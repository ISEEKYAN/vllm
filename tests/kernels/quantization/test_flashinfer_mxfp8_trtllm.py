# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.kernels.linear import (
    FlashInferTrtllmMxfp8LinearKernel,
    Mxfp8LinearLayerConfig,
)
from vllm.platforms import current_platform
from vllm.utils import flashinfer as vllm_flashinfer
from vllm.utils.flashinfer import has_flashinfer

pytestmark = pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and has_flashinfer()
    ),
    reason="FlashInfer TRTLLM MXFP8 requires an SM100-family GPU",
)


def _make_layer(weight: torch.Tensor) -> torch.nn.Module:
    from flashinfer import SfLayout, mxfp8_quantize

    weight_mxfp8, weight_scale = mxfp8_quantize(
        weight,
        sf_swizzle_layout=SfLayout.layout_linear,
    )
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight_mxfp8, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        weight_scale.view(weight.shape[0], weight.shape[1] // 32),
        requires_grad=False,
    )
    return layer


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("scale_rows", [1, 32])
@torch.inference_mode()
def test_cuda_k32_mma_tail_strides_and_batch_composition(
    grouped: bool, packed: bool, scale_rows: int
) -> None:
    """K32 products retain row semantics for strided and packed E8M0 packets."""
    from vllm.model_executor.kernels.batch_invariant import block32_scaled_mm

    generator = torch.Generator().manual_seed(1931)
    m, n, k, groups = 7, 49, 128, 2 if grouped else 1
    a = torch.randint(-8, 9, (m, groups, k), generator=generator).to(
        torch.float8_e4m3fn
    )
    weight = torch.randint(-8, 9, (groups, n, k), generator=generator).to(a.dtype)
    sa = torch.randint(126, 130, (m, groups, k // 32), generator=generator).byte()
    sw = torch.randint(
        126,
        130,
        (groups, (n + scale_rows - 1) // scale_rows, k // 32),
        generator=generator,
    ).byte()
    expected = torch.zeros(m, groups, n)
    a_factor = torch.ldexp(torch.ones_like(sa, dtype=torch.float32), sa.int() - 127)
    w_factor = torch.ldexp(torch.ones_like(sw, dtype=torch.float32), sw.int() - 127)
    w_factor = w_factor.repeat_interleave(scale_rows, dim=1)[:, :n]
    for block in range(k // 32):
        start, end = block * 32, (block + 1) * 32
        product = (
            torch.bmm(
                a[:, :, start:end].double().transpose(0, 1),
                weight[:, :, start:end].double().transpose(1, 2),
            )
            .transpose(0, 1)
            .float()
        )
        expected += product * a_factor[:, :, block, None] * w_factor[:, :, block]
    if packed:
        sa = sa.contiguous().view(torch.int32)
        sw = sw.contiguous().view(torch.int32)
    # Preserve logical values in a physically strided packet.
    a_device = torch.empty(m, groups, k * 2, device="cuda", dtype=a.dtype)[:, :, ::2]
    a_device.copy_(a)
    w_device = weight.cuda()
    sa_device = sa.cuda().transpose(0, 1).contiguous().transpose(0, 1)
    sw_device = sw.cuda()
    if not grouped:
        a_device, sa_device = a_device[:, 0], sa_device[:, 0]
        w_device, sw_device, expected = w_device[0], sw_device[0], expected[:, 0]
    actual = block32_scaled_mm(
        a_device,
        sa_device,
        w_device,
        sw_device,
        torch.float32,
        grouped=grouped,
        weight_scale_group_rows=scale_rows,
    )
    assert torch.equal(actual.cpu(), expected)
    single = block32_scaled_mm(
        a_device[:1],
        sa_device[:1],
        w_device,
        sw_device,
        torch.float32,
        grouped=grouped,
        weight_scale_group_rows=scale_rows,
    )
    assert torch.equal(actual[:1], single)
    torch.library.opcheck(
        torch.ops.vllm_batch_invariant.block32_scaled_mm.default,
        (
            a_device,
            sa_device,
            w_device,
            sw_device,
            torch.empty_like(actual),
            grouped,
            scale_rows,
        ),
        test_utils=("test_faketensor", "test_aot_dispatch_dynamic"),
    )
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = block32_scaled_mm(
            a_device,
            sa_device,
            w_device,
            sw_device,
            torch.float32,
            grouped=grouped,
            weight_scale_group_rows=scale_rows,
        )
    a_device.zero_()
    replayed.fill_(float("nan"))
    graph.replay()
    assert torch.count_nonzero(replayed).item() == 0


@pytest.mark.parametrize("shape", [(1, 130, 256), (7, 256, 512), (128, 130, 768)])
@torch.inference_mode()
def test_flashinfer_trtllm_mxfp8_linear_numerics(
    shape: tuple[int, int, int],
) -> None:
    torch.manual_seed(0)
    m, n, k = shape
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    layer = _make_layer(weight)
    kernel = FlashInferTrtllmMxfp8LinearKernel(
        Mxfp8LinearLayerConfig(weight_shape=(n, k))
    )
    kernel.process_weights_after_loading(layer)

    output = kernel.apply_weights(layer, x)
    reference = torch.mm(x, weight.t())
    similarity = F.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    )

    assert output.shape == (m, n)
    assert output.is_contiguous()
    assert similarity.item() > 0.98


@torch.inference_mode()
def test_flashinfer_trtllm_mxfp8_custom_ops() -> None:
    x = torch.randn((7, 512), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((256, 512), dtype=torch.bfloat16, device="cuda")
    layer = _make_layer(weight)
    kernel = FlashInferTrtllmMxfp8LinearKernel(
        Mxfp8LinearLayerConfig(weight_shape=(256, 512))
    )
    kernel.process_weights_after_loading(layer)

    torch.library.opcheck(
        torch.ops.vllm.flashinfer_mxfp8_quantize_8x4.default,
        (x,),
    )
    x_mxfp8, x_scale = vllm_flashinfer.flashinfer_mxfp8_quantize_8x4(x)
    # SchemaCheckMode compares inputs with allclose, which CUDA does not
    # implement for float8. The numerical tests above guard input mutation.
    torch.library.opcheck(
        torch.ops.vllm.mm_mxfp8.default,
        (
            x_mxfp8,
            layer.weight.t(),
            x_scale,
            layer.weight_scale,
            torch.bfloat16,
            "trtllm",
            True,
        ),
        test_utils=(
            "test_autograd_registration",
            "test_faketensor",
            "test_aot_dispatch_dynamic",
        ),
    )


@torch.inference_mode()
def test_flashinfer_trtllm_mxfp8_linear_cuda_graph() -> None:
    torch.manual_seed(0)
    m, n, k = 7, 130, 512
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    layer = _make_layer(weight)
    kernel = FlashInferTrtllmMxfp8LinearKernel(
        Mxfp8LinearLayerConfig(weight_shape=(n, k))
    )
    kernel.process_weights_after_loading(layer)

    static_x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    kernel.apply_weights(layer, static_x)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = kernel.apply_weights(layer, static_x)

    new_x = torch.randn_like(static_x)
    static_x.copy_(new_x)
    graph.replay()
    eager_output = kernel.apply_weights(layer, new_x)

    torch.testing.assert_close(graph_output, eager_output, rtol=0, atol=0)
