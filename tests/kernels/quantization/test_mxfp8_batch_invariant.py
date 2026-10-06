# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 products remain quantized and invariant to token partitioning."""

import pytest
import torch

from vllm.model_executor.kernels.linear.mxfp8.batch_invariant import (
    BatchInvariantMxfp8LinearKernel,
    block32_scaled_mm,
    packed_block32_grouped_mm,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.has_device_capability(89),
    reason="requires CUDA SM89+ FP8 GEMM",
)


def _k32_oracle(a, sa, b, sb):
    """Round each exact K32 dot to FP32, then scale and accumulate in order."""
    result = torch.zeros(a.shape[0], b.shape[0], device=a.device)
    sa = sa.view(torch.float8_e8m0fnu).float()
    sb = sb.view(torch.float8_e8m0fnu).float()
    for block, start in enumerate(range(0, a.shape[-1], 32)):
        dot = (
            a[:, start : start + 32].double() @ b[:, start : start + 32].double().T
        ).float()
        result += dot * sa[:, block, None] * sb[:, block][None, :]
    return result


@pytest.mark.parametrize("output_dtype", [torch.float32, torch.bfloat16])
def test_block32_scaled_mm_rounding_and_token_partition(output_dtype, record_property):
    torch.manual_seed(723)
    m, n, k = 129, 512, 4096
    qa = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    qb = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.randint(120, 135, (m, k // 32), device="cuda", dtype=torch.uint8)
    sb = torch.randint(120, 135, (n, k // 32), device="cuda", dtype=torch.uint8)
    expected = _k32_oracle(qa, sa, qb, sb).to(output_dtype)
    actual = block32_scaled_mm(qa, sa, qb, sb, output_dtype)
    assert torch.equal(actual, expected)
    # A full-K replacement must fail this same exact-equality oracle.
    da = qa.double() * sa.view(torch.float8_e8m0fnu).double().repeat_interleave(32, -1)
    db = qb.double() * sb.view(torch.float8_e8m0fnu).double().repeat_interleave(32, -1)
    full_k = (da @ db.T).float().to(output_dtype)
    record_property("full_k_mismatches", (full_k != expected).sum().item())
    assert not torch.equal(full_k, expected)
    for chunk in (1, 3):
        parts = [
            block32_scaled_mm(
                qa[i : i + chunk], sa[i : i + chunk], qb, sb, output_dtype
            )
            for i in range(0, m, chunk)
        ]
        assert torch.equal(actual, torch.cat(parts))


@pytest.mark.parametrize("grouped", [False, True])
def test_block32_same_row_decode_prefill_and_graph(grouped):
    torch.manual_seed(729)
    m, n, k = 1024, 128, 512
    a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.randint(124, 131, (m, k // 32), device="cuda", dtype=torch.uint8)
    sb = torch.randint(124, 131, (n, k // 32), device="cuda", dtype=torch.uint8)
    if grouped:
        a, b = a[:, None], b[None]
        sa = sa.view(torch.int32)[:, None].transpose(0, 2).contiguous().transpose(0, 2)
        sb = sb.view(torch.int32)[None].transpose(1, 2).contiguous().transpose(1, 2)
    mm = packed_block32_grouped_mm if grouped else block32_scaled_mm

    def run(rows):
        return mm(a[:rows], sa[:rows], b, sb, torch.bfloat16)

    eager = {rows: run(rows) for rows in (1, m)}
    assert torch.equal(eager[1], eager[m][:1])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for rows in (1, m):
            run(rows)
    torch.cuda.current_stream().wait_stream(stream)
    for rows in (1, m):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(rows)
        for _ in range(2):
            graph.replay()
            assert torch.equal(eager[rows], captured)


def test_batch_invariant_linear_selection_loading_and_apply(
    monkeypatch, default_vllm_config
):
    from vllm.model_executor.kernels.linear import init_mxfp8_linear_kernel
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    torch.manual_seed(731)
    kernel = init_mxfp8_linear_kernel()
    assert isinstance(kernel, BatchInvariantMxfp8LinearKernel)
    n, k = 128, 512
    layer = torch.nn.Module()
    weight = torch.randn(k, n, device="cuda").to(torch.float8_e4m3fn).T
    scales = torch.full((n + 8, k // 32 + 4), 255, device="cuda", dtype=torch.uint8)
    scales[:n, : k // 32] = torch.randint(
        124, 131, (n, k // 32), device="cuda", dtype=torch.uint8
    )
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
    original_weight = weight.view(torch.uint8).clone()
    original_scales = scales[:n, : k // 32].clone()
    for _ in range(2):
        kernel.process_weights_after_loading(layer)
        assert torch.equal(layer.weight.view(torch.uint8), original_weight)
        assert torch.equal(layer.weight_scale, original_scales)
    x = torch.randn(2, 3, k, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    qa, sa = mxfp8_e4m3_quantize(x.reshape(-1, k), is_sf_swizzled_layout=False)
    expected = _k32_oracle(qa, sa, layer.weight, original_scales).bfloat16()
    actual = kernel.apply_weights(layer, x, bias)
    assert torch.equal(actual, (expected + bias).reshape(2, 3, n))


def test_packed_grouped_mm_padding_strides_known_values_and_token_partition():
    torch.manual_seed(727)
    for groups, n, k in ((1, 32, 128), (2, 512, 4096)):
        m, padded_m = 7, 8
        a = torch.randn(m, groups, k, device="cuda")
        weight = torch.randn(groups, n, k, device="cuda")
        sf = torch.tensor([126, 127, 128], device="cuda", dtype=torch.uint8)
        sa = sf[torch.arange(m * groups * k // 32, device="cuda") % 3].reshape(
            m, groups, -1
        )
        sw = sf[torch.arange(groups * n * k // 32, device="cuda") % 3].reshape(
            groups, n, -1
        )
        # Genuine logical MN-major strides, four packed exponents per INT32.
        pa = torch.empty(
            (groups, k // 128, padded_m), device="cuda", dtype=torch.int32
        ).permute(2, 0, 1)
        pa.fill_(-1)  # NaN exponent bytes in padding must never be consumed.
        pa[:m].copy_(sa.contiguous().view(torch.int32))
        pw = torch.empty(
            (groups, k // 128, n), device="cuda", dtype=torch.int32
        ).permute(0, 2, 1)
        pw.copy_(sw.contiguous().view(torch.int32))
        qa, qw = a.to(torch.float8_e4m3fn), weight.to(torch.float8_e4m3fn)
        actual = packed_block32_grouped_mm(qa, pa, qw, pw, torch.bfloat16)
        expected = torch.stack(
            [_k32_oracle(qa[:, g], sa[:, g], qw[g], sw[g]) for g in range(groups)],
            dim=1,
        ).bfloat16()
        assert torch.equal(actual, expected)
        for chunk in (1, 3):
            parts = [
                packed_block32_grouped_mm(
                    qa[i : i + chunk], pa[i : i + chunk], qw, pw, torch.bfloat16
                )
                for i in range(0, m, chunk)
            ]
            assert torch.equal(actual, torch.cat(parts))
