# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped (``is_bmm``) linear LoRA, as on DeepSeek-V4 ``wo_a``: each group
adds ``B_g (A_g x_g)`` to its own output slice, and a row's result does not
depend on the batch around it; with an FP8 ``wo_a`` the adapter still reads the
BF16 rotated attention output."""

import pytest
import torch

from vllm.config.lora import LoRAConfig
from vllm.lora.layers import GroupedColumnParallelLinearWithLoRA, LoRAMapping
from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_cuda(), reason="punica kernels need CUDA")
def test_grouped_lora(dist_init, monkeypatch):
    import vllm.lora.ops.triton_ops.lora_shrink_op as shrink_op
    import vllm.lora.ops.triton_ops.utils as lora_utils

    monkeypatch.setattr(lora_utils, "is_batch_invariant", True)
    monkeypatch.setattr(shrink_op, "is_batch_invariant", True)
    lora_utils.get_lora_op_configs.cache_clear()
    torch.manual_seed(0)
    groups, k, n, rank, tokens = 8, 512, 128, 16, 77
    torch.set_default_dtype(torch.bfloat16)
    base = ColumnParallelLinear(k, groups * n, bias=False, return_bias=False).cuda()
    torch.set_default_dtype(torch.float32)
    base.is_bmm, base.bmm_batch_size = True, groups
    config = LoRAConfig(max_loras=1, max_lora_rank=rank, lora_dtype=torch.bfloat16)
    assert GroupedColumnParallelLinearWithLoRA.can_replace_layer(base, config, [])
    layer = GroupedColumnParallelLinearWithLoRA(base)
    layer.create_lora_weights(1, config)
    punica = PunicaWrapperGPU(256, 8, "cuda", lora_config=config)
    layer.set_mapping(punica)
    a = (torch.randn(groups, rank, k, device="cuda") / k**0.5).bfloat16()
    b = (torch.randn(groups, n, rank, device="cuda") / rank**0.5).bfloat16()
    layer.set_lora(0, a, b)
    ids = (1,) * (tokens - 5) + (0,) * 5  # the last 5 rows carry no adapter
    punica.update_metadata(LoRAMapping(ids, ids), [1], max_loras=1, vocab_size=0)

    x = torch.randn(tokens, groups, k, device="cuda").bfloat16()
    z0 = torch.randn(tokens, groups, n, device="cuda").bfloat16()
    z = z0.clone()
    layer.apply_grouped_lora(x, z)
    xa = torch.einsum("tgk,grk->gtr", x.float(), a.float()).bfloat16().float()
    delta = torch.einsum("gtr,gnr->tgn", xa, b.float()).bfloat16()
    ref = (z0.float() + delta.float()).bfloat16()
    torch.testing.assert_close(z[:-5].float(), ref[:-5].float(), atol=0.04, rtol=0.02)
    assert torch.equal(z[-5:].view(torch.int16), z0[-5:].view(torch.int16))

    # The same rows in a smaller batch come out bit for bit the same.
    sub = 13
    punica.update_metadata(
        LoRAMapping(ids[:sub], ids[:sub]), [1], max_loras=1, vocab_size=0
    )
    z_sub = z0[:sub].clone()
    layer.apply_grouped_lora(x[:sub], z_sub)
    assert torch.equal(z_sub.view(torch.int16), z[:sub].view(torch.int16))
    lora_utils.get_lora_op_configs.cache_clear()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="the RoPE kernel needs CUDA")
def test_fp8_wo_a_lora_reads_bf16_inv_rope(monkeypatch):
    """With an FP8 ``wo_a`` the adapter adds to the FP8 GEMM's output and reads
    BF16 ``inv_rope(o)``: the rotation the GEMM input is quantized from."""
    from types import SimpleNamespace

    import vllm.models.deepseek_v4.nvidia.ops.o_proj as o_proj
    from vllm.model_executor.layers.fusion.quant_activation import (
        QuantizedActivation,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
    )
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
        fused_inv_rope_fp8_quant,
    )

    monkeypatch.setattr(o_proj.envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(o_proj.envs, "VLLM_BI_MXFP8_GEMM", "tensor_core")
    monkeypatch.setattr(o_proj, "_compile_einsum_configs", lambda *a, **k: None)
    tokens, groups, heads_per_group, rank = 5, 2, 2, 128
    base = torch.randn(tokens, groups, rank, device="cuda").bfloat16()
    monkeypatch.setattr(o_proj, "fp8_einsum", lambda eq, a, b, z, recipe: z.copy_(base))
    seen = {}

    def lora(x, z):
        seen["x"], seen["z"] = x.clone(), z.clone()
        z.add_(1)

    wo_a = SimpleNamespace(
        weight=torch.zeros(groups * rank, heads_per_group * 512, device="cuda").to(
            torch.float8_e4m3fn
        ),
        weight_scale=torch.zeros(1, device="cuda"),
        apply_grouped_lora=lora,
    )
    o = torch.randn(tokens, groups * heads_per_group, 512, device="cuda").bfloat16()
    positions = torch.arange(tokens, device="cuda")
    cache = torch.randn(64, 64, device="cuda")
    shape = dict(n_groups=groups, heads_per_group=heads_per_group, nope_dim=448)
    out = o_proj.deep_gemm_fp8_o_proj(
        o,
        positions,
        cache,
        wo_a,
        lambda z: z,
        rope_dim=64,
        o_lora_rank=rank,
        einsum_recipe=(1, 1, 32),
        tma_aligned_scales=True,
        **shape,
    )
    rotated, _ = fused_inv_rope_fp8_quant(o, positions, cache, quantize=False, **shape)
    assert torch.equal(seen["x"].view(torch.int16), rotated.view(torch.int16))
    assert torch.equal(seen["z"], base)
    assert torch.equal(out, base.clone().add_(1).flatten(1))

    quantized = QuantizedActivation(o, o, o.dtype, o.shape, kMxfp8Dynamic)
    with pytest.raises(NotImplementedError):
        o_proj.deep_gemm_fp8_o_proj(
            quantized,
            positions,
            cache,
            wo_a,
            lambda z: z,
            rope_dim=64,
            o_lora_rank=rank,
            einsum_recipe=(1, 1, 32),
            tma_aligned_scales=True,
            **shape,
        )
