# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MegaMoE with LoRA hooks (fused kernel, DeepGEMM built with them) against the
plain fused MegaMoE and the unfused LoRA path, on the same checkpoint: zero
adapters and tokens without an adapter keep the plain kernel's bits, adapter
tokens stay close to the unfused contract, and a row does not depend on the
batch around it."""

import pytest
import torch

from tests.kernels.moe.test_deep_gemm_mega_moe import (
    _bits_equal,
    _mega_moe_experts,
    _routes,
    requires_mega_moe,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    DeepGemmMegaMoEExperts,
)


def _has_lora_hooks() -> bool:
    from vllm.utils.deep_gemm import _import_deep_gemm

    try:
        return hasattr(_import_deep_gemm(), "transform_lora_b_for_mega_moe")
    except Exception:
        return False


@pytest.fixture(autouse=True)
def sm100_mega_moe_layout(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe."
        "check_deep_gemm_mega_moe_supported",
        lambda *args: None,
    )


def _attach_adapters(experts, punica, config, rank, num_experts, zero, seed):
    from vllm.lora.layers import MoELoRASlotWithLoRA
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe_unfused import (
        LORA_SLOTS,
    )

    gen = torch.Generator(device="cuda").manual_seed(seed)
    for name in LORA_SLOTS:
        slot = MoELoRASlotWithLoRA(getattr(experts, name))
        slot.create_lora_weights(1, config)
        slot.set_mapping(punica)
        setattr(experts, name, slot)
        a_shape = (rank, slot.in_features)
        b_shape = (slot.out_features, rank)
        if name == "gate_up":
            b_shape = (num_experts, *b_shape)
        elif name == "down":
            a_shape = (num_experts, *a_shape)
        a = torch.randn(a_shape, device="cuda", generator=gen) / a_shape[-1] ** 0.5
        b = 0.05 * torch.randn(b_shape, device="cuda", generator=gen)
        slot.set_lora(0, a.bfloat16(), (b * (not zero)).bfloat16())


@requires_mega_moe
@pytest.mark.skipif(not _has_lora_hooks(), reason="DeepGEMM without LoRA hooks")
def test_mega_moe_lora_hooks(dist_init, monkeypatch):
    import vllm.lora.ops.triton_ops.lora_shrink_op as shrink_op
    import vllm.lora.ops.triton_ops.utils as lora_utils
    from vllm.config.lora import LoRAConfig
    from vllm.lora.layers import LoRAMapping
    from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe_lora import (
        DeepGemmMegaMoELoRAExperts,
    )
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe_unfused import (
        DeepGemmMegaMoEUnfusedExperts,
    )

    monkeypatch.setattr(lora_utils, "is_batch_invariant", True)
    monkeypatch.setattr(shrink_op, "is_batch_invariant", True)
    lora_utils.get_lora_op_configs.cache_clear()
    hidden, inter, num_experts, top_k, rank = 1024, 512, 16, 6, 32
    args = (hidden, inter, num_experts, top_k)
    plain = _mega_moe_experts(DeepGemmMegaMoEExperts, *args, "plain")
    zero = _mega_moe_experts(DeepGemmMegaMoELoRAExperts, *args, "zero")
    hooked = _mega_moe_experts(DeepGemmMegaMoELoRAExperts, *args, "hooked")
    unfused = _mega_moe_experts(DeepGemmMegaMoEUnfusedExperts, *args, "unfused")
    x, weights, ids = _routes(300, num_experts, top_k, hidden)
    base = plain(x, weights, ids)

    config = LoRAConfig(max_loras=1, max_lora_rank=rank, lora_dtype=torch.bfloat16)
    punica = PunicaWrapperGPU(1024, 8, "cuda", lora_config=config)
    _attach_adapters(zero, punica, config, rank, num_experts, zero=True, seed=3)
    _attach_adapters(hooked, punica, config, rank, num_experts, zero=False, seed=3)
    _attach_adapters(unfused, punica, config, rank, num_experts, zero=False, seed=3)

    def run(experts, rows):
        mapping = tuple(1 if r % 5 else 0 for r in rows.tolist())  # every 5th: none
        punica.update_metadata(LoRAMapping(mapping, mapping), [1], 1, 0)
        return experts(x[rows], weights[rows], ids[rows])

    rows = torch.arange(300, device="cuda")
    # Zero adapters: the plain kernel's bits on every row.
    assert _bits_equal(run(zero, rows), base)
    full = run(hooked, rows)
    adapter = rows % 5 != 0
    assert _bits_equal(full[~adapter], base[~adapter])
    # Adapter rows: the delta of the unfused contract, up to where it rounds.
    ref = run(unfused, rows)[adapter].float()
    effect = (ref - base[adapter].float()).norm()
    assert effect > 1e-3 * base[adapter].float().norm()
    assert (full[adapter].float() - ref).norm() < 0.05 * effect
    for sub in (rows[:37], rows[150:], rows.flip(0)):
        assert _bits_equal(run(hooked, sub), full[sub])
    lora_utils.get_lora_op_configs.cache_clear()
