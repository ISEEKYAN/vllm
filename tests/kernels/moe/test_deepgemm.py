# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit-test DeepGEMM FP8 and FP4 kernels (no DeepEP).
Compare DeepGEMM path against the Triton fallback inside vLLM's fused_experts.
"""

import importlib
import math

import pytest
import torch

# vLLM fused-expert reference (Triton fallback + DeepGEMM option)
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    deepgemm_moe_permute,
)
from vllm.model_executor.layers.fused_moe.experts.triton_deep_gemm_moe import (
    TritonOrDeepGemmExperts,
)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.utils.deep_gemm import (
    calc_diff,
    is_deep_gemm_supported,
    per_block_cast_to_fp8,
)

BLOCK_SIZE = [128, 128]


@pytest.mark.skipif(not is_deep_gemm_supported(), reason="Requires deep_gemm kernels")
def test_deepgemm_moe_permute_initializes_padding_scales(workspace_init):
    hidden_states = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    activations, scales = per_token_group_quant_fp8(
        hidden_states,
        group_size=128,
        use_ue8m0=True,
    )
    topk_ids = torch.tensor([[0], [1]], device="cuda", dtype=torch.int64)

    _, permuted_scales, expert_ids, _, _ = deepgemm_moe_permute(
        aq=activations,
        aq_scale=scales,
        topk_ids=topk_ids,
        local_num_experts=2,
        expert_map=None,
        expert_tokens_meta=None,
    )

    padding = expert_ids < 0
    assert padding.any()
    torch.testing.assert_close(
        permuted_scales[padding],
        torch.zeros_like(permuted_scales[padding]),
        rtol=0,
        atol=0,
    )


def make_block_quant_fp8_weights(
    e: int,
    n: int,
    k: int,
    block_size: list[int],
):
    """Generate (w1, w2) expert weights and their per-block scale tensors
    in FP8 block-quantized format.

      w1 shape: (E, 2N, K)
      w2 shape: (E, K, N)
    """
    dtype = torch.bfloat16
    fp8_max, fp8_min = (
        torch.finfo(torch.float8_e4m3fn).max,
        torch.finfo(torch.float8_e4m3fn).min,
    )

    # bf16 reference weights
    w1_bf16 = torch.randn(e, 2 * n, k, device="cuda", dtype=dtype) / 10
    w2_bf16 = torch.randn(e, k, n, device="cuda", dtype=dtype) / 10
    w1_bf16.clamp_(fp8_min, fp8_max)
    w2_bf16.clamp_(fp8_min, fp8_max)

    block_n, block_k = block_size
    n_tiles_w1 = math.ceil((2 * n) / block_n)
    k_tiles_w1 = math.ceil(k / block_k)
    n_tiles_w2 = math.ceil(k / block_n)
    k_tiles_w2 = math.ceil(n / block_k)

    w1 = torch.empty_like(w1_bf16, dtype=torch.float8_e4m3fn)
    w2 = torch.empty_like(w2_bf16, dtype=torch.float8_e4m3fn)
    w1_s = torch.empty(e, n_tiles_w1, k_tiles_w1, device="cuda", dtype=torch.float32)
    w2_s = torch.empty(e, n_tiles_w2, k_tiles_w2, device="cuda", dtype=torch.float32)

    for i in range(e):
        w1[i], w1_s[i] = per_block_cast_to_fp8(
            w1_bf16[i], block_size=block_size, use_ue8m0=True
        )
        w2[i], w2_s[i] = per_block_cast_to_fp8(
            w2_bf16[i], block_size=block_size, use_ue8m0=True
        )

    return w1, w2, w1_s, w2_s


def run_single_case(m, n, k, topk, num_experts, block_size):
    """Run one (M,N,K) configuration on a single GPU and assert DeepGEMM ==
    Triton baseline within tolerance.
    """
    tokens_bf16 = (
        torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        .clamp_min_(-1)
        .clamp_max_(1)
    )
    _, a1_scale = per_token_group_quant_fp8(tokens_bf16, block_size[1])

    # expert weight tensors
    w1, w2, w1_s, w2_s = make_block_quant_fp8_weights(num_experts, n, k, block_size)

    router_logits = torch.randn(m, num_experts, device="cuda", dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=-1)
    topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1)

    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w1_s,
        w2_scale=w2_s,
        a1_scale=a1_scale,
        block_shape=block_size,
    )
    moe_config = make_dummy_moe_config()

    deep_gemm_experts = mk.FusedMoEKernel(
        prepare_finalize=maybe_make_prepare_finalize(
            moe=moe_config,
            quant_config=quant_config,
            allow_new_interface=True,
            use_monolithic=False,
        ),
        fused_experts=TritonOrDeepGemmExperts(
            moe_config=moe_config,
            quant_config=quant_config,
        ),
    )

    # triton reference
    out_triton = fused_experts(
        hidden_states=tokens_bf16,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_config=quant_config,
    )

    # DeepGemm
    out_deepgemm = deep_gemm_experts.apply(
        hidden_states=tokens_bf16,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        global_num_experts=num_experts,
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
        expert_map=None,
    )
    diff = calc_diff(out_deepgemm, out_triton)
    assert diff < 0.001, f"Diff exceeded 1%: {diff}"


# Note: N <= 512 will disable the deepgemm path due to performance issues.
MNKs = [
    (1024, 768, 128),
    (2048, 768, 512),
    (512, 1024, 1024),
    (4096, 4096, 1024),
]

TOPKS = [2, 6]
NUM_EXPERTS = [32]


@pytest.mark.parametrize(("m", "n", "k"), MNKs)
@pytest.mark.parametrize("topk", TOPKS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.skipif(not is_deep_gemm_supported(), reason="Requires deep_gemm kernels")
def test_deepgemm_vs_triton(m, n, k, topk, num_experts, monkeypatch, workspace_init):
    with monkeypatch.context() as mp:
        mp.setenv("VLLM_USE_DEEP_GEMM", "1")

        _DeepGemmExperts = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe"
        ).DeepGemmExperts

        call_counter = {"cnt": 0}

        orig_fn = _DeepGemmExperts.apply

        def _spy_apply(*args, **kwargs):
            call_counter["cnt"] += 1
            return orig_fn(*args, **kwargs)

        monkeypatch.setattr(_DeepGemmExperts, "apply", _spy_apply)
        if topk > num_experts:
            pytest.skip(f"topk={topk} > num_experts={num_experts}")

        run_single_case(
            m=m,
            n=n,
            k=k,
            topk=topk,
            num_experts=num_experts,
            block_size=BLOCK_SIZE,
        )

        # ensure that the DeepGEMM path was indeed taken.
        assert call_counter["cnt"] == 1, (
            f"DeepGEMM path was not executed during the test. "
            f"Call counter: {call_counter['cnt']}"
        )


# ---------------------------------------------------------------------------
# FP4 weight tests (DeepGEMM m_grouped_fp8_fp4_gemm_nt_contiguous)
# ---------------------------------------------------------------------------


def make_mxfp4_weights(
    e: int,
    n: int,
    k: int,
):
    """Generate (w1, w2) expert weights in MXFP4 packed format with float32 scales,
    plus BF16 reference weights for validation.

      w1 shape: (E, 2N, K//2) uint8    — packed FP4
      w2 shape: (E, K, N//2)  uint8    — packed FP4
      w1_s shape: (E, 2N, K//32) float32  — per-row block-32 scales
      w2_s shape: (E, K, N//32)  float32  — per-row block-32 scales
      w1_bf16: (E, 2N, K)   — original BF16 for reference
      w2_bf16: (E, K, N)    — original BF16 for reference
    """
    from deep_gemm.utils.math import per_token_cast_to_fp4

    dtype = torch.bfloat16
    gran_k = 32  # MXFP4 block size

    # bf16 reference weights — scale by 1/sqrt(dim) for numerical stability
    w1_bf16 = torch.randn(e, 2 * n, k, device="cuda", dtype=dtype) * (k**-0.5)
    w2_bf16 = torch.randn(e, k, n, device="cuda", dtype=dtype) * (n**-0.5)

    # Quantize per-expert to FP4
    w1 = torch.empty(e, 2 * n, k // 2, device="cuda", dtype=torch.uint8)
    w2 = torch.empty(e, k, n // 2, device="cuda", dtype=torch.uint8)
    w1_s = torch.empty(
        e, 2 * n, math.ceil(k / gran_k), device="cuda", dtype=torch.float32
    )
    w2_s = torch.empty(e, k, math.ceil(n / gran_k), device="cuda", dtype=torch.float32)

    for i in range(e):
        w1[i], w1_s[i] = per_token_cast_to_fp4(
            w1_bf16[i].float(), use_ue8m0=True, gran_k=gran_k
        )
        w2[i], w2_s[i] = per_token_cast_to_fp4(
            w2_bf16[i].float(), use_ue8m0=True, gran_k=gran_k
        )

    return w1, w2, w1_s, w2_s, w1_bf16, w2_bf16


def _bf16_moe_reference(x, w1, w2, topk_weights, topk_ids):
    """BF16 token-loop MoE reference for correctness testing."""
    import torch.nn.functional as F

    num_tokens, hidden_size = x.shape
    intermediate = w1.shape[1] // 2
    top_k = topk_ids.shape[1]

    output = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=x.device)
    for t in range(num_tokens):
        for kk in range(top_k):
            e = topk_ids[t, kk].item()
            w = topk_weights[t, kk].item()
            fc1 = x[t : t + 1].float() @ w1[e].float().T
            linear = fc1[:, :intermediate]
            gate = fc1[:, intermediate:]
            act = F.silu(gate) * linear
            fc2 = act @ w2[e].float().T
            output[t] += w * fc2[0]
    return output.to(torch.bfloat16)


def run_single_fp4_case(m, n, k, topk, num_experts):
    """Run one (M,N,K) configuration with FP4 weights on DeepGEMM and assert
    DeepGEMM FP4 == BF16 reference within tolerance.
    """
    tokens_bf16 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * (k**-0.5)

    # FP4 expert weight tensors + BF16 originals for reference
    w1, w2, w1_s, w2_s, w1_bf16, w2_bf16 = make_mxfp4_weights(num_experts, n, k)

    router_logits = torch.randn(m, num_experts, device="cuda", dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(router_logits, k=topk, dim=-1)
    topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1)

    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        GroupShape,
    )
    from vllm.platforms import current_platform

    _fp8_dtype = current_platform.fp8_dtype()
    _block_shape = GroupShape(128, 128)
    quant_config = FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(_fp8_dtype, _block_shape, None, None, None, None),
        _a2=FusedMoEQuantDesc(_fp8_dtype, _block_shape, None, None, None, None),
        _w1=FusedMoEQuantDesc("mxfp4", None, w1_s, None, None, None),
        _w2=FusedMoEQuantDesc("mxfp4", None, w2_s, None, None, None),
    )
    moe_config = make_dummy_moe_config()

    from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
        DeepGemmFP4Experts,
    )

    deep_gemm_fp4_experts = mk.FusedMoEKernel(
        prepare_finalize=maybe_make_prepare_finalize(
            moe=moe_config,
            quant_config=quant_config,
            allow_new_interface=True,
            use_monolithic=False,
        ),
        fused_experts=DeepGemmFP4Experts(
            moe_config=moe_config,
            quant_config=quant_config,
        ),
    )

    # DeepGEMM FP4 path
    out_deepgemm_fp4 = deep_gemm_fp4_experts.apply(
        hidden_states=tokens_bf16,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        global_num_experts=num_experts,
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
        expert_map=None,
    )

    # BF16 reference using the same original weights
    out_ref = _bf16_moe_reference(tokens_bf16, w1_bf16, w2_bf16, topk_weights, topk_ids)

    # FP4 vs BF16 reference: quantization error from FP4 weights + FP8 activations
    diff = calc_diff(out_deepgemm_fp4, out_ref)
    assert diff < 0.05, f"FP4 diff exceeded 5%: {diff}"


# DeepSeek V4 dims: H=4096, I=2048, so N=2*I=4096, K=H=4096.
# FP4 quantization with block_k=32 needs large K for good accuracy.
FP4_MNKs = [
    (128, 4096, 4096),  # DeepSeek V4 shape
    (256, 2048, 2048),  # Half-size variant
    (128, 384, 3584),  # Kimi-K3 TP8 latent-MoE shape
]

FP4_TOPKS = [2]
FP4_NUM_EXPERTS = [8]


@pytest.mark.parametrize(
    "hidden,intermediate,experts,topk", [(512, 256, 4, 2), (2048, 768, 16, 8)]
)
def test_fp4_batch_invariant_clamp_contract(
    hidden, intermediate, experts, topk, monkeypatch, workspace_init
):
    """Compare real FP4 experts across batches and with independent CPU A8/FMA."""
    import vllm.envs as envs
    from tests.kernels.moe.w4a8_reference import quantize_a8, swiglu_clamp, topk_fma
    from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
        DeepGemmFP4Experts,
    )
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        make_mxfp4_moe_quant_config,
    )
    from vllm.platforms import current_platform
    from vllm.utils import deep_gemm as dg

    if not current_platform.is_device_capability_family(100):
        pytest.skip("requires SM100-family GPU")
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    dg._lazy_init()
    dg.enable_deep_gemm_batch_invariance()
    torch.manual_seed(123)
    w1, w2, s1, s2, _, _ = make_mxfp4_weights(experts, intermediate, hidden)
    # Same packed operands on both sides; independent dispatch/activation/combine.
    quant = make_mxfp4_moe_quant_config(
        Mxfp4MoeBackend.DEEPGEMM_MXFP4, s1, s2, swiglu_limit=10.0
    )
    config = make_dummy_moe_config(
        num_experts=experts,
        experts_per_token=topk,
        hidden_dim=hidden,
        intermediate_size=intermediate,
        max_num_tokens=1024,
    )
    implementation = DeepGemmFP4Experts(config, quant)
    kernel = mk.FusedMoEKernel(
        prepare_finalize=maybe_make_prepare_finalize(
            moe=config,
            quant_config=quant,
            allow_new_interface=True,
            use_monolithic=False,
        ),
        fused_experts=implementation,
    )
    x = torch.randn(129, hidden, device="cuda", dtype=torch.bfloat16) * 8
    x[0] = 0
    x[1] *= 1e-10
    # Leave one expert empty, keep the original top-k slot order.
    ids = torch.stack(
        [torch.randperm(experts - 1, device="cuda")[:topk] for _ in range(len(x))]
    ).int()
    weights = torch.softmax(torch.randn(len(x), topk, device="cuda"), -1)
    a1q, a1s = per_token_group_quant_fp8(x, 128, use_ue8m0=True)
    a1ref, s1ref = quantize_a8(x.cpu())
    assert torch.equal(a1q.cpu().view(torch.uint8), a1ref.view(torch.uint8))
    assert torch.equal(a1s.cpu(), s1ref)

    def run(values, routes, probs):
        return kernel.apply(
            hidden_states=values,
            w1=w1,
            w2=w2,
            topk_weights=probs,
            topk_ids=routes,
            global_num_experts=experts,
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=False,
            expert_map=None,
        ).clone()

    expected = run(x, ids, weights)
    for chunk in (1, 7, 32, 64):
        result = torch.cat(
            [
                run(x[i : i + chunk], ids[i : i + chunk], weights[i : i + chunk])
                for i in range(0, len(x), chunk)
            ]
        )
        assert torch.equal(result.view(torch.int16), expected.view(torch.int16))
    order = torch.randperm(len(x), device="cuda")
    assert torch.equal(
        run(x[order], ids[order], weights[order]).view(torch.int16),
        expected[order].view(torch.int16),
    )
    partners = torch.randn(513, hidden, device="cuda", dtype=torch.bfloat16)
    partner_ids = torch.arange(topk, device="cuda").expand(513, -1).int().contiguous()
    partner_weights = torch.full((513, topk), 1 / topk, device="cuda")
    together = run(
        torch.cat([x, partners]),
        torch.cat([ids, partner_ids]),
        torch.cat([weights, partner_weights]),
    )
    assert torch.equal(together[: len(x)].view(torch.int16), expected.view(torch.int16))
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(x, ids, weights)
    graph.replay()
    assert torch.equal(captured.view(torch.int16), expected.view(torch.int16))

    # Stable CPU dispatch deliberately differs from the atomic production scatter.
    cpu_ids = ids.cpu()
    slots = [torch.nonzero(cpu_ids == e) for e in range(experts)]
    sizes = [(len(s) + 127) // 128 * 128 for s in slots]
    total = sum(sizes)
    a = torch.zeros(total, hidden, dtype=torch.bfloat16)
    layout = torch.full((total,), -1, dtype=torch.int32)
    offset = 0
    for e, (s, size) in enumerate(zip(slots, sizes)):
        a[offset : offset + len(s)] = x.cpu()[s[:, 0]]
        layout[offset : offset + len(s)] = e
        offset += size
    aq, asc = quantize_a8(a)
    fc1 = torch.empty(total, 2 * intermediate, device="cuda", dtype=torch.bfloat16)
    fc2 = torch.empty(total, hidden, device="cuda", dtype=torch.bfloat16)
    with dg.mk_alignment_scope(128):
        dg.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (aq.cuda(), asc.cuda()),
            (w1.view(torch.int8), s1),
            fc1,
            layout.cuda(),
            recipe_a=(1, 128),
            recipe_b=(1, 32),
        )
        a2q, a2s = quantize_a8(swiglu_clamp(fc1.cpu()))
        q_gpu, scales_gpu = implementation._act_mul_quant(
            fc1,
            torch.empty_like(
                fc1[:, :intermediate], memory_format=torch.contiguous_format
            ).to(torch.float8_e4m3fn),
            MoEActivation.SILU,
        )
        real = layout >= 0
        assert torch.equal(
            q_gpu.cpu().view(torch.uint8)[real], a2q.view(torch.uint8)[real]
        )
        scale_bytes = (
            scales_gpu.cpu().contiguous().view(torch.uint8)[:, : intermediate // 128]
        )
        reference_bytes = ((a2s.contiguous().view(torch.int32) >> 23) & 255).byte()
        assert torch.equal(scale_bytes[real], reference_bytes[real])
        dg.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (a2q.cuda(), a2s.cuda()),
            (w2.view(torch.int8), s2),
            fc2,
            layout.cuda(),
            recipe_a=(1, 128),
            recipe_b=(1, 32),
        )
    rows = torch.empty(len(x), topk, hidden, dtype=torch.bfloat16)
    offset = 0
    for s, size in zip(slots, sizes):
        rows[s[:, 0], s[:, 1]] = fc2.cpu()[offset : offset + len(s)]
        offset += size
    assert torch.equal(
        topk_fma(rows, weights.cpu()).view(torch.int16),
        expected.cpu().view(torch.int16),
    )


@pytest.mark.parametrize(("m", "n", "k"), FP4_MNKs)
@pytest.mark.parametrize("topk", FP4_TOPKS)
@pytest.mark.parametrize("num_experts", FP4_NUM_EXPERTS)
@pytest.mark.skipif(not is_deep_gemm_supported(), reason="Requires deep_gemm kernels")
def test_deepgemm_fp4_vs_triton(
    m, n, k, topk, num_experts, monkeypatch, workspace_init
):
    pytest.importorskip("deep_gemm.utils.math")
    with monkeypatch.context() as mp:
        mp.setenv("VLLM_USE_DEEP_GEMM", "1")

        _DeepGemmFP4Experts = importlib.import_module(
            "vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe"
        ).DeepGemmFP4Experts

        call_counter = {"cnt": 0}

        orig_fn = _DeepGemmFP4Experts.apply

        def _spy_apply(*args, **kwargs):
            call_counter["cnt"] += 1
            return orig_fn(*args, **kwargs)

        monkeypatch.setattr(_DeepGemmFP4Experts, "apply", _spy_apply)
        if topk > num_experts:
            pytest.skip(f"topk={topk} > num_experts={num_experts}")

        run_single_fp4_case(
            m=m,
            n=n,
            k=k,
            topk=topk,
            num_experts=num_experts,
        )

        # ensure that the DeepGEMM FP4 path was indeed taken.
        assert call_counter["cnt"] == 1, (
            f"DeepGEMM FP4 path was not executed during the test. "
            f"Call counter: {call_counter['cnt']}"
        )
