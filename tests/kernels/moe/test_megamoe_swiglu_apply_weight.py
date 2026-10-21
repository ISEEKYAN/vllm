# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The batch-invariant ``swiglu_apply_weight_to_fp8`` reproduces DeepGEMM
MegaMoE's L1 epilogue bit for bit: an unfused MegaMoE (grouped FP8xFP4 GEMM ->
this op -> grouped GEMM -> shared-first FP32 combine) equals
``fp8_fp4_mega_moe``, and the shared expert's activation equals the kernel's
``shared_l2_acts``."""

import pytest
import torch

from vllm.model_executor.kernels.batch_invariant import swiglu_apply_weight_to_fp8
from vllm.model_executor.layers.fused_moe.prepare_megamoe import (
    prepare_megamoe_inputs,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import _import_deep_gemm, is_deep_gemm_supported

pytestmark = pytest.mark.skipif(
    not (
        current_platform.is_device_capability_family(100) and is_deep_gemm_supported()
    ),
    reason="DeepGEMM MegaMoE kernels require SM100 and DeepGEMM",
)


def _mn_major(sf: torch.Tensor) -> torch.Tensor:
    m, k = sf.shape
    out = torch.empty_strided(
        (m, k), (1, (m + 3) // 4 * 4), dtype=sf.dtype, device=sf.device
    )
    return out.copy_(sf)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.element_size() == 1 else t.view(torch.int16)


@pytest.mark.parametrize(
    "hidden,inter,num_experts,top_k", [(1024, 512, 16, 6), (5120, 2304, 32, 6)]
)
@pytest.mark.parametrize("fast_math", [True, False])
def test_unfused_mega_moe_matches_fused(
    dist_init, hidden, inter, num_experts, top_k, fast_math
):
    from torch.distributed import group as dist_group

    dg = _import_deep_gemm()
    from vllm.third_party.deep_gemm.utils.math import (
        align,
        per_token_cast_to_fp4,
        per_token_cast_to_fp8,
    )

    clamp, max_tokens, device = 10.0, 2048, torch.device("cuda")
    torch.manual_seed(0)

    def fp4_experts(n, k):
        w = torch.empty((num_experts, n, k // 2), device=device, dtype=torch.int8)
        sf = torch.empty((num_experts, n, k // 32), device=device, dtype=torch.float)
        for e in range(num_experts):
            b = (torch.randn((n, k), device=device) / k**0.5).bfloat16()
            w[e], sf[e] = per_token_cast_to_fp4(b, use_ue8m0=True, gran_k=32)
        return w, dg.transform_sf_into_required_layout(sf, n, k, (1, 32), num_experts)

    def fp8_dense(n, k):
        b = (torch.randn((n, k), device=device) / k**0.5).bfloat16()
        q, sf = per_token_cast_to_fp8(
            b, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
        )
        return q, _mn_major(sf)

    l1, l2 = fp4_experts(2 * inter, hidden), fp4_experts(hidden, inter)
    sh1, sh2 = fp8_dense(2 * inter, hidden), fp8_dense(hidden, inter)
    t_l1, t_l2 = dg.transform_weights_for_mega_moe(l1, l2, activation="swiglu")
    t_sh1, t_sh2 = dg.transform_weights_for_mega_moe(sh1, sh2, activation="swiglu")
    buf = dg.get_symm_buffer_for_mega_moe(
        dist_group.WORLD,
        num_experts,
        max_tokens,
        top_k,
        hidden,
        inter,
        num_shared_experts=1,
        mma_type="fp8xfp4",
        activation="swiglu",
    )
    alignment = dg.get_theoretical_mk_alignment_for_contiguous_layout()
    dg.set_mk_alignment_for_contiguous_layout(alignment)
    try:
        for num_tokens, scale in ((1, 1.0), (129, 1.0), (1000, 16.0)):
            x = (torch.randn((num_tokens, hidden), device=device) * scale).bfloat16()
            _, ids = torch.topk(
                torch.rand((num_tokens, num_experts), device=device), top_k
            )
            weights = torch.rand((num_tokens, top_k), device=device) + 0.05
            weights = weights / weights.sum(-1, keepdim=True) * 2.5
            masked = torch.rand((num_tokens, top_k), device=device) < 0.1
            ids, weights = ids.masked_fill(masked, -1), weights.masked_fill(masked, 0.0)

            block_m = dg.get_block_m_for_mega_moe(
                1,
                num_experts,
                buf.num_max_tokens_per_rank,
                num_tokens,
                top_k,
                "fp8xfp4",
            )
            prepare_megamoe_inputs(
                x,
                weights,
                ids,
                buf.x[:num_tokens],
                buf.x_sf[:num_tokens],
                buf.topk_idx[:num_tokens],
                buf.topk_weights[:num_tokens],
                shared_x_sf=buf.shared_l1_acts_sf,
                shared_block_m=block_m,
            )
            x_q, x_sf = buf.x[:num_tokens].clone(), buf.x_sf[:num_tokens].clone()
            fused = torch.empty(
                (num_tokens, hidden), device=device, dtype=torch.bfloat16
            )
            dg.fp8_fp4_mega_moe(
                fused,
                t_l1,
                t_l2,
                buf,
                shared_l1_weights=t_sh1,
                shared_l2_weights=t_sh2,
                activation="swiglu",
                activation_clamp=clamp,
                fast_math=fast_math,
            )
            shared_acts = buf.shared_l2_acts[:num_tokens].clone()

            # Unfused: routes grouped per expert in (token, slot) order, psum layout.
            flat = ids.reshape(-1)
            valid = (flat >= 0).nonzero().squeeze(1)
            order = valid[torch.argsort(flat[valid], stable=True)]
            counts = torch.bincount(flat[valid], minlength=num_experts).tolist()
            psum, starts, pos = [], [], 0
            for c in counts:
                starts.append(pos)
                psum.append(pos + c)
                pos = align(pos + c, alignment)
            rows = torch.cat(
                [torch.arange(s, s + c, device=device) for s, c in zip(starts, counts)]
            )
            m = max(pos, alignment)
            rx = torch.zeros((m, hidden), device=device, dtype=torch.uint8)
            rsf = torch.zeros((m, hidden // 128), device=device, dtype=torch.int32)
            rw = torch.zeros((m,), device=device)
            rx[rows] = x_q.view(torch.uint8)[order // top_k]
            rsf[rows] = x_sf[order // top_k]
            rw[rows] = weights.reshape(-1)[order]
            psum_t = torch.tensor(psum, device=device, dtype=torch.int32)
            l1_y = torch.empty((m, 2 * inter), device=device, dtype=torch.bfloat16)
            dg.m_grouped_fp8_fp4_gemm_nt_contiguous(
                (rx.view(torch.float8_e4m3fn), _mn_major(rsf)),
                l1,
                l1_y,
                psum_t,
                use_psum_layout=True,
                recipe=(1, 1, 32),
            )
            avail = torch.tensor([psum[-1]], device=device, dtype=torch.int32)
            q, q_sf = swiglu_apply_weight_to_fp8(
                l1_y, rw, clamp_limit=clamp, fast_math=fast_math, avail_rows=avail
            )
            l2_y = torch.empty((m, hidden), device=device, dtype=torch.bfloat16)
            dg.m_grouped_fp8_fp4_gemm_nt_contiguous(
                (q, q_sf), l2, l2_y, psum_t, use_psum_layout=True, recipe=(1, 1, 32)
            )
            sh_l1 = torch.empty(
                (num_tokens, 2 * inter), device=device, dtype=torch.bfloat16
            )
            dg.fp8_gemm_nt(
                (x_q, _mn_major(x_sf)),
                sh1,
                sh_l1,
                recipe=(1, 1, 32),
                disable_ue8m0_cast=True,
            )
            sq, ssf = swiglu_apply_weight_to_fp8(
                sh_l1, None, clamp_limit=clamp, fast_math=fast_math
            )
            assert torch.equal(_bits(sq), _bits(shared_acts)), num_tokens
            sh_y = torch.empty(
                (num_tokens, hidden), device=device, dtype=torch.bfloat16
            )
            dg.fp8_gemm_nt(
                (sq, ssf), sh2, sh_y, recipe=(1, 1, 32), disable_ue8m0_cast=True
            )

            slot_out = torch.zeros(
                (num_tokens * top_k, hidden), device=device, dtype=torch.bfloat16
            )
            slot_out[order] = l2_y[rows]
            slot_out = slot_out.view(num_tokens, top_k, hidden)
            acc = torch.zeros((num_tokens, hidden), device=device) + sh_y.float()
            for k in range(top_k):
                acc = torch.where(
                    (ids[:, k] >= 0)[:, None], acc + slot_out[:, k].float(), acc
                )
            assert torch.equal(_bits(acc.bfloat16()), _bits(fused)), num_tokens
    finally:
        buf.destroy()


def test_avail_rows_and_bf16_output():
    torch.manual_seed(0)
    x = torch.randn((37, 2 * 256), device="cuda").bfloat16()
    w = torch.rand(37, device="cuda")
    q, s = swiglu_apply_weight_to_fp8(x, w, clamp_limit=10.0)
    q2, s2, b = swiglu_apply_weight_to_fp8(x, w, clamp_limit=10.0, output_bf16=True)
    assert torch.equal(_bits(q), _bits(q2)) and torch.equal(s, s2)
    gate, up = x.float().chunk(2, -1)
    ref = (
        gate.clamp(max=10)
        * torch.sigmoid(gate.clamp(max=10))
        * up.clamp(-10, 10)
        * w[:, None]
    )
    torch.testing.assert_close(b.float(), ref, rtol=1e-2, atol=1e-2)
    # Rows from avail_rows on keep their previous contents.
    avail = torch.tensor([20], device="cuda", dtype=torch.int32)
    q3, _ = swiglu_apply_weight_to_fp8(x, w, clamp_limit=10.0, avail_rows=avail)
    assert torch.equal(_bits(q3[:20]), _bits(q[:20]))
    # Row results do not depend on the batch around them.
    q4, s4 = swiglu_apply_weight_to_fp8(x[5:9], w[5:9], clamp_limit=10.0)
    assert torch.equal(_bits(q4), _bits(q[5:9])) and torch.equal(s4, s[5:9])
