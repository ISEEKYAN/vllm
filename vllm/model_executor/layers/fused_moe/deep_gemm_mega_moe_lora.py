# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MegaMoE with LoRA hooks: the fused DeepGEMM kernel for layers whose experts
carry LoRA, instead of ``DeepGemmMegaMoEUnfusedExperts``.

The adapters are those of the unfused path (same slots and layouts): routed
shared-outer (gate_up: shared A, per-expert B; down: per-expert A, shared B)
and ordinary adapters on the shared expert's w1 / w3 / w2. The forward:

* gate_up: ``xa = x A^T`` (punica's batch-invariant shrink, rounded to BF16,
  zero for tokens without an adapter) travels with each token; the kernel adds
  ``d = sum_r xa[r] B[n, r]`` (FP32, r ascending) to the L1 accumulator before
  its BF16 rounding; the shared expert's gate / up rows use the w1 / w3 parts;
* down: the kernel exports each (token, slot)'s BF16 activation ``a``
  (SwiGLU times routing weight); ``ad = A_e a`` (BF16) is computed per slot and
  written back to the token's rank, summed in FP32 in top-k order and rounded
  to BF16 (``s``); with the shared expert's ``A_w2 a_s`` (BF16) the delta
  ``[s, A_w2 a_s] [B_dn, B_w2]^T`` (one deterministic FP32 GEMM) is added to
  MegaMoE's BF16 output once.

Every delta that is exactly zero leaves the bits unchanged, so tokens without
an adapter, and zero adapters, give the bits of the plain fused MegaMoE.
"""

import torch

from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    DeepGemmMegaMoEExperts,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe_unfused import (
    MoELoRASlot,
)
from vllm.triton_utils import tl, triton

# The down delta's K (both ranks, zero-padded): bf16_fixed_mm then takes its
# deterministic tensor-core path.
_DELTA_K = 128
_BLOCK_M = 16


@triton.jit
def _down_shrink_return_kernel(
    act_ptr,
    info_ptr,
    meta_ptr,
    a_ptr,
    a_stride_e,
    a_stride_r,
    peers_ptr,
    ret_offset,
    max_tokens,
    INTER: tl.constexpr,
    RANK: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """``ad = A_e a`` for a block of one local expert's exported rows, each row
    stored at ``lora_ret[slot, token]`` on its token's rank. Fixed tiles, so a
    row's bits do not depend on the rows around it."""
    pid = tl.program_id(0)
    expert = -1
    block = 0
    seen = 0
    for e in tl.static_range(NUM_EXPERTS):
        n = (tl.load(info_ptr + 2 * e + 1) + BLOCK_M - 1) // BLOCK_M
        hit = (pid >= seen) & (pid < seen + n)
        expert = tl.where(hit, e, expert)
        block = tl.where(hit, pid - seen, block)
        seen += n
    if expert < 0:
        return
    start = tl.load(info_ptr + 2 * expert)
    count = tl.load(info_ptr + 2 * expert + 1)
    offs = block * BLOCK_M + tl.arange(0, BLOCK_M)
    valid = offs < count
    rows = (start + offs).to(tl.int64)
    k = tl.arange(0, BLOCK_K)
    r = tl.arange(0, RANK)
    acc = tl.zeros((BLOCK_M, RANK), dtype=tl.float32)
    for k0 in range(0, INTER, BLOCK_K):
        x = tl.load(
            act_ptr + rows[:, None] * INTER + k0 + k[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        w = tl.load(
            a_ptr + expert * a_stride_e + r[None, :] * a_stride_r + k0 + k[:, None]
        )
        acc = tl.dot(x, w, acc)
    meta = meta_ptr + rows * 3
    src_rank = tl.load(meta, mask=valid, other=0)
    token = tl.load(meta + 1, mask=valid, other=0).to(tl.int64)
    slot = tl.load(meta + 2, mask=valid, other=0).to(tl.int64)
    base = tl.load(peers_ptr + src_rank, mask=valid, other=0)
    dst = base + ret_offset + (slot * max_tokens + token) * (RANK * 2)
    ptrs = (dst[:, None] + r[None, :] * 2).to(tl.pointer_type(tl.bfloat16))
    tl.store(ptrs, acc.to(tl.bfloat16), mask=valid[:, None])


@triton.jit
def _stage_xa_kernel(
    shrink_ptr,  # [3, T, RANK] FP32: gate_up, shared w1, shared w3 shrinks
    has_lora_ptr,
    xa_ptr,  # [T, 3 * RANK] BF16 (the symmetric buffer's lora_xa rows)
    num_tokens,
    RANK: tl.constexpr,
):
    t = tl.program_id(0)
    r = tl.arange(0, RANK)
    on = tl.load(has_lora_ptr + t) != 0
    for part in tl.static_range(3):
        v = tl.load(shrink_ptr + (part * num_tokens + t) * RANK + r)
        v = tl.where(on, v, 0.0)
        tl.store(xa_ptr + t * 3 * RANK + part * RANK + r, v.to(tl.bfloat16))


@triton.jit
def _down_inputs_kernel(
    ret_ptr,  # [TOPK, max_tokens, RANK] BF16: the slots' A_e a rows
    ids_ptr,  # [T, TOPK] expert ids, -1 for no slot
    has_lora_ptr,
    shared_ptr,  # [T, RANK] FP32: the shared expert's A_w2 a_s
    out_ptr,  # [T, K_OUT] BF16: [s, A_w2 a_s, 0 ...]
    max_tokens,
    TOPK: tl.constexpr,
    RANK: tl.constexpr,
    K_OUT: tl.constexpr,
):
    """``s = bf16(sum_k ret[k, t])`` (FP32, top-k order, absent slots skipped)
    and the shared part, zero for tokens without an adapter."""
    t = tl.program_id(0)
    r = tl.arange(0, RANK)
    on = tl.load(has_lora_ptr + t) != 0
    s = tl.zeros((RANK,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        row = tl.load(ret_ptr + (k * max_tokens + t) * RANK + r).to(tl.float32)
        s = s + tl.where(tl.load(ids_ptr + t * TOPK + k) >= 0, row, 0.0)
    shared = tl.load(shared_ptr + t * RANK + r)
    base = out_ptr + t * K_OUT
    tl.store(base + r, tl.where(on, s, 0.0).to(tl.bfloat16))
    tl.store(base + RANK + r, tl.where(on, shared, 0.0).to(tl.bfloat16))
    pad = tl.arange(0, K_OUT - 2 * RANK)
    tl.store(base + 2 * RANK + pad, tl.zeros((K_OUT - 2 * RANK,), tl.bfloat16))


@triton.jit
def _add_delta_kernel(y_ptr, delta_ptr, n, BLOCK: tl.constexpr):
    """``y = bf16(y + delta)`` where ``delta != 0``; other elements keep their
    bits."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    y = tl.load(y_ptr + i, mask=m)
    d = tl.load(delta_ptr + i, mask=m, other=0.0)
    out = tl.where(d != 0.0, (y.to(tl.float32) + d).to(tl.bfloat16), y)
    tl.store(y_ptr + i, out, mask=m)


class DeepGemmMegaMoELoRAExperts(DeepGemmMegaMoEExperts):
    def __init__(self, vllm_config, **kwargs):
        super().__init__(vllm_config, **kwargs)
        if not self.has_fused_shared_experts:
            raise NotImplementedError("MegaMoE LoRA hooks need the fused shared expert")
        e, h, i = self.num_experts, self.hidden_size, self.intermediate_size
        s = i * self.num_shared_experts
        local = (self.experts_start_idx, self.experts_end_idx)
        self.gate_up = MoELoRASlot(h, 2 * i, e, b_per_expert=True, local_experts=local)
        self.down = MoELoRASlot(i, h, e, local_experts=local)
        self.shared_w1 = MoELoRASlot(h, s)
        self.shared_w3 = MoELoRASlot(h, s)
        self.shared_w2 = MoELoRASlot(s, h)
        self.lora_rank = 0  # set from the adapter slots on the first LoRA forward
        self._views: dict[str, torch.Tensor] | None = None

    def _lora(self, name: str):
        """The adapter slot wrapper (``MoELoRASlotWithLoRA``), if LoRA is on."""
        slot = getattr(self, name)
        return slot if hasattr(slot, "punica_wrapper") else None

    def _ensure_views(self) -> None:
        """The kernel's layouts of the adapters (gate_up B interleaved like the
        L1 weights, the down delta's [B_dn, B_w2]): fixed tensors, rewritten on
        every adapter load, so CUDA graphs keep reading the current adapter."""
        if self._views is not None:
            return
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()
        self.lora_rank = self.gate_up.rank
        if 2 * self.lora_rank > _DELTA_K:
            raise NotImplementedError(f"LoRA rank {self.lora_rank} > {_DELTA_K // 2}")
        self._symm_buffer_kwargs = {
            "lora_rank": self.lora_rank,
            "shared_lora_rank": self.lora_rank,
        }
        r, i, h = self.lora_rank, self.intermediate_size, self.hidden_size
        s = i * self.num_shared_experts
        device = self.gate_up.expert_w.device
        self._views = {
            "gate_up": torch.zeros(
                (self.num_local_experts, 2 * i, r), dtype=torch.bfloat16, device=device
            ),
            "shared": torch.zeros((2 * s, r), dtype=torch.bfloat16, device=device),
            "delta": torch.zeros((h, _DELTA_K), dtype=torch.bfloat16, device=device),
        }
        self._deep_gemm = deep_gemm
        for name in ("gate_up", "down", "shared_w1", "shared_w3", "shared_w2"):
            getattr(self, name).on_update = self._refresh_views
        self._refresh_views()
        self._kernel_kwargs.update(
            lora_b_gate_up=self._views["gate_up"],
            lora_b_shared_gate_up=self._views["shared"],
        )

    def _refresh_views(self) -> None:
        if self._views is None:
            return
        r, dg, views = self.lora_rank, self._deep_gemm, self._views
        w1_b = self.shared_w1.lora_b_stacked[0][0, 0]
        w3_b = self.shared_w3.lora_b_stacked[0][0, 0]
        # The views are made during a forward (inference tensors); adapter
        # loads run outside inference mode, where in-place writes to them fail.
        with torch.inference_mode():
            views["gate_up"].copy_(
                dg.transform_lora_b_for_mega_moe(self.gate_up.expert_w[:, :, :r])
            )
            views["shared"].copy_(dg.transform_shared_lora_b_for_mega_moe(w1_b, w3_b))
            delta = views["delta"]
            delta.zero_()
            delta[:, :r].copy_(self.down.shared_w[:, :r])
            delta[:, r : 2 * r].copy_(self.shared_w2.lora_b_stacked[0][0, 0])

    def forward(
        self, hidden_states, topk_weights, topk_ids, shared_experts=None
    ) -> torch.Tensor:
        gu = self._lora("gate_up")
        if gu is None:
            return super().forward(
                hidden_states, topk_weights, topk_ids, shared_experts
            )
        if self.eplb_state.logical_to_physical_map is not None:
            raise NotImplementedError("MegaMoE LoRA hooks do not support EPLB")
        self._ensure_views()
        num_tokens = hidden_states.shape[0]
        meta = gu.punica_wrapper.token_mapping_meta
        has_lora = meta.token_lora_mapping[:num_tokens] >= 0
        if bool(meta.no_lora_flag_cpu[0]):
            has_lora = torch.zeros_like(has_lora)
        w1, w3 = self._lora("shared_w1"), self._lora("shared_w3")
        # One batch-invariant shrink for the three adapters that read x: the
        # same per-slice arithmetic as one shrink per adapter.
        shrink = torch.zeros(
            (3, num_tokens, self.lora_rank),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        gu.punica_wrapper.add_shrink(
            shrink,
            hidden_states,
            (gu.lora_a_stacked[0], w1.lora_a_stacked[0], w3.lora_a_stacked[0]),
            1.0,
        )
        buffer = self.get_symm_buffer()
        if num_tokens:
            _stage_xa_kernel[(num_tokens,)](
                shrink, has_lora, buffer.lora_xa, num_tokens, RANK=self.lora_rank
            )
        y = super().forward(hidden_states, topk_weights, topk_ids, shared_experts)
        return self._add_down_delta(y, buffer, topk_ids, has_lora)

    def _add_down_delta(self, y, buffer, topk_ids, has_lora) -> torch.Tensor:
        from vllm.model_executor.kernels.linear.cute_dsl.batch_invariant import (
            bf16_fixed_mm,
        )

        num_tokens, r = y.shape[0], self.lora_rank
        dn, w2 = self._lora("down"), self._lora("shared_w2")
        ep = get_ep_group()
        grid = (
            triton.cdiv(ep.world_size * num_tokens * self.top_k, _BLOCK_M)
            + self.num_local_experts,
        )
        _down_shrink_return_kernel[grid](
            buffer.lora_act,
            buffer.lora_expert_info,
            buffer.lora_pool_meta,
            dn.expert_w,
            dn.expert_w.stride(0),
            dn.expert_w.stride(1),
            buffer.lora_peer_bases,
            buffer.lora_ret_offset,
            buffer.num_max_tokens_per_rank,
            INTER=self.intermediate_size,
            RANK=r,
            NUM_EXPERTS=self.num_local_experts,
            BLOCK_M=_BLOCK_M,
            BLOCK_K=64,
        )
        ad_shared = w2.shrink(buffer.lora_shared_act[:num_tokens])
        if ep.world_size > 1:  # the peers' rows have landed in lora_ret
            buffer.handle.barrier()
        a = torch.empty((num_tokens, _DELTA_K), dtype=torch.bfloat16, device=y.device)
        if num_tokens:
            _down_inputs_kernel[(num_tokens,)](
                buffer.lora_ret,
                topk_ids,
                has_lora,
                ad_shared,
                a,
                buffer.num_max_tokens_per_rank,
                TOPK=self.top_k,
                RANK=r,
                K_OUT=_DELTA_K,
            )
        assert self._views is not None
        delta = bf16_fixed_mm(a, self._views["delta"])
        n = y.numel()
        if n:
            _add_delta_kernel[(triton.cdiv(n, 1024),)](y, delta, n, BLOCK=1024)
        return y
