# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MegaMoE's arithmetic without the fused kernel, for layers that carry LoRA.

Same weights, loader and outputs as ``DeepGemmMegaMoEExperts`` (bit for bit),
from separate steps: MegaMoE's MXFP8 input staging, an all-gather of the
quantized rows over the EP group, the batch-invariant stable expert
permutation, DeepGEMM's grouped FP8xFP4 GEMMs around
``swiglu_apply_weight_to_fp8`` (the L1 epilogue), the shared expert's FP8
GEMMs, then each top-k row sent back to its token's rank and summed in FP32
after the shared expert, slot by slot, with one BF16 rounding. Weights are kept
in the plain grouped-GEMM layout. The steps between leave room for adapters:

* routed (shared-outer): ``l1 += B_gu[e] (A_gu x)`` before the activation and
  ``l2 += B_dn (A_dn[e] a)`` with ``a = (silu(gate) * up) * w`` (routing
  weight included), each delta rounded to BF16 and added once in BF16;
* shared expert: ordinary LoRA on its gate/up and down projections.

Rows of tokens without an adapter keep the base bits.
"""

import torch
import torch.nn as nn

from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.kernels.batch_invariant import (
    gather_expert_slots,
    stable_expert_permute,
    swiglu_apply_weight_to_fp8,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    DeepGemmMegaMoEExperts,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    compute_aligned_M_and_alignment,
)
from vllm.model_executor.layers.fused_moe.prepare_megamoe import (
    prepare_megamoe_inputs,
)
from vllm.utils.deep_gemm import (
    _import_deep_gemm,
    get_mk_alignment_for_contiguous_layout,
    mk_alignment_scope,
)

LORA_SLOTS = ("gate_up", "down", "shared_w1", "shared_w3", "shared_w2")


class MoELoRASlot(nn.Module):
    """A named adapter slot of the experts: ``lora_A`` ``[*experts, r, in]``,
    ``lora_B`` ``[*experts, out, r]``; ``experts`` routed experts or none."""

    is_lora_slot = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        per_expert: int = 0,
        b_per_expert: bool = False,
        local_experts: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.per_expert = per_expert  # number of global experts, 0 = shared
        self.b_per_expert = b_per_expert  # per-expert B (else per-expert A)
        self.local_experts = local_experts


def _mn_major(sf: torch.Tensor) -> torch.Tensor:
    """Packed UE8M0 scales, MN-major and TMA-aligned, as DeepGEMM reads them."""
    m, k = sf.shape
    out = torch.empty_strided(
        (m, k), (1, (m + 3) // 4 * 4), dtype=sf.dtype, device=sf.device
    )
    return out.copy_(sf)


def _slot_add(
    base: torch.Tensor, slots: torch.Tensor, delta: torch.Tensor, mask: torch.Tensor
) -> None:
    """``base[slots] = bf16(base[slots] + delta)`` where ``mask``, one row per
    (token, slot); other rows keep their bits. Slots of other ranks' experts
    point at ``base``'s last (scratch) row."""
    rows = base[slots]
    base.index_copy_(0, slots, torch.where(mask[:, None], rows + delta, rows))


def _to_rows(
    act: tuple[torch.Tensor, ...], slots: torch.Tensor, rows: int
) -> list[torch.Tensor]:
    """The epilogue's per-(token, slot) outputs (FP8 codes, packed MN-major
    scales, optional BF16) placed at their permuted rows of ``rows``; slots of
    other ranks' experts land in a scratch row past them."""
    q, sf = act[0], act[1]
    out_q = torch.empty((rows + 1, q.shape[1]), dtype=q.dtype, device=q.device)
    out_q.view(torch.uint8).index_copy_(0, slots, q.view(torch.uint8))
    out_sf = torch.empty_strided(
        (rows + 1, sf.shape[1]),
        (1, (rows + 4) // 4 * 4),
        dtype=sf.dtype,
        device=sf.device,
    )
    out = [out_q[:rows], out_sf.index_copy_(0, slots, sf)[:rows]]
    if len(act) > 2:
        b = torch.empty(
            (rows + 1, act[2].shape[1]), dtype=act[2].dtype, device=q.device
        )
        out.append(b.index_copy_(0, slots, act[2])[:rows])
    return out


class DeepGemmMegaMoEUnfusedExperts(DeepGemmMegaMoEExperts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        e, h, i = self.num_experts, self.hidden_size, self.intermediate_size
        s = i * max(self.num_shared_experts, 1)
        local = (self.experts_start_idx, self.experts_end_idx)
        self.gate_up = MoELoRASlot(h, 2 * i, e, b_per_expert=True, local_experts=local)
        self.down = MoELoRASlot(i, h, e, local_experts=local)
        self.shared_w1 = MoELoRASlot(h, s)
        self.shared_w3 = MoELoRASlot(h, s)
        self.shared_w2 = MoELoRASlot(s, h)
        expert_map = torch.full((e,), -1, dtype=torch.int32)
        expert_map[self.experts_start_idx : self.experts_end_idx] = torch.arange(
            self.num_local_experts, dtype=torch.int32
        )
        self.register_buffer("_expert_map", expert_map, persistent=False)

    @staticmethod
    def _kernel_layout_shard(dst, src, shard_id, is_scale):
        """Plain grouped-GEMM layout: [w1; w3] rows, scales not transposed."""
        if shard_id == "w2":
            return dst, src
        rows = dst.shape[0] // 2
        half = slice(0, rows) if shard_id == "w1" else slice(rows, None)
        return dst[half], src

    def _lora(self, name: str):
        """The adapter slot wrapper (``MoELoRASlotWithLoRA``), if LoRA is on."""
        slot = getattr(self, name)
        return slot if hasattr(slot, "punica_wrapper") else None

    def forward(self, hidden_states, topk_weights, topk_ids, shared_experts=None):
        if self.eplb_state.logical_to_physical_map is not None:
            raise NotImplementedError("Unfused MegaMoE does not support EPLB")
        dg = _import_deep_gemm()
        num_tokens, hidden = hidden_states.shape
        device, top_k = hidden_states.device, self.top_k
        is_padding = None
        if is_forward_context_available():
            is_padding = get_forward_context().is_padding
            is_padding = None if is_padding is None else is_padding[:num_tokens]
        x_q = torch.empty(
            (num_tokens, hidden), dtype=torch.float8_e4m3fn, device=device
        )
        x_sf = torch.empty(
            (num_tokens, hidden // 128), dtype=torch.int32, device=device
        )
        ids = torch.empty((num_tokens, top_k), dtype=torch.int64, device=device)
        weights = torch.empty((num_tokens, top_k), dtype=torch.float32, device=device)
        prepare_megamoe_inputs(
            hidden_states,
            topk_weights,
            topk_ids,
            x_q,
            x_sf,
            ids,
            weights,
            is_padding=is_padding,
        )
        gu, dn = self._lora("gate_up"), self._lora("down")
        has_lora = None
        if gu is not None:
            meta = gu.punica_wrapper.token_mapping_meta
            # With no adapter in the batch the mapping buffer is left stale.
            has_lora = (meta.token_lora_mapping[:num_tokens] >= 0).to(torch.float32)
            if bool(meta.no_lora_flag_cpu[0]):
                has_lora.zero_()
            xa = gu.shrink(hidden_states)  # [T, r] FP32, once per token

        # EP all-gather of the quantized rows (copies only).
        ep = get_ep_group()
        sizes = self._ep_sizes(num_tokens, ep.world_size)
        parts = [x_q.view(torch.uint8), x_sf, ids, weights]
        if gu is not None:
            parts += [xa, has_lora]
        if ep.world_size > 1:
            parts = ep.all_gatherv(parts, dim=0, sizes=sizes)
        x_all = parts[0].view(torch.float8_e4m3fn)
        ids_all, w_all = parts[2], parts[3]
        # Packed UE8M0 -> FP32 powers of two (exact) for the permutation.
        sf_bytes = parts[1].view(torch.uint8).view(x_all.shape[0], hidden // 32)
        sf_all = (sf_bytes.to(torch.int32) << 23).view(torch.float32)

        # Always the batch-invariant permutation: slots of other ranks' experts
        # and of padding get inverse -1 (the generic one leaves them unset).
        capacity, align = compute_aligned_M_and_alignment(
            M=x_all.shape[0],
            num_topk=top_k,
            local_num_experts=self.num_local_experts,
            alignment=get_mk_alignment_for_contiguous_layout()[0],
            expert_tokens_meta=None,
        )
        # The grouped GEMMs take the psum layout (expert end rows): they visit
        # the occupied blocks only, not the whole padded capacity.
        expert_ends = torch.empty(
            self.num_local_experts, dtype=torch.int32, device=device
        )
        a, a_sf, _, inverse = stable_expert_permute(
            x_all,
            sf_all,
            ids_all,
            self.num_local_experts,
            capacity,
            align,
            expert_map=self._expert_map,
            expert_ends=expert_ends,
            fill_padding=False,  # the psum GEMMs read the expert rows only
        )
        rows = a.shape[0]
        # Each (token, slot)'s permuted row (other ranks' slots: row ``rows``,
        # a scratch row).
        dump = torch.where(inverse >= 0, inverse, rows).flatten().long()
        l1_full = torch.empty(
            (rows + 1, 2 * self.intermediate_size), dtype=torch.bfloat16, device=device
        )
        l1 = l1_full[:rows]
        with mk_alignment_scope(align):
            dg.m_grouped_fp8_fp4_gemm_nt_contiguous(
                (a, a_sf),
                self.l1_weights,
                l1,
                expert_ends,
                recipe_a=(1, 32),
                recipe_b=(1, 32),
                use_psum_layout=True,
            )
            # The epilogue and the LoRA work past the grouped GEMMs run on the
            # (token, slot) rows only, not on the padded capacity.
            h = l1_full[dump]
            slot_lora = None
            slot_row = dump.clamp(max=rows - 1)
            if gu is not None:
                tok = torch.arange(x_all.shape[0], device=device).repeat_interleave(
                    top_k
                )
                slot_lora = parts[5][tok].bool() & (dump < rows)
                # Each slot's shrink output, BF16, at its permuted row.
                xa = torch.zeros(
                    (rows + 1, gu.expert_w.shape[-1]),
                    dtype=torch.bfloat16,
                    device=device,
                )
                xa[:, : parts[4].shape[1]].index_copy_(
                    0, dump, parts[4][tok].to(torch.bfloat16)
                )
                delta = gu.expert_delta(xa[:rows], expert_ends, psum=True)
                h = torch.where(slot_lora[:, None], h + delta[slot_row], h)
            act = _to_rows(
                swiglu_apply_weight_to_fp8(
                    h,
                    w_all.flatten(),
                    clamp_limit=self.activation.clamp,
                    fast_math=self.fast_math,
                    output_bf16=dn is not None,
                ),
                dump,
                rows,
            )
            l2_full = torch.empty(
                (rows + 1, hidden), dtype=torch.bfloat16, device=device
            )
            l2 = l2_full[:rows]
            dg.m_grouped_fp8_fp4_gemm_nt_contiguous(
                act[:2],
                self.l2_weights,
                l2,
                expert_ends,
                recipe_a=(1, 32),
                recipe_b=(1, 32),
                use_psum_layout=True,
            )
            if dn is not None:
                ad = dn.expert_delta(act[-1], expert_ends, psum=True)
                _slot_add(l2_full, dump, dn.shared_delta(ad[slot_row]), slot_lora)

        slots = gather_expert_slots(
            l2, ids_all, inverse, self._expert_map, torch.bfloat16
        )
        if ep.world_size > 1:
            slots = self._return_rows(ep, slots, ids, sizes, num_tokens)
        y = torch.zeros((num_tokens, hidden), dtype=torch.float32, device=device)
        if self.has_fused_shared_experts:
            y += self._shared(dg, hidden_states, x_q, x_sf).float()
        # MegaMoE's combine: shared first, slot order. An absent slot adds +0.0,
        # which keeps y's bits (y starts at +0.0, so it is never -0.0).
        slots = torch.where((ids >= 0)[..., None], slots, 0).float()
        for k in range(top_k):
            y += slots[:, k]
        y = y.to(torch.bfloat16)
        if shared_experts is not None:
            y += shared_experts(hidden_states)
        return y

    def _shared_weights(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """A shared-expert weight and its packed scales, MN-major as
        ``fp8_gemm_nt`` reads them (they are loaded row-major)."""
        weight = getattr(self, f"{name}_weight")
        return weight, _mn_major(getattr(self, f"{name}_weight_scale"))

    def _shared(self, dg, hidden_states, x_q, x_sf) -> torch.Tensor:
        tokens = hidden_states.shape[0]
        width = self.shared_l1_weight.shape[0]
        l1 = torch.empty((tokens, width), dtype=torch.bfloat16, device=x_q.device)
        dg.fp8_gemm_nt(
            (x_q, _mn_major(x_sf)),
            self._shared_weights("shared_l1"),
            l1,
            recipe=(1, 1, 32),
            disable_ue8m0_cast=True,
        )
        w1, w3, w2 = (self._lora(n) for n in ("shared_w1", "shared_w3", "shared_w2"))
        if w1 is not None:
            w1.add_linear(l1, hidden_states, [w1, w3])
        act = swiglu_apply_weight_to_fp8(
            l1,
            None,
            clamp_limit=self.activation.clamp,
            fast_math=self.fast_math,
            output_bf16=w2 is not None,
        )
        y = torch.empty(
            (tokens, self.hidden_size), dtype=torch.bfloat16, device=x_q.device
        )
        dg.fp8_gemm_nt(
            act[:2],
            self._shared_weights("shared_l2"),
            y,
            recipe=(1, 1, 32),
            disable_ue8m0_cast=True,
        )
        if w2 is not None:
            w2.add_linear(y, act[-1], [w2])
        return y

    def _ep_sizes(self, num_tokens: int, world: int) -> list[int]:
        """Each EP rank's token count (EP spans the DP ranks, TP 1)."""
        dp_metadata = (
            get_forward_context().dp_metadata
            if is_forward_context_available()
            else None
        )
        if world == 1 or dp_metadata is None:
            return [num_tokens] * world
        sizes = dp_metadata.num_tokens_across_dp_cpu.tolist()
        if len(sizes) != world:
            raise NotImplementedError("Unfused MegaMoE needs EP == DP (TP 1)")
        return sizes

    def _return_rows(self, ep, slots, ids, sizes, num_tokens) -> torch.Tensor:
        """Every rank sends each token's top-k rows back to the token's rank;
        the token's rank keeps, per slot, the copy of the expert's owner (the
        others are zeros). A pure copy: no sum can flip a -0.0."""
        comm = ep.device_communicator.pynccl_comm
        rank, world = ep.rank_in_group, ep.world_size
        starts = [sum(sizes[:r]) for r in range(world)]
        recv = slots.new_empty((world, num_tokens, *slots.shape[1:]))
        comm.group_start()
        for peer in range(world):
            if peer != rank:
                if sizes[peer]:
                    comm.send(slots[starts[peer] : starts[peer] + sizes[peer]], peer)
                if num_tokens:
                    comm.recv(recv[peer], peer)
        comm.group_end()
        recv[rank].copy_(slots[starts[rank] : starts[rank] + num_tokens])
        owner = ids.clamp(min=0) // self.num_local_experts  # [T, k]
        index = owner[None, ..., None].expand(1, *recv.shape[1:])
        return recv.gather(0, index).squeeze(0)


def lora_tensor_transform(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Adapter tensors in the trainer's shared-outer layout -> the experts'
    adapter slots: ``ffn.experts.lora_gate_up_A`` -> ``gate_up.lora_A``,
    ``ffn.experts.{e}.lora_gate_up_B`` (stacked by ``e``) -> ``gate_up.lora_B``
    ``[E, out, r]``, likewise ``down`` (per-expert A), and
    ``ffn.shared_experts.{w1,w3,w2}`` -> ``ffn.experts.shared_{w1,w3,w2}``.
    Other keys pass through; a per-expert key without its stack partners fails."""
    out: dict[str, torch.Tensor] = {}
    stacks: dict[str, dict[int, torch.Tensor]] = {}
    for name, t in tensors.items():
        head, sep, tail = name.partition(".ffn.experts.")
        if ".ffn.shared_experts." in name:
            out[name.replace(".ffn.shared_experts.", ".ffn.experts.shared_")] = t
        elif not sep:
            out[name] = t
        elif tail in ("lora_gate_up_A.weight", "lora_down_B.weight"):
            slot = "gate_up.lora_A" if "gate_up" in tail else "down.lora_B"
            out[f"{head}.ffn.experts.{slot}.weight"] = t
        else:
            expert, _, factor = tail.partition(".")
            slot = {
                "lora_gate_up_B.weight": "gate_up.lora_B",
                "lora_down_A.weight": "down.lora_A",
            }[factor]
            stacks.setdefault(f"{head}.ffn.experts.{slot}.weight", {})[int(expert)] = t
    for name, per_expert in stacks.items():
        ids = sorted(per_expert)
        if ids != list(range(ids[0], ids[0] + len(ids))):
            raise ValueError(f"{name}: experts {ids} are not a contiguous range")
        out[name] = torch.stack([per_expert[e] for e in ids])
    return out
