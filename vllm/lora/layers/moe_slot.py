# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
from transformers import PreTrainedConfig

from vllm.config.lora import LoRAConfig

from .base import BaseLayerWithLoRA

# DeepGEMM's grouped BF16 GEMM wants K and N in 64s: ranks are zero-padded.
_RANK_ALIGN = 64


class MoELoRASlotWithLoRA(BaseLayerWithLoRA):
    """An adapter slot of ``DeepGemmMegaMoEUnfusedExperts``, which applies it.

    Routed slots are shared-outer, one adapter only: ``gate_up`` has a shared
    ``lora_A`` ``[r, in]`` and per-expert ``lora_B`` ``[E, out, r]``, ``down``
    per-expert ``lora_A`` ``[E, r, in]`` and a shared ``lora_B`` ``[out, r]``
    (global experts; this rank keeps its own). Shared-expert slots are
    ordinary ``[r, in]`` / ``[out, r]`` adapters run through punica.
    """

    def __init__(self, base_layer: nn.Module):
        super().__init__()
        self.base_layer = base_layer
        self.experts = base_layer.per_expert

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PreTrainedConfig | None = None,
    ) -> None:
        base, rank = self.base_layer, lora_config.max_lora_rank
        dtype, device = lora_config.lora_dtype, torch.accelerator.current_device_index()
        self.rank = rank
        self.lora_a_stacked = (
            torch.zeros(
                max_loras, 1, rank, base.in_features, dtype=dtype, device=device
            ),
        )
        self.lora_b_stacked = (
            torch.zeros(
                max_loras, 1, base.out_features, rank, dtype=dtype, device=device
            ),
        )
        if self.experts:
            if max_loras != 1:
                raise NotImplementedError("Routed-expert LoRA supports one adapter")
            pad = -(-rank // _RANK_ALIGN) * _RANK_ALIGN
            self.local = slice(*(base.local_experts or (0, self.experts)))
            n = self.local.stop - self.local.start
            # gate_up: per-expert B [E, out, r]; down: per-expert A [E, r, in].
            shape = (
                (n, base.out_features, pad)
                if self._b_per_expert
                else (n, pad, base.in_features)
            )
            self.expert_w = torch.zeros(shape, dtype=dtype, device=device)
            self.shared_w = torch.zeros(  # down's B, K-padded for the dense GEMM
                (base.out_features, pad), dtype=dtype, device=device
            )

    @property
    def _b_per_expert(self) -> bool:
        return self.base_layer.b_per_expert

    def _local(self, t: torch.Tensor) -> torch.Tensor:
        """This rank's experts of a global (or already local) expert tensor."""
        return t[self.local] if t.shape[0] == self.experts else t

    def reset_lora(self, index: int):
        for t in (*self.lora_a_stacked, *self.lora_b_stacked):
            t[index] = 0
        if self.experts:
            self.expert_w.zero_()
            self.shared_w.zero_()

    def set_lora(self, index: int, lora_a: torch.Tensor, lora_b: torch.Tensor):
        self.reset_lora(index)
        if not self.experts:
            r = lora_a.shape[0]
            self.lora_a_stacked[0][index, 0, :r].copy_(lora_a, non_blocking=True)
            self.lora_b_stacked[0][index, 0, :, :r].copy_(lora_b, non_blocking=True)
            return
        if self._b_per_expert:  # gate_up: shared A, per-expert B
            r = lora_a.shape[0]
            self.lora_a_stacked[0][index, 0, :r].copy_(lora_a, non_blocking=True)
            self.expert_w[:, :, :r].copy_(self._local(lora_b), non_blocking=True)
        else:  # down: per-expert A, shared B
            r = lora_b.shape[1]
            self.expert_w[:, :r].copy_(self._local(lora_a), non_blocking=True)
            self.shared_w[:, :r].copy_(lora_b, non_blocking=True)

    def shrink(self, x: torch.Tensor) -> torch.Tensor:
        """``x A^T`` in FP32 for every token with an adapter, punica's BI shrink."""
        out = torch.zeros(
            (1, x.shape[0], self.rank), dtype=torch.float32, device=x.device
        )
        self.punica_wrapper.add_shrink(out, x, self.lora_a_stacked, 1.0)
        return out[0]

    def expert_delta(self, x: torch.Tensor, layout: torch.Tensor) -> torch.Tensor:
        """The BF16 delta of each expert-grouped row (``layout``: its local
        expert, -1 for padding), FP32-accumulated, from its BF16 input (an
        FP32 shrink output is rounded first, as punica's expand does)."""
        from vllm.model_executor.kernels.linear.cute_dsl.batch_invariant import (
            bf16_fixed_mm,
        )
        from vllm.utils.deep_gemm import _import_deep_gemm

        dg = _import_deep_gemm()
        rows, pad = x.shape[0], self.expert_w.shape[-1 if self._b_per_expert else 1]
        x = x.to(torch.bfloat16)
        if self._b_per_expert:
            xp = torch.zeros((rows, pad), dtype=torch.bfloat16, device=x.device)
            xp[:, : x.shape[1]] = x
            out = torch.empty(
                (rows, self.expert_w.shape[1]), dtype=torch.bfloat16, device=x.device
            )
            dg.m_grouped_bf16_gemm_nt_contiguous(xp, self.expert_w, out, layout)
            return out
        ad = torch.empty((rows, pad), dtype=torch.bfloat16, device=x.device)
        dg.m_grouped_bf16_gemm_nt_contiguous(x.contiguous(), self.expert_w, ad, layout)
        return bf16_fixed_mm(ad, self.shared_w).to(torch.bfloat16)

    def add_linear(self, y: torch.Tensor, x: torch.Tensor, slots: list) -> None:
        """Ordinary LoRA of the listed slots, side by side, added into ``y``."""
        self.punica_wrapper.add_lora_linear(
            y,
            x,
            tuple(s.lora_a_stacked[0] for s in slots),
            tuple(s.lora_b_stacked[0] for s in slots),
            1.0,
            tuple(s.base_layer.out_features for s in slots),
        )

    def forward(self, *args, **kwargs):
        raise RuntimeError("The experts module applies its adapter slots")

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PreTrainedConfig | None = None,
    ) -> bool:
        from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe_unfused import (
            MoELoRASlot,
        )

        return isinstance(source_layer, MoELoRASlot)
