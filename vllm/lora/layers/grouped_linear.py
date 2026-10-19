# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
from transformers import PreTrainedConfig

from vllm.config.lora import LoRAConfig
from vllm.model_executor.layers.linear import ColumnParallelLinear

from .column_parallel_linear import ColumnParallelLinearWithLoRA


class GroupedColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    """LoRA for a grouped linear (``is_bmm``, e.g. DeepSeek-V4 ``wo_a``): group
    ``g`` maps its own input slice, ``delta_g = B_g (A_g x_g)``.

    The model runs the grouped GEMM itself from ``weight`` and calls
    ``apply_grouped_lora`` on its output; adapters are ``lora_A``
    ``[groups, r, in]`` and ``lora_B`` ``[groups, out, r]``.
    """

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PreTrainedConfig | None = None,
    ) -> None:
        if self.tp_size != 1:
            raise NotImplementedError("Grouped LoRA supports tensor parallel 1 only")
        self.lora_config = lora_config
        self.groups = self.base_layer.bmm_batch_size
        rank, out = lora_config.max_lora_rank, self.output_size // self.groups
        dtype, device = lora_config.lora_dtype, self.device
        self.lora_a_stacked = tuple(
            torch.zeros(max_loras, 1, rank, self.input_size, dtype=dtype, device=device)
            for _ in range(self.groups)
        )
        self.lora_b_stacked = tuple(
            torch.zeros(max_loras, 1, out, rank, dtype=dtype, device=device)
            for _ in range(self.groups)
        )
        self.output_slices = (out,) * self.groups
        self.n_slices = self.groups

    def set_lora(self, index: int, lora_a: torch.Tensor, lora_b: torch.Tensor):
        self.reset_lora(index)
        if lora_a.dim() == 2:  # the zero warmup adapter is built 2-D
            lora_a = lora_a.expand(self.groups, *lora_a.shape)
            lora_b = lora_b.expand(self.groups, *lora_b.shape)
        for g in range(self.groups):
            a, b = lora_a[g], lora_b[g]
            self.lora_a_stacked[g][index, 0, : a.shape[0]].copy_(a, non_blocking=True)
            self.lora_b_stacked[g][index, 0, :, : b.shape[1]].copy_(
                b, non_blocking=True
            )

    def apply_grouped_lora(self, x: torch.Tensor, out: torch.Tensor) -> None:
        """Add each group's delta to ``out[:, g]`` (in place, BF16), from
        ``x[:, g]``: the punica shrink per group, one expand over all."""
        tokens = x.shape[0]
        buffer = torch.empty(
            (self.groups, tokens, self.lora_config.max_lora_rank),
            dtype=torch.float32,
            device=x.device,
        )
        for g in range(self.groups):
            self.punica_wrapper.add_shrink(
                buffer[g : g + 1], x[:, g], self.lora_a_stacked[g : g + 1], 1.0
            )
        self.punica_wrapper.add_expand(
            out.view(tokens, -1),
            buffer,
            self.lora_b_stacked,
            self.output_slices,
            add_inputs=True,
        )

    def forward(self, input_: torch.Tensor):
        raise RuntimeError("A grouped linear's LoRA runs through apply_grouped_lora")

    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PreTrainedConfig | None = None,
    ) -> bool:
        return type(source_layer) is ColumnParallelLinear and getattr(
            source_layer, "is_bmm", False
        )
