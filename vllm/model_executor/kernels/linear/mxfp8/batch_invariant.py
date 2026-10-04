# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 visible math with a fixed K32 accumulation order for precision runs."""

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


def block32_scaled_mm(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Consume E4M3 codes/E8M0 bytes using FP8 products and FP32 accumulation.

    Each K32 product is accumulated before the next block, independently of
    token count, backend autotuning and the physical scale layout.
    """
    a_scale = a_scale.view(torch.float8_e8m0fnu).float()
    weight_scale = weight_scale.view(torch.float8_e8m0fnu).float()
    output = torch.zeros(
        a.shape[0], weight.shape[0], device=a.device, dtype=torch.float32
    )
    unit = torch.ones((), device=a.device, dtype=torch.float32)
    for block, start in enumerate(range(0, a.shape[1], 32)):
        product = torch._scaled_mm(
            a[:, start : start + 32].contiguous(),
            weight[:, start : start + 32].contiguous().T,
            unit,
            unit,
            out_dtype=torch.float32,
        )
        output += product * a_scale[:, block, None] * weight_scale[:, block][None, :]
    return output.to(output_dtype)


class BatchInvariantMxfp8LinearKernel(Mxfp8LinearKernel):
    """Fixed-block FP8 GEMM; retain quantized weights through reload."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        supported = (
            current_platform.is_cuda() and current_platform.has_device_capability(89)
        )
        return supported, "requires CUDA FP8 Tensor Cores"

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return c.bmm_batch_size is None, "grouped projection uses its own kernel"

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        n, k = weight.shape
        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(
            layer.weight_scale.data[:n, : k // 32].contiguous(), requires_grad=False
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dtype, shape = x.dtype, x.shape
        k = layer.weight.shape[-1]
        codes, scales = mxfp8_e4m3_quantize(
            x.reshape(-1, k), is_sf_swizzled_layout=False
        )
        output = block32_scaled_mm(
            codes, scales, layer.weight, layer.weight_scale, dtype
        )
        if bias is not None:
            output = output + bias
        return output.reshape(*shape[:-1], layer.weight.shape[0])
