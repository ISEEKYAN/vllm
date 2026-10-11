# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 visible math with a fixed K32 accumulation order for precision runs."""

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.determinism.mxfp8 import batched_block32_mm
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
    return batched_block32_mm(a, a_scale, weight, weight_scale, output_dtype)


def packed_block32_grouped_mm(
    a: torch.Tensor,
    packed_a_scale: torch.Tensor,
    weight: torch.Tensor,
    packed_weight_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Consume DeepGEMM's logical INT32 UE8M0 scales without changing its codec.

    Packed scales keep MN-major TMA strides and padded activation rows. Make
    their logical order contiguous before the byte view, then discard padding.
    Weights stay FP8; each group uses the same K32 visible math as ordinary
    batch-invariant MXFP8 projections.
    """
    return batched_block32_mm(
        a, packed_a_scale, weight, packed_weight_scale, output_dtype, grouped=True
    )


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
