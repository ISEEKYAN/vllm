# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA slot kernels against an independent CPU FP32 fmaf oracle.

These unit tests isolate kernel indexing and reduction. Real EP communicator
and AG/RS coverage is separate from this local-kernel test.
"""

import pytest
import torch

from tests.kernels.moe.w4a8_reference import topk_fma
from vllm.platforms import current_platform

# CPU contract hosts do not need to import the CUDA kernel module.
if current_platform.is_cuda():
    from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
        _ep_ordered_combine_kernel,
        _ep_unweighted_slots_kernel,
    )

pytestmark = [
    pytest.mark.skip_global_cleanup,
    pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required"),
]


@pytest.mark.parametrize("ep", [4, 8])
@pytest.mark.parametrize("width", [128, 1024, 5120])
@pytest.mark.parametrize("arm", ["fma-midpoint", "slot-order", "padding-sentinel"])
def test_owned_slots_and_ordered_fma(ep, width, arm):
    tokens, topk = 3, 6
    routes = torch.tensor([0, 2, 4, 6, 1, 3]).repeat(tokens, 1)
    weights = torch.ones(tokens, topk, dtype=torch.float32)
    values = torch.zeros(tokens, topk, width, dtype=torch.bfloat16)
    if arm == "slot-order":
        values[:, 0] = 2**30
        values[:, 1] = 1
        values[:, 2] = -(2**30)
    else:
        values[:, 0] = -(1 + 2**-7)
        values[:, 1] = 1 + 2**-7
        weights[:, 1] = 1 + 2**-23
    if arm == "padding-sentinel":
        routes[:, 5] = -1
        routes[::2, 0] = -1
        weights.masked_fill_(routes < 0, float("nan"))
    expected = topk_fma(
        values.masked_fill((routes < 0).unsqueeze(-1), 0),
        weights.masked_fill(routes < 0, 0),
    )
    device = torch.device("cuda", torch.accelerator.current_device_index())
    total_slots = torch.zeros(tokens, topk, width, device=device)
    block = min(width, 1024)
    gpu_routes, gpu_weights = routes.to(device), weights.to(device)
    for rank in range(ep):
        lo, hi = rank * (8 // ep), (rank + 1) * (8 // ep)
        owned = (routes >= lo) & (routes < hi)
        pairs = owned.nonzero()
        pairs = pairs[torch.argsort(routes[owned], stable=True)]
        rows = values[pairs[:, 0], pairs[:, 1]].to(device).contiguous()
        # Remote/padding inverse entries are deliberately unsafe addresses.
        inverse = torch.full(routes.shape, -999999, dtype=torch.int32)
        inverse[pairs[:, 0], pairs[:, 1]] = torch.arange(len(pairs), dtype=torch.int32)
        expert_map = torch.full((8,), -1, device=device, dtype=torch.int32)
        expert_map[lo:hi] = torch.arange(hi - lo, device=device, dtype=torch.int32)
        slots = torch.empty_like(total_slots)
        _ep_unweighted_slots_kernel[(tokens, topk, (width + block - 1) // block)](
            rows,
            gpu_routes,
            inverse.to(device),
            expert_map,
            slots,
            WIDTH=width,
            TOPK=topk,
            BLOCK=block,
            VALUE_STRIDE=rows.stride(0),
            ROUTE_STRIDE=gpu_routes.stride(0),
            INVERSE_STRIDE=inverse.stride(0),
            GLOBAL_EXPERTS=8,
            VALUE_ROWS=len(pairs),
        )
        torch.accelerator.synchronize(device)
        target = values.float().masked_fill(~owned.unsqueeze(-1), 0)
        assert torch.equal(slots.cpu(), target)
        # Single-source slots; this is local assembly, not a fake EP collective.
        total_slots += slots
    output = torch.empty(tokens, width, device=device, dtype=torch.bfloat16)
    _ep_ordered_combine_kernel[(tokens, (width + block - 1) // block)](
        total_slots,
        gpu_weights,
        gpu_routes,
        output,
        WIDTH=width,
        TOPK=topk,
        WEIGHT_STRIDE=gpu_weights.stride(0),
        ROUTE_STRIDE=gpu_routes.stride(0),
        GLOBAL_EXPERTS=8,
        OUTPUT_STRIDE=output.stride(0),
        BLOCK=block,
    )
    torch.accelerator.synchronize(device)
    assert torch.equal(output.cpu(), expected)
    assert torch.isfinite(output).all()
