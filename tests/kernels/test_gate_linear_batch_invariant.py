# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA BI router: FP32 accumulation with one reduction order for every batch."""

import math

import pytest
import torch

from vllm.model_executor.determinism.batch_invariant import matmul_persistent
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.platforms import current_platform


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.parametrize("hidden,experts", [(512, 4), (4096, 256)])
def test_bi_router_fp32_across_batches_chunks_and_graph(monkeypatch, hidden, experts):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    torch.manual_seed(17)
    gate = GateLinear.__new__(GateLinear)
    torch.nn.Module.__init__(gate)
    gate.weight = torch.nn.Parameter(
        torch.randn(experts, hidden, device="cuda", dtype=torch.bfloat16),
        requires_grad=False,
    )
    gate.allow_cublas_router_gemm = True
    x = torch.randn(129, hidden, device="cuda", dtype=torch.bfloat16)
    full, bias = gate(x)
    assert bias is None and full.dtype == torch.float32
    reference = (x.double() @ gate.weight.double().T).float()
    # Unit-variance dot products have RMS sqrt(K); compare normalized error
    # so cancellation near zero does not impose a K-independent absolute bound.
    scale = math.sqrt(hidden)
    torch.testing.assert_close(full / scale, reference / scale, rtol=2e-5, atol=2e-5)
    assert not torch.equal(full, full.bfloat16().float())
    # Default matmul output must retain its original BF16 rounding contract.
    assert torch.equal(matmul_persistent(x, gate.weight.T), full.bfloat16())
    for size in (1, 2, 4, 16, 64, 128):
        chunks = [gate(chunk)[0] for chunk in x.split(size)]
        assert torch.equal(torch.cat(chunks), full), f"chunk size {size}"
    order = torch.randperm(x.shape[0], device=x.device)
    assert torch.equal(gate(x[order])[0], full[order])
    single = x[37:38].clone()
    gate(single)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, _ = gate(single)
    graph.replay()
    assert torch.equal(captured, full[37:38])
