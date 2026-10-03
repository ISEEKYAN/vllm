# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GateLinear dispatch under VLLM_BATCH_INVARIANT.

The low-latency router tiers (cuteDSL ll_bf16_gemm, the fp32 specialized
kernel and the BF16x3 kernel) choose their kernel and K-reduction scheme from
the number of tokens, so a token's router logits and top-k experts would
depend on whether it is decoded alone or prefilled in a batch. Under batch
invariance every M must go through the same GEMM.

The dispatch tests mock the platform predicates and run device-free. The
bitwise test needs a Hopper or Blackwell GPU and runs in a fresh process,
because the cuBLAS workspace settings of batch-invariant mode must be applied
before the first cuBLAS handle is created.
"""

import os
from unittest import mock

import pytest
import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.router.gate_linear as gate_linear_mod
from tests.utils import spawn_new_process_for_each_test
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.platforms import current_platform

# Router (hidden_size, num_experts) of DeepSeek-V4-Flash and DeepSeek-V4-Pro.
DSV4_ROUTER_SHAPES = [(4096, 256), (7168, 384)]
DSV4_TOP_K = 6
# Decode, the ll_bf16 / fp32-kernel cutoffs and their neighbours, and prefill.
NUM_TOKENS = [1, 16, 17, 32, 33, 2048]


def _make_gate(
    monkeypatch,
    *,
    batch_invariant: bool,
    params_dtype: torch.dtype = torch.bfloat16,
    out_dtype: torch.dtype | None = torch.float32,
    input_size: int = 4096,
    output_size: int = 256,
    blackwell: bool = False,
) -> GateLinear:
    """Build a GateLinear with mocked Hopper or Blackwell eligibility."""
    for target in (
        "vllm.model_executor.layers.linear",
        "vllm.model_executor.parameter",
    ):
        monkeypatch.setattr(f"{target}.get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(f"{target}.get_tensor_model_parallel_world_size", lambda: 1)

    platform = gate_linear_mod.current_platform
    monkeypatch.setattr(platform, "is_cuda", lambda: True)
    monkeypatch.setattr(platform, "is_rocm", lambda: False)
    monkeypatch.setattr(
        platform,
        "is_device_capability",
        lambda capability: not blackwell and capability == (9, 0),
    )
    monkeypatch.setattr(
        platform,
        "is_device_capability_family",
        lambda family: blackwell and family == 100,
    )
    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.cute_dsl.ll_bf16.is_available",
        lambda: True,
    )
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", batch_invariant)

    gate = GateLinear(
        input_size=input_size,
        output_size=output_size,
        bias=False,
        out_dtype=out_dtype,
        params_dtype=params_dtype,
    )
    torch.nn.init.normal_(gate.weight)
    return gate


def _record_gemm_tiers(monkeypatch) -> list[tuple[str, int]]:
    """Replace every GateLinear GEMM tier with a recorder.

    Returns the list of ``(tier, num_tokens)`` calls. The recorders compute
    ``x @ w.T`` in fp32, which also stands in for ``torch.mm(out_dtype=...)``
    on CPU builds where that overload is CUDA-only.
    """
    calls: list[tuple[str, int]] = []

    # Keep the ReplicatedLinear fallback device-free on CUDA hosts too.
    # Its BI implementation otherwise launches Triton on our CPU tensors.
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.linear_batch_invariant",
        torch.nn.functional.linear,
    )

    def recorder(tier):
        def gemm(x, weight, *args, **kwargs):
            calls.append((tier, x.shape[0]))
            if tier == "cublas":
                assert kwargs == {"out_dtype": torch.float32}
                weight = weight.T
            return x.float() @ weight.float().T

        return gemm

    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.cute_dsl.ll_bf16.ll_bf16_gemm",
        recorder("ll_bf16"),
    )
    monkeypatch.setattr(
        torch.ops.vllm, "fp32_router_gemm_dispatch", recorder("fp32_kernel")
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.router."
        "bf16x3_router_gemm_cutedsl.bf16x3_router_gemm",
        recorder("bf16x3"),
    )
    monkeypatch.setattr(torch, "mm", recorder("cublas"))
    return calls


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
def test_bf16_gate_uses_cublas_for_every_m(monkeypatch, num_tokens):
    gate = _make_gate(monkeypatch, batch_invariant=True)
    # The flags are unchanged; only forward() skips the M-dependent tiers.
    assert gate.allow_ll_bf16_gemm
    assert gate.allow_cublas_router_gemm
    calls = _record_gemm_tiers(monkeypatch)

    x = torch.randn(num_tokens, gate.input_size, dtype=torch.bfloat16)
    output, bias = gate(x)

    assert bias is None
    assert output.shape == (num_tokens, gate.output_size)
    assert output.dtype == torch.float32
    assert calls == [("cublas", num_tokens)]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
def test_bf16_gate_keeps_low_latency_tier_without_batch_invariance(
    monkeypatch, num_tokens
):
    gate = _make_gate(monkeypatch, batch_invariant=False)
    calls = _record_gemm_tiers(monkeypatch)

    x = torch.randn(num_tokens, gate.input_size, dtype=torch.bfloat16)
    output, _ = gate(x)

    assert output.dtype == torch.float32
    expected_tier = (
        "ll_bf16" if num_tokens <= GateLinear.LL_BF16_MAX_TOKENS else "cublas"
    )
    assert calls == [(expected_tier, num_tokens)]


def test_set_out_dtype_does_not_reenable_low_latency_tier(monkeypatch):
    gate = _make_gate(monkeypatch, batch_invariant=True, out_dtype=None)
    gate.set_out_dtype(torch.float32)
    assert gate.allow_ll_bf16_gemm
    calls = _record_gemm_tiers(monkeypatch)

    gate(torch.randn(1, gate.input_size, dtype=torch.bfloat16))

    assert calls == [("cublas", 1)]


@pytest.mark.parametrize("batch_invariant", [True, False])
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
def test_fp32_gate_uses_fp32_linear_for_every_m(
    monkeypatch, batch_invariant, num_tokens
):
    input_size, output_size = 3072, 256
    assert (input_size, output_size) in GateLinear.FP32_SUPPORTED_SHAPES
    gate = _make_gate(
        monkeypatch,
        batch_invariant=batch_invariant,
        params_dtype=torch.float32,
        input_size=input_size,
        output_size=output_size,
    )
    assert gate.allow_fp32_router_gemm
    calls = _record_gemm_tiers(monkeypatch)

    x = torch.randn(num_tokens, input_size, dtype=torch.bfloat16)
    output, _ = gate(x)

    assert output.dtype == torch.float32
    if batch_invariant:
        # Falls through to ReplicatedLinear, which runs in fp32.
        assert calls == []
        torch.testing.assert_close(output, x.float() @ gate.weight.T)
    else:
        assert calls == [("fp32_kernel", num_tokens)]


@pytest.mark.parametrize("batch_invariant", [True, False])
@pytest.mark.parametrize("num_tokens", [1, 33, 2048])
def test_bf16x3_tier_respects_batch_invariance(
    monkeypatch, batch_invariant, num_tokens
):
    # An unsupported tier-2 shape reaches tier 3 directly on Blackwell.
    gate = _make_gate(
        monkeypatch,
        batch_invariant=batch_invariant,
        params_dtype=torch.float32,
        blackwell=True,
    )
    assert gate.allow_bf16x3_router_gemm
    assert not gate.allow_fp32_router_gemm
    calls = _record_gemm_tiers(monkeypatch)
    x = torch.randn(num_tokens, gate.input_size, dtype=torch.bfloat16)

    output, _ = gate(x)

    assert output.dtype == torch.float32
    torch.testing.assert_close(output, x.float() @ gate.weight.T)
    assert calls == ([] if batch_invariant else [("bf16x3", num_tokens)])


@pytest.mark.parametrize("capability", [(8, 0), (8, 9)])
def test_bitwise_check_skips_unsupported_device_before_workspace_asserts(
    monkeypatch, capability
):
    def init_unsupported_device():
        # Family 80 does not set the SM90/SM100 workspace settings.
        monkeypatch.setattr(torch.backends.cuda.matmul, "fp32_precision", "ieee")

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        "vllm.model_executor.determinism.batch_invariant.init_batch_invariance",
        init_unsupported_device,
    )
    monkeypatch.setattr(
        current_platform,
        "is_device_capability",
        lambda requested: requested == capability,
    )
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda family: family == 80
    )
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.delenv("CUBLASLT_WORKSPACE_SIZE", raising=False)

    with pytest.raises(pytest.skip.Exception, match="requires SM90 or SM100"):
        _bitwise_router_check(*DSV4_ROUTER_SHAPES[0])


def _bitwise_router_check(input_size: int, output_size: int) -> None:
    from vllm.model_executor.determinism.batch_invariant import (
        init_batch_invariance,
    )

    # Batch-invariant mode disables cuBLAS split-K through environment
    # variables that cuBLAS only reads when its first handle is created, so
    # it must be enabled before anything touches CUDA, as the GPU worker
    # does in init_worker_distributed_environment().
    assert not torch.cuda.is_initialized()
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    init_batch_invariance()
    assert torch.backends.cuda.matmul.fp32_precision == "ieee"
    if not (
        current_platform.is_device_capability((9, 0))
        or current_platform.is_device_capability_family(100)
    ):
        pytest.skip("Batch-invariant cuBLAS router GEMM requires SM90 or SM100.")
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert os.environ["CUBLASLT_WORKSPACE_SIZE"] == "1"

    with (
        mock.patch(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        mock.patch(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        mock.patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        mock.patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
    ):
        gate = GateLinear(
            input_size=input_size,
            output_size=output_size,
            bias=False,
            out_dtype=torch.float32,
            params_dtype=torch.bfloat16,
        ).cuda()

    torch.manual_seed(0)
    torch.nn.init.normal_(gate.weight, std=input_size**-0.5)
    max_tokens = max(NUM_TOKENS) + 64
    x = torch.randn(max_tokens, input_size, dtype=torch.bfloat16, device="cuda")

    ref_logits, _ = gate(x)
    assert ref_logits.dtype == torch.float32
    # fp32 accumulation and fp32 output, not a bf16 GEMM cast afterwards.
    torch.testing.assert_close(
        ref_logits.double(),
        x.double() @ gate.weight.double().T,
        atol=1e-4,
        rtol=1e-4,
    )
    ref_routes = torch.topk(ref_logits, DSV4_TOP_K, dim=-1).indices

    for num_tokens in NUM_TOKENS:
        # The same tokens at a different row offset of a different batch.
        for start in (0, 5):
            rows = slice(start, start + num_tokens)
            logits, _ = gate(x[rows])
            torch.testing.assert_close(logits, ref_logits[rows], atol=0, rtol=0)
            routes = torch.topk(logits, DSV4_TOP_K, dim=-1).indices
            assert torch.equal(routes, ref_routes[rows])


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA.")
@pytest.mark.parametrize("input_size,output_size", DSV4_ROUTER_SHAPES)
@spawn_new_process_for_each_test
def test_router_logits_are_bitwise_batch_invariant(input_size, output_size):
    _bitwise_router_check(input_size, output_size)
