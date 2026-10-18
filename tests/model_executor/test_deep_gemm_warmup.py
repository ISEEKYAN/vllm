# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import (
    DeepGemmFp8BlockScaledMMKernel,
)
from vllm.model_executor.warmup import deep_gemm_warmup


@pytest.mark.cpu_test
@pytest.mark.parametrize("vendored", [False, True])
def test_bi_rejects_deep_gemm_without_deterministic_control(monkeypatch, vendored):
    """A missing BI API must not be swallowed as an optional import failure."""
    from vllm import envs
    from vllm.utils import deep_gemm

    module = SimpleNamespace()

    def import_module(name):
        if name == "deep_gemm" and vendored:
            raise ImportError("use vendored module")
        return module

    monkeypatch.setattr(deep_gemm.importlib, "import_module", import_module)
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    assert deep_gemm._import_deep_gemm() is module
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    with pytest.raises(RuntimeError, match="use_deterministic_algorithms"):
        deep_gemm._import_deep_gemm()


@pytest.mark.cpu_test
def test_bi_configures_the_imported_deep_gemm_provider(monkeypatch):
    """MHC and grouped experts share the selected library's numerical mode."""
    from vllm import envs
    from vllm.utils import deep_gemm

    calls: list[bool] = []
    module = SimpleNamespace(use_deterministic_algorithms=calls.append)
    monkeypatch.setattr(deep_gemm.importlib, "import_module", lambda _: module)
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    assert deep_gemm._import_deep_gemm() is module
    assert calls == []
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    assert deep_gemm._import_deep_gemm() is module
    assert calls == [True]


def _block_fp8_layer(n: int, k: int, scale_name: str = "weight_scale"):
    layer = SimpleNamespace(
        weight=torch.empty((n, k), dtype=torch.float8_e4m3fn),
        weight_block_size=[128, 128],
        deep_gemm_warmup_provider=object.__new__(DeepGemmFp8BlockScaledMMKernel),
    )
    setattr(layer, scale_name, torch.empty((n // 128, k // 128), dtype=torch.float32))
    return layer


def _run_warmup(monkeypatch, layers) -> list[dict]:
    calls: list[dict] = []
    monkeypatch.setattr(
        deep_gemm_warmup,
        "get_mk_alignment_for_contiguous_layout",
        lambda: [128, 128],
    )
    monkeypatch.setattr(
        deep_gemm_warmup,
        "_deepgemm_fp8_gemm_nt_warmup",
        lambda **kwargs: calls.append(kwargs),
    )
    model = SimpleNamespace(modules=lambda: iter(layers))
    deep_gemm_warmup.deepgemm_fp8_gemm_nt_warmup(model, max_tokens=16)
    return calls


@pytest.mark.parametrize("scale_name", ["weight_scale", "weight_scale_inv"])
def test_registered_deep_gemm_layer_is_warmed_up(monkeypatch, scale_name) -> None:
    """Any layer stamped by the DeepGEMM kernel is warmed, regardless of the
    quantization method that owns it or the name of its scale parameter."""
    layer = _block_fp8_layer(256, 128, scale_name)

    calls = _run_warmup(monkeypatch, [layer])

    assert len(calls) == 1
    assert calls[0]["w"] is layer.weight
    assert calls[0]["ws"] is getattr(layer, scale_name)


def test_n_multiple_of_64_matches_kernel_selection(monkeypatch) -> None:
    """DeepGEMM accepts N % 64 == 0 (e.g. DeepSeek kv_a_proj_with_mqa, N=576),
    so warmup must not require N % 128 == 0."""
    assert len(_run_warmup(monkeypatch, [_block_fp8_layer(576, 256)])) == 1


def test_unstamped_and_mismatched_block_layers_are_skipped(monkeypatch) -> None:
    mismatched = _block_fp8_layer(256, 128)
    mismatched.weight_block_size = [1, 32]
    layers = [SimpleNamespace(), mismatched, _block_fp8_layer(256, 128)]

    calls = _run_warmup(monkeypatch, layers)

    assert len(calls) == 1
    assert calls[0]["w"] is layers[-1].weight


@pytest.mark.parametrize("is_bmm", [False, True])
def test_kernel_registers_itself_as_warmup_provider(is_bmm) -> None:
    """Bmm layers bypass the kernel at runtime, so they must not be warmed."""
    kernel = object.__new__(DeepGemmFp8BlockScaledMMKernel)
    kernel.is_deep_gemm_supported = False
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.empty((256, 128), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.empty((2, 1), dtype=torch.float32), requires_grad=False
    )
    layer.weight_block_size = [128, 128]
    layer.is_bmm = is_bmm

    kernel.process_weights_after_loading(layer)

    provider = getattr(layer, "deep_gemm_warmup_provider", None)
    assert provider is (None if is_bmm else kernel)


def test_mega_moe_predicate_matches_only_native_mega_moe_layers(monkeypatch) -> None:
    """Without the MegaMoE model modules loaded, nothing matches and nothing is
    imported; with them loaded, only DeepSeek-V4 and Kimi K3 MegaMoE layers match."""
    name = "vllm.models.deepseek_v4.nvidia.model"
    kimi_name = "vllm.models.kimi_k3.nvidia.model"
    monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delitem(sys.modules, kimi_name, raising=False)
    other = SimpleNamespace(use_native_mega_moe=True, use_mega_moe=True)
    assert not deep_gemm_warmup._mega_moe_may_use_deep_gemm(other)
    assert name not in sys.modules and kimi_name not in sys.modules

    moe_cls = type("DeepseekV4MoE", (), {})
    monkeypatch.setitem(sys.modules, name, SimpleNamespace(DeepseekV4MoE=moe_cls))
    native, fused = moe_cls(), moe_cls()
    native.use_native_mega_moe, fused.use_native_mega_moe = True, False
    assert deep_gemm_warmup._mega_moe_may_use_deep_gemm(native)
    assert not deep_gemm_warmup._mega_moe_may_use_deep_gemm(fused)
    assert not deep_gemm_warmup._mega_moe_may_use_deep_gemm(other)

    kimi_cls = type("KimiMoE", (), {})
    monkeypatch.setitem(sys.modules, kimi_name, SimpleNamespace(KimiMoE=kimi_cls))
    kimi_mega, kimi_fused = kimi_cls(), kimi_cls()
    kimi_mega.use_mega_moe, kimi_fused.use_mega_moe = True, False
    assert deep_gemm_warmup._mega_moe_may_use_deep_gemm(kimi_mega)
    assert not deep_gemm_warmup._mega_moe_may_use_deep_gemm(kimi_fused)


def test_mega_moe_m_values_reach_mega_gate(monkeypatch) -> None:
    """Each bf16_mega_gate config is warmed with an M above the GateLinear
    threshold (1 for 128 experts), so the forward actually reaches the kernel."""
    dg = SimpleNamespace(
        get_bf16_mega_gate_config=lambda m, *_: {"num_sms": -(-m // 16)},
        get_block_m_for_mega_moe=lambda *_: 16,
    )
    monkeypatch.setattr(deep_gemm_warmup, "_import_deep_gemm", lambda: dg)
    monkeypatch.setattr(deep_gemm_warmup, "_mega_moe_uses_mega_gate", lambda _: True)
    monkeypatch.setattr(
        deep_gemm_warmup, "get_ep_group", lambda: SimpleNamespace(world_size=8)
    )
    buffer = SimpleNamespace(num_max_tokens_per_rank=64)
    module = SimpleNamespace(
        gate=SimpleNamespace(input_size=4096, output_size=128),
        experts=SimpleNamespace(
            top_k=6, num_experts=128, get_symm_buffer=lambda: buffer
        ),
        use_sequence_parallel=False,
    )
    assert deep_gemm_warmup._get_fp8_fp4_mega_moe_m_values(module, 40) == [16, 32, 40]


@pytest.mark.cpu_test
def test_mxfp4_experts_are_warmed_with_their_gemm_recipes(monkeypatch) -> None:
    """MXFP4 experts compile at startup, per M bucket, with the activation and
    weight recipes their grouped GEMMs pass at runtime."""
    from vllm.model_executor.layers.fused_moe import MoERunner
    from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
        DeepGemmFP4Experts,
    )

    num_experts, n, k = 4, 512, 256
    w13 = torch.empty((num_experts, n, k // 2), dtype=torch.uint8)
    w2 = torch.empty((num_experts, k, n // 4), dtype=torch.uint8)
    experts = object.__new__(DeepGemmFP4Experts)
    experts._act_block_k = 32
    experts.quant_config = SimpleNamespace(
        w1_scale=torch.empty(1), w2_scale=torch.empty(1)
    )
    runner = object.__new__(MoERunner)
    runner.__dict__["routed_experts"] = SimpleNamespace(
        w13_weight=w13,
        w2_weight=w2,
        top_k=6,
        quant_method=SimpleNamespace(
            get_fused_moe_quant_config=lambda layer: None,
            moe_kernel=SimpleNamespace(impl=SimpleNamespace(fused_experts=experts)),
        ),
    )
    cases = [(128, 128, torch.zeros(128)), (256, 128, torch.zeros(256))]
    monkeypatch.setattr(
        deep_gemm_warmup, "_get_grouped_gemm_params", lambda *_: (256, 128, cases)
    )
    monkeypatch.setattr(
        deep_gemm_warmup,
        "current_workspace_manager",
        lambda: SimpleNamespace(
            get_simultaneous=lambda *specs: [torch.empty(s, dtype=d) for s, d in specs]
        ),
    )
    calls: list[tuple] = []
    monkeypatch.setattr(
        deep_gemm_warmup,
        "m_grouped_fp8_fp4_gemm_nt_contiguous",
        lambda a, b, out, ids, **kw: calls.append((a[0].shape, a[1].shape, b[0], kw)),
    )
    deep_gemm_warmup.GROUPED_FP8_FP4_GEMM_NT_CONTIGUOUS_WARMUP_CACHE.clear()
    model = SimpleNamespace(modules=lambda: iter([runner, SimpleNamespace()]))

    assert deep_gemm_warmup._count_warmup_iterations(model, 256) == 4
    deep_gemm_warmup.deepgemm_grouped_fp8_fp4_gemm_nt_contiguous_warmup(model, 256)
    deep_gemm_warmup.deepgemm_grouped_fp8_fp4_gemm_nt_contiguous_warmup(model, 256)

    recipes = {"recipe_a": (1, 32), "recipe_b": (1, 32)}
    assert [(a, s, kw) for a, s, _, kw in calls] == [
        ((128, k), (128, k // 32), recipes),
        ((256, k), (256, k // 32), recipes),
        ((128, n // 2), (128, n // 2 // 32), recipes),
        ((256, n // 2), (256, n // 2 // 32), recipes),
    ]
    assert [w.shape for *_, w, _ in calls] == [w13.shape] * 2 + [w2.shape] * 2
    assert deep_gemm_warmup._count_warmup_iterations(model, 256) == 0
