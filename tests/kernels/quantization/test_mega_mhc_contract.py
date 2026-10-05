# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

import vllm.envs as envs
from vllm.utils import deep_gemm

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.mark.parametrize("has_external", [False, True])
@pytest.mark.parametrize("vendored", [False, True])
def test_mega_provider_is_explicit_and_keeps_other_apis(
    monkeypatch, has_external, vendored
):
    external_call = lambda **_: None
    vendored_call = lambda **_: None
    external = (
        SimpleNamespace(mega_mhc=external_call) if has_external else SimpleNamespace()
    )
    imported = []
    monkeypatch.setattr(envs, "VLLM_DEEP_GEMM_MEGA_MHC_USE_VENDORED", vendored)

    def load(name):
        imported.append(name)
        return SimpleNamespace(mega_mhc=vendored_call)

    monkeypatch.setattr(deep_gemm.importlib, "import_module", load)
    actual = deep_gemm._resolve_mega_mhc_impl(external)
    assert actual is (
        vendored_call if vendored else external_call if has_external else None
    )
    assert imported == (["vllm.third_party.deep_gemm"] if vendored else [])
    assert getattr(external, "mega_mhc", None) is (
        external_call if has_external else None
    )


@pytest.mark.parametrize("implementation", [None, 1, lambda: None])
def test_capability_uses_generic_mega_call_provider(monkeypatch, implementation):
    monkeypatch.setattr(deep_gemm, "_lazy_init", lambda: None)
    monkeypatch.setattr(deep_gemm, "_mega_mhc_impl", implementation)
    assert deep_gemm.has_deep_gemm_mega_mhc() is callable(implementation)


def test_explicit_mega_vendor_missing_api_fails_loud(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_DEEP_GEMM_MEGA_MHC_USE_VENDORED", True)
    monkeypatch.setattr(
        deep_gemm.importlib, "import_module", lambda _: SimpleNamespace()
    )
    with pytest.raises(RuntimeError, match="missing mega_mhc"):
        deep_gemm._resolve_mega_mhc_impl(SimpleNamespace(mega_mhc=lambda: None))


@pytest.mark.parametrize("rows", [0, 1, 127, 128, 129, 257])
def test_bi_preserves_caller_buffers_and_token_order(monkeypatch, rows):
    import torch

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(deep_gemm, "_lazy_init", lambda: None)
    torch.manual_seed(1127)
    inputs = {
        "x": torch.randn(rows, 8),
        "residual": torch.randn(rows, 4, 8),
        "shifted_prev_mix": torch.randn(rows, 4, 1),
        "post_mix": torch.randn(rows, 4, 1),
        "comb_res_mix": torch.randn(rows, 4, 4),
    }
    originals = {key: value.clone() for key, value in inputs.items()}
    expected = {
        "new_residual": inputs["residual"] + inputs["x"].unsqueeze(1),
        "new_prev_mix": inputs["shifted_prev_mix"] + 1,
        "new_post_mix": inputs["post_mix"] + 1,
        "new_comb_res_mix": inputs["comb_res_mix"] + 1,
        "y_bf16": inputs["x"] + 1,
    }
    outputs = {key: torch.full_like(value, -17) for key, value in expected.items()}
    owners = {key: value.data_ptr() for key, value in outputs.items()}
    metadata = torch.randn(24, 32)
    calls = []

    def provider(**kw):
        # Plumbing reference only: real kernel equality is tested on SM100.
        assert kw["fn"] is metadata
        calls.append(kw["x"].shape[0])
        kw["new_residual"].copy_(kw["residual"] + kw["x"].unsqueeze(1))
        for dst, src in (
            ("new_prev_mix", "shifted_prev_mix"),
            ("new_post_mix", "post_mix"),
            ("new_comb_res_mix", "comb_res_mix"),
            ("y_bf16", "x"),
        ):
            kw[dst].copy_(kw[src] + 1)

    monkeypatch.setattr(deep_gemm, "_mega_mhc_impl", provider)
    deep_gemm.mega_mhc(**inputs, **outputs, fn=metadata)
    assert calls == [128] * ((rows + 127) // 128)
    for key, value in inputs.items():
        assert torch.equal(value, originals[key])
    for key, value in outputs.items():
        assert value.data_ptr() == owners[key]
        assert torch.equal(value, expected[key])


def test_non_bi_mega_preserves_original_call(monkeypatch):
    marker = object()
    calls = []
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(deep_gemm, "_lazy_init", lambda: None)
    monkeypatch.setattr(
        deep_gemm, "_mega_mhc_impl", lambda *a, **k: calls.append((a, k))
    )
    deep_gemm.mega_mhc(marker, unchanged=marker)
    assert calls == [((marker,), {"unchanged": marker})]


def test_bi_mega_unknown_buffer_abi_fails_loud(monkeypatch):
    import torch

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(deep_gemm, "_lazy_init", lambda: None)
    monkeypatch.setattr(deep_gemm, "_mega_mhc_impl", lambda **_: pytest.fail("kernel"))
    with pytest.raises(ValueError, match="named buffers"):
        deep_gemm.mega_mhc(torch.zeros(1, 8))
    with pytest.raises(ValueError, match="residual"):
        deep_gemm.mega_mhc(x=torch.zeros(1, 8))


@pytest.mark.parametrize("width", [1024, 5120])
def test_sm100_bi_mega_all_outputs_equal_across_batch_sizes(monkeypatch, width):
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 Mega-mHC")
    from vllm.third_party import deep_gemm as vendor

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(deep_gemm, "_lazy_init", lambda: None)
    monkeypatch.setattr(deep_gemm, "_mega_mhc_impl", vendor.mega_mhc)
    torch.manual_seed(1117)
    n = 33
    data = {
        "x": torch.randn(n, width, device="cuda", dtype=torch.bfloat16),
        "residual": torch.randn(n, 4, width, device="cuda", dtype=torch.bfloat16),
        "shifted_prev_mix": torch.randn(n, 4, 1, device="cuda"),
        "post_mix": torch.randn(n, 4, 1, device="cuda"),
        "comb_res_mix": torch.randn(n, 4, 4, device="cuda"),
    }
    metadata = {
        "fn": torch.randn(24, 4 * width, device="cuda") * 0.01,
        "mix_scales": torch.ones(3, device="cuda"),
        "mix_bases": torch.randn(24, device="cuda"),
        "hc_mult": 4,
        "hc_norm_eps": 1e-20,
        "hc_pre_eps": 1e-6,
        "hc_post_scale": 2.0,
        "sinkhorn_eps": 1e-6,
        "num_sinkhorn_iters": 20,
        "rmsnorm_weight": torch.ones(width, device="cuda", dtype=torch.bfloat16),
        "rmsnorm_eps": 1e-20,
        "rmsnorm_scale": 1.0,
    }

    def compute(rows):
        values = {
            key: value.repeat((rows + n - 1) // n, *([1] * (value.ndim - 1)))[:rows]
            for key, value in data.items()
        }
        outputs = {
            "new_residual": torch.empty_like(values["residual"]),
            "new_prev_mix": torch.empty_like(values["shifted_prev_mix"]),
            "new_post_mix": torch.empty_like(values["post_mix"]),
            "new_comb_res_mix": torch.empty_like(values["comb_res_mix"]),
            "y_bf16": torch.empty_like(values["x"]),
        }
        deep_gemm.mega_mhc(**values, **metadata, **outputs)
        return outputs

    baseline = compute(n)
    for rows in (1, 17, 33, 127, 128, 129, 257):
        outputs = compute(rows)
        common = min(n, rows)
        for key in outputs:
            assert torch.equal(outputs[key][:common], baseline[key][:common]), (
                key,
                rows,
            )
