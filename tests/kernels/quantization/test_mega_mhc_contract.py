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
