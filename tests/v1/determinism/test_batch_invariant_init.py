from types import SimpleNamespace

from vllm.model_executor.layers import batch_invariant
from vllm.utils import deep_gemm as deep_gemm_utils


def _torch_backends():
    return SimpleNamespace(
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(fp32_precision=None)),
            cudnn=SimpleNamespace(
                conv=SimpleNamespace(fp32_precision=None),
                rnn=SimpleNamespace(fp32_precision=None),
            ),
        )
    )


def test_init_batch_invariance_preserves_environment_default(monkeypatch):
    calls = []
    torch = _torch_backends()
    monkeypatch.setattr(batch_invariant, "torch", torch)
    monkeypatch.setattr(
        batch_invariant, "envs", SimpleNamespace(VLLM_BATCH_INVARIANT=False)
    )
    monkeypatch.setattr(
        batch_invariant,
        "override_envs_for_invariance",
        lambda: calls.append("env"),
    )
    monkeypatch.setattr(
        batch_invariant,
        "enable_batch_invariant_mode",
        lambda: calls.append("mode"),
    )

    batch_invariant.init_batch_invariance()

    assert calls == []
    assert torch.backends.cuda.matmul.fp32_precision is None


def test_init_batch_invariance_can_be_required_explicitly(monkeypatch):
    calls = []
    torch = _torch_backends()
    monkeypatch.setattr(batch_invariant, "torch", torch)
    monkeypatch.setattr(
        batch_invariant, "envs", SimpleNamespace(VLLM_BATCH_INVARIANT=False)
    )
    monkeypatch.setattr(
        batch_invariant,
        "override_envs_for_invariance",
        lambda: calls.append("env"),
    )
    monkeypatch.setattr(
        batch_invariant,
        "enable_batch_invariant_mode",
        lambda: calls.append("mode"),
    )

    batch_invariant.init_batch_invariance(force=True)

    assert calls == ["env", "mode"]
    assert torch.backends.cuda.matmul.fp32_precision == "ieee"
    assert torch.backends.cudnn.conv.fp32_precision == "ieee"
    assert torch.backends.cudnn.rnn.fp32_precision == "ieee"


def test_deep_gemm_invariance_initializes_runtime_before_enabling(monkeypatch):
    calls = []
    deep_gemm = SimpleNamespace(
        set_batch_invariant=lambda enabled: calls.append(("set", enabled)),
        get_batch_invariant=lambda: True,
    )
    monkeypatch.setattr(
        deep_gemm_utils, "is_deep_gemm_supported", lambda: True
    )
    monkeypatch.setattr(
        deep_gemm_utils, "_lazy_init", lambda: calls.append("init")
    )
    monkeypatch.setattr(deep_gemm_utils, "_import_deep_gemm", lambda: deep_gemm)
    monkeypatch.setattr(
        deep_gemm_utils, "supports_deep_gemm_batch_invariance", lambda: True
    )

    deep_gemm_utils.enable_deep_gemm_batch_invariance()

    assert calls == ["init", ("set", True)]
