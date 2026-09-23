# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch

import pytest
import torch


def test_nemotron_h_lm_head_receives_quant_config():
    from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM

    mock_quant_config = Mock()

    mock_hf_config = Mock()
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_config = mock_hf_config
    mock_vllm_config.model_config.dtype = None
    mock_vllm_config.scheduler_config = Mock()
    mock_vllm_config.quant_config = mock_quant_config

    with (
        patch("vllm.model_executor.models.nemotron_h.NemotronHModel") as MockModel,
        patch("vllm.model_executor.models.nemotron_h.ParallelLMHead") as MockLMHead,
        patch("vllm.model_executor.models.nemotron_h.LogitsProcessor"),
    ):
        MockModel.return_value.make_empty_intermediate_tensors = Mock()
        MockModel.return_value.has_moe = False

        NemotronHForCausalLM(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config


@pytest.mark.parametrize("batch_invariant", [False, True])
@pytest.mark.parametrize("kind", ["rms", "residual", "gated"])
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA norm kernels")
@torch.inference_mode()
def test_nemotron_norm_dispatch(monkeypatch, batch_invariant, kind, tp_size, compiled):
    """BI alone selects shared kernels; otherwise preserve native norm results."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.mamba import mamba_mixer2
    from vllm.model_executor.models import nemotron_h as model
    from vllm.model_executor.models.nemotron_h_alignment import (
        gated_forward,
        rms_forward,
    )

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", str(int(batch_invariant)))
    monkeypatch.setattr(model, "get_tensor_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(
        mamba_mixer2, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    monkeypatch.setattr(mamba_mixer2, "get_tensor_model_parallel_rank", lambda: 0)
    gated = kind == "gated"
    args = (64, 8) if gated else (64,)
    default_cls = model.Mixer2RMSNormGated if gated else model.RMSNorm
    norm_cls = model.NemotronHGatedRMSNorm if gated else model.NemotronHRMSNorm
    with set_current_vllm_config(VllmConfig()):
        norm = norm_cls(*args).to(device="cuda", dtype=torch.bfloat16)
        norm.weight.uniform_(0.5, 1.5)
        default = default_cls(*args).to(device="cuda", dtype=torch.bfloat16)
        default.load_state_dict(norm.state_dict(), strict=True)
        width = 64 // tp_size if gated else 64
        x = torch.randn(3, width, device="cuda", dtype=torch.bfloat16)
        extra = torch.randn_like(x) if kind != "rms" else None
        if batch_invariant and tp_size != 1:
            with pytest.raises(ValueError, match="TP=1"):
                norm(x, extra)
            return
        if not batch_invariant:
            if compiled:
                default = torch.compile(default, fullgraph=True)
            expected = default(x.clone(), extra.clone() if extra is not None else None)
        elif gated:
            expected = gated_forward(
                x, extra, norm.weight, norm.group_size, norm.variance_epsilon
            )
        else:
            expected = rms_forward(x, norm.weight, norm.variance_epsilon, extra)
        if compiled:
            norm = torch.compile(norm, fullgraph=True)
        actual = norm(x, extra)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
