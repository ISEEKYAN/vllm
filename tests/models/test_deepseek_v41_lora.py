# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 LoRA wiring: the registered class supports LoRA, adapter keys
in checkpoint naming reach the language model, and LoRA targets reach an
Engram WKV only when it is named in full."""

from types import SimpleNamespace

import pytest

import vllm.models.deepseek_v41.common.engram as engram
from vllm.lora.utils import parse_fine_tuned_lora_name
from vllm.model_executor.models.interfaces import supports_lora
from vllm.models.deepseek_v41.nvidia.model import DeepseekV41LLMForCausalLM
from vllm.models.deepseek_v41.nvidia.vl_model import (
    DeepseekV41ForCausalLM,
    _make_deepseek_v4_vl_weights_mapper,
)

PACKED = DeepseekV41LLMForCausalLM.packed_modules_mapping
ATTN = [
    f"language_model.model.layers.{i}.attn.{m}"
    for i in range(20, 40)
    for m in ("fused_wqa_wkv", "wq_b", "wo_b")
]


def test_registered_class_supports_lora():
    assert supports_lora(DeepseekV41ForCausalLM)
    assert DeepseekV41ForCausalLM.packed_modules_mapping == PACKED
    assert DeepseekV41ForCausalLM.lora_skip_prefixes == ["mtp."]


@pytest.mark.parametrize("module", ["wq_a", "wkv", "wq_b", "wo_b"])
def test_checkpoint_named_adapter_keys_reach_the_language_model(module):
    mapper = _make_deepseek_v4_vl_weights_mapper("fp4", "weight_scale")
    name = f"base_model.model.layers.20.attn.{module}.lora_A.weight"
    parsed, is_a = parse_fine_tuned_lora_name(name, mapper.get_rename_mapper())
    assert (parsed, is_a) == (f"language_model.model.layers.20.attn.{module}", True)


@pytest.mark.parametrize(
    "targets,expected",
    [
        (ATTN, False),
        (["language_model.model.layers.1.engram.wkv"], True),
        (None, ValueError),
        (["wkv"], ValueError),
        (["fused_wqa_wkv"], ValueError),
    ],
)
def test_engram_batch_kept_unless_wkv_named(monkeypatch, targets, expected):
    config = SimpleNamespace(lora_config=SimpleNamespace(target_modules=targets))
    monkeypatch.setattr(engram, "get_current_vllm_config", lambda: config)
    layers = [
        SimpleNamespace(
            wkv=SimpleNamespace(prefix=f"language_model.model.layers.{i}.engram.wkv")
        )
        for i in (1, 14)
    ]
    if expected is ValueError:
        with pytest.raises(ValueError):
            engram._lora_wraps_wkv(layers, PACKED)
    else:
        assert engram._lora_wraps_wkv(layers, PACKED) is expected
    monkeypatch.setattr(
        engram, "get_current_vllm_config", lambda: SimpleNamespace(lora_config=None)
    )
    assert engram._lora_wraps_wkv(layers, PACKED) is False


@pytest.mark.parametrize(
    "targets,ok",
    [
        (["language_model.model.layers.20.ffn.experts.gate_up"], True),
        (["language_model.model.layers.39.attn.wq_b"], True),  # another PP stage
        (["experts.gate_up"], False),  # would leave the fused experts in place
        (["wq_b"], False),
        (None, False),
    ],
)
def test_lora_targets_must_be_full_module_names(targets, ok):
    import torch.nn as nn

    from vllm.model_executor.models.utils import PPMissingLayer
    from vllm.models.deepseek_v41.nvidia.vl_model import _check_lora_targets

    model = nn.Module()
    layers = nn.ModuleDict({"20": nn.Module(), "39": PPMissingLayer()})
    layers["20"].ffn = nn.Module()
    layers["20"].ffn.experts = nn.Module()
    layers["20"].ffn.experts.gate_up = nn.Linear(1, 1)
    model.language_model = nn.Module()
    model.language_model.model = nn.Module()
    model.language_model.model.layers = layers
    config = SimpleNamespace(lora_config=SimpleNamespace(target_modules=targets))
    if ok:
        _check_lora_targets(config, model)
    else:
        with pytest.raises(ValueError):
            _check_lora_targets(config, model)
    _check_lora_targets(SimpleNamespace(lora_config=None), model)
