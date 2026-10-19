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


def test_lora_branch_reads_the_unquantized_activation():
    import torch

    from vllm.lora.layers.base_linear import _lora_input
    from vllm.model_executor.layers.fusion.quant_activation import (
        QuantizedActivation,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
    )

    orig = torch.randn(4, 64)
    qa = QuantizedActivation(orig, orig, orig.dtype, orig.shape, kMxfp8Dynamic)
    with pytest.raises(ValueError):
        _lora_input(qa)
    qa.orig = orig
    assert _lora_input(qa) is orig and _lora_input(orig) is orig


def test_in_memory_adapter_update_keeps_caller_tensors():
    import torch

    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.request import LoRARequest
    from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

    r, E = 4, 3
    tensors = {
        "layers.20.attn.wq_b.lora_A.weight": torch.randn(r, 16),
        "layers.20.attn.wq_b.lora_B.weight": torch.randn(32, r),
        "layers.20.ffn.experts.lora_gate_up_A.weight": torch.randn(r, 16),
        "layers.20.ffn.experts.lora_down_B.weight": torch.randn(16, r),
        "layers.20.ffn.shared_experts.w2.lora_A.weight": torch.randn(r, 8),
        "layers.20.ffn.shared_experts.w2.lora_B.weight": torch.randn(16, r),
        **{
            f"layers.20.ffn.experts.{e}.lora_gate_up_B.weight": torch.randn(8, r)
            for e in range(E)
        },
        **{
            f"layers.20.ffn.experts.{e}.lora_down_A.weight": torch.randn(r, 4)
            for e in range(E)
        },
    }
    tensors = {k: v.bfloat16() for k, v in tensors.items()}
    before = {k: v.clone() for k, v in tensors.items()}
    added: list = []

    def add_adapter(lora):
        # As the model manager does: scale B by alpha / r in place.
        for weights in lora.loras.values():
            weights.optimize()
        added.append(lora)

    manager = SimpleNamespace(
        model=DeepseekV41ForCausalLM.__new__(DeepseekV41ForCausalLM),
        remove_adapter=lambda i: None,
        add_adapter=add_adapter,
        activate_adapter=lambda i: None,
    )
    manager.model.hf_to_vllm_mapper = _make_deepseek_v4_vl_weights_mapper(
        "fp4", "weight_scale"
    )
    fake = SimpleNamespace(
        _adapter_manager=manager,
        _lora_model_cls=LoRAModel,
        vocab_size=None,
        lora_config=SimpleNamespace(max_lora_rank=r, lora_dtype=torch.bfloat16),
    )
    LRUCacheWorkerLoRAManager.add_adapter_tensors(
        fake,
        LoRARequest("policy-v7", 1, "mem://policy"),
        tensors,
        {"r": r, "lora_alpha": 2 * r, "target_modules": ["wq_b"]},
    )
    assert all(torch.equal(tensors[k], before[k]) for k in tensors)
    loras = added[0].loras
    prefix = "language_model.model.layers.20."
    gate_up = loras[prefix + "ffn.experts.gate_up"]
    assert gate_up.lora_b.shape == (E, 8, r)
    assert torch.equal(
        gate_up.lora_b[1], 2 * before["layers.20.ffn.experts.1.lora_gate_up_B.weight"]
    )
    assert loras[prefix + "ffn.experts.down"].lora_a.shape == (E, r, 4)
    assert prefix + "ffn.experts.shared_w2" in loras
