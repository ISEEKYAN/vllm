# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`engram_hash_layer_ids` keeps the release hash multipliers when a model is
built from renumbered release layers; without it the table ids seed them."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41.common import engram as engram_ops
from vllm.models.deepseek_v41.common.engram import EngramLayout, NgramHashState

COMPRESSED_VOCAB = 99092


def _config(layer_ids, **extra):
    return SimpleNamespace(
        engram_layer_ids=list(layer_ids),
        engram_num_embeddings=[384006168, 384016682],
        engram_max_ngram_size=4,
        engram_n_heads=8,
        engram_head_dim=256,
        engram_compressed_vocab_size=COMPRESSED_VOCAB,
        engram_pad_token_id=2,
        engram_vocab_size=16000000,
        **extra,
    )


def _hash_state(monkeypatch, layout):
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **k: object()
    )
    monkeypatch.setattr(
        engram_ops,
        "build_compressed_token_map",
        lambda tokenizer: ([0, 1, 2, 3], COMPRESSED_VOCAB),
    )
    vllm_config = SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(
            tokenizer="unused", trust_remote_code=False, revision=None
        ),
    )
    return NgramHashState(vllm_config, layout, SimpleNamespace(block_size=16))


def test_hash_layer_ids_default_to_table_layer_ids():
    layout = EngramLayout(_config((1, 14)))
    assert layout.hash_layer_ids == (1, 14)


def test_renumbered_tables_keep_release_primes_and_hash(monkeypatch):
    release = EngramLayout(_config((1, 14)))
    renumbered = EngramLayout(_config((0, 1), engram_hash_layer_ids=[1, 14]))

    # Tables keep their slots, so the prime bucket layout is unchanged.
    assert renumbered.primes == release.primes
    torch.testing.assert_close(renumbered.offsets, release.offsets, rtol=0, atol=0)
    assert renumbered.hash_layer_ids == (1, 14)

    state = _hash_state(monkeypatch, renumbered)
    expected = engram_ops.compute_hash_multipliers((1, 14), 4, COMPRESSED_VOCAB)
    torch.testing.assert_close(state.multipliers, expected, rtol=0, atol=0)


def test_hash_layer_ids_must_match_table_count():
    with pytest.raises(AssertionError):
        EngramLayout(_config((0, 1), engram_hash_layer_ids=[14]))
