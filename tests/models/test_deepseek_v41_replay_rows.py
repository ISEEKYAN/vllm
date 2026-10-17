# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decoder SWA bounded replay equals a full prefill with N * (W - 1) + 1 rows.

A float64 stand-in for the layers past the last KV source: every layer attends
over its own sliding window of the previous layer's rows. Replaying the last R
rows (each window clipped at the replay start, as the trimmed replay layers'
KV starts there) must reproduce the full prefill's last row and every layer's
last W - 1 rows of window KV, which is all the decode steps read.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v41.decoder_replay_layers import (
    check_decoder_replay_exact,
    decoder_replay_rows,
)

WINDOW = 8


def _layer(x, weight, window):
    """Row r attends rows max(0, r - window + 1)..r of ``x``: a replay batch
    holds only the kept rows, so its windows stop at the replay start."""
    rows = []
    for r in range(x.shape[0]):
        keys = x[max(0, r - window + 1) : r + 1]
        scores = torch.softmax(keys @ weight @ x[r], 0)
        rows.append(torch.tanh(scores @ keys + x[r]))
    return torch.stack(rows)


def _stack(x, weights, window):
    """Each layer's window KV (its input rows) and the last layer's output."""
    kv = []
    for weight in weights:
        kv.append(x)
        x = _layer(x, weight, window)
    return kv, x


def _decode_view(kv, out, window):
    """What decode reads: each layer's last W - 1 KV rows and the last row."""
    return [k[-(window - 1) :] for k in kv], out[-1]


@pytest.mark.parametrize("layers_after_cut", [1, 2, 3, 5])
@pytest.mark.parametrize("length", [WINDOW - 3, WINDOW + 1, 3 * WINDOW + 5, 9 * WINDOW])
def test_replay_rows_equal_full_prefill(layers_after_cut, length):
    generator = torch.Generator().manual_seed(7 * layers_after_cut + length)
    width = 4
    cut_out = torch.randn(length, width, generator=generator, dtype=torch.float64)
    weights = [
        torch.randn(width, width, generator=generator, dtype=torch.float64) / 2
        for _ in range(layers_after_cut)
    ]
    full_kv, full_out = _stack(cut_out, weights, WINDOW)
    kept = min(length, decoder_replay_rows(layers_after_cut, WINDOW))
    kv, out = _stack(cut_out[length - kept :], weights, WINDOW)
    want_kv, want_last = _decode_view(full_kv, full_out, WINDOW)
    got_kv, got_last = _decode_view(kv, out, WINDOW)
    assert torch.equal(got_last, want_last)
    for got, want in zip(got_kv, want_kv):
        assert torch.equal(got, want)


@pytest.mark.parametrize("layers_after_cut", [2, 3, 5])
def test_one_row_fewer_or_one_window_is_not_exact(layers_after_cut):
    """The old trim (one window) and R - 1 rows both diverge from a full prefill."""
    generator = torch.Generator().manual_seed(11)
    width, length = 4, 9 * WINDOW
    cut_out = torch.randn(length, width, generator=generator, dtype=torch.float64)
    weights = [
        torch.randn(width, width, generator=generator, dtype=torch.float64) / 2
        for _ in range(layers_after_cut)
    ]
    full_kv, full_out = _stack(cut_out, weights, WINDOW)
    want_kv, want_last = _decode_view(full_kv, full_out, WINDOW)
    for kept in (WINDOW, decoder_replay_rows(layers_after_cut, WINDOW) - 1):
        kv, out = _stack(cut_out[length - kept :], weights, WINDOW)
        got_kv, got_last = _decode_view(kv, out, WINDOW)
        exact = torch.equal(got_last, want_last) and all(
            torch.equal(g, w) for g, w in zip(got_kv, want_kv)
        )
        assert not exact, kept


def test_replay_rows_formula():
    assert decoder_replay_rows(1, 128) == 128  # one SWA layer: the old window
    assert decoder_replay_rows(2, 128) == 255  # 6L/9L proxies
    assert decoder_replay_rows(19, 128) == 2414  # release 40L, cut at layer 20
    for bad in ((0, 128), (2, 0)):
        with pytest.raises(ValueError):
            decoder_replay_rows(*bad)


def test_kept_lens_trim_to_replay_rows():
    from vllm.models.deepseek_v41.nvidia.model_state import replay_kept_lens

    query_lens = np.array([1, 254, 255, 256, 17408, 17408])
    keeps_rows = np.array([False, False, False, False, False, True])
    assert replay_kept_lens(query_lens, keeps_rows, 255).tolist() == [
        1,
        254,
        255,
        255,
        255,
        17408,
    ]


@pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1",
    reason="runs the Triton gather kernel on CPU under TRITON_INTERPRET=1",
)
def test_gather_kv_start_is_first_kept_row():
    """A trimmed request's replay-layer KV starts at its first kept row."""
    from vllm.models.deepseek_v41.nvidia.model_state import (
        _gather_replay_batch_kernel,
        replay_kept_lens,
    )

    replay_rows = 255
    query_lens = np.array([300, 100, 1000], dtype=np.int32)
    seq_lens = torch.tensor([300, 130, 1000], dtype=torch.int32)  # req 1: chunk 2
    kept = replay_kept_lens(query_lens, np.zeros(3, dtype=bool), replay_rows)
    qsl = np.concatenate([[0], np.cumsum(query_lens)]).astype(np.int32)
    kept_qsl = np.concatenate([[0], np.cumsum(kept)]).astype(np.int32)
    num_tokens, num_kept = int(qsl[-1]), int(kept_qsl[-1])
    positions = torch.cat(
        [torch.arange(s - q, s) for s, q in zip(seq_lens.tolist(), query_lens)]
    ).to(torch.int64)
    slots = torch.arange(num_tokens, dtype=torch.int64)[None]
    rows = torch.empty(num_kept, dtype=torch.int64)
    kept_loc = torch.empty(4, dtype=torch.int32)
    kept_positions = torch.empty(num_kept, dtype=torch.int64)
    kept_slots = torch.empty(1, num_kept, dtype=torch.int64)
    kv_start = torch.empty(3, dtype=torch.int32)
    _gather_replay_batch_kernel[(3,)](
        torch.from_numpy(qsl),
        torch.from_numpy(qsl - kept_qsl),
        seq_lens,
        torch.zeros(3, dtype=torch.int32),
        positions,
        slots,
        slots.stride(0),
        rows,
        kept_loc,
        kept_positions,
        kept_slots,
        kept_slots.stride(0),
        kv_start,
        replay_rows,
        NUM_GROUPS=1,
        BLOCK=64,
    )
    assert kept.tolist() == [255, 100, 255]
    assert kept_loc.tolist() == kept_qsl.tolist()
    # Trimmed requests start at seq_len - replay_rows; the untrimmed one at 0.
    assert kv_start.tolist() == [300 - 255, 0, 1000 - 255]
    assert kept_positions[:255].tolist() == list(range(45, 300))
    assert rows[255:355].tolist() == list(range(300, 400))


def _vllm_config(requested, prefix_caching, batched, max_len):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            swa_bounded_replay=requested, enable_prefix_caching=prefix_caching
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=batched),
        model_config=SimpleNamespace(max_model_len=max_len),
    )


@pytest.mark.parametrize("requested", [None, True])
def test_flag_on_without_conflicts(requested):
    from vllm.models.deepseek_v41.attention import resolve_swa_bounded_replay

    assert resolve_swa_bounded_replay(_vllm_config(requested, False, 17536, 17536))


@pytest.mark.parametrize(
    "prefix_caching,batched", [(True, 17536), (False, 8192), (True, 8192)]
)
def test_default_turns_off_and_explicit_fails_where_inexact(prefix_caching, batched):
    from vllm.models.deepseek_v41.attention import resolve_swa_bounded_replay

    config = _vllm_config(None, prefix_caching, batched, 17536)
    assert resolve_swa_bounded_replay(config) is False
    with pytest.raises(ValueError, match="does not equal a full prefill"):
        resolve_swa_bounded_replay(_vllm_config(True, prefix_caching, batched, 17536))


def test_flag_off_stays_off():
    from vllm.models.deepseek_v41.attention import resolve_swa_bounded_replay

    assert resolve_swa_bounded_replay(_vllm_config(False, False, 17536, 17536)) is False


def _layers(index_k_prefixes):
    return [
        SimpleNamespace(
            attn=SimpleNamespace(
                indexer=None
                if prefix is None
                else SimpleNamespace(k_cache=SimpleNamespace(prefix=prefix))
            )
        )
        for prefix in index_k_prefixes
    ]


def test_index_only_sources_share_the_cut_cache():
    # Release 40L shape in miniature: kv sources ..2, index-only source 4.
    layers = _layers(
        [
            None,
            None,
            "model.layers.2.attn.indexer.k_cache",
            None,
            "model.layers.2.attn.indexer.k_cache",
            None,
        ]
    )
    check_decoder_replay_exact(
        SimpleNamespace(candidate_source_layer_id=2), layers, 2, 6
    )


def test_late_index_k_owner_or_candidate_rejected():
    layers = _layers(
        [None, None, None, None, "model.layers.4.attn.indexer.k_cache", None]
    )
    with pytest.raises(ValueError, match=r"layers \[4\] own"):
        check_decoder_replay_exact(SimpleNamespace(), layers, 2, 6)
    with pytest.raises(ValueError, match="candidate source"):
        check_decoder_replay_exact(
            SimpleNamespace(candidate_source_layer_id=3), _layers([None] * 6), 2, 6
        )
