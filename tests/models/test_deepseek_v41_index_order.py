# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.models.deepseek_v41.attention as attention


@pytest.mark.parametrize("batch_invariant", [False, True])
def test_graph_topk_reads_selected_positions_in_deployment_order(
    monkeypatch, batch_invariant
):
    """Replay sees live token counts and preserves the shared buffer and selection."""
    monkeypatch.setattr(attention.envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    assert attention.envs.VLLM_BATCH_INVARIANT is batch_invariant
    indices = torch.tensor(
        [[3, 0, 2, 1, -1, -1], [1, -1, 0, -1, -1, -1], [9, 8, 7, 6, 5, 4]],
        dtype=torch.int32,
    )
    pointer = indices.data_ptr()
    metadata = SimpleNamespace(num_decode_tokens=2, num_prefill_tokens=0)
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"index": metadata}),
    )
    scored = []
    reads = []
    layer = SimpleNamespace(
        indexer=SimpleNamespace(
            k_cache=SimpleNamespace(prefix="index"),
            indexer_op=lambda *args: scored.append(args),
        ),
        topk_indices_buffer=indices,
        forward_mqa=lambda *args: reads.append(indices.clone()),
    )
    value = torch.empty(2, 1)
    positions = torch.tensor([129, 3])
    original = indices.clone()
    attention.DeepseekV4Attention._sparse_indexer_and_attn(
        layer, value, value, None, value, value, value, positions, value
    )
    expected = original.clone()
    if batch_invariant:
        expected[:2] = torch.tensor([[0, 1, 2, 3, -1, -1], [0, 1, -1, -1, -1, -1]])
    assert scored
    assert torch.equal(reads[-1], expected)
    assert indices.data_ptr() == pointer

    # A replay with a smaller live batch must not sort the previous padded row.
    metadata.num_decode_tokens = 1
    indices.copy_(original)
    attention.DeepseekV4Attention._sparse_indexer_and_attn(
        layer, value, value, None, value, value, value, positions, value
    )
    expected[1:] = original[1:]
    assert torch.equal(reads[-1], expected)

    # Eager short-context fill has no Q/indexer call and already has canonical order.
    indices.copy_(expected)
    attention.DeepseekV4Attention._sparse_indexer_and_attn(
        layer, value, None, None, None, value, value, positions, value
    )
    assert len(scored) == 2
    assert torch.equal(reads[-1], expected)


def test_profile_run_without_attention_metadata_preserves_dummy_path(monkeypatch):
    """The native profile run has no KV metadata and performs no attention read."""
    monkeypatch.setattr(attention.envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=None),
    )
    indices = torch.tensor([[3, 0, 2, 1, -1, -1]], dtype=torch.int32)
    original = indices.clone()
    calls = []
    layer = SimpleNamespace(
        indexer=SimpleNamespace(
            k_cache=SimpleNamespace(prefix="index"),
            indexer_op=lambda *args: calls.append("dummy_indexer"),
        ),
        topk_indices_buffer=indices,
        forward_mqa=lambda *args: calls.append("dummy_attention"),
    )
    value = torch.empty(1, 1)
    positions = torch.tensor([0])
    attention.DeepseekV4Attention._sparse_indexer_and_attn(
        layer, value, value, None, value, value, value, positions, value
    )
    assert calls == ["dummy_indexer", "dummy_attention"]
    assert torch.equal(indices, original)
