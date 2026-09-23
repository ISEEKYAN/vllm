# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.models.deepseek_v4.nvidia.flashmla as flashmla_module
from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention


class _WorkspaceManager:
    def get_simultaneous(self, *specs):
        return tuple(torch.empty(shape, dtype=dtype) for shape, dtype in specs)


def _attention(compress_ratio: int) -> DeepseekV4FlashMLAAttention:
    attention = DeepseekV4FlashMLAAttention.__new__(DeepseekV4FlashMLAAttention)
    nn.Module.__init__(attention)
    attention.compress_ratio = compress_ratio
    attention.window_size = 4
    attention.max_model_len = 512
    attention.scale = 0.125
    attention.attn_sink = torch.zeros(2, dtype=torch.float32)
    attention.topk_indices_buffer = torch.tensor(
        [[0, 1, 2, 3, -1, -1, -1, -1]] * 8,
        dtype=torch.int32,
    )
    attention.swa_cache_layer = SimpleNamespace(
        kv_cache=torch.empty((2, 256, 584), dtype=torch.uint8)
    )
    return attention


def _swa_metadata():
    return SimpleNamespace(
        num_decodes=2,
        num_decode_tokens=2,
        seq_lens=torch.tensor([3, 5], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([3, 5], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_valid_token=torch.tensor([True, True]),
        decode_swa_indices=torch.zeros((2, 1, 4), dtype=torch.int32),
        decode_swa_lens=torch.tensor([3, 4], dtype=torch.int32),
        block_table=torch.zeros((2, 2), dtype=torch.int32),
        block_size=256,
    )


@pytest.mark.parametrize("compress_ratio", [1, 4, 128])
def test_decode_sparse_reuses_prefill_kernel(
    monkeypatch: pytest.MonkeyPatch,
    compress_ratio: int,
):
    attention = _attention(compress_ratio)
    swa_metadata = _swa_metadata()
    q = torch.zeros((2, 2, 8), dtype=torch.bfloat16)
    output = torch.empty_like(q)
    compressed_cache = (
        None
        if compress_ratio == 1
        else torch.empty((2, 256 // compress_ratio, 584), dtype=torch.uint8)
    )
    attn_metadata = (
        None
        if compress_ratio == 1
        else SimpleNamespace(
            block_table=torch.zeros((2, 2), dtype=torch.int32),
            block_size=256,
            c128a_global_decode_topk_indices=(
                torch.zeros((2, 1, 128), dtype=torch.int32)
                if compress_ratio == 128
                else None
            ),
            c128a_decode_topk_lens=(
                torch.tensor([2, 2], dtype=torch.int32)
                if compress_ratio == 128
                else None
            ),
        )
    )
    gathered = []
    captured = {}

    def fake_gather(out, cache, **kwargs):
        out.zero_()
        gathered.append((cache, kwargs))

    def fake_combine(local_topk, *args, out, **kwargs):
        captured["local_topk"] = local_topk.clone()
        indices, lengths = out
        indices.fill_(-1)
        lengths.fill_(1)
        return indices, lengths

    def fake_fill_c128(out, lengths):
        indices = torch.arange(out.shape[1], dtype=out.dtype)
        out.copy_(torch.where(indices < lengths[:, None], indices, -1))

    def fake_sparse_fwd(**kwargs):
        captured["sparse"] = kwargs
        kwargs["out"].zero_()

    monkeypatch.setattr(
        flashmla_module.envs, "VLLM_DS4_DECODE_KERNEL", "sparse", raising=False
    )
    monkeypatch.setattr(
        flashmla_module, "current_workspace_manager", lambda: _WorkspaceManager()
    )
    monkeypatch.setattr(flashmla_module, "dequantize_and_gather_k_cache", fake_gather)
    monkeypatch.setattr(flashmla_module, "combine_topk_swa_indices", fake_combine)
    monkeypatch.setattr(flashmla_module, "fill_c128_topk", fake_fill_c128)
    monkeypatch.setattr(flashmla_module, "flash_mla_sparse_fwd", fake_sparse_fwd)
    monkeypatch.setattr(
        flashmla_module,
        "flash_mla_with_kvcache",
        lambda **kwargs: pytest.fail("paged decode kernel must not run"),
    )

    attention._forward_decode(
        q=q,
        kv_cache=compressed_cache,
        swa_metadata=swa_metadata,
        attn_metadata=attn_metadata,
        swa_only=compress_ratio == 1,
        output=output,
    )

    assert len(gathered) == (1 if compress_ratio == 1 else 2)
    assert captured["sparse"]["q"] is q
    assert captured["sparse"]["out"] is output
    if compress_ratio == 128:
        expected = torch.full((128,), -1, dtype=torch.int32)
        expected[:2] = torch.arange(2, dtype=torch.int32)
        torch.testing.assert_close(captured["local_topk"][0], expected, rtol=0, atol=0)
    elif compress_ratio == 4:
        torch.testing.assert_close(
            captured["local_topk"],
            attention.topk_indices_buffer[:2],
            rtol=0,
            atol=0,
        )


def test_decode_paged_remains_default(monkeypatch: pytest.MonkeyPatch):
    attention = _attention(1)
    swa_metadata = _swa_metadata()
    swa_metadata.tile_sched_swaonly = object()
    q = torch.zeros((2, 2, 8), dtype=torch.bfloat16)
    output = torch.empty_like(q)
    captured = {}

    def fake_paged(**kwargs):
        captured.update(kwargs)
        return kwargs["out"], None

    monkeypatch.setattr(
        flashmla_module.envs, "VLLM_DS4_DECODE_KERNEL", "paged", raising=False
    )
    monkeypatch.setattr(flashmla_module, "flash_mla_with_kvcache", fake_paged)
    monkeypatch.setattr(
        flashmla_module,
        "flash_mla_sparse_fwd",
        lambda **kwargs: pytest.fail("sparse decode kernel must not run"),
    )

    attention._forward_decode(
        q=q,
        kv_cache=None,
        swa_metadata=swa_metadata,
        attn_metadata=None,
        swa_only=True,
        output=output,
    )

    assert captured["tile_scheduler_metadata"] is swa_metadata.tile_sched_swaonly
    assert captured["out"].shape == (2, 1, 2, 8)


def test_decode_sparse_fails_closed_without_scheduler_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    attention = _attention(1)
    swa_metadata = _swa_metadata()
    swa_metadata.seq_lens_cpu = None
    monkeypatch.setattr(
        flashmla_module.envs, "VLLM_DS4_DECODE_KERNEL", "sparse", raising=False
    )

    with pytest.raises(RuntimeError, match="finalized scheduler metadata"):
        attention._forward_decode(
            q=torch.zeros((2, 2, 8), dtype=torch.bfloat16),
            kv_cache=None,
            swa_metadata=swa_metadata,
            attn_metadata=None,
            swa_only=True,
            output=torch.empty((2, 2, 8), dtype=torch.bfloat16),
        )


@pytest.mark.parametrize("compress_ratio", [1, 4, 128])
def test_sparse_decode_graph_handles_growing_sequences(monkeypatch, compress_ratio):
    """Replay must use new KV lengths, not capture-time workspace capacity."""
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        quantize_and_insert_k_cache,
    )

    if not torch.cuda.is_available():
        pytest.skip("CUDA graph test")
    torch.manual_seed(17)
    attention = _attention(compress_ratio)
    attention.attn_sink = torch.zeros(64, device="cuda")
    attention.topk_indices_buffer = attention.topk_indices_buffer.cuda()
    attention.topk_indices_buffer[:, 1:] = -1
    swa = _swa_metadata()
    for name, value in vars(swa).items():
        if isinstance(value, torch.Tensor) and not name.endswith("_cpu"):
            setattr(swa, name, value.cuda())
    swa.block_table.copy_(torch.tensor([[0, 1], [2, 3]], device="cuda"))
    swa.block_size = 512

    def make_cache(block_size):
        cache = torch.zeros((4, block_size, 584), dtype=torch.uint8, device="cuda")
        values = torch.randn((4 * block_size, 512), dtype=torch.bfloat16, device="cuda")
        slots = torch.arange(4 * block_size, dtype=torch.int64, device="cuda")
        quantize_and_insert_k_cache(values, cache.view(4, -1), slots, block_size)
        return cache

    attention.swa_cache_layer.kv_cache = make_cache(512)
    compressed_cache = (
        None if compress_ratio == 1 else make_cache(512 // compress_ratio)
    )
    metadata = SimpleNamespace(
        block_table=swa.block_table,
        block_size=512,
        c128a_global_decode_topk_indices=torch.zeros(
            (2, 1, 128), dtype=torch.int32, device="cuda"
        ),
        c128a_decode_topk_lens=torch.zeros(2, dtype=torch.int32, device="cuda"),
    )

    class Workspace:
        def get_simultaneous(self, *specs):
            return tuple(
                torch.empty(shape, dtype=dtype, device="cuda") for shape, dtype in specs
            )

    monkeypatch.setattr(flashmla_module.envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(flashmla_module.envs, "VLLM_DS4_DECODE_KERNEL", "sparse")
    monkeypatch.setattr(flashmla_module, "current_workspace_manager", Workspace)
    q = torch.randn((2, 64, 512), dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(q)

    def forward():
        attention._forward_decode(
            q=q,
            kv_cache=compressed_cache,
            swa_metadata=swa,
            attn_metadata=None if compress_ratio == 1 else metadata,
            swa_only=compress_ratio == 1,
            output=output,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(stream)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    for lengths in ([15, 17], [255, 257], [511, 512]):
        swa.seq_lens_cpu.copy_(torch.tensor(lengths, dtype=torch.int32))
        swa.seq_lens.copy_(swa.seq_lens_cpu)
        if compress_ratio == 4:
            offsets = torch.arange(8, dtype=torch.int32, device="cuda")
            counts = (swa.seq_lens // 4).clamp(max=8)
            attention.topk_indices_buffer[:2].copy_(
                torch.where(
                    offsets < counts[:, None],
                    swa.seq_lens[:, None] // 4 - counts[:, None] + offsets,
                    -1,
                )
            )
        metadata.c128a_decode_topk_lens.copy_(swa.seq_lens // 128)
        graph.replay()
        replay = output.clone()
        forward()
        torch.accelerator.synchronize()
        assert torch.isfinite(output).all()
        torch.testing.assert_close(replay, output, rtol=0, atol=0, msg=f"{lengths=}")
