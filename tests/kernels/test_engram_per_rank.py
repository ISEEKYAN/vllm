# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-rank Engram host tables: contiguous row ranges over the DP group.

Layout from sgl-project/sglang@86cc2e02ecb2 python/sglang/srt/layers/engram.py.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.kernels.test_engram import _reference_lookup
from vllm.config.engram import EngramConfig
from vllm.models.deepseek_v41.common import engram as engram_ops
from vllm.models.deepseek_v41.common.engram import (
    DEAD_ID,
    Engram,
    ParallelEngramEmbedding,
    engram_row_range,
    reduce_scatter_owned_rows,
)
from vllm.platforms import current_platform


@pytest.mark.parametrize("num_rows", [0, 3, 7, 4096, 384_006_168])
@pytest.mark.parametrize("size", [1, 2, 3, 4, 8, 16])
def test_row_ranges_cover_the_table_once(num_rows, size):
    ranges = [engram_row_range(num_rows, rank, size) for rank in range(size)]
    assert ranges[0][0] == 0 and ranges[-1][1] == num_rows
    for (_, end), (start, _) in zip(ranges, ranges[1:]):
        assert end == start
    assert all(start <= end for start, end in ranges)


class _SumGroup:
    """A DP group whose reduce_scatter sums every rank's contribution."""

    def __init__(
        self,
        rank: int,
        contributions: list[torch.Tensor],
        gathered: list[torch.Tensor] | None = None,
    ):
        self.rank_in_group, self.world_size = rank, len(contributions)
        self.contributions = contributions
        self.gathered = gathered

    def all_gather(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        assert dim == 0 and self.gathered is not None
        assert torch.equal(input_, self.gathered[self.rank_in_group])
        return torch.cat(self.gathered)

    def reduce_scatter(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        assert dim == 0 and input_.dtype == torch.int32
        own = self.contributions[self.rank_in_group]
        assert torch.equal(input_, own.reshape(own.shape[0], -1).view(torch.int32))
        total = sum(
            c.reshape(c.shape[0], -1).view(torch.int32) for c in self.contributions
        )
        return total.chunk(self.world_size)[self.rank_in_group]


def test_reduce_scatter_keeps_every_bf16_bit():
    torch.manual_seed(0)
    size, slot, heads, dim = 4, 3, 5, 8
    full = torch.randn(size * slot, heads, dim).to(torch.bfloat16)
    full[0, 0, :4] = torch.tensor([-0.0, 0.0, float("inf"), -float("inf")])
    bits = full.view(torch.int16)
    bits[1, 1, 0] = 0x7FC1  # NaN with a payload
    bits[1, 1, 1] = -0x003F  # 0xFFC1, negative NaN with a payload
    owner = torch.randint(0, size, (size * slot, heads))
    contributions = [
        full.masked_fill((owner != rank).unsqueeze(-1), 0) for rank in range(size)
    ]
    for rank in range(size):
        out = reduce_scatter_owned_rows(
            contributions[rank], _SumGroup(rank, contributions)
        )
        expected = full[rank * slot : (rank + 1) * slot]
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


def test_per_rank_config_requirements():
    with pytest.raises(ValueError, match="per_rank"):
        EngramConfig(cpu_offload=False, host_table_layout="per_rank")
    with pytest.raises(ValueError, match="per_rank"):
        EngramConfig(dp_shared_memory=True, host_table_layout="per_rank")
    config = EngramConfig(host_table_layout="per_rank")
    parallel = SimpleNamespace(
        data_parallel_size=16, tensor_parallel_size=1, enable_elastic_ep=False
    )
    config.resolve_dp_shared_memory(parallel)
    assert config.dp_shared_memory is False
    config.verify_parallel_config(parallel)
    parallel.tensor_parallel_size = 2
    with pytest.raises(ValueError, match="TP=1"):
        config.verify_parallel_config(parallel)
    assert EngramConfig().compute_hash() != config.compute_hash()


def _per_rank_layers(monkeypatch, size, num_rows, dim, head_sizes, weight, scales):
    monkeypatch.setattr(engram_ops, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(engram_ops, "get_engram_dp_size", lambda: 4)
    layers = []
    for rank in range(size):
        group = SimpleNamespace(rank_in_group=rank, world_size=size)
        monkeypatch.setattr(engram_ops, "get_dp_group", lambda group=group: group)
        with torch.device("cuda"):
            layer = ParallelEngramEmbedding(
                num_rows, dim, head_sizes, cpu_offload=True, row_sharded=True
            )
        assert layer.dp_size == 1 and layer.part_n_hash_cols == len(head_sizes)
        assert (layer.vocab_start_idx, layer.vocab_end_idx) == engram_row_range(
            num_rows, rank, size
        )
        layer.weight.weight_loader(layer.weight, weight)
        layer.weight_scale_inv.weight_loader(
            layer.weight_scale_inv, scales.view(torch.float8_e8m0fnu)
        )
        layers.append(layer)
    return layers


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize("size", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("sort_rows", [False, True])
def test_per_rank_rows_reassemble_the_full_lookup(size, sort_rows, monkeypatch):
    """Each rank keeps only its rows; the integer sum of every rank's lookup
    (zeros outside its rows) is the full-table lookup bit for bit."""
    if sort_rows:
        monkeypatch.setattr(
            engram_ops, "_engram_lookup_thresholds", lambda _: (0, None)
        )
    head_sizes = (17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73)
    num_rows, dim = sum(head_sizes) + 7, 64
    torch.manual_seed(0)
    weight = torch.randn(num_rows, dim).to(torch.float8_e4m3fn)
    weight[3, :5] = -0.0
    scales = torch.randint(120, 134, (num_rows, dim // 32), dtype=torch.uint8)
    ids = torch.empty(11, len(head_sizes), dtype=torch.int32, device="cuda")
    start = 0
    for head, hsize in enumerate(head_sizes):
        ids[:, head].random_(start, start + hsize)
        ids[0, head], ids[-1, head] = start, start + hsize - 1
        start += hsize
    ids[1, 0] = 3
    ids[2] = DEAD_ID
    expected = _reference_lookup(weight.cuda(), scales.cuda(), ids, 0, num_rows)

    layers = _per_rank_layers(
        monkeypatch, size, num_rows, dim, head_sizes, weight, scales
    )
    assert sum(layer.part_num_embeddings for layer in layers) == num_rows
    assert all(layer.weight.device.type == "cpu" for layer in layers)
    outputs = []
    for layer in layers:
        out = torch.empty(
            len(ids), len(head_sizes), dim, device="cuda", dtype=torch.bfloat16
        )
        layer.lookup(ids, out)
        outputs.append(out)
    total = sum(out.view(torch.int16).int() for out in outputs)
    assert torch.equal(total.short(), expected.view(torch.int16))
    owners = sum((out.view(torch.int16) != 0).int() for out in outputs)
    assert owners.max() <= 1


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize("sizes", [(5, 0, 3, 9), (1, 1, 1, 1), (0, 0, 2, 0)])
def test_per_rank_lookup_returns_each_rank_its_tokens(sizes, monkeypatch):
    """End to end per rank: DP gather of padded ids, owned-row lookup and the
    integer reduce-scatter, against the full-table lookup."""
    size = len(sizes)
    head_sizes = (101, 103, 107, 109)
    num_rows, dim = sum(head_sizes), 64
    torch.manual_seed(1)
    weight = torch.randn(num_rows, dim).to(torch.float8_e4m3fn)
    scales = torch.randint(120, 134, (num_rows, dim // 32), dtype=torch.uint8)
    layers = _per_rank_layers(
        monkeypatch, size, num_rows, dim, head_sizes, weight, scales
    )
    offsets = torch.tensor(head_sizes).cumsum(0) - torch.tensor(head_sizes)
    rank_ids = [
        (torch.randint(0, min(head_sizes), (n, len(head_sizes))) + offsets).int().cuda()
        for n in sizes
    ]
    slot = max(sizes)
    padded = [
        torch.cat((ids, ids.new_full((slot - len(ids), len(head_sizes)), DEAD_ID)))
        for ids in rank_ids
    ]
    gathered = torch.cat(padded)
    contributions = []
    for layer in layers:
        out = torch.zeros(
            len(gathered), len(head_sizes), dim, device="cuda", dtype=torch.bfloat16
        )
        layer.lookup(gathered, out)
        contributions.append(out)
    metadata = SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor(sizes))
    monkeypatch.setattr(
        engram_ops,
        "get_forward_context",
        lambda: SimpleNamespace(dp_metadata=metadata),
    )
    for rank, (layer, ids) in enumerate(zip(layers, rank_ids)):
        group = _SumGroup(rank, contributions, padded)
        monkeypatch.setattr(engram_ops, "get_dp_group", lambda group=group: group)
        expected = _reference_lookup(weight.cuda(), scales.cuda(), ids, 0, num_rows)
        assert torch.equal(layer(ids).view(torch.int16), expected.view(torch.int16))
        # The Engram staging path stores the same rows inline.
        module = Engram.__new__(Engram)
        torch.nn.Module.__init__(module)
        module.embed_tokens = layer
        module.use_sequence_parallel = False
        module._prefetch_stream = module._prefetch_done = None
        module.staged_rows = torch.empty(
            16, len(head_sizes), dim, device="cuda", dtype=torch.bfloat16
        )
        module.prepare_embeddings(ids)
        assert torch.equal(
            module.embed(ids).view(torch.int16), expected.view(torch.int16)
        )
