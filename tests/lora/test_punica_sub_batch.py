# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LoRA over a gathered subset of the batch rows (DeepSeek-V4.1 decoder replay).

The kernels size their tiles from the per-LoRA token counts, so the full-batch
metadata applied to a sub-batch writes past it; the sub-batch metadata must
give each row its own adapter and touch nothing else."""

import numpy as np
import pytest
import torch

from vllm.config.lora import LoRAConfig
from vllm.lora.layers import LoRAMapping
from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU
from vllm.models.deepseek_v41.decoder_replay_layers import tail_rows
from vllm.platforms import current_platform


def test_tail_rows_are_each_requests_last_rows():
    qsl = np.array([0, 5, 6, 13])
    lens = np.array([3, 1, 0])
    assert tail_rows(qsl, lens).tolist() == [2, 3, 4, 5]
    assert tail_rows(qsl, np.array([1, 0, 0])).tolist() == [4]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="punica kernels need CUDA")
@pytest.mark.parametrize("batch_invariant", [False, True])
def test_sub_batch_metadata(monkeypatch, batch_invariant):
    import vllm.lora.ops.triton_ops.lora_shrink_op as shrink_op
    import vllm.lora.ops.triton_ops.utils as lora_utils

    monkeypatch.setattr(lora_utils, "is_batch_invariant", batch_invariant)
    monkeypatch.setattr(shrink_op, "is_batch_invariant", batch_invariant)
    lora_utils.get_lora_op_configs.cache_clear()
    torch.manual_seed(0)
    rank, k, n = 8, 256, 192
    # Requests: adapter 1 x37, no LoRA x21, adapter 2 x45.
    lens = np.array([37, 21, 45])
    ids = np.repeat([1, 0, 2], lens)
    punica = PunicaWrapperGPU(
        256, 8, "cuda", lora_config=LoRAConfig(max_loras=2, max_lora_rank=rank)
    )
    punica.update_metadata(
        LoRAMapping(tuple(ids), tuple(ids)), [1, 2], max_loras=2, vocab_size=0
    )
    a = (0.1 * torch.randn(2, 1, rank, k, device="cuda")).bfloat16()
    b = (0.1 * torch.randn(2, 1, n, rank, device="cuda")).bfloat16()
    x = torch.randn(len(ids), k, device="cuda").bfloat16()
    full = torch.randn(len(ids), n, device="cuda").bfloat16()
    base = full.clone()
    punica.add_lora_linear(full, x, (a,), (b,), 1.0, (n,))

    # Each request's last 30/9/30 rows: 69 rows, not a multiple of the tile.
    qsl = np.concatenate([[0], np.cumsum(lens)])
    rows = tail_rows(qsl, np.minimum(lens, [30, 9, 30]))
    rows_t = torch.from_numpy(rows).cuda()
    canary = torch.full((len(rows) + 64, n), float("nan"), device="cuda").bfloat16()
    canary[: len(rows)] = base[rows_t]
    meta = punica.new_sub_batch_meta()
    punica.prepare_sub_batch_meta(meta, rows)
    with punica.use_token_mapping_meta(meta):
        punica.add_lora_linear(canary[: len(rows)], x[rows_t], (a,), (b,), 1.0, (n,))
    assert punica.token_mapping_meta is not meta
    assert torch.equal(
        canary[: len(rows)].view(torch.int16), full[rows_t].view(torch.int16)
    )
    assert torch.isnan(canary[len(rows) :].float()).all()

    # A sub-batch of no-LoRA rows only: the kernels skip it.
    rows = tail_rows(qsl, np.array([0, 5, 0]))
    punica.prepare_sub_batch_meta(meta, rows)
    assert bool(meta.no_lora_flag_cpu[0])
    # Rows past the mapping (an idle rank's dummy batch) get no LoRA.
    punica.prepare_sub_batch_meta(meta, np.array([len(ids) + 3]))
    assert bool(meta.no_lora_flag_cpu[0])
    lora_utils.get_lora_op_configs.cache_clear()
