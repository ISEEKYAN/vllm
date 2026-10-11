# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm import envs
from vllm.model_executor.kernels.mhc import warmup

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.mark.parametrize("batch_invariant", [False, True])
def test_entry_split_order_for_decode_prefill_and_teacher_forcing(
    monkeypatch, batch_invariant
):
    """Long teacher-forced batches must retain single-token reduction in BI."""
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=152),
    )
    warmup.compute_num_split.cache_clear()
    try:
        actual = [
            warmup.compute_mhc_pre_num_splits(5120, rows)
            for rows in (1, 8, 239, 384, 4206)
        ]
        assert actual == ([16] * 5 if batch_invariant else [16, 16, 16, 16, 4])
    finally:
        warmup.compute_num_split.cache_clear()
