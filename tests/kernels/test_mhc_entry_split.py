# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm import envs
from vllm.model_executor.kernels.mhc import warmup

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.mark.parametrize("batch_invariant", [False, True])
def test_shared_entry_selector_and_warmup_keys_are_batch_independent(
    monkeypatch, batch_invariant
):
    """Shared selector and actual warmup key construction use a fixed BI split."""
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=152),
    )
    monkeypatch.setattr(warmup, "is_deep_gemm_supported", lambda: True)
    warmup.compute_num_split.cache_clear()
    try:
        actual = [
            warmup.compute_mhc_pre_num_splits(5120, rows)
            for rows in (1, 8, 239, 384, 4206)
        ]
        assert actual == ([16] * 5 if batch_invariant else [16, 16, 16, 16, 4])
        keys = warmup.MHC_PRE_NORM_KERNEL.get_warmup_keys(
            max_tokens=4206,
            hidden_size=1280,
            rms_eps=1e-6,
            hc_pre_eps=1e-6,
            hc_sinkhorn_eps=1e-6,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=20,
            norm_eps=1e-6,
            hc_mult=4,
            use_pre_mix_in=False,
            rms_numel=5120,
        )
        assert sorted({key.n_splits for key in keys}) == (
            [16] if batch_invariant else [4, 16]
        )
    finally:
        warmup.compute_num_split.cache_clear()
