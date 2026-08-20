# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from types import SimpleNamespace

import pytest

from vllm.distributed.device_communicators.all2all import DeepEPLLAll2AllManager


class _LegacyBuffer:
    def __init__(
        self,
        group,
        num_nvl_bytes=0,
        num_rdma_bytes=0,
        low_latency_mode=False,
        num_qps_per_rank=24,
        allow_nvlink_for_low_latency_mode=True,
        allow_mnnvl=False,
        explicitly_destroy=False,
    ):
        del (
            group,
            num_nvl_bytes,
            num_rdma_bytes,
            low_latency_mode,
            num_qps_per_rank,
            allow_nvlink_for_low_latency_mode,
            allow_mnnvl,
            explicitly_destroy,
        )

    @staticmethod
    def get_low_latency_rdma_size_hint(**kwargs):
        del kwargs
        return 1


class _FaultTolerantBuffer(_LegacyBuffer):
    def __init__(self, group, enable_shrink=False, **kwargs):
        del enable_shrink
        super().__init__(group, **kwargs)


def _make_kwargs(monkeypatch, buffer_cls, *, fault_tolerance):
    monkeypatch.setitem(sys.modules, "deep_ep", SimpleNamespace(Buffer=buffer_cls))
    manager = object.__new__(DeepEPLLAll2AllManager)
    manager.cpu_group = object()
    manager.support_fault_tolerance = fault_tolerance
    return manager._make_all2all_kwargs(
        max_num_tokens_per_dp_rank=8,
        token_hidden_size=16,
        num_ep_ranks=4,
        num_global_experts=32,
        num_local_experts=8,
    )


def test_deepep_ll_omits_disabled_fault_tolerance_kwarg(monkeypatch):
    kwargs = _make_kwargs(monkeypatch, _LegacyBuffer, fault_tolerance=False)
    assert "enable_shrink" not in kwargs


def test_deepep_ll_requires_enable_shrink_for_fault_tolerance(monkeypatch):
    with pytest.raises(RuntimeError, match="Buffer\\(enable_shrink"):
        _make_kwargs(monkeypatch, _LegacyBuffer, fault_tolerance=True)


def test_deepep_ll_enables_supported_fault_tolerance(monkeypatch):
    kwargs = _make_kwargs(monkeypatch, _FaultTolerantBuffer, fault_tolerance=True)
    assert kwargs["enable_shrink"] is True
