# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepEP low-latency dispatch for the batch-invariant MXFP4 group-32 experts."""

from dataclasses import replace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.test_deepgemm import (
    _mxfp4_ep_cases,
    _mxfp4_ep_quant,
    make_mxfp4_weights,
)
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)


def _layout(count: int, begin: int) -> int:
    return count | (begin << 32)


def test_received_row_weights_follow_dispatch_layout():
    from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll import (
        received_row_weights,
    )

    # Two ranks, two local experts each, three tokens per rank, top-2. This
    # is rank 1: it owns physical experts 2 and 3.
    routes = torch.tensor(
        [
            [[2, 0], [3, 2], [-1, -1]],  # rank 0 sends token 0 and 1
            [[1, 3], [2, 3], [0, 1]],  # rank 1 sends tokens 0..2
        ]
    )
    weights = torch.tensor(
        [
            [[0.5, 0.25], [0.125, 2.0], [0.0, 0.0]],
            [[4.0, 0.75], [1.5, 3.0], [8.0, 16.0]],
        ]
    )
    # Expert 2 received rank 1's token 1 first, then rank 0's tokens 0 and 1.
    # Expert 3 received rank 0's token 1, then rank 1's tokens 0 and 1.
    src_info = torch.tensor([[1, 0, 1, 7, 7, 7], [1, 0, 1, 7, 7, 7]], dtype=torch.int32)
    layout_range = torch.tensor(
        [
            [_layout(2, 1), _layout(1, 0)],
            [_layout(1, 0), _layout(2, 1)],
        ]
    )
    counts = torch.tensor([3, 3], dtype=torch.int32)
    row_weights = received_row_weights(
        (src_info, layout_range), routes, weights, counts, rank=1
    )
    assert row_weights.dtype == torch.float32
    assert row_weights.tolist() == [
        [1.5, 0.5, 2.0, 0.0, 0.0, 0.0],
        [0.125, 0.75, 3.0, 0.0, 0.0, 0.0],
    ]


@pytest.mark.parametrize(
    ("backend", "experts_cls", "act_format", "expected"),
    [
        ("deepep_low_latency", "BatchedDeepGemmFP4Experts", "BatchedExperts", True),
        ("deepep_low_latency", "DeepGemmFP4Experts", "Standard", False),
        ("allgather_reducescatter", "DeepGemmFP4Experts", "Standard", True),
        (
            "allgather_reducescatter",
            "BatchedDeepGemmFP4Experts",
            "BatchedExperts",
            False,
        ),
    ],
)
def test_mxfp4_bi_dispatch_selection(
    monkeypatch, backend, experts_cls, act_format, expected
):
    from types import SimpleNamespace

    from vllm import envs
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
    from vllm.model_executor.layers.fused_moe.experts import (
        batched_deep_gemm_moe,
        deep_gemm_moe,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic32Sym,
        kMxfp4Static,
    )

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(
        deep_gemm_moe,
        "current_platform",
        SimpleNamespace(is_device_capability_family=lambda family: family == 100),
    )
    cls = getattr(batched_deep_gemm_moe, experts_cls, None) or getattr(
        deep_gemm_moe, experts_cls
    )
    monkeypatch.setattr(cls, "_supports_current_device", staticmethod(lambda: True))
    parallel = replace(
        FusedMoEParallelConfig.make_no_parallel(),
        all2all_backend=backend,
        dp_size=4,
        ep_size=4,
        use_ep=True,
    )
    config = replace(
        make_dummy_moe_config(
            num_experts=8, experts_per_token=6, hidden_dim=5120, intermediate_size=2304
        ),
        num_local_experts=2,
        moe_parallel_config=parallel,
    )
    assert envs.VLLM_BATCH_INVARIANT
    supported, reason = mk.FusedMoEExperts.is_supported_config(
        cls,
        config,
        kMxfp4Static,
        kFp8Dynamic32Sym,
        getattr(mk.FusedMoEActivationFormat, act_format),
    )
    assert supported == expected, reason


@torch.inference_mode()
def _mxfp4_bi_deepep_ll_worker(group, vllm_config, cpu_group, clamp):
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.determinism.batch_invariant import (
        enable_batch_invariant_mode,
    )
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
    from vllm.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
        BatchedDeepGemmFP4Experts,
    )
    from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
        DeepGemmFP4Experts,
    )
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        _pack_deepgemm_mxfp4_scales,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll import (
        DeepEPLLPrepareAndFinalize,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(group.device)
    enable_batch_invariant_mode()
    # Same seed and inputs as the AG/RS test, so both dispatches are checked
    # against one unsharded result.
    torch.manual_seed(1907)
    experts, intermediate, hidden, topk = 8, 2304, 5120, 6
    w1, w2, s1, s2, _, _ = make_mxfp4_weights(experts, intermediate, hidden)
    scales = _pack_deepgemm_mxfp4_scales(w1, w2, s1, s2)
    per_rank = experts // group.world_size
    first = group.rank * per_rank
    owned = slice(first, first + per_rank)
    expert_map = torch.full((experts,), -1, device=group.device, dtype=torch.int32)
    expert_map[owned] = torch.arange(per_rank, device=group.device, dtype=torch.int32)
    base = make_dummy_moe_config(
        num_experts=experts,
        experts_per_token=topk,
        hidden_dim=hidden,
        intermediate_size=intermediate,
    )
    parallel = FusedMoEParallelConfig.make(
        1, 1, group.world_size, 1, vllm_config.parallel_config
    )
    assert parallel.use_deepep_ll_kernels
    sharded = replace(base, num_local_experts=per_rank, moe_parallel_config=parallel)

    local_quant = _mxfp4_ep_quant(scales[0], scales[1], clamp, 32)
    local = mk.FusedMoEKernel(
        prepare_finalize=maybe_make_prepare_finalize(
            base, local_quant, allow_new_interface=True
        ),
        fused_experts=DeepGemmFP4Experts(base, local_quant),
    )
    quant = _mxfp4_ep_quant(scales[0][owned], scales[1][owned], clamp, 32)
    prepare_finalize = maybe_make_prepare_finalize(
        sharded, quant, allow_new_interface=True
    )
    assert isinstance(prepare_finalize, DeepEPLLPrepareAndFinalize)
    assert not prepare_finalize.use_fp8_dispatch
    distributed = mk.FusedMoEKernel(
        prepare_finalize=prepare_finalize,
        fused_experts=BatchedDeepGemmFP4Experts(
            sharded,
            quant,
            max_num_tokens=prepare_finalize.max_num_tokens_per_rank(),
            num_dispatchers=prepare_finalize.num_dispatchers(),
        ),
    )
    assert prepare_finalize.weights_rows_before_combine

    for sizes, values, ids, scores in _mxfp4_ep_cases(
        group.device, experts, hidden, topk
    ):
        begin = sum(sizes[: group.rank])
        rows = slice(begin, begin + sizes[group.rank])
        kwargs = dict(
            activation=MoEActivation.SILU,
            global_num_experts=experts,
            apply_router_weight_on_input=False,
        )
        expected = local.apply(
            hidden_states=values,
            w1=w1,
            w2=w2,
            topk_weights=scores,
            topk_ids=ids,
            expert_map=None,
            **kwargs,
        )[rows].clone()
        with set_forward_context(
            None,
            vllm_config,
            num_tokens=sizes[group.rank],
            num_tokens_across_dp=torch.tensor(sizes, device="cpu", dtype=torch.int32),
        ):
            actual = distributed.apply(
                hidden_states=values[rows].contiguous(),
                w1=w1[owned].contiguous(),
                w2=w2[owned].contiguous(),
                topk_weights=scores[rows].contiguous(),
                topk_ids=ids[rows].long().contiguous(),
                expert_map=expert_map,
                **kwargs,
            )
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected), (group.rank, sizes, clamp)


@pytest.mark.parametrize("clamp", [None, 10.0])
def test_mxfp4_bi_deepep_ll_matches_unsharded(clamp):
    """EP4 DeepEP low-latency group-32 experts reproduce the unsharded result
    bit for bit, as test_mxfp4_bi_ep_ag_rs_matches_unsharded[32] does for AG/RS
    on the same inputs."""
    from tests.kernels.moe.modular_kernel_tools.parallel_utils import (
        parallel_launch_with_config,
    )
    from vllm.config import ParallelConfig, VllmConfig

    assert torch.accelerator.device_count() >= 4, "EP evidence requires four GPUs"
    config = VllmConfig(
        parallel_config=ParallelConfig(
            data_parallel_size=4,
            data_parallel_size_local=4,
            enable_expert_parallel=True,
            all2all_backend="deepep_low_latency",
        )
    )
    parallel_launch_with_config(
        4, _mxfp4_bi_deepep_ll_worker, config, {"VLLM_BATCH_INVARIANT": "1"}, clamp
    )
