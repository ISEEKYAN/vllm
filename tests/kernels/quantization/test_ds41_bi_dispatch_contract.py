# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU control-flow contracts; CUDA arithmetic and NCCL are tested separately.

Compile the actual production functions without importing GPU model layers.
Only their platform, provider and kernel/collective boundaries are replaced.
"""

import ast
import os
import sys
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]
ROOT = Path(__file__).resolve().parents[3]


def load_function(relative_path, name, namespace):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    # Postponed annotations leave production signatures and decorators intact.
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name], namespace


Backends = Enum(
    "Backends",
    "FLASHINFER_MLA_SPARSE FLASHINFER_MLA_SPARSE_SM120 "
    "FLASHINFER_MLA_SPARSE_DSV4 FLASHINFER_MLA_SPARSE_DSV41 "
    "FLASHMLA_MEGA_ATTN_DSV41 FLASHMLA_SPARSE FLASHMLA_SPARSE_DSV4 "
    "FLASHMLA_SPARSE_DSV41",
)


@pytest.mark.parametrize("bi", [False, True])
@pytest.mark.parametrize("mega_available", [False, True])
@pytest.mark.parametrize(
    "backend", [None, Backends.FLASHMLA_SPARSE_DSV41, Backends.FLASHMLA_MEGA_ATTN_DSV41]
)
def test_sm100_attention_selection(bi, mega_available, backend):
    flash = type("FlashMLA", (), {})
    available = Mock(return_value=mega_available)
    mega = type("MegaAttn", (), {"is_available_for": available})
    selector, _ = load_function(
        "vllm/models/deepseek_v41/nvidia/model.py",
        "_select_dsv4_attn_cls",
        dict(
            envs=SimpleNamespace(VLLM_BATCH_INVARIANT=bi),
            AttentionBackendEnum=Backends,
            current_platform=SimpleNamespace(
                get_device_capability=lambda: SimpleNamespace(major=10)
            ),
            DeepseekV4MegaAttnAttention=mega,
            DeepseekV4FlashMLAAttention=flash,
        ),
    )
    config = SimpleNamespace(attention_config=SimpleNamespace(backend=backend))
    if bi and backend is Backends.FLASHMLA_MEGA_ATTN_DSV41:
        with pytest.raises(ValueError, match="does not support VLLM_BATCH_INVARIANT"):
            selector(config)
    else:
        expected = (
            mega
            if backend is Backends.FLASHMLA_MEGA_ATTN_DSV41
            or (backend is None and mega_available and not bi)
            else flash
        )
        assert selector(config) is expected
    # BI never probes the MegaAttn default or silently falls back for explicit Mega.
    assert available.call_count == int(backend is None and not bi)


API_NAMES = (
    "fp8_fp4_paged_mqa_logits",
    "get_paged_mqa_logits_metadata",
    "fp8_fp4_sparse_mqa_logits",
    "fp8_fp4_paged_sparse_mqa_logits",
    "get_sparse_mqa_logits_metadata",
    "get_paged_sparse_mqa_logits_metadata",
    "fp8_gemm_nt",
    "fp8_fp4_mqa_logits",
)


def lazy_provider(vendored, monkeypatch, missing=None):
    external = SimpleNamespace(
        **{name: Mock(name="external_" + name) for name in API_NAMES}
    )
    bundled = SimpleNamespace(
        **{name: Mock(name="bundled_" + name) for name in API_NAMES if name != missing}
    )
    importer = Mock(return_value=bundled)
    init, namespace = load_function(
        "vllm/utils/deep_gemm.py",
        "_lazy_init",
        dict(
            os=os,
            envs=SimpleNamespace(
                VLLM_DEEP_GEMM_PAGED_MQA_USE_VENDORED=vendored,
                VLLM_CACHE_ROOT="/unused",
            ),
            has_deep_gemm=lambda: True,
            _import_deep_gemm=lambda: external,
            importlib=SimpleNamespace(import_module=importer),
            current_platform=SimpleNamespace(is_arch_support_pdl=lambda: False),
            logger=SimpleNamespace(info_once=Mock()),
            _resolve_mega_mhc_impl=lambda _: None,
            DeepGemmQuantScaleFMT=SimpleNamespace(init_oracle_cache=Mock()),
        ),
    )
    tree = ast.parse((ROOT / "vllm/utils/deep_gemm.py").read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_lazy_init"
    )
    for node in ast.walk(function):
        if isinstance(node, ast.Global):
            namespace.update(dict.fromkeys(node.names))
    monkeypatch.setenv("DG_JIT_CACHE_DIR", "/unused")
    return init, namespace, external, bundled, importer


@pytest.mark.parametrize("vendored", [False, True])
def test_dense_paged_provider_and_external_sparse_pair(monkeypatch, vendored):
    init, namespace, external, bundled, importer = lazy_provider(vendored, monkeypatch)
    init()
    for name in API_NAMES:
        source = bundled if vendored and name in API_NAMES[:2] else external
        assert namespace["_" + name + "_impl"] is getattr(source, name)
    assert importer.call_args_list == (
        [(("vllm.third_party.deep_gemm",), {})] if vendored else []
    )


@pytest.mark.parametrize("missing", API_NAMES[:2])
def test_explicit_dense_vendor_missing_api_fails_loud(monkeypatch, missing):
    init, _, _, _, _ = lazy_provider(True, monkeypatch, missing)
    with pytest.raises(RuntimeError, match="missing " + missing):
        init()


class Launch:
    def __init__(self, callback):
        self.callback = callback
        self.calls = []

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
            self.callback(*args, **kwargs)

        return run


@pytest.mark.parametrize("rank", [0, 1, 3, 7])
def test_ep_wrapper_collective_and_single_owner(monkeypatch, rank):
    routes = torch.tensor([[0, -1], [1, 0]], dtype=torch.int32)
    weights = torch.tensor([[1.0, float("nan")], [0.5, 0.25]])
    values = torch.ones(2, 7, dtype=torch.bfloat16)
    inverse = torch.zeros_like(routes)
    expert_map = torch.tensor([0, -1], dtype=torch.int32)
    output = torch.full_like(values, -17)
    pointer = output.data_ptr()
    events = []
    reduced = torch.arange(28, dtype=torch.float32).reshape(2, 2, 7)
    reduced[:, 1].masked_fill_((routes[:, 1] < 0).unsqueeze(-1), 0)

    def gather(a, ids, inv, owners, slots, **kwargs):
        events.append("slots")
        assert a is values and ids is routes and inv is inverse and owners is expert_map
        assert slots.shape == (2, 2, 7) and slots.dtype == torch.float32
        assert kwargs["GLOBAL_EXPERTS"] == 2
        slots.zero_()

    def all_reduce(slots):
        events.append("all_reduce")
        assert torch.count_nonzero(slots) == 0
        # A distinct tensor catches wrappers that ignore the collective result.
        assert slots.data_ptr() != reduced.data_ptr()
        return reduced

    def combine(slots, route_weights, ids, out, **kwargs):
        events.append("combine")
        assert slots is reduced and route_weights is weights and ids is routes
        assert out is output and kwargs["OUTPUT_STRIDE"] == output.stride(0)
        out.fill_(13)  # Test plumbing, not a substitute numerical FMA oracle.

    group = SimpleNamespace(
        rank_in_group=rank, world_size=8, all_reduce=Mock(side_effect=all_reduce)
    )
    monkeypatch.setitem(
        sys.modules, "vllm.distributed", SimpleNamespace(get_ep_group=lambda: group)
    )
    gather_kernel, combine_kernel = Launch(gather), Launch(combine)
    wrapper, _ = load_function(
        "vllm/model_executor/layers/fused_moe/deep_gemm_utils.py",
        "deepgemm_ep_ordered_unpermute_and_reduce",
        dict(
            torch=torch,
            triton=SimpleNamespace(
                next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
                cdiv=lambda a, b: (a + b - 1) // b,
            ),
            _ep_unweighted_slots_kernel=gather_kernel,
            _ep_ordered_combine_kernel=combine_kernel,
        ),
    )
    assert wrapper(values, routes, weights, inverse, expert_map, output) is None
    assert output.data_ptr() == pointer
    assert torch.equal(output, torch.full_like(output, 13 if rank == 0 else 0))
    assert events == ["slots", "all_reduce"] + (["combine"] if rank == 0 else [])
    assert group.all_reduce.call_count == 1
    assert gather_kernel.calls[0][0] == (2, 2, 1)
    assert len(combine_kernel.calls) == int(rank == 0)


@pytest.mark.parametrize("rank", [0, 1])
def test_ep_wrapper_empty_tokens_has_no_launch_or_collective(monkeypatch, rank):
    group = SimpleNamespace(
        rank_in_group=rank,
        all_reduce=Mock(side_effect=AssertionError("unexpected collective")),
    )
    monkeypatch.setitem(
        sys.modules, "vllm.distributed", SimpleNamespace(get_ep_group=lambda: group)
    )
    kernel = Launch(Mock(side_effect=AssertionError("unexpected launch")))
    wrapper, _ = load_function(
        "vllm/model_executor/layers/fused_moe/deep_gemm_utils.py",
        "deepgemm_ep_ordered_unpermute_and_reduce",
        dict(
            torch=torch,
            _ep_unweighted_slots_kernel=kernel,
            _ep_ordered_combine_kernel=kernel,
        ),
    )
    values = torch.empty(0, 7, dtype=torch.bfloat16)
    routes = torch.empty(0, 2, dtype=torch.int32)
    assert (
        wrapper(values, routes, routes.float(), routes, torch.tensor([0, -1]), values)
        is None
    )
    assert group.all_reduce.call_count == 0
    assert kernel.calls == []
