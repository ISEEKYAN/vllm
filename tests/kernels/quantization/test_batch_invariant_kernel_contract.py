# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.layers.quantization.utils import fp8_utils

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def test_required_batch_invariant_kernel_uses_packaged_default(monkeypatch, tmp_path):
    monkeypatch.delenv("VLLM_BATCH_INVARIANT_KERNEL_LIB", raising=False)
    packaged = tmp_path / "_vllm_batch_invariant_C.so"
    monkeypatch.setattr(fp8_utils, "_PACKAGED_BATCH_INVARIANT_KERNEL", packaged)

    with pytest.raises(RuntimeError, match="library does not exist"):
        fp8_utils.require_batch_invariant_quant_kernel()


def test_required_batch_invariant_kernel_loads_configured_library(monkeypatch):
    calls = []
    monkeypatch.setenv("VLLM_BATCH_INVARIANT_KERNEL_LIB", "/test/bi-kernel.so")
    monkeypatch.setattr(
        fp8_utils,
        "_load_batch_invariant_kernel_library",
        lambda path: calls.append(path),
    )

    fp8_utils.require_batch_invariant_quant_kernel()

    assert calls == ["/test/bi-kernel.so"]


def test_required_batch_invariant_kernel_rejects_missing_library(monkeypatch, tmp_path):
    missing = tmp_path / "_vllm_batch_invariant_C.so"
    monkeypatch.setenv("VLLM_BATCH_INVARIANT_KERNEL_LIB", str(missing))

    with pytest.raises(RuntimeError, match="library does not exist"):
        fp8_utils.require_batch_invariant_quant_kernel()


def test_contiguous_deepgemm_forwards_clamp_to_bi_kernel(monkeypatch):
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts import deep_gemm_moe
    from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT

    calls = []

    def fused(value, **kwargs):
        calls.append(kwargs)
        return kwargs["output_q"], torch.ones(1)

    monkeypatch.setattr(
        deep_gemm_moe, "is_batch_invariant_quant_kernel_enabled", lambda: True
    )
    monkeypatch.setattr(
        DeepGemmQuantScaleFMT,
        "from_oracle",
        staticmethod(lambda: DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0),
    )
    monkeypatch.setattr(
        deep_gemm_moe, "fused_silu_mul_per_token_group_quant_fp8", fused
    )
    expert = SimpleNamespace(
        block_shape=[128, 128],
        gemm1_clamp_limit=10.0,
        gemm1_alpha=1.0,
        gemm1_beta=0.0,
        adjust_N_for_activation=lambda n, _activation: n // 2,
    )
    value = torch.randn(2, 256, dtype=torch.bfloat16)
    output = torch.empty(2, 128, dtype=torch.float8_e4m3fn)

    quantized, _scales = deep_gemm_moe.DeepGemmExperts._act_mul_quant(
        expert, value, output, MoEActivation.SILU
    )

    assert quantized is output
    assert len(calls) == 1
    assert calls[0]["output_q"] is output
    assert calls[0]["use_ue8m0"] is False
    assert calls[0]["round_scale"] is True
    assert calls[0]["clamp_limit"] == 10.0
    assert calls[0]["masked_m"] is None
    assert calls[0]["group_size"] == 128


def test_masked_deepgemm_forwards_clamp_to_bi_kernel(monkeypatch):
    from vllm.model_executor.layers.fused_moe.experts import batched_deep_gemm_moe
    from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT

    calls = []

    def fused(value, **kwargs):
        calls.append(kwargs)
        return torch.empty(1), torch.empty(1)

    monkeypatch.setattr(
        batched_deep_gemm_moe,
        "is_batch_invariant_quant_kernel_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        batched_deep_gemm_moe,
        "fused_silu_mul_per_token_group_quant_fp8",
        fused,
    )
    value = torch.randn(2, 3, 256, dtype=torch.bfloat16)
    counts = torch.tensor([2, 3], dtype=torch.int32)

    batched_deep_gemm_moe.persistent_masked_m_silu_mul_quant(
        value,
        counts,
        quant_scale_fmt=DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0,
        clamp_limit=10.0,
    )

    assert calls[0]["round_scale"] is True
    assert calls[0]["clamp_limit"] == 10.0
    assert calls[0]["masked_m"] is counts


def test_quant_kernel_gated_on_batch_invariant_flag(monkeypatch, tmp_path):
    """The BI fused quant kernel must not be selected when BI is disabled."""
    lib = tmp_path / "_vllm_batch_invariant_C.so"
    lib.write_bytes(b"")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT_KERNEL_LIB", str(lib))
    monkeypatch.setattr(
        fp8_utils, "_load_batch_invariant_kernel_library", lambda path: None
    )

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    assert fp8_utils.batch_invariant_quant_kernel_available()
    assert not fp8_utils.is_batch_invariant_quant_kernel_enabled()

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    assert fp8_utils.is_batch_invariant_quant_kernel_enabled()


def test_fp4_probe_does_not_accept_only_a_masked_fp8_api(monkeypatch):
    from types import SimpleNamespace

    from vllm.utils import deep_gemm

    package = SimpleNamespace(
        set_batch_invariant=lambda _: None,
        get_batch_invariant=lambda: True,
        m_grouped_fp8_gemm_nt_masked=lambda: None,
    )
    monkeypatch.setattr(deep_gemm, "_import_deep_gemm", lambda: package)
    deep_gemm.supports_deep_gemm_batch_invariance.cache_clear()
    try:
        assert deep_gemm.supports_deep_gemm_batch_invariance()
        assert not deep_gemm.supports_deep_gemm_batch_invariance(fp4_contiguous=True)
    finally:
        deep_gemm.supports_deep_gemm_batch_invariance.cache_clear()


@pytest.mark.parametrize("backend", ["auto", "marlin", "flashinfer_trtllm"])
def test_ds41_bi_rejects_a_different_activation_quantization_backend(
    monkeypatch, backend
):
    from types import SimpleNamespace

    from vllm.models.deepseek_v41 import quant_config

    monkeypatch.setattr(quant_config, "RoutedExperts", SimpleNamespace)
    monkeypatch.setattr(quant_config, "is_layer_skipped", lambda **_: False)
    selected = object()
    monkeypatch.setattr(quant_config, "Mxfp4MoEMethod", lambda _: selected)
    config = SimpleNamespace(
        weight_block_size=None,
        expert_dtype="fp4",
        moe_quant_algo=None,
        ignored_layers=[],
        packed_modules_mapping={},
    )
    layer = SimpleNamespace(moe_config=SimpleNamespace(moe_backend=backend))
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    with pytest.raises(RuntimeError, match="explicit.*moe_backend='deep_gemm'"):
        quant_config.DeepseekV4FP8Config.get_quant_method(config, layer, "experts")
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    assert (
        quant_config.DeepseekV4FP8Config.get_quant_method(config, layer, "experts")
        is selected
    )


def test_ds41_bi_rejects_mega_moe_before_distributed_initialization(monkeypatch):
    from types import SimpleNamespace

    from vllm.models.deepseek_v41.nvidia.model import DeepseekV4MoE

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(expert_dtype="fp4")),
        kernel_config=SimpleNamespace(moe_backend="deep_gemm_mega_moe"),
    )
    with pytest.raises(RuntimeError, match="does not support MegaMoE"):
        DeepseekV4MoE(config)


@pytest.mark.parametrize("m", [1, 31, 32, 33, 128, 513])
@pytest.mark.parametrize("cpu_counts", [False, True])
def test_fp4_fixed_alignment_ignores_batch_heuristic(monkeypatch, m, cpu_counts):
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
        compute_aligned_M_and_alignment,
    )
    from vllm.utils import deep_gemm

    monkeypatch.setattr(
        deep_gemm,
        "get_theoretical_mk_alignment_for_contiguous_layout",
        lambda **_: 32,
    )
    meta = (
        SimpleNamespace(expert_num_tokens_cpu=torch.tensor([m, 0]))
        if cpu_counts
        else None
    )
    capacity, alignment = compute_aligned_M_and_alignment(
        m,
        1,
        2,
        64,
        meta,
        fixed_alignment=128,
    )
    assert alignment == 128 and capacity % 128 == 0 and capacity >= m
    _, legacy_alignment = compute_aligned_M_and_alignment(m, 1, 2, 64, meta)
    assert legacy_alignment == (64 if cpu_counts else 32)


@pytest.mark.parametrize("m", [1, 511, 512, 513])
def test_fp4_a2_keeps_the_same_row_and_scale_tile(monkeypatch, m):
    launches = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: launches.append(kwargs)

    monkeypatch.setattr(fp8_utils, "_silu_mul_quant_fp8_packed_kernel", Kernel())
    x = torch.zeros(m, 256, dtype=torch.bfloat16)
    for bi in (True, False):
        fp8_utils.silu_mul_quant_fp8_packed_triton(x, batch_invariant=bi)
    assert (launches[0]["BLOCK_M"], launches[0]["PACKS_PER_CTA"]) == (1, 2)
    assert (launches[1]["BLOCK_M"], launches[1]["PACKS_PER_CTA"]) == (
        (1, 2) if m < 512 else (4, 1)
    )


@pytest.mark.parametrize(
    "failure,match",
    [
        ("device", "SM100"),
        ("api", "control APIs"),
        ("parallel", "TP=EP"),
        ("activation", "SITU"),
        ("dtype", "BF16"),
        ("shape", "divisible"),
        ("scale", "UE8M0"),
        ("clamp", "positive clamp"),
        ("bias", "biases"),
    ],
)
def test_fp4_bi_rejects_unverified_contract(monkeypatch, failure, match):
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.fused_moe import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts import deep_gemm_moe
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        make_mxfp4_moe_quant_config,
    )
    from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(
        deep_gemm_moe.current_platform,
        "is_device_capability_family",
        lambda _: failure != "device",
    )
    monkeypatch.setattr(
        deep_gemm_moe,
        "supports_deep_gemm_batch_invariance",
        lambda **_: failure != "api",
    )
    monkeypatch.setattr(
        DeepGemmQuantScaleFMT,
        "from_oracle",
        lambda: (
            DeepGemmQuantScaleFMT.FLOAT32
            if failure == "scale"
            else DeepGemmQuantScaleFMT.UE8M0
        ),
    )
    config = make_dummy_moe_config(hidden_dim=128, intermediate_size=128)
    quant = make_mxfp4_moe_quant_config(
        Mxfp4MoeBackend.DEEPGEMM_MXFP4,
        torch.ones(1),
        torch.ones(1),
        swiglu_limit=float("nan") if failure == "clamp" else 10.0,
        w1_bias=torch.ones(1) if failure == "bias" else None,
    )
    if failure == "parallel":
        config.moe_parallel_config.ep_size = 2
    elif failure == "activation":
        config.activation = MoEActivation.SITU
    elif failure == "dtype":
        config.in_dtype = torch.float16
    elif failure == "shape":
        config.intermediate_size = 192
    with pytest.raises(RuntimeError, match=match):
        deep_gemm_moe.DeepGemmFP4Experts(config, quant)
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    # BI=0 retains the previous constructor's acceptance and numeric path.
    deep_gemm_moe.DeepGemmFP4Experts(config, quant)


@pytest.mark.parametrize(
    "failure,match",
    [
        (None, None),
        ("device", "batch invariance"),
        ("api", "batch invariance"),
        ("scale", "batch invariance"),
        *[
            (size, "TP=EP")
            for size in ("tp_size", "ep_size", "dp_size", "pcp_size", "sp_size")
        ],
        ("enable_eplb", "EPLB"),
        ("SITU", "SITU"),
        ("SWIGLUSTEP", "SWIGLUSTEP"),
        ("dtype", "BF16"),
        ("hidden_dim", "divisible"),
        ("intermediate_size", "divisible"),
        ("batched", "activation format"),
    ],
)
@pytest.mark.parametrize("bi", [False, True])
def test_fp4_selector_batch_invariance_contract(monkeypatch, failure, match, bi):
    """Select the contiguous BI path and fail closed before kernel construction."""
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.fused_moe import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts import deep_gemm_moe
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend,
        make_mxfp4_moe_quant_config,
        select_deepseek_v4_mxfp4_moe_backend,
    )
    from vllm.utils import deep_gemm

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", bi)
    monkeypatch.setattr(deep_gemm_moe, "is_deep_gemm_supported", lambda: True)
    monkeypatch.setattr(
        deep_gemm_moe.current_platform,
        "is_device_capability_family",
        lambda family: family == (120 if failure == "device" else 100),
    )

    def supports_bi(*, fp4_contiguous=False):
        assert fp4_contiguous
        return failure != "api"

    monkeypatch.setattr(
        deep_gemm_moe, "supports_deep_gemm_batch_invariance", supports_bi
    )
    monkeypatch.setattr(
        deep_gemm.DeepGemmQuantScaleFMT,
        "from_oracle",
        lambda: (
            deep_gemm.DeepGemmQuantScaleFMT.FLOAT32
            if failure == "scale"
            else deep_gemm.DeepGemmQuantScaleFMT.UE8M0
        ),
    )
    config = make_dummy_moe_config(hidden_dim=128, intermediate_size=128)
    config.moe_backend = "deep_gemm"
    if failure in ("SITU", "SWIGLUSTEP"):
        config.activation = MoEActivation[failure]
    elif failure == "dtype":
        config.in_dtype = torch.float16
    elif failure in ("hidden_dim", "intermediate_size"):
        setattr(config, failure, 192)
    elif failure in ("tp_size", "ep_size", "dp_size", "pcp_size", "sp_size"):
        setattr(config.moe_parallel_config, failure, 2)
    elif failure == "enable_eplb":
        config.moe_parallel_config.enable_eplb = True
    elif failure == "batched":
        monkeypatch.setattr(
            type(config.moe_parallel_config),
            "use_batched_activation_format",
            property(lambda _: True),
        )

    if (bi and failure is not None) or failure == "batched":
        with pytest.raises(ValueError, match=match):
            select_deepseek_v4_mxfp4_moe_backend(config)
    else:
        backend, experts_cls = select_deepseek_v4_mxfp4_moe_backend(config)
        assert backend == Mxfp4MoeBackend.DEEPGEMM_MXFP4
        assert experts_cls is deep_gemm_moe.DeepGemmFP4Experts
        for clamp in (None, 10.0):
            quant = make_mxfp4_moe_quant_config(
                backend, torch.ones(1), torch.ones(1), swiglu_limit=clamp
            )
            experts_cls(config, quant)


@pytest.mark.parametrize("bi", [False, True])
@pytest.mark.parametrize("original_threshold", [1, 5])
def test_ds41_bi_classifies_all_queries_for_the_same_sparse_kernel(
    monkeypatch, bi, original_threshold
):
    """BI must not switch arithmetic for single-token or speculative queries."""
    from types import SimpleNamespace

    from vllm.models.deepseek_v41.nvidia.flashmla import (
        DeepseekSparseSWAFlashMLAMetadataBuilder,
        DeepseekV41SparseSWAMetadataBuilder,
    )
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm.v1.attention.backends.utils import split_decodes_and_prefills

    def init_metadata(self):
        self.decode_threshold = original_threshold

    monkeypatch.setattr(DeepseekV41SparseSWAMetadataBuilder, "__init__", init_metadata)
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", bi)
    builder = DeepseekSparseSWAFlashMLAMetadataBuilder()
    batch = SimpleNamespace(
        max_query_len=8,
        num_reqs=3,
        num_actual_tokens=10,
        query_start_loc_cpu=torch.tensor([0, 1, 2, 10], dtype=torch.int32),
    )
    assert split_decodes_and_prefills(batch, builder.decode_threshold) == (
        (0, 3, 0, 10) if bi else (2, 1, 2, 8)
    )
    assert builder.get_cudagraph_support(None, None) == (
        AttentionCGSupport.NEVER if bi else AttentionCGSupport.ALWAYS
    )


@pytest.mark.parametrize("bi", [False, True])
@pytest.mark.parametrize("hidden_size,splits", [(512, 4), (1280, 2)])
def test_bi_mhc_keeps_post_pre_rounding_across_token_batches(
    monkeypatch, bi, hidden_size, splits
):
    """M>32 must not introduce an intermediate BF16 store in BI mode."""
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        mhc_fused_post_pre_split_config,
    )

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", bi)
    sizes = [1, 15, 16, 32, 33, 64, 513]
    configs = [mhc_fused_post_pre_split_config(m, hidden_size, 4) for m in sizes]
    expected = (
        [(2, splits, 128)] * len(sizes)
        if bi
        else [
            (2, splits, 128),
            (2, splits, 128),
            (6, splits, 128),
            (6, splits, 128),
            None,
            None,
            None,
        ]
    )
    assert configs == expected


@pytest.mark.parametrize("bi", [False, True])
def test_bi_mhc_retains_unsupported_row_geometry(monkeypatch, bi):
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        mhc_fused_post_pre_split_config,
    )

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", bi)
    assert mhc_fused_post_pre_split_config(64, 65, 4) is None
