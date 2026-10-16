# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import deep_ep
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input,
    normalize_batched_scales_shape,
)
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_id,
    dbo_enabled,
    dbo_maybe_run_recv_hook,
)

logger = init_logger(__name__)

# DeepEP kernels quantize dispatch inputs in 128 element chunks.
DEEPEP_QUANT_BLOCK_SIZE = 128
DEEPEP_QUANT_BLOCK_SHAPE = [DEEPEP_QUANT_BLOCK_SIZE, DEEPEP_QUANT_BLOCK_SIZE]

logger = init_logger(__name__)


def dequant_fp8(
    expert_x_fp8: torch.Tensor, expert_x_scales: torch.Tensor
) -> torch.Tensor:
    """Return dequantized tensor in fp32."""
    # TODO (varun) : Optimize leverage num_tokens_per_expert counts
    assert expert_x_fp8.is_contiguous()
    expert_x_scales = expert_x_scales.contiguous()
    num_experts = expert_x_fp8.size(0)

    expert_x_fp32 = expert_x_fp8.to(torch.float32).view(
        num_experts, -1, DEEPEP_QUANT_BLOCK_SIZE
    )
    expert_x_scales = expert_x_scales.view(num_experts, -1, 1)
    return (expert_x_fp32 * expert_x_scales).view(expert_x_fp8.size())


def received_row_weights(
    handle: tuple,
    routes: torch.Tensor,
    weights: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """FP32 routing weight of each row a low-latency dispatch delivered.

    ``routes``/``weights`` are every rank's dispatched top-k ids and weights,
    ``[num_ranks, max_tokens_per_rank, topk]``. The dispatch handle records,
    per local expert, the source token of each received row and the row range
    each source rank occupies (count in the low and begin in the high 32 bits).
    Rows past ``expert_num_tokens`` get 0.
    """
    src_info, layout_range = handle[0], handle[1]
    num_local_experts, capacity = src_info.shape
    device = src_info.device
    rows = torch.arange(capacity, device=device)
    begin = (layout_range >> 32)[:, :, None]
    count = (layout_range & 0xFFFFFFFF)[:, :, None]
    owner = (rows >= begin) & (rows < begin + count)
    src_rank = owner.to(torch.int32).argmax(dim=1)
    src_token = src_info.long().clamp(0, routes.size(1) - 1)
    expert = rank * num_local_experts + torch.arange(num_local_experts, device=device)
    # Top-k ids are distinct, so exactly one slot matches each valid row.
    hit = routes[src_rank, src_token] == expert[:, None, None]
    row_weights = torch.where(hit, weights[src_rank, src_token], 0.0).sum(dim=-1)
    valid = rows < expert_num_tokens[:, None]
    return torch.where(valid, row_weights, 0.0)


class DeepEPLLPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Prepare/Finalize using DeepEP low-latency kernels."""

    # DeepEP low-latency kernels are compiled only for certain
    # specific hidden sizes.
    # NOTE: Keep this list sorted, maybe_roundup_layer_hidden_size depends
    # on it.
    SUPPORTED_HIDDEN_SIZES = [2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192]

    @staticmethod
    def maybe_roundup_layer_hidden_size(hidden_size: int) -> int:
        # Round up hidden size to the closest supported hidden size.
        _supported_hs = DeepEPLLPrepareAndFinalize.SUPPORTED_HIDDEN_SIZES
        # Check sorted
        num_supported_hs = len(_supported_hs)
        assert all(
            [
                _supported_hs[i] < _supported_hs[i + 1]
                for i in range(num_supported_hs - 1)
            ]
        )

        for x in _supported_hs:
            if x >= hidden_size:
                return x

        raise ValueError(
            f"Hidden Size {hidden_size} is greater than the "
            f"maximum supported hidden size {_supported_hs[-1]}"
        )

    def __init__(
        self,
        buffer: deep_ep.Buffer,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        use_fp8_dispatch: bool = False,
        global_to_physical: torch.Tensor | None = None,
        physical_to_global: torch.Tensor | None = None,
        local_expert_global_ids: torch.Tensor | None = None,
    ):
        super().__init__()

        self.buffer = buffer
        self.max_tokens_per_rank = max_tokens_per_rank
        self.use_fp8_dispatch = use_fp8_dispatch
        # The dispatch function returns a handle that the combine function
        # requires. We store the handle here so it is available to the
        # combine function.
        self.handles: list[tuple | None] = [None, None]
        self.num_dispatchers_ = num_dispatchers

        topk_indices_dtype = self.topk_indices_dtype()

        def _maybe_cast(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None or topk_indices_dtype is None:
                return tensor
            return tensor.to(dtype=topk_indices_dtype)

        self.global_to_physical = _maybe_cast(global_to_physical)
        self.physical_to_global = _maybe_cast(physical_to_global)
        self.local_expert_global_ids = _maybe_cast(local_expert_global_ids)

        # We don't have enough information to determine if we should dispatch
        # activation scales in a packed ue8m0 format during object construction
        # time. This setting is handled by post_init_setup.
        self.use_ue8m0_dispatch = False
        # Set by post_init_setup when the experts weight rows before FC2.
        self.weights_rows_before_combine = False
        self.route_tables: list[tuple[torch.Tensor, torch.Tensor] | None] = [
            None,
            None,
        ]

    def post_init_setup(self, fused_experts: mk.FusedMoEExperts):
        self.weights_rows_before_combine = fused_experts.weights_rows_before_combine()
        if self.weights_rows_before_combine and self.use_fp8_dispatch:
            raise RuntimeError(
                "Weighting rows before FC2 requires BF16 low-latency dispatch"
            )
        if not fused_experts.supports_packed_ue8m0_act_scales():
            # Early exit.
            return

        if self.use_fp8_dispatch:
            logger.debug_once(
                "Update DeepEPLLPrepareFinalize to do packed ue8m0 scales dispatch."
            )
            self.use_ue8m0_dispatch = True
        else:
            logger.warning_once(
                "DeepEPLLPrepareAndFinalize is setup to dispatch raw/unquantized "
                f"activations despite ({fused_experts.__class__.__name__}) being able "
                "to support quantized activations.",
            )

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int64

    def _map_global_to_physical_ids(self, topk_ids: torch.Tensor) -> torch.Tensor:
        if self.global_to_physical is None:
            return topk_ids
        return self.global_to_physical[topk_ids]

    def _do_quant(
        self,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        a1_dtype: torch.dtype,
        quant_config: FusedMoEQuantConfig,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_fp8_dispatch:
            block_k = (
                quant_config.block_shape[1]
                if quant_config.block_shape is not None
                else None
            )
            if block_k == DEEPEP_QUANT_BLOCK_SIZE:
                # DeepEP kernels did the quantization for us.
                x, x_scales = x
                return x, x_scales

            # Dequant to get back the tokens in the datatype we dispatched in.
            x_fp8, x_scales = x
            x = dequant_fp8(x_fp8, x_scales).to(dtype=a1_dtype)

        assert isinstance(x, (torch.Tensor, tuple))
        q_dtype = quant_config.quant_dtype

        if q_dtype == "nvfp4" and envs.VLLM_DEEPEPLL_NVFP4_DISPATCH:
            logger.info_once(
                "Since VLLM_DEEPEPLL_NVFP4_DISPATCH==1, make sure "
                "using the hybrid-ep branch of DeepEP"
                "(https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep)"
            )
            assert isinstance(x, tuple)
            x_scales = x[1]
            x = x[0].permute(2, 0, 1)
            num_experts, max_tokens, hidden_dim_by_2 = x.shape
            hidden_dim = hidden_dim_by_2 * 2
            logger.info_once(
                "Quantization is fused with DeepEP nvfp4 dispatch for "
                "FlashInfer CUTEDSL as VLLM_DEEPEPLL_NVFP4_DISPATCH==1"
            )
        else:
            if q_dtype == "nvfp4":
                q_dtype = None
                logger.info_once(
                    "Using DeepEP bfloat16 dispatch for FlashInfer CUTEDSL as "
                    "VLLM_DEEPEPLL_NVFP4_DISPATCH==0"
                )
            assert isinstance(x, torch.Tensor)
            num_experts, max_tokens, hidden_dim = x.size()

            # TODO (varun): Optimization - Use a batched version of quant
            x = x.view((-1, hidden_dim))
            x, x_scales = moe_kernel_quantize_input(
                x,
                quant_config.a1_scale,
                q_dtype,
                quant_config.per_act_token_quant,
                quant_config.block_shape,
                eps=quant_config.a1_eps,
            )
            x = x.view((num_experts, -1, hidden_dim))

        if q_dtype is not None and q_dtype != "nvfp4":
            assert x_scales is not None
            x_scales = normalize_batched_scales_shape(x_scales, num_experts)

        return x, x_scales

    def supports_async(self) -> bool:
        return True

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> tuple[Callable, mk.ReceiverType]:
        if defer_input_quant:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support defer_input_quant=True. "
                "Please select an MoE kernel that accepts quantized inputs."
            )

        hidden_size = a1.size(1)
        assert hidden_size in self.SUPPORTED_HIDDEN_SIZES, (
            f"Hidden Size {hidden_size} not in supported list of hidden sizes"
            f"{self.SUPPORTED_HIDDEN_SIZES}"
        )

        a2a_idx = dbo_current_ubatch_id()

        if self.use_fp8_dispatch:
            assert hidden_size % 128 == 0, (
                "DeepEP kernels quantize the inputs in blocks of shape 128"
            )

        use_nvfp4 = False
        nvfp4_dispatch = (
            quant_config.quant_dtype == "nvfp4" and envs.VLLM_DEEPEPLL_NVFP4_DISPATCH
        )
        if nvfp4_dispatch:
            use_nvfp4 = True
        qc_a1_gscale_or_scale = (
            quant_config.a1_gscale if nvfp4_dispatch else quant_config.a1_scale
        )
        has_per_token_scales = (
            qc_a1_gscale_or_scale.numel() != 1
            if qc_a1_gscale_or_scale is not None
            else (
                quant_config.a2_scale.numel() != 1
                if quant_config.a2_scale is not None
                else False
            )
        )
        if not use_nvfp4:
            assert not has_per_token_scales, (
                "low_latency kernels doesn't support dispatching per-token scales"
            )

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        # Dispatch
        dispatch_topk_ids = self._map_global_to_physical_ids(topk_ids)
        (
            expert_x,
            expert_num_tokens,
            handle,
            _,
            hook,
        ) = self.buffer.low_latency_dispatch(
            a1,
            dispatch_topk_ids,
            self.max_tokens_per_rank,
            num_experts,
            use_fp8=self.use_fp8_dispatch,
            round_scale=self.use_ue8m0_dispatch,
            use_ue8m0=self.use_ue8m0_dispatch,
            **(dict(use_nvfp4=True) if use_nvfp4 else dict()),
            **(
                dict(x_global_scale=qc_a1_gscale_or_scale)
                if qc_a1_gscale_or_scale is not None and nvfp4_dispatch
                else dict()
            ),
            async_finish=False,
            return_recv_hook=True,
        )
        self.handles[a2a_idx] = handle
        if self.weights_rows_before_combine:
            self.route_tables[a2a_idx] = self._gather_routes(
                dispatch_topk_ids, topk_weights
            )

        return (
            hook,
            lambda: self._receiver(
                expert_x,
                expert_num_tokens,
                quant_config.a1_scale,
                a1.dtype,
                quant_config,
                a2a_idx,
            ),
        )

    def _gather_routes(
        self, topk_ids: torch.Tensor, topk_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Every rank's dispatched top-k ids and FP32 weights, padded to the
        dispatch capacity so the collective has a static shape."""
        from vllm.distributed import get_ep_group

        if topk_weights.dtype != torch.float32:
            raise RuntimeError("Row weighting requires FP32 routing weights")
        num_tokens, topk = topk_ids.shape
        routes = topk_ids.new_full((self.max_tokens_per_rank, topk), -1)
        weights = topk_weights.new_zeros((self.max_tokens_per_rank, topk))
        routes[:num_tokens] = topk_ids
        weights[:num_tokens] = topk_weights
        group = get_ep_group()
        routes = group.all_gather(routes, dim=0)
        weights = group.all_gather(weights, dim=0)
        return (
            routes.view(group.world_size, -1, topk),
            weights.view(group.world_size, -1, topk),
        )

    def _receiver(
        self,
        expert_x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        expert_num_tokens: torch.Tensor,
        a1_scale: torch.Tensor | None,
        a1_dtype: torch.dtype,
        quant_config: FusedMoEQuantConfig,
        a2a_idx: int,
    ) -> mk.PrepareResultType:
        expert_x, expert_x_scale = self._do_quant(expert_x, a1_dtype, quant_config)

        row_weights = None
        if self.weights_rows_before_combine:
            from vllm.distributed import get_ep_group

            route_tables = self.route_tables[a2a_idx]
            handle = self.handles[a2a_idx]
            assert route_tables is not None and handle is not None
            row_weights = received_row_weights(
                handle,
                *route_tables,
                expert_num_tokens,
                get_ep_group().rank_in_group,
            )
            self.route_tables[a2a_idx] = None

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens,
            expert_num_tokens_cpu=None,
            expert_row_weights=row_weights,
        )

        return expert_x, expert_x_scale, expert_tokens_meta, None, None

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        if defer_input_quant:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support defer_input_quant=True. "
                "Please select an MoE kernel that accepts quantized inputs."
            )
        hook, receiver = self.prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
        )
        hook()
        return receiver()

    def _finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
        do_async: bool,
    ) -> tuple[Callable, Callable]:
        assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate), (
            "Weight application and reduction happens in the combine kernel."
        )

        a2a_idx = dbo_current_ubatch_id()
        do_recv_hook = dbo_enabled() or do_async
        handle = self.handles[a2a_idx]
        assert handle is not None

        combine_topk_weights = topk_weights
        if apply_router_weight_on_input or self.weights_rows_before_combine:
            # weights have already been applied.
            combine_topk_weights = torch.ones_like(topk_weights)

        combine_topk_ids = self._map_global_to_physical_ids(topk_ids)
        # TODO (varun) : Enable zero copy mode
        dbo_maybe_run_recv_hook()
        _, _, recv_hook = self.buffer.low_latency_combine(
            fused_expert_output,
            combine_topk_ids,
            combine_topk_weights,
            handle,
            async_finish=False,
            zero_copy=False,
            return_recv_hook=do_recv_hook,
            out=output,
        )

        return recv_hook, lambda: None

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> tuple[Callable, Callable]:
        return self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            do_async=True,
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        self._finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
            do_async=False,
        )
