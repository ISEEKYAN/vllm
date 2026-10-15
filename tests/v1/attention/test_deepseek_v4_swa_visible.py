# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DeepSeek-V4 in-image bidirectional SWA visibility (vision variant).

The reference semantics are a torch port of the official
``get_image_visible`` / ``get_window_topk_idxs_visible``: inside an image span
[span_start, span_end] (inclusive), a token at ``pos`` sees
``min(pos - span_start, max_image_tokens - 1)`` extra tokens to the left and
``min(span_end - pos, max_image_tokens)`` to the right; the window then starts
at ``max(pos - (window - 1) - max(left - (window - 1), 0), 0)`` and ends at
``pos + right`` (inclusive).
"""

import inspect

import pytest
import torch
from transformers import DeepseekV4Config
from typing_extensions import TypedDict

from tests.v1.attention.utils import create_vllm_config
from vllm.models.deepseek_v4.common.ops.cache_utils import (
    build_flashinfer_mixed_sparse_indices,
    combine_topk_swa_indices,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWAMetadataBuilder,
    _compute_image_visibility_kernel,
    _compute_swa_indices_and_lens_kernel,
)
from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

WINDOW = 8
MAX_IMG = 6
WIDTH = WINDOW + MAX_IMG
BLOCK_SIZE = 64


def ref_left_right(
    seq_lens: list[int],
    query_lens: list[int],
    spans_per_req: list[list[tuple[int, int]]],
    max_image_tokens: int,
) -> tuple[list[int], list[int]]:
    """Per-token (left, right) for the flattened decode-first token stream."""
    lefts: list[int] = []
    rights: list[int] = []
    for seq_len, query_len, spans in zip(seq_lens, query_lens, spans_per_req):
        prefix_len = seq_len - query_len
        for i in range(query_len):
            pos = prefix_len + i
            left = right = 0
            for span_start, span_end in spans:
                if span_start <= pos <= span_end:
                    left = min(pos - span_start, max_image_tokens - 1)
                    right = min(span_end - pos, max_image_tokens)
            lefts.append(left)
            rights.append(right)
    return lefts, rights


def ref_swa_bounds(
    pos: int, window: int, left: int, right: int, replay_start: int = 0
) -> tuple[int, int]:
    """Reference [start, end) window bounds for one query token. Under SWA
    bounded replay no window KV exists below ``replay_start``."""
    left_add = max(left - (window - 1), 0)
    start = max(pos - (window - 1) - left_add, 0, replay_start)
    return start, pos + right + 1


def make_batch(
    seq_lens: list[int],
    query_lens: list[int],
    device: torch.device,
):
    """Build the kernel inputs for a decode-first batch."""
    num_reqs = len(seq_lens)
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = torch.tensor(
        query_lens, dtype=torch.int32, device=device
    ).cumsum(0)
    num_tokens = int(query_start_loc[-1])
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    token_to_req = torch.repeat_interleave(
        torch.arange(num_reqs, dtype=torch.int32, device=device),
        torch.tensor(query_lens, dtype=torch.int32, device=device),
    )
    max_blocks = (max(seq_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_table = torch.arange(
        num_reqs * max_blocks, dtype=torch.int32, device=device
    ).view(num_reqs, max_blocks)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    return query_start_loc, seq_lens_t, token_to_req, slot_mapping, block_table


def ref_swa_slot_rows(
    seq_lens: list[int],
    query_lens: list[int],
    spans_per_req: list[list[tuple[int, int]]],
    block_table: torch.Tensor,
    window: int,
    max_image_tokens: int,
    width: int,
    replay_starts: list[int] | None = None,
) -> tuple[list[list[int]], list[int]]:
    """Reference paged slot-id rows and lens for every token in the batch."""
    block_table_cpu = block_table.cpu()
    lefts, rights = ref_left_right(
        seq_lens, query_lens, spans_per_req, max_image_tokens
    )
    replay_starts = replay_starts or [0] * len(seq_lens)
    rows: list[list[int]] = []
    lens: list[int] = []
    token = 0
    for req, (seq_len, query_len) in enumerate(zip(seq_lens, query_lens)):
        prefix_len = seq_len - query_len
        for i in range(query_len):
            pos = prefix_len + i
            start, end = ref_swa_bounds(
                pos, window, lefts[token], rights[token], replay_starts[req]
            )
            row = []
            for p in range(start, end):
                blk = int(block_table_cpu[req, p // BLOCK_SIZE])
                row.append(blk * BLOCK_SIZE + p % BLOCK_SIZE)
            lens.append(len(row))
            row.extend([-1] * (width - len(row)))
            rows.append(row)
            token += 1
    return rows, lens


def run_swa_kernel(
    seq_lens: list[int],
    query_lens: list[int],
    spans_per_req: list[list[tuple[int, int]]],
    window: int = WINDOW,
    max_image_tokens: int = MAX_IMG,
    with_image: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    query_start_loc, seq_lens_t, token_to_req, slot_mapping, block_table = make_batch(
        seq_lens, query_lens, device
    )
    num_tokens = int(query_start_loc[-1])
    width = window + (max_image_tokens if with_image else 0)
    swa_indices = torch.zeros(num_tokens, 1, width, dtype=torch.int32, device=device)
    swa_lens = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    is_valid = slot_mapping >= 0
    replay_start_t = torch.zeros(len(seq_lens), dtype=torch.int32, device=device)

    if with_image:
        lefts, rights = ref_left_right(
            seq_lens, query_lens, spans_per_req, max_image_tokens
        )
        left_t = torch.tensor(lefts, dtype=torch.int32, device=device)
        right_t = torch.tensor(rights, dtype=torch.int32, device=device)
    else:
        left_t = right_t = swa_lens  # unused dummies

    _compute_swa_indices_and_lens_kernel[(num_tokens,)](
        swa_indices,
        swa_indices.stride(0),
        swa_lens,
        window,
        width,
        left_t,
        right_t,
        query_start_loc,
        seq_lens_t,
        token_to_req,
        is_valid,
        block_table,
        block_table.stride(0),
        BLOCK_SIZE,
        replay_start_t,
        token_offset=0,
        HAS_IMAGE=with_image,
        TRITON_BLOCK_SIZE=1024,
    )
    return swa_indices[:, 0], swa_lens


# seq_lens, query_lens, spans per request (positions are prompt-absolute,
# inclusive on both ends, matching the reference's sentinel-bracketed spans).
class _Case(TypedDict):
    seq_lens: list[int]
    query_lens: list[int]
    spans: list[list[tuple[int, int]]]


CASES: list[_Case] = [
    # two image spans in one request, one pure-text request
    {
        "seq_lens": [30, 12],
        "query_lens": [30, 12],
        "spans": [[(4, 12), (20, 24)], []],
    },
    # span larger than max_image_tokens, straddling the window boundary
    {
        "seq_lens": [26],
        "query_lens": [26],
        "spans": [[(2, 17)]],
    },
    # chunked prefill: span fully inside the second chunk's query
    {
        "seq_lens": [40, 9],
        "query_lens": [16, 9],
        "spans": [[(30, 38)], []],
    },
    # span ending exactly at the prompt end; tiny request
    {
        "seq_lens": [7, 15],
        "query_lens": [7, 15],
        "spans": [[(0, 6)], [(10, 14)]],
    },
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("case", CASES)
def test_swa_indices_kernel_with_image_spans(case):
    rows, lens = ref_swa_slot_rows(
        case["seq_lens"],
        case["query_lens"],
        case["spans"],
        make_batch(case["seq_lens"], case["query_lens"], torch.device("cuda"))[4],
        WINDOW,
        MAX_IMG,
        WIDTH,
    )
    indices, actual_lens = run_swa_kernel(
        case["seq_lens"], case["query_lens"], case["spans"], with_image=True
    )
    assert actual_lens.cpu().tolist() == lens
    assert indices.cpu().tolist() == rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("case", CASES)
def test_swa_indices_kernel_without_image_unchanged(case):
    """HAS_IMAGE=False must reproduce the plain causal sliding window."""
    rows, lens = ref_swa_slot_rows(
        case["seq_lens"],
        case["query_lens"],
        [[] for _ in case["seq_lens"]],
        make_batch(case["seq_lens"], case["query_lens"], torch.device("cuda"))[4],
        WINDOW,
        MAX_IMG,
        WINDOW,
    )
    indices, actual_lens = run_swa_kernel(
        case["seq_lens"], case["query_lens"], case["spans"], with_image=False
    )
    assert indices.shape[-1] == WINDOW
    assert actual_lens.cpu().tolist() == lens
    assert indices.cpu().tolist() == rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_image_visibility_kernel():
    """The builder's visibility kernel must match get_image_visible."""
    device = torch.device("cuda")
    seq_lens = [30, 12]
    query_lens = [30, 12]
    spans = [[(4, 12), (20, 24)], []]
    query_start_loc, seq_lens_t, token_to_req, _, _ = make_batch(
        seq_lens, query_lens, device
    )
    num_tokens = int(query_start_loc[-1])

    # CSR span layout: request -> contiguous [start, end) span list.
    indptr = [0, 2, 2]
    starts = [4, 20]
    ends = [12, 24]
    left = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    right = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    _compute_image_visibility_kernel[(num_tokens,)](
        left,
        right,
        torch.tensor(indptr, dtype=torch.int32, device=device),
        torch.tensor(starts, dtype=torch.int32, device=device),
        torch.tensor(ends, dtype=torch.int32, device=device),
        query_start_loc,
        seq_lens_t,
        token_to_req,
        MAX_IMG,
        token_offset=0,
    )
    ref_left, ref_right = ref_left_right(seq_lens, query_lens, spans, MAX_IMG)
    assert left.cpu().tolist() == ref_left
    assert right.cpu().tolist() == ref_right


def combine_case(
    compress_ratio: int,
    topk: int,
    seq_lens: list[int],
    query_lens: list[int],
    spans: list[list[tuple[int, int]]],
    with_image: bool,
    replay_starts: list[int] | None = None,
    combine_fn=combine_topk_swa_indices,
    in_range_topk: bool = False,
):
    """Run combine_topk_swa_indices and return (indices, lens, expected).

    ``in_range_topk`` draws top-k indices below N, for combiners that drop
    out-of-range ones.
    """
    device = torch.device("cuda")
    num_reqs = len(seq_lens)
    replay_starts = replay_starts or [0] * num_reqs
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = torch.tensor(
        query_lens, dtype=torch.int32, device=device
    ).cumsum(0)
    num_tokens = int(query_start_loc[-1])
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    # The builder's gather covers only the context above the replay start.
    gather_lens = torch.tensor(
        [
            q + min(max(s - q - r, 0), WINDOW - 1)
            for s, q, r in zip(seq_lens, query_lens, replay_starts)
        ],
        dtype=torch.int32,
        device=device,
    )
    N = (max(seq_lens) + compress_ratio - 1) // compress_ratio
    M = N + int(gather_lens.max()) + 8
    gen = torch.Generator(device="cpu").manual_seed(0)
    topk_indices = torch.randint(
        0,
        N if in_range_topk else 4096,
        (num_tokens, max(topk, 1)),
        generator=gen,
        dtype=torch.int32,
    ).to(device)
    topk_indices = topk_indices[:, : max(topk, 1)]

    if with_image:
        lefts, rights = ref_left_right(seq_lens, query_lens, spans, MAX_IMG)
        left_t = torch.tensor(lefts, dtype=torch.int32, device=device)
        right_t = torch.tensor(rights, dtype=torch.int32, device=device)
    else:
        left_t = right_t = None

    # V4.1's combine has no image windows (causal SWA only).
    has_image = "left_visible" in inspect.signature(combine_fn).parameters
    image_kwargs = (
        dict(left_visible=left_t, right_visible=right_t, max_image_tokens=MAX_IMG)
        if has_image
        else {}
    )
    combined_indices, combined_lens = combine_fn(
        topk_indices,
        query_start_loc,
        seq_lens_t,
        gather_lens,
        WINDOW,
        compress_ratio,
        topk,
        M,
        N,
        **image_kwargs,
    )

    # Reference rows.
    lefts, rights = ref_left_right(
        seq_lens,
        query_lens,
        spans if with_image else [[] for _ in seq_lens],
        MAX_IMG,
    )
    topk_cpu = topk_indices.cpu()
    width = WINDOW + (MAX_IMG if has_image else 0)
    combined_topk = (topk + width + 127) // 128 * 128
    rows = []
    lens = []
    token = 0
    for b, (seq_len, query_len) in enumerate(zip(seq_lens, query_lens)):
        prefix_len = seq_len - query_len
        gather_start = seq_len - int(gather_lens[b])
        for i in range(query_len):
            pos = prefix_len + i
            topk_len = min((pos + 1) // compress_ratio, topk)
            start, end = ref_swa_bounds(pos, WINDOW, lefts[token], rights[token])
            # The window never reaches below the gathered buffer.
            start = max(start, gather_start)
            swa_len = end - start
            row = [-1] * combined_topk
            for j in range(topk_len):
                row[j] = int(topk_cpu[token, j]) + M * b
            for j in range(swa_len):
                row[topk_len + j] = M * b + N + start + j - gather_start
            rows.append(row)
            lens.append(topk_len + swa_len)
            token += 1
    return combined_indices, combined_lens, rows, lens


COMBINE_CASES = [
    dict(compress_ratio=1, topk=0),  # SWA-only layer
    dict(compress_ratio=4, topk=16),  # C4A layer
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("cfg", COMBINE_CASES)
def test_combine_topk_swa_with_image_spans(cfg):
    case = CASES[0]
    indices, lens, rows, exp_lens = combine_case(
        cfg["compress_ratio"],
        cfg["topk"],
        case["seq_lens"],
        case["query_lens"],
        case["spans"],
        with_image=True,
    )
    assert lens.cpu().tolist() == exp_lens
    assert indices.cpu().tolist() == rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("cfg", COMBINE_CASES)
@pytest.mark.parametrize("query_len", [24, 20])
def test_v41_combine_topk_swa_stops_at_replay_start(cfg, query_len):
    """SWA bounded replay: the gathered buffer starts at replay_start, so the
    window never indexes below it."""
    from vllm.models.deepseek_v41.common.ops.cache_utils import (
        combine_topk_swa_indices as combine_v41,
    )

    # Request 0 replays from 16: a 24-token chunk starts there, a 20-token
    # chunk leaves only 4 context tokens above it (fewer than the window).
    indices, lens, rows, exp_lens = combine_case(
        cfg["compress_ratio"],
        cfg["topk"],
        seq_lens=[40, 12],
        query_lens=[query_len, 12],
        spans=[[], []],
        with_image=False,
        replay_starts=[16, 0],
        combine_fn=combine_v41,
        in_range_topk=True,
    )
    assert lens.cpu().tolist() == exp_lens
    assert indices.cpu().tolist() == rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("cfg", COMBINE_CASES)
def test_combine_topk_swa_without_image_unchanged(cfg):
    """left_visible=None must reproduce the plain causal combined indices."""
    case = CASES[0]
    indices, lens, rows, exp_lens = combine_case(
        cfg["compress_ratio"],
        cfg["topk"],
        case["seq_lens"],
        case["query_lens"],
        case["spans"],
        with_image=False,
    )
    assert lens.cpu().tolist() == exp_lens
    assert indices.cpu().tolist() == rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flashinfer_mixed_sparse_indices_with_image_spans():
    device = torch.device("cuda")
    # 1 decode token (req 0) + two prefill requests (reqs 1, 2).
    seq_lens = [20, 30, 9]
    query_lens = [1, 30, 9]
    spans = [[], [(4, 12)], [(0, 6)]]
    query_start_loc = torch.tensor([0, 1, 31, 40], dtype=torch.int32, device=device)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    token_to_req = torch.tensor(
        [0] + [1] * 30 + [2] * 9, dtype=torch.int32, device=device
    )
    block_table = torch.arange(3, dtype=torch.int32, device=device).view(3, 1)
    decode_swa = torch.tensor(
        [[100, 101, 102, 103, 104, 105, 106, 107]], dtype=torch.int32, device=device
    )
    prefill_topk = torch.zeros(39, 0, dtype=torch.int32, device=device)

    lefts, rights = ref_left_right(seq_lens, query_lens, spans, MAX_IMG)
    left_t = torch.tensor(lefts, dtype=torch.int32, device=device)
    right_t = torch.tensor(rights, dtype=torch.int32, device=device)

    sparse_indices, sparse_lens = build_flashinfer_mixed_sparse_indices(
        decode_swa_indices=decode_swa,
        decode_compressed_indices=None,
        decode_compressed_topk_lens=None,
        prefill_topk_indices=prefill_topk,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens_t,
        token_to_req_indices=token_to_req,
        swa_block_table=block_table,
        swa_block_size=BLOCK_SIZE,
        compressed_block_table=None,
        compressed_block_size=BLOCK_SIZE,
        window_size=WINDOW,
        compress_ratio=1,
        topk=0,
        prefill_left_visible=left_t,
        prefill_right_visible=right_t,
        max_image_tokens=MAX_IMG,
    )

    swa_total = WINDOW + MAX_IMG
    assert sparse_indices.shape == (40, swa_total)
    assert sparse_lens.cpu().tolist() == [swa_total] * 40
    # Decode row: slots copied, image-extension columns padded with -1.
    decode_row = sparse_indices[0].cpu().tolist()
    assert decode_row[:WINDOW] == list(range(100, 108))
    assert decode_row[WINDOW:] == [-1] * MAX_IMG
    # Prefill rows: paged slot ids over the widened window.
    exp_rows, _ = ref_swa_slot_rows(
        seq_lens, query_lens, spans, block_table, WINDOW, MAX_IMG, swa_total
    )
    actual = sparse_indices[1:].cpu().tolist()
    assert actual == exp_rows[1:]


def make_builder(
    vision: bool,
    window: int = WINDOW,
    max_image_tokens: int = MAX_IMG,
    max_num_batched_tokens: int = 64,
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    builder_cls: type[
        DeepseekSparseSWAMetadataBuilder
    ] = DeepseekSparseSWAMetadataBuilder,
    deepseek_v4_config: bool = True,
    bounded_replay: bool = False,
) -> DeepseekSparseSWAMetadataBuilder:
    vision_fields = dict(vision_n_layers=2, vision_max_n_token=max_image_tokens)
    vllm_config = create_vllm_config(
        model_name=model_name,
        max_model_len=4096,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=8,
        hf_config_override={"sliding_window": window}
        | (vision_fields if vision and not deepseek_v4_config else {}),
    )
    if vision and deepseek_v4_config:
        # Only V4 configs widen the SWA window in-kernel over image spans
        vllm_config.model_config.hf_config = DeepseekV4Config(
            num_hidden_layers=1,
            layer_types=["sliding_attention"],
            sliding_window=window,
            **vision_fields,
        )
    spec = SlidingWindowMLASpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=window,
        cache_dtype_str="auto",
        model_version="deepseek_v4",
        bounded_replay=bounded_replay,
    )
    return builder_cls(
        kv_cache_spec=spec,
        layer_names=["layer0"],
        vllm_config=vllm_config,
        device=torch.device("cuda"),
    )


def build_metadata(
    builder: DeepseekSparseSWAMetadataBuilder,
    seq_lens: list[int],
    query_lens: list[int],
    mm_req_doc_ranges: dict[int, list[tuple[int, int]]] | None,
    replay_starts: list[int] | None = None,
):
    device = torch.device("cuda")
    query_start_loc, seq_lens_t, _, slot_mapping, block_table = make_batch(
        seq_lens, query_lens, device
    )
    replay_start = (
        None
        if replay_starts is None
        else torch.tensor(replay_starts, dtype=torch.int32, device=device)
    )
    return builder.build(
        0,
        CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc.cpu(),
            seq_lens=seq_lens_t,
            seq_lens_cpu_upper_bound=seq_lens_t.cpu(),
            num_reqs=len(seq_lens),
            num_actual_tokens=int(query_start_loc[-1]),
            max_query_len=max(query_lens),
            max_seq_len=max(seq_lens),
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            causal=True,
            mm_req_doc_ranges=mm_req_doc_ranges,
        ),
        replay_start=replay_start,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_builder_in_image_visibility():
    seq_lens = [30, 12]
    query_lens = [30, 12]
    spans = [[(4, 12), (20, 24)], []]
    builder = make_builder(vision=True)
    md = build_metadata(builder, seq_lens, query_lens, {0: spans[0], 1: spans[1]})
    assert md.num_prefills == 2
    assert md.num_decode_tokens == 0
    assert md.prefill_swa_indices.shape[-1] == WIDTH
    assert md.prefill_left_visible is not None
    assert md.prefill_right_visible is not None

    ref_left, ref_right = ref_left_right(seq_lens, query_lens, spans, MAX_IMG)
    assert md.prefill_left_visible.cpu().tolist() == ref_left
    assert md.prefill_right_visible.cpu().tolist() == ref_right

    _, _, _, _, block_table = make_batch(seq_lens, query_lens, torch.device("cuda"))
    rows, lens = ref_swa_slot_rows(
        seq_lens, query_lens, spans, block_table, WINDOW, MAX_IMG, WIDTH
    )
    assert md.prefill_swa_lens.cpu().tolist() == lens
    assert md.prefill_swa_indices[:, 0].cpu().tolist() == rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_builder_no_image_spans_fast_path():
    """Vision model, image-free batch: no visibility tensors, plain window."""
    seq_lens = [30, 12]
    query_lens = [30, 12]
    builder = make_builder(vision=True)
    md = build_metadata(builder, seq_lens, query_lens, {0: [], 1: []})
    assert md.prefill_left_visible is None
    assert md.prefill_right_visible is None

    _, _, _, _, block_table = make_batch(seq_lens, query_lens, torch.device("cuda"))
    rows, lens = ref_swa_slot_rows(
        seq_lens, query_lens, [[], []], block_table, WINDOW, MAX_IMG, WIDTH
    )
    assert md.prefill_swa_lens.cpu().tolist() == lens
    assert md.prefill_swa_indices[:, 0].cpu().tolist() == rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_builder_text_model_unchanged():
    """Text-only model: buffers stay window-sized and spans are ignored."""
    seq_lens = [30, 12]
    query_lens = [30, 12]
    builder = make_builder(vision=False)
    assert builder.max_image_tokens == 0
    md = build_metadata(builder, seq_lens, query_lens, None)
    assert md.prefill_swa_indices.shape[-1] == WINDOW
    assert md.prefill_left_visible is None

    _, _, _, _, block_table = make_batch(seq_lens, query_lens, torch.device("cuda"))
    rows, lens = ref_swa_slot_rows(
        seq_lens, query_lens, [[], []], block_table, WINDOW, MAX_IMG, WINDOW
    )
    assert md.prefill_swa_lens.cpu().tolist() == lens
    assert md.prefill_swa_indices[:, 0].cpu().tolist() == rows


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compress_ratio", [0, 1, 2])
@pytest.mark.parametrize("query_len", [100, 4000])
def test_v41_image_prefill_uses_causal_swa(compress_ratio, query_len):
    """V4.1 image tokens use the same causal SWA as text, including in chunks."""
    from vllm.models.deepseek_v41.common.ops.cache_utils import (
        combine_topk_swa_indices as combine_v41,
    )
    from vllm.models.deepseek_v41.sparse_mla import DeepseekV41SparseSWAMetadataBuilder

    seq_len, window, max_image_tokens = 4000, 128, 2048
    spans = [(1900, 3947)]
    builder = make_builder(
        True,
        window,
        max_image_tokens,
        query_len,
        builder_cls=DeepseekV41SparseSWAMetadataBuilder,
        deepseek_v4_config=False,
    )
    assert builder.max_image_tokens == 0
    md = build_metadata(builder, [seq_len], [query_len], {0: spans})
    assert md.prefill_left_visible is None
    assert md.prefill_right_visible is None
    assert md.prefill_swa_indices.shape[-1] == window
    plan = md.get_prefill_chunk_plan(
        compress_ratio, prefill_chunk_size=4, has_compressed=compress_ratio > 0
    )
    [(start, end, n, m)] = plan
    gather_len = int(md.prefill_gather_lens[0])
    assert (start, end, m - n) == (0, 1, gather_len)
    assert gather_len == query_len + min(seq_len - query_len, window - 1)
    indices, lens = combine_v41(
        torch.empty(query_len, 0, dtype=torch.int32, device="cuda"),
        md.query_start_loc,
        md.prefill_seq_lens,
        md.prefill_gather_lens,
        window,
        compress_ratio,
        0,
        m,
        n,
    )
    indices, lens = indices.cpu(), lens.cpu().tolist()
    paged_indices = md.prefill_swa_indices.cpu()
    paged_lens = md.prefill_swa_lens.cpu().tolist()
    for token, pos in enumerate(range(seq_len - query_len, seq_len)):
        lo, hi = max(0, pos - window + 1), pos + 1
        row = indices[token, : lens[token]]
        assert lens[token] == hi - lo
        assert torch.all((row >= n) & (row < m))
        absolute_keys = row - n + seq_len - gather_len
        assert absolute_keys.tolist() == list(range(lo, hi))
        assert paged_lens[token] == hi - lo
        assert paged_indices[token, 0, : hi - lo].tolist() == list(range(lo, hi))


# SWA bounded replay: every prefill index path clamps the window at replay_start.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_builder_replay_start_bounds_prefill_window_and_gather():
    """Paged-direct prefill indices stop at replay_start and the FlashMLA
    gather only covers the context the request may see."""
    seq_lens = [40, 12, 30]
    query_lens = [24, 12, 10]
    replay_starts = [12, 0, 0]
    builder = make_builder(vision=False)
    md = build_metadata(builder, seq_lens, query_lens, None, replay_starts)
    assert md.replay_start is not None
    assert md.replay_start.cpu().tolist() == replay_starts

    _, _, _, _, block_table = make_batch(seq_lens, query_lens, torch.device("cuda"))
    rows, lens = ref_swa_slot_rows(
        seq_lens,
        query_lens,
        [[], [], []],
        block_table,
        WINDOW,
        MAX_IMG,
        WINDOW,
        replay_starts=replay_starts,
    )
    assert md.prefill_swa_lens.cpu().tolist() == lens
    assert md.prefill_swa_indices[:, 0].cpu().tolist() == rows
    # Request 0: the first replayed row (pos 16) sees [12, 16]; its last row
    # (pos 39) is a full window above the replay start and sees all of it.
    assert md.prefill_swa_lens[0].item() == 16 - 12 + 1
    assert md.prefill_swa_lens[23].item() == WINDOW
    # gather_len = query_len + min(prefix_len - replay_start, WINDOW - 1).
    assert md.prefill_gather_lens.cpu().tolist() == [
        24 + min(16 - 12, WINDOW - 1),
        12,
        10 + min(20, WINDOW - 1),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_builder_replaying_group_builds_without_replay_start():
    """Graph captures build without model-specific kwargs: a replaying group's
    metadata then bounds nothing."""
    builder = make_builder(vision=False, bounded_replay=True)
    md = build_metadata(builder, [40, 12], [24, 12], None)
    assert not md.replay_start.any()
    assert md.prefill_swa_lens[0].item() == WINDOW


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v41_flashinfer_mixed_sparse_indices_respect_replay_start():
    from vllm.models.deepseek_v41.common.ops.cache_utils import (
        build_flashinfer_mixed_sparse_indices as build_v41,
    )

    device = torch.device("cuda")
    # 1 decode token (req 0) + two prefill requests; req 1 resumes at 16.
    seq_lens = [20, 40, 9]
    query_lens = [1, 24, 9]
    replay_starts = [0, 16, 0]
    query_start_loc = torch.tensor([0, 1, 25, 34], dtype=torch.int32, device=device)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    token_to_req = torch.tensor(
        [0] + [1] * 24 + [2] * 9, dtype=torch.int32, device=device
    )
    block_table = torch.arange(3, dtype=torch.int32, device=device).view(3, 1)
    kwargs = dict(
        decode_swa_indices=torch.zeros((1, WINDOW), dtype=torch.int32, device=device),
        decode_compressed_indices=None,
        decode_compressed_topk_lens=None,
        prefill_topk_indices=torch.empty((33, 0), dtype=torch.int32, device=device),
        query_start_loc=query_start_loc,
        seq_lens=seq_lens_t,
        token_to_req_indices=token_to_req,
        swa_block_table=block_table,
        swa_block_size=BLOCK_SIZE,
        compressed_block_table=None,
        compressed_block_size=BLOCK_SIZE,
        window_size=WINDOW,
        compress_ratio=1,
        topk=0,
    )
    plain, _ = build_v41(
        replay_start=torch.zeros(3, dtype=torch.int32, device=device), **kwargs
    )
    bounded, _ = build_v41(
        replay_start=torch.tensor(replay_starts, dtype=torch.int32, device=device),
        **kwargs,
    )
    rows, _ = ref_swa_slot_rows(
        seq_lens,
        query_lens,
        [[], [], []],
        block_table,
        WINDOW,
        MAX_IMG,
        WINDOW,
        replay_starts=replay_starts,
    )
    # Decode row is copied through; prefill rows follow the bounded window.
    assert bounded[0].cpu().tolist() == plain[0].cpu().tolist()
    assert bounded[1:].cpu().tolist() == rows[1:]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dspark_noncausal_draft_decode_update_matches_build():
    """A DSpark draft graph refreshes its non-causal SWA metadata in place
    instead of rebuilding it, so the refresh must match a fresh build."""
    from vllm.v1.attention.backends.mla.compressor_utils import (
        get_dspark_swa_index_width,
    )

    num_spec = 4
    query_lens = [num_spec] * 3

    def make_dspark_builder():
        builder = make_builder(vision=False)
        builder.is_dspark = True
        builder.noncausal_index_width = get_dspark_swa_index_width(WINDOW, num_spec)
        builder.decode_threshold = num_spec
        return builder

    def common_metadata(seq_lens):
        query_start_loc, seq_lens_t, _, slot_mapping, block_table = make_batch(
            seq_lens, query_lens, torch.device("cuda")
        )
        return CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc.cpu(),
            seq_lens=seq_lens_t,
            seq_lens_cpu_upper_bound=seq_lens_t.cpu(),
            num_reqs=len(seq_lens),
            num_actual_tokens=int(query_start_loc[-1]),
            max_query_len=num_spec,
            max_seq_len=max(seq_lens),
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            causal=False,
        )

    captured_seq_lens = [6, 40, 130]
    replay_seq_lens = [130, 9, 90]
    builder = make_dspark_builder()
    captured = common_metadata(captured_seq_lens)
    md = builder.build(0, captured)
    replay = common_metadata(replay_seq_lens)
    expected = make_dspark_builder().build(0, replay)

    # A replay sees new inputs through the same device buffers.
    captured.seq_lens.copy_(replay.seq_lens)
    captured.block_table_tensor.copy_(replay.block_table_tensor)
    builder.update_draft_decode_metadata(md)

    assert md.decode_swa_width == builder.noncausal_index_width
    torch.testing.assert_close(md.decode_swa_lens, expected.decode_swa_lens)
    torch.testing.assert_close(md.decode_swa_indices, expected.decode_swa_indices)


def _ds41_swa_bi_config(lengths):
    from types import SimpleNamespace

    from vllm.config import CacheConfig, set_current_vllm_config
    from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache

    window, block_size, dim = 128, 32, 512
    max_tokens = sum(lengths)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(sliding_window=window, compress_ratios=[0]),
            max_model_len=max(lengths),
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_tokens, max_num_seqs=3
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        speculative_config=None,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    cache_config = CacheConfig(cache_dtype="nvfp4_ds_mla")
    with set_current_vllm_config(config):
        swa = DeepseekV4SWACache(
            head_dim=dim,
            window_size=window,
            dtype=torch.uint8,
            prefix="swa",
            cache_config=cache_config,
            backend_cls=DeepseekV4FlashMLAAttention.swa_backend_cls,
            block_size=block_size,
            packed_bytes_per_token=528,
            packed_page_alignment=512,
        )
    return config, swa


def _ds41_cr2_bi_builder(config, device):
    from vllm.models.deepseek_v41.sparse_mla import (
        DeepseekV4FlashMLAMetadataBuilder,
    )
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        tokens_per_state=2,
        state_content_bytes=288,
        cache_dtype_str="nvfp4_ds_mla",
        model_version="deepseek_v4",
    )
    return DeepseekV4FlashMLAMetadataBuilder(spec, ["compressed"], config, device)


def test_ds41_flashmla_cr2_bi_builder_constructs_on_cpu(monkeypatch):
    """Construct the same CR2 builder before requesting a GPU arithmetic proof."""
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    config, _ = _ds41_swa_bi_config([12290, 65, 129])
    config.model_config.hf_config.compress_ratios = [0, 0, 2, 2]
    config.model_config.hf_config.index_topk = 512
    builder = _ds41_cr2_bi_builder(config, torch.device("cpu"))
    assert builder.compress_ratio == 2 and builder.topk_tokens == 512
    assert builder.req_id_per_token_buffer.numel() == 12484
    assert builder.compressed_slot_mapping_buffer.device.type == "cpu"


def test_ds41_flashmla_swa_bi_builder_policy(monkeypatch):
    """CPU construction must retain BI decode routing and disable engine graphs."""
    from vllm.models.deepseek_v41.nvidia.flashmla import (
        DeepseekSparseSWAFlashMLAMetadataBuilder,
    )
    from vllm.v1.attention.backend import AttentionCGSupport

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    config, swa = _ds41_swa_bi_config([12290, 65, 129])
    spec = swa.get_kv_cache_spec(config)
    builder = DeepseekSparseSWAFlashMLAMetadataBuilder(
        spec, [swa.prefix], config, torch.device("cpu")
    )
    assert builder.decode_threshold == 0
    assert builder.get_cudagraph_support(config, spec) == AttentionCGSupport.NEVER


@pytest.mark.parametrize("major,supported", [(9, False), (10, True), (12, False)])
def test_ds41_swa_bi_declaration_stays_scoped(monkeypatch, major, supported):
    """The tested SM100 route must not grant BI to DS4 or FlashInfer."""
    from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
        DeepseekSparseSWAFlashInferBackend,
    )
    from vllm.models.deepseek_v41.nvidia.flashmla import (
        DeepseekSparseSWAFlashMLABackend,
    )
    from vllm.platforms import current_platform
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend

    monkeypatch.setattr(
        type(current_platform),
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(major, 0)),
    )
    assert DeepseekSparseSWAFlashMLABackend.supports_batch_invariance() == supported
    assert not DeepseekSparseSWABackend.supports_batch_invariance()
    assert not DeepseekSparseSWAFlashInferBackend.supports_batch_invariance()


@pytest.mark.parametrize("major,supported", [(9, False), (10, True), (12, False)])
def test_ds41_cr2_indexer_bi_declaration_stays_scoped(monkeypatch, major, supported):
    """The CR2 proof must not grant BI to DS4, mega attention or sparse indexer."""
    from vllm.models.deepseek_v41.sparse_mla import (
        DeepseekV4FlashMLABackend,
        FlashMLAMegaAttnBackend,
    )
    from vllm.platforms import current_platform
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV4IndexerBackend,
        DeepseekV41IndexerBackend,
    )
    from vllm.v1.attention.backends.mla.sparse_indexer import (
        DeepseekV41SparseIndexerBackend,
    )

    monkeypatch.setattr(
        type(current_platform),
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(major, 0)),
    )
    assert DeepseekV4FlashMLABackend.supports_batch_invariance() == supported
    assert DeepseekV41IndexerBackend.supports_batch_invariance() == supported
    assert not DeepseekV4IndexerBackend.supports_batch_invariance()
    assert not FlashMLAMegaAttnBackend.supports_batch_invariance()
    assert not DeepseekV41SparseIndexerBackend.supports_batch_invariance()


@pytest.mark.parametrize("length", [127, 129, 12290])
def test_ds41_flashmla_swa_bi_batch_chunk_pd_graph(monkeypatch, length):
    _assert_ds41_flashmla_bi_schedule(monkeypatch, length, compress_ratio=0)


@pytest.mark.parametrize("length", [129, 12290])
def test_ds41_flashmla_cr2_indexer_bi_batch_chunk_pd_graph(monkeypatch, length):
    """Actual BF16 DeepGEMM -> stable Top-K -> CR2 attention is schedule invariant."""
    _assert_ds41_flashmla_bi_schedule(monkeypatch, length, compress_ratio=2)


def _assert_ds41_flashmla_bi_schedule(monkeypatch, length, compress_ratio):
    """Scheduling cannot change actual DS41 SWA attention output bits.

    Cover window edges, all query rows across 12288, request permutation,
    chunked prefill, single-token decode and static-metadata graph replay.
    Full engine graphs remain disabled by the production BI builder.
    """
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.models.deepseek_v41.common.ops.cache_utils import (
        quantize_and_insert_k_cache,
    )
    from vllm.models.deepseek_v41.nvidia.flashmla import (
        DeepseekSparseSWAFlashMLAMetadataBuilder,
        DeepseekV4FlashMLAAttention,
    )
    from vllm.v1.attention.backend import AttentionCGSupport
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        reset_workspace_manager,
    )

    if not torch.cuda.is_available():
        pytest.skip("DS41 FlashMLA requires CUDA")
    assert torch.cuda.get_device_capability()[0] == 10
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    torch.manual_seed(1949)
    device = torch.device("cuda")
    window, block_size, heads, dim = 128, 32, 64, 512
    lengths = [length, 65, 129]
    max_tokens = sum(lengths)
    config, swa = _ds41_swa_bi_config(lengths)
    config.model_config.hf_config.compress_ratios = (
        [0, 0, 2, 2] if compress_ratio else [0]
    )
    config.model_config.hf_config.index_topk = 512
    spec = swa.get_kv_cache_spec(config)
    builder = DeepseekSparseSWAFlashMLAMetadataBuilder(
        spec, [swa.prefix], config, device
    )
    assert builder.decode_threshold == 0
    assert builder.get_cudagraph_support(config, spec) == AttentionCGSupport.NEVER

    pages = (max(lengths) + block_size - 1) // block_size
    tables = torch.arange(3 * pages, dtype=torch.int32, device=device).view(3, pages)
    packed = torch.zeros(3 * pages, block_size * 528, device=device, dtype=torch.uint8)
    swa.kv_cache = packed.view(3 * pages, block_size, 528)
    queries = []
    for request, size in enumerate(lengths):
        queries.append(
            torch.randn(size, heads, dim, device=device, dtype=torch.bfloat16)
        )
        values = torch.randn(size, dim, device=device, dtype=torch.bfloat16)
        slots = torch.arange(size, device=device, dtype=torch.int64)
        slots += request * pages * block_size
        quantize_and_insert_k_cache(
            values, packed, slots, block_size=block_size, bytes_per_token=528
        )

    # Exercise the production attention subpath without allocating projections,
    # experts or embedding tables unrelated to its scheduling contract.
    attention = DeepseekV4FlashMLAAttention.__new__(DeepseekV4FlashMLAAttention)
    torch.nn.Module.__init__(attention)
    attention.compress_ratio = compress_ratio
    attention.compressed_cache_prefix = "compressed" if compress_ratio else None
    attention.swa_cache_layer = swa
    attention.window_size = window
    attention.max_num_batched_tokens = max_tokens
    attention.topk_indices_buffer = torch.full(
        (max_tokens, 512), -1, device=device, dtype=torch.int32
    )
    attention.scale = dim**-0.5
    attention.attn_sink = torch.randn(heads, device=device, dtype=torch.float32)
    compressed_builder = None
    if compress_ratio:
        from vllm.model_executor.kernels.batch_invariant import top_k_per_row_prefill
        from vllm.model_executor.layers.sparse_attn_indexer import as_fp32_logits
        from vllm.utils.deep_gemm import fp8_fp4_mqa_logits, mqa_logits_mx_inputs

        compressed_pages = (max(lengths) + 127) // 128
        compressed_tables = torch.arange(
            3 * compressed_pages, device=device, dtype=torch.int32
        ).view(3, compressed_pages)
        compressed = torch.empty(
            3 * compressed_pages, 64 * 288, device=device, dtype=torch.uint8
        )
        compressed[:, : 64 * 256] = torch.randint(
            256, (3 * compressed_pages, 64 * 256), device=device, dtype=torch.uint8
        )
        scales = (torch.rand(3 * compressed_pages, 64 * 32, device=device) + 0.25).to(
            torch.float8_e4m3fn
        )
        compressed[:, 64 * 256 :] = scales.view(torch.uint8)
        attention.is_kv_source = True
        attention.kv_cache = compressed.view(-1, 64, 288)
        attention._static_forward_context = {"compressed": attention}
        compressed_builder = _ds41_cr2_bi_builder(config, device)
        key_lengths = [(size + 1) // 2 for size in lengths]
        key_offsets = [0, key_lengths[0], sum(key_lengths[:2])]
        index_keys = torch.randint(
            -128, 128, (sum(key_lengths), 64), device=device, dtype=torch.int8
        )
        index_key_scales = torch.full(
            (sum(key_lengths),), 0x7F7F7F7F, device=device, dtype=torch.int32
        )
        index_queries = [
            torch.randint(-128, 128, (size, 32, 64), device=device, dtype=torch.int8)
            for size in lengths
        ]
        index_weights = [torch.randn(size, 32, device=device) for size in lengths]
        for weights in index_weights:
            weights[::97].zero_()
        if length > 12288:
            index_weights[0][12288].zero_()
    init_workspace_manager(device)

    def prepare(schedule):
        sizes = [end - start for _, start, end in schedule]
        ends = [end for _, _, end in schedule]
        starts_cpu = torch.tensor([0, *sizes], dtype=torch.int32).cumsum(0).int()
        starts = starts_cpu.to(device)
        q = torch.cat([queries[r][start:end] for r, start, end in schedule])
        positions = torch.cat(
            [torch.arange(start, end, device=device) for _, start, end in schedule]
        )
        common = CommonAttentionMetadata(
            query_start_loc=starts,
            query_start_loc_cpu=starts_cpu,
            seq_lens=torch.tensor(ends, device=device, dtype=torch.int32),
            seq_lens_cpu_upper_bound=torch.tensor(ends, dtype=torch.int32),
            num_reqs=len(schedule),
            num_actual_tokens=sum(sizes),
            max_query_len=max(sizes),
            max_seq_len=max(ends),
            block_table_tensor=torch.stack([tables[r] for r, _, _ in schedule]),
            slot_mapping=torch.arange(sum(sizes), device=device, dtype=torch.int64),
            causal=True,
        )
        metadata = builder.build(0, common)
        assert metadata.num_decodes == 0 and metadata.num_prefills == len(schedule)
        assert metadata.num_prefill_tokens == sum(sizes)
        all_metadata = {swa.prefix: metadata}
        if compressed_builder is not None:
            from dataclasses import replace

            compressed_common = replace(
                common,
                block_table_tensor=torch.stack(
                    [compressed_tables[r] for r, _, _ in schedule]
                ),
            )
            all_metadata["compressed"] = compressed_builder.build(0, compressed_common)
            index_q = torch.cat(
                [index_queries[r][start:end] for r, start, end in schedule]
            )
            weights = torch.cat(
                [index_weights[r][start:end] for r, start, end in schedule]
            )
            row_starts = torch.cat(
                [
                    torch.full(
                        (end - start,), key_offsets[r], device=device, dtype=torch.int32
                    )
                    for r, start, end in schedule
                ]
            )
            row_ends = row_starts + (positions.int() + 1) // 2
            valid_ends = row_ends - row_starts
            query_scales, native_weights = mqa_logits_mx_inputs(index_q, None, weights)
        context = ForwardContext({}, all_metadata, {})
        output = torch.empty_like(q)

        def forward():
            if compressed_builder is not None:
                scores = fp8_fp4_mqa_logits(
                    (index_q, query_scales),
                    (index_keys, index_key_scales),
                    native_weights,
                    row_starts,
                    row_ends,
                    max(key_lengths),
                )
                assert scores.dtype == torch.bfloat16
                logits = as_fp32_logits(scores, valid_ends)
                top_k_per_row_prefill(
                    logits,
                    torch.zeros_like(valid_ends),
                    valid_ends,
                    attention.topk_indices_buffer[: q.shape[0]],
                    q.shape[0],
                    logits.stride(0),
                    logits.stride(1),
                    512,
                )
            with override_forward_context(context):
                attention.forward_mqa(q, q[:, 0], positions, output)

        return forward, output, metadata, q

    def run(schedule):
        forward, output, _, _ = prepare(schedule)
        forward()
        return output.clone()

    try:
        expected = [run([(r, 0, size)]) for r, size in enumerate(lengths)]
        schedule = [(1, 0, lengths[1]), (0, 0, length), (2, 0, lengths[2])]
        for order in (schedule, schedule[::-1]):
            actual = run(order)
            assert torch.equal(actual, torch.cat([expected[r] for r, _, _ in order]))

        attention.PREFILL_CHUNK_SIZE = 1
        forward, output, metadata, _ = prepare(schedule)
        plan = metadata.get_prefill_chunk_plan(
            compress_ratio, 1, has_compressed=bool(compress_ratio)
        )
        # Small cases can fit one workspace chunk; the long case must split.
        if length > 12288:
            assert len(plan) > 1
        forward()
        wanted = torch.cat([expected[r] for r, _, _ in schedule])
        assert torch.equal(output, wanted)
        attention.PREFILL_CHUNK_SIZE = 4
        if compress_ratio:
            attention.is_kv_source = False
            assert torch.equal(run([(0, 0, length)]), expected[0])
            attention.is_kv_source = True

        cuts = sorted({0, 64, 127, 128, length - 2, length - 1, length})
        cuts = [cut for cut in cuts if 0 <= cut <= length]
        chunks = [run([(0, start, end)]) for start, end in zip(cuts, cuts[1:])]
        assert torch.equal(torch.cat(chunks), expected[0])
        for position in sorted({0, 126, 127, 128, length - 1}):
            if position < length:
                assert torch.equal(
                    run([(0, position, position + 1)]),
                    expected[0][position : position + 1],
                )

        forward, output, _, q = prepare(schedule)
        forward()
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        for multiplier in (1, -1):
            q.mul_(multiplier)
            forward()
            eager = output.clone()
            output.fill_(float("nan"))
            graph.replay()
            assert torch.equal(output, eager)
            assert torch.isfinite(output).all()
    finally:
        torch.accelerator.synchronize()
        reset_workspace_manager()


@pytest.mark.parametrize("bounded", [False, True])
def test_ds41_bi_nvfp4_gather_matches_paged_record_and_graph(monkeypatch, bounded):
    """BI gathers the same packed records, preserving offsets and page strides."""
    if not torch.cuda.is_available():
        pytest.skip("DS41 NVFP4 CUDA gather")
    from vllm.models.deepseek_v41.common.ops.cache_utils import (
        dequantize_and_gather_k_cache,
    )

    torch.manual_seed(1959)
    device = torch.device("cuda")
    block, requests, pages = 64, 3, 3
    page_stride = block * 288 + 512
    packed = torch.full(
        (requests * pages, page_stride), 165, dtype=torch.uint8, device=device
    )
    packed[:, : block * 256] = torch.randint(
        256, (requests * pages, block * 256), dtype=torch.uint8, device=device
    )
    scales = (torch.rand(requests * pages, block * 32, device=device) + 0.25).to(
        torch.float8_e4m3fn
    )
    packed[:, block * 256 : block * 288] = scales.view(torch.uint8)
    cache = packed.as_strided((requests * pages, block, 288), (page_stride, 288, 1))
    table = torch.randperm(requests * pages, device=device).int().view(requests, pages)
    seq_lens = torch.tensor([130, 127, 65], dtype=torch.int32, device=device)
    gather_lens = seq_lens.clamp(max=63) if bounded else None
    offset, max_gather = 17, 63 if bounded else 130
    shape = (requests, offset + max_gather + 3, 512)
    strides = (shape[1] * 512 + 64, 512, 1)

    def allocate(count=requests):
        return torch.empty_strided(
            (count, *shape[1:]), strides, dtype=torch.bfloat16, device=device
        ).fill_(-7.25)

    def gather(output, lengths=seq_lens, tables=table, limits=gather_lens):
        dequantize_and_gather_k_cache(
            output, cache, lengths, limits, tables, block, offset
        )

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    expected = allocate()
    gather(expected)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    actual = allocate()
    gather(actual)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    for request in range(requests):
        alone = allocate(1)
        gather(
            alone,
            seq_lens[request : request + 1],
            table[request : request + 1],
            None if gather_lens is None else gather_lens[request : request + 1],
        )
        assert torch.equal(alone, expected[request : request + 1])

    permutation = torch.tensor([2, 0, 1], device=device)
    shuffled = allocate()
    gather(
        shuffled,
        seq_lens[permutation],
        table[permutation],
        None if gather_lens is None else gather_lens[permutation],
    )
    assert torch.equal(shuffled, expected[permutation])
    gather(actual)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gather(actual)
    seq_lens.sub_(1)
    if gather_lens is not None:
        gather_lens.clamp_(max=62)
    expected.fill_(-7.25)
    gather(expected)
    actual.fill_(-7.25)
    graph.replay()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
