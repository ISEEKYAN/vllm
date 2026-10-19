# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C3 batch-invariance check for DeepSeek-V4.1 on vLLM with VLLM_BATCH_INVARIANT=1.

Usage: python c3.py <model_dir> <graph|graph_cache|eager|mixed> <out_dir>
Then:  python c3_compare.py <out_dir>   (prints C3_ALL_ZERO_DIFF on success)

Runs five fixed random prompts (5/131/700/2113/4900 tokens) through one engine
configuration in several scheduling scenarios (single, batch, reversed batch,
prefill-vs-decode, prefix-cache hit) and records, per generated position, the token,
its raw logprob (float.hex) and the top-5; plus a full-vocab logprob sha256 family
and a seeded-sampling family. c3_compare.py requires every scenario of every
configuration to be bitwise equal to graph's single-request reference.

Defaults follow the release recipe (sparse FlashMLA attention, MegaMoE with expert
parallelism, compilation_config unset). Overrides: C3_ATTN_BACKEND, C3_MOE_BACKEND,
C3_COMPILE=mode0, C3_HF_OVERRIDES, C3_ENGRAM_CONFIG. The VLLM_BI_* variables are read
from the environment.
"""

import hashlib
import json
import os
import random
import sys
import time

import numpy as np

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")

MODEL, CFG, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)

PROMPT_LENS = [5, 131, 700, 2113, 4900]
GEN = 24
TOPK = 5
FULL_GEN = 8
MAX_LEN = 8192

CONFIGS = {
    # FULL_AND_PIECEWISE graphs, prefix cache on (cache replay scenario).
    "graph_cache": dict(graph=True, prefix=True, mnbt=8192),
    # Same graphs, prefix cache off.
    "graph": dict(graph=True, prefix=False, mnbt=8192),
    # Eager reference for graph replay.
    "eager": dict(graph=False, prefix=False, mnbt=8192),
    # Small token budget: prefill chunks share steps with other requests' decodes.
    "mixed": dict(graph=True, prefix=False, mnbt=512),
}
cfg = CONFIGS[CFG]


def make_prompts():
    rng = random.Random(20261018)
    return [[rng.randrange(1000, 120000) for _ in range(n)] for n in PROMPT_LENS]


from vllm import LLM, SamplingParams  # noqa: E402

MOE_BACKEND = os.environ.get("C3_MOE_BACKEND", "deep_gemm_mega_moe")
# Unset compilation_config resolves to mode NONE with breakable CUDA graphs.
COMPILE = os.environ.get("C3_COMPILE", "default")
if COMPILE == "default":
    compilation_config = None
elif cfg["graph"]:
    compilation_config = {
        "mode": 0,
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "max_cudagraph_capture_size": 16,
    }
else:
    compilation_config = {"mode": 0, "cudagraph_mode": "NONE"}

llm = LLM(
    model=MODEL,
    trust_remote_code=True,
    dtype="bfloat16",
    seed=42,
    tensor_parallel_size=1,
    data_parallel_size=1,
    # MegaMoE requires expert parallelism (EP1 on one GPU).
    enable_expert_parallel=MOE_BACKEND == "deep_gemm_mega_moe",
    kv_cache_dtype="nvfp4_ds_mla",
    attention_backend=os.environ.get("C3_ATTN_BACKEND", "FLASHMLA_SPARSE_DSV41"),
    attention_config={"indexer_kv_dtype": "mxfp4", "indexer_sparse_logits": False},
    # The wo_a contract (wo_a_dtype) comes from the model config.
    hf_overrides=json.loads(
        os.environ.get("C3_HF_OVERRIDES", '{"head_dtype": "float32"}')
    ),
    moe_backend=MOE_BACKEND,
    compilation_config=compilation_config,
    enforce_eager=not cfg["graph"],
    engram_config=json.loads(
        os.environ.get(
            "C3_ENGRAM_CONFIG",
            '{"cpu_offload": true, "embedding_across_dp": true, '
            '"dp_shared_memory": false, "host_table_layout": "per_rank"}',
        )
    ),
    enable_prefix_caching=cfg["prefix"],
    enable_chunked_prefill=True,
    max_model_len=MAX_LEN,
    max_num_batched_tokens=cfg["mnbt"],
    max_num_seqs=8,
    max_logprobs=-1,
    logprobs_mode="raw_logprobs",
    gpu_memory_utilization=0.85,
    disable_custom_all_reduce=True,
    limit_mm_per_prompt={"image": 0},
)


def records(outs):
    res = []
    for o in outs:
        c = o.outputs[0]
        steps = []
        for tok, lp in zip(c.token_ids, c.logprobs):
            steps.append(
                {
                    "tok": int(tok),
                    "lp": float(lp[tok].logprob).hex(),
                    "top": sorted(
                        ((int(k), float(v.logprob).hex()) for k, v in lp.items()),
                        key=lambda kv: kv[0],
                    ),
                }
            )
        res.append(steps)
    return res


def gen(prompts, n=GEN):
    sp = SamplingParams(max_tokens=n, temperature=0.0, logprobs=TOPK, ignore_eos=True)
    return records(
        llm.generate([{"prompt_token_ids": p} for p in prompts], sp, use_tqdm=False)
    )


def full_records(outs):
    res = []
    for o in outs:
        c = o.outputs[0]
        steps = []
        for tok, lp in zip(c.token_ids, c.logprobs):
            vec = np.full(len(lp), np.nan, dtype=np.float32)
            ids = np.fromiter(lp.keys(), dtype=np.int64, count=len(lp))
            vals = np.fromiter((v.logprob for v in lp.values()), np.float32, len(lp))
            assert ids.max() < len(lp)
            vec[ids] = vals
            steps.append(
                {
                    "tok": int(tok),
                    "lp": float(lp[tok].logprob).hex(),
                    "top": [[len(lp), hashlib.sha256(vec.tobytes()).hexdigest()]],
                    "nonfinite": int((~np.isfinite(vec)).sum()),
                }
            )
        res.append(steps)
    return res


def gen_full(prompts):
    sp = SamplingParams(
        max_tokens=FULL_GEN,
        temperature=0.0,
        logprobs=-1,
        ignore_eos=True,
        detokenize=False,
    )
    return full_records(
        llm.generate([{"prompt_token_ids": p} for p in prompts], sp, use_tqdm=False)
    )


def gen_sample(prompts, seeds):
    sps = [
        SamplingParams(
            max_tokens=GEN,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            seed=s,
            logprobs=TOPK,
            ignore_eos=True,
        )
        for s in seeds
    ]
    return records(
        llm.generate([{"prompt_token_ids": p} for p in prompts], sps, use_tqdm=False)
    )


prompts = make_prompts()
seeds = [7001 + i for i in range(len(prompts))]
cc = llm.llm_engine.vllm_config.compilation_config
result = {
    "config": CFG,
    "cfg": cfg,
    "prompt_lens": PROMPT_LENS,
    "scenarios": {},
    "resolved": {
        "compile": COMPILE,
        "mode": str(cc.mode),
        "cudagraph_mode": str(cc.cudagraph_mode),
        "max_cudagraph_capture_size": cc.max_cudagraph_capture_size,
        "attention_backend": os.environ.get("C3_ATTN_BACKEND", "FLASHMLA_SPARSE_DSV41"),
        "moe_backend": MOE_BACKEND,
    },
}
print("RESOLVED", json.dumps(result["resolved"]), flush=True)
S = result["scenarios"]
t0 = time.time()

S["single"] = [gen([p])[0] for p in prompts]
S["batch"] = gen(prompts)
S["batch_rev"] = gen(prompts[::-1])[::-1]
S["full_single"] = [gen_full([p])[0] for p in prompts]
S["full_batch"] = gen_full(prompts)
S["sample_single"] = [gen_sample([p], [sd])[0] for p, sd in zip(prompts, seeds)]
S["sample_batch"] = gen_sample(prompts, seeds)
S["sample_batch_rev"] = gen_sample(prompts[::-1], seeds[::-1])[::-1]
if cfg["prefix"]:
    # Second pass over identical prompts hits the prefix cache.
    S["cache_single"] = [gen([p])[0] for p in prompts]
    S["cache_batch"] = gen(prompts)

if CFG == "graph":
    # Prefill/decode consistency: teacher-force prompt + generated tokens and
    # read the prompt logprobs at the generated positions.
    pd = []
    for p, steps in zip(prompts, S["single"]):
        toks = [s["tok"] for s in steps]
        sp = SamplingParams(
            max_tokens=1, temperature=0.0, prompt_logprobs=TOPK, ignore_eos=True
        )
        o = llm.generate([{"prompt_token_ids": p + toks}], sp, use_tqdm=False)[0]
        rows = []
        for j, tok in enumerate(toks):
            lp = o.prompt_logprobs[len(p) + j]
            rows.append(
                {
                    "tok": tok,
                    "lp": float(lp[tok].logprob).hex(),
                    "top": sorted(
                        (
                            (int(k), float(v.logprob).hex())
                            for k, v in lp.items()
                            if v.rank is not None and v.rank <= TOPK
                        ),
                        key=lambda kv: kv[0],
                    ),
                }
            )
        pd.append(rows)
    S["pd_prefill"] = pd

result["elapsed_s"] = time.time() - t0
result["env"] = {
    k: os.environ.get(k)
    for k in (
        "VLLM_BATCH_INVARIANT",
        "VLLM_USE_BREAKABLE_CUDAGRAPH",
        "VLLM_BATCH_INVARIANT_KERNEL_LIB",
        "C3_HF_OVERRIDES",
        "C3_ENGRAM_CONFIG",
        "VLLM_BI_BF16_GEMM",
        "VLLM_BI_MHC_OVERLAP",
        "VLLM_BI_MXFP8_GEMM",
        "VLLM_BI_MEGA_MHC",
        "VLLM_BI_MEGA_GATE",
        "VLLM_BI_MEGA_ATTN",
        "VLLM_BI_MXFP8_PREQUANT",
        "VLLM_BI_REPLAY_STAGE_LAYERS",
        "VLLM_USE_BREAKABLE_CUDAGRAPH",
    )
}
with open(os.path.join(OUT, f"c3v2-{CFG}.json"), "w") as f:
    json.dump(result, f)
print("C3_DRIVER_DONE", CFG, result["elapsed_s"])
