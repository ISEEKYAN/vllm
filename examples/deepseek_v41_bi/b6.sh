#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# B6 throughput: BI=0 (A) vs the batch-invariant release recipe (B) for DeepSeek-V4.1.
# Both servers stay up, one GPU each; benchmarks never overlap. Per concurrency c the
# order is A B B A (two rounds), 8K in / 1K out, ignore_eos, N_MULT*c requests per run,
# after an unmeasured warmup of c requests per server.
#
# Usage: MODEL_A=<dir> MODEL_B=<dir> OUT=<dir> bash b6.sh
#   MODEL_A  checkpoint for A (BI=0)
#   MODEL_B  checkpoint for B (config with text_config.wo_a_dtype="bf16")
#   CS       concurrencies (default "8 64"); N_MULT requests per c (default 4)
#   GPU_A / GPU_B (default 0 / 1), PORT_A / PORT_B (default 18000 / 18001)
set -euo pipefail
: "${MODEL_A:?}" "${MODEL_B:?}" "${OUT:?}"
read -r -a CS_LIST <<< "${CS:-8 64}"
N_MULT=${N_MULT:-4}
GPU_A=${GPU_A:-0}
GPU_B=${GPU_B:-1}
PORT_A=${PORT_A:-18000}
PORT_B=${PORT_B:-18001}
mkdir -p "$OUT"

SERVE_ARGS=(
  --trust-remote-code --dtype bfloat16 --seed 42 --tensor-parallel-size 1
  --kv-cache-dtype nvfp4_ds_mla --attention-backend FLASHMLA_SPARSE_DSV41
  --attention-config '{"indexer_kv_dtype":"mxfp4","indexer_sparse_logits":false}'
  --hf-overrides '{"head_dtype":"float32"}'
  --moe-backend deep_gemm_mega_moe --enable-expert-parallel
  --engram-config '{"cpu_offload":true,"embedding_across_dp":true,"dp_shared_memory":false,"host_table_layout":"per_rank"}'
  --no-enable-prefix-caching --enable-chunked-prefill --max-model-len 10240
  --max-num-batched-tokens 16384 --max-num-seqs 64 --gpu-memory-utilization 0.85
  --disable-custom-all-reduce --limit-mm-per-prompt '{"image":0}'
)
BI_ENV=(
  VLLM_BATCH_INVARIANT=1 VLLM_BI_MXFP8_GEMM=tensor_core VLLM_BI_BF16_GEMM=tensor_core
  VLLM_BI_MXFP8_PREQUANT=1 VLLM_BI_MHC_OVERLAP=1 VLLM_BI_REPLAY_STAGE_LAYERS=2
)

CUDA_VISIBLE_DEVICES=$GPU_A env VLLM_BATCH_INVARIANT=0 \
  vllm serve "$MODEL_A" --port "$PORT_A" "${SERVE_ARGS[@]}" > "$OUT/server-A.log" 2>&1 &
PID_A=$!
CUDA_VISIBLE_DEVICES=$GPU_B env "${BI_ENV[@]}" \
  vllm serve "$MODEL_B" --port "$PORT_B" "${SERVE_ARGS[@]}" > "$OUT/server-B.log" 2>&1 &
PID_B=$!
trap 'kill $PID_A $PID_B 2>/dev/null || true' EXIT

ready() {  # port
  for _ in $(seq 1 720); do
    curl -sf -m 3 "http://localhost:$1/health" > /dev/null && return 0
    sleep 5
  done
  echo "server on port $1 not ready" >&2
  return 1
}
ready "$PORT_A"
ready "$PORT_B"

bench() {  # port model c n tag
  vllm bench serve --backend vllm --base-url "http://localhost:$1" --model "$2" \
    --dataset-name random --random-input-len 8192 --random-output-len 1024 --ignore-eos \
    --max-concurrency "$3" --num-prompts "$4" --seed 2026 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
    --save-result --result-dir "$OUT" --result-filename "$5.json" > "$OUT/bench-$5.log" 2>&1
}
for C in "${CS_LIST[@]}"; do
  bench "$PORT_A" "$MODEL_A" "$C" "$C" "warmup-A-c$C"
  bench "$PORT_B" "$MODEL_B" "$C" "$C" "warmup-B-c$C"
done
for C in "${CS_LIST[@]}"; do
  N=$((N_MULT * C))
  bench "$PORT_A" "$MODEL_A" "$C" "$N" "A-r1-c$C"
  bench "$PORT_B" "$MODEL_B" "$C" "$N" "B-r1-c$C"
  bench "$PORT_B" "$MODEL_B" "$C" "$N" "B-r2-c$C"
  bench "$PORT_A" "$MODEL_A" "$C" "$N" "A-r2-c$C"
done

python3 - "$OUT" "${CS_LIST[@]}" <<'EOF'
import json
import sys

out, cs = sys.argv[1], sys.argv[2:]
print("| c | A r1 / r2 | B r1 / r2 | B/A (mean) |")
print("|---|---|---|---|")
for c in cs:
    tp = {
        k: json.load(open(f"{out}/{k}-c{c}.json"))["output_throughput"]
        for k in ("A-r1", "A-r2", "B-r1", "B-r2")
    }
    ratio = (tp["B-r1"] + tp["B-r2"]) / (tp["A-r1"] + tp["A-r2"])
    print(
        f"| {c} | {tp['A-r1']:.0f} / {tp['A-r2']:.0f} | "
        f"{tp['B-r1']:.0f} / {tp['B-r2']:.0f} | {ratio:.3f} |"
    )
EOF
