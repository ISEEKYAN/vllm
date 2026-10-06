# Paged MQA with an external batch-invariant DeepGEMM

An older external DeepGEMM can provide GEMM batch-invariance controls while
lacking the 128-entry paged-MQA kernels required by DeepSeek-V4.1 on SM100.
Use the wheel-bundled DeepGEMM for the dense paged metadata/kernel pair in
that deployment:

```bash
export VLLM_DEEP_GEMM_PAGED_MQA_USE_VENDORED=1
# Start the usual vLLM engine with its existing model and warmup configuration.
```

The flag defaults to zero. It leaves GEMM, contiguous MQA and BI control APIs
on the externally selected package. Only `get_paged_mqa_logits_metadata` and
`fp8_fp4_paged_mqa_logits` come from the bundled revision; a missing bundled
package/API is an error, with no fallback.

The opt-in `attention_config.indexer_sparse_logits=True` DS4.1 candidate
consumer instead uses `get_paged_sparse_mqa_logits_metadata` and
`fp8_fp4_paged_sparse_mqa_logits`. Those sparse APIs remain together on the
external package, including when this flag is set. The flag does not enable
sparse-logits support in an external package that lacks it.

With the default `indexer_sparse_logits=False`, DS4.1 uses the dense scorer
(and then candidate masking where applicable), so its paged decode consumes
the bundled dense pair. Dense and sparse metadata are never interchanged.
The vLLM wheel must include its pinned DeepGEMM and JIT headers.

Validate the SM100 128-entry varlen path with:

```bash
VLLM_DEEP_GEMM_PAGED_MQA_USE_VENDORED=1 .venv/bin/python -m pytest \
  tests/kernels/attention/test_deepgemm_attention.py \
  -k test_vendored_paged_mqa_128_varlen
```

This option does not establish training/rollout numerical equality; validate
that property independently with the model and batch-invariance recipe.
