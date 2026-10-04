# Paged MQA with an external batch-invariant DeepGEMM

An older external DeepGEMM can provide GEMM batch-invariance controls while
lacking the 128-entry paged-MQA kernels required by DeepSeek-V4.1 on SM100.
Use the wheel-bundled DeepGEMM for the metadata/kernel pair in that deployment:

```bash
export VLLM_DEEP_GEMM_PAGED_MQA_USE_VENDORED=1
# Start the usual vLLM engine with its existing model and warmup configuration.
```

The flag defaults to zero. It leaves GEMM, contiguous MQA and BI control APIs
on the externally selected package. Both paged-MQA functions come from the
bundled revision; a missing bundled package/API is an error, with no fallback.
The vLLM wheel must include its pinned DeepGEMM and JIT headers.

Validate the SM100 128-entry varlen path with:

```bash
VLLM_DEEP_GEMM_PAGED_MQA_USE_VENDORED=1 .venv/bin/python -m pytest \
  tests/kernels/attention/test_deepgemm_attention.py \
  -k test_vendored_paged_mqa_128_varlen
```

This option does not establish training/rollout numerical equality; validate
that property independently with the model and batch-invariance recipe.
