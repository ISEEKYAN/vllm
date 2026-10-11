# Batched DS4.1 batch invariance

Enable `VLLM_BATCH_INVARIANT=1` before importing vLLM. The MXFP8 provider launches one batched grid with ascending block-32 scaled FP32 accumulation; packed E8M0 scales are read directly. The mHC post/pre provider processes all token rows with two batched kernels, not one kernel per token. Ordinary math and native quantization remain unchanged. Generic linear BI supports SM89+; the DS4.1 o_proj and W4A8 serving path require SM100 and the existing DS4.1 provider.

```python
from vllm import LLM

llm = LLM(model="path/to/ds41", enforce_eager=False)
```

Evidence limits: the launch spy counts the mHC GEMM, not the second post/pre
kernel, and its hidden-width fixture is 512 rather than a production width.
The packed-scale launch spy uses all-ones contiguous inputs; separate native
oracle, strided-scale and graph tests support the stated numerical behavior.
No generated-PTX precision proof or malformed grouped-weight layout test is
claimed. These remain non-blocking review caveats.
