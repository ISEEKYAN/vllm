# Batched DS4.1 batch invariance

Enable `VLLM_BATCH_INVARIANT=1` before importing vLLM. The MXFP8 provider launches one batched grid with ascending block-32 scaled FP32 accumulation; packed E8M0 scales are read directly. The mHC post/pre provider processes all token rows in one launch. Ordinary math and native quantization remain unchanged. SM100 and the existing DS4.1 W4A8 provider are prerequisites.

```python
from vllm import LLM

llm = LLM(model="path/to/ds41", enforce_eager=False)
```
