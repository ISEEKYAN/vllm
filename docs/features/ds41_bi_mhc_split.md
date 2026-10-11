# Batch-invariant mHC entry reduction

Set `VLLM_BATCH_INVARIANT=1` before starting vLLM. Leave CUDA graphs enabled:

```python
import os
os.environ["VLLM_BATCH_INVARIANT"] = "1"
from vllm import LLM
llm = LLM(model="YOUR_DS41_MODEL", enforce_eager=False)
```

The shared unfused/delayed mHC entry selector uses a single-token batch in
BI mode for both warmup keys and live dispatch. Calls through that selector keep
the same split count on the same hardware. The small-batch fused post/pre and
the eager sibling are separate paths; this PR does not unify their reductions.
The count still depends on
input width and device SM count; it is not universally fixed to 16. Default mode
retains the existing batch-size heuristic.

The CPU regression mocks 152 SMs and checks 1, 8, 239, 384 and 4206 rows: BI uses
16 splits throughout, while default keeps 16/16/16/16/4. End-to-end equality was
verified with the complete DS4.1 W4A8 fix bundle on two- and four-layer text
prefixes, including 4096-token output. This boundary fix alone does not establish
batch invariance of the whole model; the full 40-layer model is not evaluated.
