# DS4.1 batch-invariant sparse attention order

Set `VLLM_BATCH_INVARIANT=1` before starting vLLM. CUDA graphs remain enabled:

```python
import os
os.environ["VLLM_BATCH_INVARIANT"] = "1"
from vllm import LLM
llm = LLM(model="YOUR_DS41_MODEL", enforce_eager=False)
```

The live eager indexer boundary sorts the selected KV positions in ascending
position order before attention reads them. This matches the native short-context
fill and the training deployment provider's reduction order. Selection is
unchanged; invalid slots remain at the end as `-1`. The shared buffer keeps its
address, and only live rows are modified on each replay. Dummy/profile forwards
without KV metadata retain the original path.

This fix addresses reduction order; it does not independently guarantee that
score computation, top-k membership or all other model operations are batch
invariant. The CPU boundary test covers BI/default, changing live batch sizes,
padded rows and missing metadata. End-to-end strict equality was verified with
the complete DS4.1 W4A8 training/serving fix bundle on two- and four-layer text
prefixes, including 4096-token output; the full 40-layer model is not evaluated.
