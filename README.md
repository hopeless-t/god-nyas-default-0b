# God Nyāzu Default 0B Asset

Official optional starter asset for **六親合体ゴッドにゃ〜ず / ROKUSHIN GATTAI GOD NYĀZU**.

## What “0B” means here

This is software, implemented in Python. **0B does not mean “not Python” or “zero bytes.”** It means the declared runtime path uses:

- **0 trainable model parameters**
- **0 language-model calls**

It is not an open-world language model and does not pretend to be one. Unsupported semantic requests return `UNRESOLVED`.

## What is actually included

The v0.2 public asset is not the old six-intent toy file. It includes working modules for:

- bounded command understanding + explicit abstention
- state-space planning
- reliability-weighted replanning after observed drift
- non-parametric case memory
- persistent text memory + deterministic retrieval
- arithmetic, linear equations, shortest path, subset sum, topological sort, CSP solving
- HTML / JSON / filesystem / process perception
- repeated-transition discovery
- deterministic worker routing and multi-worker cover routing
- independent verification and canonical digests
- combinatorial candidate generation and diversity selection
- allowlisted real-tool execution
- checkpointed/resumable workflows

All of those paths remain `trainable_parameters=0` and `model_calls=0`.

## Try it

```bash
python -m godnyas_default_0b selftest
python -m godnyas_default_0b chat "ベンチ回して"
python -m godnyas_default_0b arithmetic "(12 + 5) * 3"
python -m godnyas_default_0b capabilities
```

Python:

```python
from godnyas_default_0b import load_default_0b

model = load_default_0b()
print(model.self_test())
print(model.respond("ベンチ回して"))
print(model.reason("arithmetic", {"expr": "(12 + 5) * 3"}))
```

## Boundary

This is the **usable public 0B starter asset**, not a dump of the author's private production internals. Private model names, private composition topology, and private optimization mechanisms are intentionally absent.

## Distribution

This asset has its own license. It may be downloaded and used while officially distributed, but may not be mirrored or redistributed without written permission.

If official distribution ends, already lawfully obtained copies may continue to be used under the asset license, while new official acquisition ends.
