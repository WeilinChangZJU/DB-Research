# A-MEM wrapper and offline M0–M3 receipt

Date: 2026-11-14 (Asia/Shanghai)

The A-MEM research layer is now available in `bemula/research/amem.py`.
It wraps an existing Bemula `MemoryAdapter` and adds source-level IDs,
source-to-note mapping, serialisable note snapshots, wrapper decision events,
best-effort retrieval evidence, and forward/reverse dependency introspection.

The wrapper records `ingest`, `candidate`, `write`, `update`, `noop`, `read`,
and `select` events. Candidate positions are captured at the A-MEM retrieval
boundary and mapped back to note IDs. The trace is labelled `wrapper_delta`.
The stock adapter still has no source-level forget/delete API and does not
expose its parsed LLM branch decision; those capabilities are reported as
unsupported rather than inferred.
Stock note UUIDs remain physical snapshot IDs, so cross-run semantic equality
must use the Q01 canonical state normalizer.

## Offline verification

No DeepSeek endpoint, API key, full dataset, official A-MEM model, or paid
embedding request was used.

```text
WSL: .venv/bin/python -m pytest -q tests
30 passed in 0.15s

WSL: .venv/bin/python -m py_compile bemula/research/*.py \
  bemula/research/datasets/*.py scripts/run_research_eval.py \
  scripts/run_q01_fixture.py scripts/run_amem_fixture.py
PASS
```

The single-history fixture uses three deterministic source events and removes
`src-a`. Its result is saved at:

`C:\Users\Weilin\Desktop\AgentMemory\experiment_results_20261114\amem_fixture\summary.json`

SHA-256: `247f29c33a1390b4eeab11d19cef848197fbf7b9b31b72dc987a8bf3a06e2d63`

The fixture reports M0, M2, and M3 exact to the M0 rebuilt state. M1 is
intentionally non-exact for this shared-support history, which exercises the
weak tombstone/filter boundary. The result is an engineering contract check,
not a model-quality or paper-reproduction result.
