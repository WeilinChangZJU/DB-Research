# Q01 platform implementation receipt (historical pre-audit snapshot)

> This receipt records the pre-repair state from the earlier implementation
> pass. The current repaired status is in
> `Q01_REPAIR_RECEIPT_20261114.md`; the older test counts and efficiency
> figures below must not be used as current validation results.

Date: 2026-11-14 (Asia/Shanghai)

The platform now contains the following research modules:

- canonical state and semantic A-MEM snapshot normalization;
- source-aware A-MEM wrapper with LLM prompt/response hashes, conservative
  branch labels, retriever/index summaries, query trace and dependency graph;
- durable semantic checkpoint save/load;
- M0-M3 canonical maintenance comparison;
- deterministic deletion scenario manifests;
- history-level failure classification and paired bootstrap intervals;
- lifecycle cost vector fields and recorder merge;
- optional OpenAI-compatible provider usage capture (prompt/completion tokens)
  without storing raw prompts or responses;
- ResearchProtocol control/forget/relearn phases;
- LoCoMo/LongMemEval-S source event overlay helper.
- reproducible offline dataset-adapter smoke runner with source/evidence-preserving
  deletion manifests and provenance manifest output.

## Offline validation

```text
WSL: .venv/bin/python -m pytest -q tests
46 passed in 0.19s

WSL: .venv/bin/python -m py_compile bemula/research/*.py \
  bemula/research/datasets/*.py scripts/*.py
PASS
```

The deterministic scenario runner produced 13 deletion scenarios over one
controlled history. M0/M2/M3 were exact on all 13 scenarios; M1 had zero exact
states on this shared-support fixture. M2/M3 replayed 38.46% fewer events than
M0 on average. This is a mechanism-screen result, not evidence from a real
workload.

The official robust A-MEM code was also exercised with a scripted local LLM
and deterministic lexical embedder. The run produced 3 LLM trace entries,
captured `llm_call`, candidate, write and query events, and reported
`api_calls=0`. Its result is saved at:

`C:\Users\Weilin\Desktop\AgentMemory\experiment_results_20261114\amem_offline_smoke\summary.json`

The current official smoke reports `retriever_scores=true`,
`exact_query_evidence=true`, `raw_context_trace=true`,
`llm_branch_decision=true`, and `stable_snapshot_ids=false`. Queue/cache
hooks and physical index rehydration remain unavailable in the stock adapter;
the wrapper reports those boundaries explicitly.

The external-data boundary was also exercised with one real instance from each
workload (no model calls): LoCoMo produced 419 source turns and 199 QA items;
LongMemEval-S produced 550 source turns and one QA item. The paired adapter
summary is saved at:

`C:\Users\Weilin\Desktop\AgentMemory\experiment_results_20261114\q01_dataset_adapter_smoke\summary.json`
