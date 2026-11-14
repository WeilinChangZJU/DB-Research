# Q01 platform repair receipt

Date: 2026-11-14 (Asia/Shanghai)

This receipt records the authorized repair cycle after the independent audit
package identified measurement, provenance, protocol, baseline, and runner
defects. The changes adapt the framework for Q01 validation; they do not claim
that Q01 is scientifically established.

## Completed changes

- Unified A-MEM canonical export for every maintenance arm. M1 now changes the
  live serving state and uses an explicit `conservative_cascade` policy. When a
  retriever factory is available, the survivor index is rebuilt and counted.
- Corrected source-mask transport, explicit empty-control semantics, word
  tokenization, retrieval-trace-gated top-k scenarios, and read-set-based old
  source exposure.
- Connected source → operation → decision → object dependencies, inherited
  support for copied payloads, and removed fabricated support for NOOP inputs.
- Captured linked-neighbor context objects and source mappings. Evidence is
  marked incomplete unless the final context can be mapped and retrieval scores
  are available. Index embeddings, document mappings, and runtime counters are
  included in the state contract.
- Made canonical comparison semantic-ID aware while preserving artifact and
  relation multiplicity. Physical UUID churn is ignored; embedding, index, and
  runtime-counter changes are observable when the continuation contract covers
  them.
- Separated deterministic M2 conservative suffix replay from M3 materialized
  read/control invalidation. Both currently rebuild the prefix and suffix, so
  `checkpoint_restored=false` and `maintenance_full_execution` are recorded.
- Added run fingerprints, exclusive worker ownership for the answer-level
  runner, resumable cost loading, and an explicit serial source-level A-MEM
  paired runner.
- Added a dataset-neutral paired replay runner for LoCoMo and LongMemEval-S
  adapters. Its outputs identify `controlled_replay` evidence and preserve
  independent-history clustering.
- Excluded incorrect low-cost states from reported cost-saving claims.

## Verification

```text
platform regression suite: 73 passed
independent audit contract suite: 15 passed
python compileall bemula scripts: passed
audit diagnostics: actual observe counts match reported replay counts;
                  expanded context source/read mappings include linked neighbors
```

Controlled pilot outputs are under:

- `C:/Users/Weilin/Desktop/AgentMemory/experiment_results_20261114/q01_locomo_paired_pilot2/summary.json`
- `C:/Users/Weilin/Desktop/AgentMemory/experiment_results_20261114/q01_longmemeval_paired_pilot50/summary.json`
- `C:/Users/Weilin/Desktop/AgentMemory/experiment_results_20261114/q01_scenarios_repaired2/summary.json`

The LoCoMo pilot used one history with 419 source turns. The LongMemEval-S
pilot used one history with a documented 50-turn source prefix out of 550
available turns. These are model-free controlled replays and are not A-MEM
quality or deployment results.

## Remaining research gates

- Run the serial `run_q01_amem_paired.py` path with a pinned official A-MEM
  commit, deterministic or explicitly measured model configuration, and a
  complete source-history JSONL.
- Compare M0, M1, M2, and M3 on at least two independent real workloads under
  one frozen deletion and state contract. Report history-clustered uncertainty.
- Add executable checkpoint restoration before claiming a checkpoint-based
  efficiency improvement. The current M2/M3 prefix is re-ingested and fully
  charged.
- Verify final-context mapping, queue/cache state, and physical-index
  restoration for each target builder. Unsupported hooks must remain
  `unknown`/`incomplete`.
- Keep the scientific status `PROMISING / UNVERIFIED` until the real-workload,
  full-cost, strong-baseline gates pass. No new algorithm claim is authorized
  by this receipt.
