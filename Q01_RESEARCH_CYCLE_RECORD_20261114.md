# Q01 research validation cycle

## 2026-11-14 — authorized platform repair cycle started

- Candidate: Q01 retrieval-decision dependency under source-level deletion.
- Status: `UNVERIFIED` before this cycle.
- Formal tuple: `P=(D,Q/W,A,R,C,B,Delta,S)` is being frozen in the platform
  roadmap. The current cycle treats the state contract, paired M0–M3 arms,
  provenance, and full lifecycle cost as prerequisites before any claim about
  a new maintenance method.
- Evidence level at start: artifact evidence and controlled audit evidence;
  no independent real-workload paired maintenance result.
- User authorization: complete code and validation workflow changes needed to
  adapt the current framework for Q01 validation, with stage-by-stage review.
- Initial blockers: audit package reports 14/15 independent contract failures;
  M1 live state is not isolated, M2/M3 share replay logic, provenance and query
  evidence are incomplete, canonical comparison has false positives/negatives,
  and protocol/runner/cost accounting are not closed.
- Next gate: restore a clean baseline test run, then make the P0 contract
  regressions pass without weakening assertions.

## Append-only updates

### 2026-11-14 — G0/P0/P1/P2/P3 repair checkpoint

- P0 measurement and protocol fixes completed: one builder-specific canonical
  exporter for A-MEM arms; live conservative-cascade M1 tombstones; correct
  tokenizer; explicit empty-control handling; delete-mask propagation; read-set
  exposure checks; retrieval-trace-gated top-k scenarios.
- P1 observation/state fixes completed: typed operation/decision/object edges,
  support inheritance for copied payloads, no fabricated NOOP support, expanded
  query-context mapping, explicit incomplete evidence, embedding/index/runtime
  fingerprints, semantic score-key normalization, and multiplicity-preserving
  comparison.
- P2 baseline/cost fixes completed: deterministic M2 and M3 now use distinct
  invalidation policies; A-MEM M2/M3 record separate replay modes and status;
  prefix re-ingestion is counted as maintenance cost; checkpoint restoration is
  explicitly false until an executable checkpoint exists; incorrect low-cost
  states are excluded from savings claims.
- P3 runner fixes completed: answer-level `run_research_eval.py` remains
  separate; `run_q01_amem_paired.py` provides a serial source-level A-MEM
  paired path with fresh adapters and run fingerprints; `run_q01_paired.py`
  provides model-free paired replay on LoCoMo/LongMemEval-S adapters with
  history-clustered output and optional scaled source prefixes.
- Verification: platform suite `70 passed`; independent audit suite `15
  passed`; `compileall` passed; audit diagnostics now report actual observe
  counts equal to reported replay counts and expanded-context source/read sets
  include both seed and linked neighbor sources.
- Controlled real-source pilots completed: one LoCoMo history (419 source
  turns, one scenario) and one LongMemEval-S history using a 50-turn scaled
  prefix (550 available, 50 used). Both are `controlled_replay` evidence only;
  no LLM or GPU result is claimed.
- Decision status remains `PROMISING / UNVERIFIED`. The platform contract is
  repaired, but independent real-workload A-MEM paired correctness/cost and a
  full two-workload strong-baseline decision gate remain outstanding.

### 2026-11-14 — G4 accounting clarification

- Added `affected_set_precision_recall` as a separate diagnostic for
  conservative over-invalidation and missed objects.
- Cost-saving summaries now exclude any observation that is not exact to M0;
  the platform suite is now `73 passed`, with the independent audit suite still
  `15 passed`.

### 2026-11-14 — platform-ready decision

- The framework now reaches the requested platform-ready boundary: source
  masks, canonical visible/runtime/audit hashes, provenance and final-context
  evidence, distinct M2/M3 policies, full maintenance cost scopes, durable
  trace/fingerprint receipts, paired runners, and two workload adapter pilots
  are all executable and tested.
- Remaining work is deliberately outside code-contract repair: obtain a pinned
  official A-MEM runtime/model configuration, run independent real histories
  on both workloads, verify builder-specific index/cache/queue restoration, and
  make the keep/narrow/close research decision from full-cost paired results.
- Research status remains `PROMISING / UNVERIFIED`; no new algorithm or
  efficiency-superiority claim is supported by the controlled pilots.


### 2026-11-14 — A-MEM source freeze, explicit histories, and paired preflight

- Official A-MEM was frozen to commit `f9a173dec325d84fcc97e7828775c7b241c33466`, verified against the remote `main` ref. Archive/source/requirements/model/input digests are recorded in `experiment_results_20261114/q01_amem_freeze/freeze.json`.
- Explicit deletion histories were generated and streaming-validated: LoCoMo has 10 histories; LongMemEval-S has 500 histories. Every row contains authoritative `deleted_source_ids` that occur in the row's events. LongMemEval-S evidence is explicitly session-level for answer overlays, not exact turn provenance.
- The real serial runner was invoked with the frozen `deepseek-flash` model name and stopped before model calls because no `OPENAI_API_KEY` is configured. No real-model result is claimed.
- A controlled official-code preflight ran one deterministically reduced history from each workload through serial M0–M3. Both pilot histories were exact to M0 for M2 and M3; conservative-cascade M1 was non-equivalent on both. M2 and M3 had identical replay starts and no observed cost advantage in these scripted traces.
- Cost accounting was tightened during the stage: embedding calls/items and wall time are recorded; prefix re-ingestion is charged; mutable-arm baseline setup is recorded in a separate `baseline_setup` scope; M0 no longer performs an uncharged discarded baseline build.
- Final controlled pilot review: `experiment_results_20261114/q01_amem_freeze/Q01_AMEM_PAIRED_REVIEW_20261114.md`; machine-readable ledger: `controlled_pilot_release2_review.json`.
- Verification after the final cost changes: platform tests `75 passed`; targeted audit/history-generation tests `17 passed`; Python compilation passed.
- Decision: platform/source-level preflight gate passes, but real-workload and provider-cost gates are blocked by missing credentials. Q01 remains `PROMISING / UNVERIFIED`; no new-method or superiority claim is supported.
- The real runner now consumes JSONL histories lazily (with a compatibility list helper), so the 404 MB LongMemEval-S artifact is not duplicated as a full in-memory history list before execution.
- The controlled runner now performs a fixed post-maintenance query probe per arm/history. In the two-history pilot, all 8 probes had complete evidence mapping and none exposed the deleted source ID; query costs are in the separate query_probe scope.
- The explicit history generator now preserves dataset questions and declared source overlays while omitting gold answers. LoCoMo/LongMemEval-S JSONL and manifests were regenerated and rehashed; the controlled serving probes now use preserved dataset questions rather than synthetic deleted-text queries.
- After adding preserved item overlays and query-aware formal runner support, the platform suite is `76 passed`; history-generation tests are `3 passed`. The controlled query-aware pilot uses real dataset questions and reports 8/8 complete evidence probes with 0 deleted-source exposures.
- A streaming input validation receipt now verifies all three explicit JSONL artifacts: deleted IDs are in-row, item source overlays are in-row, and no gold-answer keys are present. Receipt: xperiment_results_20261114/q01_amem_freeze/input_validation_receipt.json.
- Full-history controlled scale preflight also completed for one complete LoCoMo history (419 events) and one complete LongMemEval-S history (550 events). Across these four controlled histories, M0/M2/M3 were exact, M1 was non-equivalent, and all 16 preserved-question probes were evidence-complete with zero deleted-source exposures. This remains controlled evidence, not a real-model workload result.
- The combined full-history controlled cost ledger covers 56,077.6 ms reference rebuild, 108,284.2 ms paired baseline setup, 75,322.6 ms post-deletion maintenance and 63.1 ms query probes; no provider money cost is claimed because the scripted path made no remote calls.
- Added strict formal orchestration and reviewer scripts. The orchestrator verifies frozen source/input hashes, refuses to start without OPENAI_API_KEY, runs LoCoMo then LongMemEval-S serially with resume, and invokes the cost/state/source reviewer. A current blocker receipt was produced without starting model calls.

### 2026-11-14 — real-provider focused-window pilot

- The first real-provider run was deliberately bounded for cost. A full-history one-history attempt was stopped after the observed trace reached about 1.87M tokens and 272 MB while still in the first mutable setup; it is retained as `formal_real_small_local` and excluded from scientific results. A separate 144-event attempt used an invalid credential placeholder and failed with provider HTTP 401 during query; it is retained as `formal_real_pilot_short` with status `AUTH_FAILED_INCOMPLETE` and is also excluded.
- A no-write provider preflight with the user-supplied credential succeeded for `deepseek-flash`. The completed pilot then used two 32-event contiguous windows around the frozen deletion source: LoCoMo `[96,128)` and LongMemEval-S `[129,161)`, each with the deleted source at local ordinal 8. Official A-MEM remained frozen at commit `f9a173dec325d84fcc97e7828775c7b241c33466`; embeddings used the cached local `sentence-transformers/all-MiniLM-L6-v2` variant because the DeepSeek endpoint returned 404 for the declared API embedding route.
- Both workloads completed serial M0–M3 with one history each. M0 was exact to itself (`1/1`); M1, M2 and M3 were non-equivalent (`0/1`) on both workloads. M1's result is consistent with conservative cascade over-removal; M2/M3 mismatches cannot yet be separated from real-model replay variability because the internal A-MEM calls remained temperature `0.7` and no deterministic response map was used.
- All 8 focused-window query probes were evidence-complete and reported zero deleted-source exposures. The old query probe path did not capture `adapter.llm.chat` usage, so query answer cost is marked unknown rather than zero. The wrapper and reviewer were patched to capture query chat usage for future runs; platform tests now pass `79`.
- Full arm costs (including mutable-arm baseline setup) were: LoCoMo M0/M1/M2/M3 = `177,808.5 / 117,969.0 / 236,269.2 / 219,782.8 ms` and `106 / 98 / 193 / 191` LLM calls; LongMemEval-S = `132,816.8 / 130,032.4 / 228,494.9 / 251,471.6 ms` and `96 / 100 / 188 / 194` LLM calls. Provider-money cost is not inferred from tokens.
- M3 metadata identifies a distinct materialized read/control-closure baseline, but the pilot does not establish a general exact-dependency implementation or an efficiency advantage over M2. Machine review: `experiment_results_20261114/q01_amem_freeze/formal_real_focused_window_review.json`; narrative review: `Q01_AMEM_FOCUSED_WINDOW_REVIEW_20261114.md`.
- Decision update: `REAL_MODEL_FOCUSED_WINDOW_COMPLETE / Q01 PROMISING-UNVERIFIED`. The real provider gate is now exercised, but the scientific gate remains open: first calibrate deterministic replay/M0 variability and query-cost accounting, then expand paired histories only if the corrected gate still shows a nontrivial, attributable gap.
