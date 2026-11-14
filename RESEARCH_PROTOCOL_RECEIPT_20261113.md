# Bemula ResearchProtocol P0 收口凭证（2026-11-13）

## Scope

完成 Bemula 研究适配层的 P0 工程改造；未恢复完整 lite、LoCoMo 或 LongMemEval-S 实验，也未恢复此前暂停的 A-MEM 全量运行。

## Delivered modules

- `bemula/research/manifest.py`：输入/配置/代码树哈希、Git/runtime、安全环境摘要、原子写入。
- `bemula/research/checkpoint.py`：每实例 JSONL、SHA-256/record-count 校验、完成清单、恢复与篡改检测。
- `bemula/research/trace.py`：线程安全 JSONL trace、span/LLM/memory 事件与 protocol event bridge。
- `bemula/research/backend.py`：MemoryBackend、evidence、forget、snapshot/diff、ResearchAdapter、InMemoryBackend。
- `bemula/research/protocol.py`：control/forget/query 阶段、显式 forget/delete、snapshot/diff、trace。
- `bemula/research/datasets/`：LoCoMo、LongMemEval-S/LongMenEval-S 适配器与 provenance 保留。
- `scripts/run_research_eval.py`：显式 opt-in 的 provenance-aware 运行器；原 `scripts.run_eval.py` 保持不变。

## Validation

- WSL offline suite: `13 passed in 0.13s`。
- Python compile check: `scripts/run_research_eval.py`、research modules and adapters all pass.
- Dataset/protocol offline integration: LoCoMo 1 instance (419 turns/199 items) and LongMemEval-S 1 instance (550 turns/1 item) both run through `ResearchProtocol + InMemoryBackend`.
- Checkpoint validation of smoke output: completed instance `tiny:inst0`; 19 raw records; no invalid checkpoint files.

## DeepSeek smoke

- Output: `experiment_results_20261113/research_smoke_deepseek/`
- Baseline: `fullcontext`; limit: 1; workers: 1; elapsed about 30 seconds.
- Manifest status: `completed`; metrics scored 11 items, all 1.0 on this tiny smoke input.
- The 19 raw `items.jsonl` records include control calibration records; scored metrics intentionally exclude duplicated control records.
- Manifest stores only `OPENAI_API_KEY.configured=true` and a short hash; no plaintext key was written to manifest, trace, items, report, or checkpoint.
- This smoke test verifies API wiring/startup and output integrity only; it is not efficacy evidence.

## Known follow-up boundary

The external dataset adapters preserve source and answer provenance, but do not invent Bemula F1-F6 forget semantics or attack labels. A next research layer must define the paired retention/forget/attack suite and metric mapping before making unlearning claims on LoCoMo or LongMemEval-S.


## Q01 state layer increment (2026-11-14)

Added `bemula/research/state.py` and `tests/test_state.py`. The state schema captures source provenance, lifecycle decisions, canonical artifacts, relations, historical query candidates/top-k order, tombstones, and provenance events. `CanonicalStateComparator` ignores storage addresses and default wall-clock metadata while preserving source IDs, content/support sets, relations, candidate order, selected order, and control dependencies. WSL full suite after this increment: `24 passed in 0.17s`.

## Q01 platform increment（2026-11-14）

新增：

- `bemula/research/state.py`：SourceProvenance、ProvenanceEvent、CanonicalState、CanonicalStateComparator；
- `bemula/research/cost.py`：操作、LLM/token、embedding、retrieval、index、rebuild 和 I/O 计数；
- `bemula/research/maintenance.py`：M0 full rebuild、M1 tombstone/filter、M2 replay/repair-all、M3 read-set replay 的确定性参考实现；
- `bemula/research/q01.py`：统一 Q01 fixture runner，连接维护臂、canonical comparator 和成本记录；
- `scripts/run_q01_fixture.py`：完全离线的可执行平台契约测试。

验证：WSL `pytest -q tests` 为 `27 passed`，所有 research modules、dataset adapters 和两个 runner 均通过 `py_compile`。未调用 DeepSeek，未启动完整数据集。

在 shared-support toy fixture 上，M1 被 canonical comparator 判定为不等价；M2/M3 与 M0 完全一致。该结果只验证平台契约和已知 fixture，不代表真实 A-MEM/LightMem 效果。

## A-MEM wrapper increment (2026-11-14)

Added `bemula/research/amem.py`, `scripts/run_amem_fixture.py`, `docs/AMEM_RESEARCH_WRAPPER.md`, and `tests/test_amem_backend.py`. `AmemResearchWrapper` exposes A-MEM source/note snapshots, wrapper-level provenance events, best-effort retrieval evidence, dependency introspection, and capability flags while failing loudly for unsupported native forget/delete. `AmemHistoryBuilder` connects the wrapper to source-aware M0-M3 replay. The deterministic single-history fixture reports M0/M2/M3 exact to M0 and M1 intentionally non-exact on a shared-support history. WSL full suite after this increment: `30 passed in 0.15s`; no DeepSeek/API call was made.
