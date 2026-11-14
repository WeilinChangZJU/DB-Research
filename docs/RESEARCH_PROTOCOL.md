# Bemula ResearchProtocol 平台说明

本目录包含 Bemula 的研究适配层。它在保留原有 `run_eval.py` 和 F1-F6/攻击评测语义的同时，增加可审计、可恢复、可接入外部数据的实验接口。

## 组件

- `bemula.research.manifest`：记录 Git 提交、运行配置、输入文件 SHA-256、prompt/数据路径、Python/依赖、平台和安全环境摘要。API key 只记录 configured 状态和短 hash，不写入明文。
- `bemula.research.checkpoint`：每个 instance 一个独立 JSONL 文件；提交清单使用原子写入；恢复前校验文件 hash 和记录数，损坏或未提交的 instance 不会被当作完成。
- `bemula.research.trace`：线程安全 JSONL 事件流，记录 instance、LLM、memory、span 和错误事件。LLM 事件只保存 prompt hash、模型、usage、latency 等审计字段。
- `bemula.research.cost`：按 build/observe/retrieve/decision/rebuild 聚合操作成本，记录 LLM/token、embedding、retrieval、memory、index、重算单元和 I/O 计数。支持 JSONL 明细和 `summary.json` 汇总，不保存 prompt 或 source 文本。
- `bemula.research.backend`：统一的 `MemoryBackend`、query evidence、forget result、snapshot/diff、introspection 接口，并提供离线 `InMemoryBackend` 作为协议开发和回归测试基线。
- `bemula.research.protocol`：固定 control -> forget -> query 的阶段顺序，支持显式 forget/delete、state snapshot/diff、evidence 和 trace sink。
- `bemula.research.datasets`：LoCoMo 与 LongMemEval-S 适配器，保留 source path/hash、turn/source hash、session/date/speaker、evidence 和 answer-session provenance。

## 新运行器

原有 `scripts.run_eval` 保持不变。研究版运行器是显式 opt-in 的：

```bash
PYTHONUNBUFFERED=1 .venv/bin/python -m scripts.run_research_eval \
  --config configs/deepseek.yaml \
  --baseline fullcontext \
  --dialogues data/quick_check/dialogues.jsonl \
  --questions data/quick_check/questions.jsonl \
  --out results/research_smoke \
  --limit 1 --workers 1
```

可选 `--resume` 只跳过已经通过 checkpoint 完整性校验的 instance。输出包括：

- `manifest.json`：provenance 和运行状态；
- `checkpoints/instances/*.jsonl`：每实例原子结果；
- `checkpoints/completed_instances.jsonl`：完成清单及 SHA-256；
- `items.jsonl`、`metrics.json`、`report.md`、`trace.jsonl`。

`items.jsonl` 保存 control/forget 两个变体的原始记录，因此其行数可能大于 `metrics.json` 中的 scored `num_items`；control attack/preservation 记录用于配对校准，不直接重复计入聚合指标。

## 外部数据适配

```python
from bemula.research.datasets import load_locomo, load_longmemeval_s

locomo = load_locomo("/path/to/locomo10.json", limit=1)
longmem = load_longmemeval_s("/path/to/longmemeval_s_cleaned.json", limit=1)
protocol_instance = locomo[0].to_protocol_instance()
```

适配器只读取原始文件，不把大型数据复制到仓库。LoCoMo 的 9 个未能完全解析 evidence 的 QA 会显式标记；LongMemEval-S 的 `answer_session_ids` 会映射到稳定 source turn IDs。外部数据适配层不会臆造 Bemula 的 F1-F6、attack 或 unlearning 标签；这些语义需要在后续 ResearchProtocol suite 中明确定义后才能比较遗忘指标。

## 验证边界

离线测试覆盖 manifest、checkpoint、trace、protocol 和两个数据适配器。DeepSeek 只做小规模启动/接线 smoke test；它不构成完整 lite 集或方法有效性的证据。完整实验以及此前暂停的 A-MEM 运行保持不恢复。

## Q01 成本轨迹

```python
from bemula.research import CostRecorder

cost = CostRecorder("results/run/cost.jsonl")
cost.record("retrieve", counters={"retrieval_calls": 1, "retrieved_items": 5})
with cost.span("rebuild", arm="M0"):
    pass
cost.record_llm(task="summary", model="local", input_tokens=120, output_tokens=32)
cost.write_summary("results/run/cost_summary.json")
```

通用计数器包括 `llm_calls`、`input_tokens`、`output_tokens`、`embedding_calls`、`retrieval_calls`、`retrieved_items`、`decision_count`、`memory_writes`、`memory_updates`、`memory_deletes`、`index_updates`、`rebuild_units` 以及读写字节数。不同 builder 可以加入额外的数值计数器；维护臂比较时应使用同一计数器口径。

## Q01 canonical state layer

`bemula.research.state` adds a serializable state oracle without changing the original evaluator:

- `SourceProvenance` records stable source identity, content hash, session/ordinal, revocation and parent provenance.
- `ProvenanceEvent` records ingest/read/candidate/select/write/update/NOOP/forget/publish/purge decisions.
- `CanonicalArtifact`, `CanonicalRelation`, and `CanonicalQuery` preserve visible content, support sets, relations, candidate order, selected top-k order and control dependencies.
- `CanonicalState.from_snapshot()` lifts an existing `MemorySnapshot` into the state schema.
- `CanonicalStateComparator` returns deterministic hashes and categorized mismatches. Physical storage IDs and default wall-clock metadata are ignored; source IDs, content, support sets, relations, candidate order and selected order remain semantic.

```python
from bemula.research import CanonicalStateComparator

diff = CanonicalStateComparator().compare(incremental_state, rebuild_state)
assert diff.equivalent  # only after the full canonical state matches
```

This comparator is an oracle component, not an unlearning algorithm. It cannot establish equivalence when the builder is nondeterministic or when provenance/decision events were not captured.

## Q01 maintenance arms and offline runner

`bemula.research.maintenance` provides a deterministic reference harness for the four fixed arms. It is intentionally a toy builder used to test the platform contracts; it is not an LLM result or a claim about A-MEM.

```python
from bemula.research import CostRecorder, SourceEvent, run_toy_experiment

events = [
    SourceEvent("e0", "src-a", "approval limit is ten thousand"),
    SourceEvent("e1", "src-b", "approval limit is ten thousand"),
]
run = run_toy_experiment(events, ["src-a"], cost=CostRecorder())
print(run.summary())
```

The runner applies the same deletion mask to M0 full rebuild, M1 tombstone/filter, M2 replay/repair-all and M3 read-set replay. It compares each arm with M0 through the canonical state comparator and records replay/retrieval/write counters. A real builder must provide equivalent source provenance, state snapshot and decision-trace contracts before its result can be used for Q01 claims.

## A-MEM research wrapper

`bemula.research.amem.AmemResearchWrapper` wraps an already-configured Bemula `AmemAdapter` (or a deterministic `MemoryAdapter` fixture) without importing the official A-MEM repository at module import time. It binds external source IDs to note IDs, emits serialisable source/note snapshots, records ingest/candidate/write/update/noop/read/select events, captures best-effort retrieval evidence, and exposes forward/reverse dependency introspection. `AmemHistoryBuilder` and `run_amem_all_arms` connect this wrapper to the same M0-M3 maintenance contract. The wrapper reports `forget=False`, `delete=False`, and `llm_branch_decision=False` for the stock adapter because it has no native source-level delete or complete parsed LLM read/control API. These limitations are explicit; the wrapper cannot by itself establish Q01 exactness.

```python
from bemula.research import AmemResearchWrapper
backend = AmemResearchWrapper(amem_adapter)
```
