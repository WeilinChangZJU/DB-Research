# Q01 实验平台增量收口凭证（2026-11-14，历史快照）

> 本文记录审计修复前的离线骨架状态。当前修复结果、回归数量和真实
> source-level paired runner 以 `Q01_REPAIR_RECEIPT_20261114.md` 为准。

## 本轮目标

保留 Bemula 为行为层，在其上实现 Q01 的离线可验证实验骨架：source/provenance、canonical state、M0-M3 maintenance arms、成本计数和统一 fixture runner。未启动完整 lite、LoCoMo、LongMemEval-S 或 A-MEM 实验。

## 已实现

- `bemula/research/state.py`
  - `SourceProvenance`
  - `ProvenanceEvent` / `ProvenanceEventType`
  - `CanonicalArtifact` / `CanonicalRelation` / `CanonicalQuery`
  - `CanonicalState`
  - `CanonicalStateComparator`
  - 默认忽略物理存储 ID、run-local state/arm/wall-clock metadata；保留 source ID、content/support、关系、candidate 顺序、selected 顺序和 control dependency。
- `bemula/research/cost.py`
  - operation JSONL、span、LLM/token、embedding、retrieval、memory、index、rebuild 和 I/O counters；敏感文本只保存 hash。
- `bemula/research/maintenance.py`
  - M0 full rebuild oracle
  - M1 tombstone/filter
  - M2 suffix replay/repair-all
  - M3 read-set/control-dependency replay
  - deterministic toy builder、decision trace 和 replay cost。
- `bemula/research/q01.py`
  - 统一 M0-M3 结果、canonical state diff、snapshot diff、cost summary 的 `run_toy_experiment`。
- `scripts/run_q01_fixture.py`
  - 不调用模型/API 的可执行平台契约测试。
- `docs/Q01_PLATFORM_ARCHITECTURE.md`
  - L0-L6 分层设计、运行生命周期、Gate 0-Gate 5 和后续实际 builder 接入要求。

## 验证

```text
WSL: .venv/bin/python -m pytest -q tests
27 passed in 0.14s

WSL: .venv/bin/python -m py_compile \
  bemula/research/*.py bemula/research/datasets/*.py \
  scripts/run_research_eval.py scripts/run_q01_fixture.py
PASS
```

离线 fixture 的结果：

- M0：canonical reference；
- M1：在 shared-support fixture 上被判定为不等价；
- M2：与 M0 canonical state 一致；
- M3：与 M0 canonical state 一致；
- cost JSONL 和 summary 正常写出。

## 证据边界

这轮证明的是平台契约、比较器和 toy maintenance arm 的可运行性，不是 A-MEM、LightMem 或任何真实 Agent memory 方法的效果。真实 builder 接入前仍需实现 source-level provenance、snapshot、decision trace、full rebuild 和可审计的 read/control dependency。
