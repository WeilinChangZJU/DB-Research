# Q01 多历史真实模型 Pilot 复核（2026-11-15）

## 结论

本轮 pilot 已经把冻结的官方 A-MEM、显式删除集合、串行 M0–M3、隔离查询和生命周期成本计量联通到真实模型路径，但它**没有通过 Q01 的科学验收门槛**。四个 history 的独立 M0 重复构建均不稳定，导致各维护臂相对 M0 的状态差异不能解释为维护失败；查询来源映射也不完整，因此“未观察到被删除来源暴露”只能作为观测下界。

当前 Q01 状态仍为：**PROMISING / UNVERIFIED**。本轮没有证据支持任何维护方法优于其他基线，也不能把 M3 的保守全历史重放称为效率优势。

## 冻结配置与实验范围

- 官方 A-MEM checkout：commit `f9a173dec325d84fcc97e7828775c7b241c33466`。
- 模型：`deepseek-flash`；thinking disabled；base URL 为冻结配置中的 DeepSeek endpoint。
- embedding：本地 `all-MiniLM-L6-v2`；`retrieve_k=5`。
- 工作负载：LoCoMo 2 个 history、LongMemEval-S 2 个 history；每个 history 16 个事件，带显式 `deleted_source_ids`。
- 每个 history 依次运行 M0、M1、M2、M3，并在每个 arm 上执行 1 个隔离 query probe；M3 使用 `conservative_full`，即完整历史 checkpoint replay 的强保守正确性基线。
- 原始 trace、checkpoint、cost JSONL 和 review JSON 保留在 WSL：
  `/home/weilin/projects/AgentMemory/q01_pilot_20261115/runs/formal_real_pilot2_userkey/`。
  这些原始产物不进入代码仓库或导师代码包。

## 执行契约复核

| 检查项 | LoCoMo | LongMemEval-S |
|---|---:|---:|
| checkpoint replay 可核验 | 通过 | 通过 |
| 共享 baseline 可核验 | 通过 | 通过 |
| 强 M3 策略可核验 | 通过 | 通过 |
| 每个 arm 均有 query probe | 通过 | 通过 |
| query probe 隔离 | 通过 | 通过 |
| M0 独立重复一致 | **未通过（0/2）** | **未通过（0/2）** |
| query evidence 完整 | **2/8** | **0/8** |
| continuation checks 通过 | **2/8** | **2/8** |
| workload review gate | **未通过** | **未通过** |

四个 history 的 16 个 arm 行均报告 `canonical state differs`，但每个 history 同时都满足 `reference_repeat_equal=false`。差异对象数量约为 22–31 个，未出现统一的 missing/extra 模式。这些差异在 M0 oracle 尚未稳定时不应归因于 M1–M3。

## 全生命周期成本

以下数值是两条 history 的 scope 汇总；token 是 provider 记录值，`unknown` 表示实现没有可靠提供该项，不能填零。

### LoCoMo

| scope | wall time | LLM calls | input tokens | output tokens | replay events | restore |
|---|---:|---:|---:|---:|---:|---:|
| common baseline | 129.5 s | 103 | 71,085 | 11,747 | 32 | 0 |
| reference rebuild (M0) | 133.1 s | 99 | 71,411 | 12,062 | 30 | 0 |
| reference repeat | 134.3 s | 98 | 71,134 | 12,056 | 30 | 0 |
| maintenance M1–M3 | 211.9 s | 184 | 119,791 | 18,748 | 57 | 4 events / 29,518 B |
| query probes | 4.8 s | 6 | 16,362 | 205 | — | score matrix 34,560 values |

按 arm 的维护执行：M1 为 0.09 s、0 次 LLM 调用；M2 为 84.7 s、84 次调用、27 次 replay；M3 为 127.1 s、100 次调用、30 次 replay。M1 的 token 成本为 unknown，不代表零成本。

### LongMemEval-S

| scope | wall time | LLM calls | input tokens | output tokens | replay events | restore |
|---|---:|---:|---:|---:|---:|---:|
| common baseline | 112.3 s | 97 | 125,171 | 9,480 | 32 | 0 |
| reference rebuild (M0) | 148.3 s | 97 | 142,103 | 12,439 | 30 | 0 |
| reference repeat | 120.7 s | 94 | 133,455 | 10,831 | 30 | 0 |
| maintenance M1–M3 | 175.8 s | 140 | 219,100 | 17,392 | 44 | 4 events / 273,725 B |
| query probes | 6.5 s | 8 | 56,492 | 95 | — | score matrix 39,936 values |

按 arm 的维护执行：M1 为 0.43 s、0 次 LLM 调用；M2 为 60.0 s、46 次调用、14 次 replay；M3 为 115.3 s、94 次调用、30 次 replay。embedding calls/items 在本轮汇总中为 unknown，后续必须单独补齐或明确不纳入成本声明。

## 状态与来源证据

- 所有 8 个 workload-arm 组合均有 probe，且 probe 运行隔离。
- `deleted_source_exposed=0` 是已捕获证据中的观测结果；LoCoMo 仅 2/8 probe 完成来源映射，LongMemEval-S 为 0/8，因此不能推出零暴露证明。
- M3 的 metadata 一致标记为 `conservative_full`、`strong_conservative_full_history_checkpoint_replay`、`complete_conservative_barrier`，并确认 checkpoint restored。它验证了一个保守正确性路径，尚未验证通用精确依赖维护或成本优势。
- query instrumentation 额外记录了 score encode 和矩阵计算：LoCoMo 6 次 encode、90 行、34,560 个值；LongMemEval-S 8 次 encode、104 行、39,936 个值。这些插桩成本应保留在完整成本账中。

## 研究结论与下一道门槛

本轮支持的工程结论是：正式 runner 能够在冻结配置下完成多历史、显式删除、串行 M0–M3、checkpoint replay、隔离 query 和全生命周期计量；M3 强保守路径已经可核验。

本轮不支持以下科学结论：

1. M1、M2 或 M3 相对 M0 的状态差异是维护算法造成的；
2. 某类维护方法优于其他基线；
3. 删除后没有来源暴露；
4. M3 已经是可泛化的精确决策依赖基线；
5. Q01 已经成立或值得直接进入新算法设计。

下一步必须先固定随机性和 replay 语义，使同一冻结配置下的 M0 repeat 稳定；同时补齐最终 query context 到 source 的逐片段 lineage 映射，并重复相同的 4-history paired pilot。只有在 M0 oracle 稳定、evidence 完整、继续执行检查通过后，才可以解释 M1–M3 的状态正确性和全成本差异。

## 排除项

- 使用旧 `.env` 中失效 key 的首轮运行返回 401，未产生完整 history，已排除。
- 48-event 的并行试跑在完成前终止，只有不完整 trace，已排除。
- 本文不包含 API key、数据集原文、原始 trace 或实验日志；它只记录可复核的聚合结果和研究判断。
