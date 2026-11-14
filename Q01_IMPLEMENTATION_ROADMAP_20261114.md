# Q01 平台开发路线图

## 冻结目标

平台必须能够在固定构建器 `B_theta` 和 source history `H` 上比较：

```text
M_ref = B_theta(H - D)
M_inc = U(B_theta(H), D)
```

并在同一个 canonical oracle、同一删除集合、同一查询契约和完整生命周期成本下判断：

1. 直接 tombstone/filter 是否因检索决策依赖而错误；
2. M2/M3 或候选方法是否恢复到 M0 的 canonical state；
3. 依赖感知维护是否降低完整删除维护成本。

完整 DeepSeek 和完整数据集实验保持暂停，直到 G0/G1/G2 gate 通过。

## 分阶段执行

| 阶段 | 交付物 | 通过条件 |
|---|---|---|
| G0 契约层 | canonical A-MEM state、稳定 semantic ID、snapshot/load、已知反例 fixture | 重复构建稳定；M0 是 oracle；M1 发现已知失败；M2/M3 可恢复 |
| G1 插桩层 | process_memory/LLM branch/retriever/index/neighbor/queue/cache trace | 每次 mutation 可回溯到 source、candidate、support、branch；trace 可重放 |
| G2 维护层 | dependency graph、affected closure、统一 M0-M3、完整 cost ledger | 在 deterministic scripted A-MEM 上 exactness 和完整成本可比较 |
| G3 场景层 | LoCoMo/LongMemEval-S source overlay、deletion mask、paired query runner | 两个数据适配器都能生成可审计 paired histories |
| G4 统计层 | failure taxonomy、affected-set precision/recall、paired bootstrap、报告 | 每个失败可重放；独立 history 统计可生成 |
| G5 问题验证 | 受控 real subset，再扩大 lite/full | 两个 workload 上依赖现象、正确性和成本 gate 同时满足 |
| G6 方法层 | dependency-aware U、certificate、ablation | 只在 G5 通过后实现和比较 |

## 实施顺序

1. 先完成 G0：修正 A-MEM runner 使用 raw snapshot 的问题，补 stable semantic state 和 durable checkpoint。
2. 并行完成 G1：在官方 robust A-MEM 边界注入 deterministic LLM/embedding hooks；所有 raw prompt/content 只保留 hash。
3. 接入 G2：让 M0-M3 使用同一 builder、同一 deletion mask、同一 canonical state 和同一 cost schema。
4. 完成 G3/G4：生成 deletion scenarios，构造 paired runs，输出 mismatch/failure/cost report。
5. 运行 deterministic scripted A-MEM 和两个数据适配器的小规模验证；不调用 DeepSeek。
6. 只有问题验证 gate 通过后，才计划 DeepSeek smoke、lite 31 和 full 数据集。

## Keep/close 规则

- `UNVERIFIED`：插桩、canonical state 或成本口径不完整。
- `PROMISING`：受控 history 已观察到检索依赖和 M1 失败，但真实 workload 或 full cost 未完成。
- `ESTABLISHED`：两个独立 workload、相同 Q0 质量契约、完整成本和强基线均通过，最小机制保持 exactness 并改善成本 frontier。
- `CLOSED`：M1 已足够、没有稳定的检索依赖现象，或依赖维护没有成本优势。

## 明确边界

平台验证的是固定 `B_theta` 下的语义状态等价和维护成本。它不自动证明任意随机 LLM 系统的数学等价，也不研究参数级 machine unlearning、磁盘物理擦除或第三方服务副本清理。

## 当前执行状态

- G0：已通过基础确定性、canonical state、semantic checkpoint、M0-M3、scenario manifest、统计回归和 A-MEM 成本账本回归；当前离线 suite 为 46 tests passed。
- G1：已通过官方 robust A-MEM 的 deterministic local LLM/embedding smoke，已经捕获 LLM hash、保守 branch label、candidate、index summary 和 query trace。
- G1 未完成项：完整 parsed branch/control dependency、cache/queue、稳定物理索引恢复仍需继续插桩；当前 official smoke 已能捕获 retriever scores、raw context 和 exact query evidence，但 branch 仍是保守标签，稳定 semantic snapshot 已用于比较而不是替换 stock adapter 的物理 UUID。
- G2：统一成本字段、dependency graph、paired failure report 和 bootstrap 已在 deterministic harness 通过；真实双 workload 尚未运行。
- G3：LoCoMo/LongMemEval-S 的 source/evidence overlay 与 deletion manifest 已用各 1 个真实实例完成无 API smoke；paired A-MEM histories 尚未运行。
- G4：failure taxonomy、history-level paired summary 和 bootstrap 已通过离线回归；独立 history 统计尚未形成科学证据。
- G5：尚未启动，完整数据集运行和 DeepSeek 完整实验仍暂停。
