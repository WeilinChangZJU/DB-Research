# Q01 实验平台架构与推进路线

日期：2026-11-14

## 定位

Bemula 保留为 agent-facing 的行为层：多会话输入、control/forget 配对、自然语言 forget request、retention/efficacy/preservation/relearning/attack probes 和原有指标。

Q01 的核心验证层独立于 Bemula 的行为指标，研究对象是固定构建程序 `B_theta` 在 source-ID 撤回后的增量维护：

```text
M_ref = B_theta(H \ D)
M_inc = U(B_theta(H), D)
CanonicalVisible(M_inc) == CanonicalVisible(M_ref)
```

`theta` 必须冻结模型/版本、prompt、embedding、检索器、top-k、tie-break、随机性、执行顺序、缓存和发布时机。若无法保证确定性，应把等价语义改为明确的分布或行为距离，不能把随机波动计入删除错误。

本架构不把参数级 machine unlearning、物理磁盘擦除或第三方服务副本清理混入第一阶段；它研究带显式 provenance 的外部记忆构建与维护。

## 分层

### L0 冻结与 provenance

输入配置、数据、prompt、代码树、依赖、运行环境和 seed 生成 manifest。所有外部数据只引用原路径并保存文件 hash；不复制大型数据。正式实验前必须冻结 commit/tag。

### L1 Source ledger

每条 turn、文档、工具输出或事件拥有稳定 `source_id`、内容 hash、时间、顺序、session、source type 和版本。删除集合 `D` 在运行前注册，支持单 source、批量、shared-support、top-k boundary、early/middle/late、连续删除和 revoke-reinsert。

### L2 Builder state 与 decision trace

每次构建操作记录真实 read set、候选集、top-k 边界、过滤条件、ADD/UPDATE/NOOP、输出对象、support set、derived_from、query/embedding 版本、异步任务和 checkpoint。NOOP 也必须记录，因为未物化的控制决策可能决定删除后的正确状态。

### L3 Maintenance arms

在同一个 builder、同一份 `H`、同一 `D` 上运行：

- M0 Full rebuild：从 `H \ D` 完整重建，唯一 canonical oracle；
- M1 Tombstone/filter：当前实现是明确标注的 conservative-cascade 弱基线，
  删除命中 support 的物化对象并同步 live serving/index；它不等于 exact
  source-only deletion；
- M2 Replay/repair-all：从最早受影响事件重放或修复所有 provenance 后继；
- M3 Exact read-set/control-dependency：基于真实读取和控制分支，保守重查可能变化的节点；
- U Candidate：只有 M0–M3 建立正确性和成本空间后才进入。

所有维护臂都返回统一的 state snapshot、decision trace、cost vector、failure list 和 optional certificate。旧 `scripts.run_eval` 不被修改。

### L4 State oracle 与 canonical comparator

比较 active/derived objects、content/version、support set、关系、索引可见项、tombstone、read/evidence set 和状态 hash。只忽略无语义物理 ID；不能忽略内容、来源、关系、时间或影响排序的元数据。

输出三类差异：missing、extra、content/version mismatch、support mismatch、relation mismatch、retrieval mismatch、old-source exposure 和 survivor loss。query answer 只作为行为层补充。

### L5 Bemula behavior shell

将同一 instance 的状态和 query 映射到 Bemula 的 retention/efficacy/preservation/relearning/attack probes。Bemula FE/PR/AR 等指标必须与 state exact match 分开报告，不能互相替代。

### L6 Cost 与统计

记录初始构建、正常写入、query、删除传播、replay、LLM calls/tokens、embedding、retrieval、index update、publish/cleanup、wall-clock、p50/p95、CPU/GPU/存储和生命周期总成本。统计单位优先是独立 history/session，不把同一历史下的问题当成独立样本。

## 运行生命周期

1. 冻结 manifest、builder config、数据 hash、删除 mask 和 seed。
2. 在 `H` 上运行 baseline builder，写入 source ledger、decision trace、checkpoint 和 `M(B_theta(H))`。
3. 对每个 maintenance arm 应用同一个 `D`。
4. 运行 M0 生成 `M_ref`，运行 M1–M3/U 生成 `M_inc`。
5. 对每个结果执行 canonical state comparison、old-source exposure、survivor retention、query equality 和 cost accounting。
6. 可选地接入 Bemula paired probes，报告行为指标与状态指标的二维结果。
7. 将失败案例保存为可重放 fixture；任何不确定性、unsupported capability 或 state instability 都进入 manifest/report。

## 阶段门

- Gate 0：manifest、source hash、seed、配置和代码树可复现；
- Gate 1：同一 builder 重复构建得到稳定 canonical state；
- Gate 2：M0/M1/M2/M3 在 toy fixture 上可运行并能制造/发现已知差异；
- Gate 3：实际 A-MEM/LightMem wrapper 支持 source provenance、snapshot 和可审计决策；
- Gate 4：LoCoMo/LongMemEval-S 只做 source overlay 与受控 deletion masks，不把原 QA 标签冒充 unlearning truth；
- Gate 5：只有在真实历史上观察到非物化决策依赖且强基线留下成本空间时，才实现候选 U。

## 当前实现状态

已完成测量基础与离线闭环：manifest、checkpoint、trace、ResearchProtocol、
MemoryBackend、InMemoryBackend、LoCoMo/LongMemEval-S adapter、canonical
state、dependency trace、M0–M3 maintenance API、cost vector、toy builder、
source-level paired runner 和统一回归测试。当前平台套件为 73 项通过，
独立审计契约为 15 项通过。

LoCoMo 与 LongMemEval-S 已各完成一个无模型 controlled-replay pilot；这
些 pilot 只验证 source overlay、mask、canonical state 和成本输出，不能
替代真实 A-MEM paired workload。完整 lite/full、真实 A-MEM 和 GPU 实验仍
保持暂停，直到 Gate 3–5 的输入、证据和强基线条件满足。

## 当前 paired runner 与成本边界

`scripts/run_q01_amem_paired.py` 是 A-MEM source-event 闭环入口。它读取每行一个独立 history 的 JSONL，要求显式提供 `deleted_source_ids`，在同一 `H,D` 上串行运行 M0--M3，并写出每个 arm 的 canonical hash、状态差异、trace、成本、checkpoint 和运行 fingerprint。串行是第一轮的安全默认值；A-MEM adapter 是可变对象，未完成独占 worker 验证前不启用并发。

M2 的成本包含从最早删除位置开始的前缀重新摄取和后缀执行。M3 使用 materialized read/control closure 选择失效起点，但当前仍是保守的前缀重建加后缀重放；`m3_status` 与 `checkpoint_restored=false` 会在结果中显式记录。保存语义 JSON 不被当作可恢复执行 checkpoint。

