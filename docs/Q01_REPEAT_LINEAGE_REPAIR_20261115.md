# Q01 M0 repeat 与最终上下文 lineage 修复复核（2026-11-15）

本阶段针对上一轮 real-model pilot 的两个阻断点进行修复：冻结模型下的
M0 repeat 不稳定，以及最终注入模型的上下文无法逐片段映射回 source。
本文区分代码验收、controlled 机制证据和 real-provider 执行证据；后者
没有完成的部分不被包装成科学结果。

## 已完成的代码修复

- 官方 robust A-MEM 的默认 metadata/evolution temperature 被显式冻结为
  `BEMULA_AMEM_TEMPERATURE`，本阶段冻结值为 `0.0`。freeze receipt 同时
  记录 `provider_replay_policy=m0_repeat_request_key_cassette` 和
  `random_seed`，formal verifier 与 paired-run fingerprint 会检查这些值。
- M0 reference repeat 使用独立 adapter/runtime，但重放第一次 M0 的
  request-key response cassette。metadata 记录 cassette-backed、命中数、
  未命中数和 replay 完整性；reviewer 要求策略正确且未命中为零。这是
  冻结执行语义下的 oracle repeat，不是独立的随机 provider 稳定性试验。
- 修复了真实 A-MEM 中由物理 UUID 排序给新对象分配 logical output slot
  的问题。logical origin 现在按语义字段排序后分配，避免两个独立构建
  在相同响应下产生不同的 source-derived canonical IDs。
- 最终上下文 lineage 现在按 pinned robust A-MEM 的实际 renderer 重建：
  seed、按顺序展开的 integer `links` neighbor，以及 `append -> j >= k`
  的停止规则。只有重建文本 hash 与 raw context hash 完全一致，才标记
  `context_segments_complete`；每个片段保存 role、object 和 source IDs。
  空的、已观测的最终 context 也被视为完整映射；无 raw finder 或 hash 不
  匹配仍保持 incomplete。

## 代码验收

Windows source snapshot 的完整测试为 **129 passed**，其中包含：

- 冻结 temperature、cassette policy、seed 和 runner fingerprint 检查；
- M0 repeat 请求账本命中与零 miss 检查；
- 随机物理 UUID 的 M0 canonical repeat 检查；
- duplicate-content、ordered-neighbor、empty-context 和 pinned renderer
  的结构化 lineage 检查。
- provider `402 Insufficient Balance` 等账户/配额拒绝现在被识别为不可重试
  的 external blocker，避免在余额不足时重复请求并留下更大的半成品 trace；
  formal runner 会写出 blocker receipt 和 stderr 路径。

无 API 的 pinned official A-MEM controlled runner 重新运行了 LoCoMo 与
LongMemEval-S 各 1 个 history，串行 M0--M3、`retrieve_k=5`、
`m3_mode=recheck`。两条 controlled run 均满足：

| workload | M0 repeat | M1 exact | M2 exact | M3 exact |
|---|---:|---:|---:|---:|
| LoCoMo | 1/1 | 0/1（预期的 tombstone 弱基线差异） | 1/1 | 1/1 |
| LongMemEval-S | 1/1 | 0/1（预期的 tombstone 弱基线差异） | 1/1 | 1/1 |

这证明修复后的 exporter、logical identity、cassette oracle 和 M2/M3
checkpoint 路径在确定性机制环境中闭合；controlled 结果不代表真实模型
质量或自然发生率。

### 受控多历史复核（无 API）

在本阶段收口时又从 pinned official A-MEM commit 重新执行了两组串行
controlled run。每组包含 2 个独立 source-level history，`retrieve_k=5`、
`m3_mode=recheck`，使用本地 scripted LLM 和 hash embedder；这些 rows 只用于
验证多历史隔离、checkpoint、状态比较和 lineage，不是 LoCoMo 或
LongMemEval-S 原始数据集的真实模型结果。

| controlled workload label | histories | M0 repeat | M1 exact | M2 exact | M3 exact | final-context complete | deleted-source exposure |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoCoMo-shaped | 2 | 2/2 | 0/2（弱 tombstone 预期差异） | 2/2 | 2/2 | 8/8 | 0
| LongMemEval-S-shaped | 2 | 2/2 | 0/2（弱 tombstone 预期差异） | 2/2 | 2/2 | 8/8 | 0

两组 controlled run 的每个 history 都完成了四个 arm 的隔离 query probe；
`context_segments_complete` 全部为 true，且读取来源集合没有命中删除来源。
这进一步验证了修复后的串行多历史执行路径，但由于使用 scripted model，
不能替代真实数据集、真实 provider 或自然发生率证据。

## real-provider pilot 复核

本阶段使用相同的冻结 A-MEM commit
`f9a173dec325d84fcc97e7828775c7b241c33466`、`deepseek-flash`、本地
`all-MiniLM-L6-v2`、thinking disabled、`retrieve_k=5`、每个 history 一个
query、串行 `m3_mode=recheck`。

LoCoMo 的 1 个 history 在 logical-origin 修复之前已经完成，因此它只可
作为本阶段的成本和失败诊断 receipt，不能作为修复后的通过证据。该次
运行记录：

- M0 reference repeat provider cassette：54/54 hits，0 miss，但
  `reference_repeat_equal=false`；差异集中在更新对象对应的 source
  canonical IDs，正是本阶段随后修复的 UUID-slot 问题。
- 4 个 arm query probe 均隔离运行，3 个非空 query 的结构化 context
  segments 完整，M1 的空检索路径在修复前未被显式标记为 empty-complete。
- `deleted_source_exposed=0` 只是已观测证据下界；本次 M0 gate 未通过，
  因而不能把 arm 差异解释为维护失败。
- 全部 lifecycle ledger 汇总为 267 次 LLM call、176,010 input tokens、
  33,988 output tokens、130 memory reads、298 generated objects、73
  checkpoint/index updates。provider 没有返回可独立核对的货币价格，金额
  不作推断；local embedding counters 仍为 unknown。

随后启动 LongMemEval-S 首个 history 时，provider 返回 HTTP **402
Insufficient Balance**，进程在 M0 尚未完成前停止，没有生成有效的
completed checkpoint/cost summary。该次 partial trace 被排除出统计，也不
被解释为算法失败。API key 临时文件已删除，未写入代码、报告或压缩包。

## Q01 结论更新

本阶段可以确认：上一轮 M0 repeat 不稳定并非只能归因于 provider sampling；
在固定响应 cassette 已 100% 命中的情况下，平台自身还存在物理 UUID 到
logical output slot 的实现错误。该错误已由随机物理 ID 回归和 controlled
official run 复现并修复。最终上下文 lineage 也已从文本反查升级为 pinned
renderer 的结构化 hash-checked 记录。

本阶段仍**不能**据此宣布 Q01 成立、不能比较 M1--M3 的真实模型优劣，也
不能进入新算法设计。原因是修复后的 real-provider LoCoMo 尚未重新完成，
LongMemEval-S real-provider pilot 又被 provider 余额阻断；因此 H1--H4 的
自然发生率、正确强基线后的可避免成本和创新性仍未得到证据支持或证伪。

当前状态保持：**PROMISING / UNVERIFIED，实验平台修复已通过，real-model
scientific gate 尚未通过。** 下一次有可用 provider 余额时，最低复现实验
是相同 freeze 下 LoCoMo 与 LongMemEval-S 各 1 个 history；验收必须同时
满足每个 history 的 M0 cassette repeat equality、零 cassette miss、每个
非空/空 final context 的结构化 lineage complete、独立 M0 continuation
检查以及完整 lifecycle cost coverage。若这些门通过，再扩到第二个
history；否则继续停留在平台校准阶段。

## 证据边界

本文件只保留聚合数字和可复现配置。数据集、原始 trace、checkpoint、
cost JSONL、provider 响应和 API key 均不进入 source repository 或导师
审阅包。
