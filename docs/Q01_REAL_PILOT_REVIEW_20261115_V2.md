# Q01 Real-Model Focused Pilot Review (2026-11-15, current)

本报告是当前代码修复后的阶段性收口记录。它把“已经完成的真实 provider receipt”“修复前有效 pilot”“修复后因认证失败而无效的尝试”分开，不把不完整 trace 当作科学结果。

## 结论

Q01 仍处于 **PROMISING / UNVERIFIED**。本阶段没有证据支持某个维护策略优于其他策略，也没有证据证明 Q01 已经成立或已经可以进入新算法设计。

已经得到的结果是：

1. 冻结的官方 A-MEM commit、DeepSeek 模型配置、本地 embedding、显式删除集合、LoCoMo/LongMemEval-S focused history、串行 M0–M3 runner、隔离 query probe 和全成本 ledger 已经联通到真实 provider 路径。
2. 上一次有效的 focused real-model pilot（每个 workload 1 个 history、32 个事件、1 个 query）发现：provider request-key cassette 命中完整，但 M0 可见状态仍不稳定；差异来自平台的物理 UUID 到 logical output/source lineage 规范化，而不是 provider response 不一致。
3. 该 UUID 平局排序和 source `derived_from` 顺序问题已经修复；第四轮又补齐共同 oracle、缓存键、recheck memo、fork、renderer 和版本闭包契约，当前 Windows 全套测试为 **139 passed**。详细记录见 `Q01_FOURTH_AUDIT_REPAIR_20261115.md`。
4. 修复后的 v2 real-provider 重跑没有形成有效实验结果：DeepSeek 返回 HTTP 401 Authentication Fails，官方 A-MEM 捕获异常并继续写入未演化 note。该 partial run 已停止、排除，不计入任何正确率、成本或 Q01 统计。
5. wrapper 现在记录 provider error ledger，并在 observe/query 后拒绝上游 fallback；下次认证失败会在首个受影响操作后 fail-fast，而不是继续生成可误读的半成品 trace。

因此，修复后的真实 M0–M3 scientific gate **尚未通过**。下一次必须在有效 provider credential 下，用同一 freeze 和同一 post-fix source tree 重新完成 LoCoMo 与 LongMemEval-S 各至少 1 个 history。

## 冻结与范围

- 官方 A-MEM commit：`f9a173dec325d84fcc97e7828775c7b241c33466`。
- 模型：`deepseek-flash`；thinking disabled；A-MEM temperature `0.0`；DeepSeek OpenAI-compatible endpoint。
- embedding：本地 `sentence-transformers/all-MiniLM-L6-v2`；`retrieve_k=5`。
- provider replay：`m0_repeat_request_key_cassette`；随机 seed 按 freeze receipt 为 null。
- workload：LoCoMo `locomo:conv-26` 和 LongMemEval-S `longmemeval_s:e47becba` 各 1 个 focused contiguous-window history，32 个事件，显式 `deleted_source_ids`，每个 arm 1 个 query。
- M3：`conservative_full`，即完整保留历史的安全 checkpoint replay；这是强正确性基线，不是已证明的高效专用算法。
- 原始数据、trace、checkpoint、cost JSONL 和 provider receipt 保留在本地 `_q01_external`，不进入 source-only package、GitHub 或本文。

## 有效 v1 real-model receipt（修复前 source tree）

有效 receipt 位于：

```text
_q01_external/q01_freeze/formal_real_focused_window_review.json
_q01_external/q01_runs/formal_real_focused_window/
```

它在 UUID/source-lineage 修复之前完成，因此只能作为真实成本、上下文证据和失败诊断 receipt，不能作为“修复后通过”证据。

| 检查项 | LoCoMo | LongMemEval-S |
|---|---:|---:|
| workload completed | 是 | 是 |
| M0 provider cassette hits / misses | 99 / 0 | 93 / 0 |
| provider replay complete | 是 | 是 |
| M0 repeat visible equality | 否 | 否 |
| M0 repeat runtime equality | 是 | 是 |
| final context segments complete | 4/4 | 4/4 |
| query evidence complete | 4/4 | 4/4 |
| query isolated | 4/4 | 4/4 |
| observed deleted-source exposure | 0/4 | 0/4 |
| continuation checks passed | 1/4 | 1/4 |
| arm state exact-to-M0 | 0/4（M0 oracle 未稳定） | 0/4（M0 oracle 未稳定） |
| reviewer gate | `INCOMPLETE_OR_CONTROLLED` | `INCOMPLETE_OR_CONTROLLED` |

M0 visible mismatch 只报告了 source lineage keys：LoCoMo 8 个、LongMemEval-S 4 个；trace 中 reference rebuild/repeat 的输入 hash、输出 hash、decision 和 provider response sequence 完全一致。根因是 UUID lexical order 参与了 duplicate semantic output 的 logical slot 分配，同时 source `derived_from` 使用了 UUID-sorted object list。当前代码已改为 adapter creation order + semantic output key + duplicate occurrence ordinal，并把 logical-origin key 统一为 bare physical UUID。

### LoCoMo 成本

| arm/scope | wall time | LLM calls | input tokens | output tokens | memory reads | generated objects | index updates | replay events | checkpoint restore | embedding counters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| M0 full rebuild | 146.8 s | 99 | 88,150 | 11,618 | 30 | 71 | 31 | 31 | 0 | unknown |
| M1 tombstone/filter | 0.07 s | 0 | unknown | unknown | 0 | 0 | 6 | 0 | 0 | unknown |
| M2 replay/repair-all | 82.6 s | 72 | 60,601 | 6,592 | 23 | 43 | 23 | 23 | 1 / 110,919 B | unknown |
| M3 conservative full | 115.8 s | 96 | 77,260 | 9,332 | 30 | 57 | 31 | 31 | 1 / 625 B | unknown |
| four query probes | 4.4 s | 4 | 22,757 | 98 | 99 | — | — | — | — | 4 encodes / 99 rows / 38,016 values |

LoCoMo all-scope totals: 470 LLM calls, 329,282 input tokens, 36,308 output tokens, 243 memory reads, 298 generated objects, 154 index updates, 148 replay events, 2 checkpoint restores, 22,420 bytes read, and 111,544 checkpoint-restore bytes. `embedding_calls/items` remained unknown and are not treated as zero.

### LongMemEval-S 成本

| arm/scope | wall time | LLM calls | input tokens | output tokens | memory reads | generated objects | index updates | replay events | checkpoint restore | embedding counters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| M0 full rebuild | 167.9 s | 93 | 124,865 | 9,061 | 30 | 47 | 31 | 31 | 0 | unknown |
| M1 tombstone/filter | 0.10 s | 0 | unknown | unknown | 0 | 0 | 3 | 0 | 0 | unknown |
| M2 replay/repair-all | 101.1 s | 76 | 116,232 | 11,328 | 23 | 68 | 23 | 23 | 1 / 114,446 B | unknown |
| M3 conservative full | 102.5 s | 90 | 113,684 | 6,969 | 30 | 35 | 31 | 31 | 1 / 625 B | unknown |
| four query probes | 4.4 s | 4 | 15,991 | 28 | 110 | — | — | — | — | 4 encodes / 127 rows / 36,864 values |

LongMemEval-S all-scope totals: 455 LLM calls, 510,912 input tokens, 38,883 output tokens, 264 memory reads, 262 generated objects, 151 index updates, 148 replay events, 2 checkpoint restores, 100,192 bytes read, and 115,071 checkpoint-restore bytes. `embedding_calls/items` remained unknown and are not treated as zero.

These figures include lifecycle scopes, reference repeat, shared baseline, maintenance and isolated query probes. They do not provide a reliable currency amount because provider billing was not independently reconciled. The score encode/matrix work is explicit instrumentation cost and must remain in any later comparison.

## 状态、来源和 continuation 复核

- 两个 workload 的四个 arm 都有独立 query probe，probe isolation 和 query cost observation 通过。
- 最终 context reconstruction 与 pinned robust A-MEM renderer hash matched；四个 arm 的 `context_segments_complete` 均为 true。该结果说明 v1 的 structured lineage path 工作，但不等于 source deletion correctness。
- `deleted_source_exposed=0` 是已观察路径中的下界。因为本轮 M0 oracle 不稳定且 history 数量为 1，不能把它表述为零暴露证明。
- 只有 M0 的 continuation check 通过；M1–M3 在删除后继续执行的检查未通过。因而即使状态差异来自维护实现，也还不能声称未来运行状态等价。
- M1 的 0 LLM call 是弱 tombstone/filter 基线的成本现象，不应与 M2/M3 的 replay cost 直接解释为算法优势；M2/M3 的 exactness 在 M0 repeat 未通过时不具可解释性。
- M3 metadata 明确为 `strong_conservative_full_history_checkpoint_replay` / `complete_conservative_barrier`。它是公平的强通用 correctness baseline 配置，但这轮未证明其达到 M0 或产生可避免成本空间。

## v2 修复后尝试：无效 partial run

在修复迭代过程中按同一 freeze 启动了 v2；该子进程随后被 401 阻断，且它未形成包含最终 source-order 与 provider fail-fast 补丁的有效完成 receipt：

```text
_q01_external/q01_runs/formal_real_focused_window_v2/locomo/runner.stderr.log
```

provider 在首个 workload 尚未提交 completed instance 前持续返回 HTTP 401 `Authentication Fails`。因为官方 A-MEM 会吞掉 evolution exception，旧 wrapper 会把失败表现成“storing without evolution”；该路径已被当前 `_provider_errors` ledger + `refuse upstream fallback` 修复拦截。v2 没有形成有效 `manifest.json`/completed checkpoint/review，不能进入任何统计。它不是余额、模型质量或维护策略失败，而是认证环境阻断。

## 代码验收

```text
$env:PYTHONPATH=report/platform python -m pytest -q tests
139 passed
```

新增验收覆盖：

- source `derived_from` 不再依赖 UUID lexical order；
- duplicate semantic outputs 的 occurrence identity 跨独立 UUID rebuild 稳定；
- provider exception 被上游官方 fallback 吞掉时，wrapper 记录 error 并 fail-fast；
- 原有 freeze、M0 cassette、renderer lineage、checkpoint、state comparator 和成本契约全部保持通过。

## Q01 门槛与下一步

当前可以保留的工程结论：平台已经具备显式删除协议、串行 paired runner、独立 M0 repeat cassette、checkpoint replay、结构化最终上下文 lineage、生命周期成本 ledger 和 provider failure guard。

当前不能作出的科学结论：

- 不能判定 M1/M2/M3 的真实状态正确率；
- 不能比较任何维护策略优劣；
- 不能据此声称 deleted-source zero exposure；
- 不能把 conservative-full M3 称作已验证的精确依赖算法；
- 不能认定 Q01 已成立或进入新方法设计。

下一次有效运行的最小验收条件：

1. 使用有效 provider credential；runner 首个 provider error 必须 fail-fast，并留下 blocker receipt，不得写入完成 history。
2. 使用当前 source tree 与同一 freeze，LoCoMo/LongMemEval-S 各至少 1 个 history，串行 M0–M3。
3. 每个 history：M0 cassette hits = calls、misses = 0、`reference_repeat_equal=true`、`visible_equivalent=true`。
4. 每个 arm 的 final-context mapping complete；空 context 也必须明确标记 complete；deleted-source exposure 仅在 evidence complete 后报告。
5. 所有 continuation checks 通过，或明确缩小研究承诺为当前可见状态。
6. embedding、checkpoint、index、retrieval、query instrumentation 和 provider tokens 全部计入 scope，unknown 不填零。

在这些条件满足前，Q01 继续保持 **PROMISING / UNVERIFIED**，新算法设计暂停。
