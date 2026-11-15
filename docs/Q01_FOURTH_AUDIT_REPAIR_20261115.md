# Q01 第四轮审计修复与受控复核（2026-11-15）

本记录对应第四轮审计后的当前源码树。审计包中的失败测试被视为待修复契约；其中两个“诊断测试”原本故意断言旧缺陷，修复后应当反转：空删除下所有维护臂必须共享同一个反事实响应函数，M3 recheck 必须与 M2 产生不同的 memo 行为。

## 已完成的 P0 修复

- M0 reference、M0 repeat、common baseline 和 M1–M3 现在使用同一个 request-keyed counterfactual response cassette。provider ledger、method memo 和 oracle cassette 命中分开记录；空删除不会因为串行 provider 响应而产生假差异。
- 请求键包含 prompt/messages、位置参数、temperature、max_tokens、model/provider identity、endpoint、seed、tools、response format、stop 等响应相关字段。直接使用 AmemResearchWrapper 默认不启用 memo；AmemHistoryBuilder 才按显式 policy 启用 method cache。
- provider 使用量与 cache 命中分开计数。cache-only 操作的 provider token 是已知零；实际 provider 调用却没有 usage 时仍保持 unknown。
- M3 recheck 从真实 checkpoint 恢复运行状态，并导入完整旧 baseline 的授权 response memo。它重新执行当前写逻辑；上下文改变的请求不会误命中旧 response。
- 逻辑对象身份不再把 stock A-MEM 的物理 link position 写入永久 origin key。关系仍在 canonical snapshot 中单独保留。
- fork 使用 deepcopy memo 共享不可复制的 provider/embedder handle，复制可变 retriever/index 状态；不会因复制外部句柄而失败。
- completion fallback 的默认 temperature 固定为 0.0；controlled loader 保留顶层 speaker/session 字段。
- 结构化 renderer 在重复内容导致来源歧义时 fail-closed；不能仅凭相同渲染字节认证错误来源。
- 版本化依赖闭包不再把旧版本消费者错误回溯为当前删除的 downstream；显式 logical root 仍保持兼容展开。

## 本地验收

当前 source-only 树运行：

```text
PYTHONPATH=report/platform python -m pytest -q -p no:cacheprovider report/platform/tests
139 passed
```

新增 tests/test_q01_round4_repairs.py 4 项均通过。第四轮独立契约测试中，13 项可执行契约有 12 项通过；余下 2 项是旧的诊断断言，分别要求 M2/M3 继续保持相同工作、以及要求空删除产生 provider-world 差异，二者正是本轮修复目标，修复后的正确断言已经加入 source tests。

## 官方 A-MEM 受控串行闭环

在不调用外部 provider 的条件下，使用官方 A-MEM commit f9a173dec325d84fcc97e7828775c7b241c33466、hash embedder、focused LoCoMo/LongMemEval-S history 各 1 条、retrieve_k=5、m3_mode=recheck 运行了完整串行 M0–M3：

- round4_controlled_v4/locomo：completed，context-sensitive scripted policy 实际触发了 NO_EVOLUTION、STRENGTHEN、UPDATE_NEIGHBOR 和 STRENGTHEN_AND_UPDATE；M0 repeat exact/visible/runtime 均通过；M1 保持保守 tombstone 的预期非等价；M2/M3 exact-to-M0 通过；4/4 query evidence complete、isolated 且 query cost observed。
- round4_controlled_v4/longmemeval_s：同样完成了四类 branch；M0 repeat exact/visible/runtime 通过；M2/M3 exact-to-M0 通过；4/4 query evidence complete、isolated 且 query cost observed。
- 受控结果只证明 wrapper、状态比较、checkpoint/recheck、上下文敏感决策分支和来源证据路径的契约，不是 DeepSeek 质量或真实 workload 发生率结果。LoCoMo 全 scope ledger 为 434 logical LLM requests、144 provider attempts、240,497 logical input tokens、10,767 logical output tokens、86,136 provider input tokens、3,311 provider output tokens、300 embedding calls/items；LongMemEval-S 为 427 logical requests、149 provider attempts、598,784 logical input tokens、10,109 logical output tokens、231,011 provider input tokens、3,287 provider output tokens、314 embedding calls/items。logical totals 包含 repeat/oracle cassette usage；provider totals 是实际到达 provider boundary 的 usage，受控结果不应当冒充 DeepSeek billing。

受控产物保留在工作区 _q01_external/round4_controlled，不进入 GitHub 或 advisor source-only package。

## 真实模型门槛与 Q01 结论

第四轮修复本身不等于真实研究结论。当前 Windows 环境没有可验证的 provider credential；WSL 另有历史 .env，但此前同类 post-fix 尝试收到 401，不能把配置文件存在当作可用凭据。因而本轮没有再次消耗 DeepSeek API，也没有把旧的 v1 real receipt 与当前 source 混合。

当前源码下的 formal preflight 已显式写出
`_q01_external/round4_real_preflight2/formal_run_blocker.json`：冻结的
DeepSeek endpoint 已对齐，但 `OPENAI_API_KEY` 未配置，因此没有启动任何真实
model call。这个 blocker 是环境状态证据，不是维护策略结果。

真实 scientific gate 仍要求：同一 freeze、当前 source tree、两个 workload 各至少 1 条 history、M0 cassette/repeat visible equality、完整 final-context lineage、continuation 检查和全生命周期成本均通过。旧 v1 receipt 仍只是修复前 UUID/source-lineage 问题下的真实成本诊断；v2 的 401 partial 仍排除。

因此 Q01 当前结论保持：

> **PROMISING / UNVERIFIED。** 平台已具备继续做最小真实闭环的条件，但尚无证据证明某个维护方法优于其他基线，也尚无依据进入新算法设计。下一步应在有效 credential 下用当前源码重跑同一 focused freeze；若 M0 稳定、M-DAG/M2/M3 正确且全成本仍有可避免空间，才进入机制发生率和方法设计阶段。
