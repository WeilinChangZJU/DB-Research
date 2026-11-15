# Q01 第五阶段：受控 v6 闭环与真实实验门槛（2026-11-15）

本记录对应当前 `report/platform` 源码树、官方 A-MEM commit
`f9a173dec325d84fcc97e7828775c7b241c33466`，以及冻结的 focused-window 输入。
它记录的是工程契约和受控机制验证，不把 scripted adapter 结果写成真实
DeepSeek 或真实 workload 的科学结论。

## 本轮固定配置

- 官方 A-MEM commit：`f9a173dec325d84fcc97e7828775c7b241c33466`。
- 每个 workload 使用冻结 focused-window JSONL 中的 1 条 history；当前冻结文件本身各含 1 条 history，不能把本轮称作多 history 发生率实验。
- 受控 runner：`run_q01_amem_controlled.py`。
- 受控模型策略：`context_sensitive_v1` scripted policy，覆盖 `NO_EVOLUTION`、`STRENGTHEN`、`UPDATE_NEIGHBOR` 和 `STRENGTHEN_AND_UPDATE`。
- 受控 embedding：明确标记为 `hash-v1`；这只用于确定性契约验证。
- `retrieve_k=5`，`m3_mode=recheck`，串行执行，启用显式可选的 `M_DAG_materialized_dependency_repair_all`。
- 真实冻结模型配置仍要求 DeepSeek endpoint、`deepseek-flash`、temperature 0.0、`sentence-transformers/all-MiniLM-L6-v2` local embedding 和统一的 request-key cassette；受控 hash embedding 不替代该真实配置。

## 当前源代码验收

在 `report/platform` 根目录运行：

```text
PYTHONPATH=. python -m pytest -q -p no:cacheprovider tests
146 passed
```

本轮覆盖了 request-key counterfactual cassette、M0 repeat、M3 recheck、最终上下文 lineage、来源证据、状态比较、多重性、可恢复 checkpoint 以及可选物化依赖 DAG baseline。M-DAG 默认关闭，只有显式 `--include-m-dag` 才会运行；其声明范围是 wrapper 可见的物化 source/object/candidate/support/control 边，不包含隐藏的未观测依赖。

## v6 受控结果

原始产物位于工作区 `_q01_external/round4_controlled_v6/`，不进入 GitHub 或 source-only 导师包。正式 reviewer 为 `review.json`，证据级别为 `controlled_scripted_official`，最终 decision 为 `INCOMPLETE_OR_CONTROLLED`。

| workload | M0 exact | M1 exact | M2 exact | M3 exact | M-DAG exact | query probes | lineage / evidence | deleted-source exposure |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| LoCoMo focused history | 1/1 | 0/1 | 1/1 | 1/1 | 1/1 | 5 | 5/5 complete | 0 |
| LongMemEval-S focused history | 1/1 | 0/1 | 1/1 | 1/1 | 1/1 | 5 | 5/5 complete | 0 |

M1 的非等价是当前保守 tombstone/filter 语义的预期结果，不能直接解释为 Q01 现象。M2、M3 和 M-DAG 的 exact-to-M0 结果只证明在这个确定性 controlled builder 和已观测依赖范围内，当前状态导出、checkpoint、重执行和比较器彼此一致；它们不证明真实模型下的强基线优越性。

每个 workload 的 5 个 query probe 均满足：`evidence_complete=true`、`context_segments_complete=true`、`query_isolated=true`、`query_cost_observed=true`，且没有观察到已删除来源暴露。查询 probe 的 `continuation_checks_passed=0`，因为本轮 focused controlled pilot 没有执行删除后的公共后缀 continuation；因此不能宣称未来运行状态等价已经被验证。

### 全成本账本

以下是 v6 `cost_summary.json` 的完整 scope totals；logical/oracle cassette 命中和实际 provider 计数分开。受控 provider attempts 是 scripted provider 边界，不是 DeepSeek 账单。

| workload | logical LLM calls | provider attempts | provider input/output tokens | logical input/output tokens | oracle cassette hits | embeddings (calls/items) | replayed / suffix events | checkpoint restore events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LoCoMo | 527 | 145 / 145 | 86,136 / 3,311 | 290,994 / 13,079 | 382 | 363 / 363 | 175 / 81 | 3 |
| LongMemEval-S | 519 | 150 / 150 | 231,011 / 3,287 | 722,909 / 12,297 | 369 | 377 / 377 | 171 / 77 | 3 |

`provider_attempts`、`provider_input_tokens` 和 `provider_output_tokens` 是受控 provider 实际边界计数；`logical_*` 包含共享 response cassette 的重放范围。两者不能相加，也不能据此估算真实 DeepSeek 费用。真实实验还必须把 trace/checkpoint、retrieve、generate、embed、index、validate/publish 和后续 serving 成本纳入生命周期账本。

## 真实模型门槛

本轮重新执行了严格 formal preflight，使用冻结输入、官方 commit、DeepSeek endpoint、`deepseek-flash`、local MiniLM embedding、`retrieve_k=5`、`m3_mode=recheck` 和可选 M-DAG。当前 Windows 进程没有 `OPENAI_API_KEY`，因此 preflight 写出：

```text
_q01_external/round4_real_preflight3/formal_run_blocker.json
status = blocked_missing_external_model_state
message = OPENAI_API_KEY is not configured; no real-model call was started
```

没有启动任何真实模型调用，也没有使用历史 WSL `.env` 中的旧凭据。该 blocker 是当前外部环境状态，不是维护策略失败。拿到当前有效 credential 后，必须用同一 freeze 和当前 source tree 重新运行；不能把旧版本 real receipt、受控 v6 receipt 或 401 partial receipt 混合成当前结果。

## Q01 当前结论

当前结论仍为：

> **PROMISING / UNVERIFIED。**

本轮已经把测量器、response cassette、最终上下文 lineage 和显式 materialized-DAG baseline 进一步校准，并证明受控确定性闭环可以通过状态、来源和成本账本验收。但由于没有真实 DeepSeek call，且没有 continuation 后缀检查，当前结果不能支持“某个维护方法优于其他基线”、不能支持真实 workload 发生率，也不能支持新算法的创新性主张。

下一阶段的唯一有效实验入口是当前 formal runner：在有效 credential 下完成两个 workload 的最小真实 paired M0–M3（可选 M-DAG）运行，确认 M0 repeat、最终上下文 lineage、continuation、全成本和独立 reviewer 全部通过；只有随后才有资格扩大 history 数量、估计自然发生率并讨论新方法或收窄/关闭 Q01。
