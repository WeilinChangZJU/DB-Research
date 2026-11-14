# Bemula ResearchProtocol 适配开发方案

日期：2026-11-13
范围：P0 工程修复、ResearchProtocol 基础层、LoCoMo/LongMemEval-S 数据适配；不恢复完整实验。

## 目标

保留 Bemula 的 control/forget 行为评测内核，同时增加可审计、可恢复、可扩展的研究实验层。第一阶段产物必须支持：

1. 数据、代码、配置、提示词、依赖、运行环境和硬件的 provenance manifest；
2. 原子 per-instance checkpoint、完成清单和安全 resume；
3. 统一 ResearchProtocol / MemoryBackend 扩展点；
4. LoCoMo 与 LongMemEval-S 的离线转换和 source provenance；
5. 离线单元测试与 adapter fixture；
6. 不超过 1 个实例的 DeepSeek smoke test，验证 manifest/checkpoint/adapter 链路，不运行完整数据集。

## 分层设计

Bemula core 保留 schemas、runner、judge、metrics 和 baseline adapters。新增 bemula.research：

- manifest.py：hash、runtime、hardware、dependency、prompt/data/config 清单；
- checkpoint.py：原子写入、完成标记、完整性检查、安全 resume；
- trace.py：LLM/memory operation 事件、latency/token/cost/error hook；
- backend.py / protocol.py：MemoryBackend、SourceRecord、snapshot/diff、forget/delete/update/introspect/explain 可选能力；
- datasets/locomo.py：LoCoMo raw -> ResearchInstance/ResearchItem；
- datasets/longmemeval.py：LongMemEval/LongMenEval-S raw -> ResearchInstance/ResearchItem；
- tests/：全部离线测试，不触发外部 API。

## P0 修改顺序

1. 先新增研究层模块和离线 fixtures，不改现有实验结果；
2. 加入运行 manifest 与 completed-instance manifest；
3. 将研究 runner 设计为新入口或显式 opt-in，不改变旧 run_eval 默认输出；
4. dataset adapter 完成后运行 schema/fixture tests；
5. 使用 quick_check 的 1 个实例做 DeepSeek smoke test，验证真实 chat/judge 只调用少量 API；
6. 记录 smoke receipt，确认不会启动 lite/full 实验；
7. 冻结变更清单、测试结果和后续待办。

## 完成标准

- 所有离线 tests 通过；
- hash 在同一输入上稳定，输入变更可检测；
- 中断/部分 JSONL 不会被误判为已完成；
- LoCoMo/LongMemEval-S fixture 能保留 source_id、source_hash、turn/span metadata；
- ResearchProtocol reference backend 能完成 observe/query/snapshot/diff/forget；
- smoke test 只运行 1 个实例并生成 manifest、checkpoint、metrics/report；
- 完整实验仍保持未启动状态。
