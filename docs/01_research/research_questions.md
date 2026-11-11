# 研究问题

每个问题后面写了打算怎么测，以及当前状态。状态只有在有真实测量后才会改为“有初步数据”。

| 编号 | 问题 | 怎么测 | 状态 |
|---|---|---|---|
| Q1 | Omni 模型推理时，一个请求的时间分布在哪些阶段？ | SGLang-Omni request event recorder 的阶段事件，按 request_id 与客户端时间对齐（x8、x5 的事件阶段） | 有初步结果，见 `docs/05_findings/research_conclusions.md` |
| Q2 | 真实 Agent 负载与常规压测有什么不同？ | 带工具调用的 Agent 采集 trace（x5），统计请求间隔、依赖链长度、模型/工具/思考时间占比 | 有初步结果，见 `docs/05_findings/research_conclusions.md` |
| Q3 | 哪个阶段在何种负载下成为延迟或吞吐瓶颈？ | 单卡上 W1/W2/W3/W3b/W4 的并发扫描（x2–x4），对照阶段分解 | 有初步结果，见 `docs/05_findings/research_conclusions.md` |
| Q4 | 什么时候 CPU 预处理让 GPU 空闲？ | 视频时长 2/5/10/20 秒的 W4 扫描（x4）；CPU 采样与 GPU 采样对齐；之后用 Nsight 看 kernel 间隙 | 有初步结果，见 `docs/05_findings/research_conclusions.md` |
| Q5 | 什么时候阶段速度不匹配导致背压？ | 语音输出负载下 Talker 前的排队时间随并发的变化（x3、x8） | 有初步结果，见 `docs/05_findings/research_conclusions.md` |
| Q6 | 什么时候调度或 batching 效率下降？ | 并发升高时每个请求的 decode 速度、首 token 时间的变化；后续结合 Nsight | 有初步结果，见 `docs/05_findings/research_conclusions.md` |
| Q7 | 什么时候 GPU 放置或拆分部署有收益？ | 2 卡预算下：两个完整副本 vs thinker/talker 拆分 vs 只复制 talker（x7a/b/c） | 有初步结果，见 `docs/05_findings/research_conclusions.md` |
| Q8 | 什么时候阶段间通信抵消拆分的收益？ | 拆分部署时的阶段间传输事件（hop）耗时；有 NVLink 与无 NVLink 的 GPU 对比（需要更多卡和拓扑信息） | 有初步结果，见 `docs/05_findings/research_conclusions.md` |
| Q9 | 多个 Agent 并发时系统如何变化？ | Agent trace 的依赖感知回放，会话复制 1/4 倍、Poisson 到达（x6） | 有初步结果，见 `docs/05_findings/research_conclusions.md` |
| Q10 | 哪些优化方向有可重复的数据支持？ | 以上结果出来后再定，每个方向配一个对照实验 | 部分有数据支持，见 `docs/05_findings/research_conclusions.md` |

## 需要导师确认的假设

1. **“Agent traces”指什么。** 我们目前理解为应用层的 Agent 执行轨迹：会话、步骤、模型请求、工具调用和它们的依赖关系。另一种可能的理解是 Serving 系统内部的请求轨迹，或者来自某个公开 Agent 数据集的 trace。三层 trace 的基础设施都已实现，理解改变时不需要推倒重来。
2. **用合成任务代替真实 Agent 应用。** 现在的 Agent 做的是可自动判分的合成任务（算术、单位换算、库存查询），工具在本地执行，可以加入模拟的远程延迟（明确标注为合成）。它能产生真实的“模型请求—工具—模型请求”依赖结构，但任务内容和工具延迟分布不代表任何真实产品。
3. **模型选择。** PR #1 建议先用 MiniCPM-o 4.5，Qwen3-Omni 作对照。MiniCPM-o 4.5 约 9B 参数，单张 A6000 能放下，适合多卡放置实验；Qwen3-Omni BF16 需要至少 3 张 A6000，放置空间小，作为第二个模型。

## 暂不研究

- 全双工（`/v1/realtime`，模型边听边说）：SGLang-Omni 对 MiniCPM-o 4.5 支持，但需要 WebSocket 客户端和单独的测量方法。先完成请求-响应式负载，再评估。
- 跨节点分布式推理：Job 409712 只有一个节点；集群节点之间没有发现 InfiniBand，10 GbE 的跨节点通信不适合当作节点内 PCIe 的替代。
