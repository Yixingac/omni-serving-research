# 下一阶段研究路线图（统一入口）

> 2026-11-18。这个目录回答一个问题：**在实验 1–19 之后，哪些系统问题真正值得继续研究，怎么做。**
> 实验编号和每个实验的状态仍以 [`research/实验总目录.md`](../实验总目录.md) 为准；本目录只放下一阶段的审计、计划和研究材料，不改已有实验的报告和数据。
> 本轮没有运行任何新的 GPU 实验；新增的实测数字全部来自已有运行的日志和 trace（CPU 离线分析），见 [`data/`](data/)。

## 怎么读

| 顺序 | 文件 | 内容 | 语言 |
|---|---|---|---|
| 1 | [research_baseline.md](research_baseline.md) | Part 0：已有实验的五类状态、实验 6–11 逐个核对、已知/未知/有条件的结论、本轮新增证据 | 中文 |
| 2 | [research_priorities.md](research_priorities.md) | 最值得做的 8 个任务、依赖、成功条件、GPU 估计、需要你决定的事 | 中文 |
| 3 | [p0_p3_experiment_plan.md](p0_p3_experiment_plan.md) | P0 talker 名额、P1 组批 + 最长等待、P2 逐位正确性、P3 短暂停顿；事先登记的假设和判定 | 中文 |
| 4 | [thunderagent_flashstates_related_work.md](thunderagent_flashstates_related_work.md) | 两篇工作的机制、源码、实测条件；我们的服务暴露了什么；系统级对照表；3 个假设 | 中文 |
| 5 | [robotics_application_intro.md](robotics_application_intro.md) | **组会报告的权威材料**：应用、架构（Mermaid）、5 个发现、研究空白、3 个 RQ、评测方法、证据表、10 页 PPT 框架、90 秒和 3–5 分钟口头介绍、预期问答 | 英文（第 11 节中文） |
| 6 | [robotics_experiment_plan.md](robotics_experiment_plan.md) | 机器人负载和评测：R0 全双工功能验证 → R4 工作集测量 | 中文 |
| 7 | [compiler_runtime_opportunities.md](compiler_runtime_opportunities.md) | 主线 C：热点清单、kernel 普查、Compiler 和 Runtime 候选表、三个重点候选、优先级、两个推荐方向 | 中文 |
| 8 | [evidence_index.md](evidence_index.md) | 每个关键结论 → 代码、报告、数据、外部来源和证据等级 | 中文 |

## 一页总结

**已确认（有实测或源码证据）**
- 语音负载下，决定吞吐和延迟的是 talker 和 code2wav，不是 thinker；8 卡里 2+2+4 最好（实验 7、9）。
- 2+1+5 变差时，唯一的 talker 在 80–83% 的时间里正好 32 个请求在跑、约 180 个在排队，卡只忙 65%——软件准入上限在起作用（本轮从实验 9 日志补的证据；因果还要 P0 验证）。
- code2wav 最大的单个 kernel 是带 padding mask 的 mem-efficient 注意力（19%），加上准备 kernel 共 25%；HiFT 全程 FP32；CUDA graph 在并发 128 时只占 0.3%（本轮 kernel 普查）。
- 当前 chat 路径上语音一次返回，首段音频 ≈ 端到端（低负载约 2 秒）；全双工路径没跑过。
- 停顿 2 是全进程同时停住（日志 3.09 s 空白、CPU 接近 0），停顿 1 停在 talker → code2wav 的交接（本轮对齐已有数据）。

**最值得做的前四件**（都需要你批准 GPU 预算）：P0（约 1.6 h × 8 卡）→ R0 全双工功能验证（约 1 h × 1 卡）→ flow 稠密 vs 打包的微基准（约 0.5 h × 1 卡）→ R4 机器人会话工作集测量（约 2.5 h × 8 卡）。

## 每个方向的 8 个问题（简版）

P0–P3、R 主线的完整版在各自文档开头；这里给 C 主线和 Agent 结合方向的简版，其余一行概括。

### P0 talker 名额（详见 `p0_p3_experiment_plan.md`）
解决 2+1+5 为什么输给 2+2+4；GPU 只忙 65% 却排队 8 秒；实验 9；SGLang 的名额 + KV 准入；只改名额、3 个区组；登记预测 +10–20%；不涨就说明是每步开销；值得，最便宜。

### P1 组批 + 最长等待
局部的 +30% 能否变成端到端；补齐 28%；实验 6、10；长度分桶 + 老化是常见手段；2+2+4 饱和区开环；上限是 code2wav 饱和时的 +11–30%；没收益说明瓶颈不在 code2wav；值得做一次。

### P2 逐位正确性
旁路复制对不对；指纹检查分辨率不够；实验 8、11；影子执行；同一运行里保存两份逐位比；正确性本身无收益；失败能定位到具体请求；值得做检查本身。

### P3 短暂停顿
两次 2–5 秒停顿拉高尾延迟；GPU 全空无报错；实验 9；PSI、抓栈；本地盘 vs NFS 对照 + 采样器；不影响吞吐、影响 P99；可能是测量方法的问题；值得，低成本。

### R 机器人主线（详见 `robotics_experiment_plan.md`）
实时交互的首段音频和打断；chat 路径首段音频 ≈ 端到端；实验 6、9、14、19；原生全双工、OmniCast、Metronome、VoxServe；先功能验证再扫会话数；数字未知；失败也能说明单卡多阶段的限制；值得。

### C 推荐 1：flow 的变长打包
1. **问题**：code2wav 的 GPU 时间有多少花在补齐和带 mask 的低效注意力上，能不能去掉。
2. **为什么**：最大的单个 kernel 是 mem-efficient 注意力（19%，占用率 33%）+ 准备 kernel 6%；长短混合时补齐 28%。
3. **哪个实验**：本轮 kernel 普查（实验 7 的 trace）；实验 6、10。
4. **别人怎么做**：变长（varlen）FlashAttention 打包是常见做法；sglang-omni 自己就有一条默认关闭的打包路径，上游注释说它（eager 时）比编译过的稠密路径慢。
5. **怎么验证**：先零代码 A/B（稠密 / 打包 / 打包 + 编译）的微基准，用实验 10 记录的真实批形状；过了再做端到端。
6. **收益**：上界约为补齐比例（W3b 15%、MIX 28%）加注意力 kernel 的效率差（Estimated），和按长度组批的收益重叠。
7. **结果不好**：说明 sm_86 上 varlen 路径不划算，再决定是否写 Triton 变长注意力。
8. **值得吗**：值得，证据最强、起步成本最低，而且是最适合 GPU compiler 背景的题目。

### C 推荐 2：HiFT 的编译期优化
1. **问题**：HiFT 的 FP32、weight-norm 重算、布局转换浪费多少。
2. **为什么**：TF32 卷积 10%、`nchwToNhwc<float>` 3%（4.9 万次）、82 个 weight-norm 每次重算。
3. **哪个实验**：本轮 kernel 普查；实验 6（HiFT 约 45 ms/请求）。
4. **别人怎么做**：上游 issue #2399 在 H20 上把 code2wav 改 FP16 + compile 后吞吐约 2.4 倍（不同版本和 GPU，只说明方向）；weight-norm 折叠是声码器推理的常规做法。
5. **怎么验证**：三步各自微基准（折叠、channels-last、半精度），半精度要过客观音质门槛。
6. **收益**：code2wav GPU 时间约 −7..−12%（Estimated）。
7. **结果不好**：半精度过不了音质门槛就只保留前两步。
8. **值得吗**：值得，但排在推荐 1 之后（音质风险高）。

### Agent 结合方向（ThunderAgent / FlashStates，详见相关工作文档）
1. **问题**：Agent 的会话和步骤信息能不能改善多阶段 omni 的准入、放置和缓存。
2. **为什么**：负载组成决定瓶颈（实验 19）；轮转绑定会抵消会话的前缀缓存（源码）。
3. **哪个实验**：实验 12、19（合成 Agent，会话短、工具微秒级）。
4. **别人怎么做**：ThunderAgent 的程序级准入和暂停/恢复；FlashStates 的按代价驱逐（只对混合模型有完整意义）。
5. **怎么验证**：先测“多模态会话的工作集 / KV 容量”（R4）；再测下游感知的准入（假设 A）。
6. **收益**：两篇论文的收益只在工作集超过容量时出现；我们现在没到这个规模，收益未知。
7. **结果不好**：证明对这个模型和 48 GB 卡不需要会话级缓存管理——也是有用的边界。
8. **值得吗**：值得先做测量（低成本）；不值得现在实现调度器。

## 本轮新增的代码和数据

| 文件 | 用途 | 测试 |
|---|---|---|
| `scripts/analysis/next_talker_slots.py` | 从 SGLang 调度日志读 talker 的在跑数、排队、graph、每步时间曲线（P0 的分析工具） | `tests/test_next_talker_slots.py` |
| `scripts/analysis/next_c2w_kernel_census.py` | code2wav 的 kernel 普查（按名字、类别、子模型） | `tests/test_next_c2w_kernel_census.py` |
| `scripts/analysis/next_stall_alignment.py` | 把一次停顿和服务日志、每进程 CPU、code2wav 批对齐（P3） | `tests/test_next_stall_alignment.py` |
| `data/p0_*_talker_slots.json` | 实验 9 的 talker 名额证据 | — |
| `data/c2w_kernel_census_*.json` | 实验 6、7 的 kernel 普查 | — |
| `data/p3_stall*.json` | 两次停顿的对齐 | — |

数据文件只含汇总数字和运行目录名，不含主机名和用户路径。
