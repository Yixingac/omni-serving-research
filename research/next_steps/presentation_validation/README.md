# 报告前的最后一轮实验验证（presentation validation）

> 分支 `research/pre-presentation-validation`，父提交 `4e0f105`（2026-11-18 22:01 的 `origin/research/agent-trace-end-to-end`）。
> 运行环境：Job 454290（compute-03，8×RTX A6000，UNLIMITED），sglang-omni `520b49ee`，SGLang 0.5.21，torch 2.13.0，MiniCPM-o 4.5（revision `fb09628e`）。
> 本文件是本轮的唯一入口；状态在实验结束后逐条更新，**只有真正跑完并通过检查的才写 COMPLETED**。

## 1. 本轮实验状态

| 优先级 | 实验 | 状态 | 实际 GPU-时 | 关键结果 | 登记的成功条件是否满足 |
|---|---|---|---|---|---|
| 1 | R0 全双工功能验证 | 准备中 | — | — | — |
| 2 | R1 单会话关键路径 | 等 R0 | — | — | — |
| 2 | R2 并发会话 | 等 R0 | — | — | — |
| 3 | P0 talker 名额 32→64 | RUNNING（区组 1） | — | — | — |
| 4 | C1 flow 稠密 vs 打包 | 准备中 | — | — | — |
| 5 | P1 饱和区的按长度组批 + 最长等待 | 准备中 | — | — | — |
| 6 | C2 HiFT 微基准 | 准备中 | — | — | — |
| 7 | R3 持续图像输入 | 准备中 | — | — | — |
| 8 | R4 Agent 工作集 | 准备中 | — | — | — |
| 9 | P2 隐藏状态逐位检查 | 准备中 | — | — | — |
| 10 | P3 停顿观测 | 随所有运行采集（压力指标、进程状态） | 0（不单独占 GPU） | — | — |

状态取值：COMPLETED / FAILED / BLOCKED / NOT RUN / NOT APPLICABLE / RUNNING / 准备中。

## 2. 预算（开始前计算）

沿用仓库已有估计（`../p0_p3_experiment_plan.md`、`../robotics_experiment_plan.md`、`../research_priorities.md`），均为 Estimated：

| 实验 | 估计 | GPU-时（估计） |
|---|---|---|
| P0（完整版：A、B、G、参照 × 3 区组） | 约 4.2 h × 8 卡 | 约 34 |
| R0 | 约 1 h × 1 卡 | 约 1 |
| R1 | 约 1.5 h × 1–8 卡 | 约 2–12 |
| R2 | 约 3 h × 1–2 卡 | 约 3–6 |
| C1 微基准 | 约 0.5 h × 1 卡 | 约 0.5 |
| C2 微基准 | 约 0.5 h × 1 卡 | 约 0.5 |
| P1 | 约 3.5 h × 8 卡 | 约 28 |
| R3 | 约 2 h × 8 卡 | 约 16 |
| R4 | 约 2.5 h × 8 卡 | 约 20 |
| P2 正确性（通过后性能对比另约 3.8 h × 8 卡） | 约 0.4 h × 8 卡 | 约 3（+30） |
| P3 本地盘 vs NFS 对照（有剩余预算时） | 约 2 h × 8 卡 | 约 16 |
| **合计** | 8 卡串行部分约 14 h + 单卡部分约 2–3 h（可并行），P2 性能对比另 3.8 h | **约 125–160** |

实际可用时间：Job 454290 没有时间上限（用户自己的 hold，开始时 8 张卡空闲、编排器空闲）；所以限制是串行的墙钟时间，而不是作业剩余时间。执行顺序见第 3 节；每个实验跑之前先满足它的 Go/No-Go 条件。

## 3. 执行顺序（按 GPU 形状安排，不改变优先级）

1. P0 区组 1（8 卡）——配置只改一个字段，最先就绪；同时并行做其他实验的 CPU 准备。
2. 单卡阶段：R0 →（通过后）R1 → R2 在一张卡上依次做；C1、C2 微基准在另外的卡上同时做。
3. P0 区组 2、3（8 卡）。
4. P1（8 卡，C1 通过时加打包 flow 组）、R4、R3、P2、P3 对照（有剩余时）。

## 4. 文件

| 文件 | 内容 |
|---|---|
| `audit.md` | Part 1 审计：开始时的版本、环境、哪些做过、Erratum |
| `preregistration.md` | 本轮对原登记的补充（每节在对应实验第一次运行前提交） |
| `experiment_manifest.md` / `.json` | 每个尝试（含失败）的配置、提交、Slurm step、起止、GPU-时、是否纳入分析 |
| `final_experiment_results.md` | 实际完成的实验结果 |
| `figures/` | 报告用图（`scripts/analysis/ppv_figures.py` 重新生成） |
| `presentation_evidence.md` | 每个报告结论 → 实验、数字、图、来源、限制 |
| `presentation_narrative.md` | 约 10 页的研究叙事 |
| `limitations_and_negative_results.md` | 限制和负面结果 |
