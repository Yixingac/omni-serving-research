# Part 1 审计：本轮开始时的研究基线（2026-11-18 22:02）

> 详细的五类划分、实验 6–11 逐个核对、“知道 / 不知道 / 有条件”三张表在上一轮的 `../research_baseline.md`（11-18 16:32，提交 `ade6a17`），本文只写本轮开始时重新核对过的状态和变化，不重复。

## 1. 版本和环境核对

| 项 | 核对结果 | 怎么核对的 |
|---|---|---|
| 分支最新提交 | `4e0f105`（= `origin/research/agent-trace-end-to-end`，22:01 `git fetch` 后核对）；本轮分支 `research/pre-presentation-validation` 从这里开出 | `git rev-parse HEAD origin/...` |
| 集群上编排器用的检出 | `5c1be74`（比 `4e0f105` 早 12 个提交）；这 12 个提交只改了文档、分析脚本和测试，`configs/`（除一个 `resource_plan.md` 说明）、`src/`、`scripts/slurm/`、`scripts/mac/` 都没变 | `git diff --stat 5c1be74 4e0f105 -- configs src scripts/slurm scripts/mac` |
| 本轮代码在集群上从哪里跑 | 新建 worktree `~/workspace/omni-ppv`（本分支）；编排器的命令先 `cd` 到这里再运行，所以运行器、配置都来自本分支，原检出不动 | `git worktree list`；每个运行的 `manifest.json` 记录仓库提交 |
| sglang-omni 运行时 | 仍是 `520b49ee` 的各个 worktree（stock、instr、batch、async 等），本轮没有改动已有 worktree；新补丁一律放在新的 `ppv-*` worktree | — |
| 集群上原检出里有一个未跟踪文件 `4d9fb37` | 不是本轮产生的，没有动 | `git status --short` |
| Job 454290 | RUNNING，UNLIMITED，compute-03，8×A6000，属于本人；22:02 时 8 张卡 0%、1 MiB、无进程；登录节点上的编排器（`--keep-alive`）还在轮询 plan.d，115 个实验全部 SUCCEEDED，没有排队或运行中的实验 | `squeue --me`、`scontrol show job`、`nvidia-smi`、编排器 `state.json` |
| 资源余量 | 节点内存可用约 786 GB；`/scratch` 97% 满（剩 219 GB）；`/tmp` 剩 2.8 GB；NFS home 剩 5.7 TB | `free -g`、`df -h` |
| 驱动 / CUDA / torch | 580.126.18 / CUDA 13.0 / torch 2.13.0+cu130（与实验 6–11 相同） | `nvidia-smi`；运行时 venv |

## 2. 哪些已经做过、哪些只是 CPU 分析、哪些登记过、哪些没跑

| 内容 | 状态 | 依据 |
|---|---|---|
| 实验 1–14、18、19 | 已运行，有报告（部分单次运行，报告标“初步”） | `research/实验总目录.md` |
| 实验 15–17 | 没有开始，没有数据 | 同上 |
| `next_steps/results/` 的 8 个 JSON（kernel 普查、talker 名额、停顿对齐） | **只是 CPU 离线分析**，用的是实验 7、9 已有的日志和 trace | `../evidence_index.md` S6–S11 |
| P0、P1、P2、P3 | 事先登记过（`../p0_p3_experiment_plan.md`，11-18 16:32），**GPU 一次都没跑** | 编排器里没有对应实验 |
| R0–R4 | 设计写过（`../robotics_experiment_plan.md`），R0 的通过条件登记过；**没有跑过**，全双工路径在本机从没启动过 | 同上 |
| C1（flow 打包）、C2（HiFT） | 门槛写在 `../compiler_runtime_opportunities.md` 第 4 节和 `../research_priorities.md`；**没有跑过** | 同上 |
| 实验 11 旁路复制 | 正确性门槛**未通过**，性能对比没做；本轮 P2 不改写这个结论 | `research/01_性能瓶颈/实验11_Thinker异步优化/README.md` |

## 3. 哪些结果可以作为报告证据

可以直接引用（带条件）：实验 1–14、18、19 报告里标为 Measured/Derived 的结论，条件见 `../research_baseline.md` 3.3；`next_steps/results/` 的 CPU 分析结果（Measured，但注意启发式：talker 行按 token/请求区分；kernel 归属按名字）。

不能作为证据：任何 P0–P3、R0–R4、C1、C2 的预期收益（都只是 Estimated 或 Hypothesis），直到本轮的运行完成；实验 11 的旁路复制；把 profiler 窗口里的 kernel 份额当成全时段数据；把 CUDA graph 的 kernel 时间份额当成 graph 命中率。

## 4. 发现的登记问题（Erratum）

| 编号 | 问题 | 处理 |
|---|---|---|
| E1 | P0 登记的 W3b 开环速率 {18 … 31} 全部高于方案 A 的已知容量（17.75，实验 9），测不出 A 的稳定容量 | 不是判定规则的错误（H0a/H0c 用闭环容量）；执行前加一档 15.5（只加不删），记在 `preregistration.md` P0 补充 2 |
| E2 | P0 登记要在 talker 上开 3 秒 torch profiler，但实验 9 的打点版运行时不能只给 talker 开，code2wav 在 profiler 下会崩溃 | 不开；每步时间改用 SGLang 解码日志，记在同一处 |
| E3 | P0 的方案 B 同时改了准入名额和 CUDA graph 覆盖范围（`cuda_graph_max_bs` 默认等于名额），原登记写了“graph 自动录到 64”但没有对照 | 新增方案 G（名额 64、graph 32），只做分解描述，不改 H0a–H0c |
