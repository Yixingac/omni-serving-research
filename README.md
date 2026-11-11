# Omni Serving Research

研究 Omni 模型（同时处理音频、视频、文本输入，并能输出文本和语音的多模态模型）在真实 Agent 负载下的 Serving 行为：时间花在哪个阶段，GPU 什么时候空闲、为什么空闲，多张 GPU 怎么放置各个阶段更划算。

> **当前状态（2026-11-11）**：基础设施已实现并有测试；第一批 A6000 实验已排队（Slurm Job 409712，3×RTX A6000），还没有正式测量结果。本 README 只把已经实现的功能写成“已实现”，其余都标为计划。实际进度见文末“研究进展”。

## 1. 这个研究在解决什么问题

普通的文本 LLM Serving 只有一条主路径：prefill（一次性处理输入）然后 decode（逐个生成 token）。Omni 模型的一个请求要经过多个阶段，例如 MiniCPM-o 4.5 和 Qwen3-Omni 的语音回答路径：

```
输入预处理(CPU) → 音频/视觉编码器 → Thinker（语言模型，生成文本）→ Talker（生成语音 token）→ Code2Wav / 声码器（生成波形）→ 流式返回客户端
```

这些阶段的计算量、显存占用和对 CPU 的依赖都不一样。于是会出现一些文本 LLM 里少见的问题：

- CPU 解码视频、重采样音频的时候，GPU 可能在等；
- Thinker 生成得快、Talker 跟不上时，请求会在 Talker 前排队，这就是 **Backpressure（背压）**：下游处理不完，上游的产出只能堆积；
- 多张 GPU 上，可以把所有阶段放在一起复制多份，也可以把阶段拆到不同 GPU（**Disaggregation，拆分部署**），哪种更好取决于负载和通信开销。

我们想用测量回答：在给定 GPU 预算下，哪个阶段是瓶颈、GPU 空闲从哪里来、什么时候拆分部署能带来收益。

## 2. 为什么要看 Agent 负载，Agent Trace 是什么

真实应用中，请求往往由 Agent 发出：Agent 先问模型“下一步做什么”，再调用工具（计算器、数据库、网页 API），拿到结果后再问模型，最后可能用语音回答用户。请求之间有**依赖关系**，中间夹着工具执行时间和用户思考时间。这和基准测试里“同时发 N 个独立请求”的负载很不一样。

**Agent Trace**（Agent 执行轨迹）就是把这个过程按时间记录下来：每个会话、每一步、每次模型请求、每次工具调用的开始和结束时间，以及它们之间的依赖。有了 trace，就可以：

1. 统计真实 Agent 负载的特征（请求间隔、突发性、多模态比例）；
2. 把同一份 trace **回放（Replay）**到不同的 Serving 配置上，在相同负载下比较配置；
3. 用 request_id 把 Agent 层、Serving 层（SGLang-Omni 的阶段事件）和 GPU 层（采样或 Nsight 时间线）的时间对齐，找出延迟花在哪里。

导师要求“收集 Agent Traces”。我们目前把它理解为**应用层的 Agent 执行轨迹**，这个理解还需要导师确认（见 `docs/01_research/research_questions.md`）。Agent trace、Serving trace、GPU trace 三层在代码和文档里严格分开。

## 3. 目前支持的模型和 Runtime

| 项目 | 版本 | 状态 |
|---|---|---|
| Serving Runtime | SGLang-Omni `520b49ee`（上游 `1845546e` + 两个 pre-Hopper GPU 修复），SGLang v0.5.21 | A6000 环境已在早先学习项目中验证（Qwen3-Omni） |
| MiniCPM-o 4.5 | `openbmb/MiniCPM-o-4_5@fb09628e`，Apache-2.0 | 已下载；A6000 和 GB10 上的运行情况待第一批实验确认 |
| Qwen3-Omni-30B-A3B-Instruct | `@26291f79`，BF16 | 早先在 3×A6000 上跑通过语音流水线（thinker TP=2 + talker） |
| Qwen3-TTS-12Hz-0.6B-Base | `@5d839924` | 只用来合成实验输入音频，不是被测系统 |

## 4. 项目结构

```
README.md            本文件
AGENTS.md            给 Coding Agent 的工作约定（代码风格、测试、数据与凭据规则）
docs/                研究文档（中文），从 docs/00_start_here.md 开始读
  01_research/       研究目标、研究问题、相关工作
  02_workloads/      Agent trace 格式、负载定义、数据来源
  03_hardware/       DGX Spark、Slurm 集群、GPU 拓扑、任务放置规则
  04_experiments/    实验矩阵、测量方法、指标定义
  05_findings/       实验结论（有真实结果后才会出现）
  06_guides/         环境安装、Mac+Spark 工作流、Slurm 用法、复现方法
configs/             可复现配置：workloads/ models/ experiments/ orchestration/ runtime/
src/omni_research/   Python 代码
  tracing/           Agent trace 格式、采集、读取、校验；工具调用 Agent（agent_harness.py）
  replay/            回放引擎、压测驱动、HTTP 后端、合成素材、CPU-only 假服务
  orchestration/     实验编排（DAG 调度、Slurm step 启动、断点恢复）、任务放置
  analysis/          区间运算（GPU 忙/闲）、延迟与吞吐统计
scripts/
  mac/               在 Mac 上管理 Spark 任务（检查、启动、状态、同步）
  spark/             在 Spark 上运行：环境准备、有状态任务、停止任务
  slurm/             Slurm：preflight、提交、作业主体、单个实验入口、清理
tests/               自动化测试（不需要 GPU）
results/             只放说明文件；实验数据在 PROJECT_DATA_ROOT，不提交到 GitHub
```

## 5. 快速开始（Quick Start）

只需要 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)，不需要 GPU：

```bash
git clone -b research/agent-trace-end-to-end https://github.com/Yixingac/omni-serving-research.git
cd omni-serving-research
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest -q          # 全部测试，约 20 秒
```

测试里用到的“服务”都是 mock 后端或 `omni_research.replay.fake_server`（一个只用 CPU 的假 SGLang-Omni 服务）。它们产生的数字不是测量结果。

## 6. Mac 连接 Spark，启动和恢复 tmux 任务

Spark 是常开的开发和测试节点（GB10、128 GB 统一内存、ARM64）。说明：你给的别名 `s01_remote` 在这台 Mac 的 SSH 配置里不存在；实际可用、并已核实指向同一台机器（spark-b935）的是 `s01-remote`。脚本默认用它，可以用 `SPARK_HOST=...` 覆盖。

```bash
scripts/mac/check_spark.sh                         # 非交互连接检查 + 环境摘要
scripts/mac/start_spark_task.sh my-task-001 omni-run '.venv/bin/python -m pytest -q'
scripts/mac/spark_status.sh                        # 所有任务的状态
scripts/mac/spark_status.sh my-task-001            # 单个任务的 status.json 和日志尾部
scripts/mac/attach_spark.sh omni-run               # 进入 tmux 会话（Ctrl-b d 退出，任务继续）
```

每个任务在 Spark 上有自己的目录 `~/research/tasks/<任务ID>/`，记录命令、PID、开始/结束时间、退出码和日志。tmux 会话只保证终端不随 SSH 断开而消失，任务是否成功以 `status.json` 为准。详见 `docs/06_guides/mac_spark_workflow.md`。

## 7. Trace 采集、回放和压测

实验通常由编排器在 Slurm 上运行（下一节），但每个部分也可以单独调用：

```bash
# 在一个已经运行的服务上跑一组实验阶段（smoke、闭环并发扫描、开环、Agent、回放、生成素材）
python -m omni_research.orchestration.experiment configs/experiments/a6000/x5_minicpm_agents.json --out /path/to/exp

# 校验一个 trace 文件
python -c "from omni_research.tracing import validate_file; print(validate_file('agent_trace.jsonl').to_dict())"
```

trace 格式见 `docs/02_workloads/agent_trace_schema.md`；回放的两种模式（请求级开环、依赖感知闭环）见 `docs/04_experiments/benchmark_methodology.md`。

## 8. 提交 Slurm 实验

```bash
ssh unicorn
cd ~/workspace/omni-serving-research
bash scripts/slurm/submit.sh hold --gpus 3 --hours 10 --dry-run     # 先看估计开始时间，不提交
bash scripts/slurm/submit.sh run --gpus 8 --hours 16 --plan configs/experiments/plan_a6000_3gpu.json
```

`submit.sh` 会先向 Slurm 查询可用的分区、GPU 类型和账户，拒绝超过 8 张 GPU 或 16 小时的申请，并把作业记入 `results/jobs.jsonl`。`cleanup.sh` 只会取消这份记录里、属于当前用户、带本项目标记的作业。作业主体 `run_all.sh` 在 Slurm 提前 15 分钟发出的 `USR1` 信号或剩余时间不足时停止启动新实验、保存结果。详见 `docs/06_guides/slurm_usage.md`。

## 9. 查看结果和报告

- 每个实验一个目录：`$PROJECT_DATA_ROOT/experiments/<日期>_<实验名>_runNN/`，里面有 `manifest.json`、`config.json`、`result.json`、`logs/`、`raw/`、`analysis/`、`figures/`，以及中文的 `report.md`。
- 最新一次成功的运行：`$PROJECT_DATA_ROOT/latest/<实验名>`。
- 汇总报告在 `reports/`，研究结论在 `docs/05_findings/`（有结果后才会出现）。
- 复现某个实验：`docs/06_guides/reproducibility.md`。

## 10. 研究进展（Research Status）

已完成（有代码和测试）：

- Agent trace 格式、采集器、读取与校验（依赖 DAG、因果顺序、时钟域）
- 回放引擎：请求级（开环）与依赖感知（闭环）两种模式，会话复制、到达过程、加速
- SGLang-Omni SSE 客户端（文本与流式音频块）、多副本路由、mock 后端和 CPU-only 假服务
- 实验执行器：启动服务、等待就绪、按阶段测量、GPU/CPU 采样、服务端请求事件、失败分类
- 实验编排：DAG、资源装箱、backfill、截止时间、有限重试、原子检查点与断点恢复
- Slurm 脚本（preflight、提交、作业主体、清理）、Mac/Spark 任务管理、任务放置规则
- 合成素材（可核对答案的语音提问、数圆圈视频）与带工具调用的 Agent

进行中：

- Slurm Job 409712（3×A6000，10 小时）排队中，启动后自动运行 `configs/experiments/plan_a6000_3gpu.json` 的 12 个实验
- Spark 上构建 aarch64 运行环境，之后在 GB10 上做 MiniCPM-o smoke test（只验证功能）

尚未完成：A6000 正式结果、Nsight/PyTorch Profiler 的时间线分析、8 卡实验、全双工（`/v1/realtime`）实验、研究结论。

## 数据与发布规则

本仓库是公开仓库。只提交原创代码、文档和合成或公开数据；不提交原始实验数据、模型权重、Profiler 大文件、凭据、内部讨论或未公开的研究材料。尚未选择开源许可证。
