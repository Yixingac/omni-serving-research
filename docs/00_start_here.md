# 从这里开始

第一次接触这个项目，建议按下面顺序阅读，大约 20 分钟。

1. **研究什么**：`01_research/research_goals.md`（研究目标）、`01_research/research_questions.md`（具体问题和待确认的假设）、`01_research/prior_work.md`（相关工作，哪些已经有人做过）。
2. **负载是什么**：`02_workloads/agent_trace_schema.md`（Agent trace 的记录格式）、`02_workloads/workload_definitions.md`（W1–W6 负载）、`02_workloads/data_sources.md`（所有输入数据从哪里来）。
3. **在什么硬件上做**：`03_hardware/slurm_cluster.md`（实验室 A6000 集群）、`03_hardware/dgx_spark.md`（个人 DGX Spark）、`03_hardware/gpu_topology.md`（GPU 互联）、`03_hardware/task_placement.md`（任务在 Mac/Spark/Slurm 之间怎么分配）。
4. **怎么测**：`04_experiments/experiment_matrix.md`（实验清单）、`04_experiments/benchmark_methodology.md`（方法）、`04_experiments/metrics.md`（每个指标的准确定义）。
5. **测到了什么**：`05_findings/`（有正式结果后才会出现）和 `../reports/`。
6. **怎么动手**：`06_guides/environment_setup.md`、`06_guides/mac_spark_workflow.md`、`06_guides/slurm_usage.md`、`06_guides/reproducibility.md`。

## 三层 trace，一个 request_id

项目里的“trace”分三层，代码和文档都严格区分：

| 层 | 记录什么 | 谁记录 | 时钟 |
|---|---|---|---|
| Application / Agent trace | 会话、步骤、模型请求、工具调用、等待，以及它们的依赖 | `omni_research.tracing`（客户端） | 客户端进程的 `monotonic_ns`，带一个墙钟锚点 |
| Serving / Pipeline trace | 请求在 SGLang-Omni 各阶段的入队、开始、首个输出、结束 | SGLang-Omni 自带的 request event recorder | 服务端各进程的 `time.time_ns()` |
| Hardware / GPU trace | GPU 采样利用率、显存；或 Nsight Systems 的 kernel 时间线 | `nvidia-smi` 采样器、`nsys` | 墙钟 / nsys 时钟 |

客户端发请求时把自己生成的 `request_id` 放进请求体，SGLang-Omni 在所有阶段事件里沿用它，所以三层可以按 request_id 和时间对齐。客户端和服务端在同一台机器上，墙钟一致；跨机器的时间对齐没有做，也不应假设。

## 一句话版本的工作流

Mac 写代码、管 Git 和远程任务 → Spark 跑功能验证和数据分析 → Slurm（Unicorn 的 A6000）跑正式测量 → 结果经 Mac 同步回来，在 Spark 上分析、出图、写中文报告。
