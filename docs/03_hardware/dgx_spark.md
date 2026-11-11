# 个人 DGX Spark

## 硬件与软件（2026-11-11 核实）

| 项目 | 内容 |
|---|---|
| 主机 | spark-b935（Mac 上的 SSH 别名 `s01-remote`，Tailscale；局域网别名 `s01`） |
| GPU | NVIDIA GB10，compute capability 12.1 |
| 内存 | 128 GB CPU/GPU 统一内存（系统显示 121 GB） |
| CPU | 20 核 ARM64 |
| 系统 | DGX OS 7.6.0，驱动 580.178.04，CUDA 13.0 |
| 磁盘 | 3.7 TB NVMe，剩余约 3.5 TB |
| 工具 | Python 3.12、uv、git、tmux、Nsight Systems（`/usr/local/bin/nsys`） |
| 网络 | 只有 Wi-Fi |

说明：你提供的别名 `s01_remote` 在这台 Mac 的 SSH 配置里不存在。`s01-remote` 已核实是同一台机器（主机名、GPU、驱动都对得上）。脚本默认用 `s01-remote`，可以用环境变量 `SPARK_HOST` 改。SSH 配置没有被修改。

## 在 Spark 上做什么

- 功能验证：模型能否在 GB10 上运行、Trace 采集和回放是否正确、Profiler 能否工作；
- 数据分析：trace 解析、统计、作图、中文报告；
- 不依赖 A6000 拓扑的开发和测试。

Spark 上的性能数字只用于验证功能，**不和 A6000 的数字放在同一张对比表里**。两者的内存结构（统一内存 vs 独立 GDDR6）、CPU 架构（ARM64 vs x86_64）、GPU 架构（sm_121 vs sm_86）以及可用的 kernel 都不同。

## 目录

| 路径 | 内容 |
|---|---|
| `~/research/repos/omni-serving-research` | 本仓库（git clone，研究分支），`.venv` 是只含 CPU 依赖的研究环境 |
| `~/research/omni-runtime` | SGLang-Omni 运行环境（`scripts/spark/setup_runtime.sh` 构建）：sglang-omni `520b49ee`、sglang v0.5.21、torch 2.13.0+cu130、flashinfer 0.6.18 |
| `~/research/models/MiniCPM-o-4_5` | 模型权重，revision `fb09628e` |
| `~/research/tasks/<任务ID>/` | 每个任务的状态和日志 |
| `~/research/data/` | Spark 实验的数据根目录（`PROJECT_DATA_ROOT`） |

## 运行环境构建记录

| 任务 | 结果 | 原因与处理 |
|---|---|---|
| spark-runtime-build-001 | 失败 | conda-forge 没有 aarch64 的 `sox` 包；改为单独尝试安装，没有就跳过 |
| spark-runtime-build-002 | 失败 | conda 装了 pynini 2.1.7，但 `nemo-text-processing 1.2.0` 要求 2.1.6.post1，pip 转而从源码编译失败；改为在 conda 中固定 `pynini=2.1.6.post1` |
| spark-runtime-build-003 | 成功 | import 检查和 GB10 上的 bf16 矩阵乘通过 |
| spark-minicpm-text-smoke-001 | 失败 | MiniCPM-o 需要的可选依赖 `onnx`（sglang-omni 的 `minicpm-o` extra）没装。**Unicorn 的 A6000 环境同样缺少它**，已在两边补装，并在实验执行器里加了启动前的依赖检查 |

已知问题：版本锁定文件里的 `grpcio==1.85.0rc1` 已被 PyPI 撤回（有安全公告）。它来自旧环境的依赖快照；我们不运行对外的 gRPC 服务，暂时记录为待升级项。

## 共享这台机器时的约束

- `gpu-agent`（Primcast GPU 租赁服务）在运行；入驻日志显示还没有上架出租。每个 GPU 任务开始前，`scripts/spark/run_experiment.sh` 会检查 GPU 上是否已有其他计算进程，有就放弃运行。
- 不用 sudo，不修改系统配置、驱动或 CUDA 安装，不碰 `gpu-agent`。
- 只停止本项目记录的进程组（`scripts/spark/stop_task.sh`）。
