# 任务放置：Mac、Spark 还是 Slurm

清单：`configs/orchestration/task_placement.yaml`。重新计算放置和理由：

```bash
PYTHONPATH=src .venv/bin/python -m omni_research.orchestration.placement
```

代码：`src/omni_research/orchestration/placement.py`，测试：`tests/test_placement.py`。

## 三台机器各做什么

| 机器 | 适合 | 不适合 |
|---|---|---|
| **Mac**（控制端） | Git/GitHub、SSH 和远程任务协调、文档、几秒到一两分钟的小任务、需要本机凭据的任务（推送代码、从 Unicorn 同步数据） | GPU 工作、几分钟以上的长任务（Mac 可能休眠、断网、关掉终端） |
| **Spark**（常开的开发与分析节点） | 环境验证、GB10 上的功能测试和 smoke test、Nsight 数据解析、trace 分析、统计、作图、写报告；所有长时间任务都放在 tmux 里 | 正式 A6000 性能测量；任何需要 Unicorn 凭据的操作 |
| **Slurm**（Unicorn 的 A6000） | 正式的基线、profiling、多卡放置和拆分部署、相同 GPU 预算下的 A/B 对比 | 纯 CPU 的数据处理（不该占用实验室 GPU 时间）、与正式测量同时运行的重负载 |

## 规则（按顺序判断）

1. **固定主机**：需要只在某台机器上存在的条件，就固定在那台机器。例如推送代码和创建 PR 需要 GitHub 凭据（只在 Mac）；从 Unicorn 同步数据需要 Unicorn 的 SSH 凭据（只在 Mac，不复制到 Spark）；构建 Spark 自己的运行环境只能在 Spark 上。
2. **正式测量去 Slurm**：A6000 上的性能测量、多卡实验，结果必须来自同一套硬件和软件，只能在 Slurm 上做，没有备选。
3. **其他 GPU 工作去 Spark**：不是正式测量的 GPU 工作（功能验证、smoke test、Profiler 调试）放在 Spark，不占实验室 GPU。ARM64 兼容性未知时先在 Spark 上试，失败再交给 Slurm；已知不支持 ARM64 的直接交给 Slurm。
4. **CPU 工作比较总耗时**：总耗时 = 准备时间 + 数据传输时间 + 计算时间 + 结果保存时间。任务很短时留在 Mac，因为 SSH 和数据传输的开销比计算本身还大；任务预计超过 5 分钟，放到常开的 Spark 上用 tmux 运行。
5. **不干扰正式测量**：一台机器正在做正式测量时，不在上面放重的 CPU 任务。

估算用历史记录校正：同一类任务在同一台机器上至少有 3 次记录后，用“实际耗时 / 预估耗时”的中位数乘到后续估算上。只有一次记录时不校正，避免一次偶然的结果左右后续判断。

## 当前的放置结果（2026-11-11）

| 任务 | 放置 | 理由 |
|---|---|---|
| 单元测试 | Mac（Spark 也跑一遍检查环境） | 半分钟以内 |
| Spark 运行环境构建 | Spark（固定） | 构建的是 Spark 自己的环境 |
| GB10 上的 MiniCPM-o smoke test | Spark，备选 Slurm | GPU 功能验证，不是正式测量 |
| Nsight Systems 采集与解析验证 | Spark，备选 Slurm | 同上 |
| Job 409712 的实验计划 | Slurm | 正式 A6000 测量和多卡实验 |
| 结果同步 | Mac（固定） | 只有 Mac 有 Unicorn 凭据 |
| trace 分析、作图、报告 | Spark | 长任务，放在常开的机器上 |
| Draft PR | Mac（固定） | 需要 GitHub 凭据 |

## 能并行的和不能混在一起的

- 能并行：Slurm 上的 A6000 实验进行时，Spark 处理上一批数据、作图、做 GB10 功能测试，Mac 写代码和文档。三台机器互不影响。
- 不能混在一起：同一个节点上的正式基线和无关的重负载；Spark 上的 GPU 功能测试和可能出现的租赁任务（每次启动前检查 GPU 是否空闲）。Slurm 节点内部，同时运行的实验各有独立的 GPU 和 CPU 核，但共享 PCIe 和内存带宽；需要严格隔离的基线可以单独安排。
