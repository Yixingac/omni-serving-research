# 实验 9–11：从这里开始

这三个实验都在一台 8×A6000 节点上做（Slurm Job 454290，compute-03），模型是 MiniCPM-o 4.5，服务框架 sglang-omni 520b49ee 加我们自己的补丁。日期 2026-11-17 到 11-18。

## 10 分钟读法

1. **先看结论**：[exp09_11_findings.md](exp09_11_findings.md)。每条结论都标了证据等级：Measured（实测）、Derived（推算）、Estimated（估算）、Hypothesis（假设）、Unknown（未知）。
2. **不熟悉术语**：先读 [exp09_11_concepts.md](exp09_11_concepts.md)，20 个概念，每个都按“定义 → 生活例子 → 本项目里的例子 → 容易混淆的地方”讲。
3. **再看各实验的“一句话”和图**：

| 实验 | 问题 | 先看的图 |
|---|---|---|
| [实验 9：8 张 GPU 怎么分工最好](../../research/01_性能瓶颈/实验09_最佳GPU分工/README.md) | thinker、talker、code2wav 怎么分 8 张卡 | 图 3（闭环吞吐）、图 4（开环延迟）、图 7（GPU 忙碌比例：最闲的卡反而是瓶颈） |
| [实验 10：Code2Wav 组批和补齐](../../research/01_性能瓶颈/实验10_Code2Wav_Batching/README.md) | code2wav 凑大批、按长度组批能不能省 GPU | 图 1（长短混合时批越大越慢）、图 5（先进先出对按长度组批）、图 9（按长度组批让谁多等），再看第六部分末尾的完整服务验证 |
| [实验 11：Thinker 异步拷贝](../../research/01_性能瓶颈/实验11_Thinker异步优化/README.md) | thinker 请求结束时的阻塞拷贝能不能安全地去掉 | 图 9（正确性检查为什么没通过）、图 11（诊断：偏差有多大、来自哪里）。正确性没通过，所以没有性能图 |

4. **想知道结论有多可靠**：读各实验目录里的 `limitations.md`。事先登记的假设和判定标准在 `hypotheses.md`；看到数据后才加的东西都写成带时间的“补充登记”。

## 各实验目录里有什么

| 文件 | 内容 |
|---|---|
| `README.md` | 给初学者的完整说明，9 个部分：一句话、为什么做、基础概念、请求经过哪里、怎么做、每张图怎么读、证据等级、研究意义、下一步 |
| `hypotheses.md` | 事先登记的假设、判定标准和补充登记 |
| `metadata.json` | 运行目录、时间、状态、代码版本、软件版本、随机种子 |
| `results/` | 汇总后的数（不含原始数据），字段见 [exp09_11_data_dictionary.md](exp09_11_data_dictionary.md) |
| `figures/` | 图，全部由 `scripts/analysis/exp09_*.py`、`exp10_*.py`、`exp11_*.py` 从 `results/` 画出 |
| `limitations.md` | 失败、偏离、限制，以及对结论的影响 |
| `reproduce.md` | 怎样从头复现 |

## 其他文件

| 文件 | 内容 |
|---|---|
| [exp09_11_baseline_audit.md](exp09_11_baseline_audit.md) | 开始前对实验 6–8 的数据和代码的审计 |
| [exp09_11_data_dictionary.md](exp09_11_data_dictionary.md) | 原始数据和结果文件的每个字段 |
| [docs/research_skills_used_exp09_11.md](../../docs/research_skills_used_exp09_11.md) | 用到的研究方法 Skill、排查记录、测试先行记录、本轮的过程问题 |
| [configs/experiments/exp09_11/resource_plan.md](../../configs/experiments/exp09_11/resource_plan.md) | GPU 时间的预算和实际用量 |
| [research/实验总目录.md](../../research/实验总目录.md) | 所有实验的编号、状态和一句话结论 |

## 怎么复现

- 配置由 `scripts/slurm/make_plan_exp09_11.py` 生成，放在 `configs/experiments/exp09_11/`。
- sglang-omni 的补丁在 `configs/runtime/patches/`，每个补丁都配有 CPU 检查脚本。
- 原始数据在集群的 `results/experiments/<运行目录>/` 下，不公开。在能读到原始数据的机器上，按各实验 `reproduce.md` 的命令重新生成 `results/` 和 `figures/`。
