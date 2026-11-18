# 实验 6–8：从这里开始

这三个实验都在一台 8×A6000 节点上做（Slurm Job 454290，compute-03），模型是 MiniCPM-o 4.5，服务框架 sglang-omni 520b49ee 加了我们自己的测量补丁。日期 2026-11-17。

## 10 分钟读法

1. **先看结论**：[exp06_08_findings.md](exp06_08_findings.md)。每条结论都标了性质：实测、推算、估算、假设、未知。
2. **再看各实验的“一句话”和图**：

| 实验 | 问题 | 先看的图 |
|---|---|---|
| [实验 6：Code2Wav 时间拆解](../../research/01_性能瓶颈/实验06_Code2Wav时间拆解/README.md) | code2wav 为什么从 0.16 秒涨到 5 秒多 | 图 2（各段时间）、图 5（凑批省不省时间） |
| [实验 7：Code2Wav 独立 GPU](../../research/01_性能瓶颈/实验07_Code2Wav独立GPU/README.md) | code2wav 搬到单独的卡上会不会更快 | 图 5（各段处理时间：同卡时互相拖慢）、图 2（同样 8 张卡的吞吐） |
| [实验 8：Thinker 提前一步发射](../../research/01_性能瓶颈/实验08_Thinker_Lookahead/README.md) | thinker 每步的 GPU 空档能不能去掉 | 图 1（时间线）、图 3（每步空档）、图 5（原型关对开） |

3. **想知道结论有多可靠**：读各实验目录里的 `limitations.md`。事先登记的假设和判定标准在 `hypotheses.md`，补充登记都写明了时间，以及当时已经看到哪些数据。

## 各实验目录里有什么

| 文件 | 内容 |
|---|---|
| `README.md` | 给初学者的完整说明：为什么做、基础概念、怎么做、每张图怎么读、结果意味着什么、下一步 |
| `hypotheses.md` | 事先登记的假设、判定标准、补充登记 |
| `metadata.json` | 运行 ID、时间、Slurm Job、软件版本、配置文件、随机种子 |
| `results/` | 汇总后的数（不含原始数据），字段见 [data_dictionary.md](data_dictionary.md) |
| `figures/` | 图，全部由 `scripts/analysis/exp0[678]_*.py` 从 `results/` 画出 |
| `limitations.md` | 失败、偏离、限制，以及对结论的影响 |

## 怎么复现

- 配置由 `scripts/slurm/make_plan_exp06_08.py` 生成，放在 `configs/experiments/exp06_08/`。
- sglang-omni 的补丁在 `configs/runtime/patches/`，每个补丁都配有 CPU 检查脚本。
- 原始数据在集群的 `results/experiments/<运行目录>/` 下。在能读到原始数据的机器上，按各 README 末尾“文件在哪里”一节列出的脚本（每个脚本开头写有运行命令）重新生成 `results/` 和 `figures/`。
- 用到的研究方法 Skill，以及它们用在哪一步，见 [docs/research_skills_used.md](../../docs/research_skills_used.md)。
