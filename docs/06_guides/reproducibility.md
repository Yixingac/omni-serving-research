# 复现

## 一个实验目录里有什么

`$PROJECT_DATA_ROOT/experiments/<日期>_<实验名>_runNN/`：

| 路径 | 内容 |
|---|---|
| `manifest.json` | 实验 id、主机、可见 GPU、本仓库 commit（是否有未提交修改）、sglang-omni commit、Python 包版本、模型 revision、起止时间、文件清单 |
| `config.json` | 实际运行的配置（展开了 runtime 和 model 引用） |
| `result.json` | 状态、失败类别、各阶段摘要、服务就绪时间、GPU 摘要 |
| `logs/` | `server_<名字>.log`、`client.log`、`env/`（nvidia-smi、拓扑、NVLink） |
| `raw/` | `requests_<阶段>.jsonl`（每个请求一行）、`agent_trace_<阶段>.jsonl`、`bench_trace.jsonl`、`replay_trace_<阶段>.jsonl`、`server_events/<阶段>/`、`gpu_samples.csv`、`cpu_samples.jsonl` |
| `analysis/` | `phases.json`、`metrics.json`、`summary.csv` |
| `figures/` | 图表 |
| `report.md` | 中文实验报告 |
| `step.log`、`srun_argv.txt` | Slurm step 的输出和完整 srun 命令 |

`$PROJECT_DATA_ROOT/latest/<实验名>` 指向该实验最近一次成功的运行。

## 复现某个实验

1. 查看 `manifest.json`，切换到对应的 commit：`git checkout <research_repo_commit>`；
2. 运行环境与 `manifest.json` 中的包版本一致（`docs/06_guides/environment_setup.md`）；
3. 在一个有相同 GPU 的分配里运行：

```bash
srun --jobid=<JOBID> --exact --gres=gpu:nvidia_rtx_a6000:<N> -c <CPU> --mem=<MEM> \
     bash scripts/slurm/run_experiment.sh <实验目录>/config.json
```

需要的素材在 `$PROJECT_DATA_ROOT/assets/synthetic-v1/`，可以用 x1_assets 按同样的种子重新生成。

## 重跑的规则

- 每次运行都是新的目录（`runNN` 递增），不会覆盖以前的运行，失败的运行也保留；
- 编排器恢复时不会重跑已经成功的实验；中断的实验算作一次失败，如果还有重试次数，会在新目录里重跑；
- 确定性的失败（例如 OOM、配置错误、不支持）不会在没有任何改动的情况下自动重试。
