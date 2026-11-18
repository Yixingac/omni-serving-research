# 实验室 Slurm 集群（Unicorn）

正式的 A6000 测量都在这里做。本页只写与实验有关、并且已经核实过的信息；节点主机名和 IP 不写进公开仓库，原始记录在 `$PROJECT_DATA_ROOT/runs/*/preflight/topology.json`（不提交）。

## 可用资源（2026-11-11 用 `sinfo`、`scontrol`、`sbatch --test-only` 核实）

| 项目 | 内容 |
|---|---|
| 调度器 | Slurm 25.05.8 |
| 组分区 | 3 个节点：节点 A（3×RTX A6000 48 GB，32 核，1 TB 内存，ECC 开启）、节点 B（8×RTX A6000 48 GB，64 核，1 TB）、节点 C（5×RTX 3090 24 GB，32 核，512 GB） |
| 优先级 | 组分区 tier 20，其他组的作业在这些节点上只能以 tier 15/10 运行，可被组分区作业抢占；同组成员之间不能互相抢占 |
| 本项目上限 | 每次分配最多 8 张 A6000、16 小时（由 `scripts/slurm/submit.sh` 强制） |
| 交互作业 | 被 job_submit 插件转到 `*-interactive` 分区，最长 48 小时；本项目用批处理作业 |
| 存储 | 家目录是组 NFS（`/home`，2026-11-11 时 15 TB，剩余约 5 TB）；计算节点有本地 `/scratch`，作业结束后不保证保留 |

## 本项目的作业

| Job | 资源 | 用途 | 状态 |
|---|---|---|---|
| 409712 | 节点 A，3×A6000，24 CPU，300 GB，10 小时 | 第一批实验：环境检查、素材、单卡基线、Agent trace、回放、2 卡放置对比、Qwen3-Omni 3 卡 | 2026-11-11 排队中，预计 19:19 开始 |

作业启动后，`run_all.sh` 会先运行 preflight（记录实际分到的 GPU、拓扑、CPU、内存、剩余时间），再自动运行 `autostart_plan.json` 里的实验计划，不需要有人在线。之后它继续保留 GPU，直到时间余量用完或出现 STOP 文件，这段时间可以手动用 `srun --jobid` 加做实验。

## GPU 隔离

- 每个实验是一个 `srun --exact` step，显式申请 `--gres=gpu:nvidia_rtx_a6000:N`、CPU 和内存。由 Slurm 决定 step 能看到哪些 GPU，并相应地重映射 `CUDA_VISIBLE_DEVICES`；代码从不访问 step 之外的 GPU。
- 只用 CPU 的 step 申请 `--gres=none`，否则它会默认继承作业的全部 GPU。
- 同一节点上并行的实验有各自的 GPU 和 CPU 核（cgroup 隔离），但共享 PCIe、内存带宽和 NFS。哪些实验同时运行，可以从编排器的 `events.jsonl` 算出来，报告里会写明。

## 已知的坑

- 非登录 shell 里没有 Slurm 命令，要用 `ssh unicorn 'bash -lc "..."'`。
- `squeue` 看不到别人的作业（PrivateData）；想知道作业何时能开始，用 `sbatch --test-only`。
- NFS 上第一次 import torch/sglang 很慢（冷启动几分钟），`run_experiment.sh` 在每个节点上只预热一次页缓存。
- 节点之间没有发现 InfiniBand，只有 10 GbE，所以不做跨节点的分布式推理。
