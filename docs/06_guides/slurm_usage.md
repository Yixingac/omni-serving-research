# Slurm 用法

所有命令都在 Unicorn 登录节点上执行（`ssh unicorn`），仓库在 `~/workspace/omni-serving-research`。

## 提交

```bash
bash scripts/slurm/submit.sh hold --gpus 3 --hours 10 --dry-run          # 只看估计开始时间
bash scripts/slurm/submit.sh hold --gpus 3 --hours 10                    # 保留 GPU，启动时自动运行 autostart_plan.json
bash scripts/slurm/submit.sh run  --gpus 8 --hours 16 --plan configs/experiments/<plan>.json
```

`submit.sh` 会：
- 用 `sacctmgr` 和 `sinfo` 查出账户、可用的组分区和 GPU 类型，不在代码里写死集群参数；
- 拒绝超过 8 张 GPU 或 16 小时的申请；
- 生成 `results/runs/<模式>-<时间>/job.sbatch`，带 `--signal=B:USR1@900`（结束前 15 分钟发信号）；
- 打印 `sbatch --test-only` 的估计开始时间，然后提交，并把作业写进 `results/jobs.jsonl`。

`hold` 模式的作业启动后会先跑 preflight，然后：如果作业目录里有 `autostart_plan.json`，就运行这个计划；之后继续保留 GPU，可以手动加实验：

```bash
srun --jobid=<JOBID> --exact --gres=gpu:nvidia_rtx_a6000:1 -c 8 --mem=90G \
     bash scripts/slurm/run_experiment.sh configs/experiments/a6000/<实验>.json
```

运行中的计划可以追加实验：把一个计划格式的 JSON 放进 `<作业目录>/orchestrator/plan.d/`，编排器下一轮就会读进来（已有实验的定义不会被修改）。

## 查看

```bash
squeue --me
PYTHONPATH=src python3 -m omni_research.orchestration.cli status --run-dir results/runs/<作业目录>/orchestrator
tail -f results/runs/<作业目录>/orchestrator.log
```

## 停止

- 停止整个计划但保留数据：`touch results/runs/<作业目录>/STOP`。编排器不再启动新实验，向正在运行的实验发 SIGTERM 并等待，实验执行器保存已有数据后退出。
- 取消本项目的作业：`bash scripts/slurm/cleanup.sh --job <JOBID>`。它只会取消 `results/jobs.jsonl` 里、属于当前用户、带 `omni-serving-research` 注释的作业。

## 截止时间

作业主体同时用两种方法判断何时停止：收到 Slurm 的 `USR1` 信号，以及自己用 `squeue %L` 查询剩余时间。剩余时间少于 15 分钟时不再启动新实验；预计时长放不进剩余时间的实验直接标为 `SKIPPED`（原因 `insufficient_time`），而不是跑到一半被杀掉。
