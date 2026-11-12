# Mac + Spark 工作流

## 连接

```bash
scripts/mac/check_spark.sh        # 非交互连接（BatchMode）+ Spark 环境摘要
```

默认主机是 `SPARK_HOST=s01-remote`。你给的别名 `s01_remote` 在 Mac 的 SSH 配置里不存在，`s01-remote` 已核实是同一台机器。如果连接需要交互认证，脚本会直接报错退出，不会卡住；这时先在终端里手动 `ssh s01-remote` 解决认证问题。

## 第一次准备（幂等，可以重复运行）

```bash
ssh s01-remote 'bash -s' < scripts/spark/bootstrap.sh            # 仓库、tmux 会话、研究 venv
ssh s01-remote 'cd ~/research/repos/omni-serving-research && bash scripts/spark/setup_runtime.sh all'
```

`bootstrap.sh` 对已经存在的仓库只做 fetch：有本地修改或不在研究分支时，不会更新也不会覆盖。

## 启动和查看任务

```bash
scripts/mac/start_spark_task.sh <任务ID> <会话> '<命令>'   # 命令在 Spark 仓库目录下执行
scripts/mac/spark_status.sh                                # 所有任务
scripts/mac/spark_status.sh <任务ID>                       # 单个任务的 status.json 和日志尾部
scripts/mac/attach_spark.sh omni-run                       # 进入 tmux（Ctrl-b d 退出）
ssh s01-remote 'bash ~/research/repos/omni-serving-research/scripts/spark/stop_task.sh <任务ID>'
```

三个 tmux 会话：`omni-dev`（环境、测试、调试）、`omni-run`（模型推理、trace 采集、回放）、`omni-analysis`（分析、作图）。

每个任务一个目录 `~/research/tasks/<任务ID>/`：

| 文件 | 内容 |
|---|---|
| `status.json` | 任务 ID、会话、主机、命令、PID、开始/结束时间、退出码、状态 |
| `stdout.log`、`stderr.log` | 输出 |
| `command.txt`、`cmd.sh`、`wrapper.sh` | 实际执行的命令 |
| `pgid` | 进程组 id，`stop_task.sh` 只向这个进程组发信号 |

任务 ID 已存在时 `run_task.sh` 拒绝启动，重跑必须换新的 ID（例如末尾加 `-002`），旧记录保留。

## 断线后恢复

SSH 断开、Mac 休眠都不影响 Spark 上的任务。重新连上后：

1. `scripts/mac/spark_status.sh`：看哪些任务在运行、哪些结束了。`status` 是 `running` 但进程已不存在时，会显示 `pid GONE`，说明任务异常终止（例如 Spark 重启），需要看日志决定是否用新 ID 重跑；
2. 看日志尾部，确认失败原因；
3. 实验数据在 `~/research/data/experiments/`，已完成的不要重跑或覆盖。

tmux 只保证终端会话不随 SSH 断开而消失；任务是否成功以 `status.json` 和实验目录里的 `result.json` 为准。

## 同步结果

```bash
scripts/mac/sync_results.sh unicorn:workspace/omni-serving-research/results/experiments/ s01-remote:research/data/unicorn-experiments/
```

Spark 上没有 Unicorn 的凭据，所以这类同步经 Mac 中转。脚本只新增或更新文件，不删除目标端文件；正式复制前会先显示要传多少文件、多大，确认后才执行（`--yes` 跳过确认）。模型权重和 `.cache` 不会同步。
