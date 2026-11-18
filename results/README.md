# results/

这个目录在 git 里只有本说明文件。实验数据、作业记录和日志都写在 `PROJECT_DATA_ROOT`：

- Unicorn：默认 `~/workspace/omni-serving-research/results`（组 NFS，持久保存）；
- Spark：默认 `~/research/data`；
- 本机测试：`<仓库>/results`。

| 子目录或文件 | 内容 |
|---|---|
| `experiments/<日期>_<实验名>_runNN/` | 每次实验运行一个目录，格式见 `docs/06_guides/reproducibility.md` |
| `latest/<实验名>` | 指向最近一次成功的运行 |
| `assets/<池名>/` | 合成的输入素材及其 `manifest.jsonl` |
| `runs/<模式>-<时间>/` | 每个 Slurm 作业的目录：`job.sbatch`、preflight、编排器状态与日志 |
| `jobs.jsonl` | 本项目提交过的 Slurm 作业记录；`cleanup.sh` 只认这里面的作业 |
| `dryrun-*/`、`runs/test-*` | 用 CPU 版假服务做的流程测试，数字没有意义 |

为什么不提交：原始数据体积大，可能包含主机和环境信息，而且仓库是公开的。经过审阅的小型汇总表和图会放进 `reports/`。
