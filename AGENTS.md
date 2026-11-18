# 给 Coding Agent 的工作约定

本文件写给在这个仓库里写代码、跑实验的 Agent（也适用于人）。先读 `README.md` 和 `docs/00_start_here.md`。

## 语言

- 研究文档、报告、实验总结、PR 说明用中文；代码、标识符、命令、配置键用英文。
- 写法像工程师写给同事的技术说明：说清楚做了什么、看到了什么、还不知道什么。不用“赋能、全方位、革命性、闭环、综上所述”之类没有信息量的词。
- 结论必须标注性质：实际测量（Measured）、计算结果（Derived）、估算（Estimated）、研究假设（Hypothesis）、未知（Unknown）。

## 代码

- Python 3.11+。核心包（tracing、replay 的 mock 路径、orchestration、metrics）只依赖标准库，这样能直接在 SGLang-Omni 的运行环境里运行。HTTP 客户端用 aiohttp，画图用 matplotlib。
- 每个模块都要有测试，测试不能依赖 GPU 或网络。需要服务端时用 `omni_research.replay.fake_server`（它的数字永远不能当结果）。
- 提交前运行：`.venv/bin/python -m pytest -q`。
- 文件名用小写 snake_case，表达用途；不要出现 `final_v2.py`、`test2.py` 这类名字。

## 实验与数据

- 实验数据写在 `PROJECT_DATA_ROOT`（Unicorn 上默认 `~/workspace/omni-serving-research/results`，Spark 上默认 `~/research/data`），不进 git。
- 每次运行是新的目录 `<日期>_<实验名>_runNN`，不要覆盖旧目录；失败的运行也保留。
- 不要把 Spark（GB10）上的性能数字和 A6000 的数字放在同一张对比表里。
- 不要为了提高 GPU 利用率跑与研究无关的负载；GPU 空闲本身可能就是研究现象。

## 资源与安全

- Slurm：最多 8 张 A6000、每次分配最多 16 小时；只用 `scripts/slurm/submit.sh` 提交，它会检查上限并记账。只能取消 `results/jobs.jsonl` 里本项目自己的作业。
- Spark：用户的个人机器，`gpu-agent`（GPU 租赁服务）在运行。启动 GPU 任务前检查没有其他计算进程；不用 sudo，不改系统配置，不碰 `gpu-agent`。
- 停止进程只能针对本项目记录的 PID / 进程组（`scripts/spark/stop_task.sh`），不要用 `pkill`、`killall`。
- 不提交、不打印凭据（API key、SSH 私钥、token）。Unicorn 的凭据只在 Mac 上，不复制到 Spark。
- 本仓库公开：只放原创代码与合成或公开数据，不放未公开的研究材料、内部讨论、原始实验数据。

## Git

- 在功能分支上开发，不修改、不合并 PR #1。PR 用 Draft，不自动 merge。
- 提交信息说明改了什么、为什么。
