#!/usr/bin/env bash
# Mac 端：在 Spark 的某个 tmux 会话里启动一个有状态记录的任务（在 Spark 仓库目录下执行）。
#   scripts/mac/start_spark_task.sh TASK_ID SESSION 'COMMAND'
# 例：scripts/mac/start_spark_task.sh tests-001 omni-dev '.venv/bin/python -m pytest -q'
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/spark_common.sh"
task_id="${1:?usage: start_spark_task.sh TASK_ID SESSION 'COMMAND'}"
session="${2:?missing session (omni-dev | omni-run | omni-analysis)}"
cmd="${3:?missing command}"
spark "cd \$HOME/$SPARK_REPO && bash scripts/spark/run_task.sh $(printf '%q' "$task_id") $(printf '%q' "$session") --cwd \$HOME/$SPARK_REPO -- bash -c $(printf '%q' "$cmd")"
