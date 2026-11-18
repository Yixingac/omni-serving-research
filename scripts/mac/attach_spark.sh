#!/usr/bin/env bash
# Mac 端：交互式进入 Spark 上的 tmux 会话（默认 omni-run）。退出 tmux 用 Ctrl-b d，任务继续运行。
#   scripts/mac/attach_spark.sh [omni-dev|omni-run|omni-analysis]
. "$(dirname -- "${BASH_SOURCE[0]}")/spark_common.sh"
exec ssh -t "$SPARK_HOST" "tmux attach -t ${1:-omni-run}"
