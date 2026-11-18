#!/usr/bin/env bash
# Spark 端：停止一个由 run_task.sh 启动的任务。只向该任务自己的进程组发信号。
#   stop_task.sh TASK_ID [GRACE_SECONDS]
set -uo pipefail
TASK_ROOT="${OMNI_TASK_ROOT:-$HOME/research/tasks}"
task_id="${1:?usage: stop_task.sh TASK_ID [GRACE_SECONDS]}"
grace="${2:-30}"
dir="$TASK_ROOT/$task_id"
[[ -f "$dir/pgid" ]] || { echo "stop_task.sh: no process group recorded for $task_id" >&2; exit 2; }
pgid="$(cat "$dir/pgid")"
# Confirm the group still belongs to this task's command before signalling it.
if ! ps -o pgid= -g "$pgid" >/dev/null 2>&1; then
  echo "task $task_id (pgid $pgid) is not running"
  exit 0
fi
touch "$dir/STOPPED"
kill -TERM -- "-$pgid" 2>/dev/null
for _ in $(seq "$grace"); do
  ps -o pgid= -g "$pgid" >/dev/null 2>&1 || { echo "task $task_id stopped"; exit 0; }
  sleep 1
done
kill -KILL -- "-$pgid" 2>/dev/null
echo "task $task_id killed after ${grace}s grace"
