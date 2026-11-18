#!/usr/bin/env bash
# Mac 端：列出 Spark 上本项目任务的状态，或查看单个任务的状态和日志尾部。
#   scripts/mac/spark_status.sh            所有任务一行一个
#   scripts/mac/spark_status.sh TASK_ID    单个任务的 status.json 和最后 20 行日志
set -uo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/spark_common.sh"
if [[ -n "${1:-}" ]]; then
  spark "d=\$HOME/research/tasks/$1; cat \$d/status.json; echo; echo '--- stdout (tail)'; tail -20 \$d/stdout.log; echo '--- stderr (tail)'; tail -20 \$d/stderr.log"
  exit
fi
# The listing script is sent over stdin so no quoting survives two shells.
spark 'bash -s' <<'REMOTE'
echo "tmux: $(tmux ls -F '#S(#{session_windows})' 2>/dev/null | tr '\n' ' ')"
python3 - <<'PY'
import glob, json, os
for f in sorted(glob.glob(os.path.expanduser("~/research/tasks/*/status.json"))):
    d = json.load(open(f))
    alive = ""
    if d.get("status") == "running" and d.get("pid"):
        alive = " (pid alive)" if os.path.exists(f"/proc/{d['pid']}") else " (pid GONE: crashed or host rebooted?)"
    print(f"{d.get('task_id', '?'):<28} {d.get('status', '?'):<10}{alive} exit={d.get('exit_code', '-')} "
          f"started={d.get('started', '-')} ended={d.get('ended', '-')}")
PY
REMOTE
