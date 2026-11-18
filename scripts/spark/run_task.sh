#!/usr/bin/env bash
# Spark 端：在指定 tmux 会话里启动一个有状态记录的任务。
#
#   run_task.sh TASK_ID SESSION [--cwd DIR] -- COMMAND...
#
# 每个任务一个目录 $OMNI_TASK_ROOT/TASK_ID/，里面有：
#   status.json   任务 ID、命令、主机、PID、开始/结束时间、退出码、状态（running/succeeded/failed/stopped）
#   stdout.log    标准输出
#   stderr.log    标准错误
# TASK_ID 已存在时拒绝启动，避免覆盖旧记录；重跑请换一个新的 TASK_ID（例如加 -r2）。
set -euo pipefail
TASK_ROOT="${OMNI_TASK_ROOT:-$HOME/research/tasks}"
task_id="${1:?usage: run_task.sh TASK_ID SESSION [--cwd DIR] -- COMMAND...}"
session="${2:?missing tmux session name}"
shift 2
cwd="$PWD"
if [[ "${1:-}" == "--cwd" ]]; then cwd="$2"; shift 2; fi
[[ "${1:-}" == "--" ]] && shift
[[ $# -gt 0 ]] || { echo "run_task.sh: no command given" >&2; exit 2; }
[[ "$task_id" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "run_task.sh: TASK_ID may only contain letters, digits, . _ -" >&2; exit 2; }

dir="$TASK_ROOT/$task_id"
if [[ -e "$dir" ]]; then
  echo "run_task.sh: task $task_id already exists ($dir); use a new TASK_ID to rerun" >&2
  exit 3
fi
mkdir -p "$dir"
printf '%q ' "$@" > "$dir/command.txt"
# cmd.sh records its own PID (= process group id under setsid) and then becomes the command.
{ printf 'echo $$ > %q\n' "$dir/pgid"; printf 'exec '; cat "$dir/command.txt"; printf '\n'; } > "$dir/cmd.sh"

# status.json is rewritten atomically by a tiny Python helper (python3 is part of DGX OS).
cat > "$dir/set_status.py" <<'PY'
import json, os, sys, tempfile, time
path, updates = sys.argv[1], dict(a.split("=", 1) for a in sys.argv[2:])
try:
    with open(path) as fh:
        doc = json.load(fh)
except FileNotFoundError:
    doc = {}
for k, v in updates.items():
    doc[k] = int(v) if v.lstrip("-").isdigit() and k in ("pid", "exit_code") else v
doc["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
with os.fdopen(fd, "w") as fh:
    json.dump(doc, fh, indent=2)
os.replace(tmp, path)
PY

python3 "$dir/set_status.py" "$dir/status.json" task_id="$task_id" session="$session" host="$(hostname -s)" \
  cwd="$cwd" command="$(cat "$dir/command.txt")" status=starting log_stdout="$dir/stdout.log" log_stderr="$dir/stderr.log"

# The wrapper runs inside tmux. setsid gives the command its own process group, so stop_task.sh
# can stop exactly this task and nothing else.
cat > "$dir/wrapper.sh" <<EOF
#!/usr/bin/env bash
cd $(printf '%q' "$cwd")
python3 "$dir/set_status.py" "$dir/status.json" status=running started="\$(date -Is)" wrapper_pid=\$\$
setsid bash "$dir/cmd.sh" > "$dir/stdout.log" 2> "$dir/stderr.log" &
child=\$!
python3 "$dir/set_status.py" "$dir/status.json" pid=\$child
wait \$child
rc=\$?
if [[ -e "$dir/STOPPED" ]]; then st=stopped; elif [[ \$rc -eq 0 ]]; then st=succeeded; else st=failed; fi
python3 "$dir/set_status.py" "$dir/status.json" status=\$st exit_code=\$rc ended="\$(date -Is)"
echo "[task $task_id finished: \$st, exit \$rc]"
EOF
chmod +x "$dir/wrapper.sh"

tmux has-session -t "$session" 2>/dev/null || tmux new-session -d -s "$session"
tmux new-window -d -t "$session" -n "$task_id" "bash '$dir/wrapper.sh'; exec bash"
echo "started task $task_id in tmux session $session (logs: $dir)"
