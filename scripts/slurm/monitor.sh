#!/usr/bin/env bash
# 在 Unicorn 登录节点的 tmux 里运行：跟踪一个作业的状态，不依赖任何人或 Agent 在线。
#
#   tmux new-session -d -s omni-monitor "bash scripts/slurm/monitor.sh JOBID RUN_DIR [INTERVAL_S]"
#
# 状态：SLURM_PENDING → SLURM_RUNNING → SLURM_COMPLETED / SLURM_FAILED / SLURM_CANCELLED / SLURM_TIMEOUT ...
# 每一轮把状态、状态变化历史和编排器的实验状态计数写进 RUN_DIR/monitor/state.json。
# 作业结束后：写 summary.txt（编排器状态表），对每个实验目录生成汇总表和中文 report.md
# （登录节点上没有 matplotlib，图表之后在 Spark 上生成），然后写 DONE 并退出。
set -uo pipefail
JOB="${1:?usage: monitor.sh JOBID RUN_DIR [INTERVAL_S]}"
RUN_DIR="${2:?missing RUN_DIR}"
INTERVAL="${3:-300}"
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
MON="$RUN_DIR/monitor"
PY="${OMNI_ANALYSIS_PYTHON:-$HOME/workspace/sglang-omni-learning/.venv/bin/python}"
mkdir -p "$MON"
[[ -e "$MON/DONE" ]] && { echo "monitor already finished for job $JOB"; exit 0; }

slurm_state() {
  local s
  s="$(squeue -h -j "$JOB" -o %T 2>/dev/null | head -1)"
  if [[ -z "$s" ]]; then
    s="$(sacct -n -X -j "$JOB" -o State%30 2>/dev/null | head -1 | awk '{print $1}')"
  fi
  echo "${s:-UNKNOWN}"
}

record() {  # record STATE
  python3 - "$MON/state.json" "$JOB" "$1" "$RUN_DIR/orchestrator/state.json" <<'PY'
import json, os, sys, time
path, job, state, orch = sys.argv[1:5]
doc = {}
if os.path.exists(path):
    doc = json.load(open(path))
phase = "SLURM_" + state.split("+")[0] if state != "UNKNOWN" else "SLURM_UNKNOWN"
now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
hist = doc.get("history", [])
if not hist or hist[-1]["phase"] != phase:
    hist.append({"phase": phase, "slurm_state": state, "at": now})
counts = None
if os.path.exists(orch):
    st = json.load(open(orch))
    counts = {}
    for e in st.get("experiments", {}).values():
        counts[e["status"]] = counts.get(e["status"], 0) + 1
doc.update(job_id=job, phase=phase, slurm_state=state, checked_at=now, history=hist, experiments=counts)
tmp = path + ".tmp"
json.dump(doc, open(tmp, "w"), indent=2)
os.replace(tmp, path)
print(f"{now} job {job}: {phase} experiments={counts}", flush=True)
PY
}

finish() {
  echo "$(date -Is) job $JOB ended; summarizing"
  if [[ -f "$RUN_DIR/orchestrator/state.json" ]]; then
    PYTHONPATH="$REPO/src" python3 -m omni_research.orchestration.cli status --run-dir "$RUN_DIR/orchestrator" > "$MON/summary.txt" 2>&1
    dirs="$(python3 -c 'import json,sys; st=json.load(open(sys.argv[1])); print("\n".join(e["output_dir"] for e in st["experiments"].values() if e.get("output_dir")))' "$RUN_DIR/orchestrator/state.json")"
    if [[ -n "$dirs" ]]; then
      # shellcheck disable=SC2086
      "$PY" "$REPO/scripts/analysis/analyze_experiment.py" --no-figures $dirs >> "$MON/summary.txt" 2>&1
    fi
  fi
  sacct -j "$JOB" -o JobID%20,JobName%40,State%15,Elapsed,AllocTRES%60 >> "$MON/summary.txt" 2>&1
  date -Is > "$MON/DONE"
}

while true; do
  st="$(slurm_state)"
  record "$st"
  case "$st" in
    PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|REQUEUED|RESIZING|UNKNOWN) sleep "$INTERVAL" ;;
    *) finish; break ;;
  esac
done
