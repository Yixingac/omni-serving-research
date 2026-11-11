#!/usr/bin/env bash
# Body of every allocation submitted by submit.sh (or run it yourself inside an existing one).
#
#   run_all.sh hold          preflight, then keep the GPUs for `srun --jobid` experiment steps
#   run_all.sh run PLAN      preflight, then the experiment orchestrator on PLAN
#
# Shutdown: Slurm sends USR1 ~15 min before the limit (--signal=B:USR1@900). We also watch the
# remaining time ourselves (squeue %L) in case the signal is lost, and stop at OMNI_STOP_MARGIN_S.
set -uo pipefail

mode="${1:?usage: run_all.sh hold|run [PLAN]}"
plan="${2:-}"
REPO="${OMNI_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_DIR="${OMNI_RUN_DIR:-$REPO/results/runs/${mode}-${SLURM_JOB_ID:-local}}"
STOP_MARGIN_S="${OMNI_STOP_MARGIN_S:-900}"
mkdir -p "$RUN_DIR"
cd "$REPO"
log() { echo "$(date -Is) [run_all:$mode] $*" | tee -a "$RUN_DIR/run_all.log"; }

time_left_s() {  # remaining walltime of this job in seconds (squeue %L: [d-]hh:mm:ss)
  local tl
  tl="$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null | head -1)"
  [[ -z "$tl" || "$tl" == INVALID || "$tl" == UNLIMITED ]] && { echo 999999; return; }
  awk -v t="$tl" 'BEGIN{d=0; if (index(t,"-")) {split(t,a,"-"); d=a[1]; t=a[2]} n=split(t,p,":");
    s=0; for(i=1;i<=n;i++) s=s*60+p[i]; print d*86400+s}'
}

child=""
stopping=0
on_signal() {
  (( stopping )) && return
  stopping=1
  log "received stop signal; asking the orchestrator to finish and save"
  touch "$RUN_DIR/STOP"
  [[ -n "$child" ]] && kill -TERM "$child" 2>/dev/null
}
trap on_signal USR1 TERM INT

log "job ${SLURM_JOB_ID:-none} on $(hostname -s); gpus=${SLURM_GPUS_ON_NODE:-?} cpus=${SLURM_CPUS_PER_TASK:-?} time_left=$(time_left_s)s"
bash "$REPO/scripts/slurm/preflight.sh" "$RUN_DIR/preflight" >> "$RUN_DIR/run_all.log" 2>&1
log "preflight done"

case "$mode" in
  hold)
    # Nothing runs here; experiment steps are launched into this job with srun --jobid.
    while [[ ! -e "$RUN_DIR/STOP" ]] && (( $(time_left_s) > STOP_MARGIN_S )); do
      sleep 30 & child=$!; wait "$child"; child=""
    done
    ;;
  run)
    [[ -f "$plan" ]] || { log "plan not found: $plan"; exit 2; }
    PY="${OMNI_PYTHON:-python3}"
    PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m omni_research.orchestration.cli run \
      --plan "$plan" --run-dir "$RUN_DIR" --stop-margin-s "$STOP_MARGIN_S" >> "$RUN_DIR/orchestrator.log" 2>&1 &
    child=$!
    # `wait` returns early when a trapped signal arrives; keep waiting until the child exits.
    while kill -0 "$child" 2>/dev/null; do wait "$child"; done
    log "orchestrator exited"
    ;;
esac

log "cleanup: stopping steps this job still runs"
bash "$REPO/scripts/slurm/cleanup.sh" --job "${SLURM_JOB_ID:-}" --steps-only >> "$RUN_DIR/run_all.log" 2>&1
log "done"
