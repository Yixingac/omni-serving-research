#!/usr/bin/env bash
# Stop only what this project started. A job qualifies when it is in results/jobs.jsonl, belongs to
# the current user, and carries the comment "omni-serving-research".
#
#   cleanup.sh --job JOBID --steps-only   cancel the job's srun steps (not the batch step)
#   cleanup.sh --job JOBID                cancel the whole job
#   cleanup.sh --all                      cancel every running job from the ledger
set -uo pipefail
REPO="${OMNI_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
LEDGER="$REPO/results/jobs.jsonl"
job="" steps_only=0 all=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --job) job="$2"; shift 2 ;;
    --steps-only) steps_only=1; shift ;;
    --all) all=1; shift ;;
    *) echo "cleanup.sh: unknown option $1" >&2; exit 2 ;;
  esac
done

owned() {  # owned JOBID -> 0 if the job is ours by ledger, user and comment
  local j="$1"
  [[ -n "$j" ]] || return 1
  grep -q "\"job_id\": \"$j\"" "$LEDGER" 2>/dev/null || return 1
  local info
  info="$(squeue -h -j "$j" -o '%u|%k' 2>/dev/null)"
  [[ -z "$info" ]] && return 1  # not running any more
  [[ "${info%%|*}" == "$(id -un)" && "${info#*|}" == omni-serving-research ]]
}

targets=()
if (( all )); then
  while read -r j; do targets+=("$j"); done < <(grep -oP '"job_id": "\K[^"]+' "$LEDGER" 2>/dev/null)
else
  targets=("$job")
fi

for j in "${targets[@]}"; do
  if ! owned "$j"; then echo "skip $j (not a running job owned by this project)"; continue; fi
  if (( steps_only )); then
    for s in $(squeue -h -s -j "$j" -o %i 2>/dev/null | grep -vE '\.(batch|extern)$'); do
      echo "cancel step $s"; scancel "$s"
    done
  else
    echo "cancel job $j"; scancel "$j"
  fi
done
