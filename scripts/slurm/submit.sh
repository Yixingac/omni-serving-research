#!/usr/bin/env bash
# Submit one bounded allocation for this project (mode B in docs/experiments/REPRODUCIBILITY.md).
# Mode A (reuse an existing allocation) needs no submission: run run_all.sh inside it.
#
#   scripts/slurm/submit.sh hold --gpus 4 --hours 10            # GPUs held for interactive srun steps
#   scripts/slurm/submit.sh run  --gpus 8 --hours 16 --plan configs/experiments/main.yaml
#   options: --partition P  --gpu-type T  --cpus-per-gpu N  --mem-per-gpu G  --nodelist N  --dry-run
#
# Hard caps (project budget): at most 8 GPUs and 16 h per allocation. Partition, GPU type and
# account are checked against Slurm before submitting; nothing cluster-specific is assumed.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
MAX_GPUS=8
MAX_HOURS=16

mode="${1:?usage: submit.sh hold|run [options]}"; shift
gpus=1 hours=4 partition="" gpu_type="" cpus_per_gpu=6 mem_per_gpu=64 nodelist="" plan="" dry=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) gpus="$2"; shift 2 ;;
    --hours) hours="$2"; shift 2 ;;
    --partition) partition="$2"; shift 2 ;;
    --gpu-type) gpu_type="$2"; shift 2 ;;
    --cpus-per-gpu) cpus_per_gpu="$2"; shift 2 ;;
    --mem-per-gpu) mem_per_gpu="${2%G}"; shift 2 ;;
    --nodelist) nodelist="$2"; shift 2 ;;
    --plan) plan="$2"; shift 2 ;;
    --dry-run) dry=1; shift ;;
    *) echo "submit.sh: unknown option $1" >&2; exit 2 ;;
  esac
done
case "$mode" in hold|run) ;; *) echo "submit.sh: mode must be hold or run" >&2; exit 2 ;; esac
[[ "$mode" == run && -z "$plan" ]] && { echo "submit.sh: run mode needs --plan" >&2; exit 2; }
(( gpus >= 1 && gpus <= MAX_GPUS )) || { echo "submit.sh: --gpus must be 1..$MAX_GPUS" >&2; exit 2; }
awk -v h="$hours" -v m="$MAX_HOURS" 'BEGIN{exit !(h>0 && h<=m)}' || { echo "submit.sh: --hours must be in (0, $MAX_HOURS]" >&2; exit 2; }

# --- discovery: which partitions and GPU types can this user actually request? -----------------
user="$(id -un)"
account="$(sacctmgr -nP show assoc user="$user" format=account 2>/dev/null | head -1)"
if [[ -z "$partition" ]]; then
  # Prefer a partition restricted to one of the user's groups (an owner partition) over shared ones.
  for p in $(sinfo -h -o %P | tr -d '*' | grep -v -- '-interactive$' | sort -u); do
    groups_allowed="$(scontrol show partition "$p" 2>/dev/null | grep -oP 'AllowGroups=\K\S+')"
    [[ "$groups_allowed" == ALL || -z "$groups_allowed" ]] && continue
    for g in $(id -Gn); do [[ ",$groups_allowed," == *",$g,"* ]] && partition="$p" && break 2; done
  done
fi
[[ -n "$partition" ]] || { echo "submit.sh: no owner partition found; pass --partition" >&2; exit 2; }
gres_on_partition="$(sinfo -h -p "$partition" -o %G | tr ',' '\n' | grep -oP '^gpu:\K[^:(]+' | sort -u)"
if [[ -z "$gpu_type" ]]; then
  gpu_type="$(grep -m1 -i a6000 <<<"$gres_on_partition" || true)"
fi
grep -qx -- "$gpu_type" <<<"$gres_on_partition" || {
  echo "submit.sh: GPU type '$gpu_type' not on partition $partition (has: $(tr '\n' ' ' <<<"$gres_on_partition"))" >&2; exit 2; }

cpus=$(( gpus * cpus_per_gpu ))
mem=$(( gpus * mem_per_gpu ))
minutes=$(awk -v h="$hours" 'BEGIN{printf "%d", h*60}')
stamp="$(date +%Y%m%dT%H%M%S)"
run_dir="$REPO/results/runs/${mode}-${stamp}"
mkdir -p "$run_dir"

script="$run_dir/job.sbatch"
cat > "$script" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=omni-research-$mode
#SBATCH --comment=omni-serving-research
#SBATCH --partition=$partition
#SBATCH --account=$account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:$gpu_type:$gpus
#SBATCH --cpus-per-task=$cpus
#SBATCH --mem=${mem}G
#SBATCH --time=$minutes
#SBATCH --signal=B:USR1@900
#SBATCH --output=$run_dir/slurm-%j.out
#SBATCH --error=$run_dir/slurm-%j.out
${nodelist:+#SBATCH --nodelist=$nodelist}
export OMNI_RUN_DIR="$run_dir"
export OMNI_REPO="$REPO"
exec bash "$REPO/scripts/slurm/run_all.sh" "$mode" ${plan:+"$REPO/$plan"}
EOF

echo "partition=$partition account=$account gres=gpu:$gpu_type:$gpus cpus=$cpus mem=${mem}G time=${hours}h"
echo "estimate: $(sbatch --test-only "$script" 2>&1 | head -1)"
if (( dry )); then echo "dry run: $script"; exit 0; fi

jobid="$(sbatch --parsable "$script")"
echo "submitted job $jobid ($script)"
# Ledger of jobs this project owns; cleanup.sh only ever touches jobs listed here.
printf '{"job_id": "%s", "mode": "%s", "partition": "%s", "gres": "gpu:%s:%s", "cpus": %s, "mem_gb": %s, "hours": %s, "run_dir": "%s", "submitted": "%s"}\n' \
  "$jobid" "$mode" "$partition" "$gpu_type" "$gpus" "$cpus" "$mem" "$hours" "$run_dir" "$(date -Is)" \
  >> "$REPO/results/jobs.jsonl"
