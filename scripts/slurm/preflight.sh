#!/usr/bin/env bash
# Read-only hardware/Slurm discovery. Run it inside an allocation (it fans out one task per
# node with `srun --overlap`) or on any single host. It never changes system state and only
# inspects GPUs that this job's cgroup exposes.
#
#   scripts/slurm/preflight.sh [OUT_DIR]
#
# Writes OUT_DIR/<host>/*.txt (raw command output) and OUT_DIR/topology.json (parsed summary).
set -uo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
OUT_DIR="${1:-$REPO/results/preflight/${SLURM_JOB_ID:-local}-$(date +%Y%m%dT%H%M%S)}"
mkdir -p "$OUT_DIR"

collect_node() {
  local out="$1/$(hostname -s)"
  mkdir -p "$out"
  run() {  # run NAME CMD...; never fails the script
    local name="$1"; shift
    { echo "\$ $*"; timeout 60 "$@" 2>&1; echo "[exit=$?]"; } > "$out/$name.txt"
  }
  run hostname hostname -f
  run uname uname -a
  run lscpu lscpu
  run numa bash -c 'command -v numactl >/dev/null && numactl -H || cat /sys/devices/system/node/node*/cpulist'
  run free free -b
  run df df -h /home "${HOME}" /scratch /tmp /dev/shm
  run cpuset bash -c 'taskset -cp $$; cat /proc/self/status | grep -E "Cpus_allowed_list|Mems_allowed_list"'
  run cgroup_mem bash -c 'cg=$(awk -F: "\$1==\"0\"{print \$3}" /proc/self/cgroup); for f in memory.max memory.current; do echo "$f=$(cat /sys/fs/cgroup${cg}/$f 2>/dev/null || cat /sys/fs/cgroup${cg%/*}/$f 2>/dev/null)"; done'
  run env_slurm bash -c 'env | grep -E "^(SLURM_|CUDA_VISIBLE_DEVICES|GPU_DEVICE_ORDINAL|SLURM_STEP_GPUS)" | grep -v -E "TOKEN|KEY" | sort'
  run nvidia_smi_L nvidia-smi -L
  run nvidia_smi_query nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.used,ecc.mode.current,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,driver_version,clocks.max.sm,clocks.max.mem,power.limit,compute_mode,persistence_mode --format=csv
  run nvidia_topo nvidia-smi topo -m
  run nvidia_nvlink nvidia-smi nvlink -s
  run nvidia_smi nvidia-smi
  run driver_params bash -c 'grep -E "RmProfilingAdminOnly" /proc/driver/nvidia/params; cat /proc/driver/nvidia/version'
  run nvcc bash -c 'command -v nvcc && nvcc --version'
  run nsight bash -c 'for t in nsys ncu; do printf "%s: " "$t"; command -v "$t" || ls -d /usr/local/cuda*/bin/$t /share/apps/*/cuda*/bin/$t /opt/nvidia/nsight-*/bin/$t /share/apps/software/*/bin/$t 2>/dev/null | head -3 || echo none; done'
  run net bash -c 'ip -br link; for n in /sys/class/net/*; do b=$(basename "$n"); [ "$b" = lo ] && continue; echo "$b speed=$(cat "$n/speed" 2>/dev/null) operstate=$(cat "$n/operstate" 2>/dev/null)"; done; ls /dev/infiniband 2>&1'
  run python bash -c 'command -v python3; python3 --version'
}
export -f collect_node

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null; then
  # One task per allocated node; --overlap so it can run next to an existing step, and the
  # task inherits only this job's GPUs (Slurm's cgroup decides what nvidia-smi can see).
  srun --overlap --ntasks-per-node=1 --nodes="${SLURM_JOB_NUM_NODES:-1}" \
       --gres="${PREFLIGHT_GRES:-gpu:${SLURM_GPUS_ON_NODE:-0}}" \
       bash -c "collect_node '$OUT_DIR'" || collect_node "$OUT_DIR"
  scontrol show job "$SLURM_JOB_ID" > "$OUT_DIR/scontrol_job.txt" 2>&1 || true
  squeue -h -j "$SLURM_JOB_ID" -o "%L" > "$OUT_DIR/time_left.txt" 2>&1 || true
else
  collect_node "$OUT_DIR"
fi

python3 "$REPO/scripts/slurm/parse_preflight.py" "$OUT_DIR" > "$OUT_DIR/topology.json" \
  && echo "preflight: wrote $OUT_DIR/topology.json" \
  || echo "preflight: raw output in $OUT_DIR (parse failed)"
