#!/usr/bin/env bash
# plan v2 的 Y0：一个实验 step，记录 GPU 拓扑、NVLink 状态、点对点拷贝带宽、NVLink 对上的 NCCL all-reduce，
# 并计时预热模型权重文件（P7）。由编排器以 `srun --exact` 启动（需要整台机器分到的全部 GPU），
# 结果写到 $OMNI_EXP_DIR：raw/topo.txt、raw/nvlink_status.txt、raw/gpus.csv、raw/p2p.json、raw/model_warm.json。
#
#   scripts/slurm/env_topology.sh RUNTIME_PROFILE [MODEL ...]     例如：unicorn minicpm_o_4_5
#
# 点对点测量都在带超时的子进程里跑（gpu_p2p_check.py），挂起只会记成一条结果，不会卡住整个 step。
set -uo pipefail
REPO="${OMNI_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
PROFILE="${1:?usage: env_topology.sh RUNTIME_PROFILE [MODEL ...]}"
shift
OUT="${OMNI_EXP_DIR:-$REPO/results/experiments/$(date +%Y%m%d)_env_topology_manual}"
mkdir -p "$OUT/raw" "$OUT/logs"
log() { echo "$(date -Is) $*" | tee -a "$OUT/logs/env_topology.log"; }

prof="$REPO/configs/models/runtime_${PROFILE}.json"
field() {  # field KEY -> value from the runtime profile (paths expanded)
  python3 -c 'import json, os, sys; v = json.load(open(sys.argv[1])).get(sys.argv[2], ""); print(os.path.expanduser(v) if isinstance(v, str) else "")' "$prof" "$1"
}
env_script="$(field env_script)"
if [[ -n "$env_script" && -f "$env_script" ]]; then
  # shellcheck disable=SC1090
  . "$env_script"
fi
py="$(field python)"
[[ -n "$py" && -x "$py" ]] || py="$(command -v python3)"

log "host=$(hostname -s) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} python=$py"
nvidia-smi topo -m > "$OUT/raw/topo.txt" 2>&1
nvidia-smi nvlink --status > "$OUT/raw/nvlink_status.txt" 2>&1
nvidia-smi --query-gpu=index,uuid,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,pcie.link.width.max,memory.total \
  --format=csv > "$OUT/raw/gpus.csv" 2>&1
nvidia-smi -q > "$OUT/raw/nvidia_smi_q.txt" 2>&1

log "peer copies and NCCL all-reduce"
timeout 2400 "$py" "$REPO/scripts/slurm/gpu_p2p_check.py" --out "$OUT/raw/p2p.json" > "$OUT/logs/p2p.log" 2>&1
p2p_rc=$?
log "gpu_p2p_check exit $p2p_rc"

log "timed read of model weights (fills the node's page cache for the following steps)"
python3 - "$REPO" "$OUT/raw/model_warm.json" "$@" <<'PY' >> "$OUT/logs/env_topology.log" 2>&1
import json, os, sys, time
repo, out, models = sys.argv[1], sys.argv[2], sys.argv[3:] or ["minicpm_o_4_5"]
res = []
for m in models:
    root = os.path.expanduser(json.load(open(os.path.join(repo, "configs", "models", f"{m}.json")))["model_path"])
    t0, n, files = time.monotonic(), 0, 0
    for dp, _, fns in os.walk(root):
        for fn in fns:
            with open(os.path.join(dp, fn), "rb") as fh:
                while chunk := fh.read(16 << 20):
                    n += len(chunk)
            files += 1
    dt = time.monotonic() - t0
    res.append({"model": m, "files": files, "GB": round(n / 1e9, 2), "seconds": round(dt, 1),
                "GBps": round(n / 1e9 / dt, 2) if dt > 0 else None})
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res))
PY
log "done"

python3 - "$OUT" "$p2p_rc" "${OMNI_EXP_ID:-v2_y0_env_topology}" <<'PY'
import json, sys
out, rc, eid = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ok = rc == 0
json.dump({"status": "ok" if ok else "failed", "experiment_id": eid, "error_class": None if ok else "nonzero_exit",
           "error": None if ok else f"gpu_p2p_check.py exit {rc} (see logs/p2p.log)"}, open(f"{out}/result.json", "w"))
PY
exit "$p2p_rc"
