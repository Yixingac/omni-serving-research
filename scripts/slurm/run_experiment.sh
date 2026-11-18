#!/usr/bin/env bash
# 一个实验 step 的入口（由编排器以 `srun --exact` 启动，也可以在已有分配里手动运行）。
#
#   scripts/slurm/run_experiment.sh configs/experiments/a6000/x2_minicpm_text_1gpu.json
#
# 1. 按实验配置里的 runtime_profile 加载 SGLang-Omni 运行环境；
# 2. 每个节点、每个作业只预热一次页缓存（NFS 上冷启动 import 要几分钟），用文件锁避免并发重复；
# 3. 运行实验执行器（omni_research.orchestration.experiment），结果写到 $OMNI_EXP_DIR。
set -uo pipefail
REPO="${OMNI_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONFIG="${1:?usage: run_experiment.sh CONFIG.json}"
[[ "$CONFIG" = /* ]] || CONFIG="$REPO/$CONFIG"

profile_field() {  # profile_field KEY -> value from configs/models/runtime_<profile>.json (or empty)
  python3 - "$CONFIG" "$REPO" "$1" <<'PY'
import json, os, sys
cfg = json.load(open(sys.argv[1]))
prof = cfg.get("runtime_profile")
val = ""
if prof:
    p = os.path.join(sys.argv[2], "configs", "models", f"runtime_{prof}.json")
    val = json.load(open(p)).get(sys.argv[3], "")
val = cfg.get("runtime", {}).get(sys.argv[3], val)
print(os.path.expanduser(val) if isinstance(val, str) else "")
PY
}

env_script="$(profile_field env_script)"
if [[ -n "$env_script" && -f "$env_script" ]]; then
  # shellcheck disable=SC1090
  . "$env_script"
fi
py="$(profile_field python)"
[[ -n "$py" && -x "$py" ]] || py="$(command -v python3)"

warm="$(profile_field warm_cache_script)"
if [[ -n "$warm" && -f "$warm" && "${OMNI_FAKE_SERVER:-0}" != 1 ]]; then
  marker="/tmp/omni-warm-$(id -u)-${SLURM_JOB_ID:-local}"
  (
    flock 9
    if [[ ! -e "$marker" ]]; then
      echo "$(date -Is) warming page cache on $(hostname -s)"
      bash "$warm" && touch "$marker"
    fi
  ) 9>"$marker.lock"
fi

echo "$(date -Is) experiment $(basename "$CONFIG") on $(hostname -s) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} python=$py"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$py" -m omni_research.orchestration.experiment "$CONFIG"
