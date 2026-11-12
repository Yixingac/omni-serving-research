#!/usr/bin/env bash
# Spark 端：用实验执行器运行一个 Spark 实验（功能验证，不是 A6000 正式测量）。
#
#   scripts/spark/run_experiment.sh configs/experiments/spark/s1_minicpm_text_smoke.json
#
# 开始前检查 GPU 上没有其他计算进程（例如租户或别人的任务）；有就放弃，不和别人抢 GPU。
# 结果写到 $PROJECT_DATA_ROOT/experiments/<日期>_<实验名>_runNN（默认 ~/research/data）。
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:?usage: run_experiment.sh CONFIG.json}"
[[ "$CONFIG" = /* ]] || CONFIG="$REPO/$CONFIG"
export PROJECT_DATA_ROOT="${PROJECT_DATA_ROOT:-$HOME/research/data}"
export OMNI_REPO="$REPO"
PY="$HOME/research/omni-runtime/.conda-runtime/bin/python"

others="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)"
if [[ -n "$others" ]]; then
  echo "GPU 上已有计算进程，放弃运行以免干扰：" >&2
  echo "$others" >&2
  exit 75
fi

name="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["experiment_id"])' "$CONFIG")"
stamp="$(date +%Y%m%d)"
n=1
while [[ -e "$PROJECT_DATA_ROOT/experiments/${stamp}_${name}_run$(printf %02d $n)" ]]; do n=$((n + 1)); done
out="$PROJECT_DATA_ROOT/experiments/${stamp}_${name}_run$(printf %02d $n)"
mkdir -p "$out"
echo "$(date -Is) $name -> $out"
export OMNI_EXP_DIR="$out"
export OMNI_PORT="${OMNI_PORT:-31000}"
PYTHONPATH="$REPO/src" exec "$PY" -m omni_research.orchestration.experiment "$CONFIG"
