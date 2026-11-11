#!/usr/bin/env bash
# Mac 端：把 Unicorn 上的实验结果（不含权重和大文件）经 Mac 同步到 Spark，然后在 Spark 的 omni-analysis
# 会话里为每个实验生成图表和中文报告，并生成跨实验对比。只增不删，旧数据保留。
#
#   scripts/mac/pull_a6000_results.sh [--yes]
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/spark_common.sh"
STAGE="${OMNI_STAGE:-$HOME/research-staging/unicorn}"
mkdir -p "$STAGE"
SRC="unicorn:workspace/omni-serving-research/results/"
# experiments/ latest/ runs/ jobs.jsonl and asset manifests; skip media/profiler blobs larger than 50 MB.
rsync -az --max-size=50m --exclude 'dryrun-*' --exclude 'runs/test-*' --include 'assets/*/manifest.jsonl' --exclude 'assets/*/*' "$SRC" "$STAGE/"
echo "staged $(du -sh "$STAGE" | cut -f1) in $STAGE"
"$HERE/sync_results.sh" "$STAGE/" "$SPARK_HOST:research/data/unicorn/" "${1:-}"
stamp="$(date +%Y%m%dT%H%M%S)"
"$HERE/start_spark_task.sh" "analyze-a6000-$stamp" omni-analysis \
  'for d in $HOME/research/data/unicorn/experiments/*/; do [ -f "$d/result.json" ] && .venv/bin/python scripts/analysis/analyze_experiment.py "$d"; done; .venv/bin/python scripts/analysis/compare_experiments.py configs/experiments/comparisons_a6000_3gpu.json --data-root $HOME/research/data/unicorn --out $HOME/research/data/unicorn/comparisons'
