#!/usr/bin/env bash
# Mac 端：在 Unicorn、Mac、Spark 之间同步实验结果目录。只增加或更新文件，从不删除目标端文件
# （不使用 rsync --delete）。先列出将传输的文件数和大小，确认目标空间足够后才真正复制。
#
#   scripts/mac/sync_results.sh SRC DST [--yes]
#   SRC/DST 例：unicorn:workspace/omni-serving-research/results/experiments/
#              ./results/experiments/
#              s01-remote:research/data/experiments/
# 远端到远端的同步经由 Mac 中转（Spark 上没有、也不应该放 Unicorn 的凭据）。
set -euo pipefail
src="${1:?usage: sync_results.sh SRC DST [--yes]}"
dst="${2:?missing DST}"
yes="${3:-}"
EXCLUDES=(--exclude '*.safetensors' --exclude '*.bin' --exclude '*.pt' --exclude '.cache/' --exclude '__pycache__/')

is_remote() { [[ "$1" == *:* ]]; }
if is_remote "$src" && is_remote "$dst"; then
  stage="$(mktemp -d)/stage/"
  "$0" "$src" "$stage" "$yes"
  "$0" "$stage" "$dst" "$yes"
  exit
fi

echo "预览：$src -> $dst"
rsync -az --dry-run --stats "${EXCLUDES[@]}" "$src" "$dst" | grep -E "Number of (regular )?files transferred|Total transferred file size" || true
if [[ "$yes" != "--yes" ]]; then
  read -r -p "继续复制？[y/N] " ans
  [[ "$ans" == y || "$ans" == Y ]] || { echo "已取消"; exit 1; }
fi
rsync -az "${EXCLUDES[@]}" "$src" "$dst"
echo "完成：$src -> $dst"
