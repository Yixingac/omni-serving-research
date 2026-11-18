#!/usr/bin/env bash
# Spark 端：幂等地准备工作目录、tmux 会话和研究代码的 Python 环境。不需要 sudo，不改系统配置。
#
#   bootstrap.sh            仓库 + tmux 会话 + 研究代码 venv（.venv，仅 CPU 依赖）
#   bootstrap.sh runtime    另外构建 SGLang-Omni 运行环境（见 scripts/spark/setup_runtime.sh）
#
# 已存在的仓库只做 fetch 和状态报告，不会覆盖本地修改。
set -euo pipefail
REPO_URL="${OMNI_REPO_URL:-https://github.com/Yixingac/omni-serving-research.git}"
BRANCH="${OMNI_BRANCH:-research/agent-trace-end-to-end}"
REPO="${OMNI_REPO:-$HOME/research/repos/omni-serving-research}"
UV="$HOME/.local/bin/uv"

for s in omni-dev omni-run omni-analysis; do
  tmux has-session -t "$s" 2>/dev/null || tmux new-session -d -s "$s" -c "$HOME/research"
done
echo "tmux sessions: $(tmux ls -F '#S' | tr '\n' ' ')"

mkdir -p "$(dirname "$REPO")" "$HOME/research/tasks" "$HOME/research/data"
if [[ -d "$REPO/.git" ]]; then
  git -C "$REPO" fetch -q origin
  dirty="$(git -C "$REPO" status --porcelain | wc -l)"
  if [[ "$dirty" -eq 0 && "$(git -C "$REPO" rev-parse --abbrev-ref HEAD)" == "$BRANCH" ]]; then
    git -C "$REPO" merge -q --ff-only "origin/$BRANCH" || echo "repo: not fast-forwardable, left as is"
  else
    echo "repo: $dirty local changes or other branch; not updating"
  fi
else
  git clone -q -b "$BRANCH" "$REPO_URL" "$REPO"
fi
echo "repo: $(git -C "$REPO" rev-parse --abbrev-ref HEAD) @ $(git -C "$REPO" rev-parse --short HEAD)"

cd "$REPO"
[[ -x .venv/bin/python ]] || "$UV" venv -q --python 3.12 .venv
"$UV" pip install -q --python .venv/bin/python -e '.[dev]'
.venv/bin/python -c 'import omni_research, aiohttp, matplotlib; print("research venv ok:", omni_research.__version__)'

if [[ "${1:-}" == "runtime" ]]; then
  bash scripts/spark/setup_runtime.sh
fi
