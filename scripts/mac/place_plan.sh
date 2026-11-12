#!/usr/bin/env bash
# Mac 端：把仓库里的一个实验计划放进 Unicorn 上某个 hold 作业的运行目录。
#
#   scripts/mac/place_plan.sh JOBID RUN_DIR PLAN [NAME]
#     RUN_DIR  相对 Unicorn 上仓库目录的运行目录，例如 results/runs/hold8-20261112T134123
#     PLAN     仓库里的计划文件，例如 configs/experiments/plan_a6000_8gpu_v2.json（必须已推送）
#     NAME     作业已在运行时，写进 plan.d 的文件名（默认取 PLAN 的文件名）
#
# 作业还在排队（PENDING）：先在 Unicorn 上 `git pull --ff-only` 更新代码，记到 RUN_DIR/code_state.txt；
#   再用编排器自己的 load_plan 检查计划，最后先写临时文件、再 mv 成 RUN_DIR/autostart_plan.json。
#   作业开始时 run_all.sh 会自动运行它。
# 作业已经在运行（RUNNING）：不更新代码（正在跑的实验不能换代码），只把计划写进
#   RUN_DIR/orchestrator/plan.d/NAME.json，编排器会追加其中的新实验（已有的 experiment_id 不会被改写）。
#   计划里新加的配置文件必须已经在 Unicorn 的检出里；否则先按实验间隙手动更新代码，并在 code_state.txt 里记一笔。
# 其他状态（已结束、取消）：什么都不做，退出码 3。
set -euo pipefail
jobid="${1:?usage: place_plan.sh JOBID RUN_DIR PLAN [NAME]}"
run_dir="${2:?missing RUN_DIR}"
plan="${3:?missing PLAN}"
name="${4:-$(basename "$plan" .json)}"
UNICORN_HOST="${UNICORN_HOST:-unicorn}"
UNICORN_REPO="${UNICORN_REPO:-workspace/omni-serving-research}"   # 相对 Unicorn 上的 $HOME

ssh -o BatchMode=yes -o ConnectTimeout=20 "$UNICORN_HOST" bash -l -s -- "$jobid" "$run_dir" "$plan" "$name" "$UNICORN_REPO" <<'REMOTE'
set -euo pipefail
jobid="$1"; run_dir="$2"; plan="$3"; name="$4"; repo="$5"
cd "$HOME/$repo"
[[ -d "$run_dir" ]] || { echo "no run dir $run_dir" >&2; exit 2; }
state="$(sacct -u "$USER" -X -n -P -j "$jobid" -o State | head -1)"
echo "job $jobid: ${state:-unknown}"
check_plan() {
  PYTHONPATH="$PWD/src" python3 -c 'import sys; from omni_research.orchestration.plan import load_plan
p = load_plan(sys.argv[1]); print(f"plan {p.plan_id}: {len(p.experiments)} experiments")' "$1"
  python3 - "$1" <<'PY'
import json, os, re, sys
doc = json.load(open(sys.argv[1]))
missing = [m.group(1) for e in doc["experiments"] for m in [re.search(r"run_experiment\.sh (\S+\.json)", e["command"])] if m and not os.path.exists(m.group(1))]
if missing:
    sys.exit(f"configs missing in this checkout: {missing}")
PY
}
case "$state" in
  PENDING)
    old="$(git rev-parse --short HEAD)"
    git pull -q --ff-only origin "$(git rev-parse --abbrev-ref HEAD)"
    new="$(git rev-parse --short HEAD)"
    check_plan "$plan"
    printf '%s checkout %s -> %s (git pull --ff-only, job %s still PENDING); autostart_plan.json <- %s\n' \
      "$(date -Is)" "$old" "$new" "$jobid" "$plan" >> "$run_dir/code_state.txt"
    cp "$plan" "$run_dir/.autostart_plan.json.tmp"
    mv "$run_dir/.autostart_plan.json.tmp" "$run_dir/autostart_plan.json"
    echo "placed $run_dir/autostart_plan.json (code $new)"
    ;;
  RUNNING)
    check_plan "$plan"
    mkdir -p "$run_dir/orchestrator/plan.d"
    target="$run_dir/orchestrator/plan.d/$name.json"
    [[ -e "$target" ]] && { echo "$target exists; pick another NAME" >&2; exit 2; }
    cp "$plan" "$run_dir/orchestrator/plan.d/.$name.json.tmp"
    mv "$run_dir/orchestrator/plan.d/.$name.json.tmp" "$target"
    printf '%s plan.d/%s.json <- %s at code %s (job running; code not changed)\n' \
      "$(date -Is)" "$name" "$plan" "$(git rev-parse --short HEAD)" >> "$run_dir/code_state.txt"
    echo "placed $target"
    ;;
  *)
    echo "job $jobid is ${state:-not found}; nothing placed" >&2
    exit 3
    ;;
esac
REMOTE
