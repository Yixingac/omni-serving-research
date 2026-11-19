#!/usr/bin/env python3
"""Conditional plan for P3 of the pre-presentation validation: do the run's own writes to the shared file system cause
the whole-server pauses? Same plan twice per arm, logs + events + client records on node-local /scratch or on NFS.

    python scripts/slurm/make_plan_ppv_p3.py          # write configs/experiments/ppv/ppv_p3_* and the plan file
    python scripts/slurm/make_plan_ppv_p3.py --check  # list files that differ from what would be written

Registered in research/next_steps/p0_p3_experiment_plan.md (P3, 2026-11-18): "同一开环计划跑两次：日志、事件、客户端记录写
节点本地盘（/scratch）对写 NFS；各约 1 小时、W3b 21 请求/秒附近". Execution details (p1_p2_p3_design.md, before any run):
- four 30-minute runs in the order NFS, scratch, scratch, NFS (ABBA, so a drift over the evening does not look like an
  arm effect), each a fresh server on t2_k2_c4 with the P1 runtime and no control file (stock FIFO16 batching), the
  gc log, stack registration and the stall watcher on;
- each run: smoke, then 6 open-loop phases of W3b at 21 requests/s for 300 s, server events on (stage hops and the
  code2wav enqueue time for the hand-off pause);
- the scratch arm runs the experiment with OMNI_EXP_DIR on /scratch/$USER/omni_ppv/<id> and copies the directory back
  to the orchestrator's result directory afterwards (then removes the scratch copy); /scratch on the node had 219 GB
  free on 11-18, one run writes a few GB. Imports, model weights and compiler caches stay on NFS in both arms: the
  control isolates the run's own writes, not every NFS access.
- outcome rule (registered): pauses (no log line from any process for > 1 s, while a load phase runs) disappear in
  the scratch runs and remain in the NFS runs, or PSI io / slow log stats spike at the pauses -> evidence for the
  shared file system; no pause in any run -> "root cause remains unknown; no stall was observed during the new
  measurement window" with the observation time per arm.
Only placed if budget remains after the higher priorities.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_plan_8gpu_v2 import W3B, base, plan_entry, smoke  # noqa: E402
from make_plan_ppv_p0 import CODE_DIR, command  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EXP_DIR = "configs/experiments/ppv"
RUNTIME = "unicorn_ppv_maxwait"
PLACEMENT_YAML = "configs/models/minicpm_o_4_5_exp_t2_k2_c4.yaml"
ORDER = ["nfs", "scratch", "scratch", "nfs"]
RATE_RPS = 21.0
PHASES = 6
PHASE_S = 300


def scratch_command(eid: str) -> str:
    """Run with the experiment directory on node-local /scratch, then copy it back to the orchestrator's directory."""
    return (f"cd {CODE_DIR} && SCR=/scratch/$USER/omni_ppv/$OMNI_EXP_ID && mkdir -p \"$SCR\" && "
            f"OMNI_REPO=\"$PWD\" OMNI_EXP_DIR=\"$SCR\" bash scripts/slurm/run_experiment.sh {EXP_DIR}/{eid}.json; rc=$?; "
            "rsync -a \"$SCR\"/ \"$OMNI_EXP_DIR\"/ && rm -rf \"$SCR\"; exit $rc")


def entries() -> list[tuple[dict, dict]]:
    out = []
    for position, arm in enumerate(ORDER, start=1):
        phases = [smoke()] + [{"kind": "open_loop", "name": f"w3b_r21_p{k}", "workload": W3B, "rates_rps": [RATE_RPS],
                               "duration_per_rate_s": PHASE_S, "min_requests_per_rate": int(RATE_RPS * PHASE_S),
                               "max_error_rate": 0.5, "server_events": True, "seed": 20261119 + k} for k in range(1, PHASES + 1)]
        srv = [{"name": "s0", "model": "minicpm_o_4_5", "gpus": list(range(8)), "args": ["--config", "{research_repo}/" + PLACEMENT_YAML],
                "ready_timeout_s": 2400, "env": {"SGLANG_OMNI_PORT_CLAIM_DIR": "{exp_dir}/port_claims",
                                                 "SGLANG_OMNI_EXP_STACK_DIR": "{exp_dir}/stacks"}}]
        cfg = base(f"ppv_p3_{arm}_{position}", 8, srv, phases,
                   f"P3 停顿对照第 {position} 个（顺序 NFS/本地盘/本地盘/NFS）：日志、事件、客户端记录写{'节点本地盘 /scratch' if arm == 'scratch' else ' NFS'}。"
                   "放置 t2_k2_c4，默认组批（没有控制文件），gc 日志、调用栈注册和停顿检测开。W3b 开环 21 请求/秒，6 段各 300 秒，开事件。")
        cfg["runtime_profile"] = RUNTIME
        cfg["stall_watch"] = True  # the point of P3
        out.append((cfg, {"priority": -(800 + position), "est": 2100, "timeout": 5400, "arm": arm}))
    return out


def render() -> dict[str, str]:
    es = entries()
    plan = {"plan_id": "ppv_p3_scratch_vs_nfs",
            "_comment": "P3 停顿对照（本地盘对 NFS，ABBA），只在预算有剩余时放进 Job 454290。由 scripts/slurm/make_plan_ppv_p3.py 生成。",
            "experiments": [plan_entry(c, m) for c, m in es]}
    for e, (_, meta) in zip(plan["experiments"], es):
        e["command"] = scratch_command(e["experiment_id"]) if meta["arm"] == "scratch" else command(e["experiment_id"])
    files = {"configs/experiments/plan_ppv_p3.json": json.dumps(plan, ensure_ascii=False, indent=1) + "\n"}
    for cfg, _ in es:
        files[f"{EXP_DIR}/{cfg['experiment_id']}.json"] = json.dumps(cfg, ensure_ascii=False, indent=1) + "\n"
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    files = render()
    stale = [p for p, text in files.items() if not (REPO / p).exists() or (REPO / p).read_text() != text]
    if a.check:
        for p in stale:
            print(f"stale: {p}")
        return 1 if stale else 0
    for p, text in files.items():
        (REPO / p).parent.mkdir(parents=True, exist_ok=True)
        (REPO / p).write_text(text)
    print(f"wrote {len(files)} files ({len(stale)} changed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
