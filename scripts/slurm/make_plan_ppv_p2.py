#!/usr/bin/env python3
"""Configs and plans for P2 of the pre-presentation validation: is the side-stream hidden-state copy bit-exact?

    python scripts/slurm/make_plan_ppv_p2.py          # write configs/experiments/ppv/ppv_p2_* and the two plan files
    python scripts/slurm/make_plan_ppv_p2.py --check  # list files that differ from what would be written

Registered in research/next_steps/p0_p3_experiment_plan.md (P2, 2026-11-18):
- one correctness run with SGLANG_OMNI_SPEECH_LOOKAHEAD=sidecopy_verify on 实验 11's placement (2+6) and 实验 11's
  check phases (greedy workloads at concurrency 1/8/32/64, text/speech, short/long and seed/no-seed mixes, the
  client-timeout abort phase and the phase after it, W3XL), events and digests on; every finished request compares
  the side-stream copy with a launch-stream copy of the same snapshot rows (exp_hidden_verify events and warnings);
- gate: every finished request of every phase bit-exact, rows as expected, 0 errors -> pass; any mismatch -> fail,
  report the evidence, no performance runs;
- only after a pass: 实验 11's registered nine performance runs (3 blocks x A/B/C in its Latin square), unchanged
  except for new IDs (ppv_p2_perf_*), from 实验 11's runtime, whose sidecopy code is the code the verify run checks.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_plan_8gpu_v2 import base, plan_entry  # noqa: E402
from make_plan_exp09_11 import PORT_CLAIM_ENV, exp11_check_phases, exp11_entries, placement_server_s26  # noqa: E402
from make_plan_ppv_p0 import command  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EXP_DIR = "configs/experiments/ppv"
RUNTIME = "unicorn_ppv_verify"
VERIFY_ID = "ppv_p2_verify"


def verify_entry() -> tuple[dict, dict]:
    srv = placement_server_s26()
    srv[0]["env"] = {"SGLANG_OMNI_SPEECH_LOOKAHEAD": "sidecopy_verify", **PORT_CLAIM_ENV,
                     "SGLANG_OMNI_EXP_TRACE": "thinker,c2w,digest"}
    cfg = base(VERIFY_ID, 8, srv, exp11_check_phases(),
               "P2 逐位正确性检查：SGLANG_OMNI_SPEECH_LOOKAHEAD=sidecopy_verify（旁路复制 + 启动流上的参照拷贝，逐位比较），2+6，"
               "实验 11 的检查段落（贪心负载并发 1/8/32/64，文字/语音、短/长、带/不带 seed 混合，客户端超时中止段及之后的段，W3XL），"
               "开事件和 digest。")
    cfg["runtime_profile"] = RUNTIME
    return cfg, {"priority": -700, "est": 1200, "timeout": 3600}


def perf_entries() -> list[tuple[dict, dict]]:
    out = []
    for cfg, meta in exp11_entries():
        c = copy.deepcopy(cfg)
        c["experiment_id"] = c["experiment_id"].replace("exp11_perf_", "ppv_p2_perf_")
        c["description"] = "P2 通过后才运行的性能对比（实验 11 登记的 9 个运行，只换 ID）：" + c["description"]
        out.append((c, {**meta, "priority": meta["priority"] - 400}))
    return out


def plan_file(plan_id: str, comment: str, entries: list[tuple[dict, dict]]) -> str:
    plan = {"plan_id": plan_id, "_comment": comment + " 通过 plan.d 放进 Job 454290；命令从集群上本分支的 worktree 运行。"
            "由 scripts/slurm/make_plan_ppv_p2.py 生成。", "experiments": [plan_entry(c, m) for c, m in entries]}
    for e in plan["experiments"]:
        e["command"] = command(e["experiment_id"])
    return json.dumps(plan, ensure_ascii=False, indent=1) + "\n"


def render() -> dict[str, str]:
    verify, perf = [verify_entry()], perf_entries()
    files = {"configs/experiments/plan_ppv_p2_verify.json": plan_file("ppv_p2_verify", "P2 逐位正确性检查（1 个运行）。", verify),
             "configs/experiments/plan_ppv_p2_perf.json": plan_file(
                 "ppv_p2_perf", "P2 的性能对比（实验 11 登记的 9 个运行）。只在逐位检查通过后放进队列。", perf)}
    for cfg, _ in verify + perf:
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
