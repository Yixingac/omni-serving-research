#!/usr/bin/env python3
"""Configs and plans for 实验 9–11 (best GPU split, code2wav batching, thinker async copy).

    python scripts/slurm/make_plan_exp09_11.py          # write configs/models/*.yaml, configs/experiments/exp09_11/...
    python scripts/slurm/make_plan_exp09_11.py --check  # list files that differ from what would be written

实验 9 reuses the placement files of 实验 7 (configs/models/minicpm_o_4_5_exp_t2_*.yaml, written by
make_plan_exp06_08.py) and adds t2_k1_c5. Every placement keeps the talker's and code2wav's memory fractions of
the S2:6 baseline (0.40 each). The open-loop rates of the confirmatory runs come from the pilot by the rule
registered in research/01_性能瓶颈/实验09_最佳GPU分工/hypotheses.md.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_plan_8gpu_v2 import CODE2WAV_FRACTION, TALKER_FRACTION, W3B, base, plan_entry, smoke  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EXP_DIR = "configs/experiments/exp09_11"
W3L = "configs/workloads/w3l_long_speech.json"
SEED = 20261118  # every randomized order of 实验 9–11 derives from this seed (hypotheses.md)
INSTR_RUNTIME = "unicorn_instr"
PLAN_EXP09_PILOT = "configs/experiments/plan_exp09_pilot.json"

# 方案 number in the reports -> placement name; name -> (thinker GPUs, talker GPUs, code2wav GPUs).
EXP09_ARMS = {1: "t2_k6_shared", 2: "t2_k2_c4", 3: "t2_k1_c5", 4: "t2_k3_c3"}
PLACEMENTS = {
    "t2_k6_shared": ([0, 1], [2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7]),
    "t2_k2_c4": ([0, 1], [2, 3], [4, 5, 6, 7]),
    "t2_k1_c5": ([0, 1], [2], [3, 4, 5, 6, 7]),
    "t2_k3_c3": ([0, 1], [2, 3, 4], [5, 6, 7]),
}
NEW_PLACEMENTS = ["t2_k1_c5"]  # the others are written by make_plan_exp06_08.py and must stay identical


def placement_yaml(name: str) -> str:
    thinker, talker, c2w = PLACEMENTS[name]
    used = sorted(set(thinker + talker + c2w))
    lst = lambda xs: ", ".join(map(str, xs))  # noqa: E731
    lines = [
        f"# MiniCPM-o 4.5 语音流水线（实验 9 的放置 {name}），一个服务，用 GPU {lst(used)}：",
        f"# thinker 所在的 pipeline 进程 {len(thinker)} 份（GPU {lst(thinker)}），talker 进程 {len(talker)} 份（GPU {lst(talker)}），",
        f"# code2wav 单独进程 {len(c2w)} 份（GPU {lst(c2w)}，单独占卡）。",
        "# 显存比例与 2+6 基线相同（talker 0.40、code2wav 0.40），只改 talker 和 code2wav 各占几张卡。",
        "# 由 scripts/slurm/make_plan_exp09_11.py 生成。",
        "config_cls: MiniCPMOSpeechPipelineConfig",
        "model_path: openbmb/MiniCPM-o-4_5",
        "",
        "stages:",
        "  talker:",
        f"    gpu_memory_fraction: {TALKER_FRACTION}",
        "  code2wav:",
        "    process: code2wav",
        f"    gpu_memory_fraction: {CODE2WAV_FRACTION}",
        "",
        "processes:",
        "  pipeline:",
        f"    num_replicas: {len(thinker)}",
        f"    replica_devices: [{lst(thinker)}]",
        "  talker:",
        f"    num_replicas: {len(talker)}",
        f"    replica_devices: [{lst(talker)}]",
        "  code2wav:",
        f"    num_replicas: {len(c2w)}",
        f"    replica_devices: [{lst(c2w)}]",
    ]
    return "\n".join(lines) + "\n"


def placement_server(name: str) -> list[dict]:
    thinker, talker, c2w = PLACEMENTS[name]
    used = sorted(set(thinker + talker + c2w))
    return [{"name": "s0", "model": "minicpm_o_4_5", "gpus": used,
             "args": ["--config", f"{{research_repo}}/configs/models/minicpm_o_4_5_exp_{name}.yaml"], "ready_timeout_s": 2400}]


def closed(name: str, workload: str, levels: list[int], waves: int, events: bool) -> dict:
    """Closed loop: per level the runner's warmup of one wave (not measured), then max(96, waves x c) measured."""
    ph = {"kind": "closed_loop", "name": name, "workload": workload, "concurrency": levels,
          "requests_per_level": 96, "requests_per_concurrency": waves}
    if events:
        ph["server_events"] = True
    return ph


def describe(name: str) -> str:
    thinker, talker, c2w = PLACEMENTS[name]
    shared = talker == c2w
    return (f"thinker {len(thinker)} 卡、talker {len(talker)} 份、code2wav {len(c2w)} 份"
            f"（{'与 talker 同卡' if shared else '单独占卡'}，用 {len(set(thinker + talker + c2w))} 张卡）")


def exp09_pilot_entries() -> list[tuple[dict, dict]]:
    """Pilot (not confirmatory): every placement once, with events, W3b at 128/256 and the long workload at
    32/64/128 (two measured waves). Used for the open-loop rates of the confirmatory runs (rule in
    hypotheses.md) and as a first look at which stage saturates; never pooled with the blocks."""
    out = []
    for position, arm in enumerate([3, 1, 2, 4]):  # the new placement first: it is the most likely not to start
        name = EXP09_ARMS[arm]
        phases = [smoke(), closed("w3b_pilot", W3B, [128, 256], waves=2, events=True),
                  closed("w3l_pilot", W3L, [32, 64, 128], waves=2, events=True)]
        cfg = base(f"exp09_pilot_{name}", 8, placement_server(name), phases,
                   f"实验 9 预跑（不是正式数据）：方案 {arm}，放置 {name}（{describe(name)}），打点版 sglang-omni。"
                   "W3b 闭环并发 128、256，长回答负载闭环并发 32、64、128，每档先预热一波再测两波，开事件记录。"
                   "只用来按登记的规则定开环速率、初看哪个阶段先满。")
        cfg["runtime_profile"] = INSTR_RUNTIME
        out.append((cfg, {"priority": 40 - position, "est": 900, "timeout": 3600}))
    return out


def render() -> dict[str, str]:
    files = {f"configs/models/minicpm_o_4_5_exp_{name}.yaml": placement_yaml(name) for name in NEW_PLACEMENTS}
    pilot = exp09_pilot_entries()
    for path, plan_id, comment, entries in (
        (PLAN_EXP09_PILOT, "exp09_pilot", "实验 9 预跑：4 种放置各一次，用来定开环速率（规则见 hypotheses.md）。", pilot),
    ):
        plan = {"plan_id": plan_id,
                "_comment": comment + "通过 plan.d 放进 Job 454290。由 scripts/slurm/make_plan_exp09_11.py 生成。",
                "experiments": [plan_entry(c, m) for c, m in entries]}
        for e in plan["experiments"]:
            e["command"] = f"bash scripts/slurm/run_experiment.sh {EXP_DIR}/{e['experiment_id']}.json"
        files[path] = json.dumps(plan, ensure_ascii=False, indent=1) + "\n"
    for cfg, _ in pilot:
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
