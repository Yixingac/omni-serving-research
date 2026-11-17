#!/usr/bin/env python3
"""Configs and plans for 实验 6–8 (code2wav time breakdown, code2wav on its own GPUs, thinker lookahead).

    python scripts/slurm/make_plan_exp06_08.py          # write configs/models/*.yaml, configs/experiments/...
    python scripts/slurm/make_plan_exp06_08.py --check  # list files that differ from what would be written

Placements are sglang-omni process layouts on one 8xA6000 node. Every layout keeps the talker's and
code2wav's memory fractions of the S2:6 baseline (0.40 each), so only where code2wav runs changes.
Replica routing in sglang-omni 520b49ee is per-process round robin at admission
(sglang_omni/pipeline/replicas.py, RoundRobinBindingPolicy), so talker and code2wav replica counts may differ.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_plan_8gpu_v2 import CODE2WAV_FRACTION, TALKER_FRACTION, W3B, base, plan_entry, smoke  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EXP_DIR = "configs/experiments/exp06_08"
PLAN_EXP07_PILOT = "configs/experiments/plan_exp07_pilot.json"

# name -> (thinker GPUs, talker GPUs, code2wav GPUs); shared when the talker and code2wav lists are equal.
PLACEMENTS = {
    "t2_k6_shared": ([0, 1], [2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7]),  # = S2:6, the baseline
    "t2_k4_c2": ([0, 1], [2, 3, 4, 5], [6, 7]),
    "t2_k3_c3": ([0, 1], [2, 3, 4], [5, 6, 7]),
    "t2_k2_c4": ([0, 1], [2, 3], [4, 5, 6, 7]),
    "t2_k3_shared": ([0, 1], [2, 3, 4], [2, 3, 4]),  # mechanism control for t2_k3_c3: same replicas, code2wav shares
}


def placement_yaml(name: str) -> str:
    thinker, talker, c2w = PLACEMENTS[name]
    shared = talker == c2w
    used = sorted(set(thinker + talker + c2w))
    lst = lambda xs: ", ".join(map(str, xs))  # noqa: E731
    lines = [
        f"# MiniCPM-o 4.5 语音流水线（实验 6–8 的放置 {name}），一个服务，用 GPU {lst(used)}：",
        f"# thinker 所在的 pipeline 进程 {len(thinker)} 份（GPU {lst(thinker)}），talker 进程 {len(talker)} 份（GPU {lst(talker)}），",
        f"# code2wav 单独进程 {len(c2w)} 份（GPU {lst(c2w)}，" + ("与 talker 同卡）。" if shared else "单独占卡）。"),
        "# 显存比例与 2+6 基线相同（talker 0.40、code2wav 0.40），只改 code2wav 放在哪。",
        "# 由 scripts/slurm/make_plan_exp06_08.py 生成。",
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


def placement_server(name: str, runtime_note: str = "") -> list[dict]:
    thinker, talker, c2w = PLACEMENTS[name]
    used = sorted(set(thinker + talker + c2w))
    return [{"name": "s0", "model": "minicpm_o_4_5", "gpus": used,
             "args": ["--config", f"{{research_repo}}/configs/models/minicpm_o_4_5_exp_{name}.yaml"], "ready_timeout_s": 2400}]


def closed_levels(name: str, levels: list[int], waves: int, **extra) -> dict:
    """Closed loop over several levels: each level measures max(96, waves x concurrency) requests after
    the runner's default warmup of one wave (concurrency requests, not measured)."""
    return {"kind": "closed_loop", "name": name, "workload": W3B, "concurrency": levels,
            "requests_per_level": 96, "requests_per_concurrency": waves, **extra}


def exp07_pilot_entries() -> list[tuple[dict, dict]]:
    """Pilot (not confirmatory): does each placement start and serve, and roughly where does its
    closed-loop throughput saturate? Used only to pick the open-loop rates of the confirmatory runs."""
    out = []
    for i, name in enumerate(["t2_k4_c2", "t2_k3_c3", "t2_k2_c4", "t2_k3_shared"]):
        thinker, talker, c2w = PLACEMENTS[name]
        used = len(set(thinker + talker + c2w))
        phases = [smoke(), closed_levels("w3b_pilot", [64, 256], waves=2, server_events=True)]
        cfg = base(f"exp07_pilot_{name}", 8, placement_server(name), phases,
                   f"实验 7 预跑（不是正式数据）：放置 {name}（thinker {len(thinker)} 卡、talker {len(talker)} 份、code2wav {len(c2w)} 份，"
                   f"{'与 talker 同卡' if talker == c2w else '单独占卡'}，共用 {used} 张卡）。W3b 闭环并发 64、256，"
                   "每档先预热一波再测两波，开事件记录。只用来确认配置能起来、估容量，以便选正式实验的开环速率。")
        out.append((cfg, {"priority": 10 - i, "est": 600, "timeout": 2400}))
    return out


def render() -> dict[str, str]:
    files = {f"configs/models/minicpm_o_4_5_exp_{name}.yaml": placement_yaml(name) for name in PLACEMENTS}
    pilot = exp07_pilot_entries()
    plan = {"plan_id": "exp07_pilot",
            "_comment": "实验 7 预跑：四种 code2wav 放置能否启动、大致容量。通过 plan.d 放进 Job 454290。由 scripts/slurm/make_plan_exp06_08.py 生成。",
            "experiments": [plan_entry(c, m) for c, m in pilot]}
    for e in plan["experiments"]:
        e["command"] = f"bash scripts/slurm/run_experiment.sh {EXP_DIR}/{e['experiment_id']}.json"
    files[PLAN_EXP07_PILOT] = json.dumps(plan, ensure_ascii=False, indent=1) + "\n"
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
