#!/usr/bin/env python3
"""Configs and plan for P0 of the pre-presentation validation: is the talker's admission cap of 32 what limits 2+1+5?

    python scripts/slurm/make_plan_ppv_p0.py          # write configs/models/*ppv_p0*.yaml, configs/experiments/ppv/p0_*
    python scripts/slurm/make_plan_ppv_p0.py --check  # list files that differ from what would be written

Design registered in research/next_steps/p0_p3_experiment_plan.md (P0, 2026-11-18) with the addendum in
research/next_steps/presentation_validation/preregistration.md (written before any P0 run):
- arm A: t2_k1_c5 as in 实验 9 (talker max_running_requests 32, CUDA graphs up to 32, KV 131072 tokens);
- arm B: only the talker's max_running_requests 64. sglang-omni then also captures CUDA graphs up to 64
  (generation_batch_policy.py: cuda_graph_max_bs defaults to max_running_requests), so B changes two things;
- arm G (addendum): max_running_requests 64 with cuda_graph_max_bs 32, which separates admission from graph coverage;
- reference: t2_k2_c4 unchanged;
- arm C (64 + max_total_tokens 262144) is written but only placed if the registered KV trigger fires in B.
Every run restarts the server; the order inside a block is random (seed 20261119); 3 blocks. A and B run W3b and
W3L, the reference and G only W3b (the registered decisions H0a/H0c use W3b). Closed-loop levels are separate phases
so each level has its own wall-clock window for the scheduler-log analysis.
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
EXP_DIR = "configs/experiments/ppv"
W3L = "configs/workloads/w3l_long_speech.json"
SEED = 20261119  # registered in p0_p3_experiment_plan.md
BLOCKS = 3
RUNTIME = "unicorn_instr"  # the runtime of 实验 9, so arm A repeats 实验 9's arm 3
# The research checkout the job runs this plan from (a worktree of this branch on the cluster, next to the
# orchestrator's own checkout); the runner takes its repository from the script's location.
CODE_DIR = "~/workspace/omni-ppv"
W3B_RATES = [15.5, 18.0, 21.0, 23.5, 26.0, 28.5, 31.0]  # registered {18 ... 31} + 15.5 (addendum: brackets A's capacity)
W3L_RATES = [8.0, 9.5, 10.5, 12.0]  # registered
CLOSED_W3B = [64, 128, 256]
CLOSED_W3L = [128]

# arm -> (placement, talker engine overrides, workloads)
ARMS = {
    "A": ("t2_k1_c5", {}, ("w3b", "w3l")),
    "B": ("t2_k1_c5", {"max_running_requests": 64}, ("w3b", "w3l")),
    "G": ("t2_k1_c5", {"max_running_requests": 64, "cuda_graph_max_bs": 32}, ("w3b",)),
    "R": ("t2_k2_c4", {}, ("w3b",)),
}
CONDITIONAL_ARMS = {"C": ("t2_k1_c5", {"max_running_requests": 64, "max_total_tokens": 262144}, ("w3b", "w3l"))}
ARM_NAMES = {"A": "cap32", "B": "cap64", "G": "cap64_graph32", "R": "ref_t2_k2_c4", "C": "cap64_kv262k"}
PLACEMENT_GPUS = {"t2_k1_c5": ([0, 1], [2], [3, 4, 5, 6, 7]), "t2_k2_c4": ([0, 1], [2, 3], [4, 5, 6, 7])}


def yaml_path(arm: str) -> str:
    placement, overrides, _ = (ARMS | CONDITIONAL_ARMS)[arm]
    if not overrides:
        return f"configs/models/minicpm_o_4_5_exp_{placement}.yaml"  # unchanged files of 实验 7/9
    return f"configs/models/minicpm_o_4_5_ppv_p0_{placement}_{ARM_NAMES[arm]}.yaml"


def arm_yaml(arm: str) -> str:
    placement, overrides, _ = (ARMS | CONDITIONAL_ARMS)[arm]
    thinker, talker, c2w = PLACEMENT_GPUS[placement]
    lst = lambda xs: ", ".join(map(str, xs))  # noqa: E731
    lines = [
        f"# P0 方案 {arm}（{ARM_NAMES[arm]}）：与实验 9 的放置 {placement} 相同（thinker 所在的 pipeline 进程 {len(thinker)} 份，",
        f"# talker {len(talker)} 份（GPU {lst(talker)}），code2wav 单独进程 {len(c2w)} 份），只改 talker 引擎的："
        + "、".join(f"{k}={v}" for k, v in overrides.items()) + "。",
        "# 由 scripts/slurm/make_plan_ppv_p0.py 生成。",
        "config_cls: MiniCPMOSpeechPipelineConfig",
        "model_path: openbmb/MiniCPM-o-4_5",
        "",
        "stages:",
        "  talker:",
        f"    gpu_memory_fraction: {TALKER_FRACTION}",
        "    engine:",
        *[f"      {k}: {v}" for k, v in overrides.items()],
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


def closed_level(workload: str, w: str, mode: str, c: int, waves: int) -> dict:
    """One closed-loop level as its own phase (A: no events, the capacity; B: request events, the stage residence)."""
    ph = {"kind": "closed_loop", "name": f"{w}_closed_{mode}_c{c}", "workload": workload, "concurrency": [c],
          "requests_per_level": 96, "requests_per_concurrency": waves}
    if mode == "B":
        ph["server_events"] = True
    return ph


def phases(arm: str) -> list[dict]:
    _, _, workloads = (ARMS | CONDITIONAL_ARMS)[arm]
    out = [smoke()]
    out += [closed_level(W3B, "w3b", "A", c, 3) for c in CLOSED_W3B]
    out.append(closed_level(W3B, "w3b", "B", 256, 3))
    out.append({"kind": "open_loop", "name": "w3b_open", "workload": W3B, "rates_rps": W3B_RATES,
                "duration_per_rate_s": 60, "min_requests_per_rate": 96, "max_error_rate": 0.5})
    if "w3l" in workloads:
        out += [closed_level(W3L, "w3l", "A", c, 2) for c in CLOSED_W3L]
        out.append({"kind": "open_loop", "name": "w3l_open", "workload": W3L, "rates_rps": W3L_RATES,
                    "duration_per_rate_s": 90, "min_requests_per_rate": 48, "max_error_rate": 0.5})
    return out


def server(arm: str) -> list[dict]:
    return [{"name": "s0", "model": "minicpm_o_4_5", "gpus": list(range(8)),
             "args": ["--config", "{research_repo}/" + yaml_path(arm)], "ready_timeout_s": 2400}]


def experiment_id(arm: str, block: int) -> str:
    return f"ppv_p0_{ARM_NAMES[arm]}_block{block}"


def entries(arms: dict = ARMS) -> list[tuple[dict, dict]]:
    out = []
    for block in range(1, BLOCKS + 1):
        rng = random.Random(SEED * 100 + block)
        order = sorted(arms)
        rng.shuffle(order)
        for position, arm in enumerate(order, start=1):
            placement, overrides, workloads = arms[arm]
            change = "、".join(f"{k}={v}" for k, v in overrides.items()) or "不改"
            desc = (f"P0 第 {block} 个区组第 {position} 个（区组内顺序随机，种子 {SEED}）：方案 {arm}（{ARM_NAMES[arm]}），放置 {placement}，"
                    f"talker 引擎 {change}。打点版 sglang-omni（与实验 9 相同）。W3b 闭环 {'/'.join(map(str, CLOSED_W3B))}（每档一段）"
                    f"+ 256 带事件，开环 {'/'.join(f'{r:g}' for r in W3B_RATES)} 请求/秒各 60 秒"
                    + (f"；W3L 闭环 128，开环 {'/'.join(f'{r:g}' for r in W3L_RATES)} 请求/秒各 90 秒。" if "w3l" in workloads else "。"))
            cfg = base(experiment_id(arm, block), 8, server(arm), phases(arm), desc)
            cfg["runtime_profile"] = RUNTIME
            est = 1700 if "w3l" in workloads else 950
            out.append((cfg, {"priority": -(500 + 10 * block + position), "est": est, "timeout": 5400}))
    return out


def command(eid: str) -> str:
    return f"cd {CODE_DIR} && bash scripts/slurm/run_experiment.sh {EXP_DIR}/{eid}.json"


def render() -> dict[str, str]:
    files = {yaml_path(arm): arm_yaml(arm) for arm in [*ARMS, *CONDITIONAL_ARMS] if (ARMS | CONDITIONAL_ARMS)[arm][1]}
    main = entries()
    cond = entries(CONDITIONAL_ARMS)
    for cfg, _ in main + cond:
        files[f"{EXP_DIR}/{cfg['experiment_id']}.json"] = json.dumps(cfg, ensure_ascii=False, indent=1) + "\n"
    for b in (1, 2, 3):
        chosen = [(c, m) for c, m in main if c["experiment_id"].endswith(f"_block{b}")]
        plan = {"plan_id": f"ppv_p0_block{b}",
                "_comment": f"P0 第 {b} 个区组（方案 A/B/G/参照，区组内顺序随机）。通过 plan.d 放进 Job 454290；命令从集群上本分支的 "
                            f"worktree（{CODE_DIR}）运行。由 scripts/slurm/make_plan_ppv_p0.py 生成。",
                "experiments": [plan_entry(c, m) for c, m in chosen]}
        for e in plan["experiments"]:
            e["command"] = command(e["experiment_id"])
        files[f"configs/experiments/plan_ppv_p0_block{b}.json"] = json.dumps(plan, ensure_ascii=False, indent=1) + "\n"
    plan_c = {"plan_id": "ppv_p0_armC",
              "_comment": "P0 方案 C（名额 64 + KV 262144），只在方案 B 触发登记的 KV 条件时放进队列。由 scripts/slurm/make_plan_ppv_p0.py 生成。",
              "experiments": [plan_entry(c, m) for c, m in cond]}
    for e in plan_c["experiments"]:
        e["command"] = command(e["experiment_id"])
    files["configs/experiments/plan_ppv_p0_armC.json"] = json.dumps(plan_c, ensure_ascii=False, indent=1) + "\n"
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
