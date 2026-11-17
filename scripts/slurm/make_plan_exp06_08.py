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
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_plan_8gpu_v2 import CODE2WAV_FRACTION, TALKER_FRACTION, W3B, base, plan_entry, smoke  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EXP_DIR = "configs/experiments/exp06_08"
PLAN_EXP07_PILOT = "configs/experiments/plan_exp07_pilot.json"
PLAN_EXP06_SMOKE = "configs/experiments/plan_exp06_smoke.json"
PLAN_EXP06_08 = "configs/experiments/plan_exp06_exp08a.json"
W3T = "configs/workloads/w3t_text_only.json"
# The research client sends "seed": 0 by default; sglang-omni keeps seeded requests off the thinker's
# lookahead, so 实验 8 adds no-seed variants (hypotheses.md, amendment of 14:58 before any data).
EXP08_WORKLOADS = {"w3t": W3T, "w3t_ns": "configs/workloads/w3t_text_only_noseed.json",
                   "w3b": W3B, "w3b_ns": "configs/workloads/w3b_text_speech_noseed.json"}
SEED = 20261117  # every randomized order below derives from this seed (hypotheses.md)
INSTR_RUNTIME = "unicorn_instr"
EXP06_LEVELS = [8, 64, 128, 256]
EXP08_LEVELS = [8, 32, 64]
BLOCKS = 3
# Where a 3 s profiler window starts inside a measured phase of max(96, 3c) requests (seconds), so it
# falls in the steady middle; from the S2:6 throughput of plan v2 (about 3.7, 14, 20, 20 req/s).
WINDOW_START_S = {8: 10, 32: 8, 64: 7, 128: 10, 256: 18}

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


def measured(name: str, workload: str, c: int, mode: str) -> dict:
    """One closed-loop phase at concurrency c: the runner's default warmup (c requests), then
    max(96, 3c) measured requests. mode A: no events, no profiler; B: request events (with the
    patch's measurement points); C: a 3 s torch profiler window with events inside it."""
    ph = {"kind": "closed_loop", "name": name, "workload": workload, "concurrency": [c],
          "requests_per_level": 96, "requests_per_concurrency": 3}
    if mode == "B":
        ph["server_events"] = True
    elif mode == "C":
        ph["torch_profile_window"] = {"start_after_s": WINDOW_START_S[c], "duration_s": 3}
    else:
        pass
    return ph


def instr(cfg: dict) -> dict:
    cfg["runtime_profile"] = INSTR_RUNTIME
    return cfg


def smoke_instr_entry() -> tuple[dict, dict]:
    phases = [smoke(), measured("w3b_c16_B", W3B, 16, "B"),
              {**measured("w3b_c64_C", W3B, 64, "C"), "torch_profile_window": {"start_after_s": 6, "duration_s": 2}}]
    cfg = instr(base("exp06_smoke_instr", 8, placement_server("t2_k6_shared"), phases,
                     "实验 6–8 打点版 sglang-omni 的 GPU 冒烟测试（不是正式数据）：2+6，并发 16 开事件、并发 64 开 2 秒 profiler。"
                     "确认补丁在 GPU 上能跑、事件和 trace 都写出来，再放正式实验。"))
    return cfg, {"priority": 30, "est": 600, "timeout": 2400}


def exp06_entries() -> list[tuple[dict, dict]]:
    """实验 6: S2:6 at concurrency 8/64/128/256, three blocks (independent server starts). Within a block
    the level order is random and A/B/C rotate (ABC, BCA, CAB) so order effects balance."""
    out = []
    rotations = ["ABC", "BCA", "CAB"]
    for block in range(1, BLOCKS + 1):
        rng = random.Random(SEED * 100 + 6 * 10 + block)
        levels = EXP06_LEVELS[:]
        rng.shuffle(levels)
        order = rotations[block - 1]
        phases = [smoke()]
        for c in levels:
            phases += [measured(f"w3b_c{c}_{mode}", W3B, c, mode) for mode in order]
        cfg = instr(base(f"exp06_c2w_block{block}", 8, placement_server("t2_k6_shared"), phases,
                         f"实验 6（code2wav 时间拆解）第 {block} 个区组：2+6，打点版 sglang-omni。并发顺序（种子 {SEED}）"
                         f"{'、'.join(map(str, levels))}；每个并发依次跑 {'、'.join(order)} 三段（A 不打点、B 打点、C 打点并在中间开 3 秒 torch profiler），"
                         "每段先预热一波再测 max(96, 3×并发) 个请求。"))
        out.append((cfg, {"priority": 20 - block, "est": 900, "timeout": 3600}))
    return out


def exp08a_entries() -> list[tuple[dict, dict]]:
    """实验 8 phase 1: S2:6, 2x2 (text only W3t vs text + speech W3b) x (seed 0 vs no seed) at concurrency
    8/32/64; three blocks with a random order of the twelve conditions; B (events) then C (3 s kernel trace)."""
    out = []
    for block in range(1, BLOCKS + 1):
        rng = random.Random(SEED * 100 + 8 * 10 + block)
        conditions = [(w, c) for w in EXP08_WORKLOADS for c in EXP08_LEVELS]
        rng.shuffle(conditions)
        phases = [smoke()]
        for w, c in conditions:
            phases += [measured(f"{w}_c{c}_B", EXP08_WORKLOADS[w], c, "B"), measured(f"{w}_c{c}_C", EXP08_WORKLOADS[w], c, "C")]
        cfg = instr(base(f"exp08_lookahead_block{block}", 8, placement_server("t2_k6_shared"), phases,
                         f"实验 8 第一阶段第 {block} 个区组：2+6，打点版 sglang-omni。2×2：只要文字（w3t）或文字+语音（w3b），"
                         f"请求带种子 0 或不带（_ns），并发 8、32、64；顺序（种子 {SEED}）{'、'.join(f'{w}@{c}' for w, c in conditions)}。"
                         "每个条件先跑开事件的一段（含 thinker 每轮记录），再跑一段在中间开 3 秒 torch profiler。"))
        out.append((cfg, {"priority": 15 - block, "est": 1100, "timeout": 3600}))
    return out


PLAN_EXP08_SUPP = "configs/experiments/plan_exp08_supp.json"


def exp08_supp_entries() -> list[tuple[dict, dict]]:
    """实验 8 phase 1 supplement. Text-only phases at concurrency 32 and 64 finish in about 8-10 s, so the
    3 s profiler window that starts 7-8 s into the phase caught few or no decode steps (block 1). Here the two
    text conditions run six waves and the window starts 5 s in; three blocks, random order (limitations.md)."""
    out = []
    for block in range(1, BLOCKS + 1):
        rng = random.Random(SEED * 100 + 88 * 10 + block)
        conditions = [(w, c) for w in ("w3t", "w3t_ns") for c in (32, 64)]
        rng.shuffle(conditions)
        phases = [smoke()]
        for w, c in conditions:
            phases.append({"kind": "closed_loop", "name": f"{w}_c{c}_C", "workload": EXP08_WORKLOADS[w], "concurrency": [c],
                           "requests_per_level": 96, "requests_per_concurrency": 6, "torch_profile_window": {"start_after_s": 5, "duration_s": 3}})
        cfg = instr(base(f"exp08_lookahead_supp_block{block}", 8, placement_server("t2_k6_shared"), phases,
                         f"实验 8 第一阶段的补测第 {block} 个区组：只要文字的两个条件（带种子 w3t、不带种子 w3t_ns）在并发 32、64 跑六波，"
                         f"第 5 秒起开 3 秒 kernel trace（原设计的窗口开得太晚，文字请求跑得快，录不到 decode 步）。顺序（种子 {SEED}）"
                         f"{'、'.join(f'{w}@{c}' for w, c in conditions)}。"))
        out.append((cfg, {"priority": 5 - block, "est": 420, "timeout": 2400}))
    return out


PLAN_EXP07 = "configs/experiments/plan_exp07.json"
EXP07_ARMS = ["t2_k6_shared", "t2_k4_c2", "t2_k3_c3", "t2_k2_c4", "t2_k3_shared"]
EXP07_LEVELS = [8, 64, 128, 256]
# Open-loop rates. Pre-registered rule (hypotheses.md): 0.5X, 0.75X, 0.9X, 1.0X of the weakest
# equal-budget arm's pilot capacity X, rounded to 0.5 req/s; the pilot's closed-loop c256 throughput gave
# t2_k4_c2 10.6, t2_k3_c3 15.1, t2_k2_c4 20.6 (t2_k3_shared 11.0; S2:6 about 20 in plan v2), so X = 10.6.
# 15 and 18 req/s were added after the pilot (exploratory, recorded in limitations.md) to compare the
# two strongest arms near their capacity.
EXP07_RATES = [5.5, 8.0, 9.5, 10.5]
EXP07_RATES_EXPLORATORY = [15.0, 18.0]


def exp07_entries() -> list[tuple[dict, dict]]:
    """实验 7: five placements x three blocks; within a block the arm order is random (seed). Each run:
    closed loop 8/64/128/256 with events (two measured waves), a 3 s kernel trace at 128, then open
    loop at the common rates for 60 s each with events."""
    out = []
    for block in range(1, BLOCKS + 1):
        rng = random.Random(SEED * 100 + 7 * 10 + block)
        arms = EXP07_ARMS[:]
        rng.shuffle(arms)
        for position, arm in enumerate(arms, start=1):
            thinker, talker, c2w = PLACEMENTS[arm]
            phases = [smoke(), closed_levels("w3b_closed", EXP07_LEVELS, waves=3, server_events=True),
                      {**measured("w3b_c128_C", W3B, 128, "C"), "requests_per_concurrency": 2},
                      {"kind": "open_loop", "name": "w3b_open", "workload": W3B, "rates_rps": EXP07_RATES,
                       "duration_per_rate_s": 90, "min_requests_per_rate": 96, "max_error_rate": 0.2, "server_events": True},
                      {"kind": "open_loop", "name": "w3b_open_hi", "workload": W3B, "rates_rps": EXP07_RATES_EXPLORATORY,
                       "duration_per_rate_s": 60, "min_requests_per_rate": 96, "max_error_rate": 0.2, "server_events": True}]
            cfg = instr(base(f"exp07_{arm}_block{block}", 8, placement_server(arm), phases,
                             f"实验 7 第 {block} 个区组第 {position} 个（区组内顺序随机，种子 {SEED}）：放置 {arm}"
                             f"（thinker {len(thinker)}、talker {len(talker)}、code2wav {len(c2w)}，"
                             f"{'与 talker 同卡' if talker == c2w else '单独占卡'}，用 {len(set(thinker + talker + c2w))} 张卡），打点版 sglang-omni。"
                             f"闭环 {'/'.join(map(str, EXP07_LEVELS))}（各三波，开事件），并发 128 开 3 秒 kernel trace，"
                             f"再开环 {'/'.join(f'{r:g}' for r in EXP07_RATES)} 请求/秒各 90 秒（预先登记），"
                             f"{'/'.join(f'{r:g}' for r in EXP07_RATES_EXPLORATORY)} 请求/秒各 60 秒（探索性），都开事件。"))
            # A unique, decreasing priority runs the blocks in order and the arms in their random order,
            # below 实验 6 and 实验 8 phase 1 (positive priorities) when all are queued together.
            out.append((cfg, {"priority": -(10 * block + position), "est": 1000, "timeout": 3600}))
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
    smoke_entries = [smoke_instr_entry()]
    main_entries = [*exp06_entries(), *exp08a_entries()]
    exp07 = exp07_entries()
    supp = exp08_supp_entries()
    for path, plan_id, comment, entries in (
        (PLAN_EXP06_SMOKE, "exp06_smoke", "打点版 sglang-omni 的 GPU 冒烟测试。先单独放，确认事件和 trace 都写出来以后，再放正式区组。", smoke_entries),
        (PLAN_EXP06_08, "exp06_exp08a", "实验 6 的 3 个区组、实验 8 第一阶段的 3 个区组，依次运行。冒烟测试检查通过后才放。", main_entries),
        (PLAN_EXP07, "exp07", "实验 7：5 种放置 × 3 个区组，区组内顺序随机，区组按 1、2、3 的顺序跑（优先级递减）。", exp07),
        (PLAN_EXP08_SUPP, "exp08_supp", "实验 8 第一阶段的补测（文字条件在并发 32、64 的 kernel trace），排在实验 7 之前。", supp),
    ):
        plan = {"plan_id": plan_id,
                "_comment": comment + "通过 plan.d 放进 Job 454290。由 scripts/slurm/make_plan_exp06_08.py 生成。",
                "experiments": [plan_entry(c, m) for c, m in entries]}
        for e in plan["experiments"]:
            e["command"] = f"bash scripts/slurm/run_experiment.sh {EXP_DIR}/{e['experiment_id']}.json"
        files[path] = json.dumps(plan, ensure_ascii=False, indent=1) + "\n"
    for cfg, _ in pilot + smoke_entries + main_entries + exp07 + supp:
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
