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
    "t2_k4_c2": ([0, 1], [2, 3, 4, 5], [6, 7]),  # 实验 10 (实验 7's B)
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


# Common open-loop rates from the pilot by the registered rule (scripts/analysis/exp09_report.py --pilot, common_rates).
# Pilot closed-loop capacities (req/s), W3b at 256: arm 1 20.69, arm 2 21.22, arm 3 15.26, arm 4 15.73;
# W3L at 128: 8.72, 9.37, 7.31, 6.62 (20261117_exp09_pilot_*, analysed 23:09).
EXP09_RATES_W3B: list[float] | None = [7.5, 12.0, 15.5, 18.0, 21.0, 23.5]
EXP09_RATES_W3L: list[float] | None = [3.5, 5.5, 6.5, 8.0, 9.5, 10.5]
PLAN_EXP09 = "configs/experiments/plan_exp09.json"
BLOCKS = 3


def exp09_entries() -> list[tuple[dict, dict]]:
    """实验 9: four placements x three blocks, random order within a block (seed), blocks in order. Phases as registered:
    W3b closed A (8/64/128/256) and B (64/128/256, events) with the A/B order alternating by block, W3b open loop at the
    common rates (60 s each), W3L closed A (16/64/128) and B (128), W3L open loop at its common rates (90 s each).
    max_error_rate 0.5 only stops a placement that fails most requests (a broken server), never an overloaded one."""
    out = []
    for block in range(1, BLOCKS + 1):
        rng = random.Random(SEED * 100 + 9 * 10 + block)
        arms = sorted(EXP09_ARMS)
        rng.shuffle(arms)
        for position, arm in enumerate(arms, start=1):
            name = EXP09_ARMS[arm]
            a = closed("w3b_closed_A", W3B, [8, 64, 128, 256], waves=3, events=False)
            b = closed("w3b_closed_B", W3B, [64, 128, 256], waves=3, events=True)
            phases = [smoke(), *([b, a] if block == 2 else [a, b]),
                      {"kind": "open_loop", "name": "w3b_open", "workload": W3B, "rates_rps": EXP09_RATES_W3B,
                       "duration_per_rate_s": 60, "min_requests_per_rate": 96, "max_error_rate": 0.5},
                      closed("w3l_closed_A", W3L, [16, 64, 128], waves=2, events=False),
                      closed("w3l_closed_B", W3L, [128], waves=2, events=True),
                      {"kind": "open_loop", "name": "w3l_open", "workload": W3L, "rates_rps": EXP09_RATES_W3L,
                       "duration_per_rate_s": 90, "min_requests_per_rate": 48, "max_error_rate": 0.5}]
            cfg = base(f"exp09_{name}_block{block}", 8, placement_server(name), phases,
                       f"实验 9 第 {block} 个区组第 {position} 个（区组内顺序随机，种子 {SEED}）：方案 {arm}，放置 {name}（{describe(name)}），"
                       f"打点版 sglang-omni。W3b 闭环 A/B 段、开环 {'/'.join(f'{r:g}' for r in EXP09_RATES_W3B)} 请求/秒各 60 秒；"
                       f"W3L 闭环 A/B 段、开环 {'/'.join(f'{r:g}' for r in EXP09_RATES_W3L)} 请求/秒各 90 秒（速率由预跑按登记规则定）。")
            cfg["runtime_profile"] = INSTR_RUNTIME
            out.append((cfg, {"priority": -(100 + 10 * block + position), "est": 1800, "timeout": 5400}))
    return out


# ---------------------------------------------------------------- 实验 10

BATCH_RUNTIME = "unicorn_batch"
# Port claims of sglang-omni go to a directory inside the run directory (实验 10 and 11 only; 实验 9 keeps /tmp). In the
# shared /tmp directory, claim files of a killed server stay behind, and two processes that start together can both
# take over the same stale claim (2026-11-18 00:38, 实验 9 方案 2 区组 1: EADDRINUSE on 29500).
PORT_CLAIM_ENV = {"SGLANG_OMNI_PORT_CLAIM_DIR": "{exp_dir}/port_claims"}
EXP10_PLACEMENT = "t2_k4_c2"  # code2wav is clearly the bottleneck there, on two cards of its own (实验 7)
EXP10_NOGRAPH_YAML = "configs/models/minicpm_o_4_5_exp10_t2_k4_c2_nograph.yaml"
GREEDY = "configs/workloads/w3b_greedy_check.json"
MIX = [{"workload": W3B, "weight": 0.5}, {"workload": W3L, "weight": 0.5}]
EXP10_SETTINGS = {"fifo1": ("fifo", 1), "fifo2": ("fifo", 2), "fifo4": ("fifo", 4), "fifo8": ("fifo", 8),
                  "fifo16": ("fifo", 16), "length8": ("length", 8), "length16": ("length", 16)}
EXP10_CONCURRENCY = {"w3b": 128, "mix": 96}
PLAN_EXP10_CHECK = "configs/experiments/plan_exp10_check.json"
PLAN_EXP10 = "configs/experiments/plan_exp10.json"


def nograph_yaml() -> str:
    text = Path(REPO / "configs/models/minicpm_o_4_5_exp_t2_k4_c2.yaml").read_text()
    text = text.replace("# 由 scripts/slurm/make_plan_exp06_08.py 生成。",
                        "# 实验 10：同一放置，只关掉 code2wav 的 flow CUDA graph（enable_flow_cuda_graph: false）。由 scripts/slurm/make_plan_exp09_11.py 生成。")
    return text.replace("    process: code2wav\n", "    process: code2wav\n    factory:\n      enable_flow_cuda_graph: false\n", 1)


def control(name: str, setting: str) -> dict:
    policy, cap = EXP10_SETTINGS[setting.rstrip("b")]
    return {"kind": "control", "name": f"ctl_{name}", "path": "{exp_dir}/c2w_control.json",
            "content": {"max_batch_size": cap, "policy": policy, "window": 64}, "settle_s": 3}


def batch_server(graph: bool, digest: bool = False) -> list[dict]:
    srv = placement_server(EXP10_PLACEMENT)
    if not graph:
        srv[0]["args"] = ["--config", f"{{research_repo}}/{EXP10_NOGRAPH_YAML}"]
    env = {"SGLANG_OMNI_C2W_CONTROL": "{exp_dir}/c2w_control.json", **PORT_CLAIM_ENV}
    if digest:
        env["SGLANG_OMNI_EXP_TRACE"] = "c2w,thinker,digest"
    srv[0]["env"] = env
    return srv


def exp10_check_entry() -> tuple[dict, dict]:
    phases = [smoke()]
    for setting in ["fifo1", "fifo16", "length16", "length8", "fifo16b"]:
        phases += [control(f"chk_{setting}", setting),
                   {**closed(f"chk_{setting}", GREEDY, [32], waves=3, events=True)}]
    cfg = base("exp10_check", 8, batch_server(graph=True, digest=True), phases,
               "实验 10 正确性检查：放置 t2_k4_c2，flow CUDA graph 开，打开 digest。贪心负载（thinker、talker 温度 0，不带种子），并发 32，"
               "依次 fifo1、fifo16、length16、length8、fifo16（控制文件切换），每段 96 个请求。比较语音编码相同的请求在各组批方式下的波形。")
    cfg["runtime_profile"] = BATCH_RUNTIME
    return cfg, {"priority": -200, "est": 900, "timeout": 2400}


def exp10_entries() -> list[tuple[dict, dict]]:
    """实验 10: 3 blocks x graph on/off (separate runs, random order in a block); inside a run both workloads (order by
    block) and the seven batching settings in a random order per block and workload, switched by control phases."""
    out = []
    for block in range(1, BLOCKS + 1):
        rng = random.Random(SEED * 100 + 10 * 10 + block)
        graphs = [True, False]
        rng.shuffle(graphs)
        for position, graph in enumerate(graphs, start=1):
            phases = [smoke()]
            for w in (["w3b", "mix"] if block != 2 else ["mix", "w3b"]):
                settings = list(EXP10_SETTINGS)
                rng.shuffle(settings)
                for setting in settings:
                    ph = closed(f"{w}_{setting}", W3B, [EXP10_CONCURRENCY[w]], waves=2, events=True)
                    if w == "mix":
                        del ph["workload"]
                        ph["workload_mix"] = MIX
                    phases += [control(f"{w}_{setting}", setting), ph]
            tag = "graph" if graph else "nograph"
            cfg = base(f"exp10_{tag}_block{block}", 8, batch_server(graph), phases,
                       f"实验 10 第 {block} 个区组第 {position} 个：放置 t2_k4_c2，flow CUDA graph {'开' if graph else '关'}。"
                       "W3b（并发 128）和 MIX（W3b、W3L 各半，并发 96）各 7 种组批设置（fifo 批上限 1/2/4/8/16、按长度 8/16），"
                       "控制文件切换，顺序随机（种子 20261118），每段预热一波、测两波，开事件。")
            cfg["runtime_profile"] = BATCH_RUNTIME
            out.append((cfg, {"priority": -(210 + 10 * block + position), "est": 1500, "timeout": 3600}))
    return out


PLAN_EXP10_VALIDATION = "configs/experiments/plan_exp10_validation.json"


def exp10_validation_entries(placement: str, setting: str) -> list[tuple[dict, dict]]:
    """实验 10 full-service validation (supplementary registration 1 in its hypotheses.md), written only when a setting
    met the registered trigger: 实验 9's best placement, flow graph on, W3b open loop at 实验 9's common rates (60 s
    each), fifo16 and the candidate inside one run (fifo16 first in blocks 1 and 3), events on, three blocks."""
    if placement not in EXP09_ARMS.values() or setting not in EXP10_SETTINGS or setting == "fifo16":
        raise ValueError(f"unknown placement or setting: {placement} {setting}")
    out = []
    for block in range(1, BLOCKS + 1):
        order = [setting, "fifo16"] if block == 2 else ["fifo16", setting]
        phases = [smoke()]
        for s in order:
            phases += [control(f"val_{s}", s),
                       {"kind": "open_loop", "name": f"val_{s}", "workload": W3B, "rates_rps": EXP09_RATES_W3B, "duration_per_rate_s": 60,
                        "min_requests_per_rate": 96, "max_error_rate": 0.5, "server_events": True}]
        srv = placement_server(placement)
        srv[0]["env"] = {"SGLANG_OMNI_C2W_CONTROL": "{exp_dir}/c2w_control.json", **PORT_CLAIM_ENV}
        cfg = base(f"exp10_validation_{placement}_{setting}_block{block}", 8, srv, phases,
                   f"实验 10 完整服务验证第 {block} 个区组：放置 {placement}（实验 9 登记容量最高的方案），flow CUDA graph 开。"
                   f"W3b 开环 {'/'.join(f'{r:g}' for r in EXP09_RATES_W3B)} 请求/秒各 60 秒，同一运行里 {order[0]} 和 {order[1]} 各一段"
                   "（控制文件切换），开事件。")
        cfg["runtime_profile"] = BATCH_RUNTIME
        out.append((cfg, {"priority": -(250 + 10 * block), "est": 1200, "timeout": 3600}))
    return out


def render_exp10_validation(placement: str, setting: str) -> dict[str, str]:
    entries = exp10_validation_entries(placement, setting)
    plan = {"plan_id": "exp10_validation",
            "_comment": f"实验 10 完整服务验证（{placement}，fifo16 对 {setting}），只在登记的触发条件满足时生成和放进队列。"
                        "通过 plan.d 放进 Job 454290。由 scripts/slurm/make_plan_exp09_11.py --exp10-validation 生成。",
            "experiments": [plan_entry(c, m) for c, m in entries]}
    for e in plan["experiments"]:
        e["command"] = f"bash scripts/slurm/run_experiment.sh {EXP_DIR}/{e['experiment_id']}.json"
    files = {PLAN_EXP10_VALIDATION: json.dumps(plan, ensure_ascii=False, indent=1) + "\n"}
    for cfg, _ in entries:
        files[f"{EXP_DIR}/{cfg['experiment_id']}.json"] = json.dumps(cfg, ensure_ascii=False, indent=1) + "\n"
    return files


# ---------------------------------------------------------------- 实验 11

ASYNC_RUNTIME = "unicorn_async"
EXP11_ARMS = {"A": "0", "B": "1", "C": "sidecopy"}
W3B_NS = "configs/workloads/w3b_text_speech_noseed.json"
W3L_NS = "configs/workloads/w3l_long_speech_noseed.json"
W3XL_NS = "configs/workloads/w3xl_long_speech_noseed.json"
W3T_NS = "configs/workloads/w3t_text_only_noseed.json"
W3L_GREEDY = "configs/workloads/w3l_greedy_check.json"
W3L_TIMEOUT = "configs/workloads/w3l_greedy_timeout.json"
EXP11_LEVELS = [8, 32, 64]
# Where the 3 s thinker-only profiler window starts in a measured C phase (seconds), as in 实验 8 phase 3.
EXP11_WINDOW_S = {8: 10, 32: 8, 64: 7}
PLAN_EXP11_CHECK = "configs/experiments/plan_exp11_check.json"
PLAN_EXP11 = "configs/experiments/plan_exp11.json"


def async_server(arm: str, digest: bool) -> list[dict]:
    srv = placement_server_s26()
    env = {"SGLANG_OMNI_SPEECH_LOOKAHEAD": EXP11_ARMS[arm.rstrip("2")], **PORT_CLAIM_ENV}
    if digest:
        env["SGLANG_OMNI_EXP_TRACE"] = "thinker,c2w,digest"
    srv[0]["env"] = env
    return srv


def placement_server_s26() -> list[dict]:
    return [{"name": "s0", "model": "minicpm_o_4_5", "gpus": list(range(8)),
             "args": ["--config", "{research_repo}/configs/models/minicpm_o_4_5_exp_t2_k6_shared.yaml"], "ready_timeout_s": 2400}]


def mixed(name: str, mix: list[dict], c: int, requests: int) -> dict:
    return {"kind": "closed_loop", "name": name, "workload_mix": mix, "concurrency": [c], "requests_per_level": requests,
            "requests_per_concurrency": 1, "server_events": True}


def exp11_check_entries() -> list[tuple[dict, dict]]:
    """Correctness gate: A, C and a repeat of A, digest on, greedy workloads (see hypotheses.md)."""
    out = []
    for position, arm in enumerate(["A", "C", "A2"]):
        phases = [smoke(),
                  {"kind": "smoke", "name": "chk_c1", "workload": GREEDY, "requests": 18, "server_events": True},
                  {**closed("chk_c8", GREEDY, [8], waves=6, events=True), "requests_per_level": 48},
                  {**closed("chk_c32", GREEDY, [32], waves=3, events=True)},
                  {**closed("chk_c64", GREEDY, [64], waves=2, events=True)},
                  mixed("chk_mix_text", [{"workload": W3T_NS, "weight": 0.5}, {"workload": GREEDY, "weight": 0.5}], 32, 96),
                  mixed("chk_mix_long", [{"workload": GREEDY, "weight": 0.5}, {"workload": W3L_GREEDY, "weight": 0.5}], 32, 96),
                  mixed("chk_mix_seed", [{"workload": W3B, "weight": 0.5}, {"workload": GREEDY, "weight": 0.5}], 32, 96),
                  {**closed("chk_timeout", W3L_TIMEOUT, [32], waves=2, events=True), "requests_per_level": 64},
                  {**closed("chk_after", GREEDY, [32], waves=3, events=True)},
                  {**closed("chk_w3xl", W3XL_NS, [8], waves=4, events=True), "requests_per_level": 32}]
        cfg = base(f"exp11_check_{arm}", 8, async_server(arm, digest=True), phases,
                   f"实验 11 正确性检查：组 {arm}（SGLANG_OMNI_SPEECH_LOOKAHEAD={EXP11_ARMS[arm.rstrip('2')]}），2+6，打开 digest。"
                   "贪心负载：并发 1（18 个）、8、32、64，混合（文字/语音、短/长、带 seed 0/不带），客户端 1.5 秒超时的中止段及之后的检查段，"
                   "W3XL 实际长度。")
        cfg["runtime_profile"] = ASYNC_RUNTIME
        out.append((cfg, {"priority": -(300 + position), "est": 900, "timeout": 3600}))
    return out


def exp11_measured(name: str, workload: str, c: int, mode: str) -> dict:
    ph = {"kind": "closed_loop", "name": name, "workload": workload, "concurrency": [c],
          "requests_per_level": 96, "requests_per_concurrency": 3}
    if mode == "B":
        ph["server_events"] = True
    elif mode == "C":
        ph["requests_per_concurrency"] = 6  # keeps the phase busy past the end of the profiler window
        ph["torch_profile_window"] = {"start_after_s": EXP11_WINDOW_S[c], "duration_s": 3}
    else:
        pass
    return ph


def exp11_entries() -> list[tuple[dict, dict]]:
    """Performance: 3 blocks x arms A, B, C in a Latin-square order; W3b 8/32/64 (random order per block, shared by
    the arms) with A/B/C phases rotated by block, then W3L and W3XL at 32 (A and B phases)."""
    out = []
    rotations = ["ABC", "BCA", "CAB"]
    arm_orders = [["A", "B", "C"], ["B", "C", "A"], ["C", "A", "B"]]
    for block in range(1, BLOCKS + 1):
        rng = random.Random(SEED * 100 + 11 * 10 + block)
        levels = EXP11_LEVELS[:]
        rng.shuffle(levels)
        for position, arm in enumerate(arm_orders[block - 1], start=1):
            phases = [smoke()]
            for c in levels:
                phases += [exp11_measured(f"w3b_ns_c{c}_{mode}", W3B_NS, c, mode) for mode in rotations[block - 1]]
            for tag, wl in (("w3l_ns", W3L_NS), ("w3xl_ns", W3XL_NS)):
                phases += [exp11_measured(f"{tag}_c32_{mode}", wl, 32, mode) for mode in "AB"]
            cfg = base(f"exp11_perf_block{block}_{arm}", 8, async_server(arm, digest=False), phases,
                       f"实验 11 性能对比第 {block} 个区组、组 {arm}（SGLANG_OMNI_SPEECH_LOOKAHEAD={EXP11_ARMS[arm]}）。2+6，seed null。"
                       f"W3b 并发顺序 {'、'.join(map(str, levels))}，每档 {rotations[block - 1]} 三段；W3L、W3XL 并发 32 各 A、B 段。"
                       "C 段只给 thinker 所在的进程开 torch profiler。")
            cfg["runtime_profile"] = ASYNC_RUNTIME
            out.append((cfg, {"priority": -(310 + 10 * block + position), "est": 1000, "timeout": 3600}))
    return out


def render() -> dict[str, str]:
    files = {f"configs/models/minicpm_o_4_5_exp_{name}.yaml": placement_yaml(name) for name in NEW_PLACEMENTS}
    files[EXP10_NOGRAPH_YAML] = nograph_yaml()
    check10, main10 = [exp10_check_entry()], exp10_entries()
    check11, main11 = exp11_check_entries(), exp11_entries()
    pilot = exp09_pilot_entries()
    main9 = exp09_entries() if EXP09_RATES_W3B and EXP09_RATES_W3L else []
    plans = [(PLAN_EXP09_PILOT, "exp09_pilot", "实验 9 预跑：4 种放置各一次，用来定开环速率（规则见 hypotheses.md）。", pilot)]
    if main9:
        plans.append((PLAN_EXP09, "exp09", "实验 9 正式运行：4 种放置 × 3 个区组，区组内顺序随机，区组依次运行。", main9))
    plans.append((PLAN_EXP10_CHECK, "exp10_check", "实验 10 正确性检查（排在实验 9 正式运行之后）。", check10))
    plans.append((PLAN_EXP10, "exp10", "实验 10 性能运行：3 个区组 × flow CUDA graph 开/关，排在正确性检查之后。", main10))
    plans.append((PLAN_EXP11_CHECK, "exp11_check", "实验 11 正确性检查：A、C、A 的重复（排在实验 10 之后）。", check11))
    plans.append((PLAN_EXP11, "exp11", "实验 11 性能对比：3 个区组 × A/B/C。只有正确性检查通过才放进队列。", main11))
    for path, plan_id, comment, entries in plans:
        plan = {"plan_id": plan_id,
                "_comment": comment + "通过 plan.d 放进 Job 454290。由 scripts/slurm/make_plan_exp09_11.py 生成。",
                "experiments": [plan_entry(c, m) for c, m in entries]}
        for e in plan["experiments"]:
            e["command"] = f"bash scripts/slurm/run_experiment.sh {EXP_DIR}/{e['experiment_id']}.json"
        files[path] = json.dumps(plan, ensure_ascii=False, indent=1) + "\n"
    for cfg, _ in pilot + main9 + check10 + main10 + check11 + main11:
        files[f"{EXP_DIR}/{cfg['experiment_id']}.json"] = json.dumps(cfg, ensure_ascii=False, indent=1) + "\n"
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--exp10-validation", nargs=2, metavar=("PLACEMENT", "SETTING"),
                    help="also write the conditional 实验 10 full-service validation (only when its registered trigger is met)")
    a = ap.parse_args()
    files = render()
    if a.exp10_validation:
        files.update(render_exp10_validation(*a.exp10_validation))
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
