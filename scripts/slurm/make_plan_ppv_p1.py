#!/usr/bin/env python3
"""Configs and plans for P1 of the pre-presentation validation: does length-aware batching with a max wait help the full
service near and past saturation, and does the max wait control the short requests' tail?

    python scripts/slurm/make_plan_ppv_p1.py          # write the pre-run (and, once P1_X is filled in, the main plan)
    python scripts/slurm/make_plan_ppv_p1.py --check  # list files that differ from what would be written

Registered in research/next_steps/p0_p3_experiment_plan.md (P1, 2026-11-18); execution details in
research/next_steps/presentation_validation/p1_p2_p3_design.md (written before any P1 run):
- placement t2_k2_c4 (实验 9's best), flow CUDA graph on (default), runtime unicorn_ppv_maxwait (实验 10's batching
  patch + max_wait_ms); the setting is switched by control phases inside a run, every key written explicitly;
- settings FIFO16, Length8, Length8 + max wait 0.5 / 1 / 2 s; workloads W3b, W3L and MIX (W3b, W3L half each);
- pre-run: closed loop W3b 256, W3L 128, MIX 192 under FIFO16, no events; X per workload = the busy-window
  throughput of that level (completions between the first and the last send of the measured requests / that window),
  the registered capacity (first send -> last completion) is reported next to it;
- main: 3 blocks (one run each, a server restart per block); workload order rotated by block; inside a workload the
  15 cells (5 settings x {0.9X, 1.0X, 1.1X}) in a random order per block (seed 20261119); one open-loop phase per
  cell, 15 s (W3L 20 s) of warm-up that the analysis drops, then the registered 75 s (W3L 90 s); events on;
  the arrival seed is fixed per block and rate, so the five settings of a block see the same arrivals and requests;
- a four-arm variant (dense / packed flow x FIFO16 / Length8 + 1 s), written but only placed if C1 passes.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_plan_8gpu_v2 import CODE2WAV_FRACTION, TALKER_FRACTION, W3B, base, plan_entry, smoke  # noqa: E402
from make_plan_ppv_p0 import command  # noqa: E402  same worktree and OMNI_REPO rule as P0

REPO = Path(__file__).resolve().parents[2]
EXP_DIR = "configs/experiments/ppv"
W3L = "configs/workloads/w3l_long_speech.json"
MIX = [{"workload": W3B, "weight": 0.5}, {"workload": W3L, "weight": 0.5}]
SEED = 20261119  # registered in p0_p3_experiment_plan.md
BLOCKS = 3
RUNTIME = "unicorn_ppv_maxwait"
PLACEMENT_YAML = "configs/models/minicpm_o_4_5_exp_t2_k2_c4.yaml"
PACKED_YAML = "configs/models/minicpm_o_4_5_ppv_p1_t2_k2_c4_packed.yaml"
CONTROL_PATH = "{exp_dir}/c2w_control.json"
SERVER_ENV = {"SGLANG_OMNI_C2W_CONTROL": CONTROL_PATH, "SGLANG_OMNI_PORT_CLAIM_DIR": "{exp_dir}/port_claims",
              "SGLANG_OMNI_EXP_STACK_DIR": "{exp_dir}/stacks"}  # stacks: SIGUSR2 dumps for the stall watcher (P3)
# setting -> control file content (every key explicit: a missing key would keep the previous setting)
SETTINGS = {
    "fifo16": {"max_batch_size": 16, "policy": "fifo", "window": 64, "max_wait_ms": None},
    "length8": {"max_batch_size": 8, "policy": "length", "window": 64, "max_wait_ms": None},
    "length8_w500": {"max_batch_size": 8, "policy": "length", "window": 64, "max_wait_ms": 500},
    "length8_w1000": {"max_batch_size": 8, "policy": "length", "window": 64, "max_wait_ms": 1000},
    "length8_w2000": {"max_batch_size": 8, "policy": "length", "window": 64, "max_wait_ms": 2000},
}
WORKLOADS = ("w3b", "w3l", "mix")
PRERUN_CONCURRENCY = {"w3b": 256, "w3l": 128, "mix": 192}  # registered
FRACTIONS = {"x090": 0.9, "x100": 1.0, "x110": 1.1}
MEASURED_S = {"w3b": 75, "w3l": 90, "mix": 75}  # registered
WARMUP_S = {"w3b": 15, "w3l": 20, "mix": 15}
# Busy-window capacity X (requests/s) per workload from the pre-run (scripts/analysis/ppv_p1_report.py --prerun).
# None until the pre-run has been analysed; the main plan is written only then.
P1_X: dict[str, float] | None = {"w3b": 28.96, "w3l": 14.52, "mix": 14.67}  # 20261119_ppv_p1_prerun_run01 (Erratum E-P1-X fixed)
PRERUN_ID = "ppv_p1_prerun"


def round_half(x: float) -> float:
    """Half up to a multiple of 0.5 requests/s, the rule of 实验 9's open-loop rates."""
    return int(2 * x + 0.5) / 2


def rates(x: dict[str, float]) -> dict[str, dict[str, float]]:
    return {w: {tag: round_half(f * x[w]) for tag, f in FRACTIONS.items()} for w in WORKLOADS}


def server(yaml: str = PLACEMENT_YAML) -> list[dict]:
    return [{"name": "s0", "model": "minicpm_o_4_5", "gpus": list(range(8)), "args": ["--config", "{research_repo}/" + yaml],
             "ready_timeout_s": 2400, "env": dict(SERVER_ENV)}]


def control(name: str, setting: str) -> dict:
    return {"kind": "control", "name": f"ctl_{name}", "path": CONTROL_PATH, "content": dict(SETTINGS[setting]), "settle_s": 3}


def with_workload(ph: dict, w: str) -> dict:
    if w == "mix":
        ph["workload_mix"] = [dict(m) for m in MIX]
    else:
        ph["workload"] = W3B if w == "w3b" else W3L
    return ph


def prerun_entry() -> tuple[dict, dict]:
    phases = [smoke(), control("prerun", "fifo16")]
    for w in WORKLOADS:
        c = PRERUN_CONCURRENCY[w]
        phases.append(with_workload({"kind": "closed_loop", "name": f"pre_{w}_c{c}", "concurrency": [c],
                                     "requests_per_level": 96, "requests_per_concurrency": 3 if w == "w3b" else 2}, w))
    cfg = base(PRERUN_ID, 8, server(), phases,
               "P1 预跑：放置 t2_k2_c4，flow CUDA graph 开，FIFO16（控制文件写明）。闭环 W3b 256、W3L 128、MIX 192，各预热一波再测，"
               "不开事件。按登记的规则用满额时段吞吐定每种负载的 X，正式运行取 0.9X/1.0X/1.1X。")
    cfg["runtime_profile"] = RUNTIME
    cfg["stall_watch"] = True  # P3: log-gap watcher (runner default is off)
    return cfg, {"priority": -600, "est": 1500, "timeout": 3600}


def cell(w: str, setting: str, tag: str, rate: float, block: int) -> list[dict]:
    name = f"{w}_{setting}_{tag}"
    duration = WARMUP_S[w] + MEASURED_S[w]
    ph = with_workload({"kind": "open_loop", "name": name, "rates_rps": [rate], "duration_per_rate_s": duration,
                        "min_requests_per_rate": int(rate * duration), "max_error_rate": 0.5, "server_events": True,
                        "warmup_s": WARMUP_S[w], "seed": SEED + block}, w)
    return [control(name, setting), ph]


def main_entries(x: dict[str, float]) -> list[tuple[dict, dict]]:
    out = []
    r = rates(x)
    for block in range(1, BLOCKS + 1):
        order = list(WORKLOADS[block - 1:] + WORKLOADS[:block - 1])  # w3b/w3l/mix, w3l/mix/w3b, mix/w3b/w3l
        phases = [smoke()]
        for k, w in enumerate(order):
            rng = random.Random(SEED * 100 + 10 * block + k)
            cells = [(s, t) for s in SETTINGS for t in FRACTIONS]
            rng.shuffle(cells)
            for setting, tag in cells:
                phases += cell(w, setting, tag, r[w][tag], block)
        cfg = base(f"ppv_p1_block{block}", 8, server(), phases,
                   f"P1 第 {block} 个区组：放置 t2_k2_c4，flow CUDA graph 开。负载顺序 {'/'.join(order)}；每种负载 5 种设置"
                   "（FIFO16、Length8、Length8+最长等待 0.5/1/2 秒）× 3 档速率（0.9X/1.0X/1.1X），顺序随机（种子 20261119），"
                   "控制文件切换，每格一个开环段（先预热再测登记的时长），开事件。")
        cfg["runtime_profile"] = RUNTIME
        cfg["stall_watch"] = True  # P3: log-gap watcher (runner default is off)
        out.append((cfg, {"priority": -(610 + block), "est": 6000, "timeout": 14400}))
    return out


def packed_yaml() -> str:
    text = (REPO / PLACEMENT_YAML).read_text()
    first, rest = text.split("\n", 1)
    header = ("# P1 的四组变体（只在 C1 通过时运行）：放置 t2_k2_c4，code2wav 的 flow 换成变长打包 + 编译"
              "（enable_flow_variable_length、enable_flow_block_compile），其余与 t2_k2_c4 相同。由 scripts/slurm/make_plan_ppv_p1.py 生成。\n")
    body = (first + "\n" + rest).replace(
        "    process: code2wav\n",
        "    process: code2wav\n    factory:\n      enable_flow_variable_length: true\n      enable_flow_block_compile: true\n", 1)
    assert "enable_flow_variable_length" in body, "t2_k2_c4 yaml no longer has the expected code2wav block"
    return header + body


def interaction_entries(x: dict[str, float]) -> list[tuple[dict, dict]]:
    """Dense / packed flow x FIFO16 / Length8 + 1 s at 1.0X and 1.1X, MIX and W3b; one run per flow variant and block
    (a restart is needed to switch the flow), the variant order alternating by block, cells in a seeded random order
    that both variants of a block share."""
    out = []
    r = rates(x)
    for block in range(1, BLOCKS + 1):
        rng = random.Random(SEED * 100 + 50 + block)
        cells = [(w, s, t) for w in ("mix", "w3b") for s in ("fifo16", "length8_w1000") for t in ("x100", "x110")]
        rng.shuffle(cells)
        variants = ["dense", "packed"] if block % 2 else ["packed", "dense"]
        for position, variant in enumerate(variants, start=1):
            phases = [smoke()]
            for w, s, t in cells:
                phases += cell(w, s, t, r[w][t], block)
            cfg = base(f"ppv_p1x_{variant}_block{block}", 8, server(PACKED_YAML if variant == "packed" else PLACEMENT_YAML), phases,
                       f"P1 四组变体第 {block} 个区组第 {position} 个（只在 C1 通过时运行）：flow {'变长打包 + 编译' if variant == 'packed' else '默认稠密'}；"
                       "MIX 和 W3b，FIFO16 与 Length8+1 秒，1.0X 与 1.1X，顺序随机且两种 flow 相同。")
            cfg["runtime_profile"] = RUNTIME
            cfg["stall_watch"] = True  # P3: log-gap watcher (runner default is off)
            out.append((cfg, {"priority": -(650 + 10 * block + position), "est": 1500, "timeout": 5400}))
    return out


def plan_file(plan_id: str, comment: str, entries: list[tuple[dict, dict]]) -> str:
    plan = {"plan_id": plan_id, "_comment": comment + " 通过 plan.d 放进 Job 454290；命令从集群上本分支的 worktree 运行。"
            "由 scripts/slurm/make_plan_ppv_p1.py 生成。", "experiments": [plan_entry(c, m) for c, m in entries]}
    for e in plan["experiments"]:
        e["command"] = command(e["experiment_id"])
    return json.dumps(plan, ensure_ascii=False, indent=1) + "\n"


def render(x: dict[str, float] | None = None) -> dict[str, str]:
    x = x if x is not None else P1_X
    pre = [prerun_entry()]
    files = {PACKED_YAML: packed_yaml(),
             "configs/experiments/plan_ppv_p1_prerun.json": plan_file("ppv_p1_prerun", "P1 预跑（定 X）。", pre)}
    entries = list(pre)
    if x is not None:
        main, inter = main_entries(x), interaction_entries(x)
        files["configs/experiments/plan_ppv_p1.json"] = plan_file("ppv_p1", "P1 正式运行：3 个区组，排在预跑之后。", main)
        files["configs/experiments/plan_ppv_p1_interaction.json"] = plan_file(
            "ppv_p1_interaction", "P1 四组变体（稠密/打包 flow × FIFO16/Length8+1 秒），只在 C1 通过时放进队列。", inter)
        entries += main + inter
    for cfg, _ in entries:
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
