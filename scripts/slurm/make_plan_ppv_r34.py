#!/usr/bin/env python3
"""Configs and plans for R3 (camera-frame replay) and R4 (robot-session working set) of the pre-presentation validation.

    python scripts/slurm/make_plan_ppv_r34.py                      # write configs/experiments/ppv/r3_*, r4_*, plans
    python scripts/slurm/make_plan_ppv_r34.py --check              # list files that differ from what would be written
    python scripts/slurm/make_plan_ppv_r34.py --r3-confirm frame_s16_f2,clip_s4_f1   # also the 3-block confirmation

Design registered in research/next_steps/presentation_validation/r3_r4_design.md:
- R3: one 1-GPU replica (--text-only, as 实验 14), runtime unicorn_ppv_kv; preprocessing max_concurrency 1 (default)
  and 4 (the patch's factory argument) as two runs that may share the node on different GPUs. Screening: shapes
  frame/clip x fps 0.5/1/2 x sessions 1/4/16, one run per concurrency, cells in a seeded random order (the same
  order in both runs), request events on. Confirmation: only the cells the registered rule picks
  (ppv_r3_report.confirm_cells), 3 blocks x both concurrencies.
- R4: 2+2+4 (configs/models/minicpm_o_4_5_exp_t2_k2_c4.yaml), runtime unicorn_ppv_kv (SGLANG_OMNI_EXP_KV=1);
  a smoke run (4 sessions) to check token and media statistics, then 3 blocks; each block runs 8, 16, 32, 64 and
  128 sessions (text history, history_media 0) and the contrast 32 sessions with history_media 2.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_plan_8gpu_v2 import plan_entry  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EXP_DIR = "configs/experiments/ppv"
CODE_DIR = "~/workspace/omni-ppv"
RUNTIME = "unicorn_ppv_kv"
SEED = 20261119  # the pre-presentation validation's seed (p0_p3_experiment_plan.md)
W4 = "configs/workloads/w4_video_text.json"
W3B = "configs/workloads/w3b_text_speech.json"
R4_YAML = "configs/models/minicpm_o_4_5_exp_t2_k2_c4.yaml"

R3_SHAPES = ("frame", "clip")
R3_FPS = (0.5, 1.0, 2.0)
R3_SESSIONS = (1, 4, 16)
R3_CONCURRENCY = (1, 4)
R3_WARMUP_S, R3_MEASURE_S = 10.0, 60.0
R4_LEVELS = (8, 16, 32, 64, 128)
R4_CONTRAST = {"sessions": 32, "history_media": 2}
BLOCKS = 3


def command(eid: str) -> str:
    # the orchestrator exports OMNI_REPO=<its own checkout>; point it at this branch's worktree on the cluster
    return f"cd {CODE_DIR} && OMNI_REPO=\"$PWD\" bash scripts/slurm/run_experiment.sh {EXP_DIR}/{eid}.json"


def base(eid: str, gpus: int, servers: list[dict], phases: list[dict], desc: str) -> dict:
    return {"runtime_profile": RUNTIME, "hardware_label": f"a6000x{gpus}", "seed": SEED, "assets": "synthetic-v1",
            "experiment_id": eid, "description": desc, "restart_dead_servers": True, "max_server_restarts": 2,
            "servers": servers, "phases": phases, "stall_watch": True}  # P3 observations ride along (runner default is off)


# ---------------------------------------------------------------- R3


def cell_key(shape: str, sessions: int, fps: float) -> str:
    return f"{shape}_s{sessions}_f{fps:g}"


def r3_cells(seed_offset: int = 0) -> list[tuple[str, int, float]]:
    cells = [(sh, s, f) for sh in R3_SHAPES for f in R3_FPS for s in R3_SESSIONS]
    random.Random(SEED * 100 + 30 + seed_offset).shuffle(cells)
    return cells


def camera_phase(shape: str, sessions: int, fps: float) -> dict:
    return {"kind": "plugin", "module": "sessions.camera", "name": f"cam_{cell_key(shape, sessions, fps).replace('.', 'p')}",
            "sessions": sessions, "fps": fps, "shape": shape, "policy": "drop_if_busy", "warmup_s": R3_WARMUP_S,
            "duration_s": R3_MEASURE_S, "capture_fps": 8.0, "clip_s": 2.0, "max_tokens": 16, "request_timeout_s": 60.0,
            "server_events": True}


def r3_server(concurrency: int) -> list[dict]:
    args = ["--text-only"] + (["--preprocessing.factory.max_concurrency", str(concurrency)] if concurrency != 1 else [])
    return [{"name": "r0", "model": "minicpm_o_4_5", "gpus": [0], "args": args, "ready_timeout_s": 2400}]


def r3_config(eid: str, concurrency: int, cells: list[tuple[str, int, float]], desc: str) -> dict:
    phases = [{"kind": "smoke", "name": "smoke_w4", "workload": W4, "requests": 2}] + [camera_phase(*c) for c in cells]
    cfg = base(eid, 1, r3_server(concurrency), phases, desc)
    cfg["ppv_preprocessing_concurrency"] = concurrency
    return cfg


def r3_screen_entries() -> list[tuple[dict, dict]]:
    cells = r3_cells()
    out = []
    for c in R3_CONCURRENCY:
        desc = (f"R3 筛选：1 张卡一个完整副本（--text-only，与实验 14 相同），预处理并发 {c}"
                f"{'（默认）' if c == 1 else '（补丁的工厂参数）'}。合成摄像头回放：帧/短片 × 每会话 0.5/1/2 帧每秒 × 1/4/16 个会话，"
                f"18 格顺序随机（种子 {SEED}，两种并发相同），每格预热 {R3_WARMUP_S:g} 秒 + 计量 {R3_MEASURE_S:g} 秒，开请求事件。")
        out.append((r3_config(f"ppv_r3_screen_c{c}", c, cells, desc),
                    {"priority": -600 - c, "est": 2100, "timeout": 4800}))
    return out


def r3_confirm_entries(keys: list[str]) -> list[tuple[dict, dict]]:
    known = {cell_key(*c): c for c in r3_cells()}
    unknown = [k for k in keys if k not in known]
    if unknown or not keys:
        raise ValueError(f"unknown or empty R3 cells: {unknown or keys}")
    out = []
    for block in range(1, BLOCKS + 1):
        cells = [known[k] for k in keys]
        random.Random(SEED * 100 + 40 + block).shuffle(cells)
        for c in R3_CONCURRENCY:
            desc = (f"R3 确认第 {block} 个区组：预处理并发 {c}，只跑筛选按登记规则选出的 {len(cells)} 格（{', '.join(keys)}），"
                    f"区组内顺序随机（种子 {SEED}），两种并发同一顺序。")
            out.append((r3_config(f"ppv_r3_confirm_c{c}_block{block}", c, cells, desc),
                        {"priority": -620 - 10 * block - c, "est": 300 + 90 * len(cells), "timeout": 3600}))
    return out


# ---------------------------------------------------------------- R4


def r4_phase(sessions: int, history_media: int = 0, name: str | None = None) -> dict:
    return {"kind": "plugin", "module": "sessions.robot_sessions", "name": name or f"r4_n{sessions}" + (f"_media{history_media}" if history_media else ""),
            "sessions": sessions, "turns": [4, 10], "preamble_tokens": [50, 300], "ramp_s": 20.0, "think_s": [2.0, 0.5],
            "tool": {"p": 0.5, "latency_s": [0.3, 0.6], "result_tokens": [40, 200]}, "history_media": history_media,
            "speak_final": True, "image": True, "audio": True, "max_tokens_plan": 32, "max_tokens_answer": 48,
            "request_timeout_s": 300.0, "server_events": True}


def r4_server() -> list[dict]:
    return [{"name": "s0", "model": "minicpm_o_4_5", "gpus": list(range(8)), "args": ["--config", "{research_repo}/" + R4_YAML],
             "ready_timeout_s": 2400}]


def r4_smoke_entry() -> tuple[dict, dict]:
    phases = [{"kind": "smoke", "name": "smoke_w3b", "workload": W3B, "requests": 2},
              {**r4_phase(4, name="r4_smoke_n4"), "turns": [2, 3], "ramp_s": 2.0},
              {**r4_phase(4, history_media=2, name="r4_smoke_n4_media2"), "turns": [3, 3], "ramp_s": 2.0},
              # same sessions without image and audio: the first request of each session differs only by its media
              {**r4_phase(4, name="r4_smoke_n4_textonly"), "turns": [2, 3], "ramp_s": 2.0, "image": False, "audio": False,
               "content_key": "r4_smoke_n4"}]
    desc = ("R4 冒烟：2+2+4，ppv-kv 运行时（thinker 前缀命中和 KV 池日志）。4 个机器人会话（每轮新图像 + 合成语音，文字历史）"
            "、4 个带 2 轮媒体历史的会话和同一批会话的纯文字版（量出每轮图像 + 语音的 token 数），用来核对每个请求的 token 数、"
            "多模态内容、exp_kv 日志与请求的对应。")
    return base("ppv_r4_smoke", 8, r4_server(), phases, desc), {"priority": -700, "est": 900, "timeout": 3600}


def r4_entries() -> list[tuple[dict, dict]]:
    out = []
    for block in range(1, BLOCKS + 1):
        phases = [{"kind": "smoke", "name": "smoke_w3b", "workload": W3B, "requests": 2}]
        phases += [r4_phase(n) for n in R4_LEVELS]
        phases.append(r4_phase(R4_CONTRAST["sessions"], R4_CONTRAST["history_media"]))
        desc = (f"R4 第 {block} 个区组：2+2+4，ppv-kv 运行时。机器人会话数 {'/'.join(map(str, R4_LEVELS))}（每档一段，从小到大；"
                f"文字历史），最后 32 个会话带 2 轮媒体历史作对照。每个会话 4–10 轮，一半的轮先调一次模拟工具，最终回答用语音；"
                f"种子 {SEED}（每个区组同一批会话内容）。开请求事件。")
        out.append((base(f"ppv_r4_block{block}", 8, r4_server(), phases, desc),
                    {"priority": -710 - block, "est": 1500, "timeout": 5400}))
    return out


# ---------------------------------------------------------------- output


def plan_file(plan_id: str, comment: str, entries: list[tuple[dict, dict]]) -> str:
    plan = {"plan_id": plan_id, "_comment": comment + " 由 scripts/slurm/make_plan_ppv_r34.py 生成。",
            "experiments": [plan_entry(c, m) for c, m in entries]}
    for e in plan["experiments"]:
        e["command"] = command(e["experiment_id"])
    return json.dumps(plan, ensure_ascii=False, indent=1) + "\n"


def render(confirm: list[str] | None = None) -> dict[str, str]:
    files: dict[str, str] = {}
    screen, smoke, main = r3_screen_entries(), [r4_smoke_entry()], r4_entries()
    groups = [("configs/experiments/plan_ppv_r3_screen.json", "ppv_r3_screen", "R3 筛选：预处理并发 1 和 4 各一个 1 卡运行，可以同时在两张卡上跑。", screen),
              ("configs/experiments/plan_ppv_r4_smoke.json", "ppv_r4_smoke", "R4 冒烟（8 卡）：先核对 token 和多模态统计，再放正式区组。", smoke),
              ("configs/experiments/plan_ppv_r4.json", "ppv_r4", "R4 正式 3 个区组（8 卡），冒烟通过后放。", main)]
    if confirm:
        groups.append(("configs/experiments/plan_ppv_r3_confirm.json", "ppv_r3_confirm",
                       "R3 确认：按登记规则选出的格子，3 个区组 × 两种并发。", r3_confirm_entries(confirm)))
    for path, plan_id, comment, entries in groups:
        files[path] = plan_file(plan_id, comment, entries)
        for cfg, _ in entries:
            files[f"{EXP_DIR}/{cfg['experiment_id']}.json"] = json.dumps(cfg, ensure_ascii=False, indent=1) + "\n"
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--r3-confirm", default=None, help="comma-separated cells picked by ppv_r3_report.confirm_cells")
    a = ap.parse_args()
    files = render(a.r3_confirm.split(",") if a.r3_confirm else None)
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
