#!/usr/bin/env python3
"""Configs and plans for R0-R2 of the pre-presentation validation: MiniCPM-o 4.5 full duplex on RTX A6000.

    python scripts/slurm/make_plan_ppv_r.py          # write configs/experiments/ppv/ppv_r*.json and plan files
    python scripts/slurm/make_plan_ppv_r.py --check  # list files that differ from what would be written

Design registered in research/next_steps/presentation_validation/preregistration.md (R0-R2), source facts
in research/next_steps/presentation_validation/r0_duplex_source_audit.md.
- R0 (functional, 1 GPU): the pinned stock sglang-omni 520b49ee (`runtime unicorn`), launched like the
  cookbook (`--variant session --enable-realtime`, default max_sessions 2); a 30 s smoke, then 1 session
  x 300 s and 2 sessions x 300 s of the seeded spoken-question script (duplex-v1 pool);
- R1 (single-session critical path, 1 GPU each): 1 session x 300 s with scripted barge-in turns, without
  (A) and with (B) the server's request-event recorder, 3 blocks (A/B order alternates); the application
  reference is the request-response path on the same single GPU (an R8 replica: `--code2wav.process
  code2wav`) with the same spoken questions (W3 audio -> speech), concurrency 1, A/B the same way;
- R2 (concurrent sessions): levels 1/2/4 on one GPU with max_sessions 4 (full-length KV reserved per
  session within the default memory fractions) and levels 4/8 on two GPUs (thinker alone on the second,
  max_sessions 8), 3 blocks each, ascending, every level 180 s, request events on; a level runs only if
  the previous level of the run had no failed session and a unit miss rate <= 1 %. Smoke runs first check whether max_sessions 8
  starts on one GPU (expected not: 9 x 8192 tokens of thinker KV exceed the thinker's 0.52 share after
  its weights) and whether the 2-GPU split starts and streams.
Every script uses the seed below, so session i receives the same input in every level, arm and block.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_plan_8gpu_v2 import base, plan_entry

REPO = Path(__file__).resolve().parents[2]
EXP_DIR = "configs/experiments/ppv"
CODE_DIR = "~/workspace/omni-ppv"
SEED = 20261119
RUNTIME = "unicorn"  # stock sglang-omni 520b49ee (repos/sglang-omni), no research patch
ATTEMPT_SUFFIX = ""  # bump (e.g. "_a2") to re-place a run: the orchestrator never re-runs an experiment_id
W3_AUDIO = "configs/workloads/w3_audio_speech.json"
BLOCKS = 3
UNIT_MISS_TARGET = 0.01  # upstream full_duplex_bench.md: a level meets the serving target at a unit miss rate <= 1 %
DUPLEX_ARGS = ["--variant", "session", "--enable-realtime"]
# The second GPU holds only the thinker; with a fraction other than the default 0.52 sglang-omni sizes its KV
# from the fraction instead of max_sessions x 8192 tokens (native_config.py stage_factory_kwargs).
TWO_GPU_ARGS = ["--thinker.gpu", "1", "--thinker.gpu_memory_fraction", "0.85"]


def duplex_phase(name: str, sessions: int, duration_s: float, *, events: bool = False, barge_in_every: int = 0,
                 skip_if_failed: list[str] | None = None, save_audio: int = 1, stop_if_unit_miss_above: float | None = None) -> dict:
    ph = {"kind": "plugin", "module": "duplex.phase", "name": name, "sessions": sessions, "duration_s": duration_s,
          "lead_in_s": 2.0, "answer_window_s": 12.0, "barge_in_every": barge_in_every, "barge_in_offset_s": 3.0,
          "stagger_s": 1.0, "session_seed": SEED, "asset_pool": "duplex-v1", "output_modalities": ["audio"],
          "save_audio_sessions": save_audio}
    if events:
        ph["server_events"] = True
    if skip_if_failed:
        ph["skip_if_failed"] = skip_if_failed
    if stop_if_unit_miss_above is not None:
        ph["stop_if_unit_miss_above"] = stop_if_unit_miss_above
    return ph


def duplex_server(max_sessions: int, gpus: int = 1) -> list[dict]:
    args = list(DUPLEX_ARGS) + (["--max_sessions", str(max_sessions)] if max_sessions != 2 else [])
    if gpus == 2:
        args += TWO_GPU_ARGS
    return [{"name": "s0", "model": "minicpm_o_4_5", "gpus": list(range(gpus)), "args": args, "ready_timeout_s": 2400}]


def chat_server() -> list[dict]:
    return [{"name": "s0", "model": "minicpm_o_4_5", "gpus": [0], "args": ["--code2wav.process", "code2wav"],
             "ready_timeout_s": 2400}]


def eid(name: str) -> str:
    return f"ppv_{name}{ATTEMPT_SUFFIX}"


def cfg(name: str, gpus: int, servers: list[dict], phases: list[dict], desc: str) -> dict:
    c = base(eid(name), gpus, servers, phases, desc)
    c["runtime_profile"] = RUNTIME
    c["seed"] = SEED
    return c


def r0_entries() -> list[tuple[dict, dict]]:
    phases = [duplex_phase("duplex_smoke", 1, 30.0),
              duplex_phase("duplex_s1", 1, 300.0, skip_if_failed=["duplex_smoke"]),
              duplex_phase("duplex_s2", 2, 300.0, skip_if_failed=["duplex_smoke"], save_audio=2)]
    c = cfg("r0_duplex_functional", 1, duplex_server(2), phases,
            "R0 全双工功能验证：固定版本 sglang-omni 520b49ee（原版），--variant session --enable-realtime，默认 max_sessions 2，"
            "1 张 A6000。30 秒冒烟，然后 1 个会话 300 秒、2 个会话 300 秒，输入是种子固定的合成语音提问（duplex-v1）。")
    return [(c, {"priority": 900, "est": 1500, "timeout": 3600})]


def r1_entries() -> list[tuple[dict, dict]]:
    out = []
    for block in range(1, BLOCKS + 1):
        a = duplex_phase("duplex_r1_A", 1, 300.0, barge_in_every=3)
        b = duplex_phase("duplex_r1_B", 1, 300.0, barge_in_every=3, events=True)
        order = [b, a] if block == 2 else [a, b]
        c = cfg(f"r1_duplex_block{block}", 1, duplex_server(2), [duplex_phase("duplex_smoke", 1, 30.0), *order],
                f"R1 第 {block} 个区组：单会话关键路径，全双工 1 个会话 300 秒，每第 3 个提问后 3 秒插入一句打断；"
                f"A 不开、B 开服务端请求事件（{'B 先' if block == 2 else 'A 先'}）。固定版本原版，1 张 A6000。")
        out.append((c, {"priority": 800 - 2 * block, "est": 1200, "timeout": 3600}))
        closed = lambda name, events: {"kind": "closed_loop", "name": name, "workload": W3_AUDIO,
                                       "concurrency": [1], "requests_per_level": 48, "requests_per_concurrency": 1,
                                       **({"server_events": True} if events else {})}
        ref = [closed("w3_c1_A", False), closed("w3_c1_B", True)]
        smoke = {"kind": "smoke", "name": "smoke_w3", "workload": W3_AUDIO, "requests": 2}
        c = cfg(f"r1_chat_ref_block{block}", 1, chat_server(), [smoke, *(ref[::-1] if block == 2 else ref)],
                f"R1 第 {block} 个区组的应用层参照：请求-响应路径（/v1/chat/completions），同一张 A6000 上的单卡语音流水线"
                "（code2wav 单独进程，与 plan v2 的 R8 副本相同），同一批合成语音提问（W3 语音进、语音出），并发 1，"
                "每段 48 个请求，A 不开、B 开服务端事件。")
        out.append((c, {"priority": 800 - 2 * block - 1, "est": 900, "timeout": 3600}))
    return out


def r2_levels(levels: list[int], events: bool = True) -> list[dict]:
    phases = []
    for i, n in enumerate(levels):
        prev = [f"duplex_n{levels[i - 1]}"] if i else None
        # Registered stop rule: the next level runs only if this one had no failed session and kept the unit miss
        # rate at or below 1 % (upstream's serving target): a server that already falls behind is not pushed further.
        phases.append(duplex_phase(f"duplex_n{n}", n, 180.0, events=events, skip_if_failed=prev,
                                   stop_if_unit_miss_above=UNIT_MISS_TARGET if prev else None))
    return phases


R2_ARMS = {  # arm -> (max_sessions, GPUs, levels)
    "ms4_1gpu": (4, 1, [1, 2, 4]),
    "ms8_2gpu": (8, 2, [4, 8]),
    "ms8_1gpu": (8, 1, [1, 2, 4, 8]),  # only if the 1-GPU smoke shows that max_sessions 8 starts and streams
}


def r2_smoke_entries() -> list[tuple[dict, dict]]:
    out = []
    for arm in ("ms8_1gpu", "ms8_2gpu"):
        ms, gpus, _ = R2_ARMS[arm]
        c = cfg(f"r2_smoke_{arm}", gpus, duplex_server(ms, gpus), [duplex_phase("duplex_smoke_n8", 8, 30.0)],
                f"R2 冒烟：max_sessions {ms}、{gpus} 张 A6000{'（thinker 单独放第 2 张卡）' if gpus == 2 else ''}能否启动，"
                "8 个会话各 30 秒能否跑通。只用来决定 R2 的放置，不进正式结果。")
        out.append((c, {"priority": 750 - len(out), "est": 900, "timeout": 3600}))
    return out


def r2_entries(arm: str) -> list[tuple[dict, dict]]:
    ms, gpus, levels = R2_ARMS[arm]
    out = []
    for block in range(1, BLOCKS + 1):
        c = cfg(f"r2_{arm}_block{block}", gpus, duplex_server(ms, gpus), r2_levels(levels),
                f"R2 第 {block} 个区组：max_sessions {ms}、{gpus} 张 A6000，并发会话 {'/'.join(map(str, levels))} 依次各 180 秒，"
                "开服务端请求事件；上一档有会话失败就不再升档。")
        out.append((c, {"priority": 700 - 10 * list(R2_ARMS).index(arm) - block, "est": 240 * len(levels) + 400,
                        "timeout": 3600}))
    return out


def r2b_entries() -> list[tuple[dict, dict]]:
    """R2b (registered after R2's data, before any R2b run; preregistration.md R2b): the registered 2-GPU split at 4 and
    8 sessions -- the same configs as ppv_r2_ms8_2gpu_block*, which were generated but never queued, under new IDs.
    Queued behind P2 verify (-714) so no other run shares the node; the plan chains the three blocks."""
    out = []
    for block, (c, m) in enumerate(r2_entries("ms8_2gpu"), start=1):
        c = json.loads(json.dumps(c))
        c["experiment_id"] = c["experiment_id"].replace("ppv_r2_", "ppv_r2b_", 1)
        c["description"] = "R2b（R2 之后登记，见 preregistration.md R2b）：" + c["description"]
        out.append((c, {**m, "priority": -714 - block}))
    return out


def command(experiment_id: str) -> str:
    return f"cd {CODE_DIR} && OMNI_REPO=\"$PWD\" bash scripts/slurm/run_experiment.sh {EXP_DIR}/{experiment_id}.json"


PLANS = {
    "plan_ppv_r0": ("R0 全双工功能验证（1 张卡）。", r0_entries),
    "plan_ppv_r1": ("R1 单会话关键路径（全双工 + 请求-响应参照，每个运行 1 张卡），只在 R0 通过后放入。", r1_entries),
    "plan_ppv_r2_smoke": ("R2 放置冒烟（max_sessions 8 能否单卡启动；双卡拆分能否跑通）。", r2_smoke_entries),
    **{f"plan_ppv_r2_{arm}": (f"R2 并发会话扫描，{arm}，3 个区组，只在 R0 通过、冒烟决定放置后放入。",
                              functools.partial(r2_entries, arm)) for arm in R2_ARMS},
    "plan_ppv_r2b": ("R2b：双卡放置（thinker 单独占第 2 张卡）下 4、8 个会话，3 个区组串行，排在 P2 逐位检查之后。", r2b_entries),
}


def render() -> dict[str, str]:
    files = {}
    for plan_name, (comment, fn) in PLANS.items():
        entries = fn()
        plan = {"plan_id": plan_name.removeprefix("plan_") + ATTEMPT_SUFFIX,
                "_comment": comment + "通过 plan.d 放进 Job 454290；命令从集群上本分支的 worktree 运行。由 scripts/slurm/make_plan_ppv_r.py 生成。",
                "experiments": [plan_entry(c, m) for c, m in entries]}
        for i, e in enumerate(plan["experiments"]):
            e["command"] = command(e["experiment_id"])
            # one latency run on the node at a time: each run waits for the previous one of its plan (a failed run
            # blocks the rest, which are then placed again by hand after looking at the failure)
            e["dependencies"] = [plan["experiments"][i - 1]["experiment_id"]] if i else []
        files[f"configs/experiments/{plan_name}{ATTEMPT_SUFFIX}.json"] = json.dumps(plan, ensure_ascii=False, indent=1) + "\n"
        for c, _ in entries:
            files[f"{EXP_DIR}/{c['experiment_id']}.json"] = json.dumps(c, ensure_ascii=False, indent=1) + "\n"
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
