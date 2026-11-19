#!/usr/bin/env python3
"""R2 (concurrent full-duplex sessions): the registered verdicts.

    # on the cluster, with the sglang-omni runtime's python (the stage verdict needs the server-side timing):
    PYTHONPATH=src <runtime python> scripts/analysis/ppv_r2_report.py --data-root <PROJECT_DATA_ROOT> --out r2.json [--md r2.md]
    # elsewhere the stage verdict is reported as unavailable, never guessed.

Rules: research/next_steps/presentation_validation/preregistration.md, R2 and "R1/R2 分析口径补充" items 9-16.
- Per placement arm (ms8_1gpu, or ms4_1gpu plus ms8_2gpu) and level, the three blocks' sessions pooled.
- A level meets the serving target when it ran in all three blocks, no block had a failed session, a 503
  (connection refused) or a pacing violation, and the pooled unit miss rate is <= 1 %. Capacity = highest level of
  the unbroken run of target-meeting levels from the lowest (takewhile); all meet -> ">= top" (not bounded).
- The deadline capacity: the same, with the pooled deadline-miss share (all sent user turns, no_response counts)
  <= 5 % in place of the unit miss rate, D = 1 s primary and 0.5 / 2 s sensitivity.
- Skipped levels (registered stop rule) are listed per block and never meet the target.
- First limiting stage: base = the arm's lowest level, target = first level with pooled unit miss > 1 % (else the
  top level); per-stage median of per-unit queue + service, target minus base, largest increase wins; the
  stage's process CPU P90 >= 95 % with its GPU < 80 % busy -> CPU-bound, its GPU >= 90 % -> GPU-bound, else
  neither saturated.
- Per-block P50 with a t interval of the block P50s (df 2); P99 is N/A below 100 samples.
- Smoke runs: the registered placement rule, and whether the arms that ran follow it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "slurm"))

from make_plan_ppv_r import BLOCKS, R2_ARMS, UNIT_MISS_TARGET  # noqa: E402

from omni_research.analysis.experiment_analysis import read_jsonl  # noqa: E402
from omni_research.duplex.metrics import DEADLINES_S, dist, pool  # noqa: E402
from omni_research.duplex.report import (  # noqa: E402
    choose_runs,
    classify_failure,
    discover_runs,
    duplex_phase_metrics,
    gpu_window,
    load_gpu_samples_utc,
    load_server_units,
    oom_events,
    pid_stages,
    run_status,
    speaking_selector,
    stage_cpu,
    stage_gpu_position,
    stage_table,
    t_interval,
)

RUN_RE = re.compile(r"^ppv_r2_(?:(smoke)_)?(ms4_1gpu|ms8_2gpu|ms8_1gpu)(?:_block(\d+))?(?:_a\d+)?$")
DEADLINE_TARGET = 0.05
CPU_BOUND_P90 = 95.0  # P0 H0b thresholds (item 13)
GPU_NOT_BUSY = 80.0
GPU_BOUND = 90.0
SMOKE_PHASE = "duplex_smoke_n8"


def phase_summaries(exp: Path) -> dict[str, dict[str, Any]]:
    p = exp / "analysis" / "phases.json"
    return {s.get("phase"): s for s in json.loads(p.read_text())} if p.exists() else {}


def server_args(exp: Path) -> list[str]:
    p = exp / "config.json"
    if not p.exists():
        return []
    servers = json.loads(p.read_text()).get("servers") or [{}]
    return [str(x) for x in servers[0].get("args", [])]


def denials(phase_dir: Path, summary: dict[str, Any]) -> int:
    """Sessions that saw at least one HTTP 503 at connect (from the phase summary, else from the traces)."""
    d = (summary.get("duplex") or {}).get("sessions_with_admission_denials")
    if d is not None:
        return int(d)
    n = 0
    for t in sorted(phase_dir.glob("session_*.jsonl")):
        n += any(r.get("d") == "admission" for r in read_jsonl(t))
    return n


def level_block(exp: Path, level: int, gpu_samples: list[dict[str, Any]], cpu_samples: list[dict[str, Any]],
                pids: dict[str, str], ooms: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    name = f"duplex_n{level}"
    summary = phase_summaries(exp).get(name)
    d = exp / "raw" / f"duplex_{name}"
    if summary is None:
        return {"ran": False, "skipped": False, "reason": "phase not reached (run ended earlier)"}
    if summary.get("skipped") or not (d / "scripts.json").exists():
        return {"ran": False, "skipped": bool(summary.get("skipped")), "reason": summary.get("reason") or summary.get("error")}
    sessions = duplex_phase_metrics(d)["sessions"]
    p = pool(sessions)
    t0, t1 = summary.get("started"), summary.get("ended")
    units, _, reason = load_server_units(exp / "raw" / "server_events" / name)
    in_window = [o for o in ooms or [] if o["t"] is not None and t0 and t1 and t0 <= o["t"] <= t1]
    return {"ran": True, "sessions_metrics": sessions, "pooled": p, "denials": denials(d, summary),
            "oom_events": len(in_window), "oom_first": in_window[0]["line"] if in_window else None,
            "mean_open_sessions": (summary.get("duplex") or {}).get("mean_open_sessions"),
            "window": [t0, t1], "gpu": gpu_window(gpu_samples, t0, t1) if t0 and t1 else {},
            "cpu_raw": stage_cpu(cpu_samples, t0, t1, pids, raw=True) if t0 and t1 else {},
            "server_units": units, "server_reason": reason, "servers_exited": summary.get("servers_exited")}


def compact_block(b: dict[str, Any]) -> dict[str, Any]:
    if not b["ran"]:
        return {k: b[k] for k in ("ran", "skipped", "reason")}
    p = b["pooled"]
    return {"ran": True, "sessions": p["sessions"], "sessions_failed": p["sessions_failed"], "denials": b["denials"],
            "pacing_violations": p["pacing_violations"], "unit_miss_rate": p["unit_miss_rate"],
            "unit_lag_p50_ms": p["unit_lag_ms"]["p50"], "ttfa_p50_ms": p["ttfa_ms"]["p50"],
            "deadline_miss_all_turns": p["deadline_miss_all_turns"], "mean_open_sessions": b["mean_open_sessions"],
            "gpu": b["gpu"], "cpu": {k: dist(v) for k, v in b["cpu_raw"].items()}, "servers_exited": b["servers_exited"],
            "oom_events": b["oom_events"], "oom_first": b["oom_first"],
            "server_stage_timing": "available" if b["server_units"] is not None else b["server_reason"],
            **unit_mix(b["sessions_metrics"])}


def unit_mix(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive (not a registered rule): share of speaking units and the unit lag of each kind. A speaking
    unit runs the talker and the vocoder, so the all-unit P50 moves with the speaking share."""
    by = {k: [x for s in sessions for x in (s.get("unit_lags_by_decision_ms") or {}).get(k, [])]
          for k in ("speak", "listen", "text_only")}
    n = sum(len(v) for v in by.values())
    return {"speak_share": len(by["speak"]) / n if n else None,
            "unit_lag_speaking_ms": dist(by["speak"]), "unit_lag_listening_ms": dist(by["listen"])}


def ttfa_parts(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive: TTFA = (a) unit fill + (b) k further units + (c) the speaking unit's service (metrics.ttfa_parts)."""
    parts = [r["parts"] for s in sessions for r in s["turn_rows"] if r.get("status") == "answered" and r.get("parts")]
    reg = [p for p in parts if not p["early_start"]]
    return {"answered": len(parts), "early_start": len(parts) - len(reg),
            "k_counts": {str(k): sum(1 for p in reg if p["k"] == k) for k in sorted({p["k"] for p in reg})},
            "a_wait_unit_end_ms": dist([p["a_wait_unit_end_ms"] for p in reg]),
            "b_listen_ms": dist([p["b_listen_ms"] for p in reg]), "c_service_ms": dist([p["c_service_ms"] for p in reg])}


def level_entry(level: int, blocks: dict[int, dict[str, Any]], expected_blocks: int) -> dict[str, Any]:
    ran = {k: v for k, v in blocks.items() if v["ran"]}
    sessions = [s for v in ran.values() for s in v["sessions_metrics"]]
    p = pool(sessions) if sessions else None
    all_ran = len(ran) == expected_blocks
    clean = all(v["pooled"]["sessions_failed"] == 0 and v["denials"] == 0 and v["pooled"]["pacing_violations"] == 0
                for v in ran.values())
    miss = p["unit_miss_rate"] if p else None
    meets = bool(all_ran and clean and miss is not None and miss <= UNIT_MISS_TARGET)
    deadline = {f"{d:g}s": (p["deadline_miss_all_turns"][f"{d:g}s"] if p else None) for d in DEADLINES_S}
    meets_deadline = {k: bool(all_ran and all(v["pooled"]["sessions_failed"] == 0 for v in ran.values())
                              and x is not None and x <= DEADLINE_TARGET) for k, x in deadline.items()}
    out: dict[str, Any] = {
        "level": level, "blocks_ran": sorted(ran), "blocks_expected": expected_blocks, "all_blocks_ran": all_ran,
        "skipped_in_blocks": {str(k): v.get("reason") for k, v in blocks.items() if not v["ran"]},
        "no_failed_no_503_no_pacing": clean, "unit_miss_rate": miss, "meets_serving_target": meets,
        "deadline_miss_all_turns": deadline, "meets_deadline_target": meets_deadline,
        "per_block": {str(k): compact_block(v) for k, v in sorted(blocks.items())},
        "block_p50": {"ttfa_ms": t_interval([v["pooled"]["ttfa_ms"]["p50"] for v in ran.values()]),
                      "unit_lag_ms": t_interval([v["pooled"]["unit_lag_ms"]["p50"] for v in ran.values()])},
    }
    if p:
        out["pooled"] = {k: p[k] for k in ("sessions", "sessions_failed", "errors", "fatal_errors", "units_planned",
                                           "units_missed", "turns", "answered", "no_response", "replies_with_audio",
                                           "replies_with_stall", "audio_s_total", "decisions", "pacing_violations")}
        out["pooled"].update(ttfa_ms=p["ttfa_ms"], unit_lag_ms=p["unit_lag_ms"], first_audio_latency_ms=p["first_audio_latency_ms"],
                             delta_gap_ms=p["delta_gap_ms"], denials=sum(v["denials"] for v in ran.values()),
                             mean_open_sessions=[v["mean_open_sessions"] for v in ran.values()],
                             oom_events=sum(v["oom_events"] for v in ran.values()))
        out["descriptive"] = {**unit_mix(sessions), "ttfa_parts": ttfa_parts(sessions)}
    return out


def capacity(levels: list[dict[str, Any]], ok: Any) -> dict[str, Any]:
    """Takewhile from the lowest level (upstream serving_report); all levels ok -> not bounded."""
    meeting = []
    for e in sorted(levels, key=lambda e: e["level"]):
        if not ok(e):
            break
        meeting.append(e["level"])
    top = max((e["level"] for e in levels), default=None)
    return {"capacity": meeting[-1] if meeting else None, "bounded": len(meeting) < len(levels),
            "levels_meeting": meeting, "statement": (f">= {top} (every level met the target)" if levels and len(meeting) == len(levels)
                                                     else (f"{meeting[-1]}" if meeting else "below the lowest level"))}


def first_limiting_stage(arm_levels: list[dict[str, Any]], raw: dict[int, dict[int, dict[str, Any]]],
                         args: list[str]) -> dict[str, Any]:
    levels = sorted(e["level"] for e in arm_levels if e["blocks_ran"])
    if len(levels) < 2:
        return {"verdict": None, "reason": "fewer than two levels ran"}
    base = levels[0]
    over = [e["level"] for e in sorted(arm_levels, key=lambda e: e["level"]) if e["unit_miss_rate"] is not None
            and e["unit_miss_rate"] > UNIT_MISS_TARGET and e["blocks_ran"]]
    target = over[0] if over else levels[-1]
    tables = {}
    for lv in (base, target):
        units, missing = [], []
        for b, v in raw[lv].items():
            if not v["ran"]:
                continue
            if v["server_units"] is None:
                missing.append(f"block {b}: {v['server_reason']}")
            else:
                units += v["server_units"]
        if missing or not units:
            return {"verdict": None, "base_level": base, "target_level": target,
                    "reason": "server-side stage timing unavailable: " + ("; ".join(missing) or "no units")}
        tables[lv] = stage_table(units)
    inc = {}
    for stage, st in tables[target]["stages"].items():
        b0 = tables[base]["stages"].get(stage, {}).get("queue_plus_service_ms", {}).get("p50")
        t1 = st.get("queue_plus_service_ms", {}).get("p50")
        inc[stage] = t1 - b0 if t1 is not None and b0 is not None else None
    valid = {k: v for k, v in inc.items() if v is not None}
    if not valid:
        return {"verdict": None, "base_level": base, "target_level": target, "reason": "no stage observed at both levels"}
    stage = max(valid, key=valid.get)
    cpu = [x for v in raw[target].values() if v["ran"] for x in v["cpu_raw"].get(stage, [])]
    cpu_p90 = dist(cpu)["p90"]
    pos = stage_gpu_position(args, stage)
    gpu_utils, mapping, power, suspect = [], [], [], []
    for v in raw[target].values():
        if not v["ran"] or not v["gpu"]:
            continue
        idx = sorted(v["gpu"], key=lambda g: int(g) if g.isdigit() else g)
        if pos < len(idx):
            w = v["gpu"][idx[pos]]
            gpu_utils.append(w["util_mean"])
            mapping.append(idx[pos])
            power.append(w.get("power_mean_w"))
            suspect.append(bool(w.get("util_suspect")))
    gpu_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else None
    if gpu_util is not None and gpu_util >= GPU_BOUND:
        resource = "GPU-bound"
    elif cpu_p90 is not None and cpu_p90 >= CPU_BOUND_P90 and (gpu_util is None or gpu_util < GPU_NOT_BUSY):
        resource = "CPU-bound"
    else:
        resource = "neither saturated (uncertain)"
    # Descriptive robustness view (not the registered rule): the same increase over speaking units only, so a change
    # in the speaking share between the two levels cannot move a stage's median.
    speaking = {}
    for lv in (base, target):
        units = [u for v in raw[lv].values() if v["ran"] and v["server_units"] for u in v["server_units"]]
        sessions = [s for v in raw[lv].values() if v["ran"] for s in v["sessions_metrics"]]
        sel, _ = speaking_selector(units, sessions)
        speaking[lv] = stage_table(units, sel)
    inc_speak = {}
    for name, st in speaking[target]["stages"].items():
        b0 = speaking[base]["stages"].get(name, {}).get("queue_plus_service_ms", {}).get("p50")
        t1 = st.get("queue_plus_service_ms", {}).get("p50")
        inc_speak[name] = t1 - b0 if t1 is not None and b0 is not None else None
    return {"verdict": stage, "base_level": base, "target_level": target,
            "target_reason": "first level with pooled unit miss > 1 %" if over else "top level (no level missed > 1 %)",
            "increase_queue_plus_service_p50_ms": inc, "resource": resource, "stage_process_cpu_p90": cpu_p90,
            "stage_gpu_position": pos, "stage_gpu_index_per_block": mapping, "stage_gpu_util_mean": gpu_util,
            "stage_gpu_power_mean_w": power, "gpu_util_suspect": any(suspect),
            "resource_input_note": ("nvidia-smi utilisation of this GPU is flagged util_suspect in a target-level block, so "
                                    "the registered GPU test has unreliable input") if any(suspect) else None,
            "descriptive_increase_speaking_units_only_ms": inc_speak,
            "single_gpu_note": "all stages share one GPU; its utilisation is the whole card's" if "--thinker.gpu" not in args else None,
            "base_table": tables[base], "target_table": tables[target]}


def model_behaviour(raw_level: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Item 16 at the lowest level: share of deadline misses that are no_response, or k >= 1 with (a) + (c) <= D."""
    turns = [r for v in raw_level.values() if v["ran"] for s in v["sessions_metrics"] for r in s["turn_rows"]
             if r.get("status") != "not_sent"]
    out = {}
    for d in DEADLINES_S:
        lim = d * 1000.0
        missed = [r for r in turns if r["status"] != "answered" or r["ttfa_ms"] > lim]
        model = [r for r in missed if r["status"] == "no_response"
                 or (r.get("parts") and not r["parts"]["early_start"] and r["parts"]["k"] >= 1
                     and r["parts"]["a_wait_unit_end_ms"] + r["parts"]["c_service_ms"] <= lim)]
        share = len(model) / len(missed) if missed else None
        out[f"{d:g}s"] = {"turns": len(turns), "missed": len(missed), "model_driven": len(model), "share": share,
                          "statement": "主要是模型的轮次决策，不是服务容量" if share is not None and share > 0.5 else None}
    return out


def smoke_report(exp: Path) -> dict[str, Any]:
    res = json.loads((exp / "result.json").read_text()) if (exp / "result.json").exists() else {}
    summary = phase_summaries(exp).get(SMOKE_PHASE)
    d = exp / "raw" / f"duplex_{SMOKE_PHASE}"
    started = bool(res.get("server_ready_s"))
    out: dict[str, Any] = {"run": exp.name, "status": res.get("status"), "server_started": started}
    if summary and (d / "scripts.json").exists():
        p = pool(duplex_phase_metrics(d)["sessions"])
        out.update(sessions=p["sessions"], sessions_failed=p["sessions_failed"], fatal_errors=p["fatal_errors"],
                   unit_miss_rate=p["unit_miss_rate"])
    passed = bool(started and out.get("sessions") == 8 and out.get("sessions_failed") == 0 and out.get("fatal_errors") == 0)
    out["passed"] = passed
    if not passed:
        log = exp / "logs" / "server_s0.log"
        tail = ""
        if log.exists():
            with open(log, "rb") as fh:
                fh.seek(max(0, log.stat().st_size - 200_000))
                tail = fh.read().decode(errors="ignore")
        out["failure"] = classify_failure(f"{res.get('error', '')}\n{tail}")
    return out


def placement(smokes: dict[str, dict[str, Any]], arms_run: list[str]) -> dict[str, Any]:
    one, two = smokes.get("ms8_1gpu"), smokes.get("ms8_2gpu")
    if one is None and two is None:
        expected = None
        why = "no smoke run found"
    elif one and one["passed"]:
        expected, why = ["ms8_1gpu"], "max_sessions 8 started on one GPU and 8 sessions x 30 s had no failure"
    elif two and two["passed"]:
        expected, why = ["ms4_1gpu", "ms8_2gpu"], "1-GPU max_sessions 8 failed its smoke; the 2-GPU split passed"
    else:
        expected, why = ["ms4_1gpu"], "both smokes failed or are missing: 8 sessions are Blocked"
    return {"expected_arms": expected, "why": why, "arms_run": arms_run,
            "consistent": None if expected is None else sorted(arms_run) == sorted(expected)}


def arm_report(arm: str, mains: dict[int, Path]) -> dict[str, Any]:
    """One placement arm over its blocks (block -> latest run): levels, capacities, first limiting stage."""
    _, _, levels = R2_ARMS[arm]
    raw: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    args: list[str] = []
    for block in range(1, BLOCKS + 1):
        exp = mains.get(block)
        if exp is None:
            for lv in levels:
                raw[lv][block] = {"ran": False, "skipped": False, "reason": "no ok run for this block"}
            continue
        args = args or server_args(exp)
        gpu = load_gpu_samples_utc(exp)
        cpu = read_jsonl(exp / "raw" / "cpu_samples.jsonl")
        pids = pid_stages([exp / "raw" / "server_events"])
        ooms = oom_events(exp)
        for lv in levels:
            raw[lv][block] = level_block(exp, lv, gpu, cpu, pids, ooms)
    entries = [level_entry(lv, raw[lv], BLOCKS) for lv in levels]
    serving = capacity(entries, lambda e: e["meets_serving_target"])
    deadline = {f"{d:g}s": capacity(entries, lambda e, k=f"{d:g}s": e["meets_deadline_target"][k]) for d in DEADLINES_S}
    lowest = min(levels)
    return {"levels": entries, "serving_capacity": serving, "deadline_capacity": deadline,
            "stop_rule_skips": {str(e["level"]): e["skipped_in_blocks"] for e in entries if e["skipped_in_blocks"]},
            "first_limiting_stage": first_limiting_stage(entries, raw, args),
            "model_behaviour_at_lowest_level": {"level": lowest, **model_behaviour(raw[lowest])},
            "first_reply_after_start": first_reply(raw, levels),
            "server_args": args}


R2B_RE = re.compile(r"^ppv_r2b_(ms8_2gpu)_block(\d+)(?:_a\d+)?$")


def verdict_r2b(entries: list[dict[str, Any]], level: int = 8) -> dict[str, Any]:
    """preregistration.md R2b. 支持: the 8-session level meets R2's serving target (all 3 blocks ran; no failed session,
    503 or pacing violation; pooled unit miss <= 1 %) and no block logged a CUDA OOM in it. 反驳: in every block the
    level was skipped by the stop rule or had a failed session or an OOM. 数据不足: a block's run does not exist.
    Otherwise 不确定."""
    e = next((x for x in entries if x["level"] == level), None)
    if e is None:
        return {"verdict": "数据不足", "reason": f"no level {level}"}
    blocks = e["per_block"]
    missing = [k for k, b in blocks.items() if not b.get("ran") and b.get("reason") == "no ok run for this block"]
    ooms = {k: b["oom_events"] for k, b in blocks.items() if b.get("ran")}
    bad = {k: (not b.get("ran")) or b["sessions_failed"] > 0 or b["oom_events"] > 0 for k, b in blocks.items()}
    if missing:
        verdict = "数据不足"
    elif e["meets_serving_target"] and len(ooms) == BLOCKS and not any(ooms.values()):
        verdict = "支持"
    elif len(bad) == BLOCKS and all(bad.values()):
        verdict = "反驳"
    else:
        verdict = "不确定"
    return {"verdict": verdict, "level": level, "meets_serving_target": e["meets_serving_target"],
            "unit_miss_rate": e["unit_miss_rate"], "oom_events_by_block": ooms, "blocks_ran": e["blocks_ran"],
            "skipped_in_blocks": e["skipped_in_blocks"], "missing_blocks": missing}


def build_r2b(exps: list[Path]) -> dict[str, Any]:
    chosen, excluded = choose_runs(exps, R2B_RE, lambda m: ("main", m.group(1), int(m.group(2))), require_ok=False)
    mains = {b: p for (_, _, b), p in chosen.items()}
    rep = arm_report("ms8_2gpu", mains)
    return {"generated_by": "scripts/analysis/ppv_r2_report.py --r2b", "rules": "preregistration.md R2b (+ R2 items 9-16)",
            "runs": {f"block{b}": p.name for b, p in sorted(mains.items())},
            "run_status": {f"block{b}": run_status(p) for b, p in sorted(mains.items())}, "excluded_runs": excluded,
            "placement": "ms8_2gpu (R2b)", "arms": {"ms8_2gpu": rep}, "verdict_h_r2b": verdict_r2b(rep["levels"])}


def build(exps: list[Path]) -> dict[str, Any]:
    # Item 9b: the latest run of each block (or smoke) whatever its status; each level is judged on the data it has.
    chosen, excluded = choose_runs(exps, RUN_RE, lambda m: (m.group(1) or "main", m.group(2), int(m.group(3) or 0)),
                                   require_ok=False)
    smokes = {arm: smoke_report(p) for (kind, arm, _), p in chosen.items() if kind == "smoke"}
    arms = sorted({arm for (kind, arm, _) in chosen if kind == "main"})
    statuses = {f"{k[0]}_{k[1]}_block{k[2]}": run_status(p) for k, p in chosen.items()}
    out: dict[str, Any] = {"generated_by": "scripts/analysis/ppv_r2_report.py", "run_status": statuses,
                           "rules": "preregistration.md R2 + R1/R2 分析口径补充 items 9-16",
                           "runs": {f"{k[0]}_{k[1]}_block{k[2]}": p.name for k, p in sorted(chosen.items(), key=lambda kv: str(kv[0]))},
                           "excluded_runs": excluded, "smoke": smokes, "placement": placement(smokes, arms), "arms": {}}
    for arm in arms:
        out["arms"][arm] = arm_report(arm, {b: p for (kind, a, b), p in chosen.items() if kind == "main" and a == arm})
    out["cross_placement"] = cross_placement(out["arms"])
    return out


def first_reply(raw: dict[int, dict[int, dict[str, Any]]], levels: list[int]) -> dict[str, Any]:
    """Descriptive: the first answered turn of the first level of each block -- the first reply after the server
    started, which pays any one-time cost the start-up warm-up did not exercise -- and the worst unit lag of that
    session's first 30 units."""
    out = {}
    first = min(levels)
    for b, v in sorted(raw[first].items()):
        if not v["ran"] or not v["sessions_metrics"]:
            continue
        s0 = v["sessions_metrics"][0]
        ans = [r for r in s0["turn_rows"] if r.get("status") == "answered"]
        later = [r["ttfa_ms"] for r in ans[1:]]
        out[str(b)] = {"first_answer_ttfa_ms": ans[0]["ttfa_ms"] if ans else None,
                       "later_answers_ttfa_p50_ms": dist(later)["p50"],
                       "first_answer_parts": ans[0].get("parts") if ans else None}
    return out


def cross_placement(arms: dict[str, Any]) -> list[dict[str, Any]]:
    """Descriptive only (item 12): the same level in two placements."""
    by: dict[int, dict[str, Any]] = defaultdict(dict)
    for arm, a in arms.items():
        for e in a["levels"]:
            if e.get("pooled"):
                by[e["level"]][arm] = {"ttfa_p50_ms": e["pooled"]["ttfa_ms"]["p50"], "unit_lag_p50_ms": e["pooled"]["unit_lag_ms"]["p50"],
                                       "unit_miss_rate": e["unit_miss_rate"]}
    return [{"level": lv, **v} for lv, v in sorted(by.items()) if len(v) > 1]


def markdown(r: dict[str, Any]) -> str:
    def f(x: Any, nd: int = 0) -> str:
        return "N/A" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    lines = ["# R2 summary (generated)", "", f"Placement: {r['placement']}", ""]
    if "verdict_h_r2b" in r:
        lines += [f"R2b verdict: {r['verdict_h_r2b']}", ""]
    for arm, a in r["arms"].items():
        cap = (f"Serving capacity: {a['serving_capacity']['statement']}; deadline capacity (1 s): "
               f"{a['deadline_capacity']['1s']['statement']}")
        lines += [f"## {arm}", "", cap, "",
                  "| level | blocks | miss rate | TTFA P50/P95/P99 ms (N) | unit lag P50/P99 ms (N) | deadline miss 1 s | target |",
                  "|---|---|---|---|---|---|---|"]
        for e in a["levels"]:
            p = e.get("pooled") or {"ttfa_ms": {"p50": None, "p95": None, "p99": None, "n": 0},
                                    "unit_lag_ms": {"p50": None, "p99": None, "n": 0}}
            lines.append(f"| {e['level']} | {len(e['blocks_ran'])}/{e['blocks_expected']} | {f(e['unit_miss_rate'], 4)} | "
                         f"{f(p['ttfa_ms']['p50'])}/{f(p['ttfa_ms']['p95'])}/{f(p['ttfa_ms']['p99'])} ({p['ttfa_ms']['n']}) | "
                         f"{f(p['unit_lag_ms']['p50'])}/{f(p['unit_lag_ms']['p99'])} ({p['unit_lag_ms']['n']}) | "
                         f"{f(e['deadline_miss_all_turns']['1s'], 3)} | {'yes' if e['meets_serving_target'] else 'no'} |")
        fl = a["first_limiting_stage"]
        lines += ["", f"First limiting stage: {fl.get('verdict')} ({fl.get('resource', fl.get('reason'))})", ""]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--exp", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--md")
    ap.add_argument("--r2b", action="store_true", help="analyse the R2b runs (ppv_r2b_ms8_2gpu_block*) instead of R2")
    a = ap.parse_args()
    if a.data_root is None and not a.exp:
        ap.error("give --data-root or --exp")
    rep = (build_r2b if a.r2b else build)(discover_runs(a.data_root, a.exp))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    if a.md:
        Path(a.md).write_text(markdown(rep))
    print(markdown(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
