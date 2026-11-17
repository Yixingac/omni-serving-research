#!/usr/bin/env python3
"""实验 7（code2wav 单独占卡）的汇总和判定。

    python scripts/analysis/exp07_report.py --data-root <results root> --run 20261117_exp07_t2_k6_shared_block1_run01 \
        --run ... --out research/01_性能瓶颈/实验07_Code2Wav独立GPU

Per run: the closed loop per concurrency and the open loop per rate (client metrics, code2wav breakdown, talker
processing per audio second and queueing, batch-1 flow GPU time per frame, GPU busy and memory by role), and the
3 s kernel trace at concurrency 128 (overlap of talker and code2wav kernels on each card). Then the verdicts of
hypotheses.md with the amendment of 17:07. Unit of replication: the run; contrasts are paired within a block.
Writes results/summary.json and results/levels.csv.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts" / "slurm"))

import exp06_c2w_breakdown as bd  # noqa: E402

BASELINE = "t2_k6_shared"
ISOLATED = ["t2_k4_c2", "t2_k3_c3", "t2_k2_c4"]
MECHANISM = ("t2_k3_shared", "t2_k3_c3")
LEVELS = [8, 64, 128, 256]
OPEN_PHASES = ("w3b_open", "w3b_open_hi")
RUN_RE = re.compile(r"^exp07_(t2_k\d_(?:shared|c\d))_block(\d+)$")
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
H7A_METRICS = ("c2w_flow_us_per_frame_b1", "talker_gen_ms_per_audio_s")


def interval(values: list[Any]) -> dict[str, Any]:
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    out: dict[str, Any] = {"runs": values, "n": len(v), "mean": float(np.mean(v)) if v else None, "ci95": None}
    if len(v) >= 2:
        half = T975.get(len(v) - 1, 2.0) * float(np.std(v, ddof=1)) / math.sqrt(len(v))
        out["ci95"] = [out["mean"] - half, out["mean"] + half]
    return out


def talker_times(events: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per request in the talker: queue (queue enter -> prefill start) and generation (prefill start -> model path end)."""
    first: dict[tuple[str, str], int] = {}
    for e in events:
        name = e.get("event_name")
        if not (e.get("stage") or "").startswith("talker") or name not in ("scheduler_queue_enter", "scheduler_prefill_start", "model_path_end"):
            continue
        key = (e["request_id"], name)
        ts = int(e["timestamp_ns"])
        first[key] = max(first.get(key, ts), ts) if name == "model_path_end" else min(first.get(key, ts), ts)
    out = {}
    for rid in {k[0] for k in first}:
        q, p, m = (first.get((rid, n)) for n in ("scheduler_queue_enter", "scheduler_prefill_start", "model_path_end"))
        if q is not None and p is not None and m is not None:
            out[rid] = {"queue_ms": (p - q) / 1e6, "gen_ms": (m - p) / 1e6}
    return out


def batches_of(batches: dict[tuple, dict[str, Any]], ids: set[str]) -> dict[tuple, dict[str, Any]]:
    """Batches most of whose requests belong to ids (levels run one after another, so a batch belongs to one)."""
    return {k: b for k, b in batches.items() if 2 * sum(r in ids for r in b.get("request_ids", [])) > len(b.get("request_ids", []))}


def flow_us_per_frame_b1(batches: dict[tuple, dict[str, Any]]) -> dict[str, Any]:
    v = [b["gpu_flow_ms"] * 1000 / b["flow_solve_frames"] for b in batches.values()
         if b.get("size") == 1 and b.get("gpu_flow_ms") is not None and b.get("flow_solve_frames")]
    return {"n": len(v), "p50": float(np.median(v)) if v else None}


def union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def overlap_ms(first: list[tuple[float, float]], second: list[tuple[float, float]], scale: float = 1e-3) -> float:
    """Time when both sets have a kernel running (inputs in microseconds by default, result in milliseconds)."""
    u, w = union(first), union(second)
    i = j = 0
    total = 0.0
    while i < len(u) and j < len(w):
        lo, hi = max(u[i][0], w[j][0]), min(u[i][1], w[j][1])
        total += max(0.0, hi - lo)
        if u[i][1] < w[j][1]:
            i += 1
        else:
            j += 1
    return total * scale


def usable(row: dict[str, Any] | None) -> bool:
    c = (row or {}).get("client") or {}
    return bool(c.get("measured")) and (c.get("errors", 0) + c.get("timeouts", 0)) <= 0.05 * c["measured"]


def get(d: dict[str, Any] | None, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def paired(runs: list[dict[str, Any]], arm: str, base: str, section: str, key: str, path: str) -> dict[str, Any]:
    """Per block: (arm value, base value, relative difference), with blocks where either level is unusable listed."""
    by = {(r["arm"], r["block"]): r for r in runs}
    blocks = sorted({r["block"] for r in runs})
    rel, unusable, a_vals, b_vals = [], [], [], []
    for blk in blocks:
        ra, rb = (by.get((x, blk), {}).get(section, {}).get(key) for x in (arm, base))
        if not (usable(ra) and usable(rb)):
            unusable.append(blk)
            rel.append(None)
            a_vals.append(None)
            b_vals.append(None)
            continue
        a, b = get(ra, path), get(rb, path)
        a_vals.append(a)
        b_vals.append(b)
        rel.append((a - b) / b if a is not None and b else None)
    return {"relative": interval(rel), "arm": interval(a_vals), "base": interval(b_vals), "unusable_blocks": unusable}


def verdicts(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    shared, isolated = MECHANISM
    by_metric = {}
    for metric in H7A_METRICS:
        for c in (128, 256):
            p = paired(runs, isolated, shared, "closed", str(c), f"{metric}.p50")
            rel = p["relative"]["runs"]
            if p["unusable_blocks"] or not rel or any(x is None for x in rel):
                v = "数据不足"
            elif all(x < -0.10 for x in rel):
                v = "支持"
            elif abs(float(np.mean(rel))) <= 0.05 or not (all(x < 0 for x in rel) or all(x > 0 for x in rel)):
                v = "反驳"
            else:
                v = "不确定"
            by_metric[f"{metric}@{c}"] = {"verdict": v, **p}
    verdict_list = [x["verdict"] for x in by_metric.values()]
    out["H7a"] = {"verdict": "支持" if all(v == "支持" for v in verdict_list) else "反驳" if "反驳" in verdict_list
                  else "数据不足" if "数据不足" in verdict_list else "不确定", "by_metric": by_metric}

    by_arm = {}
    for arm in ISOLATED:
        tp = paired(runs, arm, BASELINE, "closed", "256", "client.throughput_rps")
        p95 = paired(runs, arm, BASELINE, "open", "9.5", "client.e2e_s.p95")
        tp_rel, p95_rel = tp["relative"]["runs"], p95["relative"]["runs"]
        complete = not tp["unusable_blocks"] and not p95["unusable_blocks"] and all(x is not None for x in tp_rel + p95_rel) and tp_rel
        meets = bool(complete) and all(x > 0.10 for x in tp_rel) and all(x < 0 for x in p95_rel)
        faster_everywhere = bool(complete) and all(x > 0 for x in tp_rel)
        by_arm[arm] = {"meets": meets, "faster_in_all_blocks": faster_everywhere, "complete": bool(complete),
                       "throughput_256": tp, "p95_at_9.5": p95}
    if any(a["meets"] for a in by_arm.values()):
        v = "支持"
    elif all(a["complete"] and not a["faster_in_all_blocks"] for a in by_arm.values()):
        v = "反驳"
    elif not all(a["complete"] for a in by_arm.values()):
        v = "数据不足"
    else:
        v = "不确定"
    out["H7b"] = {"verdict": v, "by_arm": by_arm}

    h7c = {}
    for arm in ISOLATED:
        h7c[arm] = {"throughput_8": paired(runs, arm, BASELINE, "closed", "8", "client.throughput_rps"),
                    "e2e_p50_at_5.5": paired(runs, arm, BASELINE, "open", "5.5", "client.e2e_s.p50"),
                    "e2e_p50_at_9.5": paired(runs, arm, BASELINE, "open", "9.5", "client.e2e_s.p50")}
    small = [abs(x["throughput_8"]["relative"]["mean"]) < 0.05 for x in h7c.values() if x["throughput_8"]["relative"]["mean"] is not None]
    out["H7c"] = {"low_load_small": bool(small) and all(small), "by_arm": h7c}
    return out


def gpu_roles(arm: str) -> dict[str, list[int]]:
    from make_plan_exp06_08 import PLACEMENTS

    thinker, talker, c2w = PLACEMENTS[arm]
    roles = {"thinker": thinker}
    if talker == c2w:
        roles["talker+code2wav"] = talker
    else:
        roles["talker"], roles["code2wav"] = talker, c2w
    used = set(thinker) | set(talker) | set(c2w)
    if len(used) < 8:
        roles["unused"] = [g for g in range(8) if g not in used]
    return roles


def level_row(recs: list[dict[str, Any]], reqs: dict[str, Any], batches: dict[tuple, Any], talker: dict[str, Any],
              util: dict[int, list], anchor: Any, roles: dict[str, list[int]]) -> dict[str, Any]:
    from exp06_report import client_metrics, phase_window

    ids = {r["request_id"] for r in recs if r.get("request_id")}
    mine = batches_of(batches, ids)
    records = {r["request_id"]: r for r in recs if r.get("request_id")}
    ok = [r for r in recs if not r.get("warmup") and r.get("status") == "ok"]
    gen = [talker[r["request_id"]]["gen_ms"] / r["output_audio_s"] for r in ok if r["request_id"] in talker and r.get("output_audio_s")]
    queue = [talker[r["request_id"]]["queue_ms"] for r in ok if r["request_id"] in talker]
    row = {"client": client_metrics(recs), "breakdown": bd.summarize({k: v for k, v in reqs.items() if k in ids}, mine, records),
           "c2w_flow_us_per_frame_b1": flow_us_per_frame_b1(mine), "talker_gen_ms_per_audio_s": bd.stats(gen), "talker_queue_ms": bd.stats(queue)}
    win = phase_window([r for r in recs if not r.get("warmup")], anchor)
    if win:
        row["gpu"] = {}
        for role, gpus in roles.items():
            samples = [(u, m) for g in gpus for t, u, m in util.get(g, []) if win[0] <= t <= win[1]]
            row["gpu"][role] = {"busy": float(np.mean([u for u, _ in samples])) / 100 if samples else None,
                                "mem_max_mib": float(max(m for _, m in samples)) if samples else None}
    return row


def kernel_intervals(path: Path) -> list[tuple[float, float]]:
    from exp05_step_sync import DEVICE_CATS

    with gzip.open(path, "rt") if path.suffix == ".gz" else open(path) as fh:
        tr = json.load(fh)
    base_us = int(tr.get("baseTimeNanoseconds", 0)) / 1000
    return [(base_us + e["ts"], base_us + e["ts"] + e.get("dur", 0)) for e in tr.get("traceEvents", [])
            if e.get("ph") == "X" and e.get("cat") in DEVICE_CATS]


def kernel_overlap(phase_dir: Path, arm: str) -> dict[str, Any]:
    """Per replica index i: talker@ri and code2wav@ri kernels (they share a card only in the shared placements)."""
    from make_plan_exp06_08 import PLACEMENTS

    _, talker_gpus, c2w_gpus = PLACEMENTS[arm]
    files: dict[tuple[str, int], Path] = {}
    for f in phase_dir.rglob("*.trace.json*"):
        m = re.match(r"(talker|code2wav)@r(\d+)_trace", f.name)
        if m:
            files[(m.group(1), int(m.group(2)))] = f
    per_card = []
    for i, gpu in enumerate(talker_gpus):
        t, c = files.get(("talker", i)), files.get(("code2wav", c2w_gpus.index(gpu))) if gpu in c2w_gpus else None
        if t is None:
            continue
        tk = kernel_intervals(t)
        ck = kernel_intervals(c) if c is not None else []
        both = tk + ck
        window = (max(b for _, b in both) - min(a for a, _ in both)) / 1e3 if both else None
        busy = lambda k: sum(b - a for a, b in union(k)) / 1e3  # noqa: E731
        per_card.append({"gpu": gpu, "window_ms": window, "talker_busy_ms": busy(tk), "code2wav_busy_ms": busy(ck) if ck else None,
                         "overlap_ms": overlap_ms(tk, ck) if ck else None, "code2wav_trace": c is not None})
    shares = [x["overlap_ms"] / x["window_ms"] for x in per_card if x["overlap_ms"] is not None and x["window_ms"]]
    return {"cards": per_card, "overlap_share_mean": float(np.mean(shares)) if shares else None,
            "missing_code2wav_traces": sum(1 for x in per_card if not x["code2wav_trace"] and gpu_shared(arm))}


def gpu_shared(arm: str) -> bool:
    return arm.endswith("shared")


def analyze_run(exp: Path) -> dict[str, Any]:
    from exp01_gpu_timeline import load_util
    from exp06_report import read_events
    from omni_research.analysis.experiment_analysis import clock_anchor, read_jsonl

    res = json.load(open(exp / "result.json"))
    m = RUN_RE.match(res["experiment_id"])
    arm, block = m.group(1), int(m.group(2))
    util, anchor, roles = load_util(exp), clock_anchor(exp), gpu_roles(arm)
    run: dict[str, Any] = {"run": exp.name, "arm": arm, "block": block, "restarts": res.get("server_restarts") or [],
                           "closed": {}, "open": {}, "issues": {}}
    recs = read_jsonl(exp / "raw" / "requests_w3b_closed.jsonl")
    events = read_events(exp / "raw" / "server_events" / "w3b_closed")
    reqs, batches, run["issues"]["w3b_closed"] = bd.breakdown(events)
    talker = talker_times(events)
    for c in LEVELS:
        lv = [r for r in recs if r.get("concurrency") == c]
        if lv:
            run["closed"][str(c)] = level_row(lv, reqs, batches, talker, util, anchor, roles)
    for phase in OPEN_PHASES:
        path = exp / "raw" / f"requests_{phase}.jsonl"
        if not path.exists():
            continue
        recs = read_jsonl(path)
        events = read_events(exp / "raw" / "server_events" / phase)
        reqs, batches, run["issues"][phase] = bd.breakdown(events)
        talker = talker_times(events)
        for rate in sorted({r.get("rate_rps") for r in recs if r.get("rate_rps") is not None}):
            run["open"][f"{rate:g}"] = {**level_row([r for r in recs if r.get("rate_rps") == rate], reqs, batches, talker, util, anchor, roles),
                                        "phase": phase}
    restart_phases = [x.get("where", "") for x in run["restarts"]]
    run["after_restart"] = sorted({p.split()[-1] for p in restart_phases})
    trace_dir = exp / "profiling" / "torch" / "w3b_c128_C"
    run["kernel_c128"] = kernel_overlap(trace_dir, arm) if trace_dir.exists() else None
    return run


SUMMARY_PATHS = {"closed": ["client.throughput_rps", "client.audio_s_per_s", "client.e2e_s.p50", "client.e2e_s.p95", "client.e2e_s.p99",
                            "client.ttft_s.p50", "breakdown.total_ms.p50", "breakdown.queue_ms.p50", "breakdown.compute_ms.p50",
                            "breakdown.batch_size.mean", "c2w_flow_us_per_frame_b1.p50", "talker_gen_ms_per_audio_s.p50",
                            "talker_queue_ms.p50"],
                 "open": ["client.throughput_rps", "client.e2e_s.p50", "client.e2e_s.p95", "client.e2e_s.p99", "client.ttft_s.p50",
                          "breakdown.queue_ms.p50", "breakdown.total_ms.p50", "talker_queue_ms.p50"]}


def by_arm(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = defaultdict(lambda: {"closed": {}, "open": {}})
    for arm in sorted({r["arm"] for r in runs}):
        mine = sorted((r for r in runs if r["arm"] == arm), key=lambda r: r["block"])
        for section, paths in SUMMARY_PATHS.items():
            keys = sorted({k for r in mine for k in r[section]}, key=float)
            for k in keys:
                rows = [r[section].get(k) for r in mine]
                out[arm][section][k] = {p: interval([get(x, p) if usable(x) else None for x in rows]) for p in paths}
                out[arm][section][k]["usable_blocks"] = [r["block"] for r, x in zip(mine, rows) if usable(x)]
    return dict(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    runs = [analyze_run(root / name) for name in a.run]
    out = Path(a.out) / "results"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"generated_by": "scripts/analysis/exp07_report.py", "runs": [r["run"] for r in runs], "per_run": runs,
               "by_arm": by_arm(runs), "verdicts": verdicts(runs)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float) + "\n")
    with open(out / "levels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "block", "section", "level", "throughput_rps", "e2e_p50_s", "e2e_p95_s", "e2e_p99_s", "ttft_p50_s", "errors",
                    "c2w_queue_p50_ms", "c2w_total_p50_ms", "talker_gen_ms_per_audio_s", "talker_queue_p50_ms", "flow_us_per_frame_b1"])
        for r in runs:
            for section in ("closed", "open"):
                for k, x in r[section].items():
                    w.writerow([r["arm"], r["block"], section, k] + [get(x, p) for p in (
                        "client.throughput_rps", "client.e2e_s.p50", "client.e2e_s.p95", "client.e2e_s.p99", "client.ttft_s.p50", "client.errors",
                        "breakdown.queue_ms.p50", "breakdown.total_ms.p50", "talker_gen_ms_per_audio_s.p50", "talker_queue_ms.p50",
                        "c2w_flow_us_per_frame_b1.p50")])
    print(out / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
