#!/usr/bin/env python3
"""实验 9（8 张卡的最佳分工）的汇总和判定。

    python scripts/analysis/exp09_report.py --data-root <results root> --pilot --run 20261117_exp09_pilot_t2_k1_c5_run01 ...
    python scripts/analysis/exp09_report.py --data-root <results root> --run 20261118_exp09_t2_k6_shared_block1_run01 ... \
        --out research/01_性能瓶颈/实验09_最佳GPU分工

Rules are the ones registered in research/01_性能瓶颈/实验09_最佳GPU分工/hypotheses.md:
- open-loop rates from the pilot: {0.5 Xmin, 0.8 Xmin, Xmin, (Xmin + Xmax)/2, Xmax, 1.1 Xmax}, rounded half up to 0.5;
- capacity: completed requests / (last completion - first send) at the top closed-loop level of the A phase;
- steady: least-squares slope of the client's in-flight count over the second half of the sending window
  <= 0.05 x the offered rate;
- percentiles per run, with failed and timed-out requests counted as infinitely slow;
- a stage's residence: first stage_input_received -> last stage_complete of that stage (one server clock);
  the bottleneck stage holds the most summed residence (Little's law: the most requests on average).
Unit of replication: the run; contrasts are paired within a block.
"""

from __future__ import annotations

import argparse
import csv
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

STAGES = ("thinker", "talker", "code2wav")
WORKLOADS = {"w3b": 256, "w3l": 128}  # workload -> concurrency of the capacity level
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
ARMS = {1: "t2_k6_shared", 2: "t2_k2_c4", 3: "t2_k1_c5", 4: "t2_k3_c3"}
ARM_OF = {v: k for k, v in ARMS.items()}
# Cards per stage variant; talker and code2wav share their cards in arm 1.
ARM_STAGE_CARDS = {1: {"thinker": 2, "talker_shared": 6, "code2wav_shared": 6},
                   2: {"thinker": 2, "talker_iso": 2, "code2wav_iso": 4},
                   3: {"thinker": 2, "talker_iso": 1, "code2wav_iso": 5},
                   4: {"thinker": 2, "talker_iso": 3, "code2wav_iso": 3}}
RUN_RE = re.compile(r"^exp09_(?:pilot_)?(t2_k\d_(?:shared|c\d))(?:_block(\d+))?$")


# ---------------------------------------------------------------- registered rules

def round_half(x: float) -> float:
    return math.floor(2 * x + 0.5) / 2


def common_rates(capacities: list[float]) -> list[float]:
    lo, hi = min(capacities), max(capacities)
    return sorted({round_half(x) for x in (0.5 * lo, 0.8 * lo, lo, (lo + hi) / 2, hi, 1.1 * hi)})


def queue_growth(records: list[dict[str, Any]], rate_rps: float, step_s: float = 1.0) -> dict[str, Any]:
    sent = sorted(r["sent_ns"] / 1e9 for r in records if r.get("sent_ns") is not None)
    done = sorted(r["done_ns"] / 1e9 for r in records if r.get("sent_ns") is not None and r.get("done_ns") is not None)
    if len(sent) < 2:
        return {"slope_rps": None, "steady": None, "inflight_end": None, "window_s": None}
    t0, t1 = sent[0], sent[-1]
    grid = np.arange(t0 + (t1 - t0) / 2, t1 + 1e-9, step_s)
    inflight = np.searchsorted(sent, grid, side="right") - np.searchsorted(done, grid, side="right")
    slope = float(np.polyfit(grid - grid[0], inflight, 1)[0]) if len(grid) >= 3 else None
    return {"slope_rps": slope, "steady": None if slope is None else slope <= 0.05 * rate_rps,
            "inflight_end": int(inflight[-1]) if len(inflight) else None, "window_s": t1 - t0}


def busy_window_throughput(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive (not registered): completions per second while every closed-loop slot stays busy, i.e. between
    the first and the last send of a measured request. The registered capacity divides by first send -> last
    completion, which also counts the final drain, when fewer and fewer requests are in flight."""
    meas = [r["sent_ns"] for r in records if not r.get("warmup") and r.get("sent_ns") is not None]
    if len(meas) < 2:
        return {"throughput_rps": None, "window_s": None}
    s0, s1 = min(meas), max(meas)
    done = sum(1 for r in records if r.get("status") == "ok" and r.get("done_ns") is not None and s0 < r["done_ns"] <= s1)
    window = (s1 - s0) / 1e9
    return {"throughput_rps": done / window if window > 0 else None, "window_s": window}


def inflight_series(records: list[dict[str, Any]], step_s: float = 1.0) -> list[list[float]]:
    """[seconds since the first send, requests sent and not yet finished] every step_s, until the last completion."""
    sent = sorted(r["sent_ns"] / 1e9 for r in records if r.get("sent_ns") is not None)
    done = sorted(r["done_ns"] / 1e9 for r in records if r.get("sent_ns") is not None and r.get("done_ns") is not None)
    if not sent:
        return []
    t0, t1 = sent[0], max(done[-1] if done else sent[-1], sent[-1])
    grid = np.arange(t0, t1 + step_s, step_s)
    n = np.searchsorted(sent, grid, side="right") - np.searchsorted(done, grid, side="right")
    return [[round(float(t - t0), 3), int(k)] for t, k in zip(grid, n)]


def censored_percentiles(records: list[dict[str, Any]], ps: tuple[int, ...] = (50, 95, 99)) -> dict[str, Any]:
    vals = []
    for r in records:
        ok = r.get("status") == "ok" and r.get("sent_ns") is not None and r.get("done_ns") is not None
        vals.append((r["done_ns"] - r["sent_ns"]) / 1e9 if ok else math.inf)
    vals.sort()
    out: dict[str, Any] = {"n": len(vals), "failed": sum(math.isinf(v) for v in vals)}
    for p in ps:
        if not vals:
            out[f"p{p}"] = None
            continue
        h = (len(vals) - 1) * p / 100
        lo, hi = math.floor(h), math.ceil(h)
        if lo == hi or vals[lo] == vals[hi]:
            out[f"p{p}"] = vals[lo]
        elif math.isinf(vals[hi]):
            out[f"p{p}"] = math.inf
        else:
            out[f"p{p}"] = vals[lo] + (h - lo) * (vals[hi] - vals[lo])
    return out


def stage_residence(events: list[dict[str, Any]], ids: set[str]) -> dict[str, float]:
    first: dict[tuple[str, str], int] = {}
    last: dict[tuple[str, str], int] = {}
    for e in events:
        stage = (e.get("stage") or "").split("@")[0]
        rid = e.get("request_id")
        if stage not in STAGES or rid not in ids:
            continue
        key, ts = (rid, stage), int(e["timestamp_ns"])
        if e.get("event_name") == "stage_input_received":
            first[key] = min(first.get(key, ts), ts)
        elif e.get("event_name") == "stage_complete":
            last[key] = max(last.get(key, ts), ts)
    out = {s: 0.0 for s in STAGES}
    for key, t in first.items():
        if key in last and last[key] >= t:
            out[key[1]] += (last[key] - t) / 1e6
    return out


def bottleneck(residence_ms: dict[str, float]) -> dict[str, Any]:
    total = sum(residence_ms.values())
    if not total:
        return {"stage": None, "shares": {}}
    shares = {s: v / total for s, v in residence_ms.items()}
    return {"stage": max(shares, key=shares.get), "shares": shares}


# ---------------------------------------------------------------- verdicts

def usable(client: dict[str, Any]) -> bool:
    """Registered exclusion: a closed-loop level with errors + timeouts above 5 % of the measured requests is unusable."""
    bad = (client.get("errors") or 0) + (client.get("timeouts") or 0)
    return not (client.get("measured") and bad / client["measured"] > 0.05)


def workload_entry(levels: list[dict[str, Any]], w: str, top: int) -> dict[str, Any] | None:
    """Capacity (unprofiled phase, or the events phase in the pilot) and bottleneck (events phase) at the top closed-loop
    level. A level that is unusable, or in the phase where the server restarted, gives a missing value (registered)."""
    cap = [x for x in levels if x["workload"] == w and x["kind"] == "closed_loop" and x.get("concurrency") == top]
    a = [x for x in cap if x["mode"] == "A"] or [x for x in cap if x["mode"] == "B"]
    b = [x for x in cap if x["mode"] == "B"]
    if not a:
        return None
    excluded = []
    for row, what in ((a[0], "capacity"), (b[0] if b else None, "bottleneck")):
        if row is None:
            continue
        if row.get("usable") is False:
            excluded.append(f"{what}: {row['mode']} errors + timeouts > 5 %")
        elif row.get("restart_here"):
            excluded.append(f"{what}: {row['mode']} server restarted in this phase")
    cap_ok = not any(e.startswith("capacity") for e in excluded)
    bn_ok = b and not any(e.startswith("bottleneck") for e in excluded) and b[0].get("bottleneck")
    return {"capacity_rps": a[0]["client"].get("throughput_rps") if cap_ok else None,
            "audio_s_per_s": a[0]["client"].get("audio_s_per_s") if cap_ok else None, "capacity_from": a[0]["mode"],
            "bottleneck": b[0]["bottleneck"]["stage"] if bn_ok else None, "excluded": excluded}


def _caps(runs: list[dict[str, Any]], w: str) -> dict[tuple[int, int], float]:
    return {(r["arm"], r["block"]): r["workload"][w]["capacity_rps"] for r in runs
            if w in r["workload"] and r["workload"][w].get("capacity_rps") is not None}


def interval(values: list[Any]) -> dict[str, Any]:
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    out: dict[str, Any] = {"runs": values, "n": len(v), "mean": float(np.mean(v)) if v else None, "ci95": None}
    if len(v) >= 2:
        half = T975.get(len(v) - 1, 2.0) * float(np.std(v, ddof=1)) / math.sqrt(len(v))
        out["ci95"] = [out["mean"] - half, out["mean"] + half]
    return out


def paired(runs: list[dict[str, Any]], w: str, arm: int, base: int) -> dict[str, Any]:
    caps = _caps(runs, w)
    blocks = sorted({b for (_, b) in caps})
    rel = [caps[(arm, b)] / caps[(base, b)] - 1 if (arm, b) in caps and (base, b) in caps else None for b in blocks]
    return {"blocks": blocks, "relative": interval(rel)}


def verdict_h9a(runs: list[dict[str, Any]], w: str = "w3b") -> dict[str, Any]:
    caps = _caps(runs, w)
    blocks = sorted({b for (a, b) in caps if a == 3})
    complete = blocks and all((a, b) in caps for b in blocks for a in (1, 2, 3))
    vs2, vs1 = paired(runs, w, 3, 2), paired(runs, w, 3, 1)
    if not complete:
        v = "数据不足"
    elif all(caps[(3, b)] > caps[(1, b)] and caps[(3, b)] > caps[(2, b)] for b in blocks) and vs2["relative"]["mean"] >= 0.05:
        v = "支持"
    elif all(caps[(3, b)] < caps[(2, b)] for b in blocks):
        v = "反驳"
    else:
        v = "不确定"
    return {"verdict": v, "arm3_vs_arm2": vs2, "arm3_vs_arm1": vs1}


def verdict_h9b(runs: list[dict[str, Any]], w: str = "w3b") -> dict[str, Any]:
    b = [r["workload"][w].get("bottleneck") for r in runs if r["arm"] == 3 and w in r["workload"]]
    if not b or any(x is None for x in b):
        v = "数据不足"
    elif all(x == "talker" for x in b):
        v = "支持"
    elif all(x != "talker" for x in b):
        v = "反驳"
    else:
        v = "不确定"
    return {"verdict": v, "arm3_bottlenecks": b}


def verdict_h9c(runs: list[dict[str, Any]]) -> dict[str, Any]:
    means = {}
    for w in WORKLOADS:
        caps = _caps(runs, w)
        means[w] = {a: float(np.mean([c for (aa, _), c in caps.items() if aa == a])) for a in sorted({a for a, _ in caps})}
    if not means["w3b"] or not means["w3l"]:
        return {"verdict": "数据不足", "means": means}
    win = {w: max(m, key=m.get) for w, m in means.items()}
    if win["w3b"] == win["w3l"]:
        v = "不支持"
    else:
        caps = _caps(runs, "w3l")
        blocks = sorted({b for (_, b) in caps})
        v = "支持" if all(caps.get((win["w3l"], b), -1) > caps.get((win["w3b"], b), math.inf) for b in blocks) else "不确定"
    return {"verdict": v, "means": means, "winners": win}


def stage_variant(arm: int, stage: str) -> str:
    if stage == "thinker":
        return "thinker"
    return f"{stage}_shared" if f"{stage}_shared" in ARM_STAGE_CARDS[arm] else f"{stage}_iso"


def capacity_model(runs: list[dict[str, Any]], w: str) -> dict[str, Any]:
    """Leave one placement out: per-card capacity of the bottleneck stage variant from the other arms' runs."""
    per_arm = defaultdict(list)
    for r in runs:
        x = r["workload"].get(w) or {}
        if x.get("capacity_rps") is not None:
            per_arm[r["arm"]].append(r)
    preds = {}
    for arm in sorted(ARM_STAGE_CARDS):
        est = defaultdict(list)
        for other, rs in per_arm.items():
            if other == arm:
                continue
            for r in rs:
                b = r["workload"][w].get("bottleneck")
                if b:
                    v = stage_variant(other, b)
                    est[v].append(r["workload"][w]["capacity_rps"] / ARM_STAGE_CARDS[other][v])
        options = {v: cards * float(np.mean(est[v])) for v, cards in ARM_STAGE_CARDS[arm].items() if est.get(v)}
        measured = float(np.mean([r["workload"][w]["capacity_rps"] for r in per_arm[arm]])) if per_arm.get(arm) else None
        if options and measured:
            lim = min(options, key=options.get)
            preds[arm] = {"predicted_rps": options[lim], "limited_by": lim, "options": options, "measured_rps": measured,
                          "error": options[lim] / measured - 1}
        else:
            preds[arm] = {"predicted_rps": None, "limited_by": None, "options": options, "measured_rps": measured, "error": None}
    errs = [abs(p["error"]) for p in preds.values() if p["error"] is not None]
    v = "数据不足" if not errs else "支持" if all(e <= 0.10 for e in errs) else "不支持" if any(e > 0.20 for e in errs) else "不确定"
    return {"predictions": preds, "verdict": v}


# ---------------------------------------------------------------- reading runs

def gpu_roles(arm_name: str) -> dict[str, list[int]]:
    from make_plan_exp09_11 import PLACEMENTS

    thinker, talker, c2w = PLACEMENTS[arm_name]
    if talker == c2w:
        return {"thinker": thinker, "talker+code2wav": talker}
    return {"thinker": thinker, "talker": talker, "code2wav": c2w}


def gpu_window(util: dict, clocks: dict, roles: dict[str, list[int]], win: tuple[float, float] | None) -> dict[str, Any]:
    if not win:
        return {}
    out = {}
    for role, gpus in roles.items():
        s = [(u, m) for g in gpus for t, u, m in util.get(g, []) if win[0] <= t <= win[1]]
        c = [(k, p) for g in gpus for t, k, p in clocks.get(g, []) if win[0] <= t <= win[1]]
        out[role] = {"busy": float(np.mean([u for u, _ in s])) / 100 if s else None, "mem_max_mib": float(max(m for _, m in s)) if s else None,
                     "sm_clock_mhz": float(np.mean([k for k, _ in c])) if c else None, "power_w": float(np.mean([p for _, p in c])) if c else None}
    return out


def level_client(recs: list[dict[str, Any]]) -> dict[str, Any]:
    from exp06_report import client_metrics

    meas = [r for r in recs if not r.get("warmup")]
    out = {"client": client_metrics(recs), "censored": censored_percentiles(meas), "busy_window": busy_window_throughput(recs)}
    toks = [r.get("completion_tokens") for r in meas if r.get("status") == "ok" and r.get("completion_tokens")]
    audio = [r.get("output_audio_s") for r in meas if r.get("status") == "ok" and r.get("output_audio_s")]
    late = [r.get("lateness_s") for r in meas if r.get("lateness_s") is not None]
    out["output"] = {"tokens_mean": float(np.mean(toks)) if toks else None, "tokens_p95": float(np.percentile(toks, 95)) if toks else None,
                     "audio_s_mean": float(np.mean(audio)) if audio else None, "audio_s_p95": float(np.percentile(audio, 95)) if audio else None,
                     "at_max_tokens": None}
    if late:
        out["lateness_p99_s"] = float(np.percentile(late, 99))
    return out


def analyze_run(exp: Path) -> dict[str, Any]:
    import exp06_c2w_breakdown as bd
    from exp01_gpu_timeline import load_util
    from exp06_report import phase_window, read_events
    from exp07_report import load_clocks, talker_times
    from omni_research.analysis.experiment_analysis import clock_anchor, read_jsonl

    res = json.load(open(exp / "result.json"))
    m = RUN_RE.match(res["experiment_id"])
    name = m.group(1)
    run: dict[str, Any] = {"run": exp.name, "placement": name, "arm": ARM_OF[name], "block": int(m.group(2)) if m.group(2) else 0,
                           "restarts": res.get("server_restarts") or [], "workload": {}, "levels": []}
    util, clocks, anchor, roles = load_util(exp), load_clocks(exp), clock_anchor(exp), gpu_roles(name)
    restart_phases = {x.get("where", "").split()[-1] for x in run["restarts"] if x.get("where")}
    after_restart = False
    for ph in res.get("phases", []):
        phase = ph["phase"]
        restart_here = phase in restart_phases
        after_restart = after_restart or restart_here
        path = exp / "raw" / f"requests_{phase}.jsonl"
        if phase.startswith("smoke") or not path.exists():
            continue
        w = phase.split("_")[0]
        mode = "B" if ph.get("profiled") == "request_events" or phase.endswith("_B") or phase.endswith("_pilot") else "A"
        recs = read_jsonl(path)
        events = read_events(exp / "raw" / "server_events" / phase) if mode == "B" and (exp / "raw" / "server_events" / phase).exists() else None
        if events is not None:
            reqs, batches, _ = bd.breakdown(events)
            talker = talker_times(events)
        key = "concurrency" if ph.get("kind") == "closed_loop" else "rate_rps"
        for level in sorted({r.get(key) for r in recs if r.get(key) is not None}):
            lv = [r for r in recs if r.get(key) == level]
            meas = [r for r in lv if not r.get("warmup")]
            row = {"workload": w, "phase": phase, "mode": mode, "kind": ph.get("kind"), key: level, **level_client(lv),
                   "gpu": gpu_window(util, clocks, roles, phase_window(meas, anchor))}
            row.update(usable=usable(row["client"]), restart_here=restart_here, after_restart=after_restart)
            if key == "rate_rps":
                row["growth"] = queue_growth(meas, float(level))
                row["inflight_series"] = inflight_series(meas)
            if events is not None:
                ids = {r["request_id"] for r in meas if r.get("request_id")}
                resid = stage_residence(events, ids)
                wall = row["client"].get("throughput_rps") and len(ids) / row["client"]["throughput_rps"]
                row["residence_ms"] = resid
                row["in_stage_mean"] = {s: v / 1000 / wall for s, v in resid.items()} if wall else None
                row["bottleneck"] = bottleneck(resid)
                mine = {k: b for k, b in batches.items() if 2 * sum(x in ids for x in b.get("request_ids", [])) > len(b.get("request_ids", []))}
                row["code2wav"] = {"batch_size": bd.stats([b.get("size") for b in mine.values()]),
                                   "queue_ms": bd.stats([reqs[i]["queue_ms"] for i in ids if i in reqs and reqs[i].get("queue_ms") is not None])}
                gen = [talker[r["request_id"]]["gen_ms"] / r["output_audio_s"] for r in meas
                       if r.get("request_id") in talker and r.get("status") == "ok" and r.get("output_audio_s")]
                row["talker_gen_ms_per_audio_s"] = bd.stats(gen)
                row["talker_queue_ms"] = bd.stats([talker[i]["queue_ms"] for i in ids if i in talker])
            run["levels"].append(row)
    for w, top in WORKLOADS.items():
        entry = workload_entry(run["levels"], w, top)
        if entry:
            run["workload"][w] = entry
    return run


# ---------------------------------------------------------------- outputs

def by_arm(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = defaultdict(dict)
    keys = sorted({(x["workload"], x["phase"].rsplit("_", 1)[0] if x["kind"] == "closed_loop" else x["phase"], x["mode"], x["kind"],
                    x.get("concurrency", x.get("rate_rps"))) for r in runs for x in r["levels"]}, key=str)
    for arm in sorted({r["arm"] for r in runs}):
        mine = sorted((r for r in runs if r["arm"] == arm), key=lambda r: r["block"])
        for w, ph, mode, kind, level in keys:
            rows = []
            for r in mine:
                hit = [x for x in r["levels"] if x["workload"] == w and x["mode"] == mode and x["kind"] == kind
                       and x.get("concurrency", x.get("rate_rps")) == level]
                rows.append(hit[0] if hit else None)
            if not any(rows):
                continue
            cell = {}
            for label, path in (("throughput_rps", "client.throughput_rps"), ("busy_window_rps", "busy_window.throughput_rps"),
                                ("audio_s_per_s", "client.audio_s_per_s"),
                                ("e2e_p50_s", "censored.p50"), ("e2e_p95_s", "censored.p95"), ("e2e_p99_s", "censored.p99"),
                                ("ttft_p50_s", "client.ttft_s.p50"), ("first_audio_p50_s", "client.first_audio_s.p50"),
                                ("growth_slope_rps", "growth.slope_rps"), ("tokens_mean", "output.tokens_mean"), ("audio_s_mean", "output.audio_s_mean")):
                vals = [get(x, path) if x else None for x in rows]
                cell[label] = interval([None if (v is None or (isinstance(v, float) and math.isinf(v))) else v for v in vals])
                cell[label]["infinite_runs"] = sum(1 for v in vals if isinstance(v, float) and math.isinf(v))
            cell["steady_runs"] = [get(x, "growth.steady") if x else None for x in rows]
            cell["unusable_runs"] = [x.get("usable") is False if x else None for x in rows]
            cell["after_restart_runs"] = [bool(x.get("after_restart")) if x else None for x in rows]
            out[str(arm)][f"{w}|{kind}|{mode}|{level:g}"] = cell
    return dict(out)


def get(d: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--pilot", action="store_true", help="print pilot capacities and the common open-loop rates")
    ap.add_argument("--out")
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    runs = [analyze_run(root / n) for n in a.run]
    if a.pilot:
        rates = {}
        for w in WORKLOADS:
            caps = {r["arm"]: r["workload"].get(w, {}).get("capacity_rps") for r in runs}
            print(w, {k: (round(v, 2) if v else v) for k, v in caps.items()},
                  {r["arm"]: r["workload"].get(w, {}).get("bottleneck") for r in runs})
            if all(caps.values()):
                rates[w] = common_rates(list(caps.values()))
                print(w, "common rates", rates[w])
        summary = {"generated_by": "scripts/analysis/exp09_report.py --pilot", "runs": runs, "rates": rates}
    else:
        summary = {"generated_by": "scripts/analysis/exp09_report.py", "runs": runs, "by_arm": by_arm(runs),
                   "verdicts": {"H9a": verdict_h9a(runs), "H9b": verdict_h9b(runs), "H9b_w3l": verdict_h9b(runs, "w3l"),
                                "H9c": verdict_h9c(runs), "H9d": capacity_model(runs, "w3b"), "H9d_w3l": capacity_model(runs, "w3l")}}
    if a.out:
        out = Path(a.out) / "results"
        out.mkdir(parents=True, exist_ok=True)
        name = "pilot.json" if a.pilot else "summary.json"
        (out / name).write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=lambda x: None if isinstance(x, float) and math.isinf(x) else str(x)) + "\n")
        if not a.pilot:
            with open(out / "levels.csv", "w", newline="") as fh:
                wr = csv.writer(fh)
                wr.writerow(["arm", "placement", "block", "workload", "kind", "mode", "level", "throughput_rps", "busy_window_rps", "audio_s_per_s",
                             "e2e_p50_s", "e2e_p95_s", "e2e_p99_s", "failed", "ttft_p50_s", "growth_slope_rps", "steady",
                             "tokens_mean", "audio_s_mean", "bottleneck", "usable", "restart_here", "after_restart"])
                for r in runs:
                    for x in r["levels"]:
                        wr.writerow([r["arm"], r["placement"], r["block"], x["workload"], x["kind"], x["mode"], x.get("concurrency", x.get("rate_rps")),
                                     get(x, "client.throughput_rps"), get(x, "busy_window.throughput_rps"), get(x, "client.audio_s_per_s"),
                                     get(x, "censored.p50"),
                                     get(x, "censored.p95"), get(x, "censored.p99"), get(x, "censored.failed"), get(x, "client.ttft_s.p50"),
                                     get(x, "growth.slope_rps"), get(x, "growth.steady"), get(x, "output.tokens_mean"),
                                     get(x, "output.audio_s_mean"), get(x, "bottleneck.stage"), x.get("usable"), x.get("restart_here"),
                                     x.get("after_restart")])
        print(out / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
