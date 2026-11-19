#!/usr/bin/env python3
"""P1 (length-aware batching + max wait on the full service) — pre-run capacity, per-cell metrics, registered verdicts.

    python scripts/analysis/ppv_p1_report.py --data-root <results root> --prerun 20261119_ppv_p1_prerun_run01 \
        --out research/next_steps/presentation_validation/results/p1_prerun.json
    python scripts/analysis/ppv_p1_report.py --data-root <results root> --run <block 1 run> --run <block 2> --run <block 3> \
        --out research/next_steps/presentation_validation/results/p1_summary.json

Rules (research/next_steps/p0_p3_experiment_plan.md P1 and presentation_validation/p1_p2_p3_design.md, written before
the runs):
- X per workload = busy-window throughput of the pre-run's closed-loop level (completions between the first and the
  last send of the measured requests, divided by that window); rates = 0.9X, 1.0X, 1.1X rounded half up to 0.5;
- a cell = one open-loop phase (workload, setting, rate); its first warmup_s seconds of scheduled arrivals are dropped;
- offered load = the cell's rate; completed throughput = measured requests completed ok / their sending window;
  steady = 实验 9's rule (slope of in-flight requests over the second half of the sending window <= 0.05 x rate);
  stable capacity of a (block, workload, setting) = the highest steady rate (a lower bound when the top rate is
  steady, "< 0.9X" when none is); an unsteady cell's throughput is never called a capacity;
- goodput = measured requests with TTFT <= 1 s and e2e / audio seconds <= 0.5, per second of the sending window;
  failed and timed-out requests count against goodput and as infinitely slow in the percentiles;
- code2wav queue = 实验 10's queue_ms (stage dispatch -> dequeue); the scheduler's own enqueue -> dequeue is a
  cross-check; length groups split at the cell's median audio length (short < median <= long);
- padding, batch size, CUDA graph hit = share of flow solves on a graph path (a per-batch count, not kernel time);
- starvation: per length group the queue P95 / P99 / max, and the share of requests waiting more than 3 x the
  FIFO16 cell's P95 of the same group (same block, workload, rate); share of batches with a forced pick;
- H1a (MIX, 1.0X and 1.1X, all 3 blocks): Length8 + 1 s goodput >= 1.05 x FIFO16 -> 支持; <= FIFO16 everywhere ->
  反驳; else 不确定. H1b (MIX, 1.0X and 1.1X, all 3 blocks): short requests' code2wav queue P99 with 1 s <= 0.8 x
  Length8's and audio seconds per second >= 0.95 x Length8's -> 支持, else 不确定. H1c (W3b, every rate, all 3
  blocks): Length8 + 1 s e2e P95 <= 1.05 x FIFO16's -> 支持, else 不确定. A cell with errors + timeouts > 5 % or a
  server restart is excluded; a verdict that misses a block is 数据不足.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp09_report import busy_window_throughput, censored_percentiles, queue_growth, round_half  # noqa: E402
from exp10_report import graph_share, interval, padding_ratio  # noqa: E402

CELL_RE = re.compile(r"^(w3b|w3l|mix)_(fifo16|length8(?:_w\d+)?)_(x\d{3})$")
RUN_RE = re.compile(r"ppv_p1_block(\d+)")
PRE_RE = re.compile(r"^pre_(w3b|w3l|mix)_c(\d+)$")
FRACTIONS = {"x090": 0.9, "x100": 1.0, "x110": 1.1}
BLOCKS = (1, 2, 3)
SLO_TTFT_S, SLO_E2E_PER_AUDIO_S = 1.0, 0.5


# ---------------------------------------------------------------- pure functions

def pct(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def prerun_capacity(records: list[dict[str, Any]]) -> dict[str, Any]:
    """X (busy-window throughput) and, next to it, the registered closed-loop capacity (first send -> last completion)."""
    meas = [r for r in records if not r.get("warmup")]
    ok = [r for r in meas if r.get("status") == "ok" and r.get("sent_ns") is not None and r.get("done_ns") is not None]
    span = (max(r["done_ns"] for r in ok) - min(r["sent_ns"] for r in ok)) / 1e9 if ok else None
    bw = busy_window_throughput(meas)
    return {"measured": len(meas), "ok": len(ok), "x_rps": bw["throughput_rps"], "busy_window_s": bw["window_s"],
            "registered_capacity_rps": len(ok) / span if span else None}


def prerun_rates(x: dict[str, float]) -> dict[str, dict[str, float]]:
    return {w: {tag: round_half(f * v) for tag, f in FRACTIONS.items()} for w, v in x.items()}


def measured(records: list[dict[str, Any]], warmup_s: float) -> list[dict[str, Any]]:
    """Requests scheduled after the cell's warm-up (open-loop records carry scheduled_ns)."""
    sched = [r["scheduled_ns"] for r in records if r.get("scheduled_ns") is not None]
    if not sched:
        return list(records)
    t0 = min(sched) + warmup_s * 1e9
    return [r for r in records if r.get("scheduled_ns") is not None and r["scheduled_ns"] >= t0]


def meets_slo(r: dict[str, Any]) -> bool:
    if r.get("status") != "ok" or None in (r.get("sent_ns"), r.get("done_ns"), r.get("first_text_ns")):
        return False
    audio = r.get("output_audio_s") or 0.0
    e2e = (r["done_ns"] - r["sent_ns"]) / 1e9
    return (r["first_text_ns"] - r["sent_ns"]) / 1e9 <= SLO_TTFT_S and audio > 0 and e2e / audio <= SLO_E2E_PER_AUDIO_S


def client_cell(meas: list[dict[str, Any]], rate: float) -> dict[str, Any]:
    sent = [r["sent_ns"] for r in meas if r.get("sent_ns") is not None]
    window = (max(sent) - min(sent)) / 1e9 if len(sent) >= 2 else None
    ok = [r for r in meas if r.get("status") == "ok" and r.get("sent_ns") is not None and r.get("done_ns") is not None]
    bad = sum(1 for r in meas if r.get("status") != "ok")
    cen = censored_percentiles(meas)
    ttft = [(r["first_text_ns"] - r["sent_ns"]) / 1e9 for r in ok if r.get("first_text_ns") is not None]
    growth = queue_growth(meas, rate)
    met = sum(1 for r in meas if meets_slo(r))
    return {"offered_rps": rate, "measured": len(meas), "ok": len(ok), "failed_or_timeout": bad,
            "usable": not (meas and bad / len(meas) > 0.05),
            "completed_rps": len(ok) / window if window else None,
            "audio_s_per_s": sum(r.get("output_audio_s") or 0.0 for r in ok) / window if window else None,
            "goodput_rps": met / window if window else None, "goodput_fraction": met / len(meas) if meas else None,
            "e2e_p50": cen["p50"], "e2e_p95": cen["p95"], "e2e_p99": cen["p99"],
            "ttft_p50": pct(ttft, 50), "ttft_p95": pct(ttft, 95), "ttft_p99": pct(ttft, 99),
            "steady": growth["steady"], "growth_slope_rps": growth["slope_rps"], "inflight_end": growth["inflight_end"],
            "window_s": window}


def length_groups(queue_ms: dict[str, float], audio_s: dict[str, float]) -> dict[str, Any]:
    """Queue statistics of the short and the long half of the cell (split at the median audio length)."""
    ids = [r for r in queue_ms if r in audio_s]
    if not ids:
        return {"split_s": None, "short": None, "long": None}
    split = float(np.median([audio_s[r] for r in ids]))

    def summary(sel: list[str]) -> dict[str, Any] | None:
        v = [queue_ms[r] for r in sel]
        return {"n": len(v), "p50": pct(v, 50), "p95": pct(v, 95), "p99": pct(v, 99) if len(v) >= 100 else None,
                "p99_n_note": None if len(v) >= 100 else f"P99 needs >= 100 requests, have {len(v)}",
                "max": float(max(v))} if v else None

    return {"split_s": split, "short": summary([r for r in ids if audio_s[r] < split]),
            "long": summary([r for r in ids if audio_s[r] >= split])}


def batch_cell(batches: list[dict[str, Any]]) -> dict[str, Any]:
    sizes = [b["size"] for b in batches if b.get("size")]
    forced = [b for b in batches if (b.get("forced_count") or 0) > 0]
    return {"batches": len(batches), "batch_size_mean": float(np.mean(sizes)) if sizes else None,
            "padding": padding_ratio(batches), "graph_hit": graph_share(batches),
            "forced_batch_share": len(forced) / len(batches) if batches else None,
            "forced_requests": sum(b.get("forced_count") or 0 for b in batches)}


def starvation_share(queue_ms: dict[str, float], audio_s: dict[str, float], split_s: float | None,
                     fifo_groups: dict[str, Any] | None) -> dict[str, float | None]:
    """Share of each length group waiting longer than 3 x the FIFO16 cell's P95 of that group."""
    out: dict[str, float | None] = {"short": None, "long": None}
    if split_s is None or not fifo_groups:
        return out
    for g in ("short", "long"):
        ref = (fifo_groups.get(g) or {}).get("p95")
        sel = [queue_ms[r] for r in queue_ms if r in audio_s and ((audio_s[r] < split_s) == (g == "short"))]
        out[g] = (sum(1 for v in sel if v > 3 * ref) / len(sel)) if sel and ref is not None else None
    return out


def valid(c: dict[str, Any] | None) -> bool:
    return bool(c) and c["client"].get("usable") is not False and not c.get("restart_here") and c.get("control_ok") is not False


def per_block(runs: list[dict[str, Any]], key: tuple[str, str, str]) -> dict[int, dict[str, Any]]:
    return {r["block"]: r["cells"][key] for r in runs if key in r["cells"] and valid(r["cells"][key])}


def verdict_h1a(runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows, missing = [], []
    for tag in ("x100", "x110"):
        cand, base = per_block(runs, ("mix", "length8_w1000", tag)), per_block(runs, ("mix", "fifo16", tag))
        for b in BLOCKS:
            x, y = (cand.get(b) or {}).get("client", {}).get("goodput_rps"), (base.get(b) or {}).get("client", {}).get("goodput_rps")
            if x is None or y is None:
                missing.append(f"{tag} block {b}")
                continue
            rows.append({"rate": tag, "block": b, "length8_w1000": x, "fifo16": y, "rel": x / y - 1 if y else math.inf})
    if missing:
        return {"verdict": "数据不足", "missing": missing, "rows": rows}
    v = "支持" if all(r["rel"] >= 0.05 for r in rows) else "反驳" if all(r["rel"] <= 0 for r in rows) else "不确定"
    return {"verdict": v, "rows": rows}


def verdict_h1b(runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows, missing = [], []
    for tag in ("x100", "x110"):
        cand, base = per_block(runs, ("mix", "length8_w1000", tag)), per_block(runs, ("mix", "length8", tag))
        for b in BLOCKS:
            c, l8 = cand.get(b), base.get(b)
            qc = (((c or {}).get("queue") or {}).get("short") or {}).get("p99")
            ql = (((l8 or {}).get("queue") or {}).get("short") or {}).get("p99")
            ac, al = (c or {}).get("client", {}).get("audio_s_per_s"), (l8 or {}).get("client", {}).get("audio_s_per_s")
            if None in (qc, ql, ac, al):
                missing.append(f"{tag} block {b}")
                continue
            rows.append({"rate": tag, "block": b, "short_p99_ms": qc, "length8_short_p99_ms": ql, "p99_ratio": qc / ql if ql else math.inf,
                         "audio_ratio": ac / al if al else math.inf})
    if missing:
        return {"verdict": "数据不足", "missing": missing, "rows": rows}
    ok = all(r["p99_ratio"] <= 0.8 and r["audio_ratio"] >= 0.95 for r in rows)
    return {"verdict": "支持" if ok else "不确定", "rows": rows}


def verdict_h1c(runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows, missing = [], []
    for tag in FRACTIONS:
        cand, base = per_block(runs, ("w3b", "length8_w1000", tag)), per_block(runs, ("w3b", "fifo16", tag))
        for b in BLOCKS:
            x, y = (cand.get(b) or {}).get("client", {}).get("e2e_p95"), (base.get(b) or {}).get("client", {}).get("e2e_p95")
            if x is None or y is None:
                missing.append(f"{tag} block {b}")
                continue
            rows.append({"rate": tag, "block": b, "length8_w1000": x, "fifo16": y, "ratio": x / y if y else math.inf})
    if missing:
        return {"verdict": "数据不足", "missing": missing, "rows": rows}
    return {"verdict": "支持" if all(r["ratio"] <= 1.05 for r in rows) else "不确定", "rows": rows}


def stable_capacity(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Per block, workload and setting: the highest steady offered rate (never an unsteady cell's throughput)."""
    out: dict[str, Any] = {}
    for r in runs:
        keys = {(w, s) for w, s, _ in r["cells"]}
        for w, s in sorted(keys):
            cells = [(FRACTIONS[t], r["cells"][(w, s, t)]) for t in FRACTIONS if (w, s, t) in r["cells"] and valid(r["cells"][(w, s, t)])]
            steady = [c["client"]["offered_rps"] for _, c in cells if c["client"].get("steady")]
            top = max((c["client"]["offered_rps"] for _, c in cells), default=None)
            out.setdefault(f"{w}|{s}", {})[str(r["block"])] = {
                "stable_rps_lower_bound": max(steady) if steady else None,
                "note": None if not steady else ("steady at the top rate: a lower bound" if max(steady) == top else None),
                "none_steady": not steady}
    return out


def contrasts(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Paired within a block: every setting against FIFO16 and against Length8, for the main metrics."""
    out: dict[str, Any] = {}
    keys = sorted({k for r in runs for k in r["cells"]})
    for w, s, t in keys:
        for ref in ("fifo16", "length8"):
            if s == ref or (ref == "length8" and not s.startswith("length8_w")):
                continue
            for metric in ("goodput_rps", "completed_rps", "e2e_p95", "e2e_p99", "audio_s_per_s"):
                x, y = per_block(runs, (w, s, t)), per_block(runs, (w, ref, t))
                vals = [None if b not in x or b not in y or not y[b]["client"].get(metric) or x[b]["client"].get(metric) is None
                        or math.isinf(y[b]["client"][metric]) or math.isinf(x[b]["client"][metric])
                        else x[b]["client"][metric] / y[b]["client"][metric] - 1 for b in BLOCKS]
                out[f"{w}|{t}|{s}_vs_{ref}|{metric}"] = interval(vals)
    return out


# ---------------------------------------------------------------- reading runs

def control_settings(res: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {ph["phase"].removeprefix("ctl_"): ph.get("control") or {} for ph in res.get("phases", []) if ph.get("kind") == "control"}


def control_matches(batches: list[dict[str, Any]], control: dict[str, Any]) -> bool | None:
    seen = [b for b in batches if b.get("batch_cap") is not None]
    if not seen:
        return None
    want_wait = -1.0 if control.get("max_wait_ms") is None else float(control["max_wait_ms"])
    return all(b["batch_cap"] == control.get("max_batch_size") and b.get("batch_policy") == control.get("policy")
               and (b.get("max_wait_ms") is None or abs(float(b["max_wait_ms"]) - want_wait) < 1e-6) for b in seen)


def attach_forced(events: list[dict[str, Any]], batches: dict[tuple, dict[str, Any]], requests: dict[str, dict[str, Any]]) -> None:
    """Fields the P1 patch adds to exp_batch_formed: per batch the forced count and setting, per request enqueue/wait."""
    for e in events:
        md = e.get("metadata") or {}
        if e.get("event_name") != "exp_batch_formed" or md.get("batch_id") is None:
            continue
        b = batches.get((e.get("pid"), md["batch_id"]))
        if b is not None:
            for k in ("batch_cap", "batch_policy", "window_size", "forced_count", "anchor_aged", "max_wait_ms"):
                if k in md:
                    b[k] = md[k]
        rid = e.get("request_id")
        if rid in requests and md.get("enqueue_ns", -1) > 0 and md.get("dequeue_ns"):
            requests[rid]["scheduler_wait_ms"] = (md["dequeue_ns"] - md["enqueue_ns"]) / 1e6
            requests[rid]["is_forced"] = bool(md.get("is_forced"))


def analyze_run(exp: Path) -> dict[str, Any]:
    import exp06_c2w_breakdown as bd
    from exp06_report import read_events
    from exp10_report import attach_extra
    from run_records import clean, restart_flags

    from omni_research.analysis.experiment_analysis import read_jsonl

    res = json.loads((exp / "result.json").read_text())
    block = int(RUN_RE.search(res["experiment_id"]).group(1))
    settings = control_settings(res)
    restarts = res.get("server_restarts") or []
    flags = restart_flags([ph["phase"] for ph in res.get("phases", [])], restarts)
    cfg = json.loads((exp / "config.json").read_text()) if (exp / "config.json").exists() else {}
    warmups = {p["name"]: float(p.get("warmup_s", 0)) for p in cfg.get("phases", []) if p.get("kind") == "open_loop"}
    run: dict[str, Any] = {"run": exp.name, "block": block, "restarts": restarts, "cells": {}}
    for ph in res.get("phases", []):
        m = CELL_RE.match(ph["phase"])
        if not m:
            continue
        w, setting, tag = m.groups()
        recs, dropped = clean(read_jsonl(exp / "raw" / f"requests_{ph['phase']}.jsonl"))
        rate = float(next((r["rate_rps"] for r in recs if r.get("rate_rps") is not None), 0.0))
        meas = measured(recs, warmups.get(ph["phase"], 15.0))
        cell: dict[str, Any] = {"client": client_cell(meas, rate), "crash_affected_dropped": dropped,
                                "restart_here": flags[ph["phase"]][0], "after_restart": flags[ph["phase"]][1]}
        ev_dir = exp / "raw" / "server_events" / ph["phase"]
        if ev_dir.exists():
            events = read_events(ev_dir)
            reqs, batches, issues = bd.breakdown(events)
            attach_extra(events, batches)
            attach_forced(events, batches, reqs)
            ids = {r["request_id"] for r in meas if r.get("request_id")}
            audio = {r["request_id"]: r["output_audio_s"] for r in meas if r.get("request_id") in ids and r.get("status") == "ok"
                     and r.get("output_audio_s")}
            mine = [b for b in batches.values() if 2 * sum(x in ids for x in b.get("request_ids", [])) > len(b.get("request_ids", []))]
            queue = {i: reqs[i]["queue_ms"] for i in ids if i in reqs and reqs[i].get("queue_ms") is not None}
            sched_wait = {i: reqs[i]["scheduler_wait_ms"] for i in ids if i in reqs and reqs[i].get("scheduler_wait_ms") is not None}
            cell.update(queue=length_groups(queue, audio), scheduler_wait=length_groups(sched_wait, audio),
                        batches=batch_cell(mine), issues=issues, control_ok=control_matches(mine, settings.get(ph["phase"], {})),
                        queue_values={"ms": queue, "audio_s": audio})
        run["cells"][(w, setting, tag)] = cell
    # starvation against the FIFO16 cell of the same workload and rate
    for (w, setting, tag), cell in run["cells"].items():
        ref = run["cells"].get((w, "fifo16", tag))
        if "queue" in cell and ref and "queue" in ref:
            cell["starvation"] = starvation_share(cell["queue_values"]["ms"], cell["queue_values"]["audio_s"],
                                                  cell["queue"]["split_s"], ref["queue"])
    for cell in run["cells"].values():
        cell.pop("queue_values", None)
    return run


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({k for r in runs for k in r["cells"]})
    out = {}
    for key in keys:
        cells = [r["cells"].get(key) for r in sorted(runs, key=lambda r: r["block"])]
        cells = [c if valid(c) else None for c in cells]
        out["|".join(key)] = {k: interval([(c or {}).get("client", {}).get(k) for c in cells])
                              for k in ("offered_rps", "completed_rps", "goodput_rps", "e2e_p50", "e2e_p95", "e2e_p99", "audio_s_per_s")} | {
            "steady": [(c or {}).get("client", {}).get("steady") for c in cells],
            "padding": interval([((c or {}).get("batches") or {}).get("padding") for c in cells]),
            "graph_hit": interval([((c or {}).get("batches") or {}).get("graph_hit") for c in cells]),
            "forced_batch_share": interval([((c or {}).get("batches") or {}).get("forced_batch_share") for c in cells]),
            "short_queue_p99_ms": interval([(((c or {}).get("queue") or {}).get("short") or {}).get("p99") for c in cells]),
            "long_queue_p99_ms": interval([(((c or {}).get("queue") or {}).get("long") or {}).get("p99") for c in cells])}
    return out


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {("|".join(k) if isinstance(k, tuple) else str(k)): jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [jsonable(v) for v in x]
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return x


def main() -> int:
    from omni_research.analysis.experiment_analysis import read_jsonl

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--prerun")
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    if a.prerun:
        exp = root / a.prerun
        res = json.loads((exp / "result.json").read_text())
        cap = {}
        for ph in res.get("phases", []):
            m = PRE_RE.match(ph["phase"])
            if m:
                cap[m.group(1)] = {"concurrency": int(m.group(2)), **prerun_capacity(read_jsonl(exp / "raw" / f"requests_{ph['phase']}.jsonl"))}
        x = {w: c["x_rps"] for w, c in cap.items() if c.get("x_rps")}
        out = {"generated_by": "scripts/analysis/ppv_p1_report.py --prerun", "run": a.prerun, "capacity": cap, "x_rps": x,
               "rates_rps": prerun_rates(x), "restarts": res.get("server_restarts") or []}
    else:
        runs = [analyze_run(root / r) for r in a.run]
        out = {"generated_by": "scripts/analysis/ppv_p1_report.py", "runs": [{**r, "cells": r["cells"]} for r in runs],
               "summary": summarize(runs), "stable_capacity": stable_capacity(runs), "contrasts": contrasts(runs),
               "verdicts": {"H1a": verdict_h1a(runs), "H1b": verdict_h1b(runs), "H1c": verdict_h1c(runs)}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(jsonable(out), ensure_ascii=False, indent=1))
    print(json.dumps(jsonable(out.get("verdicts") or out.get("rates_rps")), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
