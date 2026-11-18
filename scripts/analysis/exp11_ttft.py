#!/usr/bin/env python3
"""实验 11：把 TTFT 拆成沿请求路径的几段，比较同步执行和提前发射。

    python scripts/analysis/exp11_ttft.py --data-root <results root> --run 20261117_exp08_proto_perf_block1_off_run01 ... \
        --phase-re '^w3b_ns_c(\\d+)_B$' --arm-re 'block(\\d+)_(off|on)' --out research/01_性能瓶颈/实验11_Thinker异步优化 --name ttft_exp08data

Clocks: the client records CLOCK_MONOTONIC (sent_ns, first_text_ns); server events carry wall-clock timestamp_ns, and the
thinker's model_path_* events also carry CLOCK_MONOTONIC in metadata.monotonic_ns (same host, same boot). The offset
wall - monotonic is the median over those events; its spread is reported as the alignment uncertainty. Every server
point is moved to the monotonic clock with that offset, so all segments are differences on one clock.
Points (first occurrence per request): client send, coordinator request_admission, thinker stage_input_received,
scheduler_queue_enter, scheduler_prefill_start, scheduler_prefill_end, scheduler_first_emit,
thinker stage_first_stream_chunk_sent, decode stage_first_stream_chunk_sent, client first text.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

POINTS = [("admission", "coordinator", "request_admission"), ("thinker_in", "thinker", "stage_input_received"),
          ("queue_enter", "thinker", "scheduler_queue_enter"), ("prefill_start", "thinker", "scheduler_prefill_start"),
          ("prefill_end", "thinker", "scheduler_prefill_end"), ("first_emit", "thinker", "scheduler_first_emit"),
          ("thinker_chunk", "thinker", "stage_first_stream_chunk_sent"), ("decode_chunk", "decode", "stage_first_stream_chunk_sent")]
SEGMENTS = [("send_to_admission", "sent", "admission"), ("admission_to_thinker", "admission", "thinker_in"),
            ("thinker_in_to_queue", "thinker_in", "queue_enter"), ("queue_wait", "queue_enter", "prefill_start"),
            ("prefill", "prefill_start", "prefill_end"), ("prefill_to_first_emit", "prefill_end", "first_emit"),
            ("first_emit_to_thinker_chunk", "first_emit", "thinker_chunk"), ("thinker_chunk_to_decode_chunk", "thinker_chunk", "decode_chunk"),
            ("decode_chunk_to_client", "decode_chunk", "first_text")]
WANTED = {n for _, _, n in POINTS} | {"model_path_start", "model_path_end"}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


def read_events(phase_dir: Path) -> list[dict[str, Any]]:
    """Only the events this breakdown needs; the per-chunk stream events are skipped (they are most of the volume)."""
    out = []
    for path in sorted(phase_dir.rglob("*.jsonl")):
        with open(path) as fh:
            for line in fh:
                if '"stage_stream_chunk_' in line or "coordinator_stream_received" in line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("event_name") in WANTED:
                    out.append(e)
    return out


def clock_offset(events: list[dict[str, Any]]) -> dict[str, Any]:
    offs = [int(e["timestamp_ns"]) - int(e["metadata"]["monotonic_ns"]) for e in events
            if (e.get("metadata") or {}).get("monotonic_ns") is not None]
    if not offs:
        return {"offset_ns": None, "n": 0, "spread_ns": None}
    return {"offset_ns": int(np.median(offs)), "n": len(offs), "spread_ns": int(max(offs) - min(offs)),
            "p05_p95_ns": [int(np.percentile(offs, 5)), int(np.percentile(offs, 95))]}


def points(events: list[dict[str, Any]], offset_ns: int) -> dict[str, dict[str, int]]:
    want = {(stage, name): label for label, stage, name in POINTS}
    out: dict[str, dict[str, int]] = {}
    for e in events:
        label = want.get(((e.get("stage") or "").split("@")[0], e.get("event_name")))
        if label is None or not e.get("request_id"):
            continue
        t = int(e["timestamp_ns"]) - offset_ns
        d = out.setdefault(e["request_id"], {})
        d[label] = min(d.get(label, t), t)
    return out


def segments(pts: dict[str, int], client: dict[str, Any]) -> dict[str, float | None]:
    p = dict(pts)
    p["sent"], p["first_text"] = client.get("sent_ns"), client.get("first_text_ns")
    out: dict[str, float | None] = {}
    for name, a, b in SEGMENTS:
        out[name] = (p[b] - p[a]) / 1e6 if p.get(a) is not None and p.get(b) is not None else None
    out["ttft"] = (p["first_text"] - p["sent"]) / 1e6 if p.get("sent") is not None and p.get("first_text") is not None else None
    return out


def interval(values: list[Any]) -> dict[str, Any]:
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    out: dict[str, Any] = {"runs": values, "n": len(v), "mean": float(np.mean(v)) if v else None, "ci95": None}
    if len(v) >= 2:
        half = T975.get(len(v) - 1, 2.0) * float(np.std(v, ddof=1)) / math.sqrt(len(v))
        out["ci95"] = [out["mean"] - half, out["mean"] + half]
    return out


def analyze_run(exp: Path, phase_re: re.Pattern, arm_re: re.Pattern) -> dict[str, Any]:
    res = json.loads((exp / "result.json").read_text())
    m = arm_re.search(res["experiment_id"])
    run = {"run": exp.name, "block": int(m.group(1)), "arm": m.group(2), "levels": {}}
    for ph in res.get("phases", []):
        pm = phase_re.match(ph["phase"])
        if not pm:
            continue
        events = read_events(exp / "raw" / "server_events" / ph["phase"])
        off = clock_offset(events)
        if off["offset_ns"] is None:
            continue
        pts = points(events, off["offset_ns"])
        recs = [r for r in (json.loads(x) for x in open(exp / "raw" / f"requests_{ph['phase']}.jsonl") if x.strip())
                if not r.get("crash_affected")]  # the runner reruns a crashed level into the same file
        rows = [segments(pts.get(r["request_id"], {}), r) for r in recs
                if not r.get("warmup") and r.get("status") == "ok" and r.get("request_id")]
        names = [n for n, _, _ in SEGMENTS] + ["ttft"]
        run["levels"][pm.group(1)] = {"clock": off, "requests": len(rows),
                                      "complete": sum(all(x.get(n) is not None for n in names) for x in rows),
                                      "median_ms": {n: (float(np.median([x[n] for x in rows if x.get(n) is not None]))
                                                        if any(x.get(n) is not None for x in rows) else None) for n in names},
                                      "mean_ms": complete_means(rows, names)}
    return run


def complete_means(rows: list[dict[str, Any]], names: list[str]) -> dict[str, float | None]:
    """Means over the requests that have every segment and the TTFT, so the segment means add up to the TTFT mean."""
    complete = [x for x in rows if all(x.get(n) is not None for n in names)]
    return {n: (float(np.mean([x[n] for x in complete])) if complete else None) for n in names}


def contrasts(runs: list[dict[str, Any]], base: str, arm: str, stat: str = "median_ms") -> dict[str, Any]:
    """Per level and segment: per-block statistic (median_ms or mean_ms) of both arms and their paired difference
    (arm - base), ms. H11a is registered on means; medians are descriptive."""
    by = {(r["arm"], r["block"]): r for r in runs}
    blocks = sorted({r["block"] for r in runs})
    levels = sorted({lv for r in runs for lv in r["levels"]}, key=float)
    out = {}
    for lv in levels:
        out[lv] = {}
        for n in [s for s, _, _ in SEGMENTS] + ["ttft"]:
            a = [((by.get((arm, b)) or {}).get("levels", {}).get(lv) or {}).get(stat, {}).get(n) for b in blocks]
            o = [((by.get((base, b)) or {}).get("levels", {}).get(lv) or {}).get(stat, {}).get(n) for b in blocks]
            out[lv][n] = {base: interval(o), arm: interval(a), "diff": interval([y - x if x is not None and y is not None else None for x, y in zip(o, a)])}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--phase-re", required=True)
    ap.add_argument("--arm-re", required=True)
    ap.add_argument("--base", default="off")
    ap.add_argument("--arm", action="append", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    runs = [analyze_run(root / n, re.compile(a.phase_re), re.compile(a.arm_re)) for n in a.run]
    arms = a.arm or sorted({r["arm"] for r in runs} - {a.base})
    summary = {"generated_by": "scripts/analysis/exp11_ttft.py", "runs": runs, "segments": [s for s, _, _ in SEGMENTS],
               "contrasts": {arm: contrasts(runs, a.base, arm) for arm in arms},
               "contrasts_mean": {arm: contrasts(runs, a.base, arm, stat="mean_ms") for arm in arms}}
    out = Path(a.out) / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    print(out / f"{a.name}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
