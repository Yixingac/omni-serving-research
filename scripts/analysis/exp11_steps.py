#!/usr/bin/env python3
"""实验 11：按调度器的逻辑步对齐 thinker 的 GPU trace，算每步之间的空档，并把阻塞拷贝归到标注上。

    python scripts/analysis/exp11_steps.py --data-root <results root> --run 20261118_exp11_perf_block1_A_run01 ... \
        --out research/01_性能瓶颈/实验11_Thinker异步优化

The 实验 11 patch wraps every launch of the thinker scheduler in a profiler range thinker.step<i>.<MODE>.<la|sync>
and the end-of-request hidden copy in thinker.request_finished_copy. A step's GPU work is every GPU event whose
correlation id belongs to a CUDA call made inside its range, so steps outside CUDA graphs are covered too. The gap
before a step is its first GPU start minus the previous step's last GPU end (never negative); steps whose index does
not follow the previous one are flagged (an untraced step in between would make the gap look like one step's).
Idle shares: between-step gaps / window (registered) and window minus the union of all GPU work / window.
Blocking calls: synchronize calls, and copies that blocked the CPU over 1 ms, sorted by the bytes of their GPU copy.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

STEP_RE = re.compile(r"^thinker\.step(\d+)\.([A-Z_]+)\.(la|sync)$")
COPY_ANNOTATION = "thinker.request_finished_copy"
GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
SYNC = {"cudaStreamSynchronize", "cudaEventSynchronize", "cudaDeviceSynchronize"}
BLOCKING_US = 1000
SMALL_BYTES = 1024
PHASE_RE = re.compile(r"^(w3b_ns)_c(\d+)_C$")
RUN_RE = re.compile(r"exp11_perf_block(\d+)_([ABC])")


def union_length(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, -math.inf
    for a, b in sorted(intervals):
        if a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def events_of(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in trace.get("traceEvents", []) if e.get("ph") == "X"]


def steps(trace: dict[str, Any]) -> list[dict[str, Any]]:
    ev = events_of(trace)
    ann = [e for e in ev if e.get("cat") == "user_annotation" and STEP_RE.match(e.get("name", ""))]
    calls_by_tid: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for e in ev:
        if e.get("cat") in ("cuda_runtime", "cuda_driver"):
            calls_by_tid[e.get("tid")].append(e)
    gpu: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for e in ev:
        corr = (e.get("args") or {}).get("correlation")
        if e.get("cat") in GPU_CATS and corr is not None:
            gpu[corr].append(e)
    out = []
    for a in ann:
        m = STEP_RE.match(a["name"])
        lo, hi = a["ts"], a["ts"] + a.get("dur", 0)
        calls = [c for c in calls_by_tid.get(a.get("tid"), []) if lo <= c["ts"] <= hi]
        kern = [g for c in calls for g in gpu.get((c.get("args") or {}).get("correlation"), [])]
        if not kern:
            continue
        iv = [(k["ts"], k["ts"] + k.get("dur", 0)) for k in kern]
        out.append({"index": int(m.group(1)), "mode": m.group(2), "launch": m.group(3),
                    "graph": any(c.get("name") == "cudaGraphLaunch" for c in calls),
                    "gpu_start": min(s for s, _ in iv), "gpu_end": max(e for _, e in iv), "busy": union_length(iv), "gpu_events": len(kern)})
    return sorted(out, key=lambda s: s["index"])


def pct(values: list[float], p: float) -> float | None:
    return float(np.percentile(values, p)) if values else None


def gaps(step_list: list[dict[str, Any]], all_gpu: list[tuple[float, float]] | None = None) -> dict[str, Any]:
    rows, jumps = [], 0
    prev = None
    for s in step_list:
        r = dict(s)
        if prev is None:
            r["gap_before"], r["index_jump"] = None, False
        else:
            r["gap_before"] = max(0.0, s["gpu_start"] - prev["gpu_end"])
            r["index_jump"] = s["index"] != prev["index"] + 1
            jumps += r["index_jump"]
        rows.append(r)
        prev = s
    if not rows:
        return {"steps": [], "window": None}
    start, end = rows[0]["gpu_start"], max(r["gpu_end"] for r in rows)
    window = end - start
    gap_list = [r["gap_before"] for r in rows if r["gap_before"] is not None]
    if all_gpu is None:
        busy = sum(r["busy"] for r in rows)  # each step's own busy time; steps do not overlap on one stream
    else:
        busy = union_length([(max(a, start), min(b, end)) for a, b in all_gpu if b > start and a < end])
    by_type: dict[str, dict[str, Any]] = {}
    for key in sorted({f"{r['mode']}.{r['launch']}" for r in rows}):
        g = [r["gap_before"] for r in rows if f"{r['mode']}.{r['launch']}" == key and r["gap_before"] is not None]
        by_type[key] = {"steps": sum(f"{r['mode']}.{r['launch']}" == key for r in rows), "gap_sum": float(sum(g)),
                        "gap_p50": pct(g, 50), "gap_p95": pct(g, 95), "share_of_window": sum(g) / window if window else None}
    return {"steps": rows, "window": window, "index_jumps": jumps,
            "gap_p50": pct(gap_list, 50), "gap_p95": pct(gap_list, 95), "gap_p99": pct(gap_list, 99),
            "between_step_idle_share": sum(gap_list) / window if window else None,
            "total_idle_share": (window - busy) / window if window else None, "by_type": by_type,
            "graph_steps": sum(r["graph"] for r in rows), "non_graph_steps": sum(not r["graph"] for r in rows)}


def blocking_attribution(trace: dict[str, Any]) -> dict[str, Any]:
    ev = events_of(trace)
    ann = [e for e in ev if e.get("cat") == "user_annotation" and e.get("name") == COPY_ANNOTATION]
    gpu_bytes = {(e.get("args") or {}).get("correlation"): (e.get("args") or {}).get("bytes") for e in ev if e.get("cat") == "gpu_memcpy"}
    out = {k: {"n": 0, "inside": 0} for k in ("sync", "small", "large", "unknown")}
    for e in ev:
        if e.get("cat") not in ("cuda_runtime", "cuda_driver") or e.get("dur", 0) <= BLOCKING_US:
            continue
        if e["name"] in SYNC:
            kind = "sync"
        elif e["name"].startswith("cudaMemcpy"):
            n = gpu_bytes.get((e.get("args") or {}).get("correlation"))
            kind = "unknown" if n is None else "small" if n <= SMALL_BYTES else "large"
        else:
            continue
        inside = any(a.get("tid") == e.get("tid") and a["ts"] <= e["ts"] <= a["ts"] + a.get("dur", 0) for a in ann)
        out[kind]["n"] += 1
        out[kind]["inside"] += inside
    for v in out.values():
        v["share_inside"] = v["inside"] / v["n"] if v["n"] else None
    return out


def load_trace(path: Path) -> dict[str, Any]:
    with (gzip.open(path, "rt") if path.suffix == ".gz" else open(path)) as fh:
        return json.load(fh)


def analyze_run(exp: Path) -> dict[str, Any]:
    res = json.loads((exp / "result.json").read_text())
    m = RUN_RE.search(res["experiment_id"])
    run = {"run": exp.name, "block": int(m.group(1)), "arm": m.group(2), "levels": {}}
    for ph in res.get("phases", []):
        pm = PHASE_RE.match(ph["phase"])
        if not pm:
            continue
        procs = []
        for f in sorted((exp / "profiling" / "torch" / ph["phase"]).rglob("*.trace.json*")):
            tr = load_trace(f)
            st = steps(tr)
            if not st:
                continue
            all_gpu = [(e["ts"], e["ts"] + e.get("dur", 0)) for e in events_of(tr) if e.get("cat") in GPU_CATS]
            g = gaps(st, all_gpu)
            g.pop("steps")
            procs.append({"file": f.name.split("_trace")[0], **g, "blocking": blocking_attribution(tr),
                          "steps_per_s": len(st) / (g["window"] / 1e6) if g["window"] else None})
        run["levels"][pm.group(2)] = procs
    return run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    runs = [analyze_run(root / n) for n in a.run]
    out = Path(a.out) / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "steps.json").write_text(json.dumps({"generated_by": "scripts/analysis/exp11_steps.py", "runs": runs}, ensure_ascii=False, indent=1) + "\n")
    print(out / "steps.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
