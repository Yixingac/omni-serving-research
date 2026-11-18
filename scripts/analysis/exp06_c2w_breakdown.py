#!/usr/bin/env python3
"""实验 6：把 code2wav 阶段的时间拆成排队、凑批、计算、交出四段，计算再拆成 CPU 各段和 GPU 跨度。

Inputs are the request events of one phase: the built-in stage_dispatch / stage_complete of the code2wav
stage plus the measurement patch's exp_batch_formed, exp_batch_done and exp_vocode
(configs/runtime/patches/sglang_omni_instr_exp06_08.patch). All of them use the same wall clock
(time.time_ns in processes on one node).

Per request (ms):  total = queue + batching + compute + handoff, where
  queue     = left the inbox (dequeue_ns)        - stage_dispatch (put into the inbox)
  batching  = batch formed                       - left the inbox
  compute   = batch done                         - compute start          (shared by the batch)
  handoff   = stage_complete                     - batch done             (result picked up and forwarded)
Per batch, once (not per request): CPU parts of the vocode call, which run one after another on the
scheduler thread and therefore add up, and GPU spans on the code2wav stream (CUDA events), which overlap
the CPU parts and must not be added to them.

A part is None (unknown) when an event is missing, the times are out of order, the batch ids of a
request's events disagree, or a span exceeds an hour (a unit or clock mix-up); each case is counted in
the issues dict. Nothing is filled in from residuals.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

MS = 1e6
MAX_SPAN_MS = 3600 * 1000.0
PATCH_EVENTS = ("exp_batch_formed", "exp_batch_done", "exp_vocode")


def first_events(events: list[dict[str, Any]]) -> tuple[dict[str, dict[str, dict[str, Any]]], int]:
    """request id -> event name -> first event of the code2wav stage; and how many duplicates were dropped."""
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    duplicates = 0
    for e in sorted(events, key=lambda e: e.get("timestamp_ns", 0)):
        stage = (e.get("stage") or "").split("/")[-1]
        if not stage.startswith("code2wav"):
            continue
        rid, name = e.get("request_id"), e.get("event_name")
        if name in out[rid]:
            duplicates += 1
        else:
            out[rid][name] = e
    return dict(out), duplicates


def span_ms(a_ns: float | None, b_ns: float | None) -> float | None:
    return None if a_ns is None or b_ns is None else (b_ns - a_ns) / MS


def breakdown(events: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[tuple, dict[str, Any]], dict[str, int]]:
    """(per request, per batch, issue counts) for one phase's events; see the module docstring."""
    by_request, duplicates = first_events(events)
    issues: Counter[str] = Counter()
    if duplicates:
        issues["duplicate_events"] += duplicates
    requests: dict[str, dict[str, Any]] = {}
    batches: dict[tuple, dict[str, Any]] = {}
    for rid, evs in by_request.items():
        dispatch, complete = evs.get("stage_dispatch"), evs.get("stage_complete")
        if dispatch is None or complete is None:
            issues["missing_stage_events"] += 1
            continue
        total = span_ms(dispatch["timestamp_ns"], complete["timestamp_ns"])
        row: dict[str, Any] = {"total_ms": total, "queue_ms": None, "batching_ms": None, "compute_ms": None, "handoff_ms": None,
                               "batch": None, "pid": dispatch.get("pid")}
        requests[rid] = row
        if total is None or total < 0 or total > MAX_SPAN_MS:
            row["total_ms"] = None
            issues["implausible_span"] += 1
            continue
        formed, done = evs.get("exp_batch_formed"), evs.get("exp_batch_done")
        if formed is None or done is None:
            issues["missing_patch_events"] += 1
            continue
        fm, dm = formed["metadata"], done["metadata"]
        vocode = evs.get("exp_vocode")
        ids = {fm.get("batch_id"), dm.get("batch_id")} | ({vocode["metadata"].get("batch_id")} if vocode else set())
        if len(ids) != 1:
            issues["uncertain_batch"] += 1
            continue
        points = [dispatch["timestamp_ns"], fm["dequeue_ns"], formed["timestamp_ns"], dm["compute_start_ns"], done["timestamp_ns"],
                  complete["timestamp_ns"]]
        if any(b < a for a, b in zip(points, points[1:])):
            issues["out_of_order"] += 1
            continue
        row.update(queue_ms=span_ms(points[0], points[1]), batching_ms=span_ms(points[1], points[2]),
                   compute_ms=span_ms(points[3], points[4]), handoff_ms=span_ms(points[4], points[5]),
                   queue_depth_at_dequeue=fm.get("queue_depth_at_dequeue"))
        key = (formed.get("pid"), fm["batch_id"])
        row["batch"] = key
        batch = batches.setdefault(key, {"size": fm["batch_size"], "batch_cost": fm.get("batch_cost"), "request_ids": [],
                                         "queue_depth_at_formed": fm.get("queue_depth_at_formed"), "compute_ms": row["compute_ms"],
                                         "formed_ns": formed["timestamp_ns"]})
        batch["request_ids"].append(rid)
        if vocode is not None and "flow_issue_ms" not in batch:
            batch.update(vocode_parts(vocode["metadata"], vocode["timestamp_ns"]))
    return requests, batches, {k: v for k, v in issues.items() if v}


def vocode_parts(m: dict[str, Any], event_ns: int) -> dict[str, Any]:
    """CPU parts (sequential on the scheduler thread, they add up) and GPU spans (overlap them) of one vocode call."""
    g = m.get
    return {
        "prepare_ms": span_ms(g("prepare_start_ns"), g("prepare_end_ns")),
        "references_ms": span_ms(g("references_start_ns"), g("references_end_ns")),
        "upload_ms": span_ms(g("upload_start_ns"), g("flow_issue_start_ns")),
        "flow_issue_ms": span_ms(g("flow_issue_start_ns"), g("flow_issue_end_ns")),
        "hift_issue_ms": span_ms(g("flow_issue_end_ns"), g("hift_issue_end_ns")),
        "download_wait_ms": span_ms(g("hift_issue_end_ns"), g("download_end_ns")),
        "split_ms": span_ms(g("download_end_ns"), g("split_end_ns")),
        "postprocess_ms": span_ms(g("postprocess_start_ns"), event_ns),
        "gpu_flow_ms": g("gpu_flow_start_to_flow_end_ms"),
        "gpu_hift_ms": g("gpu_flow_end_to_hift_end_ms"),
        "rows": g("rows"), "token_lengths": g("token_lengths"), "hift_groups": g("hift_groups"),
        "flow_solve_path": g("flow_solve_path"), "flow_solve_rows": g("flow_solve_rows"), "flow_solve_frames": g("flow_solve_frames"),
    }


def stats(values: list[float]) -> dict[str, float | int | None]:
    v = np.array([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    return {"n": int(v.size), "mean": float(v.mean()), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99))}


REQUEST_PARTS = ("total_ms", "queue_ms", "batching_ms", "compute_ms", "handoff_ms")
BATCH_PARTS = ("size", "compute_ms", "prepare_ms", "references_ms", "upload_ms", "flow_issue_ms", "hift_issue_ms", "download_wait_ms",
               "split_ms", "postprocess_ms", "gpu_flow_ms", "gpu_hift_ms", "hift_groups", "queue_depth_at_formed", "batch_cost")


def summarize(requests: dict[str, dict[str, Any]], batches: dict[tuple, dict[str, Any]], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Distributions over the measured requests that succeeded (client records), and over the phase's batches."""
    ok = [rid for rid, r in records.items() if r.get("status") == "ok" and not r.get("warmup") and rid in requests]
    out: dict[str, Any] = {"requests": len(ok), "failed": sum(1 for r in records.values() if r.get("status") != "ok")}
    for part in REQUEST_PARTS:
        out[part] = stats([requests[rid][part] for rid in ok])
    out["batches"] = len(batches)
    for part in BATCH_PARTS:
        out["batch_size" if part == "size" else f"batch_{part}"] = stats([b.get(part) for b in batches.values()])
    paths = Counter(b.get("flow_solve_path") for b in batches.values() if b.get("flow_solve_path"))
    out["flow_solve_path_share"] = {k: v / sum(paths.values()) for k, v in paths.items()} if paths else {}
    return out


# ------------------------------------------------------------------------------------------
# kernel traces (torch profiler, one code2wav process, a few seconds)

DEVICE_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
RUNTIME_CATS = {"cuda_runtime", "cuda_driver"}


def union_us(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, None
    for a, b in sorted(intervals):
        if end is None or a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def trace_stats(trace: dict[str, Any], cpu_ranges: list[tuple[str, int, int]] | None = None) -> dict[str, Any]:
    """Kernel time, launch and sync API time, device queue delay, and the kernels each CPU range launched.

    A kernel belongs to the range whose CPU interval contains the runtime call that launched it (CUDA
    correlation id). Ranges are the trace's c2w.* annotations, or cpu_ranges = (name, start_ns, end_ns) on the
    wall clock (the measurement patch's events), matched through baseTimeNanoseconds + ts * 1000. The device
    queue delay is counted once per launch call: its first kernel's start minus the call's end (later kernels
    of a graph wait for earlier ones, not for the queue). Trace times are microseconds; results are milliseconds."""
    events = [e for e in trace.get("traceEvents", []) if e.get("ph") == "X" and "ts" in e]
    kernels = [e for e in events if e.get("cat") in DEVICE_CATS]
    calls = {(e.get("args") or {}).get("correlation"): e for e in events if e.get("cat") in RUNTIME_CATS}
    if cpu_ranges is not None:
        base_ns = int(trace.get("baseTimeNanoseconds") or 0)
        ranges = [{"name": n, "ts": (a - base_ns) / 1e3, "dur": (b - a) / 1e3} for n, a, b in cpu_ranges]
    else:
        ranges = [e for e in events if e.get("cat") == "user_annotation" and str(e.get("name", "")).startswith("c2w.")]
    by_range: dict[str, dict[str, float]] = {r: {"kernels": 0, "kernel_ms": 0.0, "launch_api_ms": 0.0, "sync_api_ms": 0.0}
                                             for r in sorted({e["name"] for e in ranges})}
    for call in calls.values():
        owner = next((r["name"] for r in ranges if r["ts"] <= call["ts"] <= r["ts"] + r.get("dur", 0)), None)
        if owner is None:
            continue
        if "Launch" in call["name"]:
            by_range[owner]["launch_api_ms"] += call.get("dur", 0) / 1e3
        elif "Synchronize" in call["name"]:
            by_range[owner]["sync_api_ms"] += call.get("dur", 0) / 1e3
    first_start: dict[Any, float] = {}
    unattributed, graph_us, total_us = 0, 0.0, 0.0
    for kern in kernels:
        dur = float(kern.get("dur", 0))
        total_us += dur
        if (kern.get("args") or {}).get("graph id", 0):
            graph_us += dur
        corr = (kern.get("args") or {}).get("correlation")
        call = calls.get(corr)
        if call is None:
            unattributed += 1
            continue
        first_start[corr] = min(first_start.get(corr, kern["ts"]), kern["ts"])
        owner = next((r["name"] for r in ranges if r["ts"] <= call["ts"] <= r["ts"] + r.get("dur", 0)), None)
        if owner is None:
            unattributed += 1
        else:
            by_range[owner]["kernels"] += 1
            by_range[owner]["kernel_ms"] += dur / 1e3
    runtime = [e for e in events if e.get("cat") in RUNTIME_CATS]
    span_us = (max(e["ts"] + e.get("dur", 0) for e in events) - min(e["ts"] for e in events)) if events else 0.0
    return {"window_ms": span_us / 1e3, "kernels": len(kernels),
            "kernel_busy_ms": union_us([(e["ts"], e["ts"] + float(e.get("dur", 0))) for e in kernels]) / 1e3,
            "kernel_sum_ms": total_us / 1e3, "graph_kernel_time_share": graph_us / total_us if total_us else None,
            "launch_api_ms": sum(e.get("dur", 0) for e in runtime if "Launch" in e["name"]) / 1e3,
            "launch_calls": sum(1 for e in runtime if "Launch" in e["name"]),
            "sync_api_ms": sum(e.get("dur", 0) for e in runtime if "Synchronize" in e["name"]) / 1e3,
            "range_cpu_ms": {name: sum(e.get("dur", 0) for e in ranges if e["name"] == name) / 1e3 for name in by_range},
            "by_range": by_range, "unattributed_kernels": unattributed,
            "queue_delay_ms": stats([(ts - (calls[c]["ts"] + calls[c].get("dur", 0))) / 1e3 for c, ts in first_start.items()])}
