#!/usr/bin/env python3
"""实验 6（code2wav 的时间花在哪）：汇总三个区组、画图。

    python scripts/analysis/exp06_report.py --data-root ~/research-staging/unicorn \
        --run 20261117_exp06_c2w_block1_run01 --run ... --out research/01_性能瓶颈/实验06_Code2Wav时间拆解

Each run is one server start (a block). Phases are named w3b_c<concurrency>_<mode>: A has neither events nor
profiler (performance), B has request events with the measurement patch (the time breakdown), C has a 3 s torch
profiler window (kernel-level). Unit of replication: the run; requests inside a run are not independent. For every
(concurrency, mode) the per-run value is reported together with the mean over runs and a t-based 95 % interval
(n - 1 degrees of freedom; with 3 runs the interval is wide and no significance is claimed).
Writes results/summary.json (every number the report quotes), results/batches.csv, results/requests.csv and figures.
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
import exp06_c2w_breakdown as bd  # noqa: E402
from exp01_gpu_timeline import load_util, setup_matplotlib  # noqa: E402
from omni_research.analysis.experiment_analysis import read_jsonl  # noqa: E402
from omni_research.analysis.metrics import request_metrics  # noqa: E402

PHASE_RE = re.compile(r"^w3b_c(\d+)_([ABC])$")
SKIP = ("stream_chunk", "coordinator_stream_received")
THINKER_GPUS, TALKER_GPUS = [0, 1], [2, 3, 4, 5, 6, 7]


# Two-sided 95 % quantiles of Student's t (0.975 point) by degrees of freedom.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def t_interval(values: list[float]) -> dict[str, Any]:
    """Mean and t-based 95 % interval over runs (values as given, None dropped)."""
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    out: dict[str, Any] = {"runs": v, "n": len(v), "mean": float(np.mean(v)) if v else None, "ci95": None}
    if 2 <= len(v) <= 11:
        half = float(T975[len(v) - 1] * np.std(v, ddof=1) / math.sqrt(len(v)))
        out["ci95"] = [out["mean"] - half, out["mean"] + half]
    return out


def read_events(phase_dir: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted(phase_dir.rglob("*.jsonl")):
        with open(path) as fh:
            for line in fh:
                if any(s in line for s in SKIP):
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def client_metrics(recs: list[dict[str, Any]]) -> dict[str, Any]:
    meas = [r for r in recs if not r.get("warmup")]
    ok = [r for r in meas if r.get("status") == "ok" and r.get("sent_ns") and r.get("done_ns")]
    m = [request_metrics(r) for r in ok]
    e2e = [x["e2e_s"] for x in m if x.get("e2e_s") is not None]
    span = (max(r["done_ns"] for r in ok) - min(r["sent_ns"] for r in ok)) / 1e9 if ok else None
    first_audio = [(r["first_audio_ns"] - r["sent_ns"]) / 1e9 for r in ok if r.get("first_audio_ns")]
    audio = [r.get("output_audio_s") or 0.0 for r in ok]
    return {"measured": len(meas), "ok": len(ok), "errors": sum(1 for r in meas if r.get("status") not in ("ok", "timeout")),
            "timeouts": sum(1 for r in meas if r.get("status") == "timeout"),
            "throughput_rps": len(ok) / span if span else None, "audio_s_per_s": sum(audio) / span if span else None,
            "e2e_s": bd.stats(e2e), "ttft_s": bd.stats([x.get("ttft_s") for x in m]), "first_audio_s": bd.stats(first_audio),
            "completion_tokens_mean": float(np.mean([r.get("completion_tokens") or 0 for r in ok])) if ok else None,
            "audio_s_mean": float(np.mean(audio)) if audio else None}


def phase_window(recs: list[dict[str, Any]], anchor: tuple[int, int] | None) -> tuple[float, float] | None:
    if anchor is None or not recs:
        return None
    mono0, wall0 = anchor
    sent = [r["sent_ns"] for r in recs if r.get("sent_ns")]
    done = [r["done_ns"] for r in recs if r.get("done_ns")]
    return ((min(sent) - mono0 + wall0) / 1e9, (max(done) - mono0 + wall0) / 1e9) if sent and done else None


def analyze_run(exp: Path) -> dict[str, Any]:
    from omni_research.analysis.experiment_analysis import clock_anchor

    res = json.load(open(exp / "result.json"))
    util = load_util(exp)
    anchor = clock_anchor(exp)
    run: dict[str, Any] = {"run": exp.name, "experiment_id": res["experiment_id"], "phases": {}, "requests": [], "batches": []}
    for ph in res["phases"]:
        m = PHASE_RE.match(ph["phase"])
        if not m:
            continue
        c, mode = int(m.group(1)), m.group(2)
        recs = read_jsonl(exp / "raw" / f"requests_{ph['phase']}.jsonl")
        row: dict[str, Any] = {"concurrency": c, "mode": mode, "client": client_metrics(recs)}
        win = phase_window(recs, anchor)
        if win:
            busy = {g: float(np.mean([u for t, u, _ in util.get(g, []) if win[0] <= t <= win[1]] or [np.nan])) / 100 for g in range(8)}
            row["busy_thinker_gpus"] = float(np.nanmean([busy[g] for g in THINKER_GPUS]))
            row["busy_talker_gpus"] = float(np.nanmean([busy[g] for g in TALKER_GPUS]))
        if mode == "B":
            reqs, batches, issues = bd.breakdown(read_events(exp / "raw" / "server_events" / ph["phase"]))
            records = {r["request_id"]: r for r in recs if r.get("request_id")}
            row["breakdown"] = bd.summarize(reqs, batches, records)
            row["issues"] = issues
            for rid, r in reqs.items():
                rec = records.get(rid)
                if rec and rec.get("status") == "ok" and not rec.get("warmup"):
                    run["requests"].append({"concurrency": c, "request_id": rid, **{k: r.get(k) for k in bd.REQUEST_PARTS},
                                            "queue_depth_at_dequeue": r.get("queue_depth_at_dequeue"),
                                            "audio_s": rec.get("output_audio_s")})
            for (pid, bid), b in batches.items():
                run["batches"].append({"concurrency": c, "pid": pid, "batch_id": bid, **{k: v for k, v in b.items() if k not in ("request_ids", "token_lengths")},
                                       "audio_s": sum(b.get("token_lengths") or []) / 25.0 if b.get("token_lengths") else None,
                                       "max_tokens": max(b.get("token_lengths") or [0])})
        if mode == "C":
            ranges = cpu_ranges_by_pid(read_events(exp / "raw" / "server_events" / ph["phase"]))
            stats_by_proc = []
            for f in sorted((exp / "profiling" / "torch" / ph["phase"]).rglob("code2wav*.trace.json*")):
                pid = int(re.search(r"_pid(\d+)_", f.name).group(1))
                with gzip.open(f, "rt") if f.suffix == ".gz" else open(f) as fh:
                    stats_by_proc.append(bd.trace_stats(json.load(fh), cpu_ranges=ranges.get(pid, [])))
            row["kernel"] = aggregate_trace_stats(stats_by_proc)
        run["phases"][f"c{c}_{mode}"] = row
    return run


def cpu_ranges_by_pid(events: list[dict[str, Any]]) -> dict[int, list[tuple[str, int, int]]]:
    """pid -> CPU intervals (wall-clock ns) in which each code2wav batch issued its flow, HiFT and download work."""
    seen: set[tuple[int, int]] = set()
    out: dict[int, list[tuple[str, int, int]]] = defaultdict(list)
    for e in events:
        if e.get("event_name") != "exp_vocode":
            continue
        m, pid = e["metadata"], e.get("pid")
        key = (pid, m.get("batch_id"))
        if key in seen:
            continue
        seen.add(key)
        for name, a, b in (("c2w.prepare", "prepare_start_ns", "flow_issue_start_ns"), ("c2w.flow", "flow_issue_start_ns", "flow_issue_end_ns"),
                           ("c2w.hift", "flow_issue_end_ns", "hift_issue_end_ns"), ("c2w.download", "hift_issue_end_ns", "download_end_ns")):
            if m.get(a) and m.get(b):
                out[pid].append((name, int(m[a]), int(m[b])))
    return dict(out)


def aggregate_trace_stats(per_proc: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum over the code2wav processes of one window (each process has its own GPU)."""
    if not per_proc:
        return {}
    keys = ("window_ms", "kernels", "kernel_busy_ms", "kernel_sum_ms", "launch_api_ms", "launch_calls", "sync_api_ms", "unattributed_kernels")
    out: dict[str, Any] = {k: float(sum(p[k] for p in per_proc)) for k in keys}
    out["processes"] = len(per_proc)
    total = out["kernel_sum_ms"]
    graph = sum((p["graph_kernel_time_share"] or 0) * p["kernel_sum_ms"] for p in per_proc)
    out["graph_kernel_time_share"] = graph / total if total else None
    names = sorted({n for p in per_proc for n in p["by_range"]})
    out["by_range"] = {n: {**{key: sum(p["by_range"].get(n, {}).get(key, 0) for p in per_proc)
                              for key in ("kernels", "kernel_ms", "launch_api_ms", "sync_api_ms")},
                           "cpu_ms": sum(p["range_cpu_ms"].get(n, 0.0) for p in per_proc)} for n in names}
    out["queue_delay_ms_p50_mean"] = float(np.mean([p["queue_delay_ms"]["p50"] for p in per_proc if p["queue_delay_ms"]["p50"] is not None] or [np.nan]))
    return out


def get(d: dict[str, Any], path: str) -> Any:
    for part in path.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


SUMMARY_METRICS = {
    "A": ["client.throughput_rps", "client.audio_s_per_s", "client.e2e_s.p50", "client.e2e_s.p95", "client.e2e_s.p99", "client.ttft_s.p50",
          "client.first_audio_s.p50", "busy_thinker_gpus", "busy_talker_gpus"],
    "B": ["client.throughput_rps", "client.e2e_s.p50", "client.e2e_s.p95", "breakdown.total_ms.p50", "breakdown.queue_ms.p50",
          "breakdown.queue_ms.p95", "breakdown.batching_ms.p50", "breakdown.compute_ms.p50", "breakdown.compute_ms.p95", "breakdown.handoff_ms.p50",
          "breakdown.batch_size.mean", "breakdown.batch_size.p95", "breakdown.batch_queue_depth_at_formed.p50", "breakdown.batch_gpu_flow_ms.p50",
          "breakdown.batch_gpu_hift_ms.p50", "breakdown.batch_flow_issue_ms.p50", "breakdown.batch_hift_issue_ms.p50",
          "breakdown.batch_download_wait_ms.p50", "breakdown.batch_prepare_ms.p50", "breakdown.batch_references_ms.p50",
          "breakdown.batch_upload_ms.p50", "breakdown.batch_split_ms.p50", "breakdown.batch_postprocess_ms.p50", "breakdown.batch_hift_groups.mean",
          "busy_talker_gpus"],
    "C": ["client.throughput_rps", "client.e2e_s.p50", "kernel.kernels", "kernel.kernel_busy_ms", "kernel.kernel_sum_ms", "kernel.launch_api_ms",
          "kernel.launch_calls", "kernel.sync_api_ms", "kernel.graph_kernel_time_share", "kernel.window_ms", "kernel.queue_delay_ms_p50_mean"],
}


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    levels = sorted({int(k[1:].split("_")[0]) for r in runs for k in r["phases"]})
    out: dict[str, Any] = {}
    for c in levels:
        for mode, metrics in SUMMARY_METRICS.items():
            key = f"c{c}_{mode}"
            out[key] = {mpath: t_interval([get(r["phases"].get(key, {}), mpath) for r in runs]) for mpath in metrics}
        # Overheads: same block, B and C relative to A.
        for mode in ("B", "C"):
            for mpath in ("client.throughput_rps", "client.e2e_s.p50"):
                ratios = []
                for r in runs:
                    a, b = get(r["phases"].get(f"c{c}_A", {}), mpath), get(r["phases"].get(f"c{c}_{mode}", {}), mpath)
                    ratios.append(b / a - 1 if a and b is not None else None)
                out.setdefault(f"c{c}_overhead_{mode}", {})[mpath] = t_interval(ratios)
        paths: dict[str, list[float]] = defaultdict(list)
        for r in runs:
            share = get(r["phases"].get(f"c{c}_B", {}), "breakdown.flow_solve_path_share") or {}
            for p in ("dense_graph", "packed_graph", "packed_eager", "eager"):
                paths[p].append(share.get(p, 0.0))
        out[f"c{c}_flow_path"] = {p: t_interval(v) for p, v in paths.items()}
    return out


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    out = Path(a.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    runs = []
    for name in a.run:
        r = analyze_run(root / name)
        runs.append(r)
        print(f"{name}: {len(r['phases'])} phases, {len(r['requests'])} requests, {len(r['batches'])} batches", file=sys.stderr)
    summary = {"generated_by": "scripts/analysis/exp06_report.py", "runs": [r["run"] for r in runs],
               "per_run": [{"run": r["run"], "phases": r["phases"]} for r in runs], "by_level": summarize_runs(runs)}
    (out / "results" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float) + "\n")
    write_csv([{"run": r["run"], **x} for r in runs for x in r["requests"]], out / "results" / "requests.csv")
    write_csv([{"run": r["run"], **x} for r in runs for x in r["batches"]], out / "results" / "batches.csv")
    print(out / "results" / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
