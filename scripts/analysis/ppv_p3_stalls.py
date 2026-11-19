#!/usr/bin/env python3
"""P3 of the pre-presentation validation: pauses in every run of this round, aligned with Linux pressure (PSI).

    python scripts/analysis/ppv_p3_stalls.py --data-root <results root> --pattern '*ppv_*' \
        --out research/next_steps/presentation_validation/results/p3_stalls.json

For each run, the measured phases (closed loop, open loop, plugin, agent, replay) are scanned for
- log gaps: no line in the shared server log (every stage process writes logs/server_s0.log) for more than --gap-s;
- sampler gaps: the runner's 1 s CPU sampler (raw/cpu_samples.jsonl) itself missing a beat (> 1.8 s between records),
  i.e. the client/runner process was blocked too;
and every gap is described with the node's pressure stall information in and around it (I/O, CPU and memory stall
time from the cumulative PSI totals, in ms per second of wall time) and the number of processes in uninterruptible
sleep (state D). A log gap is only meaningful where the log is dense: runs whose median line spacing inside measured
phases exceeds --dense-s, and full-duplex runs (which log in bursts once per 1-s unit), are flagged "sparse" and their
log gaps are reported but not counted; their sampler gaps still count. The server log has naive local wall-clock stamps; the run's manifest gives the UTC offset.
Descriptive only: P3 registered no decision rule beyond "report every gap with its evidence; a run without gaps says
nothing about the cause".
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
from pathlib import Path
from typing import Any

STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) ")
# Gen-2 garbage collections logged by the P1 patch (sglang_omni.profiler.exp_trace): start in epoch ns, duration in ms.
GC = re.compile(r"exp_gc generation=2 pid=(\d+) stage=\S* start_ns=(\d+) duration_ms=([0-9.]+)")
EVENT_FILE = re.compile(r"^events_(?P<stage>[a-z0-9_]+?)(?:@r\d+)?_(?P<pid>\d+)\.jsonl$")
MEASURED_KINDS = ("closed_loop", "open_loop", "plugin", "agent", "replay")


def utc_offset_s(manifest: dict[str, Any]) -> float:
    """'-0500' -> -18000.0"""
    off = manifest.get("utc_offset") or "+0000"
    sign = -1 if off.startswith("-") else 1
    return sign * (int(off[1:3]) * 3600 + int(off[3:5]) * 60)


def log_times(lines, offset_s: float) -> list[float]:
    out = []
    for line in lines:
        m = STAMP.match(line)
        if m:
            naive = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
            out.append(naive.timestamp() - offset_s + int(m.group(2)) / 1e3)
    return sorted(out)


def gc_events(lines) -> list[dict[str, Any]]:
    out = []
    for line in lines:
        m = GC.search(line)
        if m:
            start = int(m.group(2)) / 1e9
            out.append({"pid": int(m.group(1)), "start": start, "end": start + float(m.group(3)) / 1e3,
                        "duration_ms": float(m.group(3))})
    return out


def stage_of_pid(exp: Path) -> dict[int, str]:
    """pid -> stage from the server-event file names (only runs that recorded request events)."""
    out: dict[int, str] = {}
    for f in (exp / "raw" / "server_events").rglob("events_*.jsonl"):
        m = EVENT_FILE.match(f.name)
        if m:
            out[int(m.group("pid"))] = m.group("stage")
    return out


def send_windows(exp: Path, phases: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """Per measured phase: the part of it with arriving load, in wall seconds.
    - With a client request file: first send to last send. Before the first send a plugin phase may still be
      preparing (R3 renders every frame first, and its 8 render workers starve the runner's sampler); after the last
      send the phase drains -- the last long requests finish and the log thins out -- so a silence there is not a
      pause of a loaded server (P1's W3L cells: every watcher pause fell 111-135 s into a cell whose sending ended at
      110 s).
    - Without one (full-duplex sessions stream for the whole phase): the whole phase."""
    from omni_research.analysis.experiment_analysis import clock_anchor

    anchor = clock_anchor(exp)
    out: dict[str, tuple[float, float]] = {}
    for p in phases:
        name = p.get("phase")
        f = exp / "raw" / f"requests_{name}.jsonl"
        if not f.exists():
            if name and (exp / "raw" / f"duplex_{name}").exists() and p.get("started") and p.get("ended"):
                out[name] = (p["started"], p["ended"])
            continue
        if anchor is None:
            continue
        mono0, wall0 = anchor
        sent = []
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict) and r.get("sent_ns") is not None:
                    sent.append(int(r["sent_ns"]))
        if sent:
            out[name] = ((min(sent) - mono0 + wall0) / 1e9, (max(sent) - mono0 + wall0) / 1e9)
    return out


def under_load(t: float, windows: dict[str, tuple[float, float]]) -> bool | None:
    """True inside a sending window, False outside, None when the run has no send windows (no request files)."""
    if not windows:
        return None
    return any(lo <= t <= hi for lo, hi in windows.values())


def watcher_records(exp: Path, windows: dict[str, tuple[float, float]] | None = None) -> dict[str, Any] | None:
    """The live stall watcher's records (raw/stalls_<server>.jsonl, written only when the run had stall_watch on):
    pauses of the server log while a load phase ran (split into pauses under arriving load and pauses in a phase's
    drain), the process states captured at each pause, and stats of the log on the network file system that
    themselves took > 0.2 s."""
    files = sorted((exp / "raw").glob("stalls_*.jsonl"))
    if not files:
        return None
    recs = [json.loads(x) for f in files for x in f.read_text().splitlines() if x.strip()]
    ends = [r for r in recs if r.get("event") == "pause_end"]
    load = [under_load(r["t_wall_ns"] / 1e9 - r["silent_s"], windows or {}) for r in ends]
    loaded = [r["silent_s"] for r, u in zip(ends, load) if u]
    snaps = [r for r in recs if r.get("event") == "pause_snapshot"]
    slow = [r for r in recs if r.get("event") == "slow_stat"]
    return {"pauses": sum(1 for r in recs if r.get("event") == "pause_start"),
            "pause_s": [r["silent_s"] for r in ends], "max_pause_s": max((r["silent_s"] for r in ends), default=None),
            "pauses_under_load": len(loaded), "max_pause_under_load_s": max(loaded, default=None),
            "pauses_in_drain": sum(1 for u in load if u is False), "pauses_load_unknown": sum(1 for u in load if u is None),
            "snapshots": len(snaps), "d_state_in_snapshots": sum(1 for r in snaps for p in r.get("processes", []) if p.get("state") == "D"),
            "slow_stats": len(slow), "slow_stat_max_s": max((r["stat_s"] for r in slow), default=None),
            "slow_stat_s": [r["stat_s"] for r in slow], "errors": sum(1 for r in recs if r.get("event") == "watcher_error")}


FRAME = re.compile(r'File "(.+?)", line (\d+) in (\S+)')
# innermost frames that mean "waiting for work" (event loop, queues, thread pools, pipes)
IDLE_FRAMES = {"threading.py:wait", "queue.py:get", "selectors.py:select", "thread.py:_worker",
               "threading.py:_wait_for_tstate_lock", "connection.py:_recv", "connection.py:wait", "socket.py:accept",
               "subproc_pool.py:_recv_msg", "exp_trace.py:flush_gc_records"}


def stack_summary(exp: Path) -> dict[str, Any] | None:
    """What the server processes were doing at the watcher's SIGUSR2 dumps (P1 runtime with the stack hook). A
    process's file holds one block per thread per dump with no separator between dumps, and faulthandler lists the
    signalled ("Current", i.e. main) thread wherever it falls, so a new dump starts when a thread id repeats.
    Reported: the main threads' innermost frames and every innermost frame that is not an idle wait."""
    files = sorted((exp / "stacks").glob("stack_*.txt"))
    if not files:
        return None
    main: dict[str, int] = {}
    busy: dict[str, int] = {}
    dumps = with_busy = 0
    header = re.compile(r"^(Current thread|Thread) (0x[0-9a-f]+) \(most recent call first\):", re.M)
    for f in files:
        text = f.read_text(errors="ignore")
        heads = list(header.finditer(text))
        groups: list[list[tuple[bool, str]]] = []
        seen: set[str] = set()
        for k, h in enumerate(heads):
            body = text[h.end():heads[k + 1].start() if k + 1 < len(heads) else len(text)]
            m = FRAME.search(body)
            key = f"{Path(m.group(1)).name}:{m.group(3)}" if m else "?"
            if not groups or h.group(2) in seen:
                groups.append([])
                seen = set()
            seen.add(h.group(2))
            groups[-1].append((h.group(1) == "Current thread", key))
        for g in groups:
            dumps += 1
            found = False
            for is_main, key in g:
                if is_main:
                    main[key] = main.get(key, 0) + 1
                if key not in IDLE_FRAMES and key != "?":
                    busy[key] = busy.get(key, 0) + 1
                    found = True
            with_busy += found
    top = lambda d: dict(sorted(d.items(), key=lambda kv: -kv[1])[:10])  # noqa: E731
    return {"process_dumps": dumps, "dumps_with_non_idle_thread": with_busy, "main_thread_top": top(main),
            "non_idle_top": top(busy)}


def gaps_in(times: list[float], lo: float, hi: float, gap_s: float) -> list[tuple[float, float]]:
    inside = [t for t in times if lo <= t <= hi]
    return [(a, b) for a, b in zip(inside, inside[1:]) if b - a > gap_s]


def median_spacing(times: list[float], windows: list[tuple[float, float]]) -> float | None:
    d = []
    for lo, hi in windows:
        inside = [t for t in times if lo <= t <= hi]
        d += [b - a for a, b in zip(inside, inside[1:])]
    return statistics.median(d) if d else None


def psi_per_second(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per consecutive pair of samples: stall ms per second of wall time for io/cpu/memory (some and full), the
    D-state count and the sample spacing."""
    out = []
    for a, b in zip(samples, samples[1:]):
        dt_s = (b["t_wall_ns"] - a["t_wall_ns"]) / 1e9
        if dt_s <= 0:
            continue
        row: dict[str, Any] = {"t0": a["t_wall_ns"] / 1e9, "t1": b["t_wall_ns"] / 1e9, "spacing_s": dt_s,
                               "d_state": sum((g.get("states") or {}).get("disk-sleep", 0) for g in (b.get("groups") or {}).values()
                                              if isinstance(g, dict))}
        for res in ("io", "cpu", "memory"):
            pa, pb = (a.get("psi") or {}).get(res) or {}, (b.get("psi") or {}).get(res) or {}
            for kind in ("some", "full"):
                k = f"{kind}_total"
                if k in pa and k in pb:
                    row[f"{res}_{kind}_ms_per_s"] = (pb[k] - pa[k]) / 1e3 / dt_s
        out.append(row)
    return out


def describe_gap(g: tuple[float, float], per_s: list[dict[str, Any]], margin_s: float = 1.0) -> dict[str, Any]:
    lo, hi = g[0] - margin_s, g[1] + margin_s
    rows = [r for r in per_s if r["t1"] >= lo and r["t0"] <= hi]
    peak = lambda key: max((r.get(key) or 0.0 for r in rows), default=None)  # noqa: E731
    return {"start": round(g[0], 3), "gap_s": round(g[1] - g[0], 3), "samples": len(rows),
            "io_full_ms_per_s_max": peak("io_full_ms_per_s"), "io_some_ms_per_s_max": peak("io_some_ms_per_s"),
            "cpu_some_ms_per_s_max": peak("cpu_some_ms_per_s"), "memory_some_ms_per_s_max": peak("memory_some_ms_per_s"),
            "d_state_max": max((r["d_state"] for r in rows), default=None),
            "sampler_spacing_max_s": max((r["spacing_s"] for r in rows), default=None)}


def analyze_run(exp: Path, gap_s: float = 1.0, dense_s: float = 0.25) -> dict[str, Any] | None:
    phases_path, log_path = exp / "analysis" / "phases.json", exp / "logs" / "server_s0.log"
    if not phases_path.exists():
        return None
    manifest = json.loads((exp / "manifest.json").read_text()) if (exp / "manifest.json").exists() else {}
    phases = [p for p in json.loads(phases_path.read_text()) if p.get("kind") in MEASURED_KINDS and p.get("ended")]
    windows = [(p["started"], p["ended"]) for p in phases]
    samples = []
    if (exp / "raw" / "cpu_samples.jsonl").exists():
        with open(exp / "raw" / "cpu_samples.jsonl") as fh:
            samples = [json.loads(x) for x in fh if x.strip()]
    per_s = psi_per_second(samples)
    in_measured = [r for r in per_s if any(lo <= r["t0"] <= hi for lo, hi in windows)]
    times: list[float] = []
    gcs: list[dict[str, Any]] = []
    if log_path.exists():
        with open(log_path, errors="ignore") as fh:
            lines = fh.readlines()
        times = log_times(lines, utc_offset_s(manifest))
        gcs = [g for g in gc_events(lines) if any(lo <= g["start"] <= hi for lo, hi in windows)]
    stages = stage_of_pid(exp)
    spacing = median_spacing(times, windows)
    # A full-duplex server logs in bursts once per 1-s unit and idles between them, so its log gaps are not pauses.
    duplex = any((exp / "raw").glob("duplex_*"))
    dense = spacing is not None and spacing <= dense_s and not duplex
    sends = send_windows(exp, phases)
    log_gaps = [describe_gap(g, per_s) for lo, hi in windows for g in gaps_in(times, lo, hi, gap_s)]
    for g in log_gaps:
        g["under_load"] = under_load(g["start"], sends)
    for g in log_gaps:  # a gen-2 collection overlapping the gap (its process cannot log while it collects)
        g["gc_overlap"] = [{"pid": c["pid"], "stage": stages.get(c["pid"]), "duration_ms": c["duration_ms"]}
                           for c in gcs if c["start"] < g["start"] + g["gap_s"] and c["end"] > g["start"]]
    sampler_gaps = [describe_gap((r["t0"], r["t1"]), per_s) for r in in_measured if r["spacing_s"] > 1.8]
    for g in sampler_gaps:
        g["under_load"] = under_load(g["start"], sends)
    base = lambda key: statistics.median([r.get(key) or 0.0 for r in in_measured]) if in_measured else None  # noqa: E731
    return {"run": exp.name, "measured_s": round(sum(hi - lo for lo, hi in windows), 1), "duplex": duplex,
            "log_median_spacing_s": spacing, "log_dense": dense,
            "log_gaps": log_gaps if dense else [], "log_gaps_sparse_log": [] if dense else log_gaps,
            "sampler_gaps": sampler_gaps, "watcher": watcher_records(exp, sends), "send_windows": len(sends),
            "stacks": stack_summary(exp),
            "gc_gen2": {"count": len(gcs), "duration_ms_p50": statistics.median([c["duration_ms"] for c in gcs]) if gcs else None,
                        "duration_ms_max": max((c["duration_ms"] for c in gcs), default=None),
                        "by_stage": {st: sum(1 for c in gcs if stages.get(c["pid"]) == st) for st in sorted({stages.get(c["pid"]) or "unknown" for c in gcs})}
                        if gcs else {}},
            "baseline": {"io_full_ms_per_s_median": base("io_full_ms_per_s"), "io_some_ms_per_s_median": base("io_some_ms_per_s"),
                         "cpu_some_ms_per_s_median": base("cpu_some_ms_per_s"), "d_state_seconds": sum(1 for r in in_measured if r["d_state"]),
                         "psi_seconds": len(in_measured)}}


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    dense = [r for r in runs if r["log_dense"]]
    gaps = [g for r in dense for g in r["log_gaps"]]
    return {"runs": len(runs), "dense_log_runs": len(dense), "measured_s_dense": round(sum(r["measured_s"] for r in dense), 1),
            "log_gaps_over_1s": len(gaps), "log_gaps_over_2s": sum(1 for g in gaps if g["gap_s"] > 2),
            "log_gaps_under_load": sum(1 for g in gaps if g.get("under_load")),
            "log_gaps_in_drain": sum(1 for g in gaps if g.get("under_load") is False),
            "max_log_gap_s": max((g["gap_s"] for g in gaps), default=None),
            "sampler_gaps": sum(len(r["sampler_gaps"]) for r in runs),
            "sampler_gaps_under_load": sum(1 for r in runs for g in r["sampler_gaps"] if g.get("under_load")),
            "gaps_with_io_stall": sum(1 for g in gaps if (g["io_full_ms_per_s_max"] or 0) >= 100),
            "gaps_with_d_state": sum(1 for g in gaps if (g["d_state_max"] or 0) > 0),
            "gaps_with_gc_overlap": sum(1 for g in gaps if g.get("gc_overlap")),
            "watched_runs": sum(1 for r in runs if r.get("watcher")),
            "watcher_pauses": sum(r["watcher"]["pauses"] for r in runs if r.get("watcher")),
            "watcher_max_pause_s": max((r["watcher"]["max_pause_s"] or 0 for r in runs if r.get("watcher")), default=None),
            # a watcher pause means something only where the log is dense (sparse logs fall silent by themselves)
            "watcher_pauses_under_load": sum(r["watcher"]["pauses_under_load"] for r in runs if r.get("watcher") and r["log_dense"]),
            "watcher_max_pause_under_load_s": max((r["watcher"]["max_pause_under_load_s"] or 0 for r in runs
                                                   if r.get("watcher") and r["log_dense"]), default=None),
            "watcher_pauses_under_load_sparse_log": sum(r["watcher"]["pauses_under_load"] for r in runs if r.get("watcher") and not r["log_dense"]),
            "watcher_pauses_in_drain": sum(r["watcher"]["pauses_in_drain"] for r in runs if r.get("watcher")),
            "watcher_slow_stats": sum(r["watcher"]["slow_stats"] for r in runs if r.get("watcher")),
            "watcher_slow_stat_max_s": max((r["watcher"]["slow_stat_max_s"] or 0 for r in runs if r.get("watcher")), default=None),
            "gc_gen2_in_measured_phases": sum(r["gc_gen2"]["count"] for r in runs),
            "gc_gen2_max_ms": max((r["gc_gen2"]["duration_ms_max"] or 0 for r in runs), default=None)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--pattern", default="*ppv_*")
    ap.add_argument("--gap-s", type=float, default=1.0)
    ap.add_argument("--dense-s", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    runs = [r for exp in sorted(root.glob(a.pattern)) if (r := analyze_run(exp, a.gap_s, a.dense_s))]
    doc = {"generated_by": "scripts/analysis/ppv_p3_stalls.py", "gap_s": a.gap_s, "dense_s": a.dense_s,
           "summary": summarize(runs), "runs": runs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(json.dumps(doc["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
