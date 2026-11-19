"""Offline summary of R0-R2 runs: client-side duplex metrics (recomputed from the raw traces), the server's
unit/stage timing from its request-event recorder when the phase recorded it, GPU and CPU per phase, and
the request-response reference phases of the same runs.

    python -m omni_research.duplex.report --exp <run dir> [--exp <run dir> ...] --out summary.json

Server-side unit metrics use sglang_omni.profiler.duplex_metrics.build_duplex_session_metrics, so run this
with the sglang-omni runtime's python (on the cluster) to get them; elsewhere they are reported as
unavailable, never as zeros. Pooled levels across blocks: ``--pool-by-phase``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..analysis.experiment_analysis import gpu_by_phase, load_gpu_samples, phase_level_table, read_jsonl
from .metrics import dist, pool, session_metrics

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


def t_interval(values: list[float | None]) -> dict[str, Any]:
    """Mean of per-block statistics with a two-sided 95 % t interval (df = n - 1)."""
    import math
    import statistics

    v = [x for x in values if x is not None and math.isfinite(x)]
    if len(v) < 2:
        return {"n": len(v), "mean": v[0] if v else None, "lo": None, "hi": None, "values": v}
    m, sd = statistics.mean(v), statistics.stdev(v)
    h = T975.get(len(v) - 1, 2.0) * sd / math.sqrt(len(v))
    return {"n": len(v), "mean": m, "lo": m - h, "hi": m + h, "values": v}


def run_experiment_id(exp: Path) -> str | None:
    for name in ("result.json", "config.json"):
        p = exp / name
        if p.exists():
            try:
                return json.loads(p.read_text()).get("experiment_id")
            except ValueError:
                continue
    return None


def run_status(exp: Path) -> str | None:
    p = exp / "result.json"
    try:
        return json.loads(p.read_text()).get("status") if p.exists() else "no result.json"
    except ValueError:
        return "unreadable result.json"


def discover_runs(data_root: Path | None, exps: list[str]) -> list[Path]:
    """Explicit run directories plus every directory under <data root>/experiments."""
    paths = [Path(e).expanduser() for e in exps]
    if data_root is not None:
        root = Path(data_root).expanduser() / "experiments"
        paths += [p for p in sorted(root.iterdir()) if p.is_dir()] if root.exists() else []
    return paths


def choose_runs(exps: list[Path], pattern: Any, key: Any, require_ok: bool = True) -> tuple[dict[tuple, Path], list[dict[str, Any]]]:
    """Latest run per key(match) (run directories sort by date and run number) -- with ``require_ok`` the latest
    whose status is ok; the others are returned as excluded with the reason, never dropped silently."""
    by: dict[tuple, list[Path]] = defaultdict(list)
    for e in exps:
        m = pattern.match(run_experiment_id(e) or "")
        if m:
            by[key(m)].append(e)
    chosen: dict[tuple, Path] = {}
    excluded = []
    for k, runs in sorted(by.items(), key=lambda kv: str(kv[0])):
        runs = sorted(set(runs), key=lambda p: p.name)
        ok = [r for r in runs if run_status(r) == "ok"] if require_ok else runs
        if ok:
            chosen[k] = ok[-1]
        for r in runs:
            if chosen.get(k) != r:
                st = run_status(r)
                later = "a later ok run of the same key exists" if require_ok else "a later run of the same key exists"
                excluded.append({"run": r.name, "key": list(k), "status": st,
                                 "reason": later if (st == "ok" or not require_ok) else f"status {st}"})
    return chosen, excluded


def duplex_phase_metrics(phase_dir: Path) -> dict[str, Any]:
    scripts = json.loads((phase_dir / "scripts.json").read_text())["sessions"]
    sessions = []
    for i, s in enumerate(scripts):
        trace = phase_dir / f"session_{i:02d}.jsonl"
        if not trace.exists():
            continue
        sessions.append(session_metrics(read_jsonl(trace), s))
    return {"pooled": pool(sessions), "sessions": sessions}


def load_server_units(events_dir: Path) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None, str | None]:
    """(units, summary, reason): build_duplex_session_metrics over the phase's request events; units is None with
    a reason when there are no events or sglang_omni cannot be imported here (never an empty list for "unknown")."""
    files = sorted(events_dir.rglob("*.jsonl")) if events_dir.exists() else []
    if not files:
        return None, None, "no request events for this phase"
    try:
        from sglang_omni.profiler.duplex_metrics import build_duplex_session_metrics
    except ImportError as exc:
        return None, None, f"sglang_omni not importable here ({exc.__class__.__name__}); use the runtime's python"
    m = build_duplex_session_metrics([e for f in files for e in read_jsonl(f)])
    return m["units"], m["summary"], None


def server_unit_metrics(events_dir: Path) -> dict[str, Any]:
    units, summary, reason = load_server_units(events_dir)
    if units is None:
        return {"available": False, "reason": reason}
    files = sorted(events_dir.rglob("*.jsonl"))
    return {"available": True, "files": len(files), "summary": summary, "units": len(units),
            "sessions": len({(u.get("session_id"), u.get("session_open_index")) for u in units}),
            "bypassed": dict(sorted(_bypass_counts(units).items())), "stage_table_all_units": stage_table(units)}


def _bypass_counts(units: list[dict[str, Any]]) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for u in units:
        for st in u.get("stages", []):
            if st.get("timing_status") == "bypassed":
                c[f"{st.get('stage')}:{st.get('bypass_reason')}"] += 1
    return c


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def stage_table(units: list[dict[str, Any]], select: Any = None) -> dict[str, Any]:
    """Per stage over the selected units (upstream unit reports): observed / bypassed / incomplete counts, queue,
    service, handoff and queue + service distributions, and each stage's share of the summed unit processing time
    (ready_to_finished_ms): share_s = sum(queue_s + service_s) / sum(ready_to_finished); handoff and "other"
    (ready_to_finished - covered) are their own categories (preregistration.md, R1/R2 分析口径补充 item 5)."""
    chosen = [u for u in units if (select is None or select(u)) and _num(u.get("ready_to_finished_ms"))]
    total = sum(float(u["ready_to_finished_ms"]) for u in chosen)
    stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"queue_ms": [], "service_ms": [], "handoff_ms": [],
                                                                    "queue_plus_service_ms": []})
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"observed": 0, "bypassed": 0, "incomplete": 0})
    stage_time: dict[str, float] = defaultdict(float)
    handoff_total = other_total = 0.0
    for u in chosen:
        covered = 0.0
        for st in u.get("stages", []):
            name = st.get("stage")
            status = st.get("timing_status")
            if status not in ("observed", "bypassed"):
                counts[name]["incomplete"] += 1
                continue
            counts[name][status] += 1
            if status == "bypassed":
                continue
            q, sv, h = st.get("queue_ms"), st.get("service_ms"), st.get("handoff_from_previous_ms")
            if _num(q) and _num(sv):
                stats[name]["queue_ms"].append(float(q))
                stats[name]["service_ms"].append(float(sv))
                stats[name]["queue_plus_service_ms"].append(float(q) + float(sv))
                stage_time[name] += float(q) + float(sv)
                covered += float(q) + float(sv)
            if _num(h):
                stats[name]["handoff_ms"].append(float(h))
                handoff_total += float(h)
                covered += float(h)
        other_total += float(u["ready_to_finished_ms"]) - covered
    shares = {name: (t / total if total else None) for name, t in sorted(stage_time.items())}
    top = max(stage_time, key=stage_time.get) if stage_time else None
    return {"units": len(chosen), "unit_time_ms": dist([float(u["ready_to_finished_ms"]) for u in chosen]),
            "stages": {name: {**counts[name], **{k: dist(v) for k, v in stats[name].items()}} for name in sorted(counts)},
            "shares": {**shares, "handoff": handoff_total / total if total else None,
                       "other": other_total / total if total else None},
            "top_stage": top}


def speaking_selector(units: list[dict[str, Any]], client_sessions: list[dict[str, Any]]) -> tuple[Any, str]:
    """Selector of the server units that spoke: the client saw audio with that unit id (joined on session id and
    unit index); if no server unit joins a client session, a unit that emitted outputs (preregistration item 5)."""
    speak = {(s.get("session_id"), int(i)) for s in client_sessions for i in s.get("speak_units", [])}
    ids = {s.get("session_id") for s in client_sessions if s.get("session_id")}
    if any(u.get("session_id") in ids for u in units):
        return (lambda u: (u.get("session_id"), int(u.get("input_seq", -1))) in speak), "client audio deltas (joined)"
    return (lambda u: bool(u.get("outputs"))), "server outputs (no session id joined)"


def pid_stages(event_dirs: list[Path]) -> dict[str, str]:
    """pid -> stage from the recorder's file names events_<stage>_<pid>.jsonl (one file per process)."""
    out: dict[str, str] = {}
    for d in event_dirs:
        for f in d.rglob("events_*_*.jsonl") if d.exists() else []:
            stage, _, pid = f.stem.removeprefix("events_").rpartition("_")
            if stage and pid.isdigit():
                out[pid] = stage
    return out


def stage_cpu(samples: list[dict[str, Any]], t0: float, t1: float, pids: dict[str, str], raw: bool = False) -> dict[str, Any]:
    """Per stage process CPU % (one core = 100) over [t0, t1]; a 1 s sample below 1 % is absent from by_pid and
    counts as 0. ``raw`` returns the per-sample values instead of their distribution (for pooling)."""
    rows = [s for s in samples if t0 <= s["t_wall_ns"] / 1e9 <= t1]
    per: dict[str, list[float]] = defaultdict(list)
    for s in rows:
        by = {}
        for name, g in (s.get("groups") or {}).items():
            if name != "client":
                by.update(g.get("by_pid") or {})
        for pid, stage in pids.items():
            per[stage].append(float(by.get(pid, 0.0)))
    if raw:
        return {stage: v for stage, v in sorted(per.items())}
    return {stage: dist(v) for stage, v in sorted(per.items())}


# Post-hoc data-quality flag (added 2026-11-19 after R2, changes no registered rule): nvidia-smi's utilization.gpu
# read 0 % for whole duplex phases while the card drew 110-150 W (idle with no process: ~35 W) and the server's stage
# events showed 70-350 ms of execution per 1 s unit; the same sampler reads plausibly on the aperiodic chat path. The
# duplex load is strictly periodic at 1 Hz (one unit per second of real-time input); the hypothesis is aliasing
# between that cadence and NVML's utilisation sampling. A window with at least half of its readings at 0 % while the
# mean power is at least 100 W is flagged util_suspect; its utilisation must not be used as evidence of idleness.
UTIL_SUSPECT_ZERO_SHARE = 0.5
UTIL_SUSPECT_POWER_W = 100.0


def run_tz(exp: Path) -> Any:
    """The run's time zone as a fixed offset (manifest.json utc_offset such as "-0500"), or None."""
    from datetime import timedelta, timezone

    try:
        off = json.loads((exp / "manifest.json").read_text()).get("utc_offset")
    except (OSError, ValueError):
        return None
    if not off or len(off) != 5 or off[0] not in "+-" or not off[1:].isdigit():
        return None
    return timezone((1 if off[0] == "+" else -1) * timedelta(hours=int(off[1:3]), minutes=int(off[3:5])))


def load_gpu_samples_utc(exp: Path) -> list[dict[str, Any]]:
    """nvidia-smi samples with their local timestamps converted using the run's own UTC offset (recorded by the
    runner), so windows do not depend on the time zone of the machine running the analysis; without a recorded
    offset this falls back to the analysis machine's local time zone."""
    from datetime import datetime

    samples = load_gpu_samples(exp)
    tz = run_tz(exp)
    if tz is None:
        return samples
    # load_gpu_samples read the stamps as local time of this machine; undo that and apply the run's offset
    return [{**s, "t": datetime.fromtimestamp(s["t"]).replace(tzinfo=tz).timestamp()} for s in samples]


def gpu_window(samples: list[dict[str, Any]], t0: float, t1: float) -> dict[str, Any]:
    """Per GPU index over [t0, t1]: samples, mean utilisation, share of 0 % readings, mean power, max memory, and
    util_suspect (see UTIL_SUSPECT_*)."""
    per: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        if t0 <= s["t"] <= t1:
            per[str(s["gpu"])].append(s)
    out = {}
    for g, v in sorted(per.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        power = [x["power_w"] for x in v if x.get("power_w") is not None]
        zero = sum(1 for x in v if x["util"] == 0) / len(v)
        pw = sum(power) / len(power) if power else None
        out[g] = {"samples": len(v), "util_mean": sum(x["util"] for x in v) / len(v), "util_zero_share": zero,
                  "power_mean_w": pw, "mem_max_mib": max((x["mem_mib"] for x in v if x["mem_mib"] is not None), default=None),
                  "util_suspect": bool(zero >= UTIL_SUSPECT_ZERO_SHARE and pw is not None and pw >= UTIL_SUSPECT_POWER_W)}
    return out


OOM_PATTERN = r"CUDA out of memory|OutOfMemoryError|memory allocation failed with OOM|memory mapping failed with OOM"


def oom_events(exp: Path) -> list[dict[str, Any]]:
    """CUDA out-of-memory lines of the server log with the time of the line (or of the last timestamped line
    before it: traceback lines carry none). Two stamp formats: SGLang's '2026-11-19 00:44:43,788' and torch's
    '[W1119 00:44:43.788238681 ...' (month and day only; the year comes from the last full stamp)."""
    import re
    from datetime import datetime

    log = exp / "logs" / "server_s0.log"
    if not log.exists():
        return []
    full = re.compile(r"^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}),(\d{3})")
    torch_w = re.compile(r"^\[(?:rank\d+\]:\[)?[WEI](\d{2})(\d{2}) (\d{2}):(\d{2}):(\d{2})\.(\d{6})")
    oom = re.compile(OOM_PATTERN)
    tz = run_tz(exp)
    year, last, out = None, None, []
    with open(log, errors="ignore") as fh:
        for line in fh:
            m = full.match(line)
            if m:
                year = int(m.group(1))
                last = datetime(*(int(x) for x in m.groups()[:6]), int(m.group(7)) * 1000, tzinfo=tz).timestamp()
            else:
                m = torch_w.match(line)
                if m and year:
                    mo, d, hh, mi, ss, us = (int(x) for x in m.groups())
                    last = datetime(year, mo, d, hh, mi, ss, us, tzinfo=tz).timestamp()
            if oom.search(line):
                out.append({"t": last, "line": line.strip()[:240]})
    return out


def stage_gpu_position(server_args: list[str], stage: str) -> int:
    """Position of the stage's GPU in the server's GPU list: the thinker moves with ``--thinker.gpu N``."""
    flag = f"--{stage}.gpu"
    if flag in server_args:
        return int(server_args[server_args.index(flag) + 1])
    return 0


FAILURE_PATTERNS = (  # (class, regex) checked in this order on the server log tail and error text
    ("memory", r"CUDA out of memory|OutOfMemoryError|out of memory|max_sessions needs|not enough memory|KV cache"),
    ("model_load", r"ModuleNotFoundError|ImportError|safetensors|weights|No such file or directory"),
    ("missing_feature", r"not_supported|NotImplementedError|not implemented"),
    ("model_behaviour", r"context_exhausted|context_limit"),
    ("protocol", r"buffer_overflow|invalid_state|invalid_request|not_applicable|protocol"),
)


def classify_failure(text: str) -> dict[str, Any]:
    """First matching failure class and the matching line (heuristic; read the log)."""
    import re

    for cls, pat in FAILURE_PATTERNS:
        for line in text.splitlines():
            if re.search(pat, line):
                return {"class": cls, "evidence": line.strip()[:300]}
    return {"class": "unclassified", "evidence": None}


def cpu_by_phase(samples: list[dict[str, Any]], phases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for ph in phases:
        t0, t1 = ph.get("started"), ph.get("ended")
        if not (t0 and t1):
            continue
        rows = [s for s in samples if t0 <= s["t_wall_ns"] / 1e9 <= t1]
        if not rows:
            continue
        servers = [g for s in rows for name, g in (s.get("groups") or {}).items() if name != "client"]
        top = [max((g.get("by_pid") or {"-": 0.0}).values()) for g in servers]
        client = [(s.get("groups") or {}).get("client", {}).get("cpu_percent", 0.0) for s in rows]
        out.append({"phase": ph.get("phase"), "samples": len(rows),
                    "system_cpu_percent_mean": sum(s.get("system_cpu_percent", 0.0) for s in rows) / len(rows),
                    "server_cpu_percent_mean": sum(g.get("cpu_percent", 0.0) for g in servers) / max(1, len(rows)),
                    "busiest_server_process_cpu_percent": dist(top), "client_cpu_percent_mean": sum(client) / len(client)})
    return out


def summarize_run(exp: Path) -> dict[str, Any]:
    phases = json.loads((exp / "analysis" / "phases.json").read_text()) if (exp / "analysis" / "phases.json").exists() else []
    res = json.loads((exp / "result.json").read_text()) if (exp / "result.json").exists() else {}
    out: dict[str, Any] = {"run": exp.name, "experiment_id": res.get("experiment_id"), "status": res.get("status"),
                           "server_restarts": res.get("server_restarts") or [], "phases": {}}
    samples = load_gpu_samples_utc(exp)
    gpu = gpu_by_phase(samples, phases)
    cpu_path = exp / "raw" / "cpu_samples.jsonl"
    cpu = cpu_by_phase(read_jsonl(cpu_path), phases) if cpu_path.exists() else []
    for ph in phases:
        name = ph.get("phase")
        row: dict[str, Any] = {"kind": ph.get("kind"), "started": ph.get("started"), "ended": ph.get("ended"),
                               "skipped": ph.get("skipped"), "error": ph.get("error"),
                               "server_events": ph.get("profiled") == "request_events",
                               "gpu": [g for g in gpu if g["phase"] == name], "cpu": next((c for c in cpu if c["phase"] == name), None),
                               "gpu_window": gpu_window(samples, ph["started"], ph["ended"]) if ph.get("started") and ph.get("ended") else {}}
        d = exp / "raw" / f"duplex_{name}"
        if d.exists() and (d / "scripts.json").exists():
            m = duplex_phase_metrics(d)
            row["duplex"] = m["pooled"]
            row["duplex_sessions"] = [{"status": (s["result"] or {}).get("status"), "units": s["units"],
                                       "turns": s["turns"], "barge_in": s["barge_in"], "replies": s["replies"],
                                       "playback": s["playback"], "pacing": s["pacing"], "errors": s["errors"]}
                                      for s in m["sessions"]]
            row["server_units"] = server_unit_metrics(exp / "raw" / "server_events" / name) if row["server_events"] else {
                "available": False, "reason": "phase ran without request events"}
        req = exp / "raw" / f"requests_{name}.jsonl"
        if req.exists():
            row["request_levels"] = phase_level_table(read_jsonl(req))
        out["phases"][name] = row
    return out


def pool_by_phase(runs: list[dict[str, Any]], exps: list[Path]) -> dict[str, Any]:
    """Pool every session of the same duplex phase name over runs (e.g. the three blocks of R2)."""
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for exp in exps:
        for d in sorted((exp / "raw").glob("duplex_*")):
            if (d / "scripts.json").exists():
                by[d.name.removeprefix("duplex_")].extend(duplex_phase_metrics(d)["sessions"])
    return {name: pool(sessions) for name, sessions in sorted(by.items())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool-by-phase", action="store_true")
    a = ap.parse_args()
    exps = [Path(e).expanduser() for e in a.exp]
    runs = [summarize_run(e) for e in exps]
    out: dict[str, Any] = {"generated_by": "omni_research.duplex.report", "runs": runs}
    if a.pool_by_phase:
        out["pooled_by_phase"] = pool_by_phase(runs, exps)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    for r in runs:
        for name, ph in r["phases"].items():
            d = ph.get("duplex")
            if d:
                print(f"{r['run']} {name}: sessions {d['sessions']} failed {d['sessions_failed']} unit lag p50 "
                      f"{d['unit_lag_ms']['p50']} ms miss {d['unit_miss_rate']} TTFA p50 {d['ttfa_ms']['p50']} ms "
                      f"answered {d['answered']}/{d['turns']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
