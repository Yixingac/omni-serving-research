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


def duplex_phase_metrics(phase_dir: Path) -> dict[str, Any]:
    scripts = json.loads((phase_dir / "scripts.json").read_text())["sessions"]
    sessions = []
    for i, s in enumerate(scripts):
        trace = phase_dir / f"session_{i:02d}.jsonl"
        if not trace.exists():
            continue
        sessions.append(session_metrics(read_jsonl(trace), s))
    return {"pooled": pool(sessions), "sessions": sessions}


def server_unit_metrics(events_dir: Path) -> dict[str, Any]:
    files = sorted(events_dir.rglob("*.jsonl")) if events_dir.exists() else []
    if not files:
        return {"available": False, "reason": "no request events for this phase"}
    try:
        from sglang_omni.profiler.duplex_metrics import build_duplex_session_metrics
    except ImportError as exc:
        return {"available": False, "reason": f"sglang_omni not importable here ({exc.__class__.__name__})"}
    events = [e for f in files for e in read_jsonl(f)]
    m = build_duplex_session_metrics(events)
    return {"available": True, "files": len(files), "events": len(events), "summary": m["summary"],
            "units": len(m["units"]), "sessions": len(m["sessions"]),
            "bypassed": dict(sorted(_bypass_counts(m["units"]).items()))}


def _bypass_counts(units: list[dict[str, Any]]) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for u in units:
        for st in u.get("stages", []):
            if st.get("status") == "bypassed":
                c[f"{st.get('stage')}:{st.get('bypass_reason')}"] += 1
    return c


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
    gpu = gpu_by_phase(load_gpu_samples(exp), phases)
    cpu_path = exp / "raw" / "cpu_samples.jsonl"
    cpu = cpu_by_phase(read_jsonl(cpu_path), phases) if cpu_path.exists() else []
    for ph in phases:
        name = ph.get("phase")
        row: dict[str, Any] = {"kind": ph.get("kind"), "started": ph.get("started"), "ended": ph.get("ended"),
                               "skipped": ph.get("skipped"), "error": ph.get("error"),
                               "server_events": ph.get("profiled") == "request_events",
                               "gpu": [g for g in gpu if g["phase"] == name], "cpu": next((c for c in cpu if c["phase"] == name), None)}
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
