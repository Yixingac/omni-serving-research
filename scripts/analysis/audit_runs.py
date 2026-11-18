#!/usr/bin/env python3
"""Audit of finished runs: run status, server restarts, per-phase request outcomes and profiler traces, kept apart.

    python scripts/analysis/audit_runs.py --data-root <results root> --glob '20261117_exp0[678]_*' --out <file.json>

"The orchestrator task succeeded" only means the run's script exited normally. A run can still contain a server
restart, requests that failed or timed out, and profiler phases where a crashed process wrote no trace. Missing traces
are judged per process role (pipeline, talker, code2wav and replica index) against all roles traced in the same run. A phase without a request file is
reported as unknown (None), not as zero requests.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

TRACE_RE = re.compile(r"^(.+?)@r(\d+)_trace_pid\d+")
# One pipeline process runs these stages and names its trace after whichever stage it saw first.
PIPELINE_STAGES = {"preprocessing", "image_encoder", "audio_encoder", "thinker", "decode", "pipeline"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def role(stage: str, replica: str) -> str:
    return f"{'pipeline' if stage in PIPELINE_STAGES else stage}@r{replica}"


def traced_processes(phase_dir: Path) -> tuple[set[str], int]:
    files = list(phase_dir.rglob("*.trace.json*"))
    return {role(m.group(1), m.group(2)) for f in files for m in [TRACE_RE.match(f.name)] if m}, len(files)


def audit_run(exp: Path) -> dict[str, Any]:
    res = json.loads((exp / "result.json").read_text())
    restarts = res.get("server_restarts") or []
    torch_dir = exp / "profiling" / "torch"
    traced = {d.name: traced_processes(d) for d in torch_dir.iterdir() if d.is_dir()} if torch_dir.exists() else {}
    reference = set().union(*(roles for roles, _ in traced.values())) if traced else set()
    phases = {}
    for ph in res.get("phases", []):
        name = ph["phase"]
        path = exp / "raw" / f"requests_{name}.jsonl"
        requests = None
        if path.exists():
            meas = [r for r in read_jsonl(path) if not r.get("warmup")]
            requests = {"measured": len(meas), "ok": sum(r.get("status") == "ok" for r in meas),
                        "error": sum(r.get("status") not in ("ok", "timeout") for r in meas),
                        "timeout": sum(r.get("status") == "timeout" for r in meas),
                        "crash_affected": sum(bool(r.get("crash_affected")) for r in meas)}
        traces = None
        if name in traced:
            roles, files = traced[name]
            traces = {"files": files, "missing": sorted(reference - roles)}
        phases[name] = {"kind": ph.get("kind"), "requests": requests, "traces": traces,
                        "restart_here": any(x.get("where", "").split()[-1] == name for x in restarts)}
    return {"run": exp.name, "experiment_id": res.get("experiment_id"), "run_status": res.get("status"), "wall_s": res.get("wall_s"),
            "restarts": [f"{x.get('server')} exit {x.get('exit_code')} in {x.get('where')}" for x in restarts], "phases": phases}


def summarize(audits: list[dict[str, Any]]) -> dict[str, int]:
    phases = [p for a in audits for p in a["phases"].values()]
    reqs = [p["requests"] for p in phases if p["requests"]]
    return {"runs": len(audits), "runs_ok": sum(a["run_status"] == "ok" for a in audits),
            "runs_with_restarts": sum(bool(a["restarts"]) for a in audits),
            "phases_with_request_errors": sum((r["error"] + r["timeout"]) > 0 for r in reqs),
            "measured_requests": sum(r["measured"] for r in reqs), "ok_requests": sum(r["ok"] for r in reqs),
            "error_requests": sum(r["error"] for r in reqs), "timeout_requests": sum(r["timeout"] for r in reqs),
            "phases_with_missing_traces": sum(bool(p["traces"] and p["traces"]["missing"]) for p in phases)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    audits = [audit_run(d) for d in sorted(root.glob(a.glob)) if (d / "result.json").exists()]
    out = {"generated_by": "scripts/analysis/audit_runs.py", "glob": a.glob, "summary": summarize(audits), "runs": audits}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(out["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
