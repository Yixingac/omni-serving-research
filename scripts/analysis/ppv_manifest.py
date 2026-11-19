#!/usr/bin/env python3
"""Experiment manifest of the pre-presentation validation (every ppv_* attempt, failed ones included).

    python scripts/analysis/ppv_manifest.py --state <orchestrator state.json> --data-root <results root> \
        --inclusion research/next_steps/presentation_validation/inclusion.json \
        --out-json research/next_steps/presentation_validation/experiment_manifest.json \
        --out-md research/next_steps/presentation_validation/experiment_manifest.md

Sources: the orchestrator's state.json (status, start/end, allocated GPU-seconds = GPUs x wall time of the Slurm
step), each run's manifest.json (research-repo commit, sglang-omni commit, model revision, Slurm step id) and
result.json (the runner's own status, server restarts). GPU-hours are allocated GPU-hours: the GPUs a step held times
its wall time, whether or not the GPUs computed. Inclusion in the final analysis is curated by hand in the inclusion
file ({experiment_id: {"included": bool, "reason": str}}), never inferred. Only basenames of run directories are
written, so the output carries no host or user paths. Stdlib only (runs on the cluster login node too).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

FAMILY = re.compile(r"^ppv_(p0|p1|p2|p3|r0|r1|r2|r3|r4|c1|c2)")


def iso(t: float | None) -> str | None:
    if t is None:
        return None
    return dt.datetime.fromtimestamp(t, dt.timezone(dt.timedelta(hours=-5))).strftime("%Y-%m-%d %H:%M:%S -0500")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def run_facts(run_dir: Path | None) -> dict[str, Any]:
    """Provenance from the run directory (manifest.json, result.json); missing files give None fields."""
    if run_dir is None:
        return {}
    man = read_json(run_dir / "manifest.json") or {}
    res = read_json(run_dir / "result.json") or {}
    cfg = read_json(run_dir / "config.json") or {}
    env = man.get("environment") or {}
    models = env.get("models") or {}
    slurm = env.get("slurm") or {}
    return {
        "run_dir": run_dir.name,
        "research_repo_commit": (env.get("research_repo_commit") or "")[:10] or None,
        "research_repo_dirty": env.get("research_repo_dirty"),
        "sglang_omni_commit": (env.get("sglang_omni_commit") or "")[:10] or None,
        "model_revision": next((m.get("revision") for m in models.values() if isinstance(m, dict) and m.get("revision")), None),
        "slurm_job": slurm.get("SLURM_JOB_ID"),
        "slurm_step": slurm.get("SLURM_STEP_ID"),
        "run_status": res.get("status") or man.get("status"),
        # the sglang-omni commit is 520b49ee for every worktree; the research patch is identified by the runtime profile
        "runtime_profile": cfg.get("runtime_profile"),
        "server_restarts": len(res.get("server_restarts") or []),
    }


def build(state: dict[str, Any], data_root: Path | None, inclusion: dict[str, Any], prefix: str = "ppv_") -> list[dict[str, Any]]:
    rows = []
    for eid, st in sorted(state.get("experiments", {}).items(), key=lambda kv: (kv[1].get("started_at") or 0, kv[0])):
        if not eid.startswith(prefix):
            continue
        out_dir = st.get("output_dir")
        run_dir = (data_root / "experiments" / Path(out_dir).name) if (data_root and out_dir) else None
        facts = run_facts(run_dir if run_dir and run_dir.exists() else None)
        if not facts and out_dir:
            facts = {"run_dir": Path(out_dir).name}
        gpus = (st.get("assigned") or {}).get("gpus")
        m = FAMILY.match(eid)
        inc = inclusion.get(eid) or {}
        rows.append({
            "experiment_id": eid,
            "family": m.group(1).upper() if m else None,
            "config": f"configs/experiments/ppv/{eid}.json",
            "orchestrator_status": st.get("status"),
            "attempts": st.get("attempts"),
            "failure": (st.get("failures") or [{}])[-1].get("detail") if st.get("failures") else None,
            "started": iso(st.get("started_at")),
            "ended": iso(st.get("ended_at")),
            "gpus": gpus,
            "gpu_hours_allocated": round(st["gpu_seconds"] / 3600, 3) if st.get("gpu_seconds") is not None else None,
            **facts,
            "included": inc.get("included"),
            "inclusion_reason": inc.get("reason"),
        })
    return rows


def to_markdown(rows: list[dict[str, Any]], extra: list[dict[str, Any]] | None = None) -> str:
    cols = ["experiment_id", "orchestrator_status", "run_status", "started", "ended", "gpus", "gpu_hours_allocated",
            "slurm_step", "research_repo_commit", "sglang_omni_commit", "runtime_profile", "model_revision", "run_dir",
            "included", "inclusion_reason"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows + (extra or []):
        lines.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    total = sum(r.get("gpu_hours_allocated") or 0 for r in rows + (extra or []))
    by_status: dict[str, float] = {}
    for r in rows + (extra or []):
        by_status[r.get("orchestrator_status") or "?"] = by_status.get(r.get("orchestrator_status") or "?", 0) + (r.get("gpu_hours_allocated") or 0)
    lines += ["", f"Allocated GPU-hours, all attempts: {total:.2f} ("
              + ", ".join(f"{k} {v:.2f}" for k, v in sorted(by_status.items())) + ")."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--state", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--inclusion", default=None)
    ap.add_argument("--extra", default=None, help="JSON list of rows for work run outside the orchestrator (manual GPU runs)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args()
    state = json.loads(Path(a.state).read_text())
    inclusion = json.loads(Path(a.inclusion).read_text()) if a.inclusion and Path(a.inclusion).exists() else {}
    extra = json.loads(Path(a.extra).read_text()) if a.extra and Path(a.extra).exists() else []
    rows = build(state, Path(a.data_root).expanduser() if a.data_root else None, inclusion)
    Path(a.out_json).write_text(json.dumps({"generated_by": "scripts/analysis/ppv_manifest.py", "orchestrated": rows,
                                            "manual": extra}, ensure_ascii=False, indent=1) + "\n")
    Path(a.out_md).write_text(to_markdown(rows, extra))
    print(f"{len(rows)} orchestrated + {len(extra)} manual rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
