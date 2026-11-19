#!/usr/bin/env python3
"""P2 gate: is the side-stream hidden-state copy bit-exact in every finished request of the verify run?

    python scripts/analysis/ppv_p2_report.py --data-root <results root> --run 20261119_ppv_p2_verify_run01 \
        --out research/next_steps/presentation_validation/results/p2_verify.json

Registered gate (research/next_steps/p0_p3_experiment_plan.md P2; details in presentation_validation/p1_p2_p3_design.md):
pass only if
1. every exp_hidden_verify event has is_equal_reference, is_equal_resync, is_rows_expected and is_aligned true;
2. every speech request that finished ok in a check phase has exactly one verify event (a request without one is
   unverified, which is not a pass);
3. no "exp_hidden_verify mismatch" warning in the server log (this also covers the smoke phase, which has no events);
4. no client error other than the deliberate client timeouts of chk_timeout, and no server restart.
If a reference copy itself has a row count other than the expected one (reference_rows != expected_rows), the expected
count was defined wrongly: that is reported as an erratum and rule 1 then uses reference_rows == rows for the row
check; it never turns a failed bit comparison into a pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CHECKS = ("is_equal_reference", "is_equal_resync", "is_rows_expected", "is_aligned")
ABORT_PHASE = "chk_timeout"


def gate(events: dict[str, list[dict[str, Any]]], clients: dict[str, list[dict[str, Any]]], log_lines: list[str],
         restarts: list[Any]) -> dict[str, Any]:
    """events / clients: phase -> exp_hidden_verify events (request_id, metadata) / client records."""
    reasons: list[str] = []
    failed = Counter()
    mismatches = []
    erratum = []
    per_phase = {}
    for phase, evs in events.items():
        seen = Counter(e["request_id"] for e in evs)
        dup = [r for r, n in seen.items() if n > 1]
        bad = []
        for e in evs:
            md = e["metadata"]
            if md.get("reference_rows") not in (None, -1) and md.get("reference_rows") != md.get("expected_rows"):
                erratum.append({"phase": phase, "request_id": e["request_id"], "reference_rows": md.get("reference_rows"),
                                "expected_rows": md.get("expected_rows")})
                rows_ok = md.get("reference_rows") == md.get("rows")
            else:
                rows_ok = bool(md.get("is_rows_expected"))
            ok = {k: bool(md.get(k)) for k in CHECKS} | {"is_rows_expected": rows_ok}
            if not all(ok.values()):
                bad.append(e["request_id"])
                failed.update(k for k, v in ok.items() if not v)
                mismatches.append({"phase": phase, "request_id": e["request_id"], **{k: md.get(k) for k in (
                    "first_bad_row", "rows", "reference_rows", "expected_rows", "side_sha256", "reference_sha256", "resync_sha256",
                    "step_index", "step_batch_size", "finished_in_step", "dtype", "reference_dtype")}, "failed": [k for k, v in ok.items() if not v]})
        speech_ok = {r["request_id"] for r in clients.get(phase, []) if r.get("status") == "ok" and (r.get("output_audio_s") or 0) > 0
                     and not r.get("warmup")}
        unverified = sorted(speech_ok - set(seen))
        per_phase[phase] = {"verify_events": len(evs), "speech_ok": len(speech_ok), "unverified": len(unverified),
                            "duplicates": len(dup), "failed": len(bad)}
        if unverified:
            reasons.append(f"{phase}: {len(unverified)} speech requests finished without a verify event")
        if dup:
            reasons.append(f"{phase}: {len(dup)} requests with more than one verify event")
    errors = {p: sum(1 for r in recs if r.get("status") not in ("ok",) and not (p == ABORT_PHASE and r.get("status") == "timeout"))
              for p, recs in clients.items()}
    for p, n in errors.items():
        if n:
            reasons.append(f"{p}: {n} client errors")
    warnings = [ln for ln in log_lines if "exp_hidden_verify mismatch" in ln]
    if mismatches:
        reasons.append(f"{len(mismatches)} requests failed a check: {dict(failed)}")
    if warnings:
        reasons.append(f"{len(warnings)} mismatch warnings in the server log")
    if restarts:
        reasons.append(f"{len(restarts)} server restarts")
    if not any(events.values()):
        reasons.append("no verify events at all")
    return {"gate": "通过" if not reasons else "不通过", "reasons": reasons, "per_phase": per_phase,
            "requests_verified": sum(len(v) for v in events.values()), "mismatches": mismatches[:50],
            "mismatch_count": len(mismatches), "log_warnings": len(warnings), "erratum_expected_rows": erratum[:20],
            "erratum_count": len(erratum)}


def main() -> int:
    from exp06_report import read_events

    from omni_research.analysis.experiment_analysis import read_jsonl

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    exp = Path(a.data_root).expanduser() / "experiments" / a.run
    res = json.loads((exp / "result.json").read_text())
    events, clients = {}, {}
    for ph in res.get("phases", []):
        name = ph["phase"]
        if not name.startswith("chk_"):
            continue
        clients[name] = read_jsonl(exp / "raw" / f"requests_{name}.jsonl") if (exp / "raw" / f"requests_{name}.jsonl").exists() else []
        ev_dir = exp / "raw" / "server_events" / name
        events[name] = [{"request_id": e.get("request_id"), "metadata": e.get("metadata") or {}}
                        for e in (read_events(ev_dir) if ev_dir.exists() else []) if e.get("event_name") == "exp_hidden_verify"]
    log = (exp / "logs" / "server_s0.log").read_text(errors="ignore").splitlines() if (exp / "logs" / "server_s0.log").exists() else []
    out = {"generated_by": "scripts/analysis/ppv_p2_report.py", "run": a.run,
           **gate(events, clients, log, res.get("server_restarts") or [])}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: out[k] for k in ("gate", "reasons", "requests_verified", "mismatch_count")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
