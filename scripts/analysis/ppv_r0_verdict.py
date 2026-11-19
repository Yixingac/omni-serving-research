#!/usr/bin/env python3
"""R0 (full-duplex functional validation): the five pass conditions registered in
research/next_steps/presentation_validation/preregistration.md (R0–R2, R0), applied to one run.

    python -m omni_research.duplex.report --exp <run dir> --out r0_report.json      # metrics first
    python scripts/analysis/ppv_r0_verdict.py --exp <run dir> --report r0_report.json --out r0_verdict.json

Conditions (all must hold for 通过): 1 the server started and /v1/realtime/capabilities says native_full_duplex;
2 every session of the 1- and 2-session phases sent all planned packets and closed normally after the input drained;
3 no fatal error, no server restart, no phase error; 4 every session got at least one reply with audio, more than 0 s
of it, and most of its chunks above -60 dBFS; 5 at least 95 % of each session's planned units got sglang.unit.done
matched to the sent packets. The unit miss rate (real-time target <= 1 %), TTFA and the rest are reported beside the
verdict and do not change it; a pass with a miss rate above 1 % reads "能跑但不实时".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MEASURED_PHASES = ("duplex_s1", "duplex_s2")


def check_session(s: dict[str, Any]) -> dict[str, bool | str]:
    res = s.get("status")
    units = s.get("units") or {}
    replies = s.get("replies") or {}
    chunks = sum(r.get("chunks", 0) for r in s.get("reply_rows", [])) if s.get("reply_rows") else None
    silent = replies.get("silent_chunks", 0)
    planned = units.get("planned") or 0
    lags = (units.get("lag_ms") or {}).get("n", 0)
    return {
        "completed": res == "ok",
        "no_fatal": not any(e.get("fatal") for e in s.get("errors") or []),
        "audio": (replies.get("with_audio", 0) >= 1 and (replies.get("audio_s_total") or 0) > 0
                  and (chunks is None or chunks == 0 or silent < 0.5 * chunks)),
        "units_extractable": planned > 0 and (units.get("done", 0) >= 0.95 * planned) and lags >= 0.95 * planned,
    }


def verdict(report: dict[str, Any], capabilities: dict[str, Any] | None) -> dict[str, Any]:
    run = report["runs"][0]
    out: dict[str, Any] = {"run": run["run"], "conditions": {}, "phases": {}}
    body = (capabilities or {}).get("body") or {}
    out["conditions"]["1_server_started"] = bool(capabilities and capabilities.get("status") == 200
                                                and body.get("native_full_duplex") is True)
    completed, no_fatal, audio, units_ok = True, not run.get("server_restarts"), True, True
    for name in MEASURED_PHASES:
        ph = run["phases"].get(name)
        if not ph or ph.get("skipped") or ph.get("error") or not ph.get("duplex_sessions"):
            out["phases"][name] = {"present": False}
            completed = no_fatal = audio = units_ok = False
            continue
        checks = [check_session(s) for s in ph["duplex_sessions"]]
        out["phases"][name] = {"present": True, "sessions": len(checks), "checks": checks,
                               "unit_miss_rate": (ph.get("duplex") or {}).get("units", {}).get("miss_rate"),
                               "duplex": ph.get("duplex")}
        completed &= all(c["completed"] for c in checks)
        no_fatal &= all(c["no_fatal"] for c in checks)
        audio &= all(c["audio"] for c in checks)
        units_ok &= all(c["units_extractable"] for c in checks)
    out["conditions"].update({"2_sessions_completed": completed, "3_no_fatal_error": no_fatal,
                              "4_valid_audio": audio, "5_unit_latency_extractable": units_ok})
    passed = all(out["conditions"].values())
    miss = [p.get("unit_miss_rate") for p in out["phases"].values() if p.get("present")]
    realtime = passed and all(m is not None and m <= 0.01 for m in miss)
    out["verdict"] = "通过" if realtime else ("通过（能跑但不实时：单元错过率 > 1%）" if passed else "未通过")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", required=True)
    ap.add_argument("--report", required=True, help="output of python -m omni_research.duplex.report for this run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    exp = Path(a.exp).expanduser()
    caps = None
    for d in sorted((exp / "raw").glob("duplex_*")):
        if (d / "capabilities.json").exists():
            caps = json.loads((d / "capabilities.json").read_text())
            break
    out = verdict(json.loads(Path(a.report).read_text()), caps)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({"verdict": out["verdict"], "conditions": out["conditions"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
