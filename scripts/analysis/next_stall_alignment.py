#!/usr/bin/env python3
"""下一阶段研究 P3：把一次短暂停顿和服务日志、每进程 CPU、code2wav 的批对齐（只用已有数据）。

For one open-loop level of a run (phase + rate) and a window [lo, hi] in seconds after the level's first
scheduled send, the script reports per second:
  - log_lines: lines of the shared server log (every stage process writes to logs/server_s0.log);
  - log_gaps: stretches longer than --gap-s without any log line from any process;
  - cpu: system CPU and the CPU of the server and client process groups (raw/cpu_samples.jsonl, 1 s);
  - code2wav_batches: code2wav batches started (the stock "minicpm_code2wav_batch size=N" log line) and the
    requests in them;
  - talker: the median running batch and queue of the talker's decode log lines (T / R above
    --min-tokens-per-req, see next_talker_slots.py).
Client times are monotonic; the run's clock anchor (experiment_analysis.clock_anchor) maps them onto the
wall clock that the server log and the samplers use. The server log has millisecond wall-clock stamps in
the node's local time zone, so run this where that time zone applies (the cluster login node).

Usage (in the repo checkout next to the raw runs):
    python scripts/analysis/next_stall_alignment.py --data-root <PROJECT_DATA_ROOT> \
        --run 20261118_exp09_t2_k3_c3_block3_run01 --phase w3b_open --rate 21 --lo 45 --hi 62 \
        --out research/next_steps/data/p3_stall_t2_k3_c3_b3.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

LOG_STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) ")
C2W_BATCH = re.compile(r"minicpm_code2wav_batch size=(\d+)")
DECODE = re.compile(r"Decode batch, #running-req: (\d+), #token: (\d+),.*#queue-req: (\d+)")


def log_time(line: str) -> float | None:
    m = LOG_STAMP.match(line)
    if not m:
        return None
    return dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp() + int(m.group(2)) / 1e3


def scan_log(lines: Iterable[str], t0: float, lo: float, hi: float, min_tokens_per_req: float) -> dict[str, Any]:
    """Per-second log activity, code2wav batch starts and talker decode state in [t0 + lo, t0 + hi]."""
    per_s: dict[int, int] = {}
    c2w: dict[int, list[int]] = {}
    talker: dict[int, list[tuple[int, int]]] = {}
    stamps: list[float] = []
    for line in lines:
        t = log_time(line)
        if t is None or not (t0 + lo <= t <= t0 + hi):
            continue
        rel = t - t0
        stamps.append(rel)
        sec = int(rel)
        per_s[sec] = per_s.get(sec, 0) + 1
        m = C2W_BATCH.search(line)
        if m:
            c2w.setdefault(sec, []).append(int(m.group(1)))
            continue
        m = DECODE.search(line)
        if m:
            r, tok, q = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if r and tok / r > min_tokens_per_req:
                talker.setdefault(sec, []).append((r, q))
    stamps.sort()
    seconds = range(int(lo), int(hi) + 1)
    return {
        "log_lines": {s: per_s.get(s, 0) for s in seconds},
        "stamps": stamps,
        "code2wav_batches": {s: {"batches": len(c2w.get(s, [])), "requests": sum(c2w.get(s, []))} for s in seconds},
        "talker": {s: {"running_p50": statistics.median([r for r, _ in v]), "queue_max": max(q for _, q in v)}
                   for s, v in sorted(talker.items())},
    }


def log_gaps(stamps: list[float], gap_s: float) -> list[dict[str, float]]:
    return [{"at_s": round(a, 2), "gap_s": round(b - a, 2)} for a, b in zip(stamps, stamps[1:]) if b - a > gap_s]


def cpu_rows(samples: Iterable[dict[str, Any]], t0: float, lo: float, hi: float) -> list[dict[str, Any]]:
    out = []
    for c in samples:
        rel = c["t_wall_ns"] / 1e9 - t0
        if lo <= rel <= hi:
            groups = {k: round(v.get("cpu_percent", 0.0), 1) for k, v in (c.get("groups") or {}).items() if isinstance(v, dict)}
            out.append({"t_s": round(rel, 1), "system_cpu_percent": round(c.get("system_cpu_percent", 0.0), 1),
                        "groups": groups})
    return out


def main() -> int:
    from omni_research.analysis.experiment_analysis import clock_anchor, read_jsonl

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--phase", required=True)
    ap.add_argument("--rate", type=float, required=True, help="open-loop level (requests/s)")
    ap.add_argument("--lo", type=float, required=True)
    ap.add_argument("--hi", type=float, required=True)
    ap.add_argument("--gap-s", type=float, default=0.5)
    ap.add_argument("--min-tokens-per-req", type=float, default=200.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    exp = Path(a.data_root).expanduser() / "experiments" / a.run
    anchor = clock_anchor(exp)
    if not anchor:
        raise SystemExit("no clock anchor in this run")
    mono0, wall0 = anchor
    recs = [r for r in read_jsonl(exp / "raw" / f"requests_{a.phase}.jsonl") if not r.get("warmup")
            and r.get("scheduled_ns") and abs(float(r.get("rate_rps") or r.get("rate") or a.rate) - a.rate) < 1e-6]
    t0 = (min(r["scheduled_ns"] for r in recs) - mono0 + wall0) / 1e9
    with open(exp / "logs" / "server_s0.log", errors="ignore") as fh:
        scan = scan_log(fh, t0, a.lo, a.hi, a.min_tokens_per_req)
    out = {"generated_by": "scripts/analysis/next_stall_alignment.py", "run": a.run, "phase": a.phase,
           "rate_rps": a.rate, "window_s": [a.lo, a.hi], "requests_in_level": len(recs),
           "log_lines_per_s": scan["log_lines"], "log_gaps": log_gaps(scan["stamps"], a.gap_s),
           "code2wav_batches_per_s": scan["code2wav_batches"], "talker_per_s": scan["talker"],
           "cpu": cpu_rows(read_jsonl(exp / "raw" / "cpu_samples.jsonl"), t0, a.lo, a.hi)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: out[k] for k in ("run", "log_gaps")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
