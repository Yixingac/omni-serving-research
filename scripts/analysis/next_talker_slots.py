#!/usr/bin/env python3
"""下一阶段研究 P0：talker 的“同时处理名额”是不是上限——从 SGLang 调度日志读在跑请求数、排队数和每步速度。

SGLang's scheduler prints one line per decode logging interval:
    <date> <time>,<ms> [INFO] ...metrics_reporter: Decode batch, #running-req: R, #token: T, token usage: U,
    cuda graph: True|False, gen throughput (token/s): G, #queue-req: Q
In a MiniCPM-o speech server the thinker and the talker processes write to the same server log and the
lines do not name the stage. A talker request holds the thinker's hidden-state rows plus every generated
audio code (about 25 codes per second of speech), a thinker request holds a short prompt plus at most
max_tokens text tokens, so T / R separates the two: lines with T / R above --min-tokens-per-req (default
200) are counted as talker lines. This is a heuristic; check it on each new workload (W3b and W3L thinker
requests stay below 200 tokens, talker requests of a few seconds of speech are above it).

Per window (a phase of analysis/phases.json, whose started/ended are wall-clock seconds; with
--last-level the window is the last level of the phase, ended - wall_s), it reports the share of talker
lines at the admission cap with requests waiting, the queue length, and the CUDA-graph share. With
--curve it also reports the median decode throughput per running batch size over all talker lines of
the log, and the step time R / G that follows from it.

Usage (where the raw runs are):
    python scripts/analysis/next_talker_slots.py --data-root <PROJECT_DATA_ROOT> --cap 32 \
        --run 20261117_exp09_t2_k1_c5_block1_run01 --phase w3b_closed_A --last-level [--run ...] \
        --curve --out research/next_steps/results/p0_talker_slots_exp09.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
from pathlib import Path
from typing import Any, Iterable

LINE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(?P<ms>\d{3}) .*Decode batch, #running-req: (?P<run>\d+), "
    r"#token: (?P<tok>\d+),.*cuda graph: (?P<graph>True|False), gen throughput \(token/s\): (?P<gen>[0-9.]+), "
    r"#queue-req: (?P<queue>\d+)"
)


def parse_decode_lines(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Decode-batch lines as dicts; the timestamp is the log's local wall clock in epoch seconds."""
    out = []
    for line in lines:
        m = LINE.match(line)
        if not m:
            continue
        ts = dt.datetime.strptime(m.group("date"), "%Y-%m-%d %H:%M:%S").timestamp() + int(m.group("ms")) / 1e3
        out.append({"ts": ts, "running": int(m.group("run")), "tokens": int(m.group("tok")),
                    "graph": m.group("graph") == "True", "gen_tok_s": float(m.group("gen")),
                    "queue": int(m.group("queue"))})
    return out


def talker_lines(rows: list[dict[str, Any]], min_tokens_per_req: float) -> list[dict[str, Any]]:
    return [r for r in rows if r["running"] > 0 and r["tokens"] / r["running"] > min_tokens_per_req]


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def window_stats(rows: list[dict[str, Any]], cap: int) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"lines": 0}
    at_cap = [r for r in rows if r["running"] >= cap]
    return {
        "lines": n,
        "at_cap_share": len(at_cap) / n,
        "at_cap_with_queue_share": sum(1 for r in at_cap if r["queue"] > 0) / n,
        "above_cap_lines": sum(1 for r in rows if r["running"] > cap),
        "running_p50": _pct([r["running"] for r in rows], 0.5),
        "queue_p50": _pct([r["queue"] for r in rows], 0.5),
        "queue_p90": _pct([r["queue"] for r in rows], 0.9),
        "queue_max": max(r["queue"] for r in rows),
        "graph_share": sum(1 for r in rows if r["graph"]) / n,
    }


def decode_curve(rows: list[dict[str, Any]], min_lines: int = 20, min_gen: float = 50.0) -> list[dict[str, Any]]:
    """Median decode throughput per running batch size; lines with G < min_gen cover idle gaps and are dropped."""
    by: dict[int, list[float]] = {}
    for r in rows:
        if r["gen_tok_s"] >= min_gen:
            by.setdefault(r["running"], []).append(r["gen_tok_s"])
    curve = []
    for bs in sorted(by):
        if len(by[bs]) >= min_lines:
            med = statistics.median(by[bs])
            curve.append({"batch": bs, "lines": len(by[bs]), "gen_tok_s_median": med, "step_ms": 1000.0 * bs / med})
    return curve


def phase_window(phases: list[dict[str, Any]], phase: str, last_level: bool) -> tuple[float, float]:
    for p in phases:
        if p.get("phase") == phase:
            if last_level:
                lv = p["levels"][-1]
                return p["ended"] - lv["wall_s"], p["ended"]
            return p["started"], p["ended"]
    raise KeyError(f"phase {phase} not in phases.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--phase", required=True)
    ap.add_argument("--last-level", action="store_true", help="window = last level of the phase (ended - wall_s)")
    ap.add_argument("--cap", type=int, required=True, help="the talker's max_running_requests in these runs")
    ap.add_argument("--min-tokens-per-req", type=float, default=200.0)
    ap.add_argument("--curve", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    out: dict[str, Any] = {"generated_by": "scripts/analysis/next_talker_slots.py", "phase": a.phase,
                           "last_level": a.last_level, "cap": a.cap, "min_tokens_per_req": a.min_tokens_per_req,
                           "runs": {}}
    pooled: list[dict[str, Any]] = []
    for run in a.run:
        exp = root / run
        phases = json.loads((exp / "analysis" / "phases.json").read_text())
        t0, t1 = phase_window(phases, a.phase, a.last_level)
        with open(exp / "logs" / "server_s0.log", errors="ignore") as fh:
            rows = talker_lines(parse_decode_lines(fh), a.min_tokens_per_req)
        win = [r for r in rows if t0 <= r["ts"] <= t1]
        out["runs"][run] = {"window_s": t1 - t0, **window_stats(win, a.cap)}
        pooled.extend(rows)
        print(run, json.dumps(out["runs"][run]))
    if a.curve:
        out["decode_curve_all_lines"] = decode_curve(pooled)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
