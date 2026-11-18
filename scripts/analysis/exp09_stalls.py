#!/usr/bin/env python3
"""实验 9 开环档位里的短暂停顿（看到全部数据后才查的，描述性，不进入任何判定）。

    python scripts/analysis/exp09_stalls.py --data-root <results root> --run 20261118_exp09_t2_k6_shared_block1_run01 ... \
        --out research/01_性能瓶颈/实验09_最佳GPU分工/results/stalls.json

A request has a front stall when its response headers arrive more than 1 s after it was sent, and an audio stall when
the time from its first text to its first audio is more than 5 s above the median of its level. For every level with a
stalled request, the per-second nvidia-smi busy of each GPU role is kept from 3 s before the first stalled request was
sent to 3 s after the last one finished, with the seconds in which every GPU of a role sampled 0 % while requests were
still being sent (after the last send the pipeline drains and idles anyway). Times are seconds
after the level's first scheduled send; the client's clock anchor puts its monotonic times on the sampler's clock.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slurm"))

FRONT_S = 1.0
AUDIO_EXCESS_S = 5.0
PAD_S = 3.0
OPEN_PHASES = ("w3b_open", "w3l_open")


def level_stalls(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Counts and send times of the stalled requests of one open-loop level, and those requests."""
    ok = [r for r in records if r.get("status") == "ok" and not r.get("warmup")]
    if not ok:
        return None
    t0 = min(r["scheduled_ns"] for r in ok)

    def gap(r: dict[str, Any]) -> float | None:
        return (r["first_audio_ns"] - r["first_text_ns"]) / 1e9 if r.get("first_audio_ns") and r.get("first_text_ns") else None

    gaps = [g for g in map(gap, ok) if g is not None]
    med = statistics.median(gaps) if gaps else None
    front, audio, stalled = [], [], []
    for r in ok:
        t = round((r["sent_ns"] - t0) / 1e9, 1)
        is_front = bool(r.get("headers_ns")) and (r["headers_ns"] - r["sent_ns"]) / 1e9 > FRONT_S
        g = gap(r)
        is_audio = med is not None and g is not None and g - med > AUDIO_EXCESS_S
        if is_front:
            front.append(t)
        if is_audio:
            audio.append(t)
        if is_front or is_audio:
            stalled.append(r)
    summary = {"requests": len(ok), "front_stalls": len(front), "audio_stalls": len(audio), "stalled_requests": len(stalled),
               "front_at_s": front, "audio_at_s": audio, "first_audio_gap_median_s": med,
               "lateness_max_s": max(r.get("lateness_s") or 0.0 for r in ok)}
    return summary, stalled


def gpu_seconds(util: dict[int, list[tuple[float, float, float]]], roles: dict[str, list[int]], start_s: float,
                end_s: float, origin_s: float) -> list[dict[str, Any]]:
    """Per whole second in [start, end): mean sampled busy % of each role's GPUs, and the roles whose GPUs all sampled
    0 % in that second. Seconds are counted from origin (unix seconds)."""
    by: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for role, gpus in roles.items():
        for g in gpus:
            for t, busy, _ in util.get(g, []):
                if start_s <= t < end_s:
                    by[math.floor(t - origin_s)][role].append(busy)
    return [{"t_s": k, "busy": {role: round(sum(v) / len(v), 1) for role, v in by[k].items()},
             "idle_roles": [role for role, v in by[k].items() if max(v) == 0]} for k in sorted(by)]


def idle_while_sending(secs: list[dict[str, Any]], roles: list[str], send_end_s: float) -> dict[str, int]:
    """Seconds in which a role's GPUs all sampled 0 % while requests were still being sent (after the last send the
    pipeline drains and idles anyway)."""
    return {role: sum(role in s["idle_roles"] for s in secs if s["t_s"] < send_end_s) for role in roles}


def scan_run(exp: Path) -> dict[str, Any]:
    from exp01_gpu_timeline import load_util
    from exp09_report import ARM_OF, RUN_RE, gpu_roles
    from omni_research.analysis.experiment_analysis import clock_anchor, read_jsonl

    res = json.loads((exp / "result.json").read_text())
    m = RUN_RE.match(res["experiment_id"])
    name = m.group(1)
    util, anchor, roles = load_util(exp), clock_anchor(exp), gpu_roles(name)
    levels, stalls = [], []
    for phase in OPEN_PHASES:
        path = exp / "raw" / f"requests_{phase}.jsonl"
        if not path.exists():
            continue
        recs = [r for r in read_jsonl(path) if not r.get("crash_affected")]
        for rate in sorted({r["rate_rps"] for r in recs if r.get("rate_rps") is not None}):
            found = level_stalls([r for r in recs if r.get("rate_rps") == rate])
            if found is None:
                continue
            summary, stalled = found
            row = {"phase": phase, "rate_rps": rate, **summary}
            levels.append({k: row[k] for k in ("phase", "rate_rps", "requests", "front_stalls", "audio_stalls",
                                               "stalled_requests", "lateness_max_s")})
            if stalled and anchor:
                mono0, wall0 = anchor
                wall = lambda ns: (ns - mono0 + wall0) / 1e9  # noqa: E731
                sched = [r["scheduled_ns"] for r in recs if r.get("rate_rps") == rate and not r.get("warmup")]
                t0, send_end = wall(min(sched)), wall(max(sched))
                start, end = wall(min(r["sent_ns"] for r in stalled)) - PAD_S, wall(max(r["done_ns"] for r in stalled)) + PAD_S
                secs = gpu_seconds(util, roles, start, end, t0)
                row.update(window_s=[round(start - t0, 1), round(end - t0, 1)], send_end_s=round(send_end - t0, 1), gpu=secs,
                           idle_seconds_while_sending=idle_while_sending(secs, list(roles), send_end - t0))
                stalls.append(row)
    return {"run": exp.name, "placement": name, "arm": ARM_OF[name], "block": int(m.group(2)) if m.group(2) else 0,
            "roles": roles, "levels": levels, "stalls": stalls}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    runs = [scan_run(root / r) for r in a.run]
    levels = [lv for r in runs for lv in r["levels"]]
    out = {"generated_by": "scripts/analysis/exp09_stalls.py",
           "rule": {"front_stall_s": FRONT_S, "audio_excess_s": AUDIO_EXCESS_S, "window_pad_s": PAD_S},
           "levels_scanned": len(levels), "levels_with_stalls": sum(lv["stalled_requests"] > 0 for lv in levels),
           "levels_with_10_or_more": sum(lv["stalled_requests"] >= 10 for lv in levels), "runs": runs}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(f"{out['levels_scanned']} levels, {out['levels_with_stalls']} with stalls, {out['levels_with_10_or_more']} with >= 10 -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
