#!/usr/bin/env python3
"""R4 (pre-presentation validation): do multimodal robot sessions put the thinker's KV cache under pressure?

    python scripts/analysis/ppv_r4_report.py --data-root <PROJECT_DATA_ROOT> --run <run dir> [--run ...] \
        --out research/next_steps/presentation_validation/results/r4_working_set.json

Inputs per run (sessions.robot_sessions plugin phases, runtime unicorn_ppv_kv with SGLANG_OMNI_EXP_KV=1):
- logs/server_s0.log: the patch's lines
    exp_kv_prefill pid=P gpu=G stage=thinker rid=R input_tokens=I cached_tokens=C extend_tokens=E
    exp_kv_pool pid=P gpu=G stage=thinker total_tokens=T available_tokens=A evictable_tokens=V protected_tokens=L
        running_reqs=.. waiting_reqs=.. evicted_tokens_total=X evict_calls_total=..
  A replica is the (pid, gpu) pair of a thinker scheduler.
- raw/requests_<phase>.jsonl and raw/sessions_<phase>.jsonl of each robot-session phase.

Definitions (registered in r3_r4_design.md):
- resident KV of a replica = total - available (running requests plus evictable cached prefixes);
- prefill tokens of a request = input_tokens - cached_tokens (the patch logs extend_tokens = the first chunk's new
  tokens from Req.extend_range, -1 if unset; it is kept for reference only);
- logical working set of a replica at time t = sum over live sessions whose latest request went to that replica of
  that request's prompt + completion tokens (what the session would need resident to continue without recompute);
- a request's reuse ratio = cached_tokens / input_tokens of the session's previous request;
- clean pair = previous request of the same session on the same replica with no eviction on that replica between the
  two prefills; the median reuse ratio of clean pairs per transition (plan->answer inside a turn, answer->next turn)
  is the reference;
- lost reuse of a pair = max(0, reference x previous input_tokens - cached_tokens), attributed to routing (other
  replica) or eviction (same replica, evictions in between); the shared floor is the median cached_tokens of
  session-opening requests (the system prompt every session shares).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

STAMP = r"^(?P<date>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(?P<ms>\d{3}) "
PREFILL = re.compile(STAMP + r".*exp_kv_prefill pid=(?P<pid>\d+) gpu=(?P<gpu>\S+) stage=(?P<stage>\S+) rid=(?P<rid>\S+) "
                     r"input_tokens=(?P<input>\d+) cached_tokens=(?P<cached>\d+) extend_tokens=(?P<extend>-?\d+)"
                     r"(?: evicted_tokens_total=(?P<evicted>\d+))?")
POOL = re.compile(STAMP + r".*exp_kv_pool pid=(?P<pid>\d+) gpu=(?P<gpu>\S+) stage=(?P<stage>\S+) total_tokens=(?P<total>\d+) "
                  r"available_tokens=(?P<available>\d+) evictable_tokens=(?P<evictable>\d+) protected_tokens=(?P<protected>\d+) "
                  r"running_reqs=(?P<running>\d+) waiting_reqs=(?P<waiting>\d+) evicted_tokens_total=(?P<evicted>\d+) "
                  r"evict_calls_total=(?P<calls>\d+)")
PRESSURE_RATIO = 0.8  # registered (robotics_experiment_plan.md R4), reported but not decisive on its own
LOSS_SHARE = 0.10  # lost reuse >= 10 % of the prefill tokens of the level
SLOWER = 1.2  # first-token time of the affected requests >= 1.2 x that of clean pairs


def _ts(m: re.Match, utc_offset_s: float) -> float:
    naive = dt.datetime.strptime(m.group("date"), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()
    return naive - utc_offset_s + int(m.group("ms")) / 1e3


def parse_log(lines: Iterable[str], utc_offset_s: float = 0.0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """exp_kv lines as dicts with a UTC epoch timestamp ``ts`` and a replica key ``replica`` = "pid:gpu"."""
    prefills, pools = [], []
    for line in lines:
        if "exp_kv_" not in line:
            continue
        m = PREFILL.match(line)
        if m:
            prefills.append({"ts": _ts(m, utc_offset_s), "replica": f"{m['pid']}:{m['gpu']}", "rid": m["rid"],
                             "input_tokens": int(m["input"]), "cached_tokens": int(m["cached"]), "extend_tokens": int(m["extend"]),
                             "evicted_total": int(m["evicted"]) if m["evicted"] is not None else None})
            continue
        m = POOL.match(line)
        if m:
            pools.append({"ts": _ts(m, utc_offset_s), "replica": f"{m['pid']}:{m['gpu']}", **{k: int(m[k]) for k in (
                "total", "available", "evictable", "protected", "running", "waiting", "evicted", "calls")}})
    return prefills, pools


def evicted_between(pools: list[dict[str, Any]], replica: str, t0: float, t1: float) -> int | None:
    """Tokens the replica evicted between two times, from the cumulative counter (None without pool lines around)."""
    rows = [p for p in pools if p["replica"] == replica]
    before = [p for p in rows if p["ts"] <= t0]
    after = [p for p in rows if p["ts"] >= t1]
    if not before or not after:
        return None
    return after[0]["evicted"] - before[-1]["evicted"]


def transition(prev: dict[str, Any], cur: dict[str, Any]) -> str:
    return "within_turn" if prev["turn"] == cur["turn"] else "next_turn"


def attribute(requests: list[dict[str, Any]], prefills: list[dict[str, Any]], pools: list[dict[str, Any]]) -> dict[str, Any]:
    """Per request: replica, cached/input, previous-request relation; then the reference reuse and lost reuse."""
    by_rid = {}
    for p in prefills:
        by_rid.setdefault(p["rid"], p)
    rows = []
    sessions: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in requests:
        p = by_rid.get(r["request_id"])
        if p is None:
            continue
        row = {"request_id": r["request_id"], "session": r["session"], "turn": r["turn"], "step": r["step"],
               "sent_ns": r.get("sent_ns"), "ttft_s": (r["first_text_ns"] - r["sent_ns"]) / 1e9
               if r.get("first_text_ns") and r.get("sent_ns") else None, **p}
        sessions[r["session"]].append(row)
        rows.append(row)
    openers, pairs = [], []
    for s, rs in sessions.items():
        rs.sort(key=lambda x: x["ts"])
        openers.append(rs[0]["cached_tokens"])
        for prev, cur in zip(rs, rs[1:]):
            same = prev["replica"] == cur["replica"]
            if not same:
                ev = None
            elif prev.get("evicted_total") is not None and cur.get("evicted_total") is not None:
                # counters logged with each prefill: exact up to the evictions of cur's own batch (an upper bound)
                ev = cur["evicted_total"] - prev["evicted_total"]
            else:
                ev = evicted_between(pools, cur["replica"], prev["ts"], cur["ts"])
            pairs.append({"cur": cur, "prev": prev, "same_replica": same, "evicted_between": ev,
                          "transition": transition(prev, cur),
                          "reuse_ratio": cur["cached_tokens"] / prev["input_tokens"] if prev["input_tokens"] else None})
    floor = statistics.median(openers) if openers else 0
    ref = {}
    for kind in ("within_turn", "next_turn"):
        clean = [p["reuse_ratio"] for p in pairs if p["transition"] == kind and p["same_replica"]
                 and p["evicted_between"] == 0 and p["reuse_ratio"] is not None]
        ref[kind] = {"median": statistics.median(clean) if clean else None, "n": len(clean)}
    lost = {"routing": 0, "eviction": 0, "unexplained_same_replica": 0}
    counts = {"pairs": len(pairs), "cross_replica": 0, "same_replica_evicted": 0, "clean": 0, "unknown_eviction": 0}
    for p in pairs:
        r = ref[p["transition"]]["median"]
        expected = r * p["prev"]["input_tokens"] if r is not None else None
        loss = max(0.0, expected - p["cur"]["cached_tokens"]) if expected is not None else 0.0
        p["lost_tokens"] = loss
        if not p["same_replica"]:
            counts["cross_replica"] += 1
            lost["routing"] += loss
            p["cause"] = "routing"
        elif p["evicted_between"] is None:
            counts["unknown_eviction"] += 1
            p["cause"] = "unknown"
        elif p["evicted_between"] > 0:
            counts["same_replica_evicted"] += 1
            lost["eviction"] += loss
            p["cause"] = "eviction"
        else:
            counts["clean"] += 1
            lost["unexplained_same_replica"] += loss
            p["cause"] = "clean"
    ttft_by_cause: dict[str, list[float]] = defaultdict(list)
    for p in pairs:
        if p["cur"]["ttft_s"] is not None:
            ttft_by_cause[p["cause"]].append(p["cur"]["ttft_s"])
    # tokens each request had to compute (input - cached), independent of how SGLang chunked the prefill
    prefill_tokens = sum(x["input_tokens"] - x["cached_tokens"] for x in rows)
    return {"requests_joined": len(rows), "shared_floor_tokens": floor, "reference_reuse": ref, "counts": counts,
            "lost_reuse_tokens": {k: round(v) for k, v in lost.items()},
            "prefill_tokens": prefill_tokens, "cached_tokens": sum(x["cached_tokens"] for x in rows),
            "lost_share_of_prefill": {k: (v / prefill_tokens if prefill_tokens else None) for k, v in lost.items()},
            "ttft_s_by_cause": {k: {"p50": statistics.median(v), "n": len(v)} for k, v in ttft_by_cause.items()}}


def working_set(requests: list[dict[str, Any]], sessions: list[dict[str, Any]], prefills: list[dict[str, Any]],
                step_s: float = 1.0) -> dict[str, list[tuple[float, int]]]:
    """Per replica, (time, logical working set in tokens) every step_s from each live session's latest request."""
    by_rid = {p["rid"]: p for p in prefills}
    events = []  # (ts, session, replica, tokens)
    for r in requests:
        p = by_rid.get(r["request_id"])
        if p is None:
            continue
        events.append((p["ts"], r["session"], p["replica"], p["input_tokens"] + int(r.get("completion_tokens") or 0)))
    if not events:
        return {}
    events.sort()
    end_ts = {}
    for s in sessions:
        last = [e[0] for e in events if e[1] == s["session"]]
        if last:
            end_ts[s["session"]] = max(last)
    replicas = sorted({e[2] for e in events})
    out: dict[str, list[tuple[float, int]]] = {rep: [] for rep in replicas}
    t, i, latest = events[0][0], 0, {}
    stop = events[-1][0]
    while t <= stop + 1e-9:
        while i < len(events) and events[i][0] <= t:
            latest[events[i][1]] = (events[i][2], events[i][3])
            i += 1
        for rep in replicas:
            out[rep].append((round(t, 3), sum(tok for s, (rr, tok) in latest.items() if rr == rep and end_ts.get(s, 0) >= t)))
        t += step_s
    return out


def pool_summary(pools: list[dict[str, Any]], t0: float, t1: float) -> dict[str, Any]:
    out = {}
    for rep in sorted({p["replica"] for p in pools}):
        rows = [p for p in pools if p["replica"] == rep and t0 <= p["ts"] <= t1]
        if not rows:
            continue
        total = rows[0]["total"]
        before = [p for p in pools if p["replica"] == rep and p["ts"] < t0]
        start = before[-1] if before else rows[0]
        out[rep] = {"capacity_tokens": total,
                    "resident_peak": max(p["total"] - p["available"] for p in rows),
                    "resident_peak_ratio": max(p["total"] - p["available"] for p in rows) / total if total else None,
                    "held_by_running_peak": max(p["total"] - p["available"] - p["evictable"] for p in rows),
                    "evicted_tokens": rows[-1]["evicted"] - start["evicted"],
                    "evictable_at_start": start["evictable"],
                    "waiting_peak": max(p["waiting"] for p in rows)}
    return out


def phase_report(exp: Path, phase: str, prefills: list[dict[str, Any]], pools: list[dict[str, Any]]) -> dict[str, Any]:
    from omni_research.analysis.experiment_analysis import clock_anchor, read_jsonl

    reqs = read_jsonl(exp / "raw" / f"requests_{phase}.jsonl")
    sess = read_jsonl(exp / "raw" / f"sessions_{phase}.jsonl")
    anchor = clock_anchor(exp)
    off = (anchor[1] - anchor[0]) / 1e9 if anchor else None
    sent = [r["sent_ns"] for r in reqs if r.get("sent_ns")]
    done = [r["done_ns"] for r in reqs if r.get("done_ns")]
    if not sent or off is None:
        return {"phase": phase, "error": "no requests or no clock anchor"}
    t0, t1 = min(sent) / 1e9 + off - 1.0, max(done or sent) / 1e9 + off + 1.0
    ids = {r["request_id"] for r in reqs}
    pf = [p for p in prefills if p["rid"] in ids]
    ws = working_set(reqs, sess, pf)
    pool = pool_summary(pools, t0, t1)
    ws_peak = {rep: max(v for _, v in series) for rep, series in ws.items()}
    ttft = [(r["first_text_ns"] - r["sent_ns"]) / 1e9 for r in reqs if r.get("first_text_ns") and r.get("sent_ns")]
    return {"phase": phase, "sessions": len(sess), "requests": len(reqs), "errors": sum(r.get("status") != "ok" for r in reqs),
            "window_s": round(t1 - t0, 1), "pool": pool,
            "logical_working_set_peak": ws_peak,
            "logical_ratio_peak": {rep: ws_peak[rep] / pool[rep]["capacity_tokens"] for rep in ws_peak if rep in pool},
            "attribution": attribute(reqs, pf, pools),
            "ttft_s": {"p50": statistics.median(ttft) if ttft else None,
                       "p95": sorted(ttft)[int(0.95 * len(ttft))] if len(ttft) >= 20 else None, "n": len(ttft)},
            "session_s_p50": statistics.median([(s["ended_ns"] - s["started_ns"]) / 1e9 for s in sess]) if sess else None}


def smoke_check(phases: dict[str, list[dict[str, Any]]], prefills: list[dict[str, Any]]) -> dict[str, Any]:
    """Per smoke phase: share of requests found in exp_kv_prefill, median server input tokens per turn, and the media
    tokens of one turn (image + speech) as the difference of each session's first request with and without media."""
    by_rid = {}
    for p in prefills:
        by_rid.setdefault(p["rid"], p)
    out: dict[str, Any] = {}
    first: dict[str, dict[int, int]] = {}
    for name, reqs in phases.items():
        joined = [r for r in reqs if r["request_id"] in by_rid]
        turns: dict[int, list[int]] = defaultdict(list)
        for r in joined:
            turns[r["turn"]].append(by_rid[r["request_id"]]["input_tokens"])
        out[name] = {"requests": len(reqs), "joined_share": len(joined) / len(reqs) if reqs else None,
                     "input_tokens_p50_by_turn": {t: statistics.median(v) for t, v in sorted(turns.items())},
                     "max_input_tokens": max((by_rid[r["request_id"]]["input_tokens"] for r in joined), default=None)}
        first[name] = {r["session"]: by_rid[r["request_id"]]["input_tokens"] for r in joined
                       if r["turn"] == 0 and r["step"] == ("plan" if r.get("tool_turn") else "answer")}
    media = [n for n in phases if not n.endswith("_textonly") and "_media" not in n]
    text = [n for n in phases if n.endswith("_textonly")]
    if media and text:
        a, b = first[media[0]], first[text[0]]
        deltas = [a[s] - b[s] for s in a if s in b]
        out["media_tokens_per_turn"] = {"p50": statistics.median(deltas) if deltas else None, "values": deltas}
    return out


def verdict(levels: list[dict[str, Any]]) -> dict[str, Any]:
    """Registered rule (r3_r4_design.md): pressure needs both a high ratio and own-session losses that cost time."""
    pressure = []
    for lv in levels:
        ratio = max((lv.get("logical_ratio_peak") or {}).values(), default=0.0)
        resident = max((p.get("resident_peak_ratio") or 0.0 for p in (lv.get("pool") or {}).values()), default=0.0)
        att = lv.get("attribution") or {}
        share = att.get("lost_share_of_prefill") or {}
        ev_loss = share.get("eviction") or 0.0
        rt_loss = share.get("routing") or 0.0
        by = att.get("ttft_s_by_cause") or {}
        clean = (by.get("clean") or {}).get("p50")
        slow_ev = clean and (by.get("eviction") or {}).get("p50") and by["eviction"]["p50"] >= SLOWER * clean
        slow_rt = clean and (by.get("routing") or {}).get("p50") and by["routing"]["p50"] >= SLOWER * clean
        pressure.append({"phase": lv["phase"], "sessions": lv.get("sessions"), "logical_ratio_peak": ratio,
                         "resident_ratio_peak": resident, "over_registered_ratio": ratio >= PRESSURE_RATIO,
                         "eviction_loss_share": ev_loss, "routing_loss_share": rt_loss,
                         # a ratio alone decides nothing: losses must be large and cost first-token time
                         "eviction_pressure": bool(ev_loss >= LOSS_SHARE and slow_ev),
                         "routing_cost": bool(rt_loss >= LOSS_SHARE and slow_rt)})
    if any(p["eviction_pressure"] for p in pressure):
        text = "eviction pressure observed: design the session-affinity / cost-aware eviction A/B"
    elif any(p["routing_cost"] for p in pressure):
        text = "no eviction pressure, but cross-replica routing loses reuse that costs TTFT: next is a session-affinity A/B"
    else:
        text = "当前实验条件下，尚无证据支持引入更复杂的 Agent-aware Cache Management。"
    return {"levels": pressure, "conclusion": text}


def structural_gap(levels: list[dict[str, Any]]) -> dict[str, Any]:
    """Media-history contrast: reference next-turn reuse with re-sent media vs text history at the same session count."""
    out = {}
    for lv in levels:
        if "_media" not in lv["phase"]:
            continue
        twin = next((x for x in levels if "_media" not in x["phase"] and x.get("sessions") == lv.get("sessions")), None)
        if twin is None:
            continue
        ref = lambda x: ((x.get("attribution") or {}).get("reference_reuse") or {}).get("next_turn", {}).get("median")  # noqa: E731
        out[lv["phase"]] = {"vs": twin["phase"], "next_turn_reuse_text": ref(twin), "next_turn_reuse_media": ref(lv),
                            "ttft_p50_text": (twin.get("ttft_s") or {}).get("p50"), "ttft_p50_media": (lv.get("ttft_s") or {}).get("p50")}
    return out


def main() -> int:
    from omni_research.analysis.experiment_analysis import log_utc_offset

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out: dict[str, Any] = {"generated_by": "scripts/analysis/ppv_r4_report.py", "runs": {}}
    for run in a.run:
        exp = Path(a.data_root).expanduser() / "experiments" / run
        off = log_utc_offset(exp)
        with open(exp / "logs" / "server_s0.log", errors="ignore") as fh:
            prefills, pools = parse_log(fh, off.total_seconds() if off is not None else 0.0)
        from omni_research.analysis.experiment_analysis import read_jsonl

        res = json.loads((exp / "result.json").read_text())
        phases = [p["phase"] for p in res.get("phases", []) if (exp / "raw" / f"sessions_{p['phase']}.jsonl").exists()]
        if all("smoke" in ph for ph in phases):
            out["runs"][run] = {"smoke": smoke_check({ph: read_jsonl(exp / "raw" / f"requests_{ph}.jsonl") for ph in phases},
                                                     prefills), "replicas": sorted({p["replica"] for p in pools})}
            print(run, json.dumps(out["runs"][run], ensure_ascii=False)[:2000])
            continue
        levels = [phase_report(exp, ph, prefills, pools) for ph in phases]
        out["runs"][run] = {"levels": levels, "verdict": verdict([lv for lv in levels if "_media" not in lv["phase"]]),
                            "structural": structural_gap(levels), "utc_offset_known": off is not None,
                            "replicas": sorted({p["replica"] for p in pools})}
        print(run, json.dumps(out["runs"][run]["verdict"], ensure_ascii=False))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
