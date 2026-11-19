#!/usr/bin/env python3
"""R1 (single-session critical path of the full-duplex path): descriptive report, no verdict.

    # on the cluster, with the sglang-omni runtime's python (the server-side stage timing needs sglang_omni):
    PYTHONPATH=src <runtime python> scripts/analysis/ppv_r1_report.py --data-root <PROJECT_DATA_ROOT> --out r1.json [--md r1.md]
    # anywhere else the same command works; the server-side part is then marked unavailable, never zero.

Rules: research/next_steps/presentation_validation/preregistration.md, R1 and "R1/R2 分析口径补充" items 1-8:
A phase (no request events) is the primary client-side number, B beside it, the 30 s smoke is not analysed;
pooled = the three blocks' turns/units/replies together, with each block's P50 and a t interval of the block
P50s (df 2); the latest ok run of a block is used and the others are listed as excluded; TTFA of an answered turn
= (a) wait for the unit holding the utterance's end + (b) k further units + (c) the speaking unit's service time
(metrics.ttfa_parts); stage shares from the B phase's request events over speaking units (primary) and all units;
the stage numbers carry "event recording visibly adds latency" when the pooled B unit-lag P50 differs from A's by
more than 10 %; the request-response reference (A primary) per request and per question side by side; the D1
description (block 1 against blocks 2 and 3).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from omni_research.analysis.experiment_analysis import read_jsonl  # noqa: E402
from omni_research.duplex.metrics import DEADLINES_S, RESERVES_MS, dist  # noqa: E402
from omni_research.duplex.report import (  # noqa: E402
    choose_runs,
    discover_runs,
    duplex_phase_metrics,
    load_server_units,
    speaking_selector,
    stage_table,
    t_interval,
)

RUN_RE = re.compile(r"^ppv_r1_(duplex|chat_ref)_block(\d+)(?:_a\d+)?$")
DUPLEX_PHASES = {"A": "duplex_r1_A", "B": "duplex_r1_B"}
CHAT_PHASES = {"A": "w3_c1_A", "B": "w3_c1_B"}
OVERHEAD_LIMIT = 0.10  # item 6
OVERLAP_NOTE = {1: "D1: ran at the same time as the R3 screening runs and the C1/C2 microbenchmarks on other GPUs"}
SEMANTICS = [
    ("full duplex: TTFA from the end of sending the utterance; the reply length is the model's choice; demo sampling "
     "(temperature 0.7); one session that keeps listening in 1 s units"),
    ("request-response reference: TTFA from sending the whole request (a real application also waits for an "
     "end-of-speech detector, not modelled); max_tokens 64 and 'answer in one short sentence'; temperature 0; "
     "speech returns after talker and code2wav finish"),
    ("both on one RTX A6000, sglang-omni 520b49ee stock; different pipelines (perception/thinker/talker/speech with "
     "FP32 flow vs thinker/talker/code2wav with FP16 flow), so a TTFA difference says nothing about any kernel or "
     "scheduling choice"),
]


# ---------------------------------------------------------------- runs

def choose(exps: list[Path]) -> tuple[dict[tuple[str, int], Path], list[dict[str, Any]]]:
    """Latest ok run per (kind, block); the rest are excluded with a reason (item 3)."""
    return choose_runs(exps, RUN_RE, lambda m: (m.group(1), int(m.group(2))))


# ---------------------------------------------------------------- full duplex

def duplex_view(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """One block (one session) or the pooled blocks: every list concatenated, then the registered percentiles."""
    turns = [r for s in sessions for r in s["turn_rows"] if r.get("status") != "not_sent"]
    answered = [r for r in turns if r.get("status") == "answered"]
    replies = {r["response_id"]: r for s in sessions for r in s["reply_rows"]}
    audio_reps = [r for r in replies.values() if "first_audio_ms" in r]
    parts = [r["parts"] for r in answered if r.get("parts")]
    regular = [p for p in parts if not p["early_start"]]
    before = {k: sum(p["units_before_speech"][k] for p in regular) for k in ("listen", "text_only", "speak")}
    play = {}
    for res in RESERVES_MS:
        key = f"{int(res)}ms"
        rows = [r["playback"][key] for r in audio_reps if "playback" in r]
        play[key] = {"replies": len(rows), "replies_with_stall": sum(1 for x in rows if x["stalls"] > 0),
                     "stalls": sum(x["stalls"] for x in rows), "underrun_ms_total": sum(x["underrun_ms"] for x in rows),
                     "buffer_needed_ms": dist([x["buffer_needed_ms"] for x in rows])}
    barge = [r for r in turns if r["kind"] == "barge_in"]
    return {
        "sessions": len(sessions), "session_status": [(s["result"] or {}).get("status") for s in sessions],
        "errors": sum(len(s["errors"]) for s in sessions),
        "turns": len(turns), "answered": len(answered), "no_response": sum(1 for r in turns if r["status"] == "no_response"),
        "overlapped_at_end": sum(1 for r in turns if r.get("overlapped_at_end")),
        "ttfa_ms": dist([r["ttfa_ms"] for r in answered]),
        "ttfa_ms_questions": dist([r["ttfa_ms"] for r in answered if r["kind"] == "question"]),
        "unit_lag_ms": dist([x for s in sessions for x in s["unit_lags_ms"]]),
        "units": {k: sum(s["units"][k] for s in sessions) for k in ("planned", "done", "missed")},
        "decisions": {k: sum(s["units"]["decisions"][k] for s in sessions) for k in ("speak", "text_only", "listen")},
        "first_audio_service_ms": dist([r.get("first_audio_latency_ms") for r in audio_reps]),
        "delta_gap_ms": dist([g for s in sessions for g in s["delta_gaps_ms_list"]]),
        "playback": play,
        "deadline_miss": {f"{d:g}s": {"all_turns": (sum(1 for r in turns if r["status"] != "answered" or r["ttfa_ms"] > d * 1000)
                                                    / len(turns)) if turns else None,
                                      "answered_turns": (sum(1 for r in answered if r["ttfa_ms"] > d * 1000) / len(answered))
                                      if answered else None} for d in DEADLINES_S},
        "barge_in": {"scripted": len(barge), "valid": sum(1 for r in barge if r.get("barge_in_valid")),
                     "stop_latency_ms": dist([r.get("barge_in_stop_latency_ms") for r in barge if r.get("barge_in_valid")])},
        "ttfa_parts": {"answered_with_parts": len(parts), "early_start": len(parts) - len(regular),
                       "a_wait_unit_end_ms": dist([p["a_wait_unit_end_ms"] for p in regular]),
                       "b_listen_ms": dist([p["b_listen_ms"] for p in regular]),
                       "c_service_ms": dist([p["c_service_ms"] for p in regular]),
                       "k_counts": {str(k): sum(1 for p in regular if p["k"] == k) for k in sorted({p["k"] for p in regular})},
                       "units_before_speech": before},
        "answer_audio_s": dist([r.get("reply_audio_s") for r in answered]),
        "answer_text_chars": dist([replies[r["response_id"]]["text_chars"] for r in answered if r.get("response_id") in replies]),
    }


def ttfa_turn_rows(sessions: list[dict[str, Any]], block: int, phase: str) -> list[dict[str, Any]]:
    out = []
    for s in sessions:
        for r in s["turn_rows"]:
            if r.get("status") == "answered":
                p = r.get("parts") or {}
                out.append({"block": block, "phase": phase, "turn": r["turn"], "kind": r["kind"], "asset_id": r.get("asset_id"),
                            "ttfa_ms": r["ttfa_ms"], "k": p.get("k"), "early_start": p.get("early_start"),
                            "a_wait_unit_end_ms": p.get("a_wait_unit_end_ms"), "b_listen_ms": p.get("b_listen_ms"),
                            "c_service_ms": p.get("c_service_ms")})
    return out


def server_stages(exp: Path, phase: str) -> dict[str, Any]:
    units, summary, reason = load_server_units(exp / "raw" / "server_events" / phase)
    if units is None:
        return {"available": False, "reason": reason}
    return {"available": True, "units": units, "summary": summary}


def stage_view(entries: list[dict[str, Any]], client_sessions: list[dict[str, Any]]) -> dict[str, Any]:
    if not entries or any(not e["available"] for e in entries):
        bad = next((e for e in entries if not e["available"]), None)
        return {"available": False, "reason": bad["reason"] if bad else "no B phase"}
    units = [u for e in entries for u in e["units"]]
    sel, how = speaking_selector(units, client_sessions)
    speaking, everything = stage_table(units, sel), stage_table(units)
    return {"available": True, "speaking_from": how, "speaking_units": speaking, "all_units": everything,
            "top_stage_speaking_units": speaking["top_stage"], "top_stage_all_units": everything["top_stage"]}


# ---------------------------------------------------------------- request-response reference

def chat_rows(exp: Path, phase: str) -> list[dict[str, Any]]:
    def rel(rec: dict[str, Any], key: str) -> float | None:
        t0, t = rec.get("sent_ns"), rec.get(key)
        return (t - t0) / 1e6 if t0 is not None and t is not None else None

    rows = []
    for r in read_jsonl(exp / "raw" / f"requests_{phase}.jsonl"):
        if r.get("warmup") or r.get("kind") not in (None, "model_request"):
            continue
        payload = r.get("payload") or (r.get("metadata") or {}).get("payload") or {}  # drivers._record: top level
        rows.append({"status": r.get("status"), "asset_id": payload.get("audio_asset"), "ttfa_ms": rel(r, "first_audio_ns"),
                     "ttft_ms": rel(r, "first_text_ns"), "e2e_ms": rel(r, "done_ns"), "audio_s": r.get("output_audio_s"),
                     "text_chars": r.get("text_chars"), "completion_tokens": r.get("completion_tokens"),
                     "audio_chunks": len(r.get("audio_chunks") or [])})
    return rows


def chat_view(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if r["status"] == "ok"]
    return {"requests": len(rows), "ok": len(ok), "errors": len(rows) - len(ok),
            **{k: dist([r[k] for r in ok]) for k in ("ttfa_ms", "ttft_ms", "e2e_ms", "audio_s", "text_chars",
                                                     "completion_tokens", "audio_chunks")}}


def by_question(duplex_rows: list[dict[str, Any]], chat: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dq: dict[str, list[float]] = defaultdict(list)
    for r in duplex_rows:
        dq[r["asset_id"]].append(r["ttfa_ms"])
    cq: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in chat:
        if r["status"] == "ok":
            cq[r["asset_id"]].append(r)
    out = []
    for a in sorted(set(dq) | set(cq), key=str):
        c = cq.get(a, [])
        out.append({"asset_id": a, "duplex_answered": len(dq.get(a, [])), "duplex_ttfa_ms_p50": dist(dq.get(a, []))["p50"],
                    "chat_requests": len(c), "chat_ttfa_ms_p50": dist([x["ttfa_ms"] for x in c])["p50"],
                    "chat_audio_s_p50": dist([x["audio_s"] for x in c])["p50"]})
    return out


# ---------------------------------------------------------------- assembly

def build(exps: list[Path]) -> dict[str, Any]:
    chosen, excluded = choose(exps)
    out: dict[str, Any] = {"generated_by": "scripts/analysis/ppv_r1_report.py",
                           "rules": "preregistration.md R1 + R1/R2 分析口径补充 items 1-8",
                           "runs": {f"{k}_block{b}": p.name for (k, b), p in sorted(chosen.items())}, "excluded_runs": excluded,
                           "block_notes": {str(b): n for b, n in OVERLAP_NOTE.items()}, "semantics": SEMANTICS}
    blocks = sorted({b for (_, b) in chosen})
    sessions: dict[str, dict[int, list[dict[str, Any]]]] = {arm: {} for arm in DUPLEX_PHASES}
    skipped: dict[str, list[int]] = {arm: [] for arm in DUPLEX_PHASES}
    turn_rows: list[dict[str, Any]] = []
    server: dict[int, dict[str, Any]] = {}
    for b in blocks:
        exp = chosen.get(("duplex", b))
        if exp is None:
            continue
        for arm, phase in DUPLEX_PHASES.items():
            d = exp / "raw" / f"duplex_{phase}"
            if not (d / "scripts.json").exists():
                skipped[arm].append(b)
                continue
            sessions[arm][b] = duplex_phase_metrics(d)["sessions"]
            turn_rows += ttfa_turn_rows(sessions[arm][b], b, phase)
        if b in sessions["B"]:
            server[b] = server_stages(exp, DUPLEX_PHASES["B"])
    duplex: dict[str, Any] = {}
    for arm, phase in DUPLEX_PHASES.items():
        per = {str(b): duplex_view(s) for b, s in sorted(sessions[arm].items())}
        pooled = duplex_view([x for s in sessions[arm].values() for x in s])
        duplex[arm] = {"phase": phase, "blocks": per, "pooled": pooled, "missing_blocks": skipped[arm],
                       "block_p50": {m: t_interval([v[m]["p50"] for v in per.values()]) for m in ("ttfa_ms", "unit_lag_ms")}}
    out["duplex"] = duplex
    out["ttfa_turns"] = turn_rows
    pa, pb = duplex["A"]["pooled"]["unit_lag_ms"]["p50"], duplex["B"]["pooled"]["unit_lag_ms"]["p50"]
    diff = (pb - pa) / pa if pa and pb is not None else None
    flagged = diff is not None and abs(diff) > OVERHEAD_LIMIT
    out["events_overhead"] = {"pooled_unit_lag_p50_ms": {"A": pa, "B": pb}, "relative_difference": diff, "flag": flagged,
                              "per_block": {str(b): _rel(duplex["B"]["blocks"].get(str(b)), duplex["A"]["blocks"].get(str(b)))
                                            for b in blocks},
                              "note": "stage numbers carry 'event recording visibly adds latency'" if flagged else None}
    per_block_stage = {str(b): stage_view([e], sessions["B"].get(b, [])) for b, e in sorted(server.items())}
    pooled_stage = stage_view(list(server.values()), [x for s in sessions["B"].values() for x in s]) if server else {
        "available": False, "reason": "no B phase"}
    out["server_stages"] = {"pooled": pooled_stage, "per_block": per_block_stage,
                            "events_overhead_flag": flagged}
    chat: dict[str, Any] = {}
    chat_rows_by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for arm, phase in CHAT_PHASES.items():
        per = {}
        for b in blocks:
            exp = chosen.get(("chat_ref", b))
            if exp is None:
                continue
            rows = chat_rows(exp, phase)
            per[str(b)] = chat_view(rows)
            chat_rows_by_arm[arm] += rows
        chat[arm] = {"phase": phase, "blocks": per, "pooled": chat_view(chat_rows_by_arm[arm]),
                     "block_p50": {m: t_interval([v[m]["p50"] for v in per.values()]) for m in ("ttfa_ms", "e2e_ms")}}
    out["chat_reference"] = chat
    a_turns = [r for r in turn_rows if r["phase"] == DUPLEX_PHASES["A"]]
    out["side_by_side"] = {
        "primary": "A phases (no request events) of both paths",
        "duplex": {k: duplex["A"]["pooled"][k] for k in ("ttfa_ms", "answer_audio_s", "answer_text_chars", "answered", "turns")},
        "request_response": {k: chat["A"]["pooled"][k] for k in ("ttfa_ms", "e2e_ms", "audio_s", "text_chars", "ok")},
        "by_question": by_question(a_turns, chat_rows_by_arm["A"])}
    out["d1"] = d1_description(duplex, blocks)
    return out


def _rel(b: dict[str, Any] | None, a: dict[str, Any] | None) -> float | None:
    if not a or not b:
        return None
    x, y = a["unit_lag_ms"]["p50"], b["unit_lag_ms"]["p50"]
    return (y - x) / x if x and y is not None else None


def d1_description(duplex: dict[str, Any], blocks: list[int]) -> dict[str, Any]:
    """Item 8: per block unit lag P50/P90 and TTFA P50 of A and B; block 1 against the mean of blocks 2 and 3."""
    out: dict[str, Any] = {}
    for arm in DUPLEX_PHASES:
        rows = {str(b): {"unit_lag_p50_ms": v["unit_lag_ms"]["p50"], "unit_lag_p90_ms": v["unit_lag_ms"]["p90"],
                         "ttfa_p50_ms": v["ttfa_ms"]["p50"]} for b, v in duplex[arm]["blocks"].items()}
        vs = {}
        for m in ("unit_lag_p50_ms", "unit_lag_p90_ms", "ttfa_p50_ms"):
            one = rows.get("1", {}).get(m)
            rest = [rows[str(b)][m] for b in blocks if b != 1 and str(b) in rows and rows[str(b)][m] is not None]
            ref = sum(rest) / len(rest) if rest else None
            vs[m] = {"block1": one, "blocks_2_3_mean": ref, "relative_difference": (one - ref) / ref if one is not None and ref else None}
        out[arm] = {"per_block": rows, "block1_vs_rest": vs}
    out["note"] = "descriptive only (deviation D1); no rule changes"
    return out


def markdown(r: dict[str, Any]) -> str:
    def f(x: Any, nd: int = 0) -> str:
        return "N/A" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    lines = ["# R1 summary (generated)", "", f"Runs: {', '.join(f'{k}={v}' for k, v in r['runs'].items())}", ""]
    lines += ["| phase | turns | answered | TTFA P50/P95 ms (N) | unit lag P50/P99 ms (N) | first-audio service P50 ms | stalls @80 ms |",
              "|---|---|---|---|---|---|---|"]
    for arm in ("A", "B"):
        p = r["duplex"][arm]["pooled"]
        pb = p["playback"]["80ms"]
        lines.append(f"| {arm} pooled | {p['turns']} | {p['answered']} | {f(p['ttfa_ms']['p50'])}/{f(p['ttfa_ms']['p95'])} "
                     f"({p['ttfa_ms']['n']}) | {f(p['unit_lag_ms']['p50'])}/{f(p['unit_lag_ms']['p99'])} "
                     f"({p['unit_lag_ms']['n']}) | {f(p['first_audio_service_ms']['p50'])} | "
                     f"{pb['replies_with_stall']}/{pb['replies']} |")
    parts = r["duplex"]["A"]["pooled"]["ttfa_parts"]
    lines += ["", (f"TTFA parts (A pooled, N={parts['answered_with_parts']}, early starts {parts['early_start']}): "
                   f"(a) P50 {f(parts['a_wait_unit_end_ms']['p50'])} ms, (b) P50 {f(parts['b_listen_ms']['p50'])} ms, "
                   f"(c) P50 {f(parts['c_service_ms']['p50'])} ms, k counts {parts['k_counts']}")]
    st = r["server_stages"]["pooled"]
    top = ("top stage over speaking units: " + str(st["top_stage_speaking_units"]) if st.get("available")
           else "unavailable: " + st.get("reason", ""))
    flag = " — event recording visibly adds latency" if r["events_overhead"]["flag"] else ""
    lines += ["", f"Server stages (B pooled): {top}{flag}"]
    c = r["chat_reference"]["A"]["pooled"]
    lines += ["", (f"Request-response reference (A pooled): TTFA P50 {f(c['ttfa_ms']['p50'])} ms, e2e P50 "
                   f"{f(c['e2e_ms']['p50'])} ms, audio P50 {f(c['audio_s']['p50'], 2)} s, ok {c['ok']}/{c['requests']}")]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--exp", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--md")
    a = ap.parse_args()
    if a.data_root is None and not a.exp:
        ap.error("give --data-root or --exp")
    rep = build(discover_runs(a.data_root, a.exp))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    if a.md:
        Path(a.md).write_text(markdown(rep))
    print(markdown(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
