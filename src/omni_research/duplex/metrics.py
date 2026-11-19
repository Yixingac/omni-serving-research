"""Client-observed full-duplex timing from one session trace (``client.run_session``) and its script.

Definitions (registered in research/next_steps/presentation_validation/preregistration.md, R0-R2):
- unit lag: receipt of ``sglang.unit.done`` minus the moment the server had the unit's whole input, i.e.
  when the client finished sending the packet holding the unit's last audio; for the short last unit
  that the server pads, the later of that and the ``sglang.input_audio.end`` send (upstream
  benchmarks/duplex/serving.py, same definition). A unit misses its deadline when its lag exceeds one
  unit (1 s) or it never completes;
- TTFA of a user turn: from the end of sending the packet with the utterance's last sample to the first
  assistant audio delta of the first reply whose first audio arrives after that moment and before the
  next user turn starts. No such reply = ``no_response`` (the model kept listening), never a number;
- reply first-audio latency (upstream "first audio"): first audio delta receipt minus the ready time of
  the unit that carried it (serving time only, without the units the model spent listening);
- delta gap: time between consecutive audio deltas of one reply;
- playback stall: a player that starts ``reserve`` after a reply's first audio and plays chunks in order;
  underrun = time it waits for audio, buffer needed = smallest start-up buffer without a gap;
- deadline miss ratio over user turns: no_response or TTFA > D (D = 1 s primary, 0.5 and 2 s);
- a unit "speaks" when any audio delta carries its unit_id, is "text_only" with text but no audio,
  else it "listens";
- barge-in (scripted, see script.py): valid only when a reply was producing audio when the barge-in
  utterance started; stop latency = last audio delta of that reply minus the barge-in's first packet.
Percentiles need enough samples: p50 >= 1, p90 >= 10, p95 >= 20, p99 >= 100; otherwise None (N/A).
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

UNIT_MS = 1000.0
OUTPUT_RATE = 24000
PACKET_MS = 80
MIN_N = {50: 1, 90: 10, 95: 20, 99: 100}
DEADLINES_S = (0.5, 1.0, 2.0)
RESERVES_MS = (0.0, 80.0, 250.0)


def pct(values: Iterable[float], q: int) -> float | None:
    v = sorted(x for x in values if x is not None and math.isfinite(x))
    if len(v) < MIN_N.get(q, 1) or not v:
        return None
    # linear interpolation between closest ranks (numpy's default)
    pos = (len(v) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def dist(values: Iterable[float]) -> dict[str, Any]:
    v = [x for x in values if x is not None and math.isfinite(x)]
    out: dict[str, Any] = {"n": len(v)}
    for q in (50, 90, 95, 99):
        out[f"p{q}"] = pct(v, q)
    out["mean"] = sum(v) / len(v) if v else None
    out["max"] = max(v) if v else None
    return out


def _ms(ns: float) -> float:
    return ns / 1e6


def parse_trace(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Index a session trace: packets, input end, units, replies (all times in ms of this process's
    monotonic clock)."""
    packets: dict[int, dict[str, float]] = {}
    input_end_ms = None
    units: dict[str, dict[str, Any]] = {}
    unit_audio: dict[str, float] = defaultdict(float)
    unit_text: dict[str, int] = defaultdict(int)
    replies: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    result = None
    errors = []
    for r in records:
        d = r.get("d")
        if d == "send":
            if r.get("type") == "input_audio_buffer.append":
                packets[int(r["seq"])] = {"start": _ms(r["t_mono_ns"]), "done": _ms(r.get("t_done_mono_ns", r["t_mono_ns"])),
                                          "lateness_ms": float(r.get("lateness_ms", 0.0))}
            elif r.get("type") == "sglang.input_audio.end":
                input_end_ms = _ms(r["t_mono_ns"])
        elif d == "recv":
            ev, t = r.get("ev", {}), _ms(r["t_mono_ns"])
            typ = r.get("type")
            meta = ev.get("sglang") or {}
            if typ == "sglang.unit.done":
                uid = ev.get("unit_id") or meta.get("unit_id")
                units[uid] = {"t": t, "media_time": meta.get("media_time")}
            elif typ == "response.created":
                rid = (ev.get("response") or {}).get("id")
                if rid and rid not in replies:
                    replies[rid] = {"created": t, "chunks": [], "done": None, "status": None, "units": [],
                                    "first_audio_media_time": None, "text": ""}
                    order.append(rid)
            elif typ == "response.output_audio.delta":
                rid = ev.get("response_id")
                rep = replies.setdefault(rid, {"created": None, "chunks": [], "done": None, "status": None, "units": [],
                                                "first_audio_media_time": None, "text": ""})
                if rid not in order:
                    order.append(rid)
                audio_s = float(ev.get("delta_bytes", 0)) / 2 / OUTPUT_RATE
                rep["chunks"].append((t, audio_s, float(ev.get("delta_dbfs", -240.0))))
                if rep["first_audio_media_time"] is None:
                    rep["first_audio_media_time"] = meta.get("media_time")
                uid = meta.get("unit_id")
                if uid:
                    unit_audio[uid] += audio_s
                    rep["units"].append(uid)
            elif typ in ("response.output_audio_transcript.delta", "response.output_text.delta"):
                rid = ev.get("response_id")
                if rid in replies:
                    replies[rid]["text"] += ev.get("delta", "")
                uid = meta.get("unit_id")
                if uid:
                    unit_text[uid] += 1
            elif typ == "response.done":
                resp = ev.get("response") or {}
                rid = resp.get("id")
                if rid in replies:
                    replies[rid]["done"] = t
                    replies[rid]["status"] = resp.get("status")
            elif typ == "error":
                errors.append({"t": t, **(ev.get("error") or {}), "fatal": bool(meta.get("fatal"))})
        elif d == "meta" and r.get("what") == "result":
            result = {k: v for k, v in r.items() if k not in ("d", "what")}
    return {"packets": packets, "input_end_ms": input_end_ms, "units": units, "unit_audio_s": dict(unit_audio),
            "unit_text": dict(unit_text), "replies": {k: replies[k] for k in order}, "result": result, "errors": errors}


def unit_ready_ms(media_time: dict[str, float] | None, packets: dict[int, dict[str, float]],
                  input_end_ms: float | None) -> float | None:
    if not media_time:
        return None
    end_ms = float(media_time["t_start_ms"]) + float(media_time["duration_ms"])
    seq = math.ceil(end_ms / PACKET_MS - 1e-9) - 1
    p = packets.get(seq)
    if p is None:
        return None
    if float(media_time["duration_ms"]) >= UNIT_MS - 1e-6 or input_end_ms is None:
        return p["done"]
    return max(p["done"], input_end_ms)


def playback(chunks: list[tuple[float, float, float]], reserve_ms: float) -> dict[str, float]:
    """chunks = (arrival ms, audio s, level); returns underrun ms, stalls (count) and buffer needed ms."""
    if not chunks:
        return {"underrun_ms": 0.0, "stalls": 0, "buffer_needed_ms": 0.0}
    play = chunks[0][0] + reserve_ms
    unbuffered = chunks[0][0]
    under = 0.0
    stalls = 0
    need = 0.0
    for arrival, audio_s, _ in chunks:
        if arrival > play + 1e-9:
            under += arrival - play
            stalls += 1
        play = max(play, arrival) + audio_s * 1000.0
        need = max(need, arrival - unbuffered)
        unbuffered += audio_s * 1000.0
    return {"underrun_ms": under, "stalls": stalls, "buffer_needed_ms": need}


def session_metrics(records: list[dict[str, Any]], script: dict[str, Any], *, silent_dbfs: float = -60.0) -> dict[str, Any]:
    """Everything above for one session; ``script`` is SessionScript.to_json()."""
    tr = parse_trace(records)
    packets, units, replies = tr["packets"], tr["units"], tr["replies"]
    planned = math.ceil(script["duration_s"] * 1000.0 / UNIT_MS - 1e-9)

    # Units: lag, misses, decisions.
    lags = []
    decisions = {"speak": 0, "text_only": 0, "listen": 0}
    # The server closes the input with one more unit; when the input ended on a unit boundary that unit
    # holds no audio (duration 0). It is not one of the planned units.
    empty = {uid for uid, u in units.items() if u["media_time"] and float(u["media_time"].get("duration_ms", 0)) <= 0}
    for uid, u in units.items():
        if uid in empty:
            continue
        ready = unit_ready_ms(u["media_time"], packets, tr["input_end_ms"])
        if ready is not None:
            lags.append(u["t"] - ready)
        if tr["unit_audio_s"].get(uid, 0.0) > 0:
            decisions["speak"] += 1
        elif tr["unit_text"].get(uid, 0):
            decisions["text_only"] += 1
        else:
            decisions["listen"] += 1
    missed = max(0, planned - (len(units) - len(empty))) + sum(1 for x in lags if x > UNIT_MS)

    # Replies.
    rep_rows = []
    for rid, rep in replies.items():
        chunks = sorted(rep["chunks"])
        row: dict[str, Any] = {"response_id": rid, "chunks": len(chunks), "audio_s": sum(c[1] for c in chunks),
                               "status": rep["status"], "text_chars": len(rep["text"])}
        if chunks:
            row["first_audio_ms"] = chunks[0][0]
            row["last_audio_ms"] = chunks[-1][0]
            ready = unit_ready_ms(rep["first_audio_media_time"], packets, tr["input_end_ms"])
            row["first_audio_latency_ms"] = chunks[0][0] - ready if ready is not None else None
            row["delta_gaps_ms"] = [b[0] - a[0] for a, b in itertools.pairwise(chunks)]
            row["silent_chunks"] = sum(1 for c in chunks if c[2] <= silent_dbfs)
            row["playback"] = {f"{int(r)}ms": playback(chunks, r) for r in RESERVES_MS}
        rep_rows.append(row)

    # User turns: TTFA, overlap, barge-in.
    turns = script["turns"]
    starts = sorted(packets[t["first_packet"]]["start"] for t in turns if t["first_packet"] in packets)
    end_of_input = tr["input_end_ms"] if tr["input_end_ms"] is not None else (max(p["done"] for p in packets.values()) if packets else 0.0)
    audio_reps = [r for r in rep_rows if "first_audio_ms" in r]
    turn_rows = []
    for t in turns:
        p_end, p_start = packets.get(t["last_packet"]), packets.get(t["first_packet"])
        if p_end is None or p_start is None:
            turn_rows.append({"turn": t["index"], "kind": t["kind"], "status": "not_sent"})
            continue
        t_end = p_end["done"]
        later = [s for s in starts if s > p_start["start"] + 1e-6]
        window_end = later[0] if later else end_of_input + 60_000.0
        cand = [r for r in audio_reps if t_end < r["first_audio_ms"] < window_end]
        overlapped = any(r["first_audio_ms"] <= t_end < r["last_audio_ms"] for r in audio_reps)
        row = {"turn": t["index"], "kind": t["kind"], "asset_id": t["asset_id"], "utterance_end_ms": t_end,
               "overlapped_at_end": overlapped}
        if cand:
            first = min(cand, key=lambda r: r["first_audio_ms"])
            row.update(status="answered", ttfa_ms=first["first_audio_ms"] - t_end, response_id=first["response_id"],
                       reply_audio_s=first["audio_s"])
        else:
            row["status"] = "no_response"
        if t["kind"] == "barge_in":
            t_b = p_start["start"]
            active = [r for r in audio_reps if r["first_audio_ms"] <= t_b < r["last_audio_ms"]]
            row["barge_in_valid"] = bool(active)
            if active:
                r = active[0]
                row["barge_in_stop_latency_ms"] = r["last_audio_ms"] - t_b
                row["interrupted_response_id"] = r["response_id"]
        turn_rows.append(row)

    user_turns = [r for r in turn_rows if r.get("status") != "not_sent"]
    ttfa = [r["ttfa_ms"] for r in user_turns if r.get("status") == "answered"]
    misses = {f"{d:g}s": {"all_turns": (sum(1 for r in user_turns if r.get("status") != "answered" or r["ttfa_ms"] > d * 1000)
                                        / len(user_turns)) if user_turns else None,
                          "answered_turns": (sum(1 for x in ttfa if x > d * 1000) / len(ttfa)) if ttfa else None,
                          "n_turns": len(user_turns), "n_answered": len(ttfa)} for d in DEADLINES_S}
    lateness = [p["lateness_ms"] for p in packets.values()]
    gaps = [g for r in rep_rows for g in r.get("delta_gaps_ms", [])]
    play = {}
    for res in RESERVES_MS:
        key = f"{int(res)}ms"
        rows = [r["playback"][key] for r in rep_rows if "playback" in r]
        play[key] = {"replies": len(rows), "replies_with_stall": sum(1 for x in rows if x["stalls"] > 0),
                     "stall_count": sum(x["stalls"] for x in rows), "underrun_ms_total": sum(x["underrun_ms"] for x in rows),
                     "buffer_needed_ms": dist([x["buffer_needed_ms"] for x in rows])}
    barge = [r for r in turn_rows if r["kind"] == "barge_in"]
    return {
        "result": tr["result"],
        "errors": tr["errors"],
        "units": {"planned": planned, "done": len(units) - len(empty), "empty_end_units": len(empty), "missed": missed,
                  "miss_rate": missed / planned if planned else None, "lag_ms": dist(lags), "decisions": decisions},
        "replies": {"n": len(rep_rows), "with_audio": len(audio_reps),
                    "audio_s_total": sum(r["audio_s"] for r in rep_rows),
                    "first_audio_latency_ms": dist([r.get("first_audio_latency_ms") for r in audio_reps]),
                    "silent_chunks": sum(r.get("silent_chunks", 0) for r in rep_rows),
                    "statuses": {s: sum(1 for r in rep_rows if r["status"] == s) for s in {r["status"] for r in rep_rows}}},
        "turns": {"n": len(user_turns), "answered": len(ttfa), "no_response": sum(1 for r in user_turns if r["status"] == "no_response"),
                  "overlapped_at_end": sum(1 for r in user_turns if r.get("overlapped_at_end")),
                  "ttfa_ms": dist(ttfa), "ttfa_ms_question": dist([r["ttfa_ms"] for r in user_turns if r["kind"] == "question" and "ttfa_ms" in r]),
                  "deadline_miss": misses},
        "barge_in": {"scripted": len(barge), "valid": sum(1 for r in barge if r.get("barge_in_valid")),
                     "stop_latency_ms": dist([r.get("barge_in_stop_latency_ms") for r in barge if r.get("barge_in_valid")])},
        "delta_gap_ms": dist(gaps),
        "playback": play,
        "pacing": {"lateness_ms": dist(lateness), "violations": sum(1 for x in lateness if x > PACKET_MS)},
        "turn_rows": turn_rows,
        "reply_rows": [{k: v for k, v in r.items() if k != "delta_gaps_ms"} for r in rep_rows],
        "unit_lags_ms": lags,
        "delta_gaps_ms_list": gaps,
    }


def pool(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool several sessions of one level (failed sessions stay in the counts and add all their units as
    missed, as upstream does)."""
    lags = [x for s in sessions for x in s["unit_lags_ms"]]
    ttfa = [r["ttfa_ms"] for s in sessions for r in s["turn_rows"] if r.get("status") == "answered"]
    turns = [r for s in sessions for r in s["turn_rows"] if r.get("status") != "not_sent"]
    planned = sum(s["units"]["planned"] for s in sessions)
    failed = [s for s in sessions if not s["result"] or s["result"].get("status") != "ok"]
    missed = sum(s["units"]["missed"] for s in sessions if s not in failed) + sum(s["units"]["planned"] for s in failed)
    gaps = [g for s in sessions for g in s.get("delta_gaps_ms_list", [])]
    fal = []
    stalls = {f"{int(r)}ms": 0 for r in RESERVES_MS}
    with_audio = 0
    for s in sessions:
        for r in s["reply_rows"]:
            if r.get("first_audio_latency_ms") is not None:
                fal.append(r["first_audio_latency_ms"])
            if "playback" in r:
                with_audio += 1
                for k in stalls:
                    stalls[k] += r["playback"][k]["stalls"] > 0
    return {
        "sessions": len(sessions), "sessions_failed": len(failed),
        "errors": sum(len(s["errors"]) for s in sessions), "fatal_errors": sum(1 for s in sessions for e in s["errors"] if e.get("fatal")),
        "units_planned": planned, "units_missed": missed, "unit_miss_rate": missed / planned if planned else None,
        "unit_lag_ms": dist(lags), "ttfa_ms": dist(ttfa),
        "turns": len(turns), "answered": len(ttfa), "no_response": sum(1 for r in turns if r["status"] == "no_response"),
        "deadline_miss_all_turns": {f"{d:g}s": (sum(1 for r in turns if r.get("status") != "answered" or r["ttfa_ms"] > d * 1000)
                                                / len(turns)) if turns else None for d in DEADLINES_S},
        "first_audio_latency_ms": dist(fal),
        "replies_with_audio": with_audio, "replies_with_stall": stalls,
        "audio_s_total": sum(s["replies"]["audio_s_total"] for s in sessions),
        "decisions": {k: sum(s["units"]["decisions"][k] for s in sessions) for k in ("speak", "text_only", "listen")},
        "pacing_violations": sum(s["pacing"]["violations"] for s in sessions),
        "delta_gap_ms": dist(gaps),
    }
