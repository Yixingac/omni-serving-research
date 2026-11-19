"""Duplex metrics on a hand-built trace: unit lag, TTFA, first-audio latency, stalls, N/A percentiles."""

import math

from omni_research.duplex.metrics import dist, pct, playback, pool, session_metrics

MS = 1_000_000


def _trace(second_delta_ms=4700.0, with_reply=True, fatal=False):
    recs = [{"d": "meta", "what": "anchor", "t_mono_ns": 0, "t_wall_ns": 0}]
    for k in range(75):  # 6 s of 80 ms packets
        recs.append({"d": "send", "type": "input_audio_buffer.append", "seq": k, "t_start_ms": 80.0 * k,
                     "t_mono_ns": int(80 * k * MS), "t_done_mono_ns": int((80 * k + 1) * MS), "t_wall_ns": 0,
                     "lateness_ms": 0.5})
    recs.append({"d": "send", "type": "sglang.input_audio.end", "t_mono_ns": int(6000 * MS), "t_wall_ns": 0})

    def unit_meta(k):
        return {"unit_id": f"unit_{k}", "media_time": {"t_start_ms": 1000.0 * k, "duration_ms": 1000.0}}

    def recv(t_ms, typ, ev):
        recs.append({"d": "recv", "type": typ, "t_mono_ns": int(t_ms * MS), "t_wall_ns": 0, "ev": {"type": typ, **ev}})

    for k in range(6):
        ready = 80 * (math.ceil(12.5 * (k + 1)) - 1) + 1
        if with_reply and k == 2:
            recv(3400.0, "response.created", {"response": {"id": "r0"}, "sglang": unit_meta(2)})
            recv(3500.0, "response.output_audio.delta", {"response_id": "r0", "delta_bytes": 48000, "delta_dbfs": -20.0,
                                                         "sglang": unit_meta(2)})
        if with_reply and k == 3:
            recv(second_delta_ms, "response.output_audio.delta", {"response_id": "r0", "delta_bytes": 48000,
                                                                  "delta_dbfs": -20.0, "sglang": unit_meta(3)})
            recv(second_delta_ms, "response.done", {"response": {"id": "r0", "status": "completed"}})
        recv(ready + 200.0 if k != 3 else max(ready + 200.0, second_delta_ms), "sglang.unit.done",
             {"unit_id": f"unit_{k}", "sglang": unit_meta(k)})
    if fatal:
        recs.append({"d": "recv", "type": "error", "t_mono_ns": int(5000 * MS), "t_wall_ns": 0,
                     "ev": {"type": "error", "sglang": {"fatal": True}, "error": {"code": "internal"}}})
    recs.append({"d": "meta", "what": "result", "t_mono_ns": int(6100 * MS), "t_wall_ns": 0,
                 "status": "failed" if fatal else "ok"})
    return recs


SCRIPT = {"duration_s": 6.0, "turns": [{"index": 0, "kind": "question", "asset_id": "q0", "start_sample": 16000,
                                        "end_sample": 48000, "first_packet": 12, "last_packet": 37}]}


def test_ttfa_unit_lag_first_audio_and_decisions():
    m = session_metrics(_trace(), SCRIPT)
    turn = m["turn_rows"][0]
    assert turn["status"] == "answered" and abs(turn["ttfa_ms"] - (3500 - 2961)) < 1e-6
    assert m["units"]["planned"] == 6 and m["units"]["done"] == 6 and m["units"]["missed"] == 0
    assert m["units"]["lag_ms"]["n"] == 6 and abs(m["units"]["lag_ms"]["p50"] - 200.0) < 1e-6
    assert m["units"]["decisions"] == {"speak": 2, "text_only": 0, "listen": 4}
    rep = m["reply_rows"][0]
    assert abs(rep["first_audio_latency_ms"] - 539.0) < 1e-6 and rep["audio_s"] == 2.0
    assert m["turns"]["deadline_miss"]["0.5s"]["all_turns"] == 1.0
    assert m["turns"]["deadline_miss"]["1s"]["all_turns"] == 0.0
    assert m["pacing"]["violations"] == 0


def test_playback_model_and_stall_counts():
    m = session_metrics(_trace(second_delta_ms=4700.0), SCRIPT)
    pb = m["reply_rows"][0]["playback"]
    assert abs(pb["0ms"]["underrun_ms"] - 200.0) < 1e-6 and pb["0ms"]["stalls"] == 1
    assert abs(pb["80ms"]["underrun_ms"] - 120.0) < 1e-6
    assert pb["250ms"]["stalls"] == 0 and abs(pb["250ms"]["buffer_needed_ms"] - 200.0) < 1e-6
    assert playback([], 80.0) == {"underrun_ms": 0.0, "stalls": 0, "buffer_needed_ms": 0.0}


def test_no_reply_is_no_response_not_a_number_and_a_late_unit_is_missed():
    m = session_metrics(_trace(with_reply=False), SCRIPT)
    assert m["turn_rows"][0]["status"] == "no_response" and m["turns"]["ttfa_ms"]["n"] == 0
    assert m["turns"]["deadline_miss"]["1s"]["all_turns"] == 1.0 and m["turns"]["deadline_miss"]["1s"]["answered_turns"] is None
    late = session_metrics(_trace(second_delta_ms=6500.0), SCRIPT)  # unit_3 done ~2.6 s after its input
    assert late["units"]["missed"] == 1


def test_percentiles_need_enough_samples():
    assert pct([1.0] * 19, 95) is None and pct([1.0] * 20, 95) == 1.0
    assert pct(list(range(99)), 99) is None and pct(list(range(100)), 99) is not None
    d = dist([3.0, 1.0, 2.0])
    assert d["n"] == 3 and d["p50"] == 2.0 and d["p90"] is None and d["max"] == 3.0
    assert dist([])["p50"] is None


def test_pool_counts_failed_sessions_units_as_missed():
    ok = session_metrics(_trace(), SCRIPT)
    bad = session_metrics(_trace(fatal=True), SCRIPT)
    p = pool([ok, bad])
    assert p["sessions"] == 2 and p["sessions_failed"] == 1 and p["fatal_errors"] == 1
    assert p["units_planned"] == 12 and p["units_missed"] == 6
    assert p["ttfa_ms"]["n"] == 2 and p["turns"] == 2


def _reply_trace(unit, arrival_ms):
    """6 s of packets, one reply whose first (only) audio delta is carried by ``unit`` and arrives at arrival_ms."""
    recs = [{"d": "send", "type": "input_audio_buffer.append", "seq": k, "t_start_ms": 80.0 * k,
             "t_mono_ns": int(80 * k * MS), "t_done_mono_ns": int((80 * k + 1) * MS), "t_wall_ns": 0, "lateness_ms": 0.0}
            for k in range(75)]
    meta = {"unit_id": f"unit_{unit}", "media_time": {"t_start_ms": 1000.0 * unit, "duration_ms": 1000.0}}
    recs += [{"d": "recv", "type": "session.created", "t_mono_ns": 0, "t_wall_ns": 0,
              "ev": {"type": "session.created", "session": {"id": "sess_x"}}},
             {"d": "recv", "type": "response.created", "t_mono_ns": int(arrival_ms * MS), "t_wall_ns": 0,
              "ev": {"type": "response.created", "response": {"id": "r1"}, "sglang": meta}},
             {"d": "recv", "type": "response.output_audio.delta", "t_mono_ns": int(arrival_ms * MS), "t_wall_ns": 0,
              "ev": {"type": "response.output_audio.delta", "response_id": "r1", "delta_bytes": 24000, "delta_dbfs": -20.0,
                     "sglang": meta}}]
    return recs


TURN_2_5S = {"duration_s": 6.0, "turns": [{"index": 0, "kind": "question", "asset_id": "q0", "start_sample": 16000,
                                           "end_sample": 40000, "first_packet": 12, "last_packet": 31}]}


def test_ttfa_parts_add_up_and_count_the_extra_listening_units():
    m = session_metrics(_reply_trace(3, 4100.0), TURN_2_5S)
    row = m["turn_rows"][0]
    p = row["parts"]
    t_end = 31 * 80 + 1  # the packet holding sample 39999
    assert abs(row["ttfa_ms"] - (4100 - t_end)) < 1e-6
    assert p["u0"] == 2 and p["u1"] == 3 and p["k"] == 1 and not p["early_start"]
    assert abs(p["a_wait_unit_end_ms"] - (2961 - t_end)) < 1e-6  # unit 2 complete with packet 36
    assert abs(p["b_listen_ms"] - (3921 - 2961)) < 1e-6 and abs(p["c_service_ms"] - (4100 - 3921)) < 1e-6
    assert abs(p["a_wait_unit_end_ms"] + p["b_listen_ms"] + p["c_service_ms"] - row["ttfa_ms"]) < 1e-6
    assert p["units_before_speech"] == {"listen": 1, "text_only": 0, "speak": 0}
    assert m["session_id"] == "sess_x" and m["ttfa_parts"]["k_counts"] == {"1": 1}


def test_a_reply_from_a_unit_cut_before_the_utterance_ended_is_an_early_start():
    m = session_metrics(_reply_trace(1, 2600.0), TURN_2_5S)
    p = m["turn_rows"][0]["parts"]
    assert p["k"] == -1 and p["early_start"]
    assert abs(p["a_wait_unit_end_ms"] + p["b_listen_ms"] + p["c_service_ms"] - m["turn_rows"][0]["ttfa_ms"]) < 1e-6
    assert m["ttfa_parts"]["early_start"] == 1 and m["ttfa_parts"]["a_wait_unit_end_ms"]["n"] == 0
