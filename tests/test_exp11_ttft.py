"""实验 11 TTFT breakdown: clock alignment and the segments from send to the first text chunk (synthetic data only)."""

import importlib.util
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("exp11_ttft", HERE / "exp11_ttft.py")
tt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tt)

OFFSET = 1_000_000_000_000  # wall - monotonic, ns


def ev(rid, stage, name, mono_ms, with_mono=False):
    e = {"request_id": rid, "stage": stage, "event_name": name, "timestamp_ns": int(mono_ms * 1e6) + OFFSET, "metadata": {}}
    if with_mono:
        e["metadata"] = {"clock": "CLOCK_MONOTONIC", "monotonic_ns": int(mono_ms * 1e6) + 3_000}  # read 3 us later
    return e


def request_events(rid, t0):
    pts = [("coordinator", "request_admission", 1), ("thinker@r0", "stage_input_received", 2), ("thinker@r0", "scheduler_queue_enter", 3),
           ("thinker@r0", "scheduler_prefill_start", 30), ("thinker@r0", "scheduler_prefill_end", 70), ("thinker@r0", "scheduler_first_emit", 72),
           ("thinker@r0", "stage_first_stream_chunk_sent", 73), ("decode@r0", "stage_first_stream_chunk_sent", 75)]
    out = [ev(rid, s, n, t0 + dt) for s, n, dt in pts]
    out.append(ev(rid, "thinker@r0", "model_path_start", t0 + 29, with_mono=True))
    # a later duplicate must not move a "first" point
    out.append(ev(rid, "thinker@r0", "scheduler_first_emit", t0 + 500))
    return out


def test_clock_offset_from_events_that_carry_both_clocks():
    events = request_events("a", 0) + request_events("b", 1000)
    off = tt.clock_offset(events)
    assert abs(off["offset_ns"] - (OFFSET - 3_000)) <= 1 and off["n"] == 2 and off["spread_ns"] == 0


def test_segments_add_up_to_the_client_ttft():
    events = request_events("a", 100.0)
    off = tt.clock_offset(events)["offset_ns"]
    client = {"request_id": "a", "status": "ok", "sent_ns": int(100.0 * 1e6), "first_text_ns": int(180.0 * 1e6)}
    seg = tt.segments(tt.points(events, off)["a"], client)
    assert math.isclose(seg["queue_wait"], 27.0, abs_tol=0.01)  # queue enter 3 ms -> prefill start 30 ms
    assert math.isclose(seg["prefill"], 40.0, abs_tol=0.01) and math.isclose(seg["prefill_to_first_emit"], 2.0, abs_tol=0.01)
    assert math.isclose(sum(v for k, v in seg.items() if k != "ttft"), seg["ttft"], abs_tol=0.01)
    assert math.isclose(seg["ttft"], 80.0, abs_tol=0.01)
    # 3 us of read delay between the two clocks show up as -0.003 ms in the first segment
    assert math.isclose(seg["send_to_admission"], 1.003, abs_tol=0.01)


def test_missing_points_give_none_not_zero():
    events = [e for e in request_events("a", 0) if e["event_name"] != "scheduler_first_emit"]
    off = tt.clock_offset(events)["offset_ns"]
    seg = tt.segments(tt.points(events, off)["a"], {"request_id": "a", "status": "ok", "sent_ns": 0, "first_text_ns": int(80e6)})
    assert seg["prefill_to_first_emit"] is None and seg["first_emit_to_thinker_chunk"] is None
