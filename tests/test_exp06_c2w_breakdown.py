"""实验 6 analysis: code2wav time breakdown from the measurement patch's events (synthetic events only)."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("exp06_c2w_breakdown", HERE / "exp06_c2w_breakdown.py")
bd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bd)

MS = 1_000_000
T0 = 1_800_000_000_000_000_000  # an arbitrary wall-clock origin in ns


def ev(rid, stage, name, t_ms, **meta):
    return {"request_id": rid, "stage": stage, "event_name": name, "timestamp_ns": T0 + int(t_ms * MS), "pid": 7, "metadata": meta}


def batch_events(rids, batch_id, dispatch_ms, dequeue_ms, formed_ms, start_ms, done_ms, complete_ms, vocode=None):
    """Stock dispatch/complete plus the patch's formed/done/vocode events for one code2wav batch."""
    out = []
    for i, rid in enumerate(rids):
        out.append(ev(rid, "code2wav@r0", "stage_dispatch", dispatch_ms[i]))
        out.append(ev(rid, "code2wav@r0", "exp_batch_formed", formed_ms, batch_id=batch_id, batch_size=len(rids), batch_cost=300 * len(rids),
                      dequeue_ns=T0 + int(dequeue_ms[i] * MS), queue_depth_at_dequeue=3, queue_depth_at_formed=2))
        out.append(ev(rid, "code2wav@r0", "exp_batch_done", done_ms, batch_id=batch_id, batch_size=len(rids), compute_start_ns=T0 + int(start_ms * MS)))
        out.append(ev(rid, "code2wav@r0", "stage_complete", complete_ms[i]))
        if vocode is not None:
            out.append(ev(rid, "code2wav@r0", "exp_vocode", done_ms - 1, batch_id=batch_id, **vocode))
    return out


def vocode_record(start_ms):
    ns = lambda ms: T0 + int(ms * MS)  # noqa: E731
    return {"prepare_start_ns": ns(start_ms), "prepare_end_ns": ns(start_ms + 1), "references_start_ns": ns(start_ms + 1),
            "references_end_ns": ns(start_ms + 2), "upload_start_ns": ns(start_ms + 2), "flow_issue_start_ns": ns(start_ms + 3),
            "flow_issue_end_ns": ns(start_ms + 43), "hift_issue_end_ns": ns(start_ms + 63), "download_end_ns": ns(start_ms + 103),
            "split_end_ns": ns(start_ms + 104), "postprocess_start_ns": ns(start_ms + 104), "rows": 2, "token_lengths": [300, 310],
            "hift_groups": 2, "flow_solve_path": "eager", "flow_solve_rows": 2, "flow_solve_frames": 1200,
            "gpu_flow_start_to_flow_end_ms": 70.0, "gpu_flow_end_to_hift_end_ms": 30.0}


def test_a_complete_batch_splits_into_queue_batching_compute_and_handoff():
    evs = batch_events(["a", "b"], 1, dispatch_ms=[0, 20], dequeue_ms=[50, 52], formed_ms=55, start_ms=55, done_ms=160,
                       complete_ms=[162, 163], vocode=vocode_record(55))
    reqs, batches, issues = bd.breakdown(evs)
    a, b = reqs["a"], reqs["b"]
    assert (a["queue_ms"], a["batching_ms"], a["compute_ms"], a["handoff_ms"]) == (50, 5, 105, 2)
    assert (b["queue_ms"], b["batching_ms"], b["handoff_ms"]) == (32, 3, 3)
    assert a["total_ms"] == a["queue_ms"] + a["batching_ms"] + a["compute_ms"] + a["handoff_ms"] == 162
    assert issues == {}
    # The batch's GPU spans and CPU parts are kept once, not once per request.
    (batch,) = batches.values()
    assert batch["size"] == 2 and batch["gpu_flow_ms"] == 70 and batch["gpu_hift_ms"] == 30
    assert batch["flow_issue_ms"] == 40 and batch["hift_issue_ms"] == 20 and batch["download_wait_ms"] == 40
    assert batch["request_ids"] == ["a", "b"]


def test_missing_patch_events_leave_the_parts_unknown_not_guessed():
    evs = [ev("c", "code2wav@r0", "stage_dispatch", 0), ev("c", "code2wav@r0", "stage_complete", 200)]
    reqs, batches, issues = bd.breakdown(evs)
    assert reqs["c"]["total_ms"] == 200
    assert reqs["c"]["queue_ms"] is None and reqs["c"]["compute_ms"] is None
    assert issues["missing_patch_events"] == 1 and batches == {}


def test_out_of_order_times_are_flagged_and_left_out():
    evs = batch_events(["d"], 2, dispatch_ms=[100], dequeue_ms=[90], formed_ms=105, start_ms=105, done_ms=150, complete_ms=[151])
    reqs, _, issues = bd.breakdown(evs)
    assert reqs["d"]["queue_ms"] is None and issues["out_of_order"] == 1


def test_duplicate_events_keep_the_first_and_are_counted():
    evs = batch_events(["e"], 3, dispatch_ms=[0], dequeue_ms=[10], formed_ms=12, start_ms=12, done_ms=50, complete_ms=[52])
    evs.append(ev("e", "code2wav@r0", "stage_complete", 90))
    reqs, _, issues = bd.breakdown(evs)
    assert reqs["e"]["total_ms"] == 52 and issues["duplicate_events"] == 1


def test_disagreeing_batch_ids_mark_the_attribution_uncertain():
    evs = batch_events(["f"], 4, dispatch_ms=[0], dequeue_ms=[5], formed_ms=6, start_ms=6, done_ms=40, complete_ms=[41])
    for e in evs:
        if e["event_name"] == "exp_batch_done":
            e["metadata"]["batch_id"] = 5
    reqs, _, issues = bd.breakdown(evs)
    assert reqs["f"]["compute_ms"] is None and issues["uncertain_batch"] == 1


def test_other_stages_and_thinker_steps_are_ignored_and_empty_input_is_fine():
    evs = [ev("g", "talker@r0", "stage_dispatch", 0), ev("__thinker_step__", "thinker@r0", "exp_thinker_step", 1, batch_size=3)]
    assert bd.breakdown(evs) == ({}, {}, {})
    assert bd.breakdown([]) == ({}, {}, {})


def test_level_summary_uses_measured_ok_requests_only():
    evs = batch_events(["h", "i"], 6, dispatch_ms=[0, 0], dequeue_ms=[10, 10], formed_ms=11, start_ms=11, done_ms=60, complete_ms=[61, 61])
    reqs, batches, _ = bd.breakdown(evs)
    records = {"h": {"status": "ok"}, "i": {"status": "error"}}
    s = bd.summarize(reqs, batches, records)
    assert s["requests"] == 1 and s["failed"] == 1 and s["queue_ms"]["p50"] == 10
    assert s["batches"] == 1 and s["batch_size"]["mean"] == 2


def test_nanosecond_mixups_are_caught():
    # A span above an hour can only come from mixing units or clocks.
    evs = batch_events(["j"], 7, dispatch_ms=[0], dequeue_ms=[5], formed_ms=6, start_ms=6, done_ms=10, complete_ms=[4_000_000])
    reqs, _, issues = bd.breakdown(evs)
    assert reqs["j"]["total_ms"] is None and issues["implausible_span"] == 1
