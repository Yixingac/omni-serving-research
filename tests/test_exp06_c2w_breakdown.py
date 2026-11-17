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


def test_no_seed_workloads_send_a_null_seed_and_the_originals_keep_seed_zero():
    # sglang-omni keeps any request with a sampling seed off the thinker's one-step lookahead; the
    # 实验 8 controls must therefore send "seed": null, while W3t/W3b keep the client's seed 0.
    import random

    from omni_research.replay.payloads import Workload, build_request

    root = Path(__file__).resolve().parents[1] / "configs/workloads"
    seeds = {}
    for name in ("w3t_text_only", "w3t_text_only_noseed", "w3b_text_speech", "w3b_text_speech_noseed"):
        body = build_request(Workload.from_file(root / f"{name}.json"), random.Random(0), "r0").body(None)
        seeds[name] = body.get("seed", "absent")
        assert body["modalities"] == (["text"] if name.startswith("w3t") else ["text", "audio"])
    assert seeds == {"w3t_text_only": 0, "w3t_text_only_noseed": None, "w3b_text_speech": 0, "w3b_text_speech_noseed": None}


def k(ts, dur, corr, graph=0, name="gemm"):
    return {"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": dur, "args": {"correlation": corr, "graph id": graph}}


def rt(ts, dur, corr, name="cudaLaunchKernel"):
    return {"ph": "X", "cat": "cuda_runtime", "name": name, "ts": ts, "dur": dur, "args": {"correlation": corr}}


def ann(ts, dur, name):
    return {"ph": "X", "cat": "user_annotation", "name": name, "ts": ts, "dur": dur}


def test_trace_stats_split_kernels_by_the_cpu_range_that_launched_them():
    # Times in microseconds, as in torch profiler traces. Flow range 0-100 launches two kernels that
    # overlap on the GPU; HiFT range 100-200 launches one; the sync waits 150-260.
    trace = {"traceEvents": [
        ann(0, 100, "c2w.flow"), ann(100, 100, "c2w.hift"),
        rt(10, 5, 1), k(20, 30, 1), rt(30, 5, 2), k(40, 30, 2, graph=1),
        rt(110, 5, 3, "cuLaunchKernel"), k(150, 100, 3, name="fprop"),
        rt(150, 110, 4, "cudaStreamSynchronize"),
        k(300, 10, 99),  # no matching launch call: unattributed
    ]}
    s = bd.trace_stats(trace)
    assert s["kernels"] == 4 and abs(s["kernel_busy_ms"] - (0.05 + 0.1 + 0.01)) < 1e-9  # 20-70 union, 150-250, 300-310
    assert abs(s["launch_api_ms"] - 0.015) < 1e-9 and abs(s["sync_api_ms"] - 0.11) < 1e-9
    assert s["by_range"]["c2w.flow"]["kernels"] == 2 and abs(s["by_range"]["c2w.flow"]["kernel_ms"] - 0.06) < 1e-9
    assert s["by_range"]["c2w.hift"]["kernels"] == 1 and s["unattributed_kernels"] == 1
    # CPU time inside each range spent in launch calls and in synchronization calls.
    assert abs(s["by_range"]["c2w.flow"]["launch_api_ms"] - 0.010) < 1e-9 and s["by_range"]["c2w.flow"]["sync_api_ms"] == 0
    assert abs(s["by_range"]["c2w.hift"]["launch_api_ms"] - 0.005) < 1e-9 and abs(s["by_range"]["c2w.hift"]["sync_api_ms"] - 0.11) < 1e-9
    assert abs(s["graph_kernel_time_share"] - 30 / 170) < 1e-9  # over all kernels, attributed or not
    # Device queue delay: kernel start minus the end of the call that launched it (us -> ms).
    assert s["queue_delay_ms"]["n"] == 3 and abs(s["queue_delay_ms"]["p50"] - 0.005) < 1e-9


def test_trace_stats_of_an_empty_or_truncated_trace():
    assert bd.trace_stats({"traceEvents": []})["kernels"] == 0
    assert bd.trace_stats({})["kernels"] == 0


def test_a_graph_launch_counts_one_queue_delay_and_events_can_supply_the_cpu_ranges():
    # One cudaGraphLaunch (corr 5) runs three kernels back to back: only the first kernel's start
    # minus the call's end is queueing. The CPU ranges come from the patch's events (wall-clock ns):
    # trace time ts (us) maps to wall clock as baseTimeNanoseconds + ts * 1000.
    base = 1_000_000_000
    trace = {"baseTimeNanoseconds": base, "traceEvents": [
        rt(10, 5, 5, "cudaGraphLaunch"), k(20, 10, 5, graph=1), k(30, 10, 5, graph=1), k(40, 10, 5, graph=1),
        rt(60, 5, 6), k(80, 5, 6),
    ]}
    ranges = [("c2w.flow", base + 0, base + 50_000), ("c2w.hift", base + 50_000, base + 100_000)]
    s = bd.trace_stats(trace, cpu_ranges=ranges)
    assert s["queue_delay_ms"]["n"] == 2 and abs(s["queue_delay_ms"]["mean"] - (0.005 + 0.015) / 2) < 1e-9
    assert s["by_range"]["c2w.flow"]["kernels"] == 3 and s["by_range"]["c2w.hift"]["kernels"] == 1
    assert s["unattributed_kernels"] == 0
