"""实验 8 analysis: the thinker's per-iteration events (synthetic events only)."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("exp08_lookahead", HERE / "exp08_lookahead.py")
la = importlib.util.module_from_spec(spec)
spec.loader.exec_module(la)


def step(**meta):
    return {"request_id": "__thinker_step__", "stage": "thinker@r0", "event_name": "exp_thinker_step", "timestamp_ns": 1, "pid": 3, "metadata": meta}


def decode(lookahead, n=4, blocked="none", **extra):
    return step(batch_size=n, forward_mode="DECODE", use_lookahead=lookahead, blocked_by=blocked, flush_reason="none",
                launch_ns=1_000_000 if lookahead else 0, resolve_ns=3_000_000 if lookahead else 0,
                run_ns=0 if lookahead else 25_000_000, process_ns=0 if lookahead else 2_000_000, **extra)


def test_iteration_summary_counts_lookahead_blockers_and_flushes():
    evs = [decode(True), decode(True), decode(False, blocked="runner_ineligible", blocker_speech=4, blocker_sampling=4),
           decode(False, blocked="runner_ineligible", blocker_speech=0, blocker_sampling=4),
           step(batch_size=6, forward_mode="MIXED", use_lookahead=False, blocked_by="not_decode", flush_reason="mixed_chunk",
                drain_ns=4_000_000, run_ns=40_000_000, process_ns=3_000_000),
           step(flush_reason="mixed_chunk", top_flush_ns=5_000_000),  # an iteration that only flushed
           {"request_id": "r1", "stage": "thinker@r0", "event_name": "stage_dispatch", "timestamp_ns": 2, "metadata": {}}]
    s = la.step_summary(evs)
    assert s["iterations"] == 5 and s["flush_only_iterations"] == 1
    assert s["lookahead_share"] == 2 / 5
    assert s["decode_iterations"] == 4 and s["decode_lookahead_share"] == 2 / 4
    assert s["blocked_by"] == {"runner_ineligible": 2, "not_decode": 1}
    # Of the decode iterations the runner rejected, how many had a request of each kind.
    assert s["runner_blockers"] == {"speech": 1, "sampling": 2}
    assert s["flush_reasons"] == {"mixed_chunk": 2}
    assert s["batch_size_decode"]["p50"] == 4
    assert s["mode_share"] == {"DECODE": 4 / 5, "MIXED": 1 / 5}
    assert s["cpu_ms"]["run"]["p50"] == 25 and s["cpu_ms"]["resolve"]["p50"] == 3


def test_no_step_events():
    assert la.step_summary([])["iterations"] == 0
