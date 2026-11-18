"""实验 12: how far a trace replay's numbers are from the live agent run it replays."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("exp12_report", HERE / "exp12_report.py")
r12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r12)

S = 1_000_000_000


def req(i, sess, sent, done, ttft=0.1, status="ok", error=None, out=10, prompt=100, finish="stop", kind="model_request"):
    return {"kind": kind, "request_id": f"q{i}", "session_id": sess, "status": status, "error": error, "sent_ns": int(sent * S),
            "done_ns": int(done * S) if done is not None else None, "first_text_ns": int((sent + ttft) * S) if done is not None else None,
            "completion_tokens": out, "prompt_tokens": prompt, "finish_reason": finish}


def test_summary_counts_errors_by_cause_and_skips():
    recs = [req(0, "a", 0.0, 1.0), req(1, "a", 1.5, 2.5), req(2, "b", 0.5, 2.0),
            req(3, "c", 1.0, None, status="error", error="Cannot connect: [Errno 24] Too many open files"),
            req(4, "d", 1.0, None, status="error", error="CUDA out of memory. Tried to allocate 64 MiB"),
            {"kind": "model_request", "status": "skipped", "session_id": "d"},
            {"kind": "tool_call", "status": "ok", "session_id": "a"}]
    s = r12.replay_summary(recs)
    assert s["model_requests"] == 6 and s["ok"] == 3 and s["skipped"] == 1
    assert s["errors"] == {"client fd limit": 1, "CUDA out of memory": 1}
    assert s["valid"] is False  # client-side fd failures make a run unusable for comparison
    assert abs(s["span_s"] - 2.5) < 1e-9 and abs(s["requests_per_s"] - 3 / 2.5) < 1e-9
    assert abs(s["mean_in_flight"] - (1.0 + 1.0 + 1.5) / 2.5) < 1e-9 and s["peak_in_flight"] == 2
    assert abs(s["session_p50_s"] - 2.0) < 1e-9  # sessions a: 0-2.5 s, b: 0.5-2.0 s
    assert s["finish"] == {"stop": 3}


def test_summary_is_valid_without_client_errors():
    s = r12.replay_summary([req(0, "a", 0.0, 1.0), req(1, "b", 0.0, 2.0, status="error", error="CUDA out of memory")])
    assert s["valid"] is True and s["ok"] == 1


def test_in_flight_series_per_second():
    recs = [req(0, "a", 0.0, 2.5), req(1, "b", 1.0, 1.5), req(2, "c", 3.2, 3.4)]
    series = r12.in_flight_series(recs, bin_s=1.0)
    # mean number of requests in flight in [0,1), [1,2), [2,3), [3,4)
    assert [round(x, 6) for x in series] == [1.0, 1.5, 0.5, 0.2]


def test_relative_difference():
    assert abs(r12.rel(1.1, 1.0) - 0.1) < 1e-12 and r12.rel(None, 1.0) is None and r12.rel(1.0, 0.0) is None


def test_copy_verdict_needs_the_copied_replay_to_look_faster():
    assert r12.copy_verdict(-0.15)["verdict"] == "支持"
    assert r12.copy_verdict(0.14)["verdict"] == "不支持"
    assert r12.copy_verdict(-0.05)["verdict"] == "不确定"


def test_within_verdict_needs_every_metric_inside_the_tolerance():
    assert r12.within_verdict({"e2e_p50_s": 0.05, "session_p50_s": -0.09})["verdict"] == "支持"
    assert r12.within_verdict({"e2e_p50_s": 0.05, "session_p50_s": -0.39})["verdict"] == "不支持"


def test_underestimate_verdict_needs_the_surrogate_to_look_slower():
    assert r12.underestimate_verdict(1.2)["verdict"] == "支持"
    assert r12.underestimate_verdict(-0.2)["verdict"] == "不支持"
    assert r12.underestimate_verdict(0.03)["verdict"] == "不确定"
