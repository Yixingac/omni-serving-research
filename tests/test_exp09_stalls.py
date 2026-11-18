"""Transient stalls in the 实验 9 open-loop levels: which requests count as stalled, and GPU busy per second."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("exp09_stalls", HERE / "exp09_stalls.py")
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)

S = 1_000_000_000


def rec(i, sent_s, headers_s, first_text_s, first_audio_s, warmup=False, status="ok"):
    return {"request_id": f"r{i}", "status": status, "warmup": warmup, "scheduled_ns": int(sent_s * S), "sent_ns": int(sent_s * S),
            "headers_ns": int(headers_s * S), "first_text_ns": int(first_text_s * S), "first_audio_ns": int(first_audio_s * S),
            "done_ns": int((first_audio_s + 0.1) * S), "lateness_s": 0.001}


def test_front_and_audio_stalls():
    recs = [rec(0, 10.0, 10.01, 10.1, 13.1), rec(1, 11.0, 11.01, 11.1, 14.1), rec(2, 12.0, 12.01, 12.1, 15.1),
            rec(3, 13.0, 15.5, 15.6, 18.6),   # headers 2.5 s after sending: front stall; first audio 3 s after text
            rec(4, 14.0, 14.01, 14.1, 23.1),  # first audio 9 s after text, median 3 s: audio stall
            rec(5, 9.0, 9.01, 9.1, 30.0, warmup=True), rec(6, 15.0, 15.01, 15.1, 15.2, status="error")]
    summary, stalled = st.level_stalls(recs)
    assert summary["requests"] == 5 and summary["first_audio_gap_median_s"] == 3.0
    assert summary["front_stalls"] == 1 and summary["front_at_s"] == [3.0]  # seconds after the first scheduled send
    assert summary["audio_stalls"] == 1 and summary["audio_at_s"] == [4.0]
    assert [r["request_id"] for r in stalled] == ["r3", "r4"]


def test_audio_stall_needs_more_than_five_seconds_above_the_median():
    recs = [rec(i, 10.0 + i, 10.01 + i, 10.1 + i, 13.1 + i) for i in range(4)] + [rec(9, 20.0, 20.01, 20.1, 28.1)]
    summary, stalled = st.level_stalls(recs)  # 8 s gap = 5 s above the 3 s median: not more than 5
    assert summary["audio_stalls"] == 0 and stalled == []


def test_gpu_seconds_per_role_and_idle_roles():
    util = {0: [(99.5, 40.0, 0.0), (100.2, 50.0, 0.0), (100.6, 0.0, 0.0), (101.1, 0.0, 0.0), (101.5, 0.0, 0.0)],
            1: [(100.3, 80.0, 0.0), (101.2, 0.0, 0.0)], 2: [(100.4, 30.0, 0.0), (101.3, 10.0, 0.0)]}
    roles = {"thinker": [0], "code2wav": [1, 2]}
    secs = st.gpu_seconds(util, roles, 100.0, 102.0, origin_s=100.5)
    assert [s["t_s"] for s in secs] == [-1, 0, 1]  # floor of the time after the origin
    assert secs[0]["busy"] == {"thinker": 50.0, "code2wav": 55.0}
    assert secs[1]["busy"] == {"thinker": 0.0, "code2wav": 5.0} and secs[1]["idle_roles"] == ["thinker"]
    assert secs[2]["idle_roles"] == ["thinker"]  # 101.5 - 100.5 = 1.0 belongs to second 1


def test_idle_seconds_count_only_while_requests_are_still_being_sent():
    secs = [{"t_s": 58, "busy": {}, "idle_roles": ["thinker"]}, {"t_s": 59, "busy": {}, "idle_roles": ["thinker", "talker"]},
            {"t_s": 60, "busy": {}, "idle_roles": ["thinker", "talker"]}]  # GPUs drain after the last send at 59.8 s
    assert st.idle_while_sending(secs, ["thinker", "talker", "code2wav"], 59.8) == {"thinker": 2, "talker": 1, "code2wav": 0}
