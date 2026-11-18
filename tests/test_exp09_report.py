"""实验 9 report: open-loop rate rule, queue growth, censored percentiles, stage residence, verdicts (synthetic data only)."""

import importlib.util
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("exp09_report", HERE / "exp09_report.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)


def test_common_rates_follow_the_registered_rule():
    # {0.5 Xmin, 0.8 Xmin, Xmin, (Xmin+Xmax)/2, Xmax, 1.1 Xmax}, rounded to 0.5 (half up), duplicates dropped
    assert rp.common_rates([12.0, 24.0, 18.3, 20.0]) == [6.0, 9.5, 12.0, 18.0, 24.0, 26.5]
    assert rp.common_rates([4.6, 4.7]) == [2.5, 3.5, 4.5, 5.0]  # 4.6, 4.65 and 4.7 all round to 4.5; 5.17 -> 5.0


def rec(sent_s, done_s, status="ok"):
    return {"sent_ns": int(sent_s * 1e9), "done_ns": None if done_s is None else int(done_s * 1e9), "status": status}


def test_queue_growth_slope_is_flat_in_steady_state_and_positive_under_overload():
    steady = [rec(i * 0.1, i * 0.1 + 2.0) for i in range(600)]  # 10 req/s, each takes 2 s: 20 in flight
    g = rp.queue_growth(steady, rate_rps=10.0)
    assert abs(g["slope_rps"]) < 0.2 and g["steady"] is True
    over = [rec(i * 0.1, 1.0 + i * 0.125) for i in range(600)]  # sent 10/s, served 8/s: grows ~2/s
    g = rp.queue_growth(over, rate_rps=10.0)
    assert 1.6 < g["slope_rps"] < 2.4 and g["steady"] is False


def test_percentiles_count_failed_and_timed_out_requests_as_infinite():
    recs = [rec(0, 1.0 + i) for i in range(98)] + [rec(0, 300.0, "timeout"), rec(0, 5.0, "error")]
    p = rp.censored_percentiles(recs)
    assert p["n"] == 100 and p["failed"] == 2
    assert math.isclose(p["p50"], 50.5) and math.isinf(p["p99"])
    ok_only = rp.censored_percentiles(recs[:98])
    assert math.isclose(ok_only["p95"], 0.95 * 97 + 1.0)  # numpy's linear rule when nothing failed


def ev(rid, stage, name, t_ms):
    return {"request_id": rid, "stage": stage, "event_name": name, "timestamp_ns": int(t_ms * 1e6)}


def test_stage_residence_and_the_bottleneck_stage():
    events = []
    for i in range(10):
        r, t = f"r{i}", 1000.0 * i
        events += [ev(r, "thinker@r0", "stage_input_received", t), ev(r, "thinker@r0", "stage_complete", t + 100),
                   ev(r, "talker@r0", "stage_input_received", t + 110), ev(r, "talker@r0", "stage_complete", t + 910),
                   ev(r, "code2wav@r1", "stage_input_received", t + 912), ev(r, "code2wav@r1", "stage_complete", t + 1112)]
    res = rp.stage_residence(events, {f"r{i}" for i in range(10)})
    assert res == {"thinker": 1000.0, "talker": 8000.0, "code2wav": 2000.0}  # summed ms
    b = rp.bottleneck(res)
    assert b["stage"] == "talker" and math.isclose(b["shares"]["talker"], 8 / 11)


def run(arm, block, cap, bottleneck=None, w="w3b"):
    return {"arm": arm, "block": block, "workload": {w: {"capacity_rps": cap, "bottleneck": bottleneck}}}


def test_h9a_needs_arm3_above_both_in_every_block_and_five_percent_over_arm2():
    runs = [run(1, b, 22.0) for b in (1, 2, 3)] + [run(2, b, 24.0) for b in (1, 2, 3)] + [run(3, b, 25.5) for b in (1, 2, 3)]
    assert rp.verdict_h9a(runs)["verdict"] == "支持"  # +6.25 % over arm 2
    runs[-1]["workload"]["w3b"]["capacity_rps"] = 23.0
    assert rp.verdict_h9a(runs)["verdict"] == "不确定"
    low = [run(1, b, 22.0) for b in (1, 2, 3)] + [run(2, b, 24.0) for b in (1, 2, 3)] + [run(3, b, 14.0) for b in (1, 2, 3)]
    assert rp.verdict_h9a(low)["verdict"] == "反驳"


def test_h9b_talker_must_hold_the_most_in_every_block():
    runs = [run(3, b, 14.0, bottleneck="talker") for b in (1, 2, 3)]
    assert rp.verdict_h9b(runs)["verdict"] == "支持"
    runs[0]["workload"]["w3b"]["bottleneck"] = "code2wav"
    assert rp.verdict_h9b(runs)["verdict"] == "不确定"
    assert rp.verdict_h9b([run(3, b, 14.0, bottleneck="thinker") for b in (1, 2, 3)])["verdict"] == "反驳"


def test_h9c_winner_changes_with_the_workload():
    runs = []
    for b in (1, 2, 3):
        runs += [run(1, b, 22.0), run(2, b, 24.0)]
        runs[-2]["workload"]["w3l"] = {"capacity_rps": 6.0}
        runs[-1]["workload"]["w3l"] = {"capacity_rps": 5.0}
    assert rp.verdict_h9c(runs)["verdict"] == "支持"  # arm 2 wins W3b, arm 1 wins W3L in every block
    for r in runs:
        if r["arm"] == 2:
            r["workload"]["w3l"]["capacity_rps"] = 7.0
    assert rp.verdict_h9c(runs)["verdict"] == "不支持"


def test_leave_one_out_capacity_model():
    # isolated code2wav is the bottleneck in arms 2 and 4 (6 req/s per card), the single talker in arm 3 (14 req/s),
    # shared code2wav in arm 1 (3.7 per card): arm 1 has no other run with shared cards, arm 3 none with a full talker
    runs = []
    for b in (1, 2, 3):
        runs += [run(1, b, 22.2, "code2wav"), run(2, b, 24.0, "code2wav"), run(3, b, 14.0, "talker"), run(4, b, 18.0, "code2wav")]
    m = rp.capacity_model(runs, "w3b")
    p = m["predictions"]
    assert p[1]["predicted_rps"] is None
    assert math.isclose(p[2]["predicted_rps"], 24.0) and math.isclose(p[4]["predicted_rps"], 18.0)  # min(3 x 6, 3 x 14)
    assert math.isclose(p[3]["predicted_rps"], 30.0) and p[3]["limited_by"] == "code2wav_iso"  # misses the talker limit
    assert math.isclose(p[3]["error"], 30.0 / 14.0 - 1) and m["verdict"] == "不支持"
