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
    residence = {s: (3.0 if s == bottleneck else 1.0) for s in ("thinker", "talker", "code2wav")} if bottleneck else None
    return {"arm": arm, "block": block, "workload": {w: {"capacity_rps": cap, "bottleneck": bottleneck, "residence_ms": residence}}}


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
    runs[0]["workload"]["w3b"]["residence_ms"] = {"talker": 1.0, "code2wav": 3.0, "thinker": 1.0}  # code2wav holds the most in block 1
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


def test_closed_loop_throughput_while_all_slots_are_busy():
    # concurrency 4: 4 warmup requests then 12 measured, each takes 1 s; sends follow completions
    recs = []
    for i in range(16):
        sent = (i // 4) * 1.0
        recs.append({"sent_ns": int(sent * 1e9), "done_ns": int((sent + 1.0) * 1e9), "status": "ok", "warmup": i < 4})
    recs.append({"sent_ns": int(4.0 * 1e9), "done_ns": int(9.0 * 1e9), "status": "ok"})  # a slow straggler in the tail
    s = rp.busy_window_throughput(recs)
    # measured sends span 1 s .. 4 s; 12 completions land in [1, 4] (at 1, 2, 3 and 4 s): 12 / 3 s
    assert abs(s["throughput_rps"] - 4.0) < 1e-9 and s["window_s"] == 3.0


def test_inflight_series_counts_sent_minus_done_each_second():
    recs = [rec(i * 0.5, i * 0.5 + 2.0) for i in range(10)]  # 2 per second, each takes 2 s
    s = rp.inflight_series(recs, step_s=1.0)
    assert s[0] == [0.0, 1] and s[2] == [2.0, 4] and s[-1][1] >= 0


def test_unusable_or_restarted_capacity_level_is_left_out():
    def row(mode, errors=0, timeouts=0, restart=False, stage="code2wav"):
        client = {"measured": 100, "errors": errors, "timeouts": timeouts, "throughput_rps": 20.0, "audio_s_per_s": 50.0}
        return {"workload": "w3b", "kind": "closed_loop", "mode": mode, "concurrency": 256, "client": client,
                "usable": rp.usable(client), "restart_here": restart, "bottleneck": {"stage": stage}}

    ok = rp.workload_entry([row("A"), row("B")], "w3b", 256)
    assert ok["capacity_rps"] == 20.0 and ok["bottleneck"] == "code2wav" and ok["excluded"] == []
    bad = rp.workload_entry([row("A", errors=4, timeouts=2), row("B", restart=True)], "w3b", 256)
    assert bad["capacity_rps"] is None and bad["bottleneck"] is None and len(bad["excluded"]) == 2
    assert rp.workload_entry([row("A", errors=5)], "w3b", 256)["capacity_rps"] == 20.0  # exactly 5 % is still usable
    assert rp.workload_entry([row("B")], "w3b", 256, pilot=True)["capacity_from"] == "B"  # the pilot has the events phase only


def test_per_arm_intervals_leave_out_unusable_closed_loop_levels_only():
    def lv(kind, level, rps, ok=True):
        key = "concurrency" if kind == "closed_loop" else "rate_rps"
        return {"workload": "w3b", "phase": "w3b_closed_A" if kind == "closed_loop" else "w3b_open", "mode": "A", "kind": kind, key: level,
                "client": {"throughput_rps": rps}, "usable": ok}

    runs = [{"arm": 1, "block": b, "levels": [lv("closed_loop", 256, 20.0, ok=b != 2), lv("open_loop", 18.0, 17.0, ok=b != 2)]} for b in (1, 2, 3)]
    by = rp.by_arm(runs)["1"]
    closed, opened = by["w3b|closed_loop|A|256"]["throughput_rps"], by["w3b|open_loop|A|18"]["throughput_rps"]
    assert closed["n"] == 2 and closed["runs"][1] is None and by["w3b|closed_loop|A|256"]["unusable_runs"] == [False, True, False]
    assert opened["n"] == 3  # open-loop levels are never excluded (registered)


def test_sensitivity_swaps_in_busy_window_and_stable_open_loop_capacity():
    def full(arm, block, cap, busy, steady_rates):
        r = run(arm, block, cap, "code2wav")
        r["levels"] = [{"workload": "w3b", "kind": "closed_loop", "mode": "A", "concurrency": 256, "usable": True,
                        "busy_window": {"throughput_rps": busy}}]
        r["levels"] += [{"workload": "w3b", "kind": "open_loop", "mode": "A", "rate_rps": rate, "growth": {"steady": rate in steady_rates}}
                        for rate in (15.5, 18.0, 21.0, 23.5)]
        return r

    runs = []
    for b in (1, 2, 3):
        runs += [full(1, b, 22.0, 25.0, {15.5, 18.0, 21.0}), full(2, b, 21.0, 27.0, {15.5, 18.0, 21.0, 23.5}),
                 full(3, b, 18.0, 28.5, {15.5, 18.0, 21.0, 23.5})]
    assert rp.verdict_h9a(runs)["verdict"] == "反驳"  # registered capacity: arm 3 below arm 2 in every block
    s = rp.sensitivity(runs)
    assert s["busy_window"]["H9a"]["verdict"] == "支持"  # 28.5 above 27 and 25, +5.6 % over arm 2
    assert s["stable_open_loop"]["H9a"]["verdict"] == "不确定"  # 23.5 ties arm 2
    runs[2]["levels"][0]["usable"] = False  # an unusable capacity level gives no busy-window value either
    assert rp.sensitivity(runs)["busy_window"]["H9a"]["verdict"] == "不确定"  # a missing value rules out 支持


def test_a_missing_block_can_only_leave_the_registered_outcomes():
    """The registration knows 支持 / 反驳 / 不确定 only: a missing value rules out 支持, and 反驳 is judged on the arms it
    names (H9a: arms 3 and 2)."""
    runs = [run(1, b, 22.0) for b in (1, 2, 3)] + [run(2, b, 24.0) for b in (1, 2, 3)] + [run(3, b, 25.5, "talker") for b in (1, 2)]
    a = rp.verdict_h9a(runs)
    assert a["verdict"] == "不确定" and a["missing"] == ["方案 3 区组 3"]  # arm 3 has no run in block 3
    assert rp.verdict_h9b(runs)["verdict"] == "不确定"
    low = [run(1, b, 22.0) for b in (1, 3)] + [run(2, b, 24.0) for b in (1, 2, 3)] + [run(3, b, 14.0) for b in (1, 2, 3)]
    assert rp.verdict_h9a(low)["verdict"] == "反驳"  # arm 1 is missing in block 2, but 反驳 compares arm 3 with arm 2 only
    both = []
    for b in (1, 2, 3):
        both += [run(1, b, 22.0), run(2, b, 24.0)]
        both[-2]["workload"]["w3l"] = {"capacity_rps": 6.0}
        both[-1]["workload"]["w3l"] = {"capacity_rps": 5.0 if b != 3 else None}
    assert rp.verdict_h9c(both)["verdict"] == "不确定"
    assert rp.verdict_h9a([r for r in runs if r["block"] != 3] + [run(3, b, 30.0) for b in (1, 2)])["verdict"] == "不确定"  # 2 blocks


def test_a_restart_in_the_capacity_phase_and_crash_affected_records(tmp_path):
    """The runner writes "<phase> c=<c> rep=<r>" for a restart after a closed-loop level and flags the crashed level's
    records crash_affected before running the level again into the same file."""
    import json

    exp = tmp_path / "20261118_exp09_t2_k1_c5_block1_run01"
    (exp / "raw").mkdir(parents=True)
    (exp / "raw" / "gpu_samples.csv").write_text("timestamp, index, utilization.gpu, memory.used\n")
    (exp / "result.json").write_text(json.dumps({
        "experiment_id": "exp09_t2_k1_c5_block1", "phases": [{"phase": "w3b_closed_A", "kind": "closed_loop"}],
        "server_restarts": [{"server": "s0", "exit_code": -11, "where": "w3b_closed_A c=256 rep=0"}]}))
    recs = []
    for i in range(10):  # the crashed attempt: half errors, all flagged
        recs.append({"request_id": f"x{i}", "concurrency": 256, "status": "error" if i % 2 else "ok", "crash_affected": True,
                     "sent_ns": 10**9 + i, "done_ns": 2 * 10**9 + i})
    for i in range(20):  # the rerun after the restart
        recs.append({"request_id": f"y{i}", "concurrency": 256, "status": "ok", "sent_ns": 5 * 10**9 + i * 10**8,
                     "done_ns": 7 * 10**9 + i * 10**8, "completion_tokens": 40, "output_audio_s": 12.0})
    (exp / "raw" / "requests_w3b_closed_A.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    run = rp.analyze_run(exp)
    row = run["levels"][0]
    assert row["restart_here"] and row["after_restart"] and row["crash_affected_dropped"] == 10
    assert row["client"]["measured"] == 20 and row["client"]["errors"] == 0
    w = run["workload"]["w3b"]
    assert w["capacity_rps"] is None and "restarted" in w["excluded"][0]


def test_h9b_compares_each_stage_with_the_talker_in_every_block():
    res = [{"talker": 2.0, "code2wav": 5.0, "thinker": 1.0}, {"talker": 3.0, "code2wav": 2.0, "thinker": 4.0},
           {"talker": 2.0, "code2wav": 5.0, "thinker": 1.0}]
    runs = [run(3, b, 14.0, "code2wav") for b in (1, 2, 3)]
    for r, x in zip(runs, res):
        r["workload"]["w3b"]["residence_ms"] = x
    assert rp.verdict_h9b(runs)["verdict"] == "不确定"  # neither code2wav nor thinker holds more than the talker in all blocks
    runs[1]["workload"]["w3b"]["residence_ms"] = {"talker": 2.0, "code2wav": 4.0, "thinker": 1.0}
    assert rp.verdict_h9b(runs)["verdict"] == "反驳"


def test_formal_capacity_never_falls_back_to_the_events_phase():
    def row(mode, conc, rps):
        return {"workload": "w3b", "kind": "closed_loop", "mode": mode, "concurrency": conc, "usable": True,
                "client": {"measured": 100, "errors": 0, "timeouts": 0, "throughput_rps": rps}, "bottleneck": {"stage": "code2wav"},
                "residence_ms": {"thinker": 1.0, "talker": 1.0, "code2wav": 3.0}}
    levels = [row("A", 128, 16.0), row("B", 256, 17.5)]
    e = rp.workload_entry(levels, "w3b", 256)
    assert e["capacity_rps"] is None and "A" in e["excluded"][0]
    assert rp.workload_entry(levels, "w3b", 256, pilot=True)["capacity_rps"] == 17.5  # the pilot has the events phase only


def test_a_failed_run_gives_missing_values():
    runs = [run(1, b, 22.0) for b in (1, 2, 3)] + [run(2, b, 24.0) for b in (1, 2, 3)] + [run(3, b, 25.5) for b in (1, 2, 3)]
    runs[-1]["status"] = "failed"
    assert rp.verdict_h9a(rp.drop_failed(runs))["verdict"] == "不确定"


def test_client_lateness_and_codec_cap_flags():
    recs = [{"status": "ok", "sent_ns": 0, "done_ns": 10**9, "lateness_s": 0.01 * i, "output_audio_s": 82.0 if i == 0 else 10.0,
             "completion_tokens": 40} for i in range(30)]
    out = rp.level_client(recs)
    assert out["lateness_flag"] is True  # P99 of 0 .. 0.29 s is over 100 ms
    assert out["output"]["at_codec_cap"] == 1  # 2048 codec tokens x 0.04 s


def test_stable_capacity_skips_restarted_or_failing_rates():
    lv = lambda rate, steady, **kw: {"workload": "w3b", "kind": "open_loop", "mode": "A", "rate_rps": rate, "growth": {"steady": steady}, **kw}  # noqa: E731
    r = {"levels": [lv(18.0, True), lv(21.0, True, restart_here=True), lv(23.5, True, usable=False)]}
    assert rp.stable_open_loop_capacity(r, "w3b") == 18.0


def test_paired_contrasts_against_arm_1_and_event_overhead():
    def lv(mode, conc, rps, p95):
        return {"workload": "w3b", "phase": f"w3b_closed_{mode}", "kind": "closed_loop", "mode": mode, "concurrency": conc, "usable": True,
                "client": {"throughput_rps": rps}, "censored": {"p50": 2.0, "p95": p95, "p99": p95}}
    runs = []
    for b in (1, 2, 3):
        runs.append({"arm": 1, "block": b, "levels": [lv("A", 256, 20.0, 4.0), lv("B", 256, 19.0, 4.2)], "workload": {}})
        runs.append({"arm": 2, "block": b, "levels": [lv("A", 256, 22.0, 3.0), lv("B", 256, 22.0, 3.1)], "workload": {}})
    c = rp.contrasts(runs)
    k = c["2|w3b|closed_loop|A|256"]
    assert all(math.isclose(x, 2.0) for x in k["throughput_rps"]["diff"]["runs"])
    assert all(math.isclose(x, 0.1) for x in k["throughput_rps"]["relative"]["runs"])
    assert all(math.isclose(x, -1.0) for x in k["e2e_p95_s"]["diff"]["runs"])
    o = rp.event_overhead(runs)
    assert math.isclose(o["1|w3b|256"]["mean"], -0.05) and math.isclose(o["2|w3b|256"]["mean"], 0.0)
