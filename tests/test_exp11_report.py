"""实验 11 performance verdicts H11a-H11e on synthetic per-run values (no real data)."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("exp11_report", HERE / "exp11_report.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)


def cell(e2e=2.0, ttft=60.0, queue=5.0, mspt=28.0, tokens=44.0, large=0, inside=0, steps=100, idle=0.2):
    return {"A": {"e2e_p50": e2e}, "B": {"ttft_mean_ms": ttft, "queue_wait_mean_ms": queue, "ms_per_token": mspt, "tokens_mean": tokens},
            "C": {"large_copies": large, "large_inside": inside, "steps": steps, "between_idle_share": idle}}


def world(extra_ttft=20.0, queue_share=1.0, b_large=20, b_inside=20, c_large=1, idle_drop=0.03, e2e_gain=0.03, long_gain=0.08):
    runs = []
    for block in (1, 2, 3):
        for arm in "ABC":
            la = arm != "A"
            lv = {}
            for c in (8, 32, 64):
                d = extra_ttft if la else 0.0
                lv[c] = cell(e2e=2.0 * (1 - (e2e_gain if arm == "C" else 0)), ttft=60 + d, queue=5 + d * queue_share,
                             mspt=25.0 if la else 28.0, large=(b_large if arm == "B" else c_large if arm == "C" else 0),
                             inside=(b_inside if arm == "B" else 0), idle=0.25 - (idle_drop if arm == "C" else 0))
            runs.append({"arm": arm, "block": block, "levels": {"w3b_ns": lv,
                                                                 "w3l_ns": {32: cell(e2e=4.0 * (1 - (long_gain if arm == "C" else 0)), mspt=25.0 if la else 28.0, tokens=100)},
                                                                 "w3xl_ns": {32: cell(e2e=8.0 * (1 - (long_gain if arm == "C" else 0)), mspt=25.0 if la else 28.0, tokens=200)}}})
    return runs


def test_verdicts_when_everything_works_as_hypothesised():
    v = rp.verdicts(world())
    assert v["H11a"]["verdict"] == "支持" and v["H11b"]["verdict"] == "支持" and v["H11c"]["verdict"] == "支持"
    assert v["H11d"]["C_vs_B"]["verdict"] == "支持" and v["H11e"]["verdict"] == "支持"


def test_verdicts_when_nothing_holds():
    v = rp.verdicts(world(queue_share=0.2, b_inside=2, c_large=20, idle_drop=-0.01, e2e_gain=-0.01, long_gain=-0.02))
    assert v["H11a"]["verdict"] == "反驳" and v["H11b"]["verdict"] == "反驳" and v["H11c"]["verdict"] == "不支持"
    assert v["H11d"]["C_vs_B"]["verdict"] == "不支持" and v["H11e"]["verdict"] == "不支持"


def test_mixed_evidence_is_uncertain_and_missing_data_is_reported():
    runs = world()
    runs[-1]["levels"]["w3b_ns"][64]["C"]["between_idle_share"] = 0.30  # block 3 of C does not drop at 64
    assert rp.verdicts(runs)["H11c"]["verdict"] == "不确定"
    assert rp.verdicts([r for r in world() if r["arm"] != "C"])["H11d"]["C_vs_B"]["verdict"] == "数据不足"


def test_unusable_phase_counts_as_missing():
    runs = world()
    runs[0]["levels"]["w3b_ns"][32]["A"]["usable"] = False  # block 1, arm A: errors + timeouts > 5 %
    v = rp.verdicts(runs)
    assert v["H11d"]["C_vs_A"]["verdict"] == "数据不足"
    assert v["H11d"]["C_vs_B"]["verdict"] == "支持"


def test_default_on_rule_needs_h11d_and_a_small_ttft_increase():
    runs = world()
    for r in runs:
        for c in (8, 32, 64):
            r["levels"]["w3b_ns"][c]["A"]["ttft_p50_ms"] = 60.0 + (5.0 if r["arm"] == "C" else 0.0)
    assert rp.recommend(runs, rp.verdicts(runs))["default_on"] is True
    for r in runs:
        if r["arm"] == "C":
            r["levels"]["w3b_ns"][8]["A"]["ttft_p50_ms"] = 75.0
    rec = rp.recommend(runs, rp.verdicts(runs))
    assert rec["default_on"] is False and rec["ttft_increase_ms"][8]["mean"] == 15.0


def test_queue_increase_is_reported_in_decode_steps():
    runs = world()
    for r in runs:
        for c in (8, 32):
            r["levels"]["w3b_ns"][c]["C"]["decode_busy_ms_p50"] = 20.0
    part = rp.verdicts(runs)["H11a"]["parts"]["B_vs_A_c8"]
    assert part["queue_increase_in_decode_steps"] == [1.0, 1.0, 1.0]


def test_contrast_pairs_blocks():
    t = rp.contrast(world(), "w3b_ns", 32, "A", "e2e_p50", "C", "A")
    assert t["diff"]["n"] == 3 and abs(t["diff"]["mean"] - (-0.06)) < 1e-9


def test_h11b_with_a_missing_value_is_insufficient_data():
    runs = world()
    for r in runs:
        if r["arm"] == "B" and r["block"] == 2:
            r["levels"]["w3b_ns"][64]["C"]["usable"] = False  # the c64 trace phase of B in block 2 is unusable
    assert rp.verdicts(runs)["H11b"]["verdict"] == "数据不足"


def test_h11c_undefined_copy_part_does_not_hide_a_refutation():
    runs = world(b_large=0, b_inside=0, idle_drop=-0.01)  # B has no large copies; C's idle share rises in every block
    assert rp.verdicts(runs)["H11c"]["verdict"] == "不支持"
    runs = world(b_large=0, b_inside=0, idle_drop=0.03)
    v = rp.verdicts(runs)["H11c"]
    assert v["verdict"] == "不确定" and v["copy_part"] == "数据不足"


def test_h11a_without_a_ttft_increase_is_not_missing_data():
    v = rp.verdicts(world(extra_ttft=0.0))["H11a"]
    assert v["verdict"] == "不适用（TTFT 没有变长）"
    assert v["parts"]["B_vs_A_c8"]["verdict"] == "不适用（TTFT 没有变长）"


def test_duplicate_runs_are_an_error():
    runs = world()
    runs.append(dict(runs[0]))
    try:
        rp.verdicts(runs)
    except ValueError:
        return
    raise AssertionError("a second run of the same arm and block was accepted")


def test_idle_shares_and_gap_percentiles_are_pooled_over_processes():
    procs = [{"gaps_us": [1000.0, 3000.0], "window_us": 100000.0, "idle_us": 10000.0},
             {"gaps_us": [5000.0], "window_us": 300000.0, "idle_us": 15000.0}]
    p = rp.pool_procs(procs)
    assert p["between_idle_share"] == 9000.0 / 400000.0 and p["total_idle_share"] == 25000.0 / 400000.0
    assert p["gap_p50_ms"] == 3.0 and p["steps_with_gap"] == 3


def test_ttft_segments_use_requests_with_every_segment():
    names = ["a", "b", "ttft"]
    segs = [{"a": 1.0, "b": 2.0, "ttft": 3.0}, {"a": 10.0, "b": None, "ttft": 50.0}]
    m = rp.segment_means(segs, names)
    assert m["means"] == {"a": 1.0, "b": 2.0, "ttft": 3.0} and m["complete"] == 1 and m["requests"] == 2
