"""P1 analysis on synthetic records: warm-up cut, goodput, length groups, starvation, registered verdicts."""

import importlib.util
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("ppv_p1_report", HERE / "ppv_p1_report.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

S = 1_000_000_000


def rec(i, sched_s, e2e_s, ttft_s=0.3, audio_s=6.0, status="ok"):
    return {"request_id": f"r{i}", "scheduled_ns": int(sched_s * S), "sent_ns": int(sched_s * S),
            "first_text_ns": int((sched_s + ttft_s) * S), "done_ns": int((sched_s + e2e_s) * S),
            "status": status, "output_audio_s": audio_s, "rate_rps": 10.0}


def test_prerun_capacity_uses_the_busy_window_and_reports_the_registered_capacity():
    # closed loop: 10 measured requests sent at 0..9 s, each done 2.5 s later; one warm-up request
    recs = [rec(i, i, 2.5) for i in range(10)] + [{**rec(99, 0, 0.5), "warmup": True}]
    c = rp.prerun_capacity(recs)
    assert c["measured"] == 10 and c["busy_window_s"] == 9.0
    # completions in (0, 9] s: the warm-up request done at 0.5 s and measured ones done at 2.5 .. 8.5 s (实验 9's busy
    # window counts every completion inside it; Erratum E-P1-X)
    assert math.isclose(c["x_rps"], 8 / 9)
    assert math.isclose(c["registered_capacity_rps"], 10 / 11.5)  # first send 0 s -> last completion 11.5 s
    assert rp.prerun_rates({"w3b": 29.3}) == {"w3b": {"x090": 26.5, "x100": 29.5, "x110": 32.0}}


def test_warmup_is_dropped_by_scheduled_time():
    recs = [rec(i, 0.5 * i, 1.0) for i in range(40)]  # 0 .. 19.5 s
    meas = rp.measured(recs, 15.0)
    assert [r["request_id"] for r in meas] == [f"r{i}" for i in range(30, 40)]


def test_client_cell_goodput_failures_and_usability():
    recs = [rec(i, i * 0.1, 2.0) for i in range(80)]           # e2e / audio = 0.33, ttft 0.3: meets the SLO
    recs += [rec(100 + i, 8 + i * 0.1, 4.0) for i in range(15)]  # e2e / audio = 0.67: misses
    recs += [rec(200 + i, 9.5 + i * 0.1, 1.0, ttft_s=1.5) for i in range(3)]  # TTFT over 1 s: misses
    recs += [rec(300 + i, 9.8 + i * 0.1, 1.0, status="timeout") for i in range(2)]
    c = rp.client_cell(recs, 10.0)
    assert c["measured"] == 100 and c["failed_or_timeout"] == 2 and c["usable"] is True
    assert math.isclose(c["goodput_fraction"], 0.80)
    assert c["e2e_p99"] == math.inf  # two of 100 failed: the censored P99 is infinite
    bad = rp.client_cell(recs + [rec(400 + i, 9.9, 1.0, status="error") for i in range(10)], 10.0)
    assert bad["usable"] is False


def test_length_groups_split_at_the_median_and_report_p99_only_with_enough_requests():
    q = {f"r{i}": float(i) for i in range(300)}
    a = {f"r{i}": 5.0 if i < 150 else 20.0 for i in range(300)}
    g = rp.length_groups(q, a)
    assert g["split_s"] == 12.5 and g["short"]["n"] == 150 and g["long"]["n"] == 150
    assert g["short"]["p99"] is not None and g["short"]["max"] == 149.0
    small = rp.length_groups({"a": 1.0, "b": 2.0}, {"a": 5.0, "b": 20.0})
    assert small["short"]["p99"] is None and "100" in small["short"]["p99_n_note"]


def test_starvation_share_against_the_fifo_reference():
    q = {"s1": 10.0, "s2": 400.0, "l1": 50.0}
    a = {"s1": 5.0, "s2": 5.0, "l1": 30.0}
    ref = {"short": {"p95": 100.0}, "long": {"p95": 100.0}}
    out = rp.starvation_share(q, a, 10.0, ref)
    assert out == {"short": 0.5, "long": 0.0}


def cell(goodput=10.0, short_p99=100.0, audio=50.0, e2e_p95=3.0, steady=True, offered=10.0, usable=True):
    return {"client": {"goodput_rps": goodput, "audio_s_per_s": audio, "e2e_p95": e2e_p95, "steady": steady,
                       "offered_rps": offered, "usable": usable},
            "queue": {"short": {"p99": short_p99}}}


def runs_with(mix_fifo, mix_w1000, mix_l8=None, w3b=None):
    out = []
    for b in (1, 2, 3):
        cells = {}
        for tag in ("x100", "x110"):
            cells[("mix", "fifo16", tag)] = mix_fifo(b, tag)
            cells[("mix", "length8_w1000", tag)] = mix_w1000(b, tag)
            if mix_l8:
                cells[("mix", "length8", tag)] = mix_l8(b, tag)
        for tag in ("x090", "x100", "x110"):
            if w3b:
                cells[("w3b", "fifo16", tag)], cells[("w3b", "length8_w1000", tag)] = w3b(b, tag)
        out.append({"block": b, "cells": cells})
    return out


def test_h1a_support_refute_uncertain_and_missing():
    sup = runs_with(lambda b, t: cell(goodput=10.0), lambda b, t: cell(goodput=10.6))
    assert rp.verdict_h1a(sup)["verdict"] == "支持"
    ref = runs_with(lambda b, t: cell(goodput=10.0), lambda b, t: cell(goodput=9.9 if b != 2 else 10.0))
    assert rp.verdict_h1a(ref)["verdict"] == "反驳"
    unc = runs_with(lambda b, t: cell(goodput=10.0), lambda b, t: cell(goodput=10.6 if b != 3 else 10.2))
    assert rp.verdict_h1a(unc)["verdict"] == "不确定"
    miss = runs_with(lambda b, t: cell(goodput=10.0), lambda b, t: cell(goodput=10.6, usable=b != 2))
    assert rp.verdict_h1a(miss)["verdict"] == "数据不足"


def test_h1b_needs_both_the_tail_cut_and_the_audio_rate():
    good = runs_with(lambda b, t: cell(), lambda b, t: cell(short_p99=70.0, audio=49.0), lambda b, t: cell(short_p99=100.0, audio=50.0))
    assert rp.verdict_h1b(good)["verdict"] == "支持"
    slow = runs_with(lambda b, t: cell(), lambda b, t: cell(short_p99=70.0, audio=45.0), lambda b, t: cell(short_p99=100.0, audio=50.0))
    assert rp.verdict_h1b(slow)["verdict"] == "不确定"


def test_h1c_every_rate_every_block():
    ok = runs_with(lambda b, t: cell(), lambda b, t: cell(), w3b=lambda b, t: (cell(e2e_p95=3.0), cell(e2e_p95=3.1)))
    assert rp.verdict_h1c(ok)["verdict"] == "支持"
    worse = runs_with(lambda b, t: cell(), lambda b, t: cell(),
                      w3b=lambda b, t: (cell(e2e_p95=3.0), cell(e2e_p95=3.3 if t == "x110" and b == 1 else 3.0)))
    assert rp.verdict_h1c(worse)["verdict"] == "不确定"


def test_stable_capacity_never_uses_an_unsteady_cell():
    runs = [{"block": 1, "cells": {("w3b", "fifo16", "x090"): cell(offered=26.5, steady=True),
                                   ("w3b", "fifo16", "x100"): cell(offered=29.5, steady=False),
                                   ("w3b", "fifo16", "x110"): cell(offered=32.0, steady=False),
                                   ("w3b", "length8", "x090"): cell(offered=26.5, steady=True),
                                   ("w3b", "length8", "x100"): cell(offered=29.5, steady=True),
                                   ("w3b", "length8", "x110"): cell(offered=32.0, steady=True)}}]
    s = rp.stable_capacity(runs)
    assert s["w3b|fifo16"]["1"]["stable_rps_lower_bound"] == 26.5 and s["w3b|fifo16"]["1"]["note"] is None
    assert s["w3b|length8"]["1"]["stable_rps_lower_bound"] == 32.0 and "lower bound" in s["w3b|length8"]["1"]["note"]


def test_control_matches_checks_cap_policy_and_max_wait():
    ctl = {"max_batch_size": 8, "policy": "length", "max_wait_ms": 1000}
    assert rp.control_matches([{"batch_cap": 8, "batch_policy": "length", "max_wait_ms": 1000.0}], ctl) is True
    assert rp.control_matches([{"batch_cap": 8, "batch_policy": "length", "max_wait_ms": -1.0}], ctl) is False
    assert rp.control_matches([{"batch_cap": 16, "batch_policy": "fifo", "max_wait_ms": -1.0}],
                              {"max_batch_size": 16, "policy": "fifo", "max_wait_ms": None}) is True
    assert rp.control_matches([], ctl) is None


def test_analyze_run_reads_cells_from_a_run_directory(tmp_path):
    import json

    exp = tmp_path / "20261119_ppv_p1_block2_run01"
    (exp / "raw").mkdir(parents=True)
    phases = [{"phase": "smoke_w3b", "kind": "smoke"},
              {"phase": "ctl_mix_length8_w1000_x100", "kind": "control",
               "control": {"max_batch_size": 8, "policy": "length", "window": 64, "max_wait_ms": 1000}},
              {"phase": "mix_length8_w1000_x100", "kind": "open_loop"}]
    (exp / "result.json").write_text(json.dumps({"experiment_id": "ppv_p1_block2", "phases": phases}))
    (exp / "config.json").write_text(json.dumps({"phases": [{"kind": "open_loop", "name": "mix_length8_w1000_x100", "warmup_s": 15}]}))
    recs = [rec(i, 0.1 * i, 2.0) for i in range(900)]  # 0 .. 89.9 s
    (exp / "raw" / "requests_mix_length8_w1000_x100.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    run = rp.analyze_run(exp)
    assert run["block"] == 2 and list(run["cells"]) == [("mix", "length8_w1000", "x100")]
    c = run["cells"][("mix", "length8_w1000", "x100")]["client"]
    assert c["measured"] == 750 and c["offered_rps"] == 10.0 and c["usable"] is True and c["steady"] is True


def test_busy_window_x_is_never_below_the_registered_capacity_in_a_closed_loop():
    """Closed loop at 4 in flight: a warm-up wave of 4, then 12 measured requests, each 1 s long."""
    recs, t = [], 0.0
    for wave in range(4):
        for k in range(4):
            r = rec(wave * 4 + k, wave * 1.0, 1.0)
            if wave == 0:
                r["warmup"] = True
            recs.append(r)
    c = rp.prerun_capacity(recs)
    assert c["x_rps"] >= c["registered_capacity_rps"]
