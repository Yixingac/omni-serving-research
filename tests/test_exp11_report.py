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
