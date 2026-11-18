"""实验 19: which split is best for mixed and agent workloads, from the plan v2 runs."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("exp19_report", HERE / "exp19_report.py")
r19 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r19)

S = 1_000_000_000


def mix_rec(i, wid, conc, ttft_s, e2e_s, audio_s=None, warmup=False, status="ok"):
    out = ["text", "audio"] if wid.startswith("W3b") else ["text"]
    return {"request_id": f"m{i}", "workload_id": wid, "concurrency": conc, "warmup": warmup, "status": status,
            "output_modalities": out, "sent_ns": 0, "first_text_ns": int(ttft_s * S), "done_ns": int(e2e_s * S),
            "output_audio_s": audio_s}


def test_thinker_gpus_per_layout():
    assert [r19.thinker_gpus(x) for x in ("s2x6", "s3x5", "s5x3", "s6x2", "p4", "r8")] == [2, 3, 5, 6, 4, 8]


def test_near_best_keeps_everything_within_the_tolerance_of_the_best():
    assert r19.near_best({"a": 10.0, "b": 9.8, "c": 9.0}, tol=0.03) == ["a", "b"]
    assert r19.near_best({"a": 1.0, "b": 1.02, "c": 1.2}, tol=0.03, higher_is_better=False) == ["a", "b"]
    assert r19.near_best({"a": 5.0, "b": 5.0}, tol=0.0) == ["a", "b"]  # an exact tie keeps both


def test_more_thinker_verdict_needs_every_near_best_layout_to_have_more_thinker_gpus():
    assert r19.more_thinker_verdict(["p4", "s4x4"], reference=2)["verdict"] == "支持"
    assert r19.more_thinker_verdict(["s4x4", "s2x6"], reference=2)["verdict"] == "不确定"
    assert r19.more_thinker_verdict(["s2x6"], reference=2)["verdict"] == "不支持"
    assert r19.more_thinker_verdict(["s1x7", "s2x6"], reference=2)["verdict"] == "不支持"


def test_mix_level_stats_by_type_and_slo():
    recs = [mix_rec(0, "W1_text_text", 16, 0.2, 1.0), mix_rec(1, "W1_text_text", 16, 1.5, 2.0),   # second misses TTFT <= 1 s
            mix_rec(2, "W3b_text_speech", 16, 0.2, 4.0, audio_s=10.0),                               # 0.4 s per audio s: ok
            mix_rec(3, "W3b_text_speech", 16, 0.2, 6.0, audio_s=10.0),                               # 0.6: misses
            mix_rec(4, "W2_audio_text", 16, 0.3, 1.0), mix_rec(5, "W2_audio_text", 16, 0.3, 1.0, status="error"),
            mix_rec(6, "W1_text_text", 16, 9.0, 9.0, warmup=True)]
    lv = r19.mix_level_stats(recs)[16]
    assert lv["n"] == 6 and lv["slo_all"] == 3 / 6
    assert lv["by_type"]["W1"]["n"] == 2 and lv["by_type"]["W1"]["slo"] == 0.5
    assert lv["by_type"]["W3b"]["slo"] == 0.5 and lv["by_type"]["W2"]["slo"] == 0.5 and lv["by_type"]["W2"]["errors"] == 1
    assert abs(lv["by_type"]["W1"]["ttft_p90_s"] - 1.37) < 1e-9  # linear interpolation between 0.2 and 1.5


def test_ttft_stats_skips_failed_requests():
    recs = [{"status": "ok", "sent_ns": 0, "first_text_ns": int(x * S)} for x in (0.1, 0.2, 0.3, 0.4)]
    recs.append({"status": "error", "sent_ns": 0, "first_text_ns": None})
    st = r19.ttft_stats(recs)
    assert st["n"] == 4 and abs(st["p50_s"] - 0.25) < 1e-9


def test_routing_comparison_and_verdict():
    lo = {"text_c64": {"prefix_hit": 0.86, "ttft_p50_s": 0.100}, "text_c256": {"prefix_hit": 0.86, "ttft_p50_s": 0.200}}
    sa = {"text_c64": {"prefix_hit": 0.88, "ttft_p50_s": 0.101}, "text_c256": {"prefix_hit": 0.88, "ttft_p50_s": 0.196}}
    cmp = r19.routing_comparison(lo, sa)
    assert [round(c["hit_gain_pts"], 6) for c in cmp] == [2.0, 2.0]
    assert [round(c["ttft_p50_rel"], 3) for c in cmp] == [0.01, -0.02]
    v = r19.routing_verdict(cmp)
    assert v["hit"] == "小（< 3 个百分点）" and v["ttft"] == "不支持"
    faster = [{"level": "x", "hit_gain_pts": 5.0, "ttft_p50_rel": -0.10}, {"level": "y", "hit_gain_pts": 6.0, "ttft_p50_rel": -0.08}]
    assert r19.routing_verdict(faster)["ttft"] == "支持"


def test_role_busy_in_a_window():
    util = {0: [(10.0, 50.0, 0), (11.0, 70.0, 0), (20.0, 99.0, 0)], 1: [(10.5, 20.0, 0)], 2: [(10.2, 0.0, 0)]}
    busy = r19.role_busy(util, {"thinker": [0], "talker": [1, 2]}, 10.0, 12.0)
    assert busy == {"thinker": 60.0, "talker": 10.0}


def _agent(layout, sessions):
    phases = {f"text_c{c}": {"agent": "text", "concurrency": c, "routing": "least_outstanding", "tool_latency_s": 0.0, "session_p50_s": v}
              for c, v in sessions.items()}
    return {"layout": layout, "phases": phases}


def test_verdicts_widen_the_near_best_set_with_the_tolerance():
    w3b = [{"layout": "s2x6", "closed_max_rps": 20.0}, {"layout": "s3x5", "closed_max_rps": 17.0}]
    mix = [{"layout": "s3x5", "closed_max_rps": 30.0, "closed": [{"concurrency": 256, "slo_all": 0.8}]},
           {"layout": "s2x6", "closed_max_rps": 28.5, "closed": [{"concurrency": 256, "slo_all": 0.1}]}]
    agent = [_agent("r8", {256: 3.0, 512: 4.0}), _agent("s2x6", {256: 8.0, 512: 14.0})]
    narrow = r19.verdicts(w3b, mix, agent, tol=0.03)
    wide = r19.verdicts(w3b, mix, agent, tol=0.06)
    assert narrow["B-H1"]["near_best"] == ["s3x5"] and narrow["B-H1"]["verdict"] == "支持"
    assert wide["B-H1"]["near_best"] == ["s3x5", "s2x6"] and wide["B-H1"]["verdict"] == "不确定"  # 28.5 is 5 % below 30
    assert narrow["B-H2"]["verdict"] == "支持" and wide["B-H2"]["verdict"] == "支持"


def test_the_speech_only_reference_stays_two_thinker_gpus_when_full_replicas_come_near():
    # 8 full replicas share every GPU between thinker and talker, so they are not a point on the thinker:talker axis;
    # their being within the tolerance of 2+6 on W3b must not move the reference to 8 GPUs.
    w3b = [{"layout": "s2x6", "closed_max_rps": 20.0}, {"layout": "r8", "closed_max_rps": 19.0}]
    mix = [{"layout": "s3x5", "closed_max_rps": 30.0, "closed": [{"concurrency": 256, "slo_all": 0.8}]}]
    v = r19.verdicts(w3b, mix, [], tol=0.06)
    assert v["w3b_reference"]["near_best"] == ["s2x6", "r8"]
    assert v["B-H1"]["reference_thinker_gpus"] == 2 and v["B-H1"]["verdict"] == "支持"
