"""实验 10 report: padding, GPU time per audio second, graph share, control check, request-audio correspondence,
fairness, verdicts and the correctness pairing (synthetic data only)."""

import importlib.util
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("exp10_report", HERE / "exp10_report.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)


def batch(tokens, prompts=None, flow=100.0, hift=20.0, path="eager", cap=16, policy="fifo", rids=None):
    rids = rids or [f"r{i}" for i in range(len(tokens))]
    return {"size": len(tokens), "token_lengths": tokens, "prompt_token_lengths": prompts or [10] * len(tokens),
            "gpu_flow_ms": flow, "gpu_hift_ms": hift, "flow_solve_path": path, "batch_cap": cap, "batch_policy": policy,
            "request_ids": rids, "request_codec_tokens": tokens}


def test_padding_is_frame_weighted_over_batches():
    b1 = batch([100, 50])  # effective 110 + 60 = 170, padded 2 x 110 = 220
    b2 = batch([40])  # no padding: 50 of 50
    assert math.isclose(rp.padding_ratio([b1]), 1 - 170 / 220)
    assert math.isclose(rp.padding_ratio([b1, b2]), 1 - (170 + 50) / (220 + 50))
    assert rp.padding_ratio([]) is None


def test_gpu_time_per_effective_audio_second_and_graph_share():
    b1 = batch([100, 50], flow=300.0, hift=60.0)  # 150 tokens = 6 s of audio
    b2 = batch([25], flow=90.0, hift=10.0, path="dense_graph")  # 1 s
    assert math.isclose(rp.gpu_ms_per_audio_s([b1, b2]), (360 + 100) / 7.0)
    assert math.isclose(rp.graph_share([b1, b2, batch([5], path="packed_graph"), batch([5], path="packed_eager")]), 0.5)


def test_control_check_compares_every_batch_with_the_phase_setting():
    ok = rp.control_check([batch([1], cap=4, policy="length"), batch([2], cap=4, policy="length")], cap=4, policy="length")
    assert ok == {"batches": 2, "mismatched": 0, "ok": True}
    bad = rp.control_check([batch([1], cap=16, policy="fifo"), batch([2], cap=4, policy="length")], cap=4, policy="length")
    assert bad["mismatched"] == 1 and bad["ok"] is False


def test_each_request_gets_the_audio_of_its_own_codec_tokens():
    batches = [batch([100, 50], rids=["a", "b"])]
    client = {"a": {"output_audio_s": 4.0, "status": "ok"}, "b": {"output_audio_s": 2.0, "status": "ok"}}
    assert rp.correspondence(batches, client) == {"checked": 2, "mismatched": 0}
    client["b"]["output_audio_s"] = 4.0  # b received a's waveform
    assert rp.correspondence(batches, client)["mismatched"] == 1


def test_queue_wait_by_length_class():
    reqs = {f"s{i}": {"queue_ms": 10.0 + i} for i in range(4)} | {f"l{i}": {"queue_ms": 100.0 + i} for i in range(4)}
    audio = {f"s{i}": 5.0 for i in range(4)} | {f"l{i}": 40.0 for i in range(4)}
    w = rp.wait_by_length(reqs, audio)
    assert w["long"]["max"] == 103.0 and w["short"]["max"] == 13.0 and w["split_s"] == 22.5


def cell(e=None, pad=None, audio=None, long_p95=None, long_max=None):
    return {"gpu_ms_per_audio_s": e, "padding": pad, "audio_s_per_s": audio, "wait_long_p95": long_p95, "wait_long_max": long_max,
            "control": {"ok": True}}


def world(e1=12.0, e16=12.5, pad_mix=0.25, pad_w3b=0.10, e_len=10.0, e_fifo=12.0, wait_ratio=1.5, graph_gain1=0.2, graph_gain16=0.0):
    runs = []
    for b in (1, 2, 3):
        for graph in (True, False):
            g = 1.0 / (1 + graph_gain1) if graph else 1.0
            g16 = 1.0 / (1 + graph_gain16) if graph else 1.0
            cells = {("w3b", "fifo1"): cell(e1 * g), ("w3b", "fifo16"): cell(e16 * g16, pad_w3b),
                     ("mix", "fifo16"): cell(e_fifo, pad_mix, 100.0, 50.0, 80.0),
                     ("mix", "length16"): cell(e_len, pad_mix / 3, 110.0, 50.0 * wait_ratio, 80.0 * wait_ratio)}
            runs.append({"block": b, "graph": graph, "cells": cells})
    return runs


def test_verdicts_follow_the_registration():
    v = rp.verdicts(world())
    assert v["H10a"]["verdict"] == "支持"  # 12.5 / 12 >= 0.9
    assert v["H10b"]["verdict"] == "支持"  # 0.25 >= 0.15 and >= 0.10 + 0.05
    assert v["H10c"]["verdict"] == "支持"  # padding lower and E 10 vs 12 (-17 %)
    assert v["H10d"]["verdict"] == "支持"  # long waits +50 %
    assert v["H10e"]["verdict"] == "支持"  # graph gain 20 % at batch 1, 0 at 16
    v = rp.verdicts(world(e16=8.0, pad_mix=0.02, e_len=12.5, wait_ratio=1.05, graph_gain1=0.01))
    assert v["H10a"]["verdict"] == "反驳" and v["H10b"]["verdict"] == "反驳" and v["H10c"]["verdict"] == "反驳"
    assert v["H10d"]["verdict"] == "不支持" and v["H10e"]["verdict"] == "反驳"


def test_correctness_pairs_requests_by_codec_hash_across_phases():
    phase_a = {"x": {"sha": "h1", "stats": [960.0, 1.0, 4.0, 0.5]}, "y": {"sha": "h2", "stats": [1920.0, 2.0, 9.0, 0.6]}}
    phase_b = {"x2": {"sha": "h1", "stats": [960.0, 1.0, 4.0004, 0.5]}, "z": {"sha": "h3", "stats": [960.0, 0.0, 1.0, 0.1]}}
    c = rp.compare_waves(phase_a, phase_b)
    assert c["pairs"] == 1 and c["sample_mismatch"] == 0
    assert math.isclose(c["rel_diff_max"], math.sqrt(0.0004 / 4.0004))  # the reference (b) is the denominator
    phase_b["x2"]["stats"][0] = 961.0
    assert rp.compare_waves(phase_a, phase_b)["sample_mismatch"] == 1


def test_unusable_and_invalid_cells_are_left_out():
    runs = world()
    runs[1]["cells"][("w3b", "fifo1")]["usable"] = False  # block 1, graph off: errors + timeouts > 5 %
    assert rp.verdicts(runs)["H10a"]["verdict"] == "数据不足"
    runs = world()
    runs[0]["cells"][("mix", "length16")]["control"] = {"ok": False}  # block 1, graph on: the control file did not take effect
    v = rp.verdicts(runs)
    assert v["H10c"]["verdict"] == "数据不足" and v["H10d"]["verdict"] == "数据不足"


def test_length_aware_conclusions_stop_when_correctness_fails():
    v = rp.verdicts(world(), correctness_ok=False)
    assert v["H10c"]["verdict"] == v["H10d"]["verdict"] == "停止（正确性规则未通过）"
    assert v["H10a"]["verdict"] == "支持"


def test_validation_candidates_need_five_percent_more_audio_in_every_block():
    runs = world()  # graph on, MIX: length16 110 vs fifo16 100 audio s/s in every block
    cands = rp.validation_candidates(runs)["candidates"]
    assert [(c["workload"], c["setting"]) for c in cands] == [("mix", "length16")]
    assert all(math.isclose(g, 0.1) for g in cands[0]["gain"])
    runs[2]["cells"][("mix", "length16")]["audio_s_per_s"] = 104.0  # block 2, graph on: only +4 %
    assert rp.validation_candidates(runs) == {"candidates": [], "chosen": None}


def test_summary_intervals_leave_out_invalid_cells():
    runs = world()
    for r in runs:
        for c in r["cells"].values():
            c.update(batch_size={"mean": 4.0}, client={"throughput_rps": 1.0, "e2e_s": {"p50": 1.0, "p95": 2.0}},
                     control={"ok": True}, correspondence={"mismatched": 0})
    runs[0]["cells"][("mix", "fifo16")]["usable"] = False  # block 1, graph on
    s = rp.summarize(runs)["graph|mix|fifo16"]["audio_s_per_s"]
    assert s["n"] == 2 and s["runs"][0] is None and s["runs"][1:] == [100.0, 100.0]


def vrun(block, cand_p50=(2.0, 2.0), base_p50=(2.2, 2.4), cand_rps=(20.0, 22.0), base_rps=(20.0, 22.0)):
    rates = (21.0, 23.5)
    lv = lambda p50, rps: {r: {"e2e_p50": p, "throughput_rps": t, "steady": True} for r, p, t in zip(rates, p50, rps)}  # noqa: E731
    return {"block": block, "setting": "length8", "phases": {"fifo16": lv(base_p50, base_rps), "length8": lv(cand_p50, cand_rps)},
            "control": {"fifo16": {"ok": True}, "length8": {"ok": True}}, "restarts": []}


def test_validation_verdict_follows_supplementary_registration_1():
    assert rp.validation_verdict([vrun(b) for b in (1, 2, 3)])["verdict"] == "完整服务上有改善"  # -9 % and -17 %
    small = [vrun(1), vrun(2, cand_p50=(2.15, 2.0)), vrun(3)]  # block 2 at 21 rps: only -2 %
    assert rp.validation_verdict(small)["verdict"] == "不确定"
    slower = [vrun(b, cand_rps=(19.0, 21.0)) for b in (1, 2, 3)]  # throughput below 98 % of fifo16
    assert rp.validation_verdict(slower)["verdict"] == "不确定"
    none = [vrun(b, cand_p50=(2.3, 2.5)) for b in (1, 2, 3)]
    assert rp.validation_verdict(none)["verdict"] == "完整服务上没有改善"
    failed = [vrun(1, cand_p50=(float("inf"), 2.0)), vrun(2), vrun(3)]  # a censored P50 never counts as lower
    assert rp.validation_verdict(failed)["verdict"] == "不确定"
    assert rp.validation_verdict([vrun(1), vrun(2)])["verdict"] == "数据不足"


def test_padding_needs_every_prompt_length():
    assert rp.padding_ratio([batch([100, 200], prompts=[50, None])]) is None  # never counted as 0
    assert math.isclose(rp.padding_ratio([batch([100, 200], prompts=[50, 50])]), 1 - 400 / 500)


def test_audio_is_paired_with_the_codec_tokens_of_the_vocode_order():
    b = {"request_ids": ["a", "c"], "vocode_request_ids": ["a", "b", "c"], "request_codec_tokens": [100, 200, 300]}
    client = {r: {"status": "ok", "output_audio_s": t * 0.04} for r, t in (("a", 100), ("b", 200), ("c", 300))}
    assert rp.correspondence([b], client) == {"checked": 3, "mismatched": 0}


def test_every_request_with_the_same_codec_hash_must_have_the_same_length():
    phases = {"chk_fifo1": {"x": {"sha": "h", "stats": [960.0, 1.0, 4.0, 0.5]}, "y": {"sha": "h", "stats": [1920.0, 1.0, 4.0, 0.5]}},
              "chk_fifo16": {"z": {"sha": "h", "stats": [960.0, 1.0, 4.0, 0.5]}}}
    c = rp.length_consistency(phases)
    assert c == {"hashes": 1, "requests": 3, "inconsistent_hashes": 1}


def test_correctness_needs_evidence():
    comps = ("chk_fifo16", "chk_length16", "chk_length8", "chk_fifo16b", "chk_fifo16b_vs_chk_fifo16")
    good = {"run": "chk", "vs_fifo1": {k: {"pairs": 20, "sample_mismatch": 0} for k in comps},
            "lengths": {"inconsistent_hashes": 0, "hashes": 30}, "control_ok": True, "correspondence": {"checked": 96, "mismatched": 0}}
    runs = [{"cells": {("w3b", "fifo16"): {"correspondence": {"checked": 50, "mismatched": 0}}}}]
    assert rp.correctness(runs, good)["ok"] is True
    empty = {**good, "vs_fifo1": {**good["vs_fifo1"], "chk_length8": {"pairs": 0, "sample_mismatch": 0}}}
    assert rp.correctness(runs, empty)["ok"] is False  # no pairs is no evidence
    missing = {**good, "vs_fifo1": {k: v for k, v in good["vs_fifo1"].items() if k != "chk_length16"}}
    assert rp.correctness(runs, missing)["ok"] is False
    assert rp.correctness(runs, {**good, "lengths": {"inconsistent_hashes": 1, "hashes": 30}})["ok"] is False
    assert rp.correctness(runs, None)["ok"] is None


def test_verdicts_need_all_three_registered_blocks():
    runs = [r for r in world() if r["block"] != 3]
    v = rp.verdicts(runs)
    assert all(v[h]["verdict"].startswith("数据不足") for h in ("H10a", "H10b", "H10c", "H10d", "H10e"))
    assert rp.validation_candidates(runs)["candidates"] == []


def test_validation_candidate_choice_and_correctness():
    runs = world()  # graph on, MIX: length16 +10 % in every block
    for r in runs:
        if r["graph"]:
            r["cells"][("mix", "fifo8")] = {**r["cells"][("mix", "fifo16")], "audio_s_per_s": 110.0}
    c = rp.validation_candidates(runs)
    assert {(x["workload"], x["setting"]) for x in c["candidates"]} == {("mix", "fifo8"), ("mix", "length16")}
    assert (c["chosen"]["workload"], c["chosen"]["setting"]) == ("mix", "fifo8")  # equal gains: the smaller cap
    c = rp.validation_candidates(runs, correctness_ok=False)  # length-aware settings cannot trigger it then
    assert [(x["workload"], x["setting"]) for x in c["candidates"]] == [("mix", "fifo8")]


def test_validation_with_the_control_file_not_in_effect_is_invalid():
    runs = [vrun(b) for b in (1, 2, 3)]
    for r in runs:
        r["control"] = {"fifo16": {"ok": True}, "length8": {"ok": True}}
    assert rp.validation_verdict(runs)["verdict"] == "完整服务上有改善"
    runs[1]["control"]["length8"] = {"ok": False}
    assert rp.validation_verdict(runs)["verdict"].startswith("无效")


def test_h10d_or_is_between_statistics_not_within_a_block():
    runs = world(wait_ratio=1.0)
    p95 = {1: 1.25, 2: 1.0, 3: 1.21}
    mx = {1: 1.0, 2: 1.3, 3: 0.9}
    for r in runs:
        if r["graph"]:
            r["cells"][("mix", "length16")]["wait_long_p95"] = 50.0 * p95[r["block"]]
            r["cells"][("mix", "length16")]["wait_long_max"] = 80.0 * mx[r["block"]]
    assert rp.verdicts(runs)["H10d"]["verdict"] == "不确定"  # neither statistic rises >= 20 % in all blocks


def test_contrasts_are_paired_within_blocks():
    c = rp.contrasts(world())
    k = c["graph|mix|length16_vs_fifo16|gpu_ms_per_audio_s"]
    assert k["n"] == 3 and all(math.isclose(x, 10.0 / 12.0 - 1) for x in k["runs"])  # E 10 vs 12 in every block
    g = c["nograph_vs_graph|w3b|fifo1|gpu_ms_per_audio_s"]
    assert all(math.isclose(x, 0.2) for x in g["runs"])  # graph saves 1 - 1/1.2 of G0's time: G0 / G1 - 1 = 0.2


def test_check_run_restart_fails_correctness():
    comps = ("chk_fifo16", "chk_length16", "chk_length8", "chk_fifo16b", "chk_fifo16b_vs_chk_fifo16")
    chk = {"run": "chk", "vs_fifo1": {k: {"pairs": 20, "sample_mismatch": 0} for k in comps}, "lengths": {"inconsistent_hashes": 0},
           "control_ok": True, "correspondence": {"mismatched": 0}, "restarts": [{"where": "chk_fifo16 c=32 rep=0"}]}
    assert rp.correctness([], chk)["ok"] is False
