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
    return {"gpu_ms_per_audio_s": e, "padding": pad, "audio_s_per_s": audio, "wait_long_p95": long_p95, "wait_long_max": long_max}


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
    cands = rp.validation_candidates(runs)
    assert [(c["workload"], c["setting"]) for c in cands] == [("mix", "length16")]
    assert all(math.isclose(g, 0.1) for g in cands[0]["gain"])
    runs[2]["cells"][("mix", "length16")]["audio_s_per_s"] = 104.0  # block 2, graph on: only +4 %
    assert rp.validation_candidates(runs) == []


def test_summary_intervals_leave_out_invalid_cells():
    runs = world()
    for r in runs:
        for c in r["cells"].values():
            c.update(batch_size={"mean": 4.0}, client={"throughput_rps": 1.0, "e2e_s": {"p50": 1.0, "p95": 2.0}},
                     control={"ok": True}, correspondence={"mismatched": 0})
    runs[0]["cells"][("mix", "fifo16")]["usable"] = False  # block 1, graph on
    s = rp.summarize(runs)["graph|mix|fifo16"]["audio_s_per_s"]
    assert s["n"] == 2 and s["runs"][0] is None and s["runs"][1:] == [100.0, 100.0]
