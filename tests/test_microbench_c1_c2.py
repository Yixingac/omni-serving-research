"""C1/C2 microbenchmark harness logic and quality gates, without a GPU (torch is imported only inside GPU code)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1] / "scripts/microbench"
sys.path.insert(0, str(HERE))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


c1 = _load("c1_flow_packed")
c2 = _load("c2_hift")
qe = _load("quality_eval")


def test_row_tokens_is_deterministic_real_content_of_the_requested_length():
    pool = list(range(100))
    a = c1.row_tokens(pool, 30, "g", 1, 2)
    assert a == c1.row_tokens(pool, 30, "g", 1, 2) and len(a) == 30
    assert a != c1.row_tokens(pool, 30, "g", 1, 3)
    assert all(x in pool for x in a)
    assert len(c1.row_tokens(pool, 250, "g", 0, 0)) == 250  # longer than the pool: tiled
    with pytest.raises(ValueError):
        c1.row_tokens([], 5, "g")


def test_select_batches_uses_all_or_a_seeded_subset():
    replay = {"groups": {"g": {"batches": [[i] for i in range(50)]}}}
    assert c1.select_batches(replay, "g", 0) == [[i] for i in range(50)]
    s = c1.select_batches(replay, "g", 10)
    assert len(s) == 10 and s == c1.select_batches(replay, "g", 10)


def test_classify_path_from_runner_counters():
    b = {"graph_replays": 5, "packed_step_replays": 1, "packed_graph_replays": 2}
    assert c1.classify_path(b, {**b, "graph_replays": 15}, variable_length=False, rows=1) == "dense_graph"
    assert c1.classify_path(b, b, variable_length=False, rows=4) == "eager"
    assert c1.classify_path(b, b, variable_length=True, rows=4) == "packed_eager"
    assert c1.classify_path(b, b, variable_length=True, rows=1) == "eager"
    assert c1.classify_path(b, {**b, "packed_graph_replays": 12, "packed_step_replays": 11, "graph_replays": 25},
                            variable_length=True, rows=3) == "packed_graph"
    assert c1.classify_path(b, {**b, "packed_graph_replays": 12}, variable_length=True, rows=3) == "packed_dit_graph"


def test_summarize_repeat_computes_audio_weighted_E_and_path_shares():
    rows = [{"batch": 1, "useful_audio_s": 10.0, "flow_ms": 60.0, "hift_ms": 20.0, "wall_ms": 90.0, "path": "dense_graph"},
            {"batch": 3, "useful_audio_s": 30.0, "flow_ms": 240.0, "hift_ms": 60.0, "wall_ms": 320.0, "path": "eager"}]
    s = c1.summarize_repeat(rows)
    assert s["E_flow_ms_per_audio_s"] == pytest.approx(300 / 40) and s["E_c2w_ms_per_audio_s"] == pytest.approx(380 / 40)
    assert s["request_share_by_path"] == {"dense_graph": 0.25, "eager": 0.75} and s["graph_request_share"] == 0.25


def test_paired_change_and_c1_gate():
    ch = c1.paired_change([9.0, 9.4, 9.3], [10.0, 10.0, 10.0])
    assert ch["n"] == 3 and ch["max"] == pytest.approx(-0.06)
    g = c1.PRIMARY_GROUPS
    per = {g[0]: {"E_c2w_change_P1_vs_D": ch}, g[1]: {"E_c2w_change_P1_vs_D": c1.paired_change([9.6, 9.0, 9.0], [10.0] * 3)}}
    assert c1.c1_gate(per, True) == {"go": True, "groups_passing": [g[0]], "reason": "E(P1) <= 0.95 E(D) in every repeat"}
    assert c1.c1_gate(per, False)["go"] is False and c1.c1_gate(per, None)["reason"] == "quality gate not evaluated"
    one_bad = {g[0]: {"E_c2w_change_P1_vs_D": c1.paired_change([9.0, 9.0, 9.6], [10.0] * 3)}}
    assert c1.c1_gate(one_bad, True)["go"] is False  # stable across repeats is required
    assert c1.paired_change([1.0], [1.0, 2.0]) == {"n": 0}


def test_quality_batches_and_arm_table():
    items = [{"asset_id": str(i)} for i in range(19)]
    qb = c1.quality_batches(items)
    assert [len(b) for b in qb] == [8, 8, 3]
    assert c1.ARMS["P0"]["variable_length"] and not c1.ARMS["P0"]["block_compile"] and not c1.ARMS["P0"]["packed_graphs"]
    assert c1.ARMS["P1"]["block_compile"] and not c1.ARMS["D"]["variable_length"] and c1.ARMS["REF"]["dtype"] == "float32"


def test_c2_arms_orders_and_hift_calls():
    assert c2.arm_parts("BCD") == {"B", "C", "D"} and c2.arm_parts("Dbf") == {"Dbf"}
    with pytest.raises(ValueError):
        c2.arm_parts("E")
    orders = c2.interleaved_orders(["A", "B", "C", "D"], 3, 1)
    assert len(orders) == 3 and all(sorted(o) == ["A", "B", "C", "D"] for o in orders)
    assert orders == c2.interleaved_orders(["A", "B", "C", "D"], 3, 1)
    assert c2.hift_calls([300, 200, 300]) == [(300, [0, 2]), (200, [1])]  # same length -> one HiFT call


def test_c2_perf_gates():
    e = {"g": {"A": [2.0, 2.0, 2.0], "BC": [1.9, 1.9, 1.9], "D": [1.5, 1.55, 1.58]}}
    out = c2.c2_perf_gates(e, {"g": 8.0})
    assert out["g"]["fold_plus_layout_c2w_share"] == pytest.approx(0.1 / 10.0)
    assert out["g"]["fold_plus_layout_go"] is False  # 1 % of code2wav time < 3 %
    assert out["g"]["half_perf_go"] is True and len(out["g"]["half_change_per_repeat"]) == 3
    e2 = {"g": {"A": [2.0] * 3, "B": [1.8] * 3, "C": [1.9] * 3, "D": [1.7, 1.5, 1.5]}}
    out2 = c2.c2_perf_gates(e2, {"g": 8.0})
    assert out2["g"]["fold_plus_layout_source"] == "B + C (summed)" and out2["g"]["fold_plus_layout_go"] is True
    assert out2["g"]["half_perf_go"] is False  # one repeat only -15 %


def test_wav_bytes_round_trip():
    import io
    import wave

    x = np.sin(np.linspace(0, 100, 2400)) * 0.5
    with wave.open(io.BytesIO(qe.wav_bytes(x)), "rb") as w:
        assert w.getframerate() == 24000 and w.getnframes() == 2400 and w.getsampwidth() == 2


def _write_arm(root, arm, labels):
    """labels: {label: {asset_id: waveform}}; tokens = len // 960."""
    (root / arm / "quality").mkdir(parents=True, exist_ok=True)
    quality = {}
    for label, waves in labels.items():
        files = []
        for aid, w in waves.items():
            np.save(root / arm / "quality" / f"{label}_{aid}_wave.npy", w.astype(np.float32))
            np.save(root / arm / "quality" / f"{label}_{aid}_mel.npy", np.abs(w[:800]).reshape(80, 10).astype(np.float32))
            files.append({"asset_id": aid, "tokens": w.size // 960, "transcript": "how many bolts",
                          "wave": f"quality/{label}_{aid}_wave.npy", "mel": f"quality/{label}_{aid}_mel.npy"})
        quality[label] = files
    (root / arm / "arm_result.json").write_text(json.dumps({"quality": quality}))


def _speechy(seed, n=9600):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 24000
    return (0.1 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(n)).astype(np.float64)


def test_quality_eval_c1_gate_without_asr_is_not_passed(tmp_path):
    ref = {"q1": _speechy(1), "q2": _speechy(2)}
    near = {k: v + 1e-5 for k, v in ref.items()}
    _write_arm(tmp_path, "REF", {"one": ref})
    _write_arm(tmp_path, "D", {"batched": near, "repeat": near, "alone": near})
    _write_arm(tmp_path, "P1", {"batched": {k: v + 2e-5 for k, v in ref.items()}})
    a = type("A", (), {"root": str(tmp_path), "asr_url": None, "asr_model": "m"})()
    out = qe.c1(a)
    g = out["gates"]["P1"]
    assert g["length_ok"] and g["mcd_ok"] and g["asr_ok"] is None and g["pass"] is False  # ASR is part of the gate
    assert g["registered_rule_literal"] is False  # D's literal repeat is identical, P1 is not
    assert out["floor_repeat_vs_d"]["summary"]["bit_identical_all"] is True


def test_quality_eval_c2_fold_layout_and_half_gates(tmp_path):
    ref = {"q1": _speechy(1), "q2": _speechy(2)}
    _write_arm(tmp_path, "A", {"s1": ref, "s2": {k: v + 1e-3 for k, v in ref.items()}, "eager_s1": ref})
    _write_arm(tmp_path, "B", {"s1": ref, "eager_s1": ref})
    res_b = json.loads((tmp_path / "B" / "arm_result.json").read_text())
    res_b["fold_check"] = {"modules": 82, "max_ulp": 0}
    (tmp_path / "B" / "arm_result.json").write_text(json.dumps(res_b))
    _write_arm(tmp_path, "C", {"s1": ref, "eager_s1": {k: v * (1 + 1e-3) for k, v in ref.items()}})  # 1e-3 > 1e-4
    _write_arm(tmp_path, "D", {"s1": {k: v + 1e-6 for k, v in ref.items()}})
    a = type("A", (), {"root": str(tmp_path), "asr_url": None, "asr_model": "m"})()
    gates = qe.c2(a)["gates"]
    assert gates["B"]["kind"] == "fold" and gates["B"]["pass"] is True
    assert gates["C"]["kind"] == "layout" and gates["C"]["pass"] is False
    assert gates["D"]["kind"] == "half" and gates["D"]["mcd_ok"] is True and gates["D"]["pass"] is False  # no ASR yet


def test_asr_pass_rule():
    d = {"exact_matches": 20, "wer": 0.05}
    floor = {"exact_matches": 19, "wer": 0.06}
    assert qe.asr_pass({"exact_matches": 18, "wer": 0.064}, [d, floor]) is True  # one more miss, WER within +0.5 pt
    assert qe.asr_pass({"exact_matches": 17, "wer": 0.05}, [d, floor]) is False
    assert qe.asr_pass({"exact_matches": 20, "wer": 0.07}, [d, floor]) is False
    assert qe.asr_pass(None, [d]) is None and qe.asr_pass(d, [None]) is None
