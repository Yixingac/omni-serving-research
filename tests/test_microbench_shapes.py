"""C1 shape analysis: the 520b49ee flow-path rules and the batch-event export (no GPU, no torch)."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "scripts/microbench"
sys.path.insert(0, str(HERE))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fs = _load("flow_shapes")
cs = _load("c1_shapes")


def test_default_dense_graph_table_has_155_shapes_and_44_batch1_buckets():
    shapes = fs.default_dense_shapes()
    assert len(shapes) == 155 and len(set(shapes)) == 155
    assert len([s for s in shapes if s[0] == 1]) == 44
    assert max(b for b, _ in shapes) == 10 and max(f for _, f in shapes) == 1024
    assert len(fs.SEEDTTS_EN_DENSE_PACKED_DIT_CUDA_GRAPH_SHAPES) == 100


@pytest.mark.skipif(not os.environ.get("SGLANG_OMNI_REPO"), reason="set SGLANG_OMNI_REPO to a 520b49ee checkout")
def test_copied_tables_match_the_pinned_sglang_omni_source():
    src = Path(os.environ["SGLANG_OMNI_REPO"]) / "sglang_omni/models/minicpm_o/components/token2wav/flow_graph_shapes.py"
    spec = importlib.util.spec_from_file_location("upstream_shapes", src)
    up = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(up)
    assert tuple(up.build_default_flow_cuda_graph_shapes()) == fs.default_dense_shapes()
    assert tuple(up.SEEDTTS_EN_DENSE_PACKED_DIT_CUDA_GRAPH_SHAPES) == fs.SEEDTTS_EN_DENSE_PACKED_DIT_CUDA_GRAPH_SHAPES


def test_dense_graph_fit_takes_the_smallest_bucket_of_the_same_batch():
    # 195 codec tokens + the 151-token reference = 692 frames (recorded: dense_graph)
    assert fs.row_frames([195]) == [692]
    assert fs.dense_graph_bucket(1, 692) == 704
    assert fs.dense_graph_bucket(1, 1024) == 1024 and fs.dense_graph_bucket(1, 1025) is None
    assert fs.dense_graph_bucket(2, 770) is None  # batch 2 stops at 768
    assert fs.dense_graph_bucket(11, 200) is None  # no graphs above batch 10


def test_solve_path_rules():
    short, long_ = fs.row_frames([150]), fs.row_frames([652])  # 602 and 1606 frames
    assert fs.solve_path(short, variable_length=False) == "dense_graph"
    assert fs.solve_path(long_, variable_length=False) == "eager"
    assert fs.solve_path(short, variable_length=False, flow_graphs=False) == "eager"
    two = fs.row_frames([100, 120])
    assert fs.solve_path(two, variable_length=True, packed_shapes=()) == "packed_eager"  # P0/P1: no packed graphs
    assert fs.solve_path(two, variable_length=True, flow_graphs=False) == "packed_eager"
    assert fs.solve_path(short, variable_length=True) == "dense_graph"  # batch 1 never packs
    # a batch the packed table covers: frames 532 and 542, valid 2 * 1074 = 2148 -> capacity 2256 (slack 108 <= 542),
    # and batch 2 has a dense boundary bucket (544) for the padded 542 frames
    two_fit = fs.row_frames([115, 120])
    assert fs.packed_capacity(2, max(two_fit), 2 * sum(two_fit)) == 2256
    assert fs.solve_path(two_fit, variable_length=True) == "packed_graph"
    # padded above 1024 frames: no packed graph whatever the capacity table holds
    assert fs.packed_capacity(2, 1030, 2 * 2000) is None


def test_batch_shape_padding_and_attention_work():
    s = fs.batch_shape([349, 200], [151, 151])
    assert s["frames"] == [1000, 702] and s["padded_frames"] == 1000
    assert s["padding_ratio"] == pytest.approx(1 - 1702 / 2000)
    assert s["useful_audio_s"] == pytest.approx(549 * 0.04)
    assert s["dense_attention_elems"] == 2 * 2 * 1000 * 1000
    assert s["packed_attention_elems"] == 2 * (1000 ** 2 + 702 ** 2)
    assert s["dense_sdpa_shape"] == [4, 8, 1000, 64]


def test_summarize_group_and_path_reproduction():
    recs = [{"token_lengths": [195], "prompt_token_lengths": [151], "flow_solve_path": "dense_graph"},
            {"token_lengths": [652], "prompt_token_lengths": [151], "flow_solve_path": "eager"},
            {"token_lengths": [300, 200, 100], "prompt_token_lengths": [151] * 3, "flow_solve_path": "eager"}]
    g = fs.summarize_group(recs)
    assert g["batches"] == 3 and g["requests"] == 5
    assert g["recorded_dense_path_reproduced"] == 1.0
    d = g["request_share_by_path"]["dense_default"]
    assert d == {"dense_graph": pytest.approx(0.2), "eager": pytest.approx(0.8)}
    assert g["request_share_by_path"]["packed_no_graphs"]["packed_eager"] == pytest.approx(0.6)
    nog = fs.summarize_group([{**r, "flow_solve_path": "eager"} for r in recs], recorded_with_graphs=False)
    assert nog["recorded_dense_path_reproduced"] == 1.0


def test_sample_batches_is_seeded_uniform_and_keeps_order():
    batches = [{"token_lengths": [i]} for i in range(100)]
    a, b = fs.sample_batches(batches, 10, 7), fs.sample_batches(batches, 10, 7)
    assert a == b and len(a) == 10
    assert [x["token_lengths"][0] for x in a] == sorted(x["token_lengths"][0] for x in a)
    assert fs.sample_batches(batches[:5], 10, 7) == batches[:5]


def test_gpu_ms_per_audio_s_is_audio_weighted():
    rows = [{"gpu_ms": 10.0, "useful_audio_s": 1.0}, {"gpu_ms": 30.0, "useful_audio_s": 9.0}, {"gpu_ms": None, "useful_audio_s": 5.0}]
    assert fs.gpu_ms_per_audio_s(rows) == pytest.approx(4.0)


def _event(name, pid, batch_id, **md):
    return json.dumps({"event_name": name, "pid": pid, "timestamp_ns": 10 * batch_id, "metadata": {"batch_id": batch_id, **md}})


def test_vocode_batches_dedupes_per_batch_and_derives_missing_prompts():
    lines = [
        _event("exp_batch_formed", 1, 5, batch_cap=8, batch_policy="length", window_size=64),
        _event("exp_vocode", 1, 5, token_lengths=[195, 100], prompt_token_lengths=[151, 151], flow_solve_path="eager",
               flow_solve_frames=692, gpu_flow_start_to_flow_end_ms=70.0, gpu_flow_end_to_hift_end_ms=15.0),
        _event("exp_vocode", 1, 5, token_lengths=[195, 100], prompt_token_lengths=[151, 151]),  # second request, same batch
        _event("exp_vocode", 2, 5, token_lengths=[652], flow_solve_path="eager", flow_solve_frames=1606),  # instr: no prompts
        '{"event_name": "stage_dispatch", "metadata": {}}',
    ]
    out = cs.vocode_batches(lines)
    assert len(out) == 2
    a = next(r for r in out if r["pid"] == 1)
    assert a["rows"] == 2 and a["batch_policy"] == "length" and a["gpu_flow_ms"] == 70.0 and not a["prompt_derived"]
    b = next(r for r in out if r["pid"] == 2)
    assert b["prompt_token_lengths"] == [151] and b["prompt_derived"]


def test_run_and_phase_classification():
    assert cs.classify_run("20261118_exp10_graph_block2_run01") == ("exp10_t2_k4_c2_graph", 2)
    assert cs.classify_run("20261118_exp10_validation_t2_k2_c4_length8_block3_run01") == ("exp10val_t2_k2_c4", 3)
    assert cs.classify_run("20261117_exp09_t2_k1_c5_block1_run01") == ("exp09_t2_k1_c5", 1)
    assert cs.classify_run("20261118_exp09_t2_k6_shared_block1_run01") == ("exp09_t2_k6_shared", 1)
    assert cs.classify_run("20261118_exp10_check_run01") is None
    assert cs.classify_phase("mix_length8") == ("mix", "length8")
    assert cs.classify_phase("val_fifo16") == ("w3b", "fifo16")
    assert cs.classify_phase("w3l_closed_B") == ("w3l", "fifo16")
    assert cs.classify_phase("w3b_closed_A") is None and cs.classify_phase("smoke_w3b") is None


def test_export_reads_code2wav_event_files(tmp_path):
    run = tmp_path / "experiments" / "20261118_exp10_graph_block1_run01" / "raw" / "server_events" / "w3b_fifo16" / "s0"
    run.mkdir(parents=True)
    (run / "events_code2wav@r0_1.jsonl").write_text(
        _event("exp_vocode", 1, 3, token_lengths=[300], prompt_token_lengths=[151], flow_solve_path="eager") + "\n")
    (run / "events_talker@r0_2.jsonl").write_text("")
    out = tmp_path / "b.jsonl"
    assert cs.export(tmp_path, ["20261118_exp10_graph_block1_run01"], out) == 1
    rec = json.loads(out.read_text())
    assert rec["source"] == "exp10_t2_k4_c2_graph" and rec["workload"] == "w3b" and rec["setting"] == "fifo16" and "pid" not in rec
    rep = cs.replay_sets([rec], per_group=5, seed=1)
    assert rep["exp10_t2_k4_c2_graph|w3b|fifo16"]["batches"] == [[300]]
