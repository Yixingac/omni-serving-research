"""Pure helpers of the 实验 1 scripts (scripts/analysis/exp01_*.py); no traces or GPUs needed."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tl = load("exp01_gpu_timeline")
kt = load("exp01_kernel_timeline")


def test_utc_offset_from_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"utc_offset": "-0500"}))
    assert tl.utc_offset_s(tmp_path) == -5 * 3600
    (tmp_path / "manifest.json").write_text(json.dumps({"utc_offset": "+0530"}))
    assert tl.utc_offset_s(tmp_path) == 5 * 3600 + 30 * 60


def test_request_segments_follow_the_serial_chain():
    ms = 1_000_000
    ev = lambda stage, name, t: {"stage": stage, "event_name": name, "timestamp_ns": t * ms}  # noqa: E731
    evs = [ev("coordinator", "request_admission", 0),
           ev("s0/thinker@r0", "scheduler_queue_enter", 2), ev("s0/thinker@r0", "scheduler_prefill_start", 3),
           ev("s0/thinker@r0", "scheduler_prefill_end", 30), ev("s0/thinker@r0", "model_path_end", 1300),
           ev("s0/talker@r1", "scheduler_queue_enter", 1305), ev("s0/talker@r1", "scheduler_prefill_start", 1306),
           ev("s0/talker@r1", "model_path_end", 1850),
           ev("s0/code2wav@r1", "stage_input_received", 1852), ev("s0/code2wav@r1", "stage_dispatch", 1852),
           ev("s0/code2wav@r1", "stage_complete", 2000), ev("coordinator", "terminal_response", 2010)]
    _, segs = tl.request_segments(evs)
    got = {lab: round(b - a) for lab, a, b in segs}
    assert got["thinker 排队"] == 1 and got["thinker 读入提示（prefill）"] == 27
    assert got["thinker 逐个生成文字 token（decode）"] == 1270 and got["talker 生成语音 token"] == 544
    assert got["code2wav 合成波形"] == 148 and got["总计（到最后一个响应）"] == 2010


def test_interval_helpers():
    u = kt.union(np.array([[5, 7], [0, 2], [1, 3], [10, 12]], dtype=float))
    assert u.tolist() == [[0, 3], [5, 7], [10, 12]]
    assert kt.total(kt.clip(u, 1, 11)) == 2 + 2 + 1
    assert kt.intersect(u, kt.union(np.array([[2, 6], [11, 20]], dtype=float))) == 1 + 1 + 1
    assert kt.complement(u, -1, 13).tolist() == [[-1, 0], [3, 5], [7, 10], [12, 13]]


def test_stage_names_of_trace_files():
    assert kt.stage_of("preprocessing@r1_trace_pid1_rank0.trace.json.gz") == ("thinker", "r1")
    assert kt.stage_of("decode_trace_pid7_rank0.trace.json.gz") == ("thinker", "")
    assert kt.stage_of("talker@r3_trace_pid2_rank0.trace.json.gz") == ("talker", "r3")
    assert kt.stage_of("code2wav_trace_pid3_rank0.trace.json.gz") == ("code2wav", "")
    assert kt.role_label({"thinker", "talker", "code2wav"}) == "全部\n三段"
    assert kt.role_label({"talker", "code2wav"}) == "talker\n+c2w"


def test_exp02_rows_pick_the_phase_family(tmp_path):
    ex = load("exp02_placement")
    (tmp_path / "analysis").mkdir()
    table = [{"phase": "smoke_w3b", "level": "1"}, {"phase": "w3b_open_abs", "level": "16.0"}, {"phase": "w3b_closed", "level": "64"}]
    (tmp_path / "analysis" / "analysis.json").write_text(json.dumps({"table": table}))
    assert [r["level"] for r in ex.rows(tmp_path, "w3b_open")] == ["16.0"]
    assert ex.rows(tmp_path / "missing", "w3b_open") == []


def test_exp05_code2wav_kernel_kinds():
    ex = load("exp05_talker")
    assert ex.code2wav_kind("fmha_cutlassF_f16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel") == "注意力（flow）"
    assert ex.code2wav_kind("sm86_xmma_fprop_implicit_gemm_tf32f32") == "卷积（HiFT 为主）"
    assert ex.code2wav_kind("triton_poi_fused__weight_norm_interface_add_convolution_mul_pow_reciprocal_sin_u") == "HiFT：Snake 激活"
    assert ex.code2wav_kind("triton_per_fused__to_copy_add_constant_pad_nd_convolution_mish_mul") == "融合小 kernel（flow）"
    assert ex.code2wav_kind("ampere_fp16_s16816gemm_fp16_128x64_ldg8_relu_f2f_tn") == "矩阵乘（flow 为主）"
