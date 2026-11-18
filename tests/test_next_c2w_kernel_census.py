"""Next-step research, compiler candidate B: code2wav kernel census (synthetic traces only).

The kernel names below are copied from real code2wav traces (A6000, sglang-omni 520b49ee) so that the
name heuristics are tested on the strings they will actually see.
"""

import gzip
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("next_c2w_kernel_census", HERE / "next_c2w_kernel_census.py")
kc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kc)

FMHA = ("fmha_cutlassF_f16_aligned_64x64_rf_sm80(PyTorchMemEffAttention::AttentionKernel<cutlass::half_t, "
        "cutlass::arch::Sm80, true, 64, 64, 64, true, true>::Params)")
SDPA_PREP = ("triton_poi_fused__scaled_dot_product_efficient_attention__to_copy_constant_pad_nd_expand_"
             "native_layer_norm_scalar_tensor_slice_transpose_unsqueeze_view_where_3")
GEMM = "ampere_fp16_s16816gemm_fp16_256x128_ldg8_relu_f2f_stages_32x3_tn"
CUBLAS = "sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize96x128x32_stage4_warpsize2x2x1_tensor16x8x16_execute_kernel__5x_cublas"
CONV_TF32 = ("sm86_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc_nchw_tilesize128x128x16_stage3_"
             "warpsize2x2x1_g1_tensor16x8x8_alignc4_execute_kernel__5x_cudnn")
CONV_F16 = ("sm86_xmma_fprop_implicit_gemm_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize256x128x64_stage2_"
            "warpsize4x2x1_g1_tensor16x8x16_execute_kernel__5x_cudnn")
NCHW2NHWC = ("void cudnn::engines_precompiled::nchwToNhwcKernel<float, float, float, false, true, "
             "(cudnnKernelDataType_t)2>(cudnn::engines_precompiled::nchw2nhwc_params_t<float>, float")
SNAKE = "triton_poi_fused__weight_norm_interface_add_convolution_mul_pow_reciprocal_sin_unsqueeze_31"
GELU = "triton_poi_fused_gelu_view_9"
MISH = "triton_per_fused__to_copy_add_constant_pad_nd_convolution_mish_mul_native_layer_norm_split_squeeze_transpose_unsqueeze_view_8"
LEAKY = "triton_poi_fused_convolution_leaky_relu_12"
PLAIN_TRITON = "triton_poi_fused_add_mul_split_view_10"
SOFTMAX = ("void at::native::(anonymous namespace)::cunn_SoftMaxForward<8, c10::Half, float, float, "
           "at::native::(anonymous namespace)::SoftMaxForwardEpilogue>(float*, c10::Half const")
GROUPED = ("void cudnn::cnn::conv2d_grouped_direct_kernel<false, true, false, false, false, false, 0, 0, int, "
           "float, float, float, float, float, float>(cudnn::cnn::GroupedDirectFprop")
SCUDNN = "_5x_cudnn_ampere_scudnn_128x32_relu_small_nn_v1"
CAST = ("void at::native::vectorized_elementwise_kernel<4, at::native::float16_copy_kernel_cuda(at::TensorIteratorBase&)"
        "::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::nati")
FILL = "void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel"
FFT = "void regular_fft_factor<256u, EPT<8u>, 32u, 8u, (padding_t)1, (twiddle_t)0>(float2*, float2*)"


def kernel(name, dur, graph=0, grid=(1, 1, 1), block=(128, 1, 1)):
    return {"ph": "X", "cat": "kernel", "name": name, "ts": 0.0, "dur": dur,
            "args": {"graph id": graph, "grid": list(grid), "block": list(block), "registers per thread": 64,
                     "shared memory": 1024, "est. achieved occupancy %": 50}}


def test_families_follow_the_names_seen_in_real_traces():
    expect = {FMHA: "attention", SDPA_PREP: "attention_prep", GEMM: "gemm", CUBLAS: "gemm",
              CONV_TF32: "conv_f32_tf32", CONV_F16: "conv_f16", NCHW2NHWC: "layout_conversion",
              SNAKE: "snake_weight_norm", GELU: "inductor_fused", MISH: "inductor_fused",
              SOFTMAX: "softmax", GROUPED: "conv_f32_tf32", SCUDNN: "conv_f32_tf32", CAST: "copy_cast",
              FILL: "elementwise", FFT: "fft"}
    for name, fam in expect.items():
        assert kc.kernel_family(name) == fam, name


def test_sub_model_uses_family_then_inductor_op_tags():
    assert kc.likely_submodel("attention", FMHA) == "flow"
    assert kc.likely_submodel("conv_f32_tf32", CONV_TF32) == "hift"
    assert kc.likely_submodel("inductor_fused", MISH) == "flow"
    assert kc.likely_submodel("inductor_fused", LEAKY) == "hift"
    assert kc.likely_submodel("inductor_fused", PLAIN_TRITON) == "unknown"
    assert kc.likely_submodel("copy_cast", CAST) == "unknown"


def test_precision_tags():
    assert kc.precision_tag(CONV_TF32) == "tf32"
    assert kc.precision_tag(FMHA) == "fp16"
    assert kc.precision_tag(NCHW2NHWC) == "fp32"
    assert kc.precision_tag(PLAIN_TRITON) == "-"


def test_census_shares_graph_and_order():
    trace_a = {"traceEvents": [kernel(FMHA, 600.0), kernel(FMHA, 600.0, graph=5), kernel(CONV_TF32, 200.0),
                               {"ph": "X", "cat": "cpu_op", "name": "aten::conv1d", "ts": 0, "dur": 999.0},
                               {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy DtoH", "ts": 0, "dur": 999.0}]}
    trace_b = {"traceEvents": [kernel(NCHW2NHWC, 100.0), kernel(PLAIN_TRITON, 500.0)]}
    s = kc.summarize(kc.census([trace_a, trace_b]), top=3)
    assert s["kernel_total_ms"] == 2.0          # memcpy and CPU ops are not kernels
    assert s["kernel_count"] == 5
    assert abs(s["graph_share"] - 0.3) < 1e-9   # 600 of 2000 us replayed from a graph
    names = [r["name"] for r in s["top_kernels"]]
    assert names == [FMHA, PLAIN_TRITON, CONV_TF32]  # top 3 by total time
    fmha = s["top_kernels"][0]
    assert fmha["count"] == 2 and fmha["mean_us"] == 600.0 and abs(fmha["graph_share"] - 0.5) < 1e-9
    assert abs(s["top_kernels"][-1]["cumulative_share"] - 0.95) < 1e-9  # 0.6 + 0.25 + 0.1
    assert abs(sum(v["share"] for v in s["families"].values()) - 1.0) < 1e-9
    sub = s["likely_submodel_share"]
    assert abs(sub["flow"] - 0.6) < 1e-9 and abs(sub["hift"] - 0.15) < 1e-9 and abs(sub["unknown"] - 0.25) < 1e-9


def test_main_writes_summary_without_paths(tmp_path, monkeypatch):
    run = "20990101_exp07_t2_k2_c4_block1_run01"
    d = tmp_path / "experiments" / run / "profiling" / "torch" / "w3b_c128_C" / "s0"
    d.mkdir(parents=True)
    for proc, name in (("code2wav@r0", FMHA), ("code2wav@r1", CONV_TF32)):
        with gzip.open(d / f"{proc}_trace_pid1_rank0.trace.json.gz", "wt") as fh:
            json.dump({"traceEvents": [kernel(name, 100.0)]}, fh)
    with gzip.open(d / "talker@r0_trace_pid2_rank0.trace.json.gz", "wt") as fh:  # not a code2wav process
        json.dump({"traceEvents": [kernel(GEMM, 1e6)]}, fh)
    out = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", ["x", "--data-root", str(tmp_path), "--run", run, "--phase", "w3b_c128_C",
                                      "--label", "t", "--out", str(out)])
    assert kc.main() == 0
    res = json.loads(out.read_text())
    assert res["pooled"]["kernel_total_ms"] == 0.2
    assert res["sources"] == [{"run": run, "phase": "w3b_c128_C", "processes": ["code2wav@r0", "code2wav@r1"]}]
    assert str(tmp_path) not in out.read_text()
