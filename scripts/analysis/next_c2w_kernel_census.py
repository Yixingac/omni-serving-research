#!/usr/bin/env python3
"""下一阶段研究（Compiler/Runtime 候选 B）：code2wav 的 kernel 普查——GPU 时间具体花在哪些 kernel 上。

Reads torch-profiler Chrome traces of code2wav processes (one file per process and profiler window),
sums kernel durations per kernel name, and reports each name's share of the code2wav kernel time,
its call count, mean duration, the share replayed from CUDA graphs, and its launch configuration.

Each name is also put into an op family (attention, GEMM, cuDNN convolution by precision, layout
conversion, FFT, Inductor-fused Triton, ...) and a likely sub-model (flow or HiFT). Both labels come
from the kernel name alone (precision, op and fusion tags in the name); they are a heuristic, not an
attribution through the launching CPU code. Use the measured CUDA-event split of 实验 6 (flow about
150 ms and HiFT about 45 ms GPU per request) as the cross-check for the sub-model shares.

Kernel durations come from the GPU timestamps of CUPTI, so they hold under the profiler, but the
profiler slows the CPU side down: launch gaps and throughput inside these windows are not used here.

Usage:
    python scripts/analysis/next_c2w_kernel_census.py --data-root <PROJECT_DATA_ROOT> \
        --run 20261117_exp07_t2_k2_c4_block1_run01 --phase w3b_c128_C [--run ...] \
        --label exclusive_c128 --out research/next_steps/results/c2w_kernel_census_exclusive_c128.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

FAMILY_ORDER = [
    "attention", "attention_prep", "softmax", "gemm", "conv_f16", "conv_f32_tf32", "layout_conversion",
    "fft", "snake_weight_norm", "inductor_fused", "elementwise", "reduction_norm", "copy_cast", "other",
]

# Families whose kernels only one of the two sub-models issues, judged from the names and the
# precisions seen in the traces: the flow estimator is a compiled FP16 transformer with causal FP16
# convolutions (fused SDPA, GELU, Mish, LayerNorm), HiFT runs FP32 convolutions (TF32 tensor cores) with
# Snake activations, LeakyReLU, reflection padding and an iSTFT. Other Inductor kernels are told apart by
# the op tags in their names; the rest stay "unknown".
FLOW_FAMILIES = {"attention", "attention_prep", "softmax", "gemm", "conv_f16"}
HIFT_FAMILIES = {"conv_f32_tf32", "layout_conversion", "fft", "snake_weight_norm"}
FLOW_TAGS = ("mish", "gelu", "native_layer_norm", "scaled_dot_product", "silu")
HIFT_TAGS = ("leaky_relu", "reflection_pad", "tanh", "stft", "exp_", "_sin", "sin_")


def kernel_family(name: str) -> str:
    """Op family of one kernel, from its name."""
    n = name.lower()
    if "fmha" in n or "flash" in n or "attentionkernel" in n:
        return "attention"
    if n.startswith("triton_") and "scaled_dot_product" in n:
        return "attention_prep"
    if "softmax" in n:
        return "softmax"
    if n.startswith("triton_") and ("sin_" in n or "_sin" in n or "weight_norm" in n):
        return "snake_weight_norm"
    if n.startswith("triton_"):
        return "inductor_fused"
    if "nchwtonhwc" in n or "nhwctonchw" in n:
        return "layout_conversion"
    if "fft" in n:
        return "fft"
    if "fprop" in n or "conv" in n or "scudnn" in n or "implicit_gemm" in n:
        if "tf32" in n or "<float" in n or "float, float" in n or "scudnn" in n or "s1688fprop" in n:
            return "conv_f32_tf32"
        return "conv_f16"
    if "gemm" in n or "gemv" in n or "cublas" in n or "cutlass" in n:
        return "gemm"
    if "copy" in n or "catarray" in n:  # not "cast": every gpu_kernel_impl_nocast name contains it
        return "copy_cast"
    if "reduce" in n or "norm" in n:
        return "reduction_norm"
    if "elementwise" in n or "masked_fill" in n or "fill" in n:
        return "elementwise"
    return "other"


def likely_submodel(family: str, name: str = "") -> str:
    if family in FLOW_FAMILIES:
        return "flow"
    if family in HIFT_FAMILIES:
        return "hift"
    if family == "inductor_fused":
        n = name.lower()
        flow = any(t in n for t in FLOW_TAGS)
        hift = any(t in n for t in HIFT_TAGS)
        if flow != hift:
            return "flow" if flow else "hift"
    return "unknown"


def precision_tag(name: str) -> str:
    n = name.lower()
    if "tf32" in n:
        return "tf32"
    if "bf16" in n or "bfloat16" in n:
        return "bf16"
    if "f16" in n or "fp16" in n or "half" in n:
        return "fp16"
    if "<float" in n or "float, float" in n or "scudnn" in n:
        return "fp32"
    return "-"


def _mode(counter: Counter) -> Any:
    return counter.most_common(1)[0][0] if counter else None


def census(traces: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per kernel name: count, total and graph-replayed microseconds, and the most common launch shape."""
    agg: dict[str, dict[str, Any]] = {}
    for trace in traces:
        for ev in trace.get("traceEvents", []):
            if ev.get("ph") != "X" or ev.get("cat") != "kernel":
                continue
            name = str(ev.get("name", "?"))
            dur = float(ev.get("dur", 0.0))
            args = ev.get("args") or {}
            row = agg.setdefault(name, {"count": 0, "total_us": 0.0, "graph_us": 0.0, "durations": [],
                                        "grid": Counter(), "block": Counter(), "regs": Counter(),
                                        "smem": Counter(), "occupancy": Counter()})
            row["count"] += 1
            row["total_us"] += dur
            row["durations"].append(dur)
            if args.get("graph id", 0):
                row["graph_us"] += dur
            if "grid" in args:
                row["grid"][tuple(args["grid"])] += 1
            if "block" in args:
                row["block"][tuple(args["block"])] += 1
            if "registers per thread" in args:
                row["regs"][args["registers per thread"]] += 1
            if "shared memory" in args:
                row["smem"][args["shared memory"]] += 1
            if "est. achieved occupancy %" in args:
                row["occupancy"][args["est. achieved occupancy %"]] += 1
    return agg


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def summarize(agg: dict[str, dict[str, Any]], top: int = 25) -> dict[str, Any]:
    total = sum(r["total_us"] for r in agg.values())
    graph = sum(r["graph_us"] for r in agg.values())
    fam: dict[str, dict[str, float]] = {f: {"total_ms": 0.0, "count": 0} for f in FAMILY_ORDER}
    sub: dict[str, float] = {"flow": 0.0, "hift": 0.0, "unknown": 0.0}
    rows = []
    for name, r in agg.items():
        f = kernel_family(name)
        s = likely_submodel(f, name)
        fam[f]["total_ms"] += r["total_us"] / 1e3
        fam[f]["count"] += r["count"]
        sub[s] += r["total_us"]
        rows.append({
            "name": name, "family": f, "likely_submodel": s, "precision": precision_tag(name),
            "count": r["count"], "total_ms": r["total_us"] / 1e3,
            "share": r["total_us"] / total if total else None,
            "mean_us": r["total_us"] / r["count"], "median_us": _median(r["durations"]),
            "graph_share": r["graph_us"] / r["total_us"] if r["total_us"] else None,
            "grid": list(_mode(r["grid"]) or []), "block": list(_mode(r["block"]) or []),
            "registers_per_thread": _mode(r["regs"]), "shared_memory_bytes": _mode(r["smem"]),
            "est_occupancy_pct": _mode(r["occupancy"]),
        })
    rows.sort(key=lambda x: -x["total_ms"])
    cum = 0.0
    for x in rows:
        cum += x["share"] or 0.0
        x["cumulative_share"] = cum
    return {
        "kernel_total_ms": total / 1e3,
        "kernel_count": sum(r["count"] for r in agg.values()),
        "distinct_kernels": len(agg),
        "graph_share": graph / total if total else None,
        "families": {f: {"total_ms": v["total_ms"], "share": v["total_ms"] * 1e3 / total if total else None,
                         "count": v["count"]} for f, v in fam.items() if v["count"]},
        "likely_submodel_share": {k: v / total if total else None for k, v in sub.items()},
        "top_kernels": rows[:top],
    }


def load_trace(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return json.load(fh)


def code2wav_traces(exp_dir: Path, phase: str) -> list[Path]:
    return sorted((exp_dir / "profiling" / "torch" / phase).rglob("code2wav*.trace.json*"))


def process_label(path: Path) -> str:
    m = re.match(r"(code2wav@r\d+)", path.name)
    return m.group(1) if m else path.name.split("_trace")[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True, help="PROJECT_DATA_ROOT (contains experiments/<run>)")
    ap.add_argument("--run", action="append", required=True, help="run directory name; repeat for several")
    ap.add_argument("--phase", required=True, help="profiler phase directory, e.g. w3b_c128_C")
    ap.add_argument("--label", required=True)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    sources, per_run = [], {}
    all_agg: dict[str, dict[str, Any]] = {}
    for run in a.run:
        files = code2wav_traces(root / run, a.phase)
        traces = [load_trace(p) for p in files]
        agg = census(traces)
        per_run[run] = summarize(agg, top=10)
        sources.append({"run": run, "phase": a.phase, "processes": [process_label(p) for p in files]})
        for name, r in agg.items():
            dst = all_agg.setdefault(name, {"count": 0, "total_us": 0.0, "graph_us": 0.0, "durations": [],
                                            "grid": Counter(), "block": Counter(), "regs": Counter(),
                                            "smem": Counter(), "occupancy": Counter()})
            for k in ("count", "total_us", "graph_us"):
                dst[k] += r[k]
            dst["durations"].extend(r["durations"])
            for k in ("grid", "block", "regs", "smem", "occupancy"):
                dst[k].update(r[k])
    out = {"generated_by": "scripts/analysis/next_c2w_kernel_census.py", "label": a.label, "sources": sources,
           "note": "kernel durations from CUPTI GPU timestamps inside short torch-profiler windows; family and "
                   "sub-model labels are name heuristics, not CPU-side attribution",
           "pooled": summarize(all_agg, top=a.top), "per_run": per_run}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    pooled = out["pooled"]
    print(f"{a.label}: {pooled['kernel_total_ms']:.0f} ms kernel time, graph share {pooled['graph_share']:.3f}")
    for f, v in sorted(pooled["families"].items(), key=lambda kv: -kv[1]["total_ms"]):
        print(f"  {f:20s} {100 * v['share']:5.1f}%")
    print("  sub-model (heuristic):", {k: round(v, 3) for k, v in pooled["likely_submodel_share"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
