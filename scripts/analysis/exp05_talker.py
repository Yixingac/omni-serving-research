#!/usr/bin/env python3
"""实验 5（GPU 为什么会等待）第一轮：高负载时 2+6 分工的 talker 卡和 thinker 卡被什么卡住。

    python scripts/analysis/exp05_talker.py --data-root ~/research-staging/unicorn \
        --run 20261117_exp05_talker_s2x6_run01 --exp01 research/01_性能瓶颈/实验01_8张GPU在做什么/results/kernel_summary.json \
        --out research/01_性能瓶颈/实验05_GPU为什么会等待

Inputs: the run's raw/cpu_samples.jsonl (per-process CPU, pid -> %), profiling/torch/w3b_prof_c128 (a 3 s
torch profiler window inside a steady concurrency-128 phase), server events, client records. A process's
stage comes from the pid in its trace file name. Writes figures/*.png and results/summary.json.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics as st
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp01_kernel_timeline as kt  # noqa: E402
from exp01_gpu_timeline import setup_matplotlib  # noqa: E402
from omni_research.analysis.experiment_analysis import read_jsonl  # noqa: E402
from omni_research.analysis.metrics import request_metrics  # noqa: E402

PHASES = ["smoke_w3b", "w3b_warm_c128", "w3b_base_c128", "w3b_prof_c128", "w3b_base2_c128"]
PHASE_ZH = {"smoke_w3b": "冒烟（并发 1）", "w3b_warm_c128": "预热（并发 128 第一波）", "w3b_base_c128": "并发 128（不开 profiler）",
            "w3b_prof_c128": "并发 128（中间 3 秒开 profiler）", "w3b_base2_c128": "并发 128（再跑一次）"}


def code2wav_kind(name: str) -> str:
    n = name.lower()
    if "fft" in n:
        return "HiFT：逆傅里叶变换"
    if "sin_" in n or "snake" in n:
        return "HiFT：Snake 激活"
    if "fprop" in n or "cudnn" in n or "nchwtonhwc" in n or "nhwc" in n:
        return "卷积（HiFT 为主）"
    if "fmha" in n or "attention" in n:
        return "注意力（flow）"
    if "mish" in n or "gelu" in n or "layer_norm" in n:
        return "融合小 kernel（flow）"
    if "gemm" in n or "cutlass" in n or "cublas" in n:
        return "矩阵乘（flow 为主）"
    return "其他"


def code2wav_breakdown(phase_dir: Path) -> dict[str, Any]:
    by: dict[str, float] = {}
    graph = total = 0.0
    for f in sorted(phase_dir.rglob("code2wav*.trace.json*")):
        with gzip.open(f, "rt") as fh:
            tr = json.load(fh)
        for e in tr.get("traceEvents", []):
            if e.get("cat") != "kernel":
                continue
            d = float(e.get("dur", 0))
            total += d
            by[code2wav_kind(e["name"])] = by.get(code2wav_kind(e["name"]), 0.0) + d
            if (e.get("args") or {}).get("graph id", 0):
                graph += d
    return {"gpu_s": total / 1e6, "share": {k: v / total for k, v in sorted(by.items(), key=lambda x: -x[1])} if total else {},
            "graph_share": graph / total if total else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--exp01", required=True, help="实验 1 results/kernel_summary.json (concurrency 8 and 64 for comparison)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    exp = Path(a.data_root).expanduser() / "experiments" / a.run
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib()
    summary: dict[str, Any] = {"generated_by": "scripts/analysis/exp05_talker.py", "run": a.run}

    # Latency per phase (order check: base and base2 around the profiled phase).
    lat = {}
    for ph in PHASES[1:]:
        recs = [r for r in read_jsonl(exp / "raw" / f"requests_{ph}.jsonl") if not r.get("warmup") and r.get("status") == "ok"]
        ms = [request_metrics(r) for r in recs]
        e2e = sorted(m["e2e_s"] for m in ms if m.get("e2e_s") is not None)
        span = (max(r["done_ns"] for r in recs) - min(r["sent_ns"] for r in recs)) / 1e9
        lat[ph] = {"n": len(recs), "e2e_p50": e2e[len(e2e) // 2], "e2e_p90": e2e[int(len(e2e) * 0.9)], "tp": len(recs) / span}
    summary["latency"] = lat

    # CPU per process, by stage and phase (pid -> stage from trace file names).
    pid_stage = {}
    for f in (exp / "profiling" / "torch").rglob("*.trace.json*"):
        m = re.match(r"(.+?)_trace_pid(\d+)_", f.name)
        if m:
            pid_stage[m.group(2)] = kt.stage_of(f.name)[0]
    res = json.load(open(exp / "result.json"))
    spans = {p["phase"]: (p["started"], p["ended"]) for p in res["phases"]}
    cpu: dict[str, dict[str, list[float]]] = {}
    for s in read_jsonl(exp / "raw" / "cpu_samples.jsonl"):
        t = s["t_wall_ns"] / 1e9
        ph = next((p for p, (x, y) in spans.items() if x <= t <= y), None)
        if ph is None or "s0" not in s["groups"]:
            continue
        for pid, c in s["groups"]["s0"].get("by_pid", {}).items():
            stage = pid_stage.get(pid)
            if stage in ("thinker", "talker", "code2wav"):
                cpu.setdefault(ph, {}).setdefault(stage, []).append(c)
    summary["cpu_percent_per_process"] = {ph: {stage: st.mean(v) for stage, v in d.items()} for ph, d in cpu.items()}

    # Kernel-level split in the profiler window, the thinker's steps, code2wav's GPU time.
    win, procs = kt.analyze_phase(exp, "w3b_prof_c128")
    steps = {}
    for p in procs:
        if p["stage"] == "thinker":
            s = kt.thinker_steps(exp / "profiling" / "torch" / "w3b_prof_c128" / p["server"] / p["file"])
            if s:
                steps[p["replica"]] = s
    win["thinker_steps"] = steps
    summary["window"] = win
    summary["code2wav"] = code2wav_breakdown(exp / "profiling" / "torch" / "w3b_prof_c128")
    e1 = json.load(open(a.exp01))
    e1c = e1["runs"]["C"]["phases"]
    summary["thinker_step_by_load"] = {
        c: {k: float(st.median(v[k] for v in ph["thinker_steps"].values())) for k in ("busy_ms_p50", "idle_ms_p50")}
        for c, ph in (("8", e1c["w3b_prof_c8"]), ("64", e1c["w3b_prof_c64"]), ("128", win))}

    # Figure 1: CPU per process by phase.
    fig, ax = plt.subplots(figsize=(10, 3.8))
    stages = ["thinker", "talker", "code2wav"]
    colors = {"thinker": "#d94801", "talker": "#41ab5d", "code2wav": "#6a51a3"}
    phs = [p for p in PHASES if p in summary["cpu_percent_per_process"]]
    for i, stage in enumerate(stages):
        vals = [summary["cpu_percent_per_process"][p].get(stage, 0) for p in phs]
        ax.bar([j + (i - 1) * 0.27 for j in range(len(phs))], vals, 0.27, color=colors[stage], label=f"{stage} 进程")
    ax.axhline(100, color="gray", ls="--", lw=1)
    ax.text(len(phs) - 0.6, 102, "100% = 用满一个 CPU 核", fontsize=8, color="gray", ha="right")
    ax.set_xticks(range(len(phs)))
    ax.set_xticklabels([PHASE_ZH[p] for p in phs], fontsize=8)
    ax.set_ylabel("每个进程的 CPU 占用 %（同阶段的进程取平均）", fontsize=8)
    ax.set_ylim(0, 125)
    ax.legend(fontsize=8)
    ax.set_title("图 1　负载升高后，每个阶段进程都逼近用满一个 CPU 核", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "fig1_cpu_per_process.png")
    plt.close(fig)

    # Figure 2: where each GPU's time goes in the window.
    roles = {g: kt.role_label({p["stage"] for p in procs if g in p["gpus"]}) for g in range(8)}
    kt.fig_gpu_breakdown(plt, {"方案 C：2+6，并发 128（稳定段中的 3 秒）": win}, {"方案 C：2+6，并发 128（稳定段中的 3 秒）": roles},
                         out / "figures" / "fig2_gpu_time_c128.png", "图 2　并发 128：每张卡的时间去哪了（kernel 级）")

    # Figure 3: code2wav GPU time by component.
    share = summary["code2wav"]["share"]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.barh(list(share)[::-1], [v * 100 for v in list(share.values())[::-1]], color="#6a51a3")
    for i, v in enumerate(list(share.values())[::-1]):
        ax.text(v * 100 + 0.5, i, f"{v * 100:.0f}%", va="center", fontsize=8)
    ax.set_xlabel("占 code2wav GPU 时间的百分比（按 kernel 名字归类，推算）", fontsize=8)
    ax.set_title(f"图 3　code2wav 的 GPU 时间花在哪（并发 128；只有 {summary['code2wav']['graph_share'] * 100:.0f}% 来自 CUDA graph）", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "figures" / "fig3_code2wav_kernels.png")
    plt.close(fig)

    # Figure 4: the thinker's step against load.
    sl = summary["thinker_step_by_load"]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    keys = list(sl)
    ax.barh(range(len(keys)), [sl[k]["busy_ms_p50"] for k in keys], color="#d94801", label="GPU 在算")
    ax.barh(range(len(keys)), [sl[k]["idle_ms_p50"] for k in keys], left=[sl[k]["busy_ms_p50"] for k in keys], color="#fdd0a2", label="GPU 空着（等 CPU）")
    for i, k in enumerate(keys):
        ax.text(sl[k]["busy_ms_p50"] + sl[k]["idle_ms_p50"] + 0.3, i, f"{sl[k]['busy_ms_p50']:.1f} + {sl[k]['idle_ms_p50']:.1f} 毫秒", va="center", fontsize=8)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"并发 {k}" for k in keys], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 50)
    ax.set_xlabel("方案 C 的 thinker 每生成一个 token 的时间（毫秒，中位数）", fontsize=8)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("图 4　负载越高，thinker 每一步等 CPU 的时间越长", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "figures" / "fig4_thinker_step_by_load.png")
    plt.close(fig)

    (out / "results" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
