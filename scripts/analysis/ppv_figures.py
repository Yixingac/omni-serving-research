#!/usr/bin/env python3
"""Figures for the pre-presentation validation (16:9 slides): drawn only from committed summary JSON files.

    python scripts/analysis/ppv_figures.py --out research/next_steps/presentation_validation/figures [--only placement ...]

Every figure states its conditions (model, runtime, GPUs, workload, repetitions) in the caption box, shows blocks as
small points and t-based 95 % intervals over blocks as error bars, and draws nothing where a measurement is missing
(no zeros for failed or unrun conditions). A figure whose inputs do not exist is skipped and reported.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

PV = REPO / "research/next_steps/presentation_validation"
EXP09 = REPO / "research/01_性能瓶颈/实验09_最佳GPU分工/results/summary.json"
COMMON = "MiniCPM-o 4.5 · SGLang-Omni 520b49ee · 8× RTX A6000 · synthetic workloads"
PLACEMENT_LABEL = {1: "2+6 shared (talker & code2wav share 6 GPUs)", 2: "2+2+4", 3: "2+1+5", 4: "2+3+3"}
PLACEMENT_COLOR = {1: "#000000", 2: "#E69F00", 3: "#CC79A7", 4: "#009E73"}
PLACEMENT_MARKER = {1: "o", 2: "D", 3: "s", 4: "^"}


def plt_setup():  # noqa: ANN201
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12, "legend.fontsize": 10,
                         "figure.dpi": 150, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
    return plt


def caption(fig, text: str) -> None:  # noqa: ANN001
    fig.text(0.01, -0.02, text, ha="left", va="top", fontsize=9, color="#444444", wrap=True)


def save(fig, out: Path, name: str) -> list[Path]:  # noqa: ANN001
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.png", out / f"{name}.svg"]
    for p in paths:
        fig.savefig(p)
    return paths


def cell(summary: dict[str, Any], arm: int, key: str) -> dict[str, Any] | None:
    return summary.get("by_arm", {}).get(str(arm), {}).get(key)


def errbar(ax, x: float, stat: dict[str, Any] | None, color: str, marker: str, label: str | None = None, scale: float = 1.0):  # noqa: ANN001
    """Mean with the t interval and the blocks as small points; nothing when the mean is missing."""
    if not stat or stat.get("mean") is None:
        return
    mean = stat["mean"] * scale
    ci = stat.get("ci95")
    yerr = [[mean - ci[0] * scale], [ci[1] * scale - mean]] if ci else None
    ax.errorbar([x], [mean], yerr=yerr, fmt=marker, color=color, capsize=4, ms=8, label=label)
    runs = [r * scale for r in stat.get("runs", []) if isinstance(r, (int, float))]
    ax.scatter([x] * len(runs), runs, s=12, color=color, alpha=0.5, zorder=3)


def fig_placement(out: Path) -> list[Path]:
    """Experiment 9 (existing data): W3b capacity per placement and open-loop P95 end-to-end latency vs offered load."""
    s = json.loads(EXP09.read_text())
    plt = plt_setup()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.6]})
    for i, arm in enumerate((1, 2, 3, 4)):
        errbar(a1, i, (cell(s, arm, "w3b|closed_loop|A|256") or {}).get("throughput_rps"), PLACEMENT_COLOR[arm],
               PLACEMENT_MARKER[arm])
    a1.set_xticks(range(4), [PLACEMENT_LABEL[a].split(" (")[0] for a in (1, 2, 3, 4)])
    a1.set_ylabel("completed requests / s")
    a1.set_title("Closed-loop capacity (W3b, 256 in flight)")
    a1.set_ylim(bottom=0)
    for arm in (1, 2, 3, 4):
        pts = []
        for key, c in (s.get("by_arm", {}).get(str(arm), {})).items():
            w, kind, mode, lv = key.split("|")
            if w == "w3b" and kind == "open_loop":
                pts.append((float(lv), c))
        pts.sort()
        xs = [x for x, c in pts if c["e2e_p95_s"]["mean"] is not None]
        ys = [c["e2e_p95_s"]["mean"] for x, c in pts if c["e2e_p95_s"]["mean"] is not None]
        a2.plot(xs, ys, color=PLACEMENT_COLOR[arm], marker=PLACEMENT_MARKER[arm], label=PLACEMENT_LABEL[arm])
        for x, c in pts:
            steady = c.get("steady_runs") or []
            if c["e2e_p95_s"]["mean"] is not None and steady and not all(steady):  # some block not steady: ring
                a2.scatter([x], [c["e2e_p95_s"]["mean"]], s=160, facecolors="none", edgecolors=PLACEMENT_COLOR[arm])
    a2.set_xlabel("offered load (requests / s, open loop)")
    a2.set_ylabel("end-to-end P95 (s)")
    a2.set_title("Open-loop tail latency (W3b)")
    a2.legend(loc="upper left")
    caption(fig, f"Experiment 9 (existing data, 3 blocks per placement; points = blocks, bars = t 95 % CI). Thinker on 2 GPUs in "
                 f"every placement; '2+1+5' = 1 talker GPU + 5 code2wav GPUs. Rings: at least one block not steady "
                 f"(in-flight slope > 5 % of the offered rate). Capacity = completed / (last completion − first send). "
                 f"{COMMON}; /v1/chat/completions.")
    return save(fig, out, "fig01_gpu_placement")


P0_SUMMARY = PV / "results/p0_summary.json"
# Arm G (cap 64, graphs <= 32) could not start at the pin (Erratum E-P0-G), so it has no data and no position here.
P0_ARMS = [("A", "A: cap 32 (default)", "#CC79A7", "s"), ("B", "B: cap 64, graphs ≤64", "#0072B2", "D"),
           ("R", "Ref: 2+2+4, cap 32", "#E69F00", "o")]


def _interval(values: list[float]) -> dict[str, Any]:
    import math

    import numpy as np

    v = [x for x in values if x is not None]
    out: dict[str, Any] = {"runs": v, "mean": float(np.mean(v)) if v else None, "ci95": None, "n": len(v)}
    if len(v) >= 2:
        t = {1: 12.706, 2: 4.303, 3: 3.182}.get(len(v) - 1, 2.0)
        half = t * float(np.std(v, ddof=1)) / math.sqrt(len(v))
        out["ci95"] = [out["mean"] - half, out["mean"] + half]
    return out


def _p0_level(run: dict[str, Any], phase: str) -> dict[str, Any] | None:
    return next((x for x in run["levels"] if x["phase"] == phase), None)


def fig_p0_admission(out: Path) -> list[Path]:
    """P0: talker scheduler state at W3b 256 in flight (running vs queued) and the decode step time per batch size."""
    doc = json.loads(P0_SUMMARY.read_text())
    runs = doc["runs"]
    plt = plt_setup()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for i, (arm, label, color, marker) in enumerate(P0_ARMS):
        rows = [_p0_level(r, "w3b_closed_A_c256") for r in runs if r["arm"] == arm]
        run_p50 = _interval([x["talker"].get("running_p50") for x in rows if x and x["talker"].get("lines")])
        q_p50 = _interval([x["talker"].get("queue_p50") for x in rows if x and x["talker"].get("lines")])
        errbar(a1, i - 0.12, run_p50, color, marker, label=label)
        errbar(a1, i + 0.12, q_p50, color, "x")
    a1.axhline(32, color="#888888", ls=":", lw=1)
    a1.axhline(64, color="#888888", ls=":", lw=1)
    a1.set_xticks(range(len(P0_ARMS)), [x[0] if x[0] != "R" else "Ref" for x in P0_ARMS])
    a1.set_ylabel("requests per talker (median)")
    a1.set_title("Talker at W3b, 256 in flight:\nrunning (filled) vs queued (×)")
    a1.legend(loc="center right", fontsize=9)
    for arm, label, color, marker in P0_ARMS:
        pooled: dict[int, list[float]] = {}
        for r in runs:
            if r["arm"] == arm:
                for p in r.get("step_curve", []):
                    pooled.setdefault(p["batch"], []).append(p["step_ms"])
        xs = sorted(pooled)
        if xs:
            a2.plot(xs, [sum(pooled[x]) / len(pooled[x]) for x in xs], marker=marker, ms=4, color=color, label=label)
    a2.axvline(32, color="#888888", ls=":", lw=1)
    a2.set_xlabel("running requests in the decode batch")
    a2.set_ylabel("decode step time (ms) = batch / tokens per s")
    a2.set_title("Talker decode step time vs batch size")
    a2.legend(loc="upper left", fontsize=9)
    n = {arm: sum(1 for r in runs if r["arm"] == arm) for arm, *_ in P0_ARMS}
    caption(fig, "P0 (this round): 2+1+5 placement (1 talker GPU) with talker max_running_requests 32 (A) or 64 (B, which also "
                 "captures CUDA graphs up to 64 — the pinned version refuses a graph limit below the cap, so arm G could not run); "
                 "reference 2+2+4. Runs per arm: " + ", ".join(f"{k} {v}" for k, v in n.items())
            + ". Left: medians over SGLang decode-log lines in the closed-loop W3b 256 level (talker lines separated by tokens per "
              "running request > 200); points = blocks, bars = t 95 % CI. Right: step time from the median log throughput per batch size "
              f"of each run (≥ 20 lines per size and run), averaged over the arm's runs. {COMMON}.")
    return save(fig, out, "fig02_talker_admission")


def fig_p0_capacity(out: Path) -> list[Path]:
    """P0: registered capacity per arm (W3b, W3L) and open-loop goodput vs offered load (W3b)."""
    doc = json.loads(P0_SUMMARY.read_text())
    runs = doc["runs"]
    plt = plt_setup()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.5]})
    for i, (arm, label, color, marker) in enumerate(P0_ARMS):
        for j, w in enumerate(("w3b", "w3l")):
            st = _interval([r["capacity"].get(w) for r in runs if r["arm"] == arm and r["capacity"].get(w) is not None])
            if st["n"]:
                errbar(a1, j * 4 + i, st, color, marker, label=label if j == 0 else None)
    a1.set_xticks([1, 5], ["W3b (256 in flight)", "W3L (128 in flight)"])
    a1.set_ylabel("completed requests / s")
    a1.set_title("Closed-loop capacity")
    a1.set_ylim(bottom=0)
    a1.legend(loc="lower left", fontsize=9)
    for arm, label, color, marker in P0_ARMS:
        by_rate: dict[float, list[float]] = {}
        steady: dict[float, list[bool]] = {}
        for r in runs:
            if r["arm"] != arm:
                continue
            for x in r["levels"]:
                if x["workload"] == "w3b" and x["kind"] == "open_loop" and x["goodput"].get("goodput_rps") is not None:
                    by_rate.setdefault(x["rate_rps"], []).append(x["goodput"]["goodput_rps"])
                    steady.setdefault(x["rate_rps"], []).append(bool((x.get("growth") or {}).get("steady")))
        xs = sorted(by_rate)
        if xs:
            ys = [sum(by_rate[x]) / len(by_rate[x]) for x in xs]
            a2.plot(xs, ys, marker=marker, color=color, label=label)
            for x, y in zip(xs, ys):
                if not all(steady[x]):
                    a2.scatter([x], [y], s=160, facecolors="none", edgecolors=color)
    a2.set_xlabel("offered load (requests / s, open loop, W3b)")
    a2.set_ylabel("goodput (requests / s meeting the SLO)")
    a2.set_title("Open-loop goodput: TTFT ≤ 1 s and e2e / audio s ≤ 0.5")
    a2.legend(loc="upper left", fontsize=9)
    caption(fig, "P0 (this round), 3 blocks per arm (points = blocks, bars = t 95 % CI); the reference ran W3b only (registered addendum). "
                 "Capacity = completed / (last completion − first send) at the top closed-loop level without request events (registered, "
                 "as in Experiment 9). Right: mean "
                 "over blocks; rings = at least one block not steady (in-flight slope > 5 % of the offered rate), so that "
                 f"rate's throughput is not a stable capacity. {COMMON}; /v1/chat/completions.")
    return save(fig, out, "fig03_talker_cap_capacity")


SHAPES = REPO / "research/next_steps/results/c1_real_shapes.json"
CENSUS = REPO / "research/next_steps/results/c2w_kernel_census_exclusive_c128.json"
SETTING_ORDER = ["fifo1", "fifo2", "fifo4", "fifo8", "fifo16", "length8", "length16"]


def fig_c2w_shapes(out: Path) -> list[Path]:
    """Existing data (实验 10, t2_k4_c2, code2wav saturated): padding and CUDA-graph eligibility per batching setting."""
    d = json.loads(SHAPES.read_text())["groups"]
    plt = plt_setup()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=False)
    for ax, wl, title in ((axes[0], "w3b", "W3b (short answers)"), (axes[1], "mix", "MIX (half short, half long)")):
        rows = [(s, d.get(f"exp10_t2_k4_c2_graph|{wl}|{s}")) for s in SETTING_ORDER]
        rows = [(s, g) for s, g in rows if g]
        xs = range(len(rows))
        pad = [g["padding_ratio_frame_weighted"] * 100 for _, g in rows]
        graph = [(g["request_share_by_path"]["dense_default"].get("dense_graph", 0.0)) * 100 for _, g in rows]
        b1 = ax.bar([x - 0.2 for x in xs], pad, width=0.4, color="#D55E00", label="padded frames (%)")
        b2 = ax.bar([x + 0.2 for x in xs], graph, width=0.4, color="#0072B2", label="requests on a flow CUDA graph (%)")
        for x, (_, g) in zip(xs, rows):
            ax.text(x, max(pad[x], graph[x]) + 1.5, f"b̄={g['batch_size']['mean']:.1f}\nn={g['batches']}", ha="center", fontsize=8)
        ax.set_xticks(list(xs), [s for s, _ in rows], rotation=20)
        ax.set_title(title)
        ax.set_ylim(0, 70)
        ax.set_ylabel("percent")
    axes[0].legend(loc="upper right", fontsize=9)
    caption(fig, "Existing data (Experiment 10, placement 2+4+2 with code2wav saturated, CUDA graphs on; batches pooled over 3 blocks). "
                 "fifoN = first-in-first-out up to N requests; lengthN = length-aware up to N. b̄ = mean requests per batch, n = batches. "
                 "Graph share = requests whose batch matched a captured flow shape (B=1: 44 buckets up to 1024 frames; B 2–10 narrow; "
                 f"none above 10), reproduced from the recorded path for 100 % of batches. {COMMON}.")
    return save(fig, out, "fig04_code2wav_batch_shapes")


def fig_c2w_kernels(out: Path) -> list[Path]:
    """Existing data (实验 7 traces, code2wav on its own GPUs at 128 in flight): GPU kernel time by kernel family."""
    d = json.loads(CENSUS.read_text())["pooled"]
    fam = sorted(d["families"].items(), key=lambda kv: -kv[1]["share"])
    plt = plt_setup()
    fig, ax = plt.subplots(figsize=(11, 4.6))
    names = [k for k, _ in fam][:12]
    shares = [d["families"][k]["share"] * 100 for k in names]
    colors = ["#0072B2" if k in ("attention", "attention_prep", "softmax", "gemm", "conv_f16") else
              "#D55E00" if k in ("conv_f32_tf32", "layout_conversion", "fft", "snake_weight_norm") else "#999999" for k in names]
    ax.barh(range(len(names))[::-1], shares, color=colors)
    ax.set_yticks(range(len(names))[::-1], names)
    for i, v in enumerate(shares):
        ax.text(v + 0.3, len(names) - 1 - i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("share of code2wav GPU kernel time (%)")
    ax.set_title("Where code2wav spends GPU time (blue: flow DiT families, orange: HiFT families, grey: shared/other)")
    caption(fig, f"Existing traces (Experiment 7, 2+2+4 and 2+4+2 placements, W3b at 128 in flight, 3 blocks each): {d['kernel_count']:,} "
                 f"kernels, {d['kernel_total_ms'] / 1000:.1f} s of kernel time inside short torch-profiler windows (CUPTI timestamps). "
                 "Families and flow/HiFT attribution are name heuristics. CUDA-graph kernels are "
                 f"{d['graph_share'] * 100:.1f} % of kernel time — a time share, not a graph hit rate. {COMMON}.")
    return save(fig, out, "fig05_code2wav_kernels")


R1_SUMMARY = PV / "results/r1_summary.json"


def fig_duplex_ttfa(out: Path) -> list[Path]:
    """R1: where the full-duplex TTFA goes (per answered turn) and which stage dominates a speaking unit."""
    import statistics

    d = json.loads(R1_SUMMARY.read_text())
    turns = [t for t in d["ttfa_turns"] if t["phase"].endswith("_A") and not t.get("early_start")]
    plt = plt_setup()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    parts = [("a_wait_unit_end_ms", "(a) wait for the 1-s unit to fill", "#999999"),
             ("b_listen_ms", "(b) extra listening units", "#E69F00"),
             ("c_service_ms", "(c) speaking unit's processing", "#0072B2")]
    groups = [(f"k = {k}\n(n = {sum(1 for t in turns if t['k'] == k)})", [t for t in turns if t["k"] == k])
              for k in sorted({t["k"] for t in turns})]
    for x, (label, ts) in enumerate(groups):
        bottom = 0.0
        for key, name, color in parts:  # means: the three parts add up to the mean TTFA exactly
            v = statistics.fmean(t[key] for t in ts) / 1000
            a1.bar(x, v, bottom=bottom, color=color, label=name if x == 0 else None, width=0.6)
            bottom += v
        p50 = statistics.median(t["ttfa_ms"] for t in ts) / 1000
        top = max(t["ttfa_ms"] for t in ts) / 1000
        a1.scatter([x] * len(ts), [t["ttfa_ms"] / 1000 for t in ts], s=10, color="#333333", zorder=3)
        a1.text(x, top + 0.06, f"mean {bottom:.2f} s\nP50 {p50:.2f} s", ha="center", fontsize=9)
    ref = d["chat_reference"]["A"]["pooled"]["ttfa_ms"]
    xr = len(groups)
    a1.bar(xr, ref["p50"] / 1000, color="white", edgecolor="#555555", hatch="//", width=0.6,
           label="request-response reference (different semantics)")
    a1.text(xr, ref["p50"] / 1000 + 0.08, f"P50 {ref['p50'] / 1000:.2f} s\n(n = {ref['n']})", ha="center", fontsize=9)
    a1.set_xticks(range(len(groups) + 1), [g[0] for g in groups] + ["chat path\n1 request at a time"])
    a1.set_ylabel("seconds")
    a1.set_title("Full-duplex TTFA, end of utterance → first audio\n(stack = mean of each part; dots = turns)")
    a1.set_ylim(0, 2.9)
    a1.legend(loc="upper left", fontsize=8)
    st = d["server_stages"]["pooled"]["speaking_units"]
    order = ["perception", "thinker", "talker", "speech"]
    vals = [st["stages"][s]["queue_plus_service_ms"]["p50"] for s in order]
    colors = ["#56B4E9", "#009E73", "#CC79A7", "#D55E00"]
    left = 0.0
    for s, v, c in zip(order, vals, colors):
        a2.barh(0, v, left=left, color=c, height=0.4, label=f"{s} {v:.0f} ms")
        left += v
    a2.set_yticks([])
    a2.set_xlabel("milliseconds (median queue + service per speaking unit)")
    a2.set_title(f"Inside one speaking unit (n = {st['units']}, unit time P50 {st['unit_time_ms']['p50']:.0f} ms)")
    a2.legend(loc="upper center", ncol=2, fontsize=9)
    a2.set_ylim(-0.5, 0.9)
    caption(fig, "R1 (this round): one full-duplex session per run, 3 blocks × 300 s (phase A without request events for TTFA; phase B "
                 "with events for stage times), one RTX A6000, stock SGLang-Omni 520b49ee (--variant session --enable-realtime), "
                 "synthetic spoken questions. k = extra 1-s units the model listened before speaking. The chat-path bar answers the "
                 "same questions with max_tokens 64 and one short sentence, measured from sending the whole request (no end-of-speech "
                 "detection), so it is not a like-for-like speed comparison.")
    return save(fig, out, "fig06_duplex_ttfa")


C1_BENCH = PV / "results/c1_microbench.json"
C1_QUALITY = PV / "results/c1_quality.json"
C2_BENCH = PV / "results/c2_microbench.json"
C1_GROUPS = [("exp10_t2_k4_c2_graph|w3b|fifo16", "W3b\nfifo16"), ("exp10_t2_k4_c2_graph|mix|fifo16", "MIX\nfifo16"),
             ("exp10_t2_k4_c2_graph|w3b|length8", "W3b\nlength8"), ("exp10_t2_k4_c2_graph|mix|length8", "MIX\nlength8"),
             ("exp09_t2_k2_c4|w3l|fifo16", "W3L\n2+2+4"), ("exp10val_t2_k2_c4|w3b|fifo16", "W3b open loop\n~1.2 req/batch")]
C1_ARMS = [("D", "D dense FP16 (default)", "#555555"), ("P0", "P0 packed, no compile (BF16)", "#E69F00"),
           ("P1", "P1 packed + compile (BF16, registered)", "#0072B2"), ("P1h", "P1h packed + compile FP16 (exploratory)", "#56B4E9")]


def fig_c1(out: Path) -> list[Path]:
    """C1: code2wav GPU time per audio second on real batch shapes, and the quality gate (MCD vs FP32 reference)."""
    import statistics

    b = json.loads(C1_BENCH.read_text())["per_group"]
    q = json.loads(C1_QUALITY.read_text())
    plt = plt_setup()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [2.1, 1]})
    w = 0.2
    for j, (arm, label, color) in enumerate(C1_ARMS):
        xs, ys, errs = [], [], []
        for i, (g, _) in enumerate(C1_GROUPS):
            e = b.get(g, {}).get(arm, {}).get("E_c2w")
            if not e:
                continue  # arm not run on this group: nothing drawn
            xs.append(i + (j - 1.5) * w)
            ys.append(statistics.mean(e))
            errs.append((max(e) - min(e)) / 2)
        a1.bar(xs, ys, width=w, color=color, yerr=errs, capsize=2, label=label)
    for i, (g, _) in enumerate(C1_GROUPS):
        d, p1 = b.get(g, {}).get("D", {}).get("E_c2w"), b.get(g, {}).get("P1", {}).get("E_c2w")
        if d and p1:
            rel = statistics.mean(p1) / statistics.mean(d) - 1
            a1.text(i + 0.5 * w, statistics.mean(p1) + 0.4, f"{rel * 100:+.0f}%", ha="center", fontsize=8, color="#0072B2")
    a1.set_xticks(range(len(C1_GROUPS)), [g[1] for g in C1_GROUPS], fontsize=9)
    a1.set_ylabel("code2wav GPU ms per audio second (E)")
    a1.set_title("C1: dense vs packed flow on real batch shapes (3 timed passes)")
    a1.set_ylim(0, 15.5)
    a1.legend(fontsize=8, loc="upper center", ncol=2)
    gates = q["gates"]
    d_mcd = q["vs_ref"]["D"]["summary"]["mcd_db"]
    names = ["D_ng", "P0", "P1", "P2", "P1h"]
    vals = [gates[n]["mcd_db"] - d_mcd for n in names]
    colors = ["#999999", "#E69F00", "#0072B2", "#CC79A7", "#56B4E9"]
    a2.bar(range(len(names)), vals, color=colors)
    a2.axhline(0.1, color="#D55E00", ls="--", lw=1, label="registered margin +0.1 dB")
    a2.legend(fontsize=8, loc="upper left")
    for i, (n, v) in enumerate(zip(names, vals)):
        a2.text(i, v + 0.03, f"{v:+.2f}\n{'pass' if gates[n]['pass'] else 'fail'}", ha="center", fontsize=8)
    a2.set_xticks(range(len(names)), names)
    a2.set_ylabel("MCD vs FP32 reference, minus D's (dB)")
    a2.set_title("Quality gate (26 segments)")
    a2.set_ylim(-0.1, 1.5)
    caption(fig, "C1 microbenchmark (this round), one RTX A6000: flow + HiFT of code2wav loaded alone, replaying 112 real batches per arm sampled "
                 "from the 47,484 exported from Experiments 9–10 (32 in each of the two fifo16 groups, 12 in each other group). Bars = mean of 3 timed passes (error bars = half range, ≤ 0.3 %). "
                 f"MCD of D vs the FP32 reference is {d_mcd:.2f} dB because the sampled excitation differs between runs, so only the "
                 "difference to D is shown; D_ng (same math as D) fails the ASR rule, which marks the noise floor. Microbenchmark, "
                 "not an end-to-end result.")
    return save(fig, out, "fig07_c1_flow_packed")


def fig_c2(out: Path) -> list[Path]:
    """C2: HiFT GPU time per audio second for default, fold, channels-last and FP16 arms."""
    import statistics

    c2 = json.loads(C2_BENCH.read_text())
    plt = plt_setup()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
    groups = list(c2["E_hift"])
    labels = {"A": "A default (FP32)", "B": "B weight-norm folded", "C": "C channels-last", "D": "D FP16 body"}
    colors = {"A": "#555555", "B": "#009E73", "C": "#E69F00", "D": "#CC79A7"}
    w = 0.2
    for j, arm in enumerate("ABCD"):
        ys = [statistics.mean(c2["E_hift"][g][arm]) for g in groups]
        errs = [(max(c2["E_hift"][g][arm]) - min(c2["E_hift"][g][arm])) / 2 for g in groups]
        a1.bar([i + (j - 1.5) * w for i in range(len(groups))], ys, width=w, yerr=errs, capsize=2, color=colors[arm], label=labels[arm])
    short = {"exp10_t2_k4_c2_graph|w3b|fifo16": "W3b fifo16", "exp10_t2_k4_c2_graph|mix|fifo16": "MIX fifo16", "exp09_t2_k2_c4|w3l|fifo16": "W3L 2+2+4"}
    a1.set_xticks(range(len(groups)), [short.get(g, g) for g in groups])
    a1.set_ylabel("HiFT GPU ms per audio second")
    a1.set_title("C2: HiFT time per arm (3 timed passes)")
    a1.legend(fontsize=8, loc="upper right")
    prof = c2["profile"]
    E = {a: statistics.mean(statistics.mean(v[a]) for v in c2["E_hift"].values()) for a in "ABCD"}
    kern = {a: sum(prof[a]["families_ms_per_audio_s"].values()) for a in "ABCD"}
    a2.bar(range(4), [E[a] for a in "ABCD"], color="#DDDDDD", label="event-timed HiFT time")
    a2.bar(range(4), [kern[a] for a in "ABCD"], color=[colors[a] for a in "ABCD"], width=0.5, label="sum of kernel time (profiler)")
    for i, a in enumerate("ABCD"):
        a2.text(i, E[a] + 0.05, f"{kern[a] / E[a] * 100:.0f}% kernels", ha="center", fontsize=8)
    a2.set_xticks(range(4), list("ABCD"))
    a2.set_ylabel("ms per audio second (mean of groups)")
    a2.set_title("Kernel time vs event-timed time")
    a2.legend(fontsize=8, loc="upper left")
    caption(fig, "C2 microbenchmark (this round), one RTX A6000, fixed real-length mels and fixed excitation noise, decode_body compiled "
                 "as in the stock component. Quality gates: B bit-identical (pass); C rel. RMS 9.1e-4, MCD 0.18 dB (fail); D MCD 0.83 dB "
                 "(fail); quality compared on 26 re-synthesized segments. Right: kernel sums come from a separate profiler pass. Microbenchmark, not an end-to-end result.")
    return save(fig, out, "fig08_c2_hift")


R2_SUMMARY = PV / "results/r2_summary.json"
R2B_SUMMARY = PV / "results/r2b_summary.json"


def fig_duplex_sessions(out: Path) -> list[Path]:
    """R2: concurrent full-duplex sessions on one A6000 vs unit lag by unit kind, unit misses, deadline misses, stalls."""
    d = json.loads(R2_SUMMARY.read_text())
    levels = d["arms"]["ms8_1gpu"]["levels"]
    xs = [lv["level"] for lv in levels]
    plt = plt_setup()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    sp50 = [lv["descriptive"]["unit_lag_speaking_ms"].get("p50") for lv in levels]
    sp99 = [lv["descriptive"]["unit_lag_speaking_ms"].get("p99") for lv in levels]
    li50 = [lv["descriptive"]["unit_lag_listening_ms"].get("p50") for lv in levels]
    a1.plot(xs, sp50, marker="o", color="#0072B2", label="speaking units, P50")
    a1.plot(xs, sp99, marker="o", ls="--", color="#0072B2", label="speaking units, P99")
    a1.plot(xs, li50, marker="s", color="#999999", label="listening units, P50")
    a1.axhline(1000, color="#D55E00", ls=":", lw=1)
    a1.text(xs[0], 1030, "unit length (1 s): later units fall behind", color="#D55E00", fontsize=8)
    for x, lv in zip(xs, levels):
        miss = lv["unit_miss_rate"] * 100
        oom = sum((b.get("oom_events") or 0) for b in lv["per_block"].values())
        per = [b["unit_miss_rate"] * 100 for b in lv["per_block"].values() if b.get("ran") and b.get("unit_miss_rate") is not None]
        split = len(per) > 1 and max(per) - min(per) > 5  # pooled rate hides blocks that differ (R2 at 8 sessions)
        note = (f"miss {miss:.1f}% pooled\n(blocks {', '.join(f'{x:.1f}%' for x in per)})" if split else f"miss {miss:.1f}%") \
            + (f"\n{oom} OOM log lines" if oom else "")
        a1.text(x, (lv["descriptive"]["unit_lag_speaking_ms"].get("p99") or 0) + 60, note, ha="center", fontsize=8)
    r2b_note = ""
    if R2B_SUMMARY.exists():  # R2b: the registered 2-GPU split (thinker on its own GPU), drawn only when it ran
        b = json.loads(R2B_SUMMARY.read_text())
        blv = [lv for lv in b["arms"]["ms8_2gpu"]["levels"] if lv.get("descriptive")]
        bx = [lv["level"] for lv in blv]
        a1.plot(bx, [lv["descriptive"]["unit_lag_speaking_ms"].get("p50") for lv in blv], marker="D", ls="none", color="#D55E00",
                ms=8, label="2 GPUs (R2b), speaking units, P50")
        a1.plot(bx, [lv["descriptive"]["unit_lag_speaking_ms"].get("p99") for lv in blv], marker="D", ls="none", color="#D55E00",
                mfc="white", ms=8, label="2 GPUs (R2b), speaking units, P99")
        verdict = (b.get("verdict_h_r2b") or {}).get("verdict")
        r2b_note = (" Diamonds: R2b, the registered 2-GPU split (thinker on its own GPU) at 4 and 8 sessions, "
                    f"{len(b.get('runs') or {})} block(s); registered verdict: {VERDICT_EN.get(verdict, verdict)}; "
                    + "; ".join(f"{lv['level']} sessions miss {lv['unit_miss_rate'] * 100:.1f}%, "
                                f"{(lv.get('pooled') or {}).get('oom_events', 0)} OOM" for lv in blv if lv.get("unit_miss_rate") is not None) + ".")
    a1.set_xscale("log", base=2)
    a1.set_xticks(xs, [str(x) for x in xs])
    a1.set_xlabel("concurrent sessions on one A6000")
    a1.set_ylabel("unit lag (ms): last input sample → unit done")
    a1.set_title("1-s unit processing vs sessions")
    a1.set_ylim(0, 1800)
    a1.legend(fontsize=8, loc="upper left")
    for deadline, ls in (("0.5s", ":"), ("1s", "-"), ("2s", "--")):
        ys = [lv["deadline_miss_all_turns"][deadline] * 100 for lv in levels]
        a2.plot(xs, ys, marker="o", ls=ls, color="#CC79A7", label=f"turns missing a {deadline} TTFA deadline")
    stall = []
    for lv in levels:
        p = lv["pooled"]
        stall.append(100 * p["replies_with_stall"]["80ms"] / p["replies_with_audio"] if p.get("replies_with_audio") else None)
    a2.plot(xs, stall, marker="^", color="#009E73", label="replies with a playback stall (80 ms buffer)")
    a2.set_xscale("log", base=2)
    a2.set_xticks(xs, [str(x) for x in xs])
    a2.set_ylim(0, 105)
    a2.set_xlabel("concurrent sessions on one A6000")
    a2.set_ylabel("percent")
    a2.set_title("Interaction quality vs sessions")
    a2.legend(fontsize=8, loc="lower right")
    blocks = {lv["level"]: len(lv["blocks_ran"]) for lv in levels}
    caption(fig, "R2 (this round): stock SGLang-Omni 520b49ee, --variant session --enable-realtime --max_sessions 8 on one RTX A6000; synthetic "
                 "spoken questions; 180 s per level, levels 1 → 8 within a run. Blocks per level: " + ", ".join(f"{k}: {v}" for k, v in blocks.items())
            + " (block 2 stopped after level 1 under the registered stop rule: a first-reply warm-up missed 1.67 % of units). "
              "Unit lag is bimodal, so speaking and listening units are shown separately. Registered capacity verdicts: "
              "below the lowest level (see the results). Deadline misses at 1 session are mostly the model's turn-taking. At 8 sessions "
              "the two blocks differ: sessions failed with CUDA OOM in one block only." + r2b_note)
    return save(fig, out, "fig09_duplex_sessions")


R3_SCREEN = PV / "results/r3_screen.json"
R3_CONFIRM = PV / "results/r3_confirm.json"


def fig_r3_camera(out: Path) -> list[Path]:
    """R3 screening: accepted vs offered camera requests per shape and preprocessing concurrency, and where a request's
    server time goes at 1 vs 16 sessions."""
    d = json.loads(R3_SCREEN.read_text())
    by: dict[tuple, dict[int, dict[str, Any]]] = {}
    for run in d["runs"]:
        for c in run["cells"]:
            by.setdefault((c["shape"], c["sessions"], c["fps"]), {})[c["max_concurrency"]] = c
    plt = plt_setup()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    from matplotlib.lines import Line2D

    colors = {"frame": "#0072B2", "clip": "#E69F00"}
    styles = {1: ":", 4: "--", 16: "-"}
    for shape in ("frame", "clip"):
        for sessions in (1, 4, 16):
            cells = sorted((k, v) for k, v in by.items() if k[0] == shape and k[1] == sessions)
            xs = [k[1] * k[2] for k, _ in cells]
            a1.plot(xs, [v[1]["sent"] / 60.0 for _, v in cells], marker="o", ls=styles[sessions], color=colors[shape])
            a1.plot(xs, [v[4]["sent"] / 60.0 for _, v in cells], marker="D", ls="none", color=colors[shape], mfc="white", ms=7)
    lim = [0.4, 40]
    a1.plot(lim, lim, color="#999999", lw=1, ls=":")
    a1.text(0.5, 0.62, "accepted = offered", color="#777777", fontsize=8, rotation=37)
    confirm_note = ""
    if R3_CONFIRM.exists():  # 3-block confirmation of the registered cells: block range (min-max) at concurrency 1 and 4
        conf = json.loads(R3_CONFIRM.read_text())["confirmation"]
        for key, v in conf.items():
            x = v.get("offered_per_s")
            for conc, dx, mk in (("c1", 0.94, "o"), ("c4", 1.06, "D")):
                ys = [y for y in (v["sent_per_s"].get(conc) or {}).values() if y is not None]
                if x and ys:
                    a1.errorbar([x * dx], [statistics.fmean(ys)], yerr=[[statistics.fmean(ys) - min(ys)], [max(ys) - statistics.fmean(ys)]],
                                fmt=mk, color="black", mfc="black" if conc == "c1" else "white", ms=4, capsize=3, lw=1)
        verdicts = {k: VERDICT_EN.get(v["verdict"], v["verdict"]) for k, v in conf.items()}
        confirm_note = (" Black: 3-block confirmation of the registered cells (mean, whiskers = block range; filled = concurrency 1). "
                        "Registered verdicts for concurrency 4: " + ", ".join(f"{k} {v}" for k, v in sorted(verdicts.items())) + ".")
    handles = [Line2D([], [], color=colors["frame"], marker="o", label="frame (one PNG)"),
               Line2D([], [], color=colors["clip"], marker="o", label="clip (2 s H.264)"),
               *[Line2D([], [], color="#555555", ls=styles[n], label=f"{n} session{'s' if n > 1 else ''}, fps 0.5 / 1 / 2") for n in (1, 4, 16)],
               Line2D([], [], color="#555555", marker="D", mfc="white", ls="none", label="preprocessing concurrency 4 (filled: 1)")]
    a1.legend(handles=handles, fontsize=8, loc="upper left")
    a1.set_xscale("log", base=2)
    a1.set_yscale("log", base=2)
    ticks = [0.5, 1, 2, 4, 8, 16, 32]
    a1.set_xticks(ticks, [f"{t:g}" for t in ticks])
    a1.set_yticks(ticks, [f"{t:g}" for t in ticks])
    a1.set_xlabel("offered requests/s (sessions × fps)")
    a1.set_ylabel("accepted requests/s (sent, 60 s window)")
    a1.set_title("Camera requests per A6000: accepted vs offered")
    stages = [("preprocessing_service_ms_p50", "CPU preprocessing (service)", "#009E73"),
              ("image_encoder_service_ms_p50", "image encoder (GPU)", "#D55E00"),
              ("thinker_prefill_ms_p50", "thinker prefill (GPU)", "#56B4E9")]
    groups = [("frame", 1, 1.0), ("frame", 16, 1.0), ("clip", 1, 1.0), ("clip", 16, 1.0)]
    w = 0.25
    for i, (key, label, color) in enumerate(stages):
        vals = [by[g][1]["server"][key] for g in groups]
        xs = [j + (i - 1) * w for j in range(len(groups))]
        a2.bar(xs, vals, width=w, color=color, label=label)
        for x, v in zip(xs, vals):
            a2.text(x, v * 1.08, f"{v:.0f}", ha="center", fontsize=7)
    a2.set_yscale("log")
    a2.set_xticks(range(len(groups)), [f"{s}\n{n} session{'s' if n > 1 else ''}" for s, n, f in groups])
    a2.set_ylim(8, 20000)
    a2.set_ylabel("median per request (ms, log scale)")
    a2.set_title("Where a request's server time goes (1 fps, concurrency 1)")
    a2.legend(fontsize=8, loc="upper center", ncol=3)
    n = {g: by[g][1]["sent"] for g in groups}
    caption(fig, "R3 screening (this round): one --text-only replica on one RTX A6000 and 7 CPU cores, SGLang-Omni 520b49ee + "
                 "default-off preprocessing-concurrency patch; synthetic 640×360 camera frames (frame = one PNG, clip = last 2 s at "
                 "8 fps as H.264), one request in flight per session (late ticks dropped), 60 s per cell, one run per concurrency "
                 "(no blocks yet). Sessions 1/4/16 × 0.5/1/2 fps. Right: stage medians from request events (not additive); "
                 "requests per bar group: " + ", ".join(f"{s} {k}×{f:g}: {v}" for (s, k, f), v in n.items())
            + ". At concurrency 1 all 5 saturated cells are GPU-bound by the registered rule (at concurrency 4 two of them are 'cause unclear', "
              "74–75 % GPU busy); concurrency 4 improved no cell." + confirm_note)
    return save(fig, out, "fig10_r3_camera")


P1_SUMMARY = PV / "results/p1_summary.json"
P1_SETTINGS = [("fifo16", "FIFO16", "#999999", "o"), ("length8", "Length8", "#0072B2", "s"),
               ("length8_w500", "Length8 + max wait 0.5 s", "#56B4E9", "v"),
               ("length8_w1000", "Length8 + max wait 1 s", "#D55E00", "D"),
               ("length8_w2000", "Length8 + max wait 2 s", "#CC79A7", "^")]
VERDICT_EN = {"支持": "supported", "反驳": "refuted", "不确定": "inconclusive", "数据不足": "insufficient data", "不支持": "not supported"}


def _num(x: Any) -> float | None:
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None


def fig_p1_batching(out: Path) -> list[Path]:
    """P1: code2wav batching policy on the full 2+2+4 service near saturation -- the three registered contrasts."""
    d = json.loads(P1_SUMMARY.read_text())
    summ = d["summary"]
    plt = plt_setup()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    panels = [("mix", "goodput_rps", 1.0, "MIX: goodput (req/s)", "H1a"),
              ("mix", "short_queue_p99_ms", 1e-3, "MIX: short requests' code2wav queue P99 (s)", "H1b"),
              ("w3b", "e2e_p95", 1.0, "W3b: end-to-end P95 (s)", "H1c")]
    unbounded = []
    for ax, (w, metric, scale, title, h) in zip(axes, panels):
        for j, (st, label, color, mk) in enumerate(P1_SETTINGS):
            pts = []
            for tag in ("x090", "x100", "x110"):
                c = summ.get(f"{w}|{st}|{tag}")
                if not c or not c.get(metric):
                    continue
                mean, x = _num(c[metric].get("mean")), _num((c.get("offered_rps") or {}).get("mean"))
                if x is None:
                    continue
                if mean is None:
                    unbounded.append(f"{w} {st} {tag}")
                    continue
                ci = c[metric].get("ci95")
                steady = all(bool(v) for v in c.get("steady") or [])
                pts.append((x + (j - 2) * 0.09, mean * scale,
                            (mean - ci[0]) * scale if ci and _num(ci[0]) is not None else 0.0,
                            (ci[1] - mean) * scale if ci and _num(ci[1]) is not None else 0.0, steady))
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=1.2, label=label)
            for x, y, lo, hi, steady in pts:
                ax.errorbar([x], [y], yerr=[[lo], [hi]], marker=mk, color=color, mfc=color if steady else "white", capsize=3, ls="none")
        verdict = (d.get("verdicts") or {}).get(h) or {}
        ax.set_title(f"{title}\n{h}: {VERDICT_EN.get(verdict.get('verdict'), verdict.get('verdict') or 'n/a')}", fontsize=10)
        ax.set_xlabel("offered load (req/s): 0.9X / 1.0X / 1.1X")
    axes[0].legend(fontsize=8, loc="best")
    caption(fig, "P1 (this round): full service on 2+2+4, runtime 520b49ee + experiment 10's batching patch + max wait (code2wav only), open loop "
                 "with Poisson arrivals at 0.9X/1.0X/1.1X where X is the pre-run's busy-window throughput (FIFO16); 75 s measured per cell "
                 "(W3b, MIX) after warm-up, same arrival sequence for all settings within a block; 3 blocks; mean with 95 % t interval "
                 "(df 2). Hollow markers: not steady in every block (an unsteady cell's throughput is not a capacity). Goodput = TTFT ≤ 1 s "
                 "and e2e ≤ 0.5 × audio length; failures count against it. Short = below the cell's median audio length."
            + (f" Not drawn (unbounded mean): {', '.join(unbounded)}." if unbounded else ""))
    return save(fig, out, "fig11_p1_batching")


R4_SUMMARY = PV / "results/r4_working_set.json"


def fig_r4_working_set(out: Path) -> list[Path]:
    """R4: KV working set vs capacity per session count, and reuse lost to eviction or routing (registered thresholds)."""
    d = json.loads(R4_SUMMARY.read_text())
    runs = {run: r for run, r in sorted(d["runs"].items()) if r.get("levels")}
    if not runs:
        raise FileNotFoundError(2, "no R4 main run in the summary", str(R4_SUMMARY))
    plt = plt_setup()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    shades = ["#0072B2", "#56B4E9", "#009E73"]
    by_level: dict[int, dict[str, list[float]]] = {}
    for (run, r), color in zip(runs.items(), shades):
        lv = sorted((x for x in r["levels"] if "_media" not in x["phase"] and x.get("sessions")), key=lambda x: x["sessions"])
        xs = [x["sessions"] for x in lv]
        logical = [max((x.get("logical_ratio_peak") or {}).values(), default=None) for x in lv]
        resident = [max((p.get("resident_peak_ratio") or 0.0 for p in (x.get("pool") or {}).values()), default=None) for x in lv]
        block = re.search(r"block(\d+)", run)
        tag = f"block {block.group(1)}" if block else run
        a1.plot(xs, logical, marker="o", color=color, label=f"logical working set / capacity ({tag})")
        a1.plot(xs, resident, marker="s", ls="--", color=color, mfc="white", label=f"resident KV / capacity ({tag})")
        for x in lv:
            share = ((x.get("attribution") or {}).get("lost_share_of_prefill") or {})
            slot = by_level.setdefault(x["sessions"], {"eviction": [], "routing": []})
            for cause in ("eviction", "routing"):
                slot[cause].append((share.get(cause) or 0.0) * 100)
    a1.axhline(0.8, color="#D55E00", ls=":", lw=1)
    a1.text(8, 0.82, "0.8: flagged, does not decide", color="#D55E00", fontsize=8)
    a1.set_xscale("log", base=2)
    levels = sorted(by_level)
    a1.set_xticks(levels, [str(v) for v in levels])
    a1.set_xlabel("concurrent robot sessions (2+2+4, two thinker replicas)")
    a1.set_ylabel("peak ratio (max over thinker replicas)")
    a1.set_title("KV working set vs capacity")
    a1.legend(fontsize=7, loc="upper left")
    w = 0.38
    for i, (cause, color) in enumerate((("eviction", "#CC79A7"), ("routing", "#E69F00"))):
        xs = [k + (i - 0.5) * w for k in range(len(levels))]
        means = [statistics.fmean(by_level[v][cause]) if by_level[v][cause] else 0.0 for v in levels]
        a2.bar(xs, means, width=w, color=color, label=f"lost to {cause} (mean over blocks)")
        for x, v in zip(xs, levels):
            a2.scatter([x] * len(by_level[v][cause]), by_level[v][cause], color="black", s=8, zorder=3)
    a2.axhline(10, color="#D55E00", ls=":", lw=1)
    a2.text(-0.4, 10.4, "registered threshold: 10 % of prefill tokens (and TTFT ≥ 1.2× clean)", color="#D55E00", fontsize=8)
    a2.set_xticks(range(len(levels)), [str(v) for v in levels])
    a2.set_xlabel("concurrent robot sessions")
    a2.set_ylabel("lost reuse (% of the level's prefill tokens)")
    a2.set_title("Reuse lost to eviction or to replica routing")
    a2.legend(fontsize=8, loc="upper left")
    comb = (d.get("combined") or {}).get("conclusion")
    caption(fig, "R4 (this round): full service on 2+2+4, runtime 520b49ee + default-off KV logging patch, synthetic multimodal robot "
                 "sessions (one new camera frame and a unique spoken request per turn, half the turns with a tool call; earlier turns "
                 "sent back as text), 8 → 128 sessions in one run per block, " + f"{len(runs)} block(s). "
                 "Logical working set = the latest prompt + completion of every live session on a replica; resident = cached + running "
                 "KV. Dots = blocks. The media-history contrast (32 sessions) is reported in the text, not here. "
            + ("Registered conclusion across blocks: see the results." if comb else ""))
    return save(fig, out, "fig12_r4_working_set")


FIGURES: dict[str, Callable[[Path], list[Path]]] = {"placement": fig_placement, "p0_admission": fig_p0_admission,
                                                    "p0_capacity": fig_p0_capacity, "c2w_shapes": fig_c2w_shapes,
                                                    "c2w_kernels": fig_c2w_kernels, "duplex_ttfa": fig_duplex_ttfa,
                                                    "c1": fig_c1, "c2": fig_c2, "duplex_sessions": fig_duplex_sessions,
                                                    "r3_camera": fig_r3_camera, "p1_batching": fig_p1_batching,
                                                    "r4_working_set": fig_r4_working_set}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(PV / "figures"))
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    for name, fn in FIGURES.items():
        if a.only and name not in a.only:
            continue
        try:
            paths = fn(out)
            print(f"{name}: " + ", ".join(p.name for p in paths))
        except FileNotFoundError as exc:
            print(f"{name}: skipped (missing input {exc.filename})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
