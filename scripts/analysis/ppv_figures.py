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
P0_ARMS = [("A", "A: cap 32 (default)", "#CC79A7", "s"), ("G", "G: cap 64, graphs ≤32", "#56B4E9", "v"),
           ("B", "B: cap 64, graphs ≤64", "#0072B2", "D"), ("R", "Ref: 2+2+4, cap 32", "#E69F00", "o")]


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
    caption(fig, "P0 (this round): 2+1+5 placement (1 talker GPU) with talker max_running_requests 32 (A), 64 with CUDA graphs "
                 "up to 32 (G) or up to 64 (B); reference 2+2+4. Runs per arm: " + ", ".join(f"{k} {v}" for k, v in n.items())
            + ". Left: medians over SGLang decode-log lines in the closed-loop W3b 256 level (talker lines separated by tokens per "
              "running request > 200); points = blocks, bars = t 95 % CI. Right: median log throughput per batch size, "
              f"pooled over all talker lines of each arm's runs (≥ 20 lines per size). {COMMON}.")
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
                errbar(a1, j * 5 + i, st, color, marker, label=label if j == 0 else None)
    a1.set_xticks([1.5, 6.5], ["W3b (256 in flight)", "W3L (128 in flight)"])
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
    caption(fig, "P0 (this round), 3 blocks per arm (points = blocks, bars = t 95 % CI). Capacity = completed / (last completion − "
                 "first send) at the top closed-loop level without request events (registered, as in Experiment 9). Right: mean "
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


FIGURES: dict[str, Callable[[Path], list[Path]]] = {"placement": fig_placement, "p0_admission": fig_p0_admission,
                                                    "p0_capacity": fig_p0_capacity, "c2w_shapes": fig_c2w_shapes,
                                                    "c2w_kernels": fig_c2w_kernels, "duplex_ttfa": fig_duplex_ttfa}


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
