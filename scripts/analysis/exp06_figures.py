#!/usr/bin/env python3
"""实验 6 的图：从 exp06_report.py 写出的 results/ 重新画，另可给一组 kernel trace 画 CPU/GPU 时间线。

    python scripts/analysis/exp06_figures.py --out research/01_性能瓶颈/实验06_Code2Wav时间拆解 \
        [--timeline CODE2WAV_TRACE TALKER_TRACE CODE2WAV_EVENTS]

Inputs: results/summary.json, results/requests.csv, results/batches.csv. Unit of replication is the run (block);
points are per-run values, bars are means over runs, error bars are t-based 95 % intervals over runs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp01_gpu_timeline import setup_matplotlib  # noqa: E402

LEVELS = [8, 64, 128, 256]
# Okabe-Ito colors with a marker per concurrency (color is never the only cue).
LEVEL_STYLE = {8: ("#0072B2", "o"), 64: ("#009E73", "s"), 128: ("#E69F00", "^"), 256: ("#D55E00", "D")}
PART_STYLE = [("queue_ms", "在收件队列里排队", "#56B4E9"), ("batching_ms", "凑批", "#F0E442"),
              ("compute_ms", "计算（这一批）", "#CC79A7"), ("handoff_ms", "交出结果", "#999999")]
COMPUTE_PARTS = [("setup", "准备（参考音频、上传）", "#F0E442"), ("flow_issue_ms", "CPU 发 flow", "#0072B2"),
                 ("hift_issue_ms", "CPU 发 HiFT（含等 flow 算完的同步）", "#009E73"),
                 ("download_wait_ms", "等波形拷回", "#D55E00"), ("rest", "其他", "#999999")]


def read_csv(path: Path) -> list[dict[str, Any]]:
    def num(v: str) -> Any:
        try:
            return float(v)
        except ValueError:
            return v if v != "" else None
    with open(path) as fh:
        return [{k: num(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def t95(values: list[float]) -> tuple[float, float]:
    """Mean and half-width of the t-based 95 % interval over runs."""
    t = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
    v = np.array([x for x in values if x is not None], dtype=float)
    if v.size < 2:
        return (float(v.mean()) if v.size else float("nan"), 0.0)
    return float(v.mean()), float(t.get(v.size - 1, 2.0) * v.std(ddof=1) / np.sqrt(v.size))


def fig_breakdown_by_level(plt, requests: list[dict[str, Any]], path: Path) -> dict[str, Any]:  # noqa: ANN001
    runs = sorted({r["run"] for r in requests})
    per_run: dict[tuple[str, int], dict[str, float]] = {}
    for run in runs:
        for c in LEVELS:
            rows = [r for r in requests if r["run"] == run and r["concurrency"] == c and r.get("total_ms") is not None and r.get("queue_ms") is not None]
            if rows:
                per_run[(run, c)] = {p: float(np.mean([r[p] for r in rows])) for p, _, _ in PART_STYLE} | {"total_ms": float(np.mean([r["total_ms"] for r in rows])), "n": len(rows)}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    out: dict[str, Any] = {}
    for i, c in enumerate(LEVELS):
        vals = [per_run[(run, c)] for run in runs if (run, c) in per_run]
        if not vals:
            continue
        left = 0.0
        means = {p: float(np.mean([v[p] for v in vals])) for p, _, _ in PART_STYLE}
        total_mean, total_half = t95([v["total_ms"] for v in vals])
        for p, label, color in PART_STYLE:
            a1.bar(i, means[p] / 1e3, bottom=left / 1e3, color=color, edgecolor="black", linewidth=0.4, label=label if i == 0 else None)
            share = means[p] / sum(means.values())
            a2.bar(i, share * 100, bottom=sum(means[q] for q, _, _ in PART_STYLE[:[x[0] for x in PART_STYLE].index(p)]) / sum(means.values()) * 100,
                   color=color, edgecolor="black", linewidth=0.4)
            if share > 0.06:
                a2.text(i, (sum(means[q] for q, _, _ in PART_STYLE[:[x[0] for x in PART_STYLE].index(p)]) / sum(means.values()) + share / 2) * 100,
                        f"{share * 100:.0f}%", ha="center", va="center", fontsize=8)
            left += means[p]
        a1.errorbar(i, total_mean / 1e3, yerr=total_half / 1e3, color="black", capsize=4, lw=1)
        a1.scatter([i + 0.28] * len(vals), [v["total_ms"] / 1e3 for v in vals], color="black", marker="o", s=12, zorder=3,
                   label="每个区组的平均总时间" if i == 0 else None)
        out[str(c)] = {"runs": len(vals), "mean_parts_ms": means, "total_mean_ms": total_mean, "total_ci95_half_ms": total_half,
                       "per_run_total_ms": [v["total_ms"] for v in vals], "requests_per_run": [v["n"] for v in vals]}
    for ax in (a1, a2):
        ax.set_xticks(range(len(LEVELS)))
        ax.set_xticklabels([f"并发 {c}" for c in LEVELS])
    a1.set_ylabel("code2wav 阶段的平均时间（秒）")
    a1.set_title("（a）绝对时间：各段均值相加 = 总时间均值；误差线为 3 个区组的 95% 区间", fontsize=9, loc="left")
    a2.set_ylabel("占 code2wav 阶段时间的比例（%）")
    a2.set_ylim(0, 100)
    a2.set_title("（b）组成比例", fontsize=9, loc="left")
    a1.legend(fontsize=8, loc="upper left")
    fig.suptitle("图 2　并发 8 → 256，code2wav 阶段的时间变成了什么（2+6，W3b，每个请求，打点段 B）", fontsize=10, x=0.01, ha="left")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return out


def fig_typical_request(plt, requests: list[dict[str, Any]], batches: list[dict[str, Any]], path: Path) -> dict[str, Any]:  # noqa: ANN001
    run0 = sorted({r["run"] for r in requests})[0]
    by_batch = {(b["run"], b["pid"], b["batch_id"]): b for b in batches}
    fig, axes = plt.subplots(2, 1, figsize=(12, 3.8), layout="constrained")
    out: dict[str, Any] = {}
    for ax, c in zip(axes, (8, 256)):
        rows = [r for r in requests if r["run"] == run0 and r["concurrency"] == c and r.get("queue_ms") is not None and r.get("batch_id") is not None]
        if not rows:
            continue
        med = float(np.median([r["total_ms"] for r in rows]))
        req = min(rows, key=lambda r: abs(r["total_ms"] - med))
        b = by_batch.get((run0, req["batch_pid"], req["batch_id"]), {})
        compute_parts = {"setup": sum((b.get(k) or 0) for k in ("prepare_ms", "references_ms", "upload_ms")),
                         **{k: b.get(k) or 0 for k in ("flow_issue_ms", "hift_issue_ms", "download_wait_ms")}}
        compute_parts["rest"] = max(0.0, req["compute_ms"] - sum(compute_parts.values()))
        segs = [("在收件队列里排队", req["queue_ms"], "#56B4E9"), ("凑批", req["batching_ms"], "#F0E442")]
        segs += [(label, compute_parts[key], color) for key, label, color in COMPUTE_PARTS]
        segs += [("交出结果", req["handoff_ms"], "#999999")]
        left = 0.0
        for label, w, color in segs:
            ax.barh(0, w, left=left, color=color, edgecolor="black", linewidth=0.4, label=label)
            if w > 0.06 * req["total_ms"]:
                ax.text(left + w / 2, 0, f"{w:.0f}", ha="center", va="center", fontsize=8)
            left += w
        ax.set_yticks([])
        ax.set_xlim(0, req["total_ms"] * 1.02)
        ax.set_xlabel("毫秒（从放进 code2wav 收件队列开始）")
        ax.set_title(f"并发 {c}：总时间最接近中位数（{med:.0f} 毫秒）的那个请求，它所在的批有 {int(b.get('size') or 0)} 个请求、"
                     f"flow 走 {b.get('flow_solve_path', '?')}；GPU 上 flow {b.get('gpu_flow_ms') or 0:.0f} 毫秒、HiFT {b.get('gpu_hift_ms') or 0:.0f} 毫秒",
                     fontsize=9, loc="left")
        out[str(c)] = {"request_id": req["request_id"], "total_ms": req["total_ms"], "segments_ms": {k: v for k, v, _ in segs},
                       "batch_size": b.get("size"), "flow_solve_path": b.get("flow_solve_path"), "gpu_flow_ms": b.get("gpu_flow_ms"), "gpu_hift_ms": b.get("gpu_hift_ms")}
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("图 1　一个典型请求在 code2wav 里的时间：低负载几乎全是计算，高负载排队和“等同批算完”都变长（区组 1，打点段 B；两行横轴范围不同）",
                 fontsize=10, x=0.01, ha="left")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return out


def fig_queue_depth(plt, requests: list[dict[str, Any]], path: Path) -> dict[str, Any]:  # noqa: ANN001
    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    out: dict[str, Any] = {}
    for c in LEVELS:
        rows = [r for r in requests if r["concurrency"] == c and r.get("queue_ms") is not None and r.get("queue_depth_at_dequeue") is not None]
        if not rows:
            continue
        color, marker = LEVEL_STYLE[c]
        x = np.array([r["queue_depth_at_dequeue"] for r in rows]) + np.random.default_rng(c).uniform(-0.2, 0.2, len(rows))
        y = np.maximum([r["queue_ms"] for r in rows], 0.05)
        ax.scatter(x, y, s=6, alpha=0.35, color=color, marker=marker, label=f"并发 {c}（{len(rows)} 个请求）")
        out[str(c)] = {"requests": len(rows), "depth_p50": float(np.median([r["queue_depth_at_dequeue"] for r in rows])),
                       "queue_ms_p50": float(np.median([r["queue_ms"] for r in rows]))}
    ax.set_yscale("log")
    ax.set_xlabel("这个请求被取出时，收件队列里还有几个请求在等（加了 ±0.2 的抖动，只为看清点）")
    ax.set_ylabel("它在收件队列里等了多久（毫秒，对数坐标；0 记为 0.05）")
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(alpha=0.3)
    ax.set_title("图 4　队列深度和排队时间：取出时队列几乎总是空的，等待主要来自前一批还没算完", fontsize=10, loc="left")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return out


def fig_batch_speed(plt, batches: list[dict[str, Any]], path: Path) -> dict[str, Any]:  # noqa: ANN001
    panels = [("gpu_flow_ms", "flow 的 GPU 时间 ÷ 批大小"), ("gpu_hift_ms", "HiFT 的 GPU 时间 ÷ 批大小"), ("compute_ms", "这一批的计算墙钟时间 ÷ 批大小")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained", sharey=True)
    out: dict[str, Any] = {}
    for ax, (key, title) in zip(axes, panels):
        by_size: dict[int, list[float]] = defaultdict(list)
        for c in LEVELS:
            rows = [b for b in batches if b["concurrency"] == c and b.get(key) is not None and b.get("size")]
            if not rows:
                continue
            color, marker = LEVEL_STYLE[c]
            sizes = np.array([b["size"] for b in rows])
            per = np.array([b[key] / b["size"] for b in rows])
            ax.scatter(sizes + np.random.default_rng(c).uniform(-0.15, 0.15, len(rows)), per, s=7, alpha=0.4, color=color, marker=marker,
                       label=f"并发 {c}" if key == "gpu_flow_ms" else None)
            for s_, p_ in zip(sizes, per):
                by_size[int(s_)].append(float(p_))
        xs = sorted(k for k, v in by_size.items() if len(v) >= 5)
        ax.plot(xs, [np.median(by_size[x]) for x in xs], color="black", lw=1.5, label="中位数（≥5 批的批大小）" if key == "gpu_flow_ms" else None)
        out[key] = {str(x): {"batches": len(by_size[x]), "median_ms_per_request": float(np.median(by_size[x]))} for x in xs}
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("批大小（一批里的请求数）")
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("每个请求分摊到的时间（毫秒）")
    axes[0].legend(fontsize=7, markerscale=2)
    fig.suptitle("图 5　批大小和每个请求的处理成本：批变大，每个请求分摊的 GPU 时间并没有变少", fontsize=10, x=0.01, ha="left")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return out


def fig_kernel(plt, summary: dict[str, Any], path: Path) -> dict[str, Any]:  # noqa: ANN001
    by_level = summary["by_level"]
    metrics = [("kernel.graph_kernel_time_share", "来自 CUDA graph 的 kernel 时间比例", 100, "%"),
               ("launch_share", "CPU 在发 kernel 调用里的时间 ÷ 窗口时间", 100, "%"),
               ("sync_share", "CPU 在同步调用里的时间 ÷ 窗口时间", 100, "%"),
               ("busy_share", "code2wav 的 kernel 在跑的时间 ÷ 窗口时间", 100, "%")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), layout="constrained")
    out: dict[str, Any] = {}
    for ax, (key, title, scale, unit) in zip(axes, metrics):
        for i, c in enumerate(LEVELS):
            entry = by_level.get(f"c{c}_C", {})
            win = entry.get("kernel.window_ms", {}).get("runs") or []
            if key == "launch_share":
                vals = [a / w for a, w in zip(entry.get("kernel.launch_api_ms", {}).get("runs") or [], win) if w]
            elif key == "sync_share":
                vals = [a / w for a, w in zip(entry.get("kernel.sync_api_ms", {}).get("runs") or [], win) if w]
            elif key == "busy_share":
                vals = [a / w for a, w in zip(entry.get("kernel.kernel_busy_ms", {}).get("runs") or [], win) if w]
            else:
                vals = entry.get(key, {}).get("runs") or []
            if not vals:
                continue
            m, h = t95(vals)
            color, marker = LEVEL_STYLE[c]
            ax.bar(i, m * scale, color=color, alpha=0.5, edgecolor="black", linewidth=0.4)
            ax.errorbar(i, m * scale, yerr=h * scale, color="black", capsize=3, lw=1)
            ax.scatter([i] * len(vals), [v * scale for v in vals], color="black", marker=marker, s=12, zorder=3)
            out.setdefault(key, {})[str(c)] = {"runs": vals, "mean": m, "ci95_half": h}
        ax.set_xticks(range(len(LEVELS)))
        ax.set_xticklabels([str(c) for c in LEVELS])
        ax.set_xlabel("并发")
        ax.set_ylabel(unit)
        ax.set_ylim(bottom=0)
        ax.set_title(title, fontsize=8.5)
    fig.suptitle("图 6　kernel trace 里的 code2wav 进程（每个区组一个 3 秒窗口，6 个进程相加；点为各区组，误差线为 95% 区间；开着 profiler，发 kernel 的开销偏高）",
                 fontsize=9.5, x=0.01, ha="left")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return out


def fig_timeline(plt, c2w_trace: Path, talker_trace: Path, events_file: Path, path: Path) -> dict[str, Any]:  # noqa: ANN001
    def load(p: Path) -> dict[str, Any]:
        with gzip.open(p, "rt") if p.suffix == ".gz" else open(p) as fh:
            return json.load(fh)
    c2w, talker = load(c2w_trace), load(talker_trace)
    base = int(c2w.get("baseTimeNanoseconds") or 0)
    tbase = int(talker.get("baseTimeNanoseconds") or 0)
    vocodes = {}
    with open(events_file) as fh:
        for line in fh:
            if "exp_vocode" in line:
                e = json.loads(line)
                vocodes.setdefault(e["metadata"]["batch_id"], e["metadata"])
    # The batch with the median compute time among those whose whole vocode call lies inside the trace window.
    ev = [e for e in c2w["traceEvents"] if e.get("ph") == "X" and "ts" in e]
    t_lo = base + min(e["ts"] for e in ev) * 1e3
    t_hi = base + max(e["ts"] + e.get("dur", 0) for e in ev) * 1e3
    inside = [m for m in vocodes.values() if m["prepare_start_ns"] >= t_lo and m["download_end_ns"] <= t_hi]
    if not inside:
        return {}
    inside.sort(key=lambda m: m["download_end_ns"] - m["prepare_start_ns"])
    m = inside[len(inside) // 2]
    t0 = m["prepare_start_ns"]
    to_ms = lambda ns: (ns - t0) / 1e6  # noqa: E731
    fig, ax = plt.subplots(figsize=(13, 3.6), layout="constrained")
    ranges = [("准备", "prepare_start_ns", "flow_issue_start_ns", "#F0E442"), ("CPU 发 flow", "flow_issue_start_ns", "flow_issue_end_ns", "#0072B2"),
              ("CPU 发 HiFT（含同步等待）", "flow_issue_end_ns", "hift_issue_end_ns", "#009E73"), ("等拷回", "hift_issue_end_ns", "download_end_ns", "#D55E00")]
    for label, a, b, color in ranges:
        ax.broken_barh([(to_ms(m[a]), to_ms(m[b]) - to_ms(m[a]))], (2.55, 0.7), color=color, label=label)
    for e in ev:
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and "Synchronize" in e["name"]:
            s = to_ms(base + e["ts"] * 1e3)
            ax.broken_barh([(s, e.get("dur", 0) / 1e3)], (2.1, 0.35), color="#7f0000")
    flow_a, flow_b, hift_b = (to_ms(m["flow_issue_start_ns"]), to_ms(m["flow_issue_end_ns"]), to_ms(m["hift_issue_end_ns"]))
    calls = {(e.get("args") or {}).get("correlation"): e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver")}
    span_end = to_ms(m["download_end_ns"]) * 1.05
    for e in ev:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        s = to_ms(base + e["ts"] * 1e3)
        if s < -5 or s > span_end:
            continue
        call = calls.get((e.get("args") or {}).get("correlation"))
        cs = to_ms(base + call["ts"] * 1e3) if call else None
        color = "#0072B2" if cs is not None and flow_a <= cs <= flow_b else "#009E73" if cs is not None and flow_b < cs <= hift_b else "#bbbbbb"
        ax.broken_barh([(s, max(e.get("dur", 0) / 1e3, 0.02))], (1.2, 0.7), color=color, lw=0)
    for e in talker["traceEvents"]:
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            s = (tbase + e["ts"] * 1e3 - t0) / 1e6
            if -5 <= s <= span_end:
                ax.broken_barh([(s, max(e.get("dur", 0) / 1e3, 0.02))], (0.2, 0.7), color="#999999", lw=0)
    ax.set_yticks([0.55, 1.55, 2.3, 2.9])
    ax.set_yticklabels(["GPU：同一张卡上 talker 的 kernel", "GPU：这批 code2wav 的 kernel\n（蓝 = flow，绿 = HiFT）", "CPU：同步调用（暗红）",
                        "CPU：code2wav 调度线程在做什么"], fontsize=8)
    ax.set_xlim(-5, span_end)
    ax.set_xlabel("毫秒（从这一批开始准备算起）")
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)
    sizes = len(m.get("token_lengths") or [])
    ax.set_title(f"图 3　一批 code2wav 的 CPU/GPU 时间线（区组 1、并发 128、开着 profiler；这一批 {sizes} 个请求、flow 走 {m.get('flow_solve_path')}）",
                 fontsize=10, loc="left")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {"batch_rows": sizes, "flow_solve_path": m.get("flow_solve_path"), "cpu_ranges_ms": {label: to_ms(m[b]) - to_ms(m[a]) for label, a, b, _ in ranges},
            "gpu_flow_ms": m.get("gpu_flow_start_to_flow_end_ms"), "gpu_hift_ms": m.get("gpu_flow_end_to_hift_end_ms")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeline", nargs=3, metavar=("CODE2WAV_TRACE", "TALKER_TRACE", "CODE2WAV_EVENTS"))
    a = ap.parse_args()
    out = Path(a.out)
    plt = setup_matplotlib()
    summary = json.load(open(out / "results" / "summary.json"))
    requests = read_csv(out / "results" / "requests.csv")
    batches = read_csv(out / "results" / "batches.csv")
    figs: dict[str, Any] = {"generated_by": "scripts/analysis/exp06_figures.py"}
    figs["fig1_typical_request"] = fig_typical_request(plt, requests, batches, out / "figures" / "fig1_typical_request.png")
    figs["fig2_breakdown_by_level"] = fig_breakdown_by_level(plt, requests, out / "figures" / "fig2_breakdown_by_level.png")
    if a.timeline:
        figs["fig3_cpu_gpu_timeline"] = fig_timeline(plt, *map(Path, a.timeline), out / "figures" / "fig3_cpu_gpu_timeline.png")
    figs["fig4_queue_depth"] = fig_queue_depth(plt, requests, out / "figures" / "fig4_queue_depth.png")
    figs["fig5_batch_size"] = fig_batch_speed(plt, batches, out / "figures" / "fig5_batch_size.png")
    figs["fig6_kernel"] = fig_kernel(plt, summary, out / "figures" / "fig6_kernel.png")
    (out / "results" / "figures_data.json").write_text(json.dumps(figs, ensure_ascii=False, indent=1, default=float) + "\n")
    print(out / "results" / "figures_data.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
