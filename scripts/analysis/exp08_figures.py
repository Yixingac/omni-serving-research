#!/usr/bin/env python3
"""实验 8 的图：从 results/ 里的 phase1.json、zoom.json（以及第三阶段的 phase3.json，若已有）画。

    python scripts/analysis/exp08_figures.py --out research/01_性能瓶颈/实验08_Thinker_Lookahead

Unit of replication is the run (block): bars are means over blocks, points are per-block values, error bars are
t-based 95 % intervals over blocks. Cells of the text conditions at concurrency 32 and 64 in C phases come from
the supplement blocks (hatched), every other cell from the main blocks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp01_gpu_timeline import setup_matplotlib  # noqa: E402

LEVELS = ["8", "32", "64"]
CONDITIONS = [("w3t", "只要文字，带种子", "#E69F00", "o"), ("w3t_ns", "只要文字，不带种子", "#0072B2", "s"),
              ("w3b", "文字 + 语音，带种子", "#D55E00", "^"), ("w3b_ns", "文字 + 语音，不带种子", "#009E73", "D")]
SUPP = {("w3t", "32"), ("w3t", "64"), ("w3t_ns", "32"), ("w3t_ns", "64")}
BLOCKED_BY = {"w3t": "挡住它的：种子", "w3t_ns": "没有被挡", "w3b": "挡住它的：语音 + 种子", "w3b_ns": "挡住它的：语音"}


def grouped(ax, cells: dict[str, Any], metric: str, ylabel: str, hatch_supp: bool = False) -> None:  # noqa: ANN001
    width = 0.2
    for i, (cond, label, color, marker) in enumerate(CONDITIONS):
        for j, lv in enumerate(LEVELS):
            cell = cells[cond][lv][metric]
            x = j + (i - 1.5) * width
            if cell["mean"] is None:
                ax.text(x, 0, "无数据", rotation=90, ha="center", va="bottom", fontsize=6, color="#666666")
                continue
            hatch = "///" if hatch_supp and (cond, lv) in SUPP else None
            ax.bar(x, cell["mean"], width * 0.92, color=color, alpha=0.55, edgecolor=color, hatch=hatch, label=label if j == 0 else None)
            if cell["ci95"]:
                lo, hi = cell["ci95"]
                ax.errorbar(x, cell["mean"], yerr=[[cell["mean"] - lo], [hi - cell["mean"]]], color="#333333", capsize=2, lw=0.8)
            runs = [r for r in cell["runs"] if r is not None]
            ax.scatter([x] * len(runs), runs, s=12, color=color, edgecolor="black", linewidths=0.4, marker=marker, zorder=3)
    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels([f"并发 {lv}" for lv in LEVELS], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)


def fig_timeline(plt, zoom: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    from matplotlib.patches import Patch

    panels = [("w3t_c8_C", "只要文字、带种子（同步执行）"), ("w3t_ns_c8_C", "只要文字、不带种子（提前一步发射）"),
              ("w3b_ns_c8_C", "文字 + 语音、不带种子（语音挡住，同步执行）")]
    panels = [(k, t) for k, t in panels if zoom["phases"].get(k)]
    span = max(zoom["phases"][k]["span_ms"] for k, _ in panels)
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 1.55 * len(panels) + 1.2), sharex=True)
    for ax, (key, title) in zip(np.atleast_1d(axes), panels):
        z = zoom["phases"][key]
        s = z["summary"]
        graph = [(a, max(d, 0.02)) for a, d, g in z["gpu"] if g]
        other = [(a, max(d, 0.02)) for a, d, g in z["gpu"] if not g]
        ax.broken_barh(graph, (1.55, 0.7), color="#d94801", lw=0)
        ax.broken_barh(other, (1.55, 0.7), color="#fd8d3c", lw=0)
        ax.broken_barh([(0, z["span_ms"])], (0.15, 0.7), color="#fdd0a2", lw=0)
        for kind, color, minimum in (("api", "#969696", 0.01), ("launch", "#2171b5", 0.05), ("sync", "#cb181d", 0.01),
                                     ("copy_wait", "#67000d", 0.01)):
            ax.broken_barh([(a, max(d, minimum)) for a, d, k in z["cpu"] if k == kind], (0.15, 0.7), color=color, lw=0)
        ax.set_yticks([0.5, 1.9])
        ax.set_yticklabels(["CPU（调度线程）", "GPU（thinker 的卡）"], fontsize=7)
        ax.set_ylim(0, 2.4)
        ax.set_title(f"{title}：这一段窗口里每步 GPU 空档中位数 {s['idle_ms_p50']:.2f} 毫秒，"
                     f"CPU 等这一步算完才发下一步的比例 {s['waited_share']:.0%}", fontsize=8, loc="left")
    ax.set_xlim(0, span)
    ax.set_xlabel("时间（毫秒，从截取的第一步发射前 2 毫秒算起；每幅截取 5 步）", fontsize=8)
    fig.legend(handles=[Patch(color="#d94801", label="GPU：CUDA graph 里的 kernel（一步的主体）"),
                        Patch(color="#fd8d3c", label="GPU：graph 之外的小 kernel 和拷贝"),
                        Patch(color="#cb181d", label="CPU：卡在同步上，等 GPU 算完"),
                        Patch(color="#67000d", label="CPU：把这一步的 token id 拷回内存，同时等 GPU 算完（也是同步）"),
                        Patch(color="#2171b5", label="CPU：发出下一步（cudaGraphLaunch）"),
                        Patch(color="#969696", label="CPU：其他 CUDA 调用"),
                        Patch(color="#fdd0a2", label="CPU：Python 和调度")],
               fontsize=7, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("图 1　thinker 连续几步的 CPU/GPU 时间线（区组 1，并发 8，3 秒 kernel trace 的中段）", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_lookahead(plt, cells: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [3, 2]})
    grouped(a1, cells, "steps.decode_lookahead_share", "纯 decode 轮次里提前发射的比例")
    a1.set_ylim(0, 1.15)
    a1.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    for i, (cond, _, color, _) in enumerate(CONDITIONS):
        a1.text(0.01, 0.97 - 0.07 * i, BLOCKED_BY[cond], transform=a1.transAxes, fontsize=7, color=color, va="top")
    a1.legend(fontsize=7, loc="upper right", frameon=False)
    a1.set_title("（a）提前一步发射用上了多少（每轮事件）", fontsize=9, loc="left")
    width = 0.5
    cell = [cells["w3t_ns"][lv]["flush_share"] for lv in LEVELS]
    for j, c in enumerate(cell):
        a2.bar(j, c["mean"], width, color="#0072B2", alpha=0.55, edgecolor="#0072B2")
        if c["ci95"]:
            a2.errorbar(j, c["mean"], yerr=[[c["mean"] - c["ci95"][0]], [c["ci95"][1] - c["mean"]]], color="#333333", capsize=2, lw=0.8)
        a2.scatter([j] * len(c["runs"]), c["runs"], s=12, color="#0072B2", edgecolor="black", linewidths=0.4, marker="s", zorder=3)
    a2.set_xticks(range(len(LEVELS)))
    a2.set_xticklabels([f"并发 {lv}" for lv in LEVELS], fontsize=8)
    a2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    a2.set_ylabel("先收尾（flush）的轮次占比", fontsize=8)
    a2.tick_params(labelsize=7)
    a2.set_title("（b）只要文字、不带种子：新请求插进来时要先收尾", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_idle(plt, cells: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, ax = plt.subplots(figsize=(10, 4))
    grouped(ax, cells, "graph_steps.idle_ms_p50", "thinker 每步 GPU 空档的中位数（毫秒）", hatch_supp=True)
    busy = [cells[c][lv]["graph_steps.busy_ms_p50"]["mean"] for c, *_ in CONDITIONS for lv in LEVELS]
    busy = [b for b in busy if b is not None]
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.set_title(f"图 3　每一步 GPU 空了多久（每步 GPU 忙的时间约 {min(busy):.0f}–{max(busy):.0f} 毫秒；斜线 = 补测区组）",
                 fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_token(plt, cells: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8))
    grouped(a1, cells, "thinker_ms_per_token.p50", "thinker 每个 token 的时间，中位数（毫秒）")
    a1.set_title("（a）thinker 每生成一个 token 要多久", fontsize=9, loc="left")
    a1.legend(fontsize=7, loc="upper left", frameon=False)
    grouped(a2, cells, "client.throughput_rps", "吞吐（请求/秒）")
    a2.set_title("（b）整个服务每秒完成多少请求", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


P3_ARMS = [("off", "原型关（语音同步执行）", "#D55E00"), ("on", "原型开（语音也提前发射，发射时快照）", "#009E73")]
P3_PANELS = [("C", "graph_steps.idle_ms_p50", "（a）每步 GPU 空档，中位数（毫秒）", 1.0),
             ("C", "sync_calls_per_step", "（b）每步 CPU 停下来等 GPU 的次数", 1.0),
             ("B", "steps.decode_lookahead_share", "（c）提前发射的 decode 步（%）", 100.0),
             ("B", "thinker_ms_per_token.p50", "（d）thinker 每个 token，中位数（毫秒）", 1.0),
             ("A", "client.throughput_rps", "（e）吞吐（请求/秒）", 1.0),
             ("A", "client.e2e_s.p50", "（f）端到端 P50（秒）", 1.0),
             ("A", "client.e2e_s.p95", "（g）端到端 P95（秒）", 1.0),
             ("A", "client.e2e_s.p99", "（h）端到端 P99（秒）", 1.0),
             ("A", "client.ttft_s.p50", "（i）TTFT 中位数（毫秒）", 1000.0),
             ("A", "thinker_mem_max_mib", "（j）thinker 卡显存峰值（GiB）", 1 / 1024)]


def fig_phase3(plt, phase3: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    levels = sorted(phase3["contrasts"], key=int)
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.6))
    width = 0.36
    for ax, (mode, metric, title, scale) in zip(axes.flat, P3_PANELS):
        for j, lv in enumerate(levels):
            cell = phase3["contrasts"][lv][mode][metric]
            xs = {arm: j + (i - 0.5) * width for i, (arm, *_) in enumerate(P3_ARMS)}
            for arm, label, color in P3_ARMS:
                c = cell[arm]
                if c["mean"] is None:
                    ax.text(xs[arm], 0, "无数据", rotation=90, ha="center", va="bottom", fontsize=6, color="#666666")
                    continue
                ax.bar(xs[arm], c["mean"] * scale, width * 0.9, color=color, alpha=0.5, edgecolor=color, label=label if j == 0 else None)
                if c["ci95"]:
                    lo, hi = c["ci95"]
                    ax.errorbar(xs[arm], c["mean"] * scale, yerr=[[(c["mean"] - lo) * scale], [(hi - c["mean"]) * scale]],
                                color="#333333", capsize=2, lw=0.8)
            for a, b in zip(cell["off"]["runs"], cell["on"]["runs"]):
                if a is not None and b is not None:
                    ax.plot([xs["off"], xs["on"]], [a * scale, b * scale], color="#555555", lw=0.6, marker="o", ms=2.5, zorder=3)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([f"并发 {lv}" for lv in levels], fontsize=7)
        ax.set_title(title, fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("图 5　文字 + 语音请求（不带种子）：原型关对比原型开。柱 = 3 个区组的均值，误差线 = 95% 区间，灰线连起同一区组的两次运行",
                 fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    plt = setup_matplotlib()
    phase1 = json.loads((out / "results" / "phase1.json").read_text())
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig_timeline(plt, json.loads((out / "results" / "zoom.json").read_text()), figs / "fig1_step_timeline.png")
    fig_lookahead(plt, phase1["cells"], figs / "fig2_lookahead_share.png")
    fig_idle(plt, phase1["cells"], figs / "fig3_step_idle.png")
    fig_token(plt, phase1["cells"], figs / "fig4_token_time.png")
    if (out / "results" / "phase3.json").exists():
        fig_phase3(plt, json.loads((out / "results" / "phase3.json").read_text()), figs / "fig5_phase3_off_on.png")
    print(sorted(p.name for p in figs.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
