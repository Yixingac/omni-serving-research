#!/usr/bin/env python3
"""实验 7 的图：从 exp07_report.py 写出的 results/summary.json 画。

    python scripts/analysis/exp07_figures.py --out research/01_性能瓶颈/实验07_Code2Wav独立GPU

Points are per-block values, lines and bars are means over blocks, error bars are t-based 95 % intervals over
blocks. Levels with more than 5 % errors are left out (exp07_report.py marks them unusable).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slurm"))

from exp01_gpu_timeline import setup_matplotlib  # noqa: E402
from make_plan_exp06_08 import PLACEMENTS  # noqa: E402

ARMS = [("t2_k6_shared", "方案 A：talker、code2wav 同在 6 张卡（2+6，基线）", "#000000", "o", "-"),
        ("t2_k4_c2", "方案 B：talker 4 卡 + code2wav 独占 2 卡", "#0072B2", "s", "-"),
        ("t2_k3_c3", "方案 C：talker 3 卡 + code2wav 独占 3 卡", "#009E73", "^", "-"),
        ("t2_k2_c4", "方案 D：talker 2 卡 + code2wav 独占 4 卡", "#E69F00", "D", "-"),
        ("t2_k3_shared", "方案 E：talker、code2wav 同在 3 张卡（只用 5 卡，机制对照）", "#999999", "v", "--")]
LETTER = {arm: label.split("：")[0] for arm, label, *_ in ARMS}


def plain_log(ax, axis: str = "y") -> None:  # noqa: ANN001
    """Log axis with ordinary numbers (2, 3, 5, 10) instead of 2×10⁰."""
    from matplotlib.ticker import FuncFormatter, LogLocator

    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(FuncFormatter(lambda v, _: ""))
ROLE_COLORS = {"thinker": "#CC79A7", "talker": "#56B4E9", "code2wav": "#D55E00", "unused": "#FFFFFF"}


def series(by_arm: dict[str, Any], arm: str, section: str, path: str) -> tuple[list[float], list[dict[str, Any]]]:
    rows = by_arm.get(arm, {}).get(section, {})
    keys = sorted(rows, key=float)
    return [float(k) for k in keys], [rows[k][path] for k in keys]


def plot_series(ax, by_arm: dict[str, Any], section: str, path: str, scale: float = 1.0) -> None:  # noqa: ANN001
    for arm, label, color, marker, ls in ARMS:
        xs, cells = series(by_arm, arm, section, path)
        pts = [(x, c) for x, c in zip(xs, cells) if c["mean"] is not None]
        if not pts:
            continue
        ax.plot([x for x, _ in pts], [c["mean"] * scale for _, c in pts], ls, color=color, marker=marker, ms=5, label=label, lw=1.4)
        for x, c in pts:
            if c["ci95"]:
                ax.errorbar(x, c["mean"] * scale, yerr=[[(c["mean"] - c["ci95"][0]) * scale], [(c["ci95"][1] - c["mean"]) * scale]],
                            color=color, capsize=2, lw=0.8)
            runs = [r * scale for r in c["runs"] if r is not None]
            ax.scatter([x] * len(runs), runs, s=10, color=color, alpha=0.5, marker=marker, linewidths=0)


def fig_layouts(plt, path: Path) -> None:  # noqa: ANN001
    from matplotlib.patches import Patch, Rectangle

    fig, ax = plt.subplots(figsize=(10, 3.6))
    for row, (arm, label, *_) in enumerate(ARMS):
        thinker, talker, c2w = PLACEMENTS[arm]
        y = len(ARMS) - 1 - row
        for g in range(8):
            roles = [r for r, gs in (("thinker", thinker), ("talker", talker), ("code2wav", c2w)) if g in gs] or ["unused"]
            for i, r in enumerate(roles):
                ax.add_patch(Rectangle((g + i / len(roles), y + 0.1), 1 / len(roles), 0.8, facecolor=ROLE_COLORS[r],
                                       edgecolor="#444444", lw=0.6, hatch="//" if r == "unused" else None))
        ax.text(-0.15, y + 0.5, label, ha="right", va="center", fontsize=8)
    ax.set_xlim(-0.05, 8.05)
    ax.set_ylim(0, len(ARMS))
    ax.set_xticks([g + 0.5 for g in range(8)])
    ax.set_xticklabels([f"GPU {g}" for g in range(8)], fontsize=8)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Patch(facecolor=ROLE_COLORS["thinker"], label="thinker（pipeline 进程）"),
                       Patch(facecolor=ROLE_COLORS["talker"], label="talker"),
                       Patch(facecolor=ROLE_COLORS["code2wav"], label="code2wav"),
                       Patch(facecolor="#FFFFFF", edgecolor="#444444", hatch="//", label="空着")],
              fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.set_title("图 1　五种放置：每张卡上跑什么（一格分成两半 = 两个进程同卡）", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_closed_throughput(plt, by_arm: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
    fig.suptitle("图 2　闭环：同时在系统里的请求越多，每种放置每秒能完成多少", fontsize=10, x=0.01, ha="left")
    plot_series(a1, by_arm, "closed", "client.throughput_rps")
    a1.set_xscale("log", base=2)
    a1.set_xticks([8, 64, 128, 256])
    a1.set_xticklabels(["8", "64", "128", "256"])
    a1.set_xlabel("闭环并发（同时在系统里的请求数）", fontsize=8)
    a1.set_ylabel("吞吐（成功完成的请求/秒）", fontsize=8)
    a1.set_title("（a）闭环吞吐", fontsize=9, loc="left")
    a1.legend(fontsize=6.5, loc="upper left", frameon=False)
    plot_series(a2, by_arm, "closed", "client.audio_s_per_s")
    a2.set_xscale("log", base=2)
    a2.set_xticks([8, 64, 128, 256])
    a2.set_xticklabels(["8", "64", "128", "256"])
    a2.set_xlabel("闭环并发", fontsize=8)
    a2.set_ylabel("每秒合成的语音秒数", fontsize=8)
    a2.set_title("（b）每秒能合成多少秒语音", fontsize=9, loc="left")
    for ax in (a1, a2):
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_open_latency(plt, by_arm: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    fig.suptitle("图 3　开环：请求按固定速率随机到达时，用户要等多久", fontsize=10, x=0.01, ha="left")
    for ax, p, title in ((axes[0], "client.e2e_s.p50", "（a）端到端 P50"), (axes[1], "client.e2e_s.p95", "（b）端到端 P95")):
        plot_series(ax, by_arm, "open", p)
        ax.axvspan(12.5, 19, color="#eeeeee", zorder=0)
        ax.text(12.7, ax.get_ylim()[1] * 0.95, "探索性速率（预跑之后才加）", fontsize=7, va="top", color="#666666")
        ax.set_yscale("log")
        plain_log(ax)
        ax.set_xlabel("开环到达速率（请求/秒，泊松到达）", fontsize=8)
        ax.set_ylabel("秒（对数刻度）", fontsize=8)
        ax.set_title(title, fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=6.5, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_gpu(plt, summary: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [3, 2]})
    fig.suptitle("图 4　闭环并发 256 时每种卡有多忙，以及同卡时 talker 和 code2wav 抢卡的时间", fontsize=10, x=0.01, ha="left")
    roles = ["thinker", "talker", "code2wav", "talker+code2wav"]
    colors = {"thinker": ROLE_COLORS["thinker"], "talker": ROLE_COLORS["talker"], "code2wav": ROLE_COLORS["code2wav"], "talker+code2wav": "#8c6bb1"}
    width = 0.2
    labeled: set[str] = set()
    for i, (arm, label, *_rest) in enumerate(ARMS):
        runs = [r for r in summary["per_run"] if r["arm"] == arm]
        for j, role in enumerate(roles):
            vals = [get(r, f"closed.256.gpu.{role}.busy") for r in runs]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            x = i + (j - 1.5) * width
            a1.bar(x, np.mean(vals), width * 0.9, color=colors[role], edgecolor="#444444", lw=0.5, label=None if role in labeled else role)
            labeled.add(role)
            a1.scatter([x] * len(vals), vals, s=8, color="black", zorder=3)
    a1.set_xticks(range(len(ARMS)))
    a1.set_xticklabels([LETTER[a[0]] for a in ARMS], fontsize=8)
    a1.set_ylim(0, 1.05)
    a1.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    a1.set_ylabel("nvidia-smi 忙碌比例（闭环并发 256）", fontsize=8)
    handles, labels = a1.get_legend_handles_labels()
    names = {"thinker": "thinker 卡", "talker": "talker 卡", "code2wav": "code2wav 卡", "talker+code2wav": "talker、code2wav 同卡"}
    a1.legend(handles, [names[x] for x in labels], fontsize=7, loc="lower right", frameon=False)
    a1.set_title("（a）每种卡有多忙", fontsize=9, loc="left")
    for i, (arm, label, color, *_r) in enumerate(ARMS):
        runs = [r for r in summary["per_run"] if r["arm"] == arm and r.get("kernel_c128")]
        vals = [r["kernel_c128"]["overlap_share_mean"] for r in runs if r["kernel_c128"].get("overlap_share_mean") is not None]
        if vals:
            a2.bar(i, np.mean(vals), 0.6, color=color, alpha=0.6)
            a2.scatter([i] * len(vals), vals, s=10, color="black", zorder=3)
        else:
            a2.text(i, 0.01, "不同卡\n不会重叠" if not arm.endswith("shared") else "无数据", ha="center", fontsize=6.5, color="#666666")
    a2.set_xticks(range(len(ARMS)))
    a2.set_xticklabels([LETTER[a[0]] for a in ARMS], fontsize=8)
    a2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    a2.set_ylabel("同一张卡上 talker、code2wav 的 kernel 同时在跑的时间占比", fontsize=7)
    a2.set_title("（b）同卡时两者有多少时间在抢（并发 128，3 秒 trace）", fontsize=9, loc="left")
    for ax in (a1, a2):
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def get(d: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def fig_stages(plt, by_arm: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    from matplotlib.ticker import FuncFormatter

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    fig.suptitle("图 5　各段的处理时间和排队时间（闭环，中位数）", fontsize=10, x=0.01, ha="left")
    for ax, p, title, ylabel in ((axes[0], "talker_gen_ms_per_audio_s.p50", "（a）talker 生成 1 秒语音要多久", "毫秒 / 每秒语音（中位数）"),
                                 (axes[1], "c2w_flow_us_per_frame_b1.p50", "（b）code2wav：单个请求的批，flow 每帧 GPU 时间", "微秒 / 帧（中位数）"),
                                 (axes[2], "breakdown.queue_ms.p50", "（c）code2wav 收件队列里的排队", "毫秒（中位数）")):
        plot_series(ax, by_arm, "closed", p)
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 64, 128, 256])
        ax.set_xticklabels(["8", "64", "128", "256"])
        ax.set_xlabel("闭环并发", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
    axes[2].set_yscale("log")
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_tradeoff(plt, by_arm: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for arm, label, color, marker, ls in ARMS:
        _, tp = series(by_arm, arm, "closed", "client.throughput_rps")
        levels, p50 = series(by_arm, arm, "closed", "client.e2e_s.p50")
        pts = [(t["mean"], p["mean"], lv) for t, p, lv in zip(tp, p50, levels) if t["mean"] is not None and p["mean"] is not None]
        if not pts:
            continue
        ax.plot([x for x, _, _ in pts], [y for _, y, _ in pts], ls, color=color, marker=marker, label=label, lw=1.4)
        for x, y, lv in pts:
            ax.annotate(f"{lv:g}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=6, color=color)
    ax.set_yscale("log")
    plain_log(ax)
    ax.set_xlabel("吞吐（请求/秒）", fontsize=8)
    ax.set_ylabel("端到端 P50（秒，对数刻度）", fontsize=8)
    ax.set_title("图 6　吞吐和延迟的取舍（闭环，点旁边的数字是并发）", fontsize=10, loc="left")
    ax.legend(fontsize=6.5, loc="upper left", frameon=False)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    summary = json.loads((out / "results" / "summary.json").read_text())
    plt = setup_matplotlib()
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig_layouts(plt, figs / "fig1_layouts.png")
    fig_closed_throughput(plt, summary["by_arm"], figs / "fig2_closed_throughput.png")
    fig_open_latency(plt, summary["by_arm"], figs / "fig3_open_latency.png")
    fig_gpu(plt, summary, figs / "fig4_gpu_busy.png")
    fig_stages(plt, summary["by_arm"], figs / "fig5_stages.png")
    fig_tradeoff(plt, summary["by_arm"], figs / "fig6_tradeoff.png")
    print(sorted(p.name for p in figs.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
