#!/usr/bin/env python3
"""实验 9 的图：从 exp09_report.py 写出的 results/summary.json 画。

    python scripts/analysis/exp09_figures.py --out research/01_性能瓶颈/实验09_最佳GPU分工

Lines and bars are means over blocks, small points are the blocks, error bars are t-based 95 % intervals over blocks.
Open-loop points that the registered slope rule marks non-steady are drawn hollow. Every figure is drawn from
results/summary.json only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slurm"))

from exp01_gpu_timeline import setup_matplotlib  # noqa: E402
from make_plan_exp09_11 import PLACEMENTS  # noqa: E402

ARMS = [(1, "t2_k6_shared", "方案 1：talker、code2wav 同在 6 张卡（2+6，对照）", "#000000", "o"),
        (2, "t2_k2_c4", "方案 2：talker 2 张 + code2wav 4 张", "#E69F00", "D"),
        (3, "t2_k1_c5", "方案 3：talker 1 张 + code2wav 5 张", "#CC79A7", "s"),
        (4, "t2_k3_c3", "方案 4：talker 3 张 + code2wav 3 张", "#009E73", "^")]
ROLE_COLORS = {"thinker": "#56B4E9", "talker": "#0072B2", "code2wav": "#D55E00", "talker+code2wav": "#8c6bb1"}
WL_NAME = {"w3b": "W3b（短回答）", "w3l": "W3L（长回答）"}


def get(d: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def levels(summary: dict[str, Any], arm: int, w: str, kind: str, mode: str) -> list[tuple[float, dict[str, Any]]]:
    cells = summary["by_arm"].get(str(arm), {})
    out = []
    for key, cell in cells.items():
        ww, kk, mm, lv = key.split("|")
        if ww == w and kk == kind and mm == mode:
            out.append((float(lv), cell))
    return sorted(out)


def draw_series(ax, xs, cells, label, color, marker, field, scale=1.0, ls="-", hollow=None):  # noqa: ANN001
    pts = [(x, c[field]) for x, c in zip(xs, cells) if c[field]["mean"] is not None]
    if not pts:
        return
    ax.plot([x for x, _ in pts], [v["mean"] * scale for _, v in pts], ls, color=color, lw=1.3, label=label)
    for i, (x, v) in enumerate(pts):
        face = "white" if hollow and hollow[i] else color
        ax.plot([x], [v["mean"] * scale], marker, color=color, mfc=face, ms=5)
        if v["ci95"]:
            ax.errorbar(x, v["mean"] * scale, yerr=[[(v["mean"] - v["ci95"][0]) * scale], [(v["ci95"][1] - v["mean"]) * scale]],
                        color=color, capsize=2, lw=0.7)
        runs = [r * scale for r in v["runs"] if r is not None]
        ax.scatter([x] * len(runs), runs, s=8, color=color, alpha=0.45, linewidths=0)


def fig_layouts(plt, path: Path) -> None:  # noqa: ANN001
    from matplotlib.patches import Patch, Rectangle

    fig, ax = plt.subplots(figsize=(10, 3.2))
    for row, (arm, name, label, *_) in enumerate(ARMS):
        thinker, talker, c2w = PLACEMENTS[name]
        y = len(ARMS) - 1 - row
        for g in range(8):
            roles = [r for r, gs in (("thinker", thinker), ("talker", talker), ("code2wav", c2w)) if g in gs]
            for i, r in enumerate(roles):
                ax.add_patch(Rectangle((g + i / len(roles), y + 0.1), 1 / len(roles), 0.8, facecolor=ROLE_COLORS[r], edgecolor="#444444", lw=0.6))
        ax.text(-0.15, y + 0.5, label, ha="right", va="center", fontsize=8)
    ax.set_xlim(-0.05, 8.05)
    ax.set_ylim(0, len(ARMS))
    ax.set_xticks([g + 0.5 for g in range(8)])
    ax.set_xticklabels([f"GPU {g}" for g in range(8)], fontsize=8)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Patch(facecolor=ROLE_COLORS[r], label=n) for r, n in (("thinker", "thinker（pipeline 进程）"), ("talker", "talker"),
                                                                            ("code2wav", "code2wav"))],
              fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.set_title("图 1　四种分法：每张卡上跑什么（一格分成两半 = 两个进程同卡）", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def per_card(summary: dict[str, Any], w: str, field: str) -> dict[str, list[float]]:
    """Capacity / cards of the bottleneck stage variant, per run, from runs where that stage is the bottleneck."""
    cards = {1: {"code2wav": 6, "talker": 6, "thinker": 2}, 2: {"code2wav": 4, "talker": 2, "thinker": 2},
             3: {"code2wav": 5, "talker": 1, "thinker": 2}, 4: {"code2wav": 3, "talker": 3, "thinker": 2}}
    out: dict[str, list[float]] = {}
    for r in summary["runs"]:
        x = r["workload"].get(w) or {}
        stage = x.get("bottleneck")
        if not stage:
            continue
        cap = x.get("capacity_rps") if field == "registered" else next(
            (get(lv, "busy_window.throughput_rps") for lv in r["levels"] if lv["workload"] == w and lv["kind"] == "closed_loop"
             and lv["mode"] == "A" and lv.get("concurrency") == (256 if w == "w3b" else 128)), None)
        if cap is None:
            continue
        variant = f"{stage}（{'同卡' if r['arm'] == 1 and stage != 'thinker' else '独占'}）"
        out.setdefault(variant, []).append(cap / cards[r["arm"]][stage])
    return out


def fig_stage_capacity(plt, summary: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, w in zip(axes, ("w3b", "w3l")):
        width = 0.38
        for j, (field, label, color) in enumerate((("registered", "登记的容量 ÷ 卡数", "#444444"), ("busy", "满额时段吞吐 ÷ 卡数（描述性）", "#56B4E9"))):
            data = per_card(summary, w, field)
            names = sorted(data)
            for i, n in enumerate(names):
                v = data[n]
                ax.bar(i + (j - 0.5) * width, np.mean(v), width * 0.9, color=color, alpha=0.6, label=label if i == 0 else None)
                ax.scatter([i + (j - 0.5) * width] * len(v), v, s=10, color="black", zorder=3)
                ax.text(i + (j - 0.5) * width, np.mean(v), f"{np.mean(v):.1f}", ha="center", va="bottom", fontsize=7)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("每张卡每秒完成的请求", fontsize=8)
        ax.set_title(f"（{'a' if w == 'w3b' else 'b'}）{WL_NAME[w]}：瓶颈阶段每张卡的处理能力", fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("图 2　按“瓶颈阶段”算出的每张卡的处理能力（只用该阶段是瓶颈的运行；点是每个运行）", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_closed(plt, summary: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, w in zip(axes, ("w3b", "w3l")):
        for arm, _, label, color, marker in ARMS:
            lv = levels(summary, arm, w, "closed_loop", "A")
            xs, cells = [x for x, _ in lv], [c for _, c in lv]
            draw_series(ax, xs, cells, label, color, marker, "throughput_rps")
            draw_series(ax, xs, cells, None, color, marker, "busy_window_rps", ls=":")
        ax.set_xscale("log", base=2)
        ticks = [8, 64, 128, 256] if w == "w3b" else [16, 64, 128]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.xaxis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())
        ax.set_xlabel("闭环并发（同时在系统里的请求数）", fontsize=8)
        ax.set_ylabel("每秒完成的请求", fontsize=8)
        ax.set_title(f"（{'a' if w == 'w3b' else 'b'}）{WL_NAME[w]}：实线 = 登记的容量，虚线 = 满额时段吞吐（描述性）", fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("图 3　闭环吞吐（A 段，不打点）", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_open(plt, summary: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    from matplotlib.ticker import FuncFormatter

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6))
    for row, w in enumerate(("w3b", "w3l")):
        for col, (field, title) in enumerate((("e2e_p50_s", "P50"), ("e2e_p95_s", "P95"), ("e2e_p99_s", "P99"))):
            ax = axes[row][col]
            for arm, _, label, color, marker in ARMS:
                lv = levels(summary, arm, w, "open_loop", "A")
                xs, cells = [x for x, _ in lv], [c for _, c in lv]
                hollow = [any(s is False for s in c["steady_runs"]) for c in cells]
                draw_series(ax, xs, cells, label, color, marker, field, hollow=hollow)
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if f"{v:g}"[0] in "12357" else ""))
            ax.set_xlabel("开环到达速率（请求/秒）", fontsize=8)
            ax.set_ylabel("端到端延迟（秒，对数）", fontsize=8)
            ax.set_title(f"{WL_NAME[w]} {title}", fontsize=9, loc="left")
            ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("图 4　开环延迟（每个运行内算百分位再汇总；失败和超时按无穷大计入；空心点 = 至少一个区组被斜率规则判为非稳态）",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_queue(plt, summary: dict[str, Any], path: Path, rates: tuple[float, ...]) -> None:  # noqa: ANN001
    fig, axes = plt.subplots(1, len(rates), figsize=(5 * len(rates), 3.8), squeeze=False)
    for ax, rate in zip(axes[0], rates):
        for arm, _, label, color, _m in ARMS:
            for r in summary["runs"]:
                if r["arm"] != arm:
                    continue
                for lv in r["levels"]:
                    if lv["workload"] == "w3b" and lv["kind"] == "open_loop" and lv.get("rate_rps") == rate and lv.get("inflight_series"):
                        s = np.array(lv["inflight_series"])
                        ax.plot(s[:, 0], s[:, 1], color=color, lw=0.9, alpha=0.8, label=label if r["block"] == min(
                            x["block"] for x in summary["runs"] if x["arm"] == arm) else None)
        ax.axvline(60, color="#999999", lw=0.8, ls="--")
        ax.text(61, ax.get_ylim()[1] * 0.92, "停止发送", fontsize=7, color="#666666")
        ax.set_xlabel("开始发送后的秒数", fontsize=8)
        ax.set_ylabel("已发出、未完成的请求数", fontsize=8)
        ax.set_title(f"W3b，到达速率 {rate:g} 请求/秒", fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("图 5　排队随时间的变化（每条线是一个运行；线在 60 秒前一直上升 = 处理不过来）", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_audio(plt, summary: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, w in zip(axes, ("w3b", "w3l")):
        for arm, _, label, color, marker in ARMS:
            lv = levels(summary, arm, w, "open_loop", "A")
            draw_series(ax, [x for x, _ in lv], [c for _, c in lv], label, color, marker, "audio_s_per_s")
        ax.set_xlabel("开环到达速率（请求/秒）", fontsize=8)
        ax.set_ylabel("每秒完成的语音秒数", fontsize=8)
        ax.set_title(f"{WL_NAME[w]}", fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("图 6　每秒真正完成了多少秒语音（开环）", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_gpu(plt, summary: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    roles = ["thinker", "talker", "code2wav", "talker+code2wav"]
    width = 0.2
    for ax, field, ylabel in ((axes[0], "busy", "nvidia-smi 忙碌比例"), (axes[1], "mem_max_mib", "显存峰值（GiB）")):
        for i, (arm, _, _label, _c, _m) in enumerate(ARMS):
            for j, role in enumerate(roles):
                vals = [get(lv, f"gpu.{role}.{field}") for r in summary["runs"] if r["arm"] == arm for lv in r["levels"]
                        if lv["workload"] == "w3b" and lv["kind"] == "closed_loop" and lv["mode"] == "A" and lv.get("concurrency") == 256]
                vals = [v / 1024 if field == "mem_max_mib" else v for v in vals if v is not None]
                if not vals:
                    continue
                x = i + (j - 1.5) * width
                ax.bar(x, np.mean(vals), width * 0.9, color=ROLE_COLORS[role], edgecolor="#444444", lw=0.5,
                       label=role if i == 0 or (role == "talker+code2wav" and arm == 1) else None)
                ax.scatter([x] * len(vals), vals, s=8, color="black", zorder=3)
        ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels([f"方案 {a[0]}" for a in ARMS], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    axes[0].legend(seen.values(), seen.keys(), fontsize=7, frameon=False, loc="lower right")
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.suptitle("图 7　W3b 闭环并发 256 时各类卡的忙碌比例和显存（忙碌比例只表示采样时有 kernel 在跑，不是算力利用率）",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_workloads(plt, summary: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.38
    for j, (w, color) in enumerate((("w3b", "#0072B2"), ("w3l", "#D55E00"))):
        base = [r["workload"][w]["capacity_rps"] for r in summary["runs"] if r["arm"] == 1 and w in r["workload"]]
        b = np.mean(base) if base else None
        for i, (arm, *_r) in enumerate(ARMS):
            v = [r["workload"][w]["capacity_rps"] / b for r in summary["runs"] if r["arm"] == arm and w in r["workload"] and b]
            if not v:
                continue
            ax.bar(i + (j - 0.5) * width, np.mean(v), width * 0.9, color=color, alpha=0.6, label=WL_NAME[w] if i == 0 else None)
            ax.scatter([i + (j - 0.5) * width] * len(v), v, s=10, color="black", zorder=3)
    ax.axhline(1.0, color="#999999", lw=0.8, ls="--")
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([f"方案 {a[0]}" for a in ARMS], fontsize=8)
    ax.set_ylabel("容量 ÷ 方案 1 的容量（登记的指标）", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("图 8　回答变长后，各种分法的相对容量有没有变", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_model(plt, summary: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in ((axes[0], "H9d", "W3b"), (axes[1], "H9d_w3l", "W3L")):
        preds = summary["verdicts"].get(key, {}).get("predictions", {})
        hi = 0.0
        for arm, _, label, color, marker in ARMS:
            p = preds.get(str(arm)) or preds.get(arm)
            if not p or p.get("measured_rps") is None:
                continue
            hi = max(hi, p["measured_rps"], p.get("predicted_rps") or 0)
            if p.get("predicted_rps") is None:
                ax.scatter([p["measured_rps"]], [p["measured_rps"]], marker="x", color=color, s=40, label=f"{label}（无法预测）")
            else:
                ax.scatter([p["measured_rps"]], [p["predicted_rps"]], marker=marker, color=color, s=40,
                           label=f"{label}：误差 {p['error'] * 100:+.0f}%")
        lim = [0, hi * 1.15 if hi else 1]
        ax.plot(lim, lim, color="#999999", lw=0.8)
        ax.fill_between(lim, [x * 0.9 for x in lim], [x * 1.1 for x in lim], color="#dddddd", alpha=0.5, label="±10%")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("实测容量（请求/秒，登记的指标）", fontsize=8)
        ax.set_ylabel("留一法预测的容量（请求/秒）", fontsize=8)
        ax.set_title(f"{title}（判定：{summary['verdicts'].get(key, {}).get('verdict')}）", fontsize=9, loc="left")
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
        ax.tick_params(labelsize=7)
    fig.suptitle("图 9　用“瓶颈阶段每张卡的处理能力”预测整条流水线的容量（预测某种分法时只用其他分法的运行）",
                 fontsize=10, x=0.01, ha="left")
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
    fig_stage_capacity(plt, summary, figs / "fig2_stage_capacity.png")
    fig_closed(plt, summary, figs / "fig3_closed_throughput.png")
    fig_open(plt, summary, figs / "fig4_open_latency.png")
    fig_queue(plt, summary, figs / "fig5_queue_depth.png", (18.0, 23.5))
    fig_audio(plt, summary, figs / "fig6_audio_rate.png")
    fig_gpu(plt, summary, figs / "fig7_gpu.png")
    fig_workloads(plt, summary, figs / "fig8_workloads.png")
    fig_model(plt, summary, figs / "fig9_model.png")
    print(sorted(p.name for p in figs.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
