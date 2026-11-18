#!/usr/bin/env python3
"""实验 12 的图：从 exp12_report.py 写出的 results/summary.json 画。

    python scripts/analysis/exp12_figures.py --out research/01_性能瓶颈/实验12_回放能不能代替真实负载

Every replay ran once. The four first 4x replays ran out of client file descriptors and are drawn only in figure 5.
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

NAME = {"dep_orig_1x": "依赖感知、原文", "req_orig_1x": "请求级、原文", "dep_orig_copy4_1x": "依赖感知、原文、512 个会话各复制 4 份",
        "dep_prefix_1x": "依赖感知、保留前缀的替代", "dep_sur_1x": "依赖感知、完全替代"}
COLOR = {"实时运行": "#000000", "dep_orig_1x": "#0072B2", "req_orig_1x": "#56B4E9", "dep_orig_copy4_1x": "#E69F00",
         "dep_prefix_1x": "#009E73", "dep_sur_1x": "#D55E00"}


def get(s: dict[str, Any], label: str) -> dict[str, Any]:
    return s["live"] if label == "实时运行" else next(r for r in s["replays"] if r["label"] == label)


def finish(fig, path: Path, title: str) -> None:  # noqa: ANN001
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


def fig1(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    labels = ["req_orig_1x", "dep_orig_1x", "dep_orig_copy4_1x", "dep_prefix_1x", "dep_sur_1x"]
    metrics = [("requests_per_s", "吞吐"), ("e2e_p50_s", "单次请求中位数"), ("e2e_p90_s", "单次请求 P90"), ("ttft_p50_s", "TTFT 中位数"),
               ("session_p50_s", "会话中位数")]
    x = np.arange(len(metrics))
    w = 0.16
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.axhspan(0.9, 1.1, color="#DDDDDD", alpha=0.7, label="和实时运行相差不到 10%")
    ax.axhline(1.0, color="black", linewidth=0.7)
    for j, lab in enumerate(labels):
        r = get(s, lab)
        ax.bar(x + (j - 2) * w, [r[m] / s["live"][m] for m, _ in metrics], w * 0.95, color=COLOR[lab], label=NAME[lab])
    ax.set_xticks(x, [z for _, z in metrics])
    ax.set_ylabel("回放 ÷ 实时运行")
    ax.set_yscale("log")
    from matplotlib.ticker import NullFormatter

    ax.set_yticks([0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0], ["0.5", "0.7", "1.0", "1.5", "2.0", "3.0", "5.0", "8.0"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    finish(fig, path, "图 1　1 倍速回放和被回放的实时运行比（8 个完整副本、文本 Agent、256 个会话的 trace；每个回放跑 1 次）。数据：results/summary.json")
    plt.close(fig)


def fig2(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, labels, title in ((a, ["实时运行", "req_orig_1x", "dep_orig_1x"], "（a）两种回放方式"),
                              (b, ["dep_orig_1x", "dep_orig_copy4_1x", "dep_prefix_1x", "dep_sur_1x"], "（b）换内容、复制会话（都是依赖感知）")):
        for lab in labels:
            r = get(s, lab)
            ax.plot(np.arange(len(r["series"])), r["series"], color=COLOR[lab], linewidth=1.3,
                    label=f"{'实时运行' if lab == '实时运行' else NAME[lab]}（最多 {r['peak_in_flight']} 个）")
        ax.axhline(256, color="#999999", linestyle=":", linewidth=0.8)
        ax.text(0.5, 262, "实时运行最多 256 个会话", fontsize=7, color="#666666")
        ax.set_xlabel("从第一个请求发出起的秒数")
        ax.set_title(title, fontsize=9, loc="left")
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    a.set_ylabel("同时在服务里的模型请求数（每秒平均）")
    finish(fig, path, "图 2　同时在服务里的请求数随时间的变化：依赖感知回放保留了会话结构，但没有保留“最多 256 个会话”这个上限。数据：results/summary.json")
    plt.close(fig)


def fig3(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    labels = ["dep_orig_1x", "dep_prefix_1x", "dep_sur_1x"]
    rs = [get(s, lab) for lab in labels]
    names = ["原文", "保留前缀的替代", "完全替代"]
    colors = [COLOR[lab] for lab in labels]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    axes[0].bar(x, [r["prefix_hit"] * 100 for r in rs], color=colors)
    axes[0].set_ylabel("前缀缓存命中（%）")
    axes[0].set_title("（a）命中率", fontsize=9, loc="left")
    axes[1].bar(x - 0.18, [r["prompt_tokens_p50"] for r in rs], 0.34, color="#999999", label="输入 token 中位数")
    ax1b = axes[1].twinx()
    ax1b.bar(x + 0.18, [r["output_tokens_mean"] for r in rs], 0.34, color="#333333", label="输出 token 均值")
    axes[1].set_title("（b）输入和输出的长度", fontsize=9, loc="left")
    axes[1].set_ylabel("输入 token")
    ax1b.set_ylabel("输出 token")
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    axes[1].legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    stop = [r["finish"].get("stop", 0) / r["ok"] * 100 for r in rs]
    axes[2].bar(x, stop, color=colors)
    axes[2].set_ylabel("遇到结束符就停的请求（%）")
    axes[2].set_title("（c）遇到结束符就停的请求：原文本来如此；\n两种替代本该按录下的长度输出", fontsize=9, loc="left")
    axes[3].bar(x - 0.18, [r["e2e_p50_s"] for r in rs], 0.34, color=colors, label="中位数")
    axes[3].bar(x + 0.18, [r["e2e_p90_s"] for r in rs], 0.34, color=colors, alpha=0.45, label="P90")
    axes[3].axhline(s["live"]["e2e_p50_s"], color="black", linestyle="--", linewidth=0.8)
    axes[3].text(-0.45, s["live"]["e2e_p50_s"] + 0.08, "实时运行的中位数", fontsize=7)
    axes[3].set_ylabel("单次请求耗时（秒）")
    axes[3].set_title("（d）单次请求多久", fontsize=9, loc="left")
    axes[3].legend(fontsize=7, frameon=False)
    for ax in axes:
        ax.set_xticks(x, names, fontsize=8)
    finish(fig, path, "图 3　请求内容换掉以后（都是依赖感知、1 倍速）：命中率、长度和延迟一起变。数据：results/summary.json")
    plt.close(fig)


def fig4(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    series = [("依赖感知、原文", "#0072B2", ["dep_orig_1x", "dep_orig_2x", "dep_orig_4x_r2"]),
              ("请求级、原文", "#56B4E9", ["req_orig_1x", "req_orig_4x_r2"]),
              ("依赖感知、保留前缀的替代", "#009E73", ["dep_prefix_1x", "dep_prefix_4x_r2"])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    live = s["live"]["requests_per_s"]
    for name, color, labels in series:
        rs = [get(s, lab) for lab in labels]
        sp = [r["speed"] for r in rs]
        axes[0].plot(sp, [r["requests_per_s"] for r in rs], marker="o", color=color, label=name)
        axes[1].plot(sp, [r["e2e_p50_s"] for r in rs], marker="o", color=color, label=f"{name}：中位数")
        axes[1].plot(sp, [r["e2e_p90_s"] for r in rs], marker=".", linestyle=":", color=color, label=f"{name}：P90")
        axes[2].plot(sp, [r["ttft_p50_s"] for r in rs], marker="o", color=color, label=name)
    axes[0].plot([1, 2, 4], [live, 2 * live, 4 * live], color="#999999", linestyle="--", label="按倍速放大的实时运行吞吐（发出的速率）")
    axes[0].set_ylabel("每秒完成的模型请求")
    axes[0].set_title("（a）吞吐：倍速再高，也到不了发出的速率", fontsize=9, loc="left")
    axes[1].set_ylabel("单次请求耗时（秒）")
    axes[1].set_title("（b）单次请求", fontsize=9, loc="left")
    axes[2].set_ylabel("TTFT 中位数（秒）")
    axes[2].set_title("（c）首个文字", fontsize=9, loc="left")
    for ax in axes:
        ax.set_xticks([1, 2, 4], ["1 倍速", "2 倍速", "4 倍速"])
        ax.legend(fontsize=6.5, frameon=False)
    finish(fig, path, "图 4　加快回放：8 个完整副本对这种文本 Agent 每秒最多完成约 170–290 个请求，取决于回放方式和内容（4 倍速用的是修复句柄上限后的重跑）。数据：results/summary.json")
    plt.close(fig)


def fig5(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    labels = [r["label"] for r in s["replays"] if r["speed"] == 4.0]
    labels.sort(key=lambda lab: (lab.replace("_r2", ""), lab.endswith("_r2")))
    rs = [get(s, lab) for lab in labels]
    x = np.arange(len(rs))
    parts = [("成功", "#009E73", lambda r: r["ok"]), ("客户端文件句柄用完", "#999999", lambda r: r["errors"].get("client fd limit", 0)),
             ("服务端显存不足", "#D55E00", lambda r: r["errors"].get("CUDA out of memory", 0)),
             ("前面的请求失败而跳过", "#E69F00", lambda r: r["skipped"])]
    fig, ax = plt.subplots(figsize=(11, 4.0))
    bottom = np.zeros(len(rs))
    for name, color, fn in parts:
        v = np.array([fn(r) for r in rs], dtype=float)
        ax.bar(x, v, 0.6, bottom=bottom, color=color, label=name)
        bottom += v
    zh = {"dep_orig": "依赖感知、原文", "dep_prefix": "依赖感知、保留前缀的替代", "dep_sur": "依赖感知、完全替代", "req_orig": "请求级、原文"}
    ax.set_xticks(x, [zh[lab.split("_4x")[0]] + ("\n（重跑）" if lab.endswith("_r2") else "\n（第一次）") for lab in labels], fontsize=7.5)
    ax.set_ylabel("模型请求数（共 4128 个）")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    finish(fig, path, "图 5　4 倍速回放：第一次因为客户端句柄不够而作废；修复后重跑，完全替代内容的那次又在服务端显存不足。数据：results/summary.json")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    s = json.loads((out / "results" / "summary.json").read_text())
    plt = setup_matplotlib()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    for fn, name in ((fig1, "fig1_vs_live"), (fig2, "fig2_in_flight"), (fig3, "fig3_content"), (fig4, "fig4_speed"), (fig5, "fig5_4x_failures")):
        fn(plt, s, out / "figures" / f"{name}.png")
        print(f"figures/{name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
