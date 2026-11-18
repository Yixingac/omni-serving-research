#!/usr/bin/env python3
"""实验 13 的图：从 exp13_report.py 写出的 results/summary.json 画。

    python scripts/analysis/exp13_figures.py --out research/01_性能瓶颈/实验13_GPU间连接快慢

Every configuration ran once.
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

LINK = {"NV4": ("NVLink（NV4）", "#0072B2"), "PIX": ("同一 PCIe 交换机（PIX）", "#009E73"),
        "NODE": ("同一 CPU 下跨 PCIe 主桥（NODE）", "#E69F00"), "SYS": ("跨 CPU 插槽（SYS）", "#D55E00")}
ORDER = ["NV4", "PIX", "NODE", "SYS"]


def finish(fig, path: Path, title: str) -> None:  # noqa: ANN001
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


def fig1(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    from matplotlib.colors import LogNorm

    topo = s["topology"]
    m = np.full((8, 8), np.nan)
    lab = [["" for _ in range(8)] for _ in range(8)]
    for p in topo["pairs"]:
        a, b = p["pair"]
        m[a, b], m[b, a] = p["GBps"][0], p["GBps"][1]
        lab[a][b] = lab[b][a] = p["link"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [1.1, 1]})
    im = ax.imshow(m, cmap="viridis", norm=LogNorm(vmin=1, vmax=60))
    for i in range(8):
        for j in range(8):
            if i != j and not np.isnan(m[i, j]):
                ax.text(j, i, f"{lab[i][j]}\n{m[i, j]:.1f}", ha="center", va="center", fontsize=6.5, color="white" if m[i, j] < 20 else "black")
    ax.set_xticks(range(8), [f"GPU {i}" for i in range(8)], fontsize=7)
    ax.set_yticks(range(8), [f"GPU {i}" for i in range(8)], fontsize=7)
    ax.set_xlabel("拷贝到")
    ax.set_ylabel("从")
    ax.set_title("（a）每对 GPU 之间直接拷贝的带宽（GB/s）和连接类型", fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, label="GB/s（对数色标）")
    bw = topo["bandwidth_by_link"]
    names = [k for k in ORDER if k in bw]
    x = np.arange(len(names))
    bx.bar(x, [bw[k]["median_GBps"] for k in names], color=[LINK[k][1] for k in names])
    for i, k in enumerate(names):
        bx.text(i, bw[k]["median_GBps"] * 1.08, f"{bw[k]['median_GBps']:.1f}（{bw[k]['pairs']} 对）", ha="center", fontsize=8)
    for a in topo["allreduce"]:
        bx.axhline(a["algbw_GBps"], color="#666666", linestyle="--" if a["p2p_disabled"] else ":", linewidth=0.9)
        bx.text(len(names) - 0.5, a["algbw_GBps"] * 1.05,
                f"NCCL all-reduce，GPU {a['devices'][0]}–{a['devices'][1]}，{'关掉点对点、经主机内存' if a['p2p_disabled'] else 'NVLink'}：{a['algbw_GBps']:.1f}",
                fontsize=7, ha="right", color="#444444")
    bx.set_yscale("log")
    bx.set_ylim(1, 120)
    bx.set_xticks(x, [LINK[k][0] for k in names], fontsize=7.5, rotation=10)
    bx.set_ylabel("直接拷贝带宽（GB/s，各对的中位数）")
    bx.set_title("（b）按连接类型：PCIe 的三种连接都只有 2.7 GB/s", fontsize=9, loc="left")
    finish(fig, path, "图 1　compute-03 的 GPU 拓扑和点对点带宽（拓扑测量，2026-11-16）。数据：results/summary.json")
    plt.close(fig)


def _services(s: dict[str, Any], ph: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for r in s["runs"]:
        for name, sv in r["phases"][ph].items():
            group = "第 1 组" if r["experiment_id"].endswith("g1a_links") else "第 2 组"
            out.append((f"GPU {sv['gpus'][0]}→{sv['gpus'][1]}（{group}）", sv))
    return sorted(out, key=lambda t: (ORDER.index(t[1]["link"]), t[0]))


def fig2(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), sharey=True)
    for ax, ph, title in ((axes[0], "w3b_closed", "（a）闭环，并发 16、32（每个服务 48 个请求）"), (axes[1], "w3b_open_abs", "（b）开环，合计 6–15 请求/秒（每个服务 1,320 个请求）")):
        sv = _services(s, ph)
        x = np.arange(len(sv))
        ax.bar(x, [v["hop_thinker_talker_p50_ms"] for _, v in sv], 0.6, color=[LINK[v["link"]][1] for _, v in sv])
        ax.plot(x, [v["hop_thinker_talker_p90_ms"] for _, v in sv], "k_", markersize=14, mew=1.5, label="P90")
        ax.plot(x, [v["hop_talker_code2wav_p50_ms"] for _, v in sv], "o", color="#999999", markersize=4, label="talker 交给 code2wav（同一张卡）的中位数")
        ax.set_xticks(x, [n for n, _ in sv], fontsize=6.5, rotation=25, ha="right")
        ax.set_title(title, fontsize=9, loc="left")
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    axes[0].set_ylabel("thinker 交给 talker 的时间（毫秒，柱：中位数）")
    handles = [plt.Rectangle((0, 0), 1, 1, color=LINK[k][1]) for k in ORDER]
    fig.legend(handles, [LINK[k][0] for k in ORDER], fontsize=7, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    finish(fig, path, "图 2　thinker 交给 talker 要 3.4–4.5 毫秒，和两张卡之间是什么连接没有关系。数据：results/summary.json")
    plt.close(fig)


def fig3(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for name, sv in _services(s, "w3b_open_abs"):
        lv = sorted(sv["levels"].items(), key=lambda kv: float(kv[0]))
        ax.plot([float(k) for k, _ in lv], [v["e2e_p50_s"] for _, v in lv], marker="o", color=LINK[sv["link"]][1],
                linestyle="-" if "第 1 组" in name else "--", label=f"{name}，{LINK[sv['link']][0]}")
    ax.set_xlabel("开环到达速率（4 个服务合计，请求/秒）")
    ax.set_ylabel("端到端时间的中位数（秒）")
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    finish(fig, path, "图 3　8 个服务的端到端时间（实线：第 1 组运行，虚线：第 2 组）：颜色（连接类型）之间没有固定的高低。数据：results/summary.json")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    s = json.loads((out / "results" / "summary.json").read_text())
    plt = setup_matplotlib()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    for fn, name in ((fig1, "fig1_topology"), (fig2, "fig2_handoff"), (fig3, "fig3_e2e")):
        fn(plt, s, out / "figures" / f"{name}.png")
        print(f"figures/{name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
