#!/usr/bin/env python3
"""实验 14 的图：从 exp14_report.py 写出的 results/summary.json 画。

    python scripts/analysis/exp14_figures.py --out research/01_性能瓶颈/实验14_视频请求加副本

Every configuration ran once; levels are short (24-192 requests), so throughput is shown with two estimates.
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

VIDEO = {"w4_video_10s": ("10 秒视频", "#0072B2"), "w4_video_20s": ("20 秒视频", "#D55E00")}
REP_COLOR = {1: "#000000", 2: "#56B4E9", 4: "#009E73", 8: "#E69F00"}


def finish(fig, path: Path, title: str) -> None:  # noqa: ANN001
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


def fig1(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.4))
    for ph, (name, color) in VIDEO.items():
        sc = s["scaling"][ph]
        reps = sorted(int(k) for k in sc["best_rps"])
        best = [sc["best_rps"][str(n)] for n in reps]
        little = [sc["best_little_rps"][str(n)] for n in reps]
        a.plot(reps, best, marker="o", color=color, label=f"{name}：首个请求到最后完成")
        a.plot(reps, little, marker="s", mfc="white", linestyle="--", color=color, label=f"{name}：并发 ÷ 平均耗时")
        a.plot(reps, [best[0] * n for n in reps], color=color, linestyle=":", linewidth=0.8, alpha=0.7, label=f"{name}：1 个副本 × 副本数")
        b.plot(reps, [sc["efficiency"][str(n)] for n in reps], marker="o", color=color, label=f"{name}：首个请求到最后完成")
        b.plot(reps, [sc["efficiency_little"][str(n)] for n in reps], marker="s", mfc="white", linestyle="--", color=color, label=f"{name}：并发 ÷ 平均耗时")
    a.set_ylabel("各档里最高的吞吐（请求/秒）")
    a.set_title("（a）吞吐随副本数", fontsize=9, loc="left")
    b.axhline(0.8, color="#CC0000", linestyle="--", linewidth=0.8)
    b.text(1.05, 0.81, "0.8：判定“近似线性”的界限", fontsize=7, color="#CC0000")
    b.set_ylim(0.6, 1.15)
    b.set_ylabel("扩展效率 = 吞吐 ÷（副本数 × 1 个副本的吞吐）")
    b.set_title("（b）扩展效率：两种算法给出 0.78–0.98", fontsize=9, loc="left")
    for ax in (a, b):
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
        ax.set_xlabel("副本数（每个副本 1 张卡）")
        ax.legend(fontsize=6.5, frameon=False)
    finish(fig, path, "图 1　视频请求的吞吐随副本数增长（每种配置 1 次；每档只有 24–192 个请求，所以给出两种估计）。数据：results/summary.json")
    plt.close(fig)


def fig2(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    lv = s["events"]["phases"]["w4_video_10s"]
    concs = sorted(lv, key=int)
    parts = [("预处理（CPU：解码视频、抽帧）", "#D55E00", ["preprocessing.service"]),
             ("图像编码器", "#E69F00", ["image_encoder.service"]),
             ("thinker：排队", "#999999", ["thinker.queue"]),
             ("thinker：prefill 和其余", "#0072B2", ["thinker.service"]),
             ("阶段之间的传递、decode", "#009E73", ["hop.preprocessing->image_encoder", "hop.image_encoder->thinker", "hop.thinker->decode", "decode.service"])]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.2))
    x = np.arange(len(concs))
    bottom = np.zeros(len(concs))
    for name, color, keys in parts:
        v = np.array([sum((lv[c].get(k) or 0.0) for k in keys) for c in concs])
        a.bar(x, v, 0.6, bottom=bottom, color=color, label=name)
        bottom += v
    a.plot(x, [lv[c]["request.admission_to_last_terminal"] for c in concs], "k_", markersize=26, mew=2, label="服务端从接收到完成（中位数）")
    a.set_xticks(x, [f"并发 {c}" for c in concs])
    a.set_ylabel("毫秒（各阶段的中位数）")
    a.set_title("（a）一个 10 秒视频请求的时间花在哪", fontsize=9, loc="left")
    a.legend(fontsize=7, frameon=False, loc="upper left")
    b.plot([int(c) for c in concs], [lv[c]["preprocessing.service"] for c in concs], marker="o", color="#D55E00", label="预处理")
    b.plot([int(c) for c in concs], [lv[c]["thinker.service"] for c in concs], marker="o", color="#0072B2", label="thinker")
    b.plot([int(c) for c in concs], [lv[concs[0]]["preprocessing.service"] * int(c) for c in concs], linestyle=":", color="#D55E00",
           label="预处理 × 并发（一次只做一个时的预期）")
    b.set_xticks([int(c) for c in concs])
    b.set_xlabel("并发（1 个副本）")
    b.set_ylabel("毫秒（中位数）")
    b.set_title("（b）并发加倍，预处理的时间也加倍；thinker 不变", fontsize=9, loc="left")
    b.legend(fontsize=7, frameon=False)
    finish(fig, path, "图 2　1 个副本、10 秒视频的服务端事件（事件补测，每档 24–28 个请求）。数据：results/summary.json")
    plt.close(fig)


def fig3(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=False)
    for ax, ph in zip(axes, VIDEO):
        for r in s["runs"]:
            lvs = sorted(r["phases"][ph].values(), key=lambda lv: lv["per_replica_concurrency"])
            ax.plot([lv["per_replica_concurrency"] for lv in lvs], [lv["e2e_p50_s"] for lv in lvs], marker="o", color=REP_COLOR[r["replicas"]],
                    label=f"{r['replicas']} 个副本")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
        ax.set_xlabel("每个副本平均分到的并发")
        ax.set_ylabel("单次请求耗时的中位数（秒）")
        ax.set_title(f"（{'ab'[list(VIDEO).index(ph)]}）{VIDEO[ph][0]}", fontsize=9, loc="left")
        ax.legend(fontsize=7, frameon=False)
    finish(fig, path, "图 3　每个副本分到的请求一多，耗时就成比例变长；副本数不同，曲线几乎重合。数据：results/summary.json")
    plt.close(fig)


def fig4(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.0))
    reps = [r["replicas"] for r in s["runs"]]
    x = np.arange(len(reps))
    for j, ph in enumerate(VIDEO):
        name, color = VIDEO[ph]
        cores = [max((lv.get("cpu_cores_per_replica") or 0.0) for lv in r["phases"][ph].values()) for r in s["runs"]]
        a.bar(x + (j - 0.5) * 0.36, cores, 0.34, color=color, label=name)
        b.bar(x + (j - 0.5) * 0.36, [r["phase_gpu_busy"][ph] for r in s["runs"]], 0.34, color=color, label=name)
    a.axhline(7, color="#999999", linestyle="--", linewidth=0.8)
    a.text(-0.45, 6.6, "每个副本分到的 CPU：7 核", fontsize=7, color="#666666")
    a.set_ylabel("每个副本用的 CPU（核，各档里的最大值）")
    a.set_title("（a）CPU：每个副本只用 1–1.3 个核", fontsize=9, loc="left")
    b.set_ylim(0, 100)
    b.set_ylabel("GPU 忙碌比例（%，整段平均）")
    b.set_title("（b）GPU：大部分时间空着", fontsize=9, loc="left")
    for ax in (a, b):
        ax.set_xticks(x, [f"{n} 个副本" for n in reps])
        ax.legend(fontsize=7, frameon=False)
    finish(fig, path, "图 4　视频请求用的 CPU 和 GPU：都远没有用满。数据：results/summary.json")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    s = json.loads((out / "results" / "summary.json").read_text())
    plt = setup_matplotlib()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    for fn, name in ((fig1, "fig1_scaling"), (fig2, "fig2_stages"), (fig3, "fig3_latency"), (fig4, "fig4_cpu_gpu")):
        fn(plt, s, out / "figures" / f"{name}.png")
        print(f"figures/{name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
