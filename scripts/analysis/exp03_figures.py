#!/usr/bin/env python3
"""实验 3 的图：从 exp03_report.py 写出的 results/summary.json 画。

    python scripts/analysis/exp03_figures.py --out research/01_性能瓶颈/实验03_同时处理多少请求最好

Every configuration ran once. Concurrency 1-8 and 16-256 come from two runs of the same split.
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

COLOR = {"s2x6": "#0072B2", "s1x7": "#56B4E9", "s3x5": "#009E73", "s4x4": "#E69F00", "p4": "#999999", "r8": "#000000"}
CONC = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def finish(fig, path: Path, title: str) -> None:  # noqa: ANN001
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


def fig1(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for k, c in s["curves"].items():
        x = [lv["concurrency"] for lv in c["levels"]]
        axes[0].plot(x, [lv["rps"] for lv in c["levels"]], marker="o", color=COLOR[k], label=c["label"])
        axes[1].plot(x, [lv["e2e_p50_s"] for lv in c["levels"]], marker="o", color=COLOR[k], label=f"{c['label']}：中位数")
        axes[1].plot(x, [lv["e2e_p90_s"] for lv in c["levels"]], marker=".", linestyle=":", color=COLOR[k], alpha=0.7)
        axes[2].plot(x, [lv["slo"] * 100 for lv in c["levels"]], marker="o", color=COLOR[k], label=c["label"])
    axes[0].set_ylabel("吞吐（请求/秒）")
    axes[0].set_title("（a）吞吐：到 64 以前接近翻倍，之后越来越平", fontsize=9, loc="left")
    axes[1].set_ylabel("端到端时间（秒；实线中位数，点线 P90）")
    axes[1].set_title("（b）延迟：到 8 都不变，64 以后快速变长", fontsize=9, loc="left")
    axes[2].axhline(95, color="#CC0000", linestyle="--", linewidth=0.8)
    axes[2].text(1.1, 90, "95%", fontsize=7, color="#CC0000")
    axes[2].set_ylabel("满足 SLO 的请求（%）")
    axes[2].set_title("（c）四种分法都是 64 还合格、128 开始不合格", fontsize=9, loc="left")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(CONC, [str(c) for c in CONC], fontsize=7)
        ax.set_xlabel("闭环并发（同时在系统里的请求数）")
        ax.axvspan(12, 20, color="#EEEEEE", zorder=0)
        ax.legend(fontsize=7, frameon=False)
    finish(fig, path, "图 1　闭环：并发从 1 到 256（灰色竖条左边是小 batch 运行，右边是另一次运行；4 组 1+1 和 1+7 的 16–256 在第一批节点上）。数据：results/summary.json")
    plt.close(fig)


def fig2(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for k, c in s["curves"].items():
        lv = c["levels"]
        ax.plot([x["rps"] for x in lv], [x["e2e_p50_s"] for x in lv], marker="o", color=COLOR[k], label=c["label"])
        for x in lv:
            if int(x["concurrency"]) in (8, 64, 128, 256) and k in ("s2x6", "r8"):
                ax.annotate(f"{int(x['concurrency'])}", (x["rps"], x["e2e_p50_s"]), textcoords="offset points", xytext=(4, 4), fontsize=7, color=COLOR[k])
    ax.set_xlabel("吞吐（请求/秒）")
    ax.set_ylabel("端到端时间的中位数（秒）")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    finish(fig, path, "图 2　吞吐和延迟的取舍：每个点是一个并发（数字标在 2+6 和 8 个完整副本旁边）；曲线拐上去的地方就是“再加请求只会排队”。数据：results/summary.json")
    plt.close(fig)


def fig3(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    parts = [("thinker 排队", "#BBBBBB", lambda st: st["thinker.queue"]), ("thinker 生成", "#0072B2", lambda st: st["thinker.model_path"]),
             ("talker 生成（含排队）", "#009E73", lambda st: (st["talker.queue"] or 0) + (st["talker.model_path"] or 0)),
             ("code2wav（含排队）", "#D55E00", lambda st: st["code2wav.service"])]
    keys = [k for k in ("s2x6", "r8", "p4", "s1x7") if k in s["curves"]]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4.4), sharey=True)
    for ax, k in zip(np.atleast_1d(axes), keys):
        lv = s["curves"][k]["levels"]
        x = np.arange(len(lv))
        bottom = np.zeros(len(lv))
        for name, color, fn in parts:
            v = np.array([(fn(r["stages_ms"]) or 0.0) / 1e3 for r in lv])
            ax.bar(x, v, 0.7, bottom=bottom, color=color, label=name)
            bottom += v
        ax.plot(x, [r["e2e_p50_s"] for r in lv], "k_", markersize=12, mew=2, label="端到端中位数")
        ax.set_xticks(x, [str(int(r["concurrency"])) for r in lv], fontsize=7)
        ax.set_xlabel("闭环并发")
        ax.set_title(s["curves"][k]["label"], fontsize=9, loc="left")
    np.atleast_1d(axes)[0].set_ylabel("秒（各阶段的中位数）")
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False, loc="upper left")
    finish(fig, path, "图 3　负载变高以后多出来的时间在哪个阶段（服务端事件；各阶段中位数叠起来，和端到端中位数不完全相等）。数据：results/summary.json")
    plt.close(fig)


def fig4(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.4))
    for k, o in s["open"].items():
        lv = o["levels"]
        a.plot([x["rate_rps"] for x in lv], [x["slo"] * 100 for x in lv], marker="o", color=COLOR[k],
               label=f"{o['label']}：容量 {o['capacity']:.0f} 请求/秒，那时约 {o['in_flight_at_capacity']:.0f} 个请求在系统里")
        b.plot([x["rate_rps"] for x in lv], [x["e2e_p50_s"] for x in lv], marker="o", color=COLOR[k], label=o["label"])
    a.axhline(95, color="#CC0000", linestyle="--", linewidth=0.8)
    a.set_ylabel("满足 SLO 的请求（%）")
    a.set_title("（a）满足 SLO 的比例和 SLO 容量", fontsize=9, loc="left")
    from matplotlib.ticker import NullFormatter

    b.set_yscale("log")
    b.set_yticks([2, 3, 5, 10, 20, 30], ["2", "3", "5", "10", "20", "30"])
    b.yaxis.set_minor_formatter(NullFormatter())
    b.set_ylabel("端到端时间的中位数（秒，对数）")
    b.set_title("（b）超过容量以后，延迟按十几秒地涨", fontsize=9, loc="left")
    for ax in (a, b):
        ax.set_xlabel("开环到达速率（请求/秒）")
        ax.legend(fontsize=6.5, frameon=False)
    finish(fig, path, "图 4　开环：6 种分法在同一组到达速率下（compute-03，每档 90 秒）。数据：results/summary.json")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    s = json.loads((out / "results" / "summary.json").read_text())
    plt = setup_matplotlib()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    for fn, name in ((fig1, "fig1_closed_loop"), (fig2, "fig2_tradeoff"), (fig3, "fig3_where_time_goes"), (fig4, "fig4_open_loop")):
        fn(plt, s, out / "figures" / f"{name}.png")
        print(f"figures/{name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
