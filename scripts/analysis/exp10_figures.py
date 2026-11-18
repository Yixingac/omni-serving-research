#!/usr/bin/env python3
"""实验 10 的图：从 exp10_report.py 写出的 results/summary.json 画。

    python scripts/analysis/exp10_figures.py --out research/01_性能瓶颈/实验10_Code2Wav_Batching

Bars and lines are means over the three blocks, small points are the blocks, error bars are t-based 95 % intervals.
G1 = flow CUDA graph on (default), G0 = off. Every figure is drawn from results/summary.json only.
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

CAPS = [1, 2, 4, 8, 16]
WL = {"w3b": "W3b（短，长度相近）", "mix": "MIX（短长各半，长度差异大）"}
GRAPH = {"graph": ("G1：flow CUDA graph 开（默认）", "#0072B2", "o", "-"), "nograph": ("G0：关", "#D55E00", "s", "--")}


def valid_cell(c: dict[str, Any]) -> bool:
    """Same rule as exp10_report.valid: usable, the control file in effect, no server restart in the phase."""
    return c.get("usable") is not False and (c.get("control") or {}).get("ok") is True and not c.get("restart_here")


def cell(s: dict[str, Any], g: str, w: str, setting: str) -> dict[str, Any] | None:
    return s["by_setting"].get(f"{g}|{w}|{setting}")


def draw(ax, xs, cells, field, label, color, marker, ls, scale=1.0):  # noqa: ANN001
    pts = [(x, c[field]) for x, c in zip(xs, cells) if c and c[field]["mean"] is not None]
    if not pts:
        return
    ax.plot([x for x, _ in pts], [v["mean"] * scale for _, v in pts], ls, color=color, marker=marker, ms=5, lw=1.3, label=label)
    for x, v in pts:
        if v["ci95"]:
            ax.errorbar(x, v["mean"] * scale, yerr=[[(v["mean"] - v["ci95"][0]) * scale], [(v["ci95"][1] - v["mean"]) * scale]],
                        color=color, capsize=2, lw=0.7)
        runs = [r * scale for r in v["runs"] if r is not None]
        ax.scatter([x] * len(runs), runs, s=8, color=color, alpha=0.45, linewidths=0)


def by_cap(plt, s: dict[str, Any], field: str, ylabel: str, title: str, path: Path, scale: float = 1.0, graphs=("graph", "nograph")) -> None:  # noqa: ANN001
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, w in zip(axes, WL):
        for g in graphs:
            label, color, marker, ls = GRAPH[g]
            draw(ax, CAPS, [cell(s, g, w, f"fifo{k}") for k in CAPS], field, label, color, marker, ls, scale)
        ax.set_xscale("log", base=2)
        ax.set_xticks(CAPS)
        ax.set_xticklabels([str(k) for k in CAPS])
        ax.xaxis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())
        ax.set_xlabel("code2wav 批大小上限（先进先出）", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(WL[w], fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def padding_vs_gpu(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for g in ("graph", "nograph"):
        label, color, marker, _ = GRAPH[g]
        for r in s["runs"]:
            if r["graph"] != (g == "graph"):
                continue
            ok = [c for c in r["cells"].values() if valid_cell(c) and c.get("padding") is not None and c.get("gpu_ms_per_audio_s") is not None]
            xs, ys = [c["padding"] for c in ok], [c["gpu_ms_per_audio_s"] for c in ok]
            ax.scatter(xs, ys, s=14, color=color, marker=marker, alpha=0.7, label=label if r["block"] == 1 else None)
    ax.set_xlabel("补齐比例（按帧加权）", fontsize=8)
    ax.set_ylabel("每秒有效语音的 GPU 时间（毫秒）", fontsize=8)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)
    ax.set_title("图 4　补齐越多，GPU 时间是不是越多？（每个点是一个运行里的一段：一种负载 × 一种组批设置）", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fifo_vs_length(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fields = [("gpu_ms_per_audio_s", "每秒有效语音的 GPU 时间（毫秒）", 1.0), ("padding", "补齐比例", 1.0),
              ("audio_s_per_s", "每秒完成的语音秒数", 1.0)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    settings = ["fifo8", "length8", "fifo16", "length16"]
    width = 0.2
    for ax, (field, ylabel, scale) in zip(axes, fields):
        for i, (g, w) in enumerate([("graph", "w3b"), ("graph", "mix"), ("nograph", "w3b"), ("nograph", "mix")]):
            for j, st in enumerate(settings):
                c = cell(s, g, w, st)
                if not c or c[field]["mean"] is None:
                    continue
                x = i + (j - 1.5) * width
                color = "#999999" if st.startswith("fifo") else "#009E73"
                ax.bar(x, c[field]["mean"] * scale, width * 0.9, color=color, alpha=0.4 if st.endswith("8") else 0.8,
                       label=st if i == 0 else None)
                runs = [v * scale for v in c[field]["runs"] if v is not None]
                ax.scatter([x] * len(runs), runs, s=8, color="black", zorder=3)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["G1 W3b", "G1 MIX", "G0 W3b", "G0 MIX"], fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("图 5　先进先出（灰）对按长度组批（绿），批上限 8（浅）和 16（深）", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def length_gain(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.35
    for j, (w, color) in enumerate((("w3b", "#0072B2"), ("mix", "#D55E00"))):
        for i, g in enumerate(("graph", "nograph")):
            # paired within a block by the report (exp10_report.contrasts); saving = minus the relative change
            k = (s.get("contrasts") or {}).get(f"{g}|{w}|length16_vs_fifo16|gpu_ms_per_audio_s") or {}
            gains = [-x for x in k.get("runs", []) if x is not None]
            if not gains:
                continue
            x = i + (j - 0.5) * width
            ax.bar(x, np.mean(gains), width * 0.9, color=color, alpha=0.6, label=WL[w] if i == 0 else None)
            ax.scatter([x] * len(gains), gains, s=10, color="black", zorder=3)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xticks(range(2))
    ax.set_xticklabels(["G1（graph 开）", "G0（graph 关）"], fontsize=8)
    ax.set_ylabel("按长度组批省下的 GPU 时间（相对先进先出，批上限 16）", fontsize=8)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.0%}")
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)
    ax.set_title("图 8　长度分布不同，按长度组批的收益是否不同（点 = 区组）", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def long_short_wait(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, stat in zip(axes, ("p95", "max")):
        width = 0.35
        for i, st in enumerate(("fifo16", "length16")):
            for j, (cls, color) in enumerate((("short", "#56B4E9"), ("long", "#D55E00"))):
                vals = [((r["cells"].get(f"mix|{st}") or {}).get("wait") or {}).get(cls) for r in s["runs"]
                        if r["graph"] and valid_cell(r["cells"].get(f"mix|{st}") or {})]
                vals = [v[stat] for v in vals if v]
                if not vals:
                    continue
                x = i + (j - 0.5) * width
                ax.bar(x, np.mean(vals), width * 0.9, color=color, alpha=0.7, label=("短请求" if cls == "short" else "长请求") if i == 0 else None)
                ax.scatter([x] * len(vals), vals, s=10, color="black", zorder=3)
        ax.set_xticks(range(2))
        ax.set_xticklabels(["先进先出，上限 16", "按长度，上限 16"], fontsize=8)
        ax.set_ylabel(f"code2wav 收件队列等待的 {stat.upper()}（毫秒）", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("图 9　MIX、graph 开：按长度组批让长请求或短请求多等了吗？（按语音长度的中位数分成两类）", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    s = json.loads((out / "results" / "summary.json").read_text())
    plt = setup_matplotlib()
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    by_cap(plt, s, "audio_s_per_s", "每秒完成的语音秒数", "图 1　批大小上限和吞吐（code2wav 是瓶颈，所以这就是它的处理能力）", figs / "fig1_throughput.png")
    by_cap(plt, s, "e2e_p50_s", "端到端 P50（秒）", "图 2　批大小上限和延迟（闭环：同时在系统里的请求数固定）", figs / "fig2_latency.png")
    by_cap(plt, s, "gpu_ms_per_audio_s", "每秒有效语音的 GPU 时间（毫秒）",
           "图 3　批大小上限和 GPU 效率（flow + HiFT 的 CUDA event 跨度之和 ÷ 语音秒数；越低越省）", figs / "fig3_gpu_per_audio.png")
    padding_vs_gpu(plt, s, figs / "fig4_padding.png")
    fifo_vs_length(plt, s, figs / "fig5_fifo_vs_length.png")
    by_cap(plt, s, "wait_long_p95", "长请求在收件队列里等待的 P95（毫秒）", "图 6　批大小上限和排队（MIX 只有长请求那一类时更明显）",
           figs / "fig6_queue_wait.png")
    by_cap(plt, s, "graph_share", "走 CUDA graph 的批的比例", "图 7　CUDA graph 命中率（批大于 10 或太长的批没有录好的形状）",
           figs / "fig7_graph_hits.png", graphs=("graph",))
    length_gain(plt, s, figs / "fig8_length_gain.png")
    long_short_wait(plt, s, figs / "fig9_long_short_wait.png")
    print(sorted(p.name for p in figs.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
