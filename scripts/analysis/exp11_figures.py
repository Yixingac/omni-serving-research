#!/usr/bin/env python3
"""实验 11 的图：从 exp11_report.py 写出的 results/performance.json 和 exp11_check.py 写出的 results/check.json 画。

    python scripts/analysis/exp11_figures.py --out research/01_性能瓶颈/实验11_Thinker异步优化

Groups A (synchronous), B (lookahead + snapshot) and C (lookahead + snapshot + side-stream copy) are separate server
starts; inside each run the "unprofiled", "events" and "trace" phases (A/B/C in the data) measure the same setting in
three ways. Bars are means over the blocks, points are the blocks, error bars are t-based 95 % intervals. The schematic
timelines (figure 0) come from exp09_11_diagrams.py.
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

from exp01_gpu_timeline import setup_matplotlib  # noqa: E402

ARMS = {"A": ("A：同步执行", "#666666"), "B": ("B：提前发射 + 快照", "#0072B2"), "C": ("C：B + 旁路复制", "#009E73")}
LEVELS = [8, 32, 64]
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
SEGMENTS = [("send_to_admission", "发出→接纳"), ("admission_to_thinker", "接纳→thinker"), ("thinker_in_to_queue", "thinker→进队列"),
            ("queue_wait", "调度队列等待"), ("prefill", "prefill"), ("prefill_to_first_emit", "prefill→产出"),
            ("first_emit_to_thinker_chunk", "产出→thinker 发出"), ("thinker_chunk_to_decode_chunk", "thinker→decode"),
            ("decode_chunk_to_client", "decode→客户端")]


def per_block(s: dict[str, Any], arm: str, w: str, c: int, mode: str, key: str, sub: str | None = None) -> list[float | None]:
    """One value per registered block (1, 2, 3), None where the run or value is missing, so lists of two arms pair up."""
    out = []
    for b in (1, 2, 3):
        r = next((x for x in s["runs"] if x["arm"] == arm and x["block"] == b), None)
        if r is None:
            out.append(None)
            continue
        row = ((r["levels"].get(w) or {}).get(str(c)) or {}).get(mode) or {}
        v = None if row.get("usable") is False else row.get(key)
        if sub is not None:
            v = (v or {}).get(sub)
        out.append(float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None)
    return out


def mean_ci(vals: list[float | None]) -> tuple[float | None, float | None]:
    v = [x for x in vals if x is not None]
    if not v:
        return None, None
    half = T975.get(len(v) - 1, 2.0) * float(np.std(v, ddof=1)) / math.sqrt(len(v)) if len(v) >= 2 else None
    return float(np.mean(v)), half


def bars(ax, s, w, mode, key, scale=1.0, levels=LEVELS, arms="ABC"):  # noqa: ANN001
    width = 0.8 / len(arms)
    for j, arm in enumerate(arms):
        label, color = ARMS[arm]
        for i, c in enumerate(levels):
            vals = [None if v is None else v * scale for v in per_block(s, arm, w, c, mode, key)]
            m, half = mean_ci(vals)
            if m is None:
                continue
            x = i + (j - (len(arms) - 1) / 2) * width
            ax.bar(x, m, width * 0.9, color=color, alpha=0.75, label=label if i == 0 else None,
                   yerr=half, capsize=2, error_kw={"lw": 0.7})
            pts = [v for v in vals if v is not None]
            ax.scatter([x] * len(pts), pts, s=9, color="black", zorder=3)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f"并发 {c}" for c in levels], fontsize=8)
    ax.tick_params(labelsize=7)


def save(fig, path: Path) -> None:  # noqa: ANN001
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def fig_ttft_split(plt, s, path):  # noqa: ANN001
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    colors = plt.get_cmap("tab10")
    for ax, c in zip(axes, LEVELS):
        for j, arm in enumerate("ABC"):
            bottom = 0.0
            for k, (seg, name) in enumerate(SEGMENTS):
                m, _ = mean_ci(per_block(s, arm, "w3b_ns", c, "B", "ttft_segments_mean_ms", seg))
                if m is None:
                    continue
                ax.bar(j, m, 0.6, bottom=bottom, color=colors(k), label=name if (c == LEVELS[0] and j == 0) else None)
                bottom += m
            total, half = mean_ci(per_block(s, arm, "w3b_ns", c, "B", "ttft_mean_ms"))
            if total is not None:
                ax.text(j, bottom + 1, f"{total:.0f}", ha="center", fontsize=7)
        ax.set_xticks(range(3))
        ax.set_xticklabels([ARMS[a][0] for a in "ABC"], fontsize=7)
        ax.set_title(f"W3b 并发 {c}", fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("TTFT 均值（毫秒），按 9 段堆叠", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle("图 1　TTFT 多出来的时间花在哪一段（事件段；柱顶数字是 TTFT 均值，各段均值之和等于它）", fontsize=10, x=0.01, ha="left")
    save(fig, path)


def fig_gaps(plt, s, path):  # noqa: ANN001
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, q in zip(axes, ("p50", "p95", "p99")):
        bars(ax, s, "w3b_ns", "C", f"gap_{q}_ms")
        ax.set_ylabel(f"每步之前空档的 {q.upper()}（毫秒）", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("图 2　thinker 每一步之前 GPU 空了多久（trace 段，按调度步对齐；两个 thinker 进程的空档合在一起算分位数）", fontsize=10, x=0.01, ha="left")
    save(fig, path)


def fig_idle(plt, s, path):  # noqa: ANN001
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, label in ((axes[0], "between_idle_share", "步与步之间的空档之和 ÷ 时间窗（H11c 用这个）"),
                           (axes[1], "total_idle_share", "GPU 完全没有工作的时间 ÷ 时间窗（两个都要看）")):
        bars(ax, s, "w3b_ns", "C", key, scale=100)
        ax.set_ylabel(label + "（%）", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("图 3　按时间加权的空档比例（trace 段，每次 3 秒）", fontsize=10, x=0.01, ha="left")
    save(fig, path)


def fig_blocking(plt, s, path):  # noqa: ANN001
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bars(axes[0], s, "w3b_ns", "C", "large_copies_per_1k_steps")
    axes[0].set_ylabel("每 1000 步里 > 1 KB、阻塞 > 1 毫秒的拷贝数", fontsize=8)
    width = 0.8 / 3
    for j, arm in enumerate("ABC"):
        for i, c in enumerate(LEVELS):
            n = per_block(s, arm, "w3b_ns", c, "C", "large_copies")
            k = per_block(s, arm, "w3b_ns", c, "C", "large_inside")
            share = [b / a if a else None for a, b in zip(n, k) if a is not None and b is not None]
            m, _ = mean_ci(share)
            if m is None:
                continue
            x = i + (j - 1) * width
            axes[1].bar(x, m * 100, width * 0.9, color=ARMS[arm][1], alpha=0.75)
            axes[1].scatter([x] * len([v for v in share if v is not None]), [v * 100 for v in share if v is not None], s=9, color="black", zorder=3)
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels([f"并发 {c}" for c in LEVELS], fontsize=8)
    axes[1].set_ylabel("这些拷贝里落在“请求结束拷贝”标注内的比例（%）", fontsize=8)
    axes[1].tick_params(labelsize=7)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("图 4　请求结束时的大块阻塞拷贝：有多少、是不是都来自 on_request_finished（trace 段）", fontsize=10, x=0.01, ha="left")
    save(fig, path)


def fig_e2e(plt, s, path):  # noqa: ANN001
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, q in zip(axes, ("e2e_p50", "e2e_p95", "e2e_p99")):
        bars(ax, s, "w3b_ns", "A", q)
        ax.set_ylabel(f"端到端 {q[4:].upper()}（秒，失败按无穷大）", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("图 5　端到端延迟（不打点段，W3b）", fontsize=10, x=0.01, ha="left")
    save(fig, path)


def fig_throughput_ttft(plt, s, path):  # noqa: ANN001
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    bars(axes[0], s, "w3b_ns", "A", "throughput_rps")
    axes[0].set_ylabel("每秒完成的请求", fontsize=8)
    bars(axes[1], s, "w3b_ns", "A", "ttft_p50_ms")
    axes[1].set_ylabel("客户端 TTFT P50（毫秒）", fontsize=8)
    bars(axes[2], s, "w3b_ns", "B", "ms_per_token")
    axes[2].set_ylabel("thinker 每个 token 的时间（毫秒，事件段）", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("图 6　吞吐、TTFT 和每个 token 的时间（W3b）", fontsize=10, x=0.01, ha="left")
    save(fig, path)


def fig_long(plt, s, path):  # noqa: ANN001
    wl = [("w3b_ns", "W3b（短）"), ("w3l_ns", "W3L（约 100 词）"), ("w3xl_ns", "W3XL（约 180 词）")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    width = 0.35
    for j, arm in enumerate("BC"):
        for i, (w, _) in enumerate(wl):
            a = per_block(s, "A", w, 32, "A", "e2e_p50")
            x_ = per_block(s, arm, w, 32, "A", "e2e_p50")
            gain = [1 - b / a_ for a_, b in zip(a, x_) if a_ and b is not None]  # paired by block
            ma, mx = per_block(s, "A", w, 32, "B", "ms_per_token"), per_block(s, arm, w, 32, "B", "ms_per_token")
            tok = per_block(s, arm, w, 32, "B", "tokens_mean")
            saving = [(p - q) * t for p, q, t in zip(ma, mx, tok) if None not in (p, q, t)]
            x = i + (j - 0.5) * width
            for ax, vals, scale in ((axes[0], gain, 100), (axes[1], saving, 1)):
                m, half = mean_ci(vals)
                if m is None:
                    continue
                ax.bar(x, m * scale, width * 0.9, color=ARMS[arm][1], alpha=0.75, label=ARMS[arm][0] if i == 0 else None,
                       yerr=None if half is None else half * scale, capsize=2, error_kw={"lw": 0.7})
                ax.scatter([x] * len(vals), [v * scale for v in vals], s=9, color="black", zorder=3)
    for ax, label in ((axes[0], "端到端 P50 相对 A 的改善（%）"), (axes[1], "每个请求在 thinker 里省下的时间（毫秒）")):
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_xticks(range(3))
        ax.set_xticklabels([n for _, n in wl], fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("图 7　回答越长，收益越大吗？（并发 32；省下的时间 = 每个 token 省下的时间 × token 数）", fontsize=10, x=0.01, ha="left")
    save(fig, path)


def fig_memory(plt, s, path):  # noqa: ANN001
    fig, ax = plt.subplots(figsize=(8, 4))
    bars(ax, s, "w3b_ns", "A", "thinker_mem_max_mib", scale=1 / 1024)
    ax.set_ylabel("thinker 两张卡的最大显存（GiB）", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("图 8　显存：快照和旁路复制多占了多少（nvidia-smi，每 0.2 秒采样）", fontsize=10, loc="left")
    save(fig, path)


def fig_correctness(plt, chk: dict[str, Any] | None, path: Path) -> None:  # noqa: ANN001
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if chk:
        g = chk["gate"]
        e = g.get("c1_exact") or {}
        names = ["pairs", "text", "hidden", "codec"]
        axes[0].bar(range(4), [e.get(n) or 0 for n in names], color=["#999999", "#0072B2", "#009E73", "#D55E00"])
        axes[0].set_xticks(range(4))
        axes[0].set_xticklabels(["比较的请求", "文字相同", "隐藏状态哈希相同", "语音编码哈希相同"], fontsize=7)
        axes[0].set_ylabel("请求数", fontsize=8)
        axes[0].set_title("（a）并发 1：C 和 A 逐个请求比较", fontsize=9, loc="left")
        phases = [p for p, row in g["phases"].items() if row.get("C_vs_A")]
        for j, (key, color, label) in enumerate((("C_vs_A", "#009E73", "C 对 A"), ("A2_vs_A", "#666666", "A2 对 A（同样设置重跑一次）"))):
            vals = [((g["phases"][p].get(key) or {}).get("share_over") or 0) * 100 for p in phases]
            axes[1].bar(np.arange(len(phases)) + (j - 0.5) * 0.35, vals, 0.32, color=color, label=label)
        axes[1].set_xticks(range(len(phases)))
        axes[1].set_xticklabels([p.replace("chk_", "") for p in phases], fontsize=7, rotation=20)
        axes[1].set_ylabel("指纹相对偏差 > 0.05 的请求比例（%）", fontsize=8)
        axes[1].set_title("（b）输出相同的请求里，隐藏状态指纹偏差超限的比例", fontsize=9, loc="left")
        axes[1].legend(fontsize=7, frameon=False)
        verdict = "通过" if g.get("pass") else "未通过"
        fig.suptitle(f"图 9　正确性检查（{verdict}；并发 > 1 时批的组成会变，A 自己重跑也会有数值差异，所以拿 A2 当参照）",
                     fontsize=10, x=0.01, ha="left")
    for ax in axes:
        ax.tick_params(labelsize=7)
    save(fig, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    plt = setup_matplotlib()
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    chk_path = out / "results" / "check.json"
    if chk_path.exists():
        fig_correctness(plt, json.loads(chk_path.read_text()), figs / "fig9_correctness.png")
    perf = out / "results" / "performance.json"
    if perf.exists():
        s = json.loads(perf.read_text())
        fig_ttft_split(plt, s, figs / "fig1_ttft_split.png")
        fig_gaps(plt, s, figs / "fig2_gaps.png")
        fig_idle(plt, s, figs / "fig3_idle_share.png")
        fig_blocking(plt, s, figs / "fig4_blocking_copies.png")
        fig_e2e(plt, s, figs / "fig5_e2e.png")
        fig_throughput_ttft(plt, s, figs / "fig6_throughput_ttft.png")
        fig_long(plt, s, figs / "fig7_long_answers.png")
        fig_memory(plt, s, figs / "fig8_memory.png")
    print(sorted(p.name for p in figs.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
