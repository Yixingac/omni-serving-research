#!/usr/bin/env python3
"""实验 19 的图：从 exp19_report.py 写出的 results/summary.json 画。

    python scripts/analysis/exp19_figures.py --out research/01_性能瓶颈/实验19_负载类型和最佳分工

Every configuration ran once (no repeats), so there are no error bars; the run-to-run difference of the same split
in plan v2 was within 3 %. Every figure is drawn from results/summary.json only.
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

COLOR = {"r8": "#000000", "p4": "#999999", "s1x7": "#56B4E9", "s2x6": "#0072B2", "s3x5": "#009E73", "s4x4": "#E69F00",
         "s5x3": "#CC79A7", "s6x2": "#D55E00"}
ORDER = ["s1x7", "s2x6", "s3x5", "s4x4", "p4", "s5x3", "s6x2", "r8"]  # by thinker GPUs; a pair split next to 4+4
TYPE_ZH = {"W1": "W1 文字聊天", "W2": "W2 语音提问、文字回答", "W3b": "W3b 文字提问、语音回答"}
TYPE_COLOR = {"W1": "#0072B2", "W2": "#009E73", "W3b": "#D55E00"}
THINKER_SLOTS = 64  # running requests per thinker replica (实验 2); used only to mark the figure


def label(layout: str) -> str:
    if layout.startswith("s"):
        t, k = layout[1:].split("x")
        return f"{t}+{k}"
    return "4 组 1+1" if layout == "p4" else "8 个完整副本"


def ordered(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda m: ORDER.index(m["layout"]))


def finish(fig, path: Path, title: str) -> None:  # noqa: ANN001
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


def fig1(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.4))
    for m in ordered(s["mix"]):
        xs = [x["concurrency"] for x in m["closed"]]
        a.plot(xs, [x["rps"] for x in m["closed"]], marker="o", color=COLOR[m["layout"]], label=f"{m['label']}（{m['node']}）")
    a.set_xscale("log", base=2)
    a.set_xticks([16, 32, 64, 128, 256], ["16", "32", "64", "128", "256"])
    a.set_xlabel("闭环并发")
    a.set_ylabel("吞吐（请求/秒）")
    a.set_title("（a）混合负载：吞吐随并发", fontsize=9, loc="left")
    a.legend(fontsize=7, frameon=False)
    w3b: dict[str, float] = {}
    for r in s["w3b"]:
        w3b[r["layout"]] = max(w3b.get(r["layout"], 0.0), r["closed_max_rps"])
    mix = {m["layout"]: m["closed_max_rps"] for m in s["mix"]}
    for name, vals, style in (("W3b 纯语音回答（实验 2 的数据）", w3b, dict(marker="s", mfc="white", linestyle="--")),
                              ("混合负载", mix, dict(marker="o", linestyle="-"))):
        pooled = sorted((int(k[1:].split("x")[0]), v) for k, v in vals.items() if k.startswith("s"))
        b.plot([x for x, _ in pooled], [v for _, v in pooled], color="#333333", label=f"{name}：t+k 合用", **style)
        for k in ("p4", "r8"):
            if k in vals:
                b.plot([4.25 if k == "p4" else 8], [vals[k]], color=COLOR[k], linestyle="none",
                       **{kk: vv for kk, vv in style.items() if kk != "linestyle"})
                b.annotate(label(k), (4.25 if k == "p4" else 8, vals[k]), textcoords="offset points", xytext=(6, -3), fontsize=7)
    b.set_xticks(range(1, 9))
    b.set_xlabel("放 thinker 的 GPU 数（8 个完整副本：每张卡都放）")
    b.set_ylabel("闭环最高吞吐（请求/秒，并发 ≤ 256）")
    b.set_title("（b）最高吞吐出现在哪种分法：纯语音是 2+6，混合负载移到 3+5", fontsize=9, loc="left")
    b.legend(fontsize=7, frameon=False)
    finish(fig, path, "图 1　混合负载下的吞吐（每种分法跑 1 次；最高吞吐都出现在测到的最高并发 256，真实容量可能更高）。数据：results/summary.json")
    plt.close(fig)


def fig2(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    mix = ordered(s["mix"])
    at = [next(x for x in m["closed"] if x["concurrency"] == 256) for m in mix]
    x = np.arange(len(mix))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    w = 0.26
    for j, t in enumerate(("W1", "W2", "W3b")):
        axes[0].bar(x + (j - 1) * w, [lv["by_type"][t]["slo"] * 100 for lv in at], w * 0.95, color=TYPE_COLOR[t], label=TYPE_ZH[t])
        axes[1].bar(x + (j - 1) * w, [lv["by_type"][t]["ttft_p90_s"] for lv in at], w * 0.95, color=TYPE_COLOR[t], label=TYPE_ZH[t])
    axes[0].plot(x, [lv["slo_all"] * 100 for lv in at], "k_", markersize=22, mew=2, label="全部请求")
    axes[0].set_ylabel("满足 SLO 的请求比例（%）")
    axes[0].set_title("（a）满足 SLO 的比例", fontsize=9, loc="left")
    axes[1].axhline(1.0, color="#CC0000", linestyle="--", linewidth=0.8)
    axes[1].text(-0.4, 1.05, "SLO：TTFT ≤ 1 秒", fontsize=7, color="#CC0000")
    axes[1].set_ylabel("TTFT 的 P90（秒）")
    axes[1].set_title("（b）首个文字等多久", fontsize=9, loc="left")
    axes[2].bar(x, [lv["by_type"]["W3b"]["e2e_per_audio_p90"] for lv in at], 0.6, color=TYPE_COLOR["W3b"])
    axes[2].axhline(0.5, color="#CC0000", linestyle="--", linewidth=0.8)
    axes[2].text(-0.4, 0.52, "SLO：端到端 ÷ 语音秒 ≤ 0.5", fontsize=7, color="#CC0000")
    axes[2].set_ylabel("W3b 端到端 ÷ 语音秒的 P90")
    axes[2].set_title("（c）语音回答是不是太慢", fontsize=9, loc="left")
    for ax in axes:
        ax.set_xticks(x, [m["label"] for m in mix], fontsize=7, rotation=15)
    h, l = axes[0].get_legend_handles_labels()
    axes[1].legend(h, l, fontsize=7, frameon=False, loc="upper right")
    finish(fig, path, "图 2　混合负载、闭环并发 256：thinker 太少时三类请求的 TTFT 一起超标，talker 太少时语音回答超标。数据：results/summary.json")
    plt.close(fig)


def fig3(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for m in ordered(s["mix"]):
        xs = [x["concurrency"] for x in m["closed"]]
        ax.plot(xs, [x["by_type"]["W1"]["ttft_p90_s"] for x in m["closed"]], marker="o", color=COLOR[m["layout"]],
                label=f"{m['label']}：{m['thinker_gpus']} 个 thinker 副本，最多 {m['thinker_gpus'] * THINKER_SLOTS} 个请求同时在算")
    ax.axhline(1.0, color="#CC0000", linestyle="--", linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([16, 32, 64, 128, 256], ["16", "32", "64", "128", "256"])
    ax.set_xlabel("闭环并发（同时在系统里的请求数）")
    ax.set_ylabel("W1 文字聊天的 TTFT P90（秒，对数）")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    finish(fig, path, "图 3　2+6、3+5 的 TTFT 在并发超过“thinker 副本数 × 64”的那一档跳过 1 秒（推算的解释，见正文；6+2 在并发 256 时也变长，原因不同）。数据：results/summary.json")
    plt.close(fig)


def _agent_phase(a: dict[str, Any], agent: str, conc: int, routing: str = "least_outstanding", tool: float = 0.0) -> dict[str, Any] | None:
    return next((p for p in a["phases"].values() if p["agent"] == agent and p["concurrency"] == conc and p["routing"] == routing
                 and p["tool_latency_s"] == tool), None)


def fig4(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, agent, title in ((axes[0], "text", "（a）文本 Agent"), (axes[1], "voice", "（b）语音 Agent（最后一步语音回答）")):
        for a in ordered(s["agent"]):
            ps = [p for c in (64, 128, 256, 512) if (p := _agent_phase(a, agent, c))]
            xs = [p["concurrency"] for p in ps]
            ax.plot(xs, [p["session_p50_s"] for p in ps], marker="o", color=COLOR[a["layout"]], label=f"{a['label']}：中位数")
            ax.plot(xs, [p["session_p90_s"] for p in ps], marker=".", linestyle=":", color=COLOR[a["layout"]], alpha=0.7, label=f"{a['label']}：P90")
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 128, 256, 512], ["64", "128", "256", "512"])
        ax.set_xlabel("同时进行的会话数")
        ax.set_ylabel("一个会话从开始到结束（秒）")
        ax.set_title(title, fontsize=9, loc="left")
        ax.legend(fontsize=6.5, frameon=False, ncol=2)
    finish(fig, path, "图 4　Agent 负载：会话时长（第二批加长版，每档 512–2048 个会话；8 个完整副本用“在途请求最少”路由）。数据：results/summary.json")
    plt.close(fig)


def fig5(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    agents = ordered(s["agent"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
    x = np.arange(len(agents))
    for ax, agent, title in ((axes[0], "text", "（a）文本 Agent，512 个会话"), (axes[1], "voice", "（b）语音 Agent，512 个会话")):
        th, tk = [], []
        for a in agents:
            busy = (_agent_phase(a, agent, 512) or {}).get("role_busy") or {}
            th.append(busy.get("thinker", busy.get("full")))
            tk.append(busy.get("talker"))
        ax.bar(x - 0.18, th, 0.34, color="#0072B2", label="放 thinker 的卡（8 个完整副本：每张卡）")
        ax.bar(x + 0.18, [v if v is not None else 0 for v in tk], 0.34, color="#D55E00", label="只放 talker 和 code2wav 的卡")
        for i, v in enumerate(tk):
            if v is not None and v < 1:
                ax.text(x[i] + 0.18, 2, "0%", ha="center", fontsize=7, color="#D55E00")
        ax.set_xticks(x, [a["label"] for a in agents], fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("nvidia-smi 忙碌比例（%，这一档的均值）")
        ax.set_title(title, fontsize=9, loc="left")
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    finish(fig, path, "图 5　文本 Agent 不产生语音：拆开的分法里，talker 卡全程空着。数据：results/summary.json")
    plt.close(fig)


def fig6(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    cmp = s["verdicts"]["B-H3"]["comparison"]
    r8 = next(a for a in s["agent"] if a["layout"] == "r8")
    levels = [c["level"] for c in cmp]
    x = np.arange(len(levels))
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.2))
    lo = [r8["phases"][f"lo_{lv}"]["prefix_hit"] * 100 for lv in levels]
    sa = [r8["phases"][f"sa_{lv}"]["prefix_hit"] * 100 for lv in levels]
    a.bar(x - 0.18, lo, 0.34, color="#999999", label="按在途请求最少（默认）")
    a.bar(x + 0.18, sa, 0.34, color="#009E73", label="按会话固定")
    a.set_ylim(80, 92)
    a.set_ylabel("前缀缓存命中（输入 token 的比例，%）")
    a.set_title("（a）命中率只高 0.7–1.8 个百分点", fontsize=9, loc="left")
    a.legend(fontsize=7, frameon=False)
    for key, color, name in (("ttft_p50_rel", "#0072B2", "TTFT 中位数"), ("session_p50_rel", "#E69F00", "会话时长中位数")):
        b.plot(x, [c[key] * 100 for c in cmp], marker="o", color=color, label=name)
    b.axhspan(-5, 5, color="#DDDDDD", alpha=0.6, label="±5%：判定用的界限（看过数据后定的，见 hypotheses.md）")
    b.axhline(0, color="black", linewidth=0.6)
    b.set_ylabel("按会话固定 ÷ 默认 − 1（%）")
    b.set_title("（b）延迟有快有慢，没有一致的方向", fontsize=9, loc="left")
    b.legend(fontsize=7, frameon=False)
    for ax in (a, b):
        ax.set_xticks(x, [lv.replace("text_c", "文本 ").replace("voice_c", "语音 ") for lv in levels], fontsize=7)
    finish(fig, path, "图 6　8 个完整副本上两种路由的对比（每档各跑 1 次）。数据：results/summary.json")
    plt.close(fig)


def fig7(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    a = next(x for x in s["agent"] if x["layout"] == "s2x6")
    ps = [_agent_phase(a, "text", 256, tool=t) for t in (0.0, 0.5, 2.0)]
    names = ["工具 0 秒", "工具约 0.5 秒", "工具约 2 秒"]
    fig, (l, r) = plt.subplots(1, 2, figsize=(12, 4.0))
    x = np.arange(4)
    for j, (p, n) in enumerate(zip(ps, names)):
        l.bar(x + (j - 1) * 0.26, [p["ttft_p50_s"], p["request_p50_s"], p["request_p90_s"], p["session_p50_s"]], 0.25,
              color=["#56B4E9", "#0072B2", "#003F6B"][j], label=n)
    l.set_xticks(x, ["TTFT 中位数", "单次请求中位数", "单次请求 P90", "会话中位数"], fontsize=8)
    l.set_ylabel("秒")
    l.set_title("（a）工具越慢，模型请求反而越快", fontsize=9, loc="left")
    l.legend(fontsize=7, frameon=False)
    xs = np.arange(3)
    r.bar(xs - 0.18, [p["mean_in_flight"] for p in ps], 0.34, color="#999999", label="同时在模型里的请求数（平均）")
    r2 = r.twinx()
    r2.bar(xs + 0.18, [p["role_busy"]["thinker"] for p in ps], 0.34, color="#0072B2", label="thinker 卡忙碌比例（%）")
    r.set_xticks(xs, names, fontsize=8)
    r.set_ylabel("请求数")
    r2.set_ylabel("%")
    r2.set_ylim(0, 100)
    r.set_title("（b）等工具的会话不占模型", fontsize=9, loc="left")
    h1, l1 = r.get_legend_handles_labels()
    h2, l2 = r2.get_legend_handles_labels()
    r.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper right")
    finish(fig, path, "图 7　2+6、文本 Agent、256 个会话：模拟的工具延迟从 0 到约 2 秒（对数正态分布的中位数）。数据：results/summary.json")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    s = json.loads((out / "results" / "summary.json").read_text())
    plt = setup_matplotlib()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    for n, fn in ((1, fig1), (2, fig2), (3, fig3), (4, fig4), (5, fig5), (6, fig6), (7, fig7)):
        name = {1: "fig1_mix_throughput", 2: "fig2_mix_c256", 3: "fig3_thinker_slots", 4: "fig4_agent_sessions",
                5: "fig5_agent_gpu_busy", 6: "fig6_routing", 7: "fig7_tool_latency"}[n]
        fn(plt, s, out / "figures" / f"{name}.png")
        print(f"figures/{name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
