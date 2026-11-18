#!/usr/bin/env python3
"""实验 5 第二版：把 bubble 普查和逐步拆分画成图，并汇总报告里引用的数字。

    python scripts/analysis/exp05_bubble_figures.py --out research/01_性能瓶颈/实验05_GPU为什么会等待 \
        --exp01 research/01_性能瓶颈/实验01_8张GPU在做什么/results/kernel_summary.json

Reads results/bubble_census.json (exp05_bubble_census.py), results/step_sync.json (exp05_step_sync.py) and
results/summary.json (first round) from --out, and 实验 1's kernel-level summary. Writes figures 5-9 and 12
and results/bubble_summary.json.

The W3b sweep combines two batches: concurrency 1-8 from plan v2's second batch (compute-03, runs v2b_sb_*),
16-256 from the first batch (the shared-partition node, runs v2_a_*_run1 and _run2, averaged). Both are
8xA6000 with the same software; the overlap at concurrency 8 / 16 is checked in the report.
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

LAYOUTS = [("r8", "8 个完整副本"), ("p4", "4 对两卡流水线"), ("s1x7", "1+7"), ("s2x6", "2+6"), ("s3x5", "3+5"), ("s4x4", "4+4 合用")]
LEVELS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
ROLE_SHORT = {"完整副本": "完整副本", "thinker 卡": "thinker 卡", "talker 卡": "talker 卡"}
# Stack order, label, color. idle_with_work = work - busy; the no-work parts are exclusive (census).
PARTS = [("busy", "GPU 在算", "#525252"),
         ("idle_with_work", "有活却空着（类型 3/4/6）", "#cb181d"),
         ("type1", "等上游 thinker（类型 1）", "#fd8d3c"),
         ("type5", "请求在传递中（类型 5）", "#807dba"),
         ("type3", "等 CPU 预处理（类型 3）", "#8c6d31"),
         ("imbalance", "同阶段别的卡有活", "#fed976"),
         ("type2", "请求都在下游（类型 2）", "#6baed6"),
         ("type7", "没有请求（类型 7）", "#d9d9d9")]
CAUSE_SHORT = {"idle_with_work": "有活空着", "type1": "等上游", "type5": "传递中", "type3": "等CPU",
               "imbalance": "别卡有活", "type2": "在下游", "type7": "没请求"}


def runs_for(layout: str) -> list[tuple[str, str]]:
    """(experiment id, phase) feeding one layout's W3b row: low concurrency first, then the sweep."""
    out = [(f"v2b_sb_{layout}", "w3b_small")] if layout in ("r8", "p4", "s1x7", "s2x6") else []
    return out + [(f"v2_a_{layout}_run1", "w3b_closed"), (f"v2_a_{layout}_run2", "w3b_closed")]


def cell(census: dict[str, Any], layout: str, level: int) -> dict[str, dict[str, float]]:
    """Role -> mean fractions at one concurrency, over the runs that have it."""
    acc: dict[str, list[dict[str, float]]] = {}
    for eid, phase in runs_for(layout):
        run = census["runs"].get(eid)
        if not run or phase not in run["phases"]:
            continue
        for row in run["phases"][phase]["levels"]:
            if row["kind"] == "closed" and row["level"] == level:
                for role, v in row["roles"].items():
                    acc.setdefault(role, []).append(v)
    return {role: {k: float(np.nanmean([v[k] for v in vs if k in v])) for k in vs[0]} for role, vs in acc.items()}


def path_at(census: dict[str, Any], layout: str, level: int, key: str = "path_ms_p50") -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for eid, phase in runs_for(layout):
        run = census["runs"].get(eid)
        if not run or phase not in run["phases"]:
            continue
        for row in run["phases"][phase]["levels"]:
            if row["kind"] == "closed" and row["level"] == level:
                for k, v in (row.get(key) or {}).items():
                    acc.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def stack(v: dict[str, float]) -> dict[str, float]:
    """Parts that add up to 1: busy, idle with work, and the no-work causes scaled into 1 - max(busy, work)."""
    busy = v.get("busy", float("nan"))
    if "work" not in v:
        return {"busy": busy}
    work = v["work"]
    out = {"busy": busy, "idle_with_work": max(0.0, work - busy)}
    nowork = sum(v.get(k, 0.0) for k, _, _ in PARTS[2:])
    room = 1.0 - max(busy, work)
    for k, _, _ in PARTS[2:]:
        out[k] = v.get(k, 0.0) * (room / nowork if nowork > 0 else 0.0)
    return out


def dominant(v: dict[str, float]) -> str:
    s = stack(v)
    idle = {k: s.get(k, 0.0) for k in CAUSE_SHORT}
    k = max(idle, key=idle.get)
    return CAUSE_SHORT[k] if idle[k] > 0.03 else ""


# ------------------------------------------------------------------------------------------
# figures


def fig_map(plt, census: dict[str, Any], path: Path) -> list[dict[str, Any]]:  # noqa: ANN001
    rows, labels = [], []
    for lay, name in LAYOUTS:
        roles = ["完整副本"] if lay == "r8" else ["thinker 卡", "talker 卡"]
        for role in roles:
            vals = []
            for lv in LEVELS:
                v = cell(census, lay, lv).get(role)
                vals.append(v)
            rows.append(vals)
            labels.append(f"{name}：{role}")
    fig, ax = plt.subplots(figsize=(12, 0.55 * len(rows) + 1.6))
    grid = np.array([[1 - v["busy"] if v else np.nan for v in r] for r in rows])
    im = ax.imshow(grid * 100, cmap="OrRd", vmin=0, vmax=100, aspect="auto")
    table = []
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            if not v:
                continue
            idle = 1 - v["busy"]
            cause = dominant(v)
            ax.text(j, i, f"{idle * 100:.0f}%\n{cause}", ha="center", va="center", fontsize=6.5, color="white" if idle > 0.6 else "black")
            table.append({"row": labels[i], "level": LEVELS[j], "idle": idle, "dominant": cause, **stack(v)})
    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels([f"并发 {x}" for x in LEVELS], fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(3.5, color="black", lw=0.8, ls="--")
    ax.text(1.5, len(rows) + 0.15, "↑ 第二批，compute-03", ha="center", va="top", fontsize=7.5)
    ax.text(6, len(rows) + 0.15, "↑ 第一批节点（每格是 2 次运行的平均）", ha="center", va="top", fontsize=7.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("GPU 空着的比例（%，nvidia-smi）", fontsize=8)
    ax.set_title("图 5　Bubble 地图：W3b 语音负载下，每种分工、每种卡在不同并发时空着多少、主要为什么空着\n"
                 "（格子里：空着的比例 + 空着时最大的原因；颜色越深越空）", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return table


def fig_breakdown(plt, census: dict[str, Any], path: Path) -> list[dict[str, Any]]:  # noqa: ANN001
    sel = [("r8", "8 个完整副本"), ("p4", "4 对两卡流水线"), ("s1x7", "1+7"), ("s2x6", "2+6")]
    bars, out = [], []
    for lay, name in sel:
        for lv in (8, 64, 256):
            c = cell(census, lay, lv)
            for role in (["完整副本"] if lay == "r8" else ["thinker 卡", "talker 卡"]):
                if role in c:
                    bars.append((f"{name}　{role}　并发 {lv}", stack(c[role])))
                    out.append({"layout": lay, "role": role, "level": lv, **stack(c[role])})
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(bars) + 1.8))
    for i, (_, s) in enumerate(bars):
        left = 0.0
        for k, lab, col in PARTS:
            w = s.get(k, 0.0) * 100
            if w > 0:
                ax.barh(i, w, left=left, color=col)
                if w >= 7:
                    ax.text(left + w / 2, i, f"{w:.0f}", ha="center", va="center", fontsize=6.5, color="white" if k in ("busy", "idle_with_work") else "black")
                left += w
    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels([b[0] for b in bars], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("占这一档运行时间的百分比（GPU 在算 = nvidia-smi；有活却空着 = 有活比例 − 在算比例；其余按服务端事件实测）", fontsize=8)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=c, label=lab) for _, lab, c in PARTS], fontsize=7.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.06), ncol=4, frameon=False)
    ax.set_title("图 6　每张卡的时间去哪了：并发 8、64、256，按 bubble 类型拆开", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_stage_lines(plt, census: dict[str, Any], keys: list[tuple[str, str]], ylabel: str, title: str, path: Path,
                    logy: bool = False) -> dict[str, Any]:  # noqa: ANN001
    sel = [("s2x6", "2+6", "#d94801"), ("p4", "4 对两卡流水线", "#2171b5"), ("s1x7", "1+7", "#41ab5d"), ("r8", "8 个完整副本", "#6a51a3")]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.6), squeeze=False)
    data: dict[str, Any] = {}
    for ax, (key, lab) in zip(axes[0], keys):
        for lay, name, col in sel:
            xs, ys = [], []
            for lv in LEVELS:
                v = path_at(census, lay, lv).get(key)
                if v is not None:
                    xs.append(lv)
                    ys.append(v)
            ax.plot(xs, ys, marker="o", ms=3, color=col, label=name)
            data.setdefault(key, {})[lay] = dict(zip(xs, ys))
        ax.set_xscale("log", base=2)
        ax.set_xticks(LEVELS)
        ax.set_xticklabels([str(x) for x in LEVELS], fontsize=7)
        if logy:
            from matplotlib.ticker import FuncFormatter, LogLocator

            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("并发", fontsize=8)
        ax.set_title(lab, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(ylabel, fontsize=8)
    axes[0][0].legend(fontsize=7)
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return data


PATH_PARTS = [("pre", "进入 thinker 前（CPU）", "#bdbdbd"), ("thinker_queue", "thinker 排队", "#fdae6b"),
              ("thinker_prefill", "thinker 读提示", "#fd8d3c"), ("thinker_decode", "thinker 逐字生成", "#d94801"),
              ("hop1", "交给 talker", "#807dba"), ("talker_queue", "talker 排队", "#a1d99b"),
              ("talker_gen", "talker 生成语音 token", "#41ab5d"), ("hop2", "交给 code2wav", "#9e9ac8"),
              ("code2wav_run", "code2wav 排队 + 合成", "#6a51a3"), ("tail", "返回最后一段", "#969696")]


def fig_path(plt, census: dict[str, Any], path: Path) -> dict[str, Any]:  # noqa: ANN001
    sel = [("s2x6", "2+6"), ("p4", "4 对两卡流水线"), ("s1x7", "1+7"), ("r8", "8 个完整副本")]
    rows = []
    for lay, name in sel:
        for lv in (8, 256):
            p = path_at(census, lay, lv)
            if p:
                rows.append((f"{name}，并发 {lv}", p))
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(rows) + 2))
    for i, (_, p) in enumerate(rows):
        left = 0.0
        for k, lab, col in PATH_PARTS:
            w = p.get(k, 0.0) / 1e3
            ax.barh(i, w, left=left, color=col, label=lab if i == 0 else None)
            if w > 0.35:
                ax.text(left + w / 2, i, f"{w:.1f}", ha="center", va="center", fontsize=6.5, color="white")
            left += w
        ax.text(left + 0.1, i, f"合计 {p.get('total', left * 1e3) / 1e3:.1f} 秒", va="center", fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("一个请求在服务端经过的每一段（秒，各段取中位数，所以合计和端到端中位数略有出入）", fontsize=8)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, frameon=False)
    ax.set_title("图 9　典型请求的关键路径：低负载时 thinker 逐字生成最长，高负载时 code2wav 排队最长（1+7 是 thinker 排队）", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {r[0]: r[1] for r in rows}


def fig_rank(plt, entries: list[dict[str, Any]], path: Path) -> None:  # noqa: ANN001
    colors = {"决定吞吐": "#cb181d", "影响延迟": "#fd8d3c", "特定负载下决定一切": "#8c6d31", "基本不影响": "#bdbdbd"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 0.6 * len(entries) + 2.2), sharey=True, gridspec_kw={"width_ratios": [1, 1]})
    for i, e in enumerate(entries):
        c = colors[e["impact"]]
        a1.barh(i, e["gpu_share"] * 100, color=c)
        a1.text(e["gpu_share"] * 100 + 1, i, e["gpu_note"], va="center", fontsize=7)
        a2.barh(i, e["e2e_share"] * 100, color=c)
        a2.text(e["e2e_share"] * 100 + 1, i, e["e2e_note"], va="center", fontsize=7)
    a1.set_yticks(range(len(entries)))
    a1.set_yticklabels([f"{i + 1}. {e['name']}" for i, e in enumerate(entries)], fontsize=8)
    a1.invert_yaxis()
    a1.set_xlim(0, 210)
    a2.set_xlim(0, 210)
    for ax in (a1, a2):
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.axvline(100, color="#bdbdbd", lw=0.6)
    a1.set_xlabel("这类空闲占这些卡时间的比例（%，典型情况）", fontsize=8)
    a2.set_xlabel("对一个请求端到端时间的影响（%，典型情况）", fontsize=8)
    from matplotlib.patches import Patch

    a1.legend(handles=[Patch(color=v, label=k) for k, v in colors.items()], fontsize=7.5, loc="lower right", title="对吞吐和延迟的影响", title_fontsize=7.5)
    fig.suptitle("图 12　Bubble 根因排序：按对吞吐和延迟的影响排，不按空闲多少排（左边最长的条不一定最重要）", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------------------------------


def rank_entries(census: dict[str, Any], steps: dict[str, Any], k1: dict[str, Any], r1: dict[str, Any]) -> list[dict[str, Any]]:
    """The ranked bubbles with the numbers behind each bar (see the report's section 6)."""
    def role(lay: str, lv: int, r: str) -> dict[str, float]:
        return stack(cell(census, lay, lv)[r])

    p8, p256 = path_at(census, "s2x6", 8), path_at(census, "s2x6", 256)
    c2w = {lay: path_at(census, lay, 256) for lay in ("s2x6", "s3x5", "s4x4", "p4")}
    c2w_share = {lay: p["code2wav_run"] / p["total"] for lay, p in c2w.items()}
    th = steps["runs"]["C"]["median_over_processes"]["w3b_prof_c8"]["thinker"]
    th_share8 = th["idle_ms_p50"] / th["span_ms_p50"]
    r8_8, r8_256 = path_at(census, "r8", 8), path_at(census, "r8", 256)
    s17 = role("s1x7", 256, "thinker 卡")
    s17p = path_at(census, "s1x7", 256)
    k64 = k1["runs"]["A"]["phases"]["w3b_prof_c64"]["gpus"]
    r8_cont = float(np.mean([g["contended"] for g in k64.values()]))
    kc = k1["runs"]["C"]["phases"]["w3b_prof_c64"]["gpus"]
    talk_cont64 = float(np.mean([kc[g]["contended"] for g in kc if g not in ("0", "1")]))
    talk_cont128 = [r1["window"]["gpus"][g]["contended"] for g in r1["window"]["gpus"] if g not in ("0", "1")]
    video = census["runs"]["v2b_f1_r1_events"]["phases"]["w4_video_10s"]["levels"]
    v1 = video[0]
    text = census["runs"]["v2b_b2l_s2x6"]["phases"]["text_c256"]["levels"][0]["roles"]["talker 卡"]
    s26_8 = role("s2x6", 8, "talker 卡")
    s26_256t = role("s2x6", 256, "thinker 卡")
    hops = (p8.get("hop1", 0) + p8.get("hop2", 0)) / p8["total"]
    e = [
        {"name": "talker 卡忙不过来，请求在 code2wav 排队（类型 2）", "impact": "决定吞吐",
         "gpu_share": 1 - role("s2x6", 256, "talker 卡")["busy"], "gpu_note": f"talker 卡只空 {(1 - role('s2x6', 256, 'talker 卡')['busy']) * 100:.0f}%：它是瓶颈，不是 bubble",
         "e2e_share": c2w_share["s2x6"], "e2e_note": f"并发 256：2+6 {c2w_share['s2x6'] * 100:.0f}%，4+4 {c2w_share['s4x4'] * 100:.0f}%，两卡流水线 {c2w_share['p4'] * 100:.0f}%"},
        {"name": "thinker 每一步同步等待（类型 4，CPU 准备 = 类型 3）", "impact": "影响延迟",
         "gpu_share": role("s2x6", 64, "thinker 卡")["idle_with_work"],
         "gpu_note": f"2+6：{role('s2x6', 64, 'thinker 卡')['idle_with_work'] * 100:.0f}%（并发 64）；1+7 并发 256：{s17['idle_with_work'] * 100:.0f}%",
         "e2e_share": p8["thinker_decode"] * th_share8 / p8["total"],
         "e2e_note": f"并发 8 约 {p8['thinker_decode'] * th_share8 / p8['total'] * 100:.0f}%；1+7 时卡住 thinker，排队 {s17p.get('thinker_queue', 0) / 1e3:.1f} 秒"},
        {"name": "三段挤在一张卡上互相抢（类型 6，8 个完整副本）", "impact": "决定吞吐",
         "gpu_share": r8_cont, "gpu_note": f"并发 64 时 {r8_cont * 100:.0f}% 的时间在抢（kernel 级）",
         "e2e_share": (r8_256["thinker_decode"] - r8_8["thinker_decode"]) / r8_256["total"],
         "e2e_note": f"thinker 逐字生成从 {r8_8['thinker_decode'] / 1e3:.1f} 秒变成 {r8_256['thinker_decode'] / 1e3:.1f} 秒"},
        {"name": "talker 和 code2wav 抢同一张卡（类型 6）", "impact": "决定吞吐",
         "gpu_share": float(np.mean(talk_cont128)), "gpu_note": f"并发 64：{talk_cont64 * 100:.0f}%；并发 128：{min(talk_cont128) * 100:.0f}–{max(talk_cont128) * 100:.0f}%",
         "e2e_share": 0.0, "e2e_note": "包含在第 1 条里，单独的影响还没测"},
        {"name": "视频：GPU 等 CPU 解码视频（类型 3）", "impact": "特定负载下决定一切",
         "gpu_share": v1["roles"]["thinker 卡"].get("type3", 0.0), "gpu_note": f"单副本并发 1：{v1['roles']['thinker 卡'].get('type3', 0) * 100:.0f}%",
         "e2e_share": v1["path_ms_p50"]["pre"] / v1["path_ms_p50"]["total"], "e2e_note": f"预处理占端到端 {v1['path_ms_p50']['pre'] / v1['path_ms_p50']['total'] * 100:.0f}%（并发 4 时 {video[2]['path_ms_p50']['pre'] / video[2]['path_ms_p50']['total'] * 100:.0f}%）"},
        {"name": "thinker 卡没活：请求都在下游（类型 2 在上游的表现）", "impact": "基本不影响",
         "gpu_share": s26_256t["type2"], "gpu_note": f"并发 256：2+6 {s26_256t['type2'] * 100:.0f}%，两卡流水线 {role('p4', 256, 'thinker 卡')['type2'] * 100:.0f}%：thinker 卡给多了",
         "e2e_share": 0.0, "e2e_note": "不在请求的关键路径上"},
        {"name": "talker 卡等 thinker 交活（类型 1）", "impact": "基本不影响",
         "gpu_share": s26_8["type1"], "gpu_note": f"2+6 并发 8：{s26_8['type1'] * 100:.0f}%；并发 256：{role('s2x6', 256, 'talker 卡')['type1'] * 100:.0f}%",
         "e2e_share": 0.0, "e2e_note": "请求这时在 thinker 那边，不是在等"},
        {"name": "没有请求，或没有要说话的请求（类型 7）", "impact": "基本不影响",
         "gpu_share": text.get("type7", 0.0) + text.get("type2", 0.0), "gpu_note": f"Agent 文字负载下 talker 卡 100% 空着；8 个完整副本并发 1 时 {role('r8', 1, '完整副本')['type7'] * 100:.0f}%",
         "e2e_share": 0.0, "e2e_note": "部署选择的问题：卡给了用不上的阶段"},
        {"name": "阶段之间传递数据（类型 5）", "impact": "基本不影响",
         "gpu_share": role("s2x6", 8, "talker 卡")["type5"], "gpu_note": "≤ 1%",
         "e2e_share": hops, "e2e_note": f"每个请求两次传递共约 {(p8.get('hop1', 0) + p8.get('hop2', 0)):.0f} 毫秒"},
    ]
    return e


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--exp01", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    census = json.load(open(out / "results" / "bubble_census.json"))
    steps = json.load(open(out / "results" / "step_sync.json"))
    r1 = json.load(open(out / "results" / "summary.json"))
    k1 = json.load(open(a.exp01))
    plt = setup_matplotlib()
    fig = out / "figures"
    summary: dict[str, Any] = {"generated_by": "scripts/analysis/exp05_bubble_figures.py"}
    summary["map"] = fig_map(plt, census, fig / "fig5_bubble_map.png")
    summary["breakdown"] = fig_breakdown(plt, census, fig / "fig6_gpu_time_by_type.png")
    summary["stage_time"] = fig_stage_lines(
        plt, census, [("thinker_ms_per_token", "thinker：每生成一个文字 token（毫秒）"),
                      ("talker_gen_ms_per_audio_s", "talker：每生成 1 秒语音（毫秒）"),
                      ("code2wav_run_ms_per_audio_s", "code2wav：每合成 1 秒语音，含排队（毫秒）")],
        "毫秒（对数坐标）", "图 7　各阶段的处理时间：负载升高后，2+6 和两卡流水线的 thinker 只慢一点，8 个完整副本和 1+7 的 thinker 明显变慢；"
        "talker 慢 2–4 倍；code2wav（含排队）慢几十倍",
        fig / "fig7_stage_time.png", logy=True)
    summary["stage_wait"] = fig_stage_lines(
        plt, census, [("thinker_queue", "thinker 排队（毫秒）"), ("talker_queue", "talker 排队（毫秒）"),
                      ("hop1", "thinker 交给 talker（毫秒）"), ("hop2", "talker 交给 code2wav（毫秒）")],
        "毫秒（对数坐标）", "图 8　各阶段门口的等待时间：只有 1+7 的 thinker 排队变长（最多 6 秒）；talker 排队和阶段之间的传递一直是几到二十几毫秒"
        "（code2wav 的排队和合成分不开，在图 7）",
        fig / "fig8_stage_wait.png", logy=True)
    summary["path"] = fig_path(plt, census, fig / "fig9_critical_path.png")
    entries = rank_entries(census, steps, k1, r1)
    summary["rank"] = entries
    fig_rank(plt, entries, fig / "fig12_root_cause_rank.png")
    (out / "results" / "bubble_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float) + "\n")
    print(out / "results" / "bubble_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
