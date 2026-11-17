#!/usr/bin/env python3
"""实验 1（8 张 GPU 在做什么）：每张 GPU 的忙闲时间线、各阶段的处理和等待时间、一个请求的关键路径。

    python scripts/analysis/exp01_gpu_timeline.py --data-root ~/research-staging/unicorn \
        --scheme A=20261115_v2_a_r8_run1_run01 --scheme B=20261115_v2_a_p4_run1_run01 \
        --scheme C=20261115_v2_a_s2x6_run1_run01 \
        --low A=20261116_v2b_sb_r8_run01 --low B=20261116_v2b_sb_p4_run01 --low C=20261116_v2b_sb_s2x6_run01 \
        --out research/01_性能瓶颈/实验01_8张GPU在做什么

--scheme: the run with the closed-loop sweep (phase w3b_closed, concurrency 16-256); --low: the run
with the small-batch sweep (phase w3b_small, concurrency 1-8). Each run directory needs
raw/gpu_samples.csv (nvidia-smi every 200 ms), raw/requests_<phase>.jsonl, raw/server_events/<phase>/
and manifest.json (the node's UTC offset; nvidia-smi writes local time). Writes figures/fig*.png
and results/summary.json with every number the report quotes.

"Busy" here is nvidia-smi's utilization.gpu: the share of each sample period during which at least
one kernel was running on the GPU. It is not SM occupancy: a GPU running one small kernel all the
time shows 100%.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from omni_research.analysis.experiment_analysis import clock_anchor, read_jsonl  # noqa: E402
from stage_by_level import breakdown, level_of, load_phase_events, role_of  # noqa: E402
from summarize_plan_v2 import eid_layout, layout_roles, pct  # noqa: E402

SCHEME_NAMES = {"r8": "8 个完整副本", "p4": "4 条两卡流水线", "s2x6": "2 张 thinker 卡 + 6 张 talker 卡"}
ROLE_ZH = {"thinker": "thinker", "talker": "talker+code2wav", "full replica": "完整副本"}


# ------------------------------------------------------------------------------------------
# loading


def utc_offset_s(exp: Path) -> int:
    """Seconds east of UTC of the node's local clock ("-0500" -> -18000)."""
    try:
        z = json.load(open(exp / "manifest.json")).get("utc_offset") or "+0000"
    except (OSError, json.JSONDecodeError):
        z = "+0000"
    sign = -1 if z.startswith("-") else 1
    return sign * (int(z[1:3]) * 3600 + int(z[3:5]) * 60)


def load_util(exp: Path) -> dict[int, list[tuple[float, float, float]]]:
    """Per GPU index: (unix time s, sampled utilization %, memory used MiB)."""
    off = utc_offset_s(exp)
    out: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    with open(exp / "raw" / "gpu_samples.csv") as fh:
        rd = csv.reader(fh)
        hdr = [h.strip() for h in next(rd, [])]
        for row in rd:
            if len(row) != len(hdr):
                continue
            d = dict(zip(hdr, (x.strip() for x in row)))
            try:
                t = datetime.strptime(d["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp() - off
                out[int(d["index"])].append((t, float(d["utilization.gpu"]), float(d["memory.used"])))
            except (KeyError, ValueError):
                continue
    return out


def level_windows(exp: Path, phase: str) -> list[tuple[float, float, float]]:
    """(level, start, end) in unix seconds per load level, warmup requests included (they occupy the GPUs)."""
    anchor = clock_anchor(exp)
    if anchor is None:
        return []
    mono0, wall0 = anchor
    by: dict[float, list[float]] = defaultdict(list)
    for r in read_jsonl(exp / "raw" / f"requests_{phase}.jsonl"):
        lv = level_of(r)
        if lv is None or r.get("sent_ns") is None:
            continue
        by[lv].append((r["sent_ns"] - mono0 + wall0) / 1e9)
        if r.get("done_ns"):
            by[lv].append((r["done_ns"] - mono0 + wall0) / 1e9)
    return [(lv, min(ts), max(ts)) for lv, ts in sorted(by.items())]


def gpu_roles(exp: Path) -> tuple[str, dict[int, str]]:
    eid = json.load(open(exp / "result.json"))["experiment_id"]
    layout = eid_layout(eid)
    roles = {g: role for role, gs in layout_roles(layout).items() for g in gs}
    return layout, roles


def mean_util(util: dict[int, list[tuple[float, float, float]]], t0: float, t1: float) -> dict[int, float]:
    out = {}
    for g, s in util.items():
        v = [u for t, u, _ in s if t0 <= t <= t1]
        if v:
            out[g] = sum(v) / len(v)
    return out


# ------------------------------------------------------------------------------------------
# one request's critical path


SEGMENTS = [  # (label, role, start event, end event); the role's first start and last end are used
    ("预处理", "preprocessing", "stage_input_received", "stage_complete"),
    ("thinker 排队", "thinker", "scheduler_queue_enter", "scheduler_prefill_start"),
    ("thinker 读入提示（prefill）", "thinker", "scheduler_prefill_start", "scheduler_prefill_end"),
    ("thinker 逐个生成文字 token（decode）", "thinker", "scheduler_prefill_end", "model_path_end"),
    ("talker 排队", "talker", "scheduler_queue_enter", "scheduler_prefill_start"),
    ("talker 生成语音 token", "talker", "scheduler_prefill_start", "model_path_end"),
    ("code2wav 等待开始", "code2wav", "stage_input_received", "stage_dispatch"),
    ("code2wav 合成波形", "code2wav", "stage_dispatch", "stage_complete"),
]


def request_segments(evs: list[dict[str, Any]]) -> tuple[int, list[tuple[str, float, float]]]:
    """(admission ns, [(label, start ms, end ms) relative to admission]) for one request's events."""
    first: dict[tuple[str, str], int] = {}
    last: dict[tuple[str, str], int] = {}
    for e in evs:
        role, name, t = role_of(e.get("stage") or ""), e.get("event_name"), e.get("timestamp_ns")
        if t is None:
            continue
        k = (role, name)
        first[k] = min(first.get(k, t), t)
        last[k] = max(last.get(k, t), t)
    t_adm = first.get(("coordinator", "request_admission")) or min(first.values())
    segs = []
    for label, role, a, b in SEGMENTS:
        if (role, a) in first and (role, b) in last and last[(role, b)] >= first[(role, a)]:
            segs.append((label, (first[(role, a)] - t_adm) / 1e6, (last[(role, b)] - t_adm) / 1e6))
    if ("coordinator", "terminal_response") in last:
        end = (last[("coordinator", "terminal_response")] - t_adm) / 1e6
        segs.append(("总计（到最后一个响应）", 0.0, end))
    return t_adm, segs


def typical_request(exp: Path, phase: str, level: float) -> tuple[str, list[tuple[str, float, float]]] | None:
    """The measured request whose server-side total is the median of its level."""
    recs = {r["request_id"]: r for r in read_jsonl(exp / "raw" / f"requests_{phase}.jsonl")}
    by_rid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in load_phase_events(exp, phase):
        rid = e.get("request_id")
        if rid in recs and level_of(recs[rid]) == level and not recs[rid].get("warmup") and recs[rid].get("status") == "ok":
            by_rid[rid].append(e)
    paths = []
    for rid, evs in by_rid.items():
        _, segs = request_segments(evs)
        total = next((b for lab, a, b in segs if lab.startswith("总计")), None)
        if total is not None and len(segs) >= 6:
            paths.append((total, rid, segs))
    if not paths:
        return None
    paths.sort()
    _, rid, segs = paths[len(paths) // 2]
    return rid, segs


# ------------------------------------------------------------------------------------------
# figures


def setup_matplotlib():  # noqa: ANN201
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Hiragino Sans GB", "Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    return plt


def gpu_label(g: int, roles: dict[int, str]) -> str:
    return f"GPU {g}（{ROLE_ZH.get(roles.get(g, ''), roles.get(g, '?'))}）"


def fig_timeline(plt, schemes: list[dict[str, Any]], path: Path) -> None:  # noqa: ANN001
    import numpy as np

    fig, axes = plt.subplots(len(schemes), 1, figsize=(12, 2.9 * len(schemes)), squeeze=False)
    for ax, s in zip(axes[:, 0], schemes):
        wins = s["windows"]
        t0, t1 = wins[0][1], wins[-1][2]
        nbin = int(t1 - t0) + 1
        grid = np.full((8, nbin), np.nan)
        for g, samples in s["util"].items():
            acc: dict[int, list[float]] = defaultdict(list)
            for t, u, _ in samples:
                if t0 <= t <= t1:
                    acc[int(t - t0)].append(u)
            for b, v in acc.items():
                grid[g, b] = sum(v) / len(v)
        im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=100, interpolation="nearest",
                       extent=(0, nbin, 7.5, -0.5))
        for lv, a, b in wins:
            ax.axvline(a - t0, color="white", lw=0.8, ls="--")
            ax.text((a - t0) + 1, -0.75, f"并发 {lv:g}", fontsize=8, va="bottom")
        ax.set_yticks(range(8))
        ax.set_yticklabels([gpu_label(g, s["roles"]) for g in range(8)], fontsize=8)
        ax.set_title(f"方案 {s['key']}：{s['name']}", fontsize=10, loc="left", pad=14)
        ax.set_xlabel("时间（秒，从并发 16 开始）", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01).set_label("忙碌比例 %", fontsize=8)
    fig.suptitle("图 1　8 张 GPU 的工作时间线（颜色越亮越忙；虚线是并发档的分界）\n"
                 "忙碌比例 = nvidia-smi 每 0.2 秒采样里“有 kernel 在跑”的时间比例，按 1 秒取平均", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_busy(plt, schemes: list[dict[str, Any]], levels: list[float], path: Path) -> None:  # noqa: ANN001
    import numpy as np

    fig, axes = plt.subplots(1, len(schemes), figsize=(4.6 * len(schemes), 3.6), squeeze=False, sharey=True)
    width = 0.8 / len(levels)
    colors = ["#9ecae1", "#4292c6", "#08519c", "#08306b"]
    for ax, s in zip(axes[0], schemes):
        for i, lv in enumerate(levels):
            u = s["busy"].get(lv, {})
            ax.bar(np.arange(8) + (i - (len(levels) - 1) / 2) * width, [u.get(g, np.nan) for g in range(8)], width,
                   label=f"并发 {lv:g}", color=colors[i % len(colors)])
        ax.set_xticks(range(8))
        ax.set_xticklabels([f"{g}\n{ROLE_ZH.get(s['roles'].get(g, ''), '')[:7]}" for g in range(8)], fontsize=7)
        ax.set_ylim(0, 100)
        ax.set_title(f"方案 {s['key']}：{s['name']}", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("平均忙碌比例 %")
    axes[0][-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("图 2　每张 GPU 的平均忙碌比例（横轴：GPU 编号和它负责的阶段；空白 = 100% 减去柱高）", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


STACK = [  # (label, role, metric, color); thinker decode = model path - prefill
    ("thinker 排队", "thinker", "queue", "#fdd0a2"),
    ("thinker prefill", "thinker", "prefill", "#fd8d3c"),
    ("thinker 解码", "thinker", "decode", "#d94801"),
    ("talker 排队", "talker", "queue", "#c7e9c0"),
    ("talker 生成", "talker", "model_path", "#41ab5d"),
    ("code2wav（含等待）", "code2wav", "service", "#6a51a3"),
]


def stage_medians(row: dict[str, Any]) -> dict[tuple[str, str], float]:
    sm = row["stage_ms"]
    out = {}
    for _, role, metric, _ in STACK:
        if metric == "decode":
            mp, pf = pct(sm.get(("thinker", "model_path")) or [], 0.5), pct(sm.get(("thinker", "prefill")) or [], 0.5)
            if mp is not None:
                out[(role, metric)] = mp - (pf or 0)
        else:
            v = pct(sm.get((role, metric)) or [], 0.5)
            if v is not None:
                out[(role, metric)] = v
    return out


def fig_stages(plt, rows: list[tuple[str, dict[tuple[str, str], float]]], path: Path, title: str) -> None:  # noqa: ANN001
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(rows) + 1.6))
    for i, (label, med) in enumerate(rows):
        x = 0.0
        for name, role, metric, color in STACK:
            v = med.get((role, metric))
            if v is None:
                continue
            ax.barh(i, v / 1000, left=x, color=color, edgecolor="white", label=name if i == 0 else None)
            if v > 150:
                ax.text(x + v / 2000, i, f"{v / 1000:.2f}", ha="center", va="center", fontsize=7)
            x += v / 1000
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("每个请求的服务端时间（秒，各段分别取中位数后相加，所以总长不等于端到端中位数）", fontsize=8)
    ax.legend(fontsize=8, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_load(plt, schemes: list[dict[str, Any]], path: Path) -> None:  # noqa: ANN001
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.7))
    specs = [("thinker 每个文字 token 的解码时间（毫秒）", "thinker", "ms_per_token"),
             ("talker 每个请求的生成时间（毫秒）", "talker", "model_path"),
             ("code2wav 每个请求的时间（毫秒，含等待）", "code2wav", "service"),
             ("thinker 排队时间（毫秒）", "thinker", "queue")]
    ticks = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for ax, (title, role, metric) in zip(axes, specs):
        for s in schemes:
            pts = [(row["level"], pct(row["stage_ms"].get((role, metric)) or [], 0.5)) for row in s["rows_all"]]
            pts = [(a, b) for a, b in pts if b is not None]
            if pts:
                ax.plot([a for a, _ in pts], [b for _, b in pts], marker="o", label=f"方案 {s['key']}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks], fontsize=7)
        ax.set_xlabel("并发（同时在系统里的请求数）", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("图 4　负载变大时，哪一段变慢（中位数；并发 1–8 与 16–256 来自两次运行）", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_critical(plt, paths: list[tuple[str, list[tuple[str, float, float]]]], path: Path) -> None:  # noqa: ANN001
    colors = {"预处理": "#bdbdbd", "thinker 排队": "#fdd0a2", "thinker 读入提示（prefill）": "#fd8d3c",
              "thinker 逐个生成文字 token（decode）": "#d94801", "talker 排队": "#c7e9c0", "talker 生成语音 token": "#41ab5d",
              "code2wav 等待开始": "#dadaeb", "code2wav 合成波形": "#6a51a3", "总计（到最后一个响应）": "#252525"}
    fig, axes = plt.subplots(len(paths), 1, figsize=(11, 2.6 * len(paths)), squeeze=False)
    for ax, (title, segs) in zip(axes[:, 0], paths):
        for i, (label, a, b) in enumerate(segs):
            ax.barh(i, (b - a) / 1000, left=a / 1000, color=colors.get(label, "#999999"), height=0.6)
            ax.text(b / 1000 + 0.02, i, f"{(b - a) / 1000:.2f} s", va="center", fontsize=7)
        ax.set_yticks(range(len(segs)))
        ax.set_yticklabels([s[0] for s in segs], fontsize=7)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("从请求进入服务端开始的时间（秒）", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("图 5　一个典型请求的关键路径（服务端时间居中位的那个请求；每一段都要等上一段做完）", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--scheme", action="append", required=True, help="KEY=RUN_DIR with the w3b_closed sweep (16-256)")
    ap.add_argument("--low", action="append", default=[], help="KEY=RUN_DIR with the w3b_small sweep (1-8)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    low = dict(x.split("=", 1) for x in a.low)
    plt = setup_matplotlib()
    schemes = []
    summary: dict[str, Any] = {"generated_by": "scripts/analysis/exp01_gpu_timeline.py", "schemes": {}}
    for spec in a.scheme:
        key, run = spec.split("=", 1)
        exp = root / run
        layout, roles = gpu_roles(exp)
        util = load_util(exp)
        wins = level_windows(exp, "w3b_closed")
        busy = {lv: mean_util(util, t0, t1) for lv, t0, t1 in wins}
        rows = breakdown(exp, "w3b_closed")
        s = {"key": key, "name": SCHEME_NAMES.get(layout, layout), "layout": layout, "run": run, "roles": roles, "util": util,
             "windows": wins, "busy": busy, "rows": rows, "rows_all": list(rows)}
        if key in low:
            lexp = root / low[key]
            lutil = load_util(lexp)
            for lv, t0, t1 in level_windows(lexp, "w3b_small"):
                s["busy"][lv] = mean_util(lutil, t0, t1)
            s["low_run"] = low[key]
            s["low_rows"] = breakdown(lexp, "w3b_small")
            s["rows_all"] = s["low_rows"] + list(rows)
        schemes.append(s)
        summary["schemes"][key] = {
            "name": s["name"], "layout": layout, "run": run, "low_run": s.get("low_run"),
            "gpu_roles": {str(g): r for g, r in roles.items()},
            "busy_by_level": {f"{lv:g}": {str(g): round(v, 1) for g, v in sorted(u.items())} for lv, u in sorted(s["busy"].items())},
            "mem_max_mib": {str(g): max(m for _, _, m in v) for g, v in sorted(util.items()) if v},
            "levels": [{"level": row["level"], "n": row["n"], "tp_rps": row["tp_rps"], "e2e_p50_s": pct(row["e2e_s"], 0.5),
                        "ttft_p50_s": pct(row["ttft_s"], 0.5), "slo": row["slo"], "eff_batch": row["eff_batch"],
                        "replicas": row["replicas"], "gpu_util_by_role": row["gpu_util"],
                        "stage_p50_ms": {f"{r}.{m}": v for (r, m), v in stage_medians(row).items()},
                        "thinker_ms_per_token_p50": pct(row["stage_ms"].get(("thinker", "ms_per_token")) or [], 0.5),
                        "hop_thinker_talker_p50_ms": pct(row["stage_ms"].get(("hop", "thinker->talker")) or [], 0.5)}
                       for row in s["rows_all"]],
        }
    fig_timeline(plt, schemes, out / "figures" / "fig1_gpu_timeline.png")
    shown = [lv for lv in (8.0, 64.0, 256.0) if all(lv in s["busy"] for s in schemes)]
    fig_busy(plt, schemes, shown, out / "figures" / "fig2_gpu_busy_by_level.png")
    for lv in (8.0, 64.0):
        rows = []
        for s in schemes:
            row = next((r for r in s["rows_all"] if r["level"] == lv), None)
            if row:
                rows.append((f"方案 {s['key']}：{s['name']}", stage_medians(row)))
        if rows:
            fig_stages(plt, rows, out / "figures" / f"fig3_stage_time_c{lv:g}.png",
                       f"图 3　并发 {lv:g} 时，一个请求在每个阶段花多少时间（浅色 = 排队等待，深色 = 真正在算）")
    fig_load(plt, schemes, out / "figures" / "fig4_stage_time_vs_load.png")
    crit = []
    for s in schemes:
        for lv, run, phase in ((8.0, s.get("low_run"), "w3b_small"), (64.0, s["run"], "w3b_closed")):
            if not run:
                continue
            got = typical_request(root / run, phase, lv)
            if got:
                rid, segs = got
                crit.append((f"方案 {s['key']}（{s['name']}），并发 {lv:g}", segs))
                summary["schemes"][s["key"]].setdefault("critical_path", {})[f"{lv:g}"] = {
                    "request_id": rid, "segments_ms": [[lab, round(x, 1), round(y, 1)] for lab, x, y in segs]}
    if crit:
        fig_critical(plt, crit, out / "figures" / "fig5_critical_path.png")
    (out / "results" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
