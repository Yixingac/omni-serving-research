#!/usr/bin/env python3
"""实验 2（不同 GPU 分工的比较）：8 张卡的六种分工在同一负载下的吞吐、延迟、SLO 和各角色 GPU 的忙碌程度。

    python scripts/analysis/exp02_placement.py --data-root ~/research-staging/unicorn \
        --out research/01_性能瓶颈/实验02_不同GPU分工的比较

Runs (directory prefixes under <data-root>/experiments):
- same absolute open-loop rates, 16-40 req/s, 90 s each (plan v2 batch 2, compute-03): 20261116_v2b_aoh_<layout>
- closed loop at concurrency 16-256 (batch 1, two runs; plus compute-03 repeats for r8 and s2x6):
  20261115_v2_a_<layout>_run{1,2}, 20261116_v2b_a_<layout>_c03
- small batch, concurrency 1-8 (batch 2): 20261116_v2b_sb_<layout>
Each needs analysis/analysis.json (analyze_experiment.py --slo ttft_s=1.0,e2e_per_audio_s=0.5) and
raw/gpu_samples.csv. Writes figures/*.png and results/summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp01_gpu_timeline import level_windows, load_util, mean_util, setup_matplotlib  # noqa: E402
from summarize_plan_v2 import layout_roles, slo_capacity  # noqa: E402

SCHEMES = [  # key, internal layout, Chinese name, thinker GPUs, talker GPUs
    ("A", "r8", "8 个完整副本", None, None),
    ("B", "p4", "4 条两卡流水线", 4, 4),
    ("C", "s2x6", "2 张 thinker 卡 + 6 张 talker 卡", 2, 6),
    ("D", "s1x7", "1 张 thinker 卡 + 7 张 talker 卡", 1, 7),
    ("E", "s3x5", "3 张 thinker 卡 + 5 张 talker 卡", 3, 5),
    ("F", "s4x4", "4 张 thinker 卡 + 4 张 talker 卡", 4, 4),
]
COLORS = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#d62728", "E": "#9467bd", "F": "#8c564b"}


def rows(exp: Path, prefix: str) -> list[dict[str, Any]]:
    try:
        a = json.load(open(exp / "analysis" / "analysis.json"))
    except OSError:
        return []
    return [r for r in a["table"] if r["phase"].startswith(prefix)]


def role_busy(exp: Path, phase: str, layout: str) -> dict[float, dict[str, float]]:
    """Mean nvidia-smi busy share per role at each load level of a phase."""
    util = load_util(exp)
    roles = layout_roles(layout)
    out = {}
    for lv, t0, t1 in level_windows(exp, phase):
        u = mean_util(util, t0, t1)
        out[lv] = {role: sum(u[g] for g in gs if g in u) / max(1, sum(1 for g in gs if g in u)) for role, gs in roles.items()}
    return out


def mem_by_role(exp: Path, layout: str) -> dict[str, float]:
    util = load_util(exp)
    return {role: max(max(m for _, _, m in util[g]) for g in gs if g in util) / 1024 for role, gs in layout_roles(layout).items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    E = Path(a.data_root).expanduser() / "experiments"
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib()
    summary: dict[str, Any] = {"generated_by": "scripts/analysis/exp02_placement.py", "schemes": {}}
    for key, layout, name, t, k in SCHEMES:
        s: dict[str, Any] = {"name": name, "layout": layout, "thinker_gpus": t, "talker_gpus": k}
        aoh = E / f"20261116_v2b_aoh_{layout}_run01"
        opens = rows(aoh, "w3b_open")
        s["open"] = [{"rate": float(r["level"]), "tp": r["tp_requests_per_s"], "slo": r["goodput_fraction"], "errors": r["errors"], "n": r["n"],
                      "e2e_p50": r["e2e_s_median"], "e2e_p90": r["e2e_s_p90"], "ttft_p90": r["ttft_s_p90"],
                      "e2e_per_audio_p90": r["e2e_per_audio_s_p90"]} for r in opens]
        cap, reached = slo_capacity([(o["rate"], o["slo"]) for o in s["open"]])
        s["slo_capacity_rps"] = cap
        s["slo_capacity_above_range"] = not reached
        s["open_role_busy"] = {f"{lv:g}": v for lv, v in role_busy(aoh, "w3b_open_abs", layout).items()} if aoh.exists() else {}
        s["mem_gib_by_role"] = mem_by_role(aoh, layout) if aoh.exists() else {}
        s["closed"] = {}
        for run in (f"20261115_v2_a_{layout}_run1_run01", f"20261115_v2_a_{layout}_run2_run01", f"20261116_v2b_a_{layout}_c03_run01"):
            rr = rows(E / run, "w3b_closed")
            if rr:
                s["closed"][run] = [{"c": int(r["level"]), "tp": r["tp_requests_per_s"], "e2e_p50": r["e2e_s_median"], "e2e_p90": r["e2e_s_p90"],
                                     "ttft_p50": r["ttft_s_median"], "ttft_p90": r["ttft_s_p90"], "slo": r["goodput_fraction"]} for r in rr]
        sb = rows(E / f"20261116_v2b_sb_{layout}_run01", "w3b_small")
        s["small"] = [{"c": int(r["level"]), "tp": r["tp_requests_per_s"], "e2e_p50": r["e2e_s_median"], "e2e_p90": r["e2e_s_p90"],
                       "ttft_p50": r["ttft_s_median"], "slo": r["goodput_fraction"]} for r in sb]
        summary["schemes"][key] = s

    # Figure 1: SLO share against the offered rate (same absolute rates for every layout).
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for key, s in summary["schemes"].items():
        o = s["open"]
        if not o:
            continue
        axes[0].plot([x["rate"] for x in o], [x["slo"] * 100 for x in o], marker="o", color=COLORS[key], label=f"方案 {key}：{s['name']}")
        axes[1].plot([x["rate"] for x in o], [x["e2e_p90"] for x in o], marker="o", color=COLORS[key], label=f"方案 {key}")
    axes[0].axhline(95, color="gray", ls="--", lw=1)
    axes[0].text(16.2, 96, "95%：达标线", fontsize=8, color="gray")
    axes[0].set_xlabel("每秒发来的请求数（开环，按泊松分布到达）", fontsize=8)
    axes[0].set_ylabel("满足 SLO 的请求比例 %", fontsize=8)
    axes[0].legend(fontsize=7, loc="lower left")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("每秒发来的请求数", fontsize=8)
    axes[1].set_ylabel("端到端时间 p90（秒，对数刻度）", fontsize=8)
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("图 1　同样的到达速率下，每种分工能扛住多少（plan v2 第二批，compute-03，每档 90 秒）", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "figures" / "fig1_slo_vs_rate.png")
    plt.close(fig)

    # Figure 2: throughput-latency curves from the closed loop (first run) plus small batch.
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for key, s in summary["schemes"].items():
        run1 = next(iter(s["closed"].values()), [])
        pts = sorted(s["small"] + run1, key=lambda x: x["c"])
        if not pts:
            continue
        ax.plot([p["tp"] for p in pts], [p["e2e_p50"] for p in pts], marker="o", color=COLORS[key], label=f"方案 {key}：{s['name']}")
        for p in pts:
            if p["c"] in (8, 64, 256):
                ax.annotate(f"{p['c']}", (p["tp"], p["e2e_p50"]), fontsize=6, xytext=(3, 3), textcoords="offset points", color=COLORS[key])
    ax.set_xlabel("吞吐（每秒完成的请求数）", fontsize=8)
    ax.set_ylabel("端到端时间中位数（秒）", fontsize=8)
    ax.set_title("图 2　吞吐和延迟的取舍（闭环；点旁的数字是并发；越靠右下越好）", fontsize=10, loc="left")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "fig2_throughput_latency.png")
    plt.close(fig)

    # Figure 3: which role's GPUs fill up first as the rate grows.
    fig, axes = plt.subplots(1, 6, figsize=(17, 3.4), sharey=True)
    for ax, (key, s) in zip(axes, summary["schemes"].items()):
        rb = s["open_role_busy"]
        rates = sorted(float(r) for r in rb)
        for role, ls in (("thinker", "-"), ("talker", "--"), ("full replica", "-")):
            vals = [rb[f"{r:g}"].get(role) for r in rates]
            if any(v is not None for v in vals):
                ax.plot(rates, vals, ls, marker="o", label={"thinker": "thinker 卡", "talker": "talker 卡", "full replica": "完整副本"}[role])
        if s["slo_capacity_rps"]:
            ax.axvline(s["slo_capacity_rps"], color="gray", ls=":", lw=1)
        ax.set_title(f"方案 {key}：{s['name']}", fontsize=8)
        ax.set_xlabel("请求/秒", fontsize=7)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, loc="lower right")
    axes[0].set_ylabel("平均忙碌比例 %（nvidia-smi）", fontsize=8)
    fig.suptitle("图 3　速率升高时，哪种卡先忙满（点线 = 这种分工满足 SLO 的最高速率）", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "figures" / "fig3_role_busy_vs_rate.png")
    plt.close(fig)

    (out / "results" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
