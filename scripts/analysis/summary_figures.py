#!/usr/bin/env python3
"""Summary figures for the Job 409712 report, drawn from the per-experiment analysis.json files.

    python scripts/analysis/summary_figures.py --data-root ~/research/data/unicorn --out reports/experiment_reports/figures

Each figure uses the latest run directory of the named experiments under <data-root>/experiments
(``20261111_<name>_runNN``). Missing experiments are skipped and listed on stdout; requests that
failed (e.g. after a server crash) are not in the medians, and levels with failures are marked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from omni_research.analysis.plots import _plt  # noqa: E402


def load(data_root: Path, name: str) -> dict | None:
    runs = sorted((data_root / "experiments").glob(f"*_{name}_run*"))
    runs = [r for r in runs if (r / "analysis" / "analysis.json").exists()]
    if not runs:
        print(f"missing: {name}")
        return None
    return json.loads((runs[-1] / "analysis" / "analysis.json").read_text())


def rows(a: dict | None, phase: str) -> list[dict]:
    if a is None:
        return []
    out = [r for r in a["table"] if r["phase"] == phase]
    return sorted(out, key=lambda r: float(r["level"]))


def placement(data_root: Path, out: Path) -> Path:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    series = [("1 卡（x3）", "x3_minicpm_speech_1gpu", "w3b"),
              ("2 卡：两个副本（x7a）", "x7a_minicpm_2gpu_two_replicas", "w3b"),
              ("2 卡：两个副本，重复", "x7a_minicpm_2gpu_two_replicas_rep2", "w3b"),
              ("2 卡：拆分（x7b）", "x7b_minicpm_2gpu_disagg", "w3b"),
              ("2 卡：拆分，重复", "x7b_minicpm_2gpu_disagg_rep2", "w3b"),
              ("2 卡：拆分，开事件记录（x7e）", "x7e_minicpm_2gpu_disagg_events", "w3b_events"),
              ("2 卡：只复制 talker，重复", "x7c_minicpm_2gpu_talker_replica_rep2", "w3b"),
              ("3 卡：thinker + 两个 talker（x7f）", "x7f_minicpm_3gpu_split_talker2", "w3b_events")]
    for label, name, phase in series:
        # A level with fewer requests than its concurrency is a single burst, not a steady closed loop.
        rs = [r for r in rows(load(data_root, name), phase) if not r.get("errors") and r["n"] >= float(r["level"])]
        if not rs:
            continue
        x = [float(r["level"]) for r in rs]
        style = ":" if "3 卡" in label else ("--" if ("重复" in label or "x7e" in label) else "-")
        axes[0].plot(x, [r.get("e2e_per_audio_s_median") for r in rs], style, marker="o", label=label)
        axes[1].plot(x, [r.get("tp_output_audio_s_per_s") for r in rs], style, marker="o", label=label)
    axes[0].set_title("端到端 ÷ 输出语音时长（中位数，越低越好）")
    axes[1].set_title("每秒生成的语音秒数（越高越好）")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("总并发")
        ax.legend(fontsize=7)
    fig.suptitle("MiniCPM-o 4.5，W3b 文本提问、语音回答，RTX A6000（每条线是一次运行）", fontsize=9)
    fig.tight_layout()
    p = out / "placement_2gpu_w3b.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def saturation(data_root: Path, out: Path) -> Path:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    x2 = load(data_root, "x2_minicpm_text_1gpu")
    text = rows(x2, "w1_low") + rows(x2, "w1_high") + rows(load(data_root, "f1_minicpm_text_saturation"), "w1_c128_256")
    speech = rows(load(data_root, "x3_minicpm_speech_1gpu"), "w3b") + rows(load(data_root, "f2_minicpm_speech_saturation"), "w3b_c32_64")
    for ax, rs, title, key, unit in ((axes[0], text, "文本（W1，--text-only）", "tp_requests_per_s", "请求/秒"),
                                     (axes[1], speech, "语音输出（W3b）", "tp_requests_per_s", "请求/秒")):
        if not rs:
            continue
        x = [float(r["level"]) for r in rs]
        ax.plot(x, [r.get(key) for r in rs], marker="o", color="tab:blue", label=f"吞吐（{unit}）")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("闭环并发")
        ax.set_ylabel(unit)
        ax2 = ax.twinx()
        ax2.plot(x, [r.get("e2e_s_median") for r in rs], marker="s", color="tab:red", label="端到端中位数（秒）")
        ax2.set_ylabel("秒")
        ax2.grid(False)
        ax.set_title(title)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    fig.suptitle("单张 RTX A6000 的饱和点（MiniCPM-o 4.5；文本约 12 请求/秒，语音约 2.9 请求/秒）", fontsize=9)
    fig.tight_layout()
    p = out / "saturation_1gpu.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def replay(data_root: Path, out: Path) -> Path:
    plt = _plt()
    a = load(data_root, "f4_minicpm_replay_large")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    if a is not None:
        by = {(r["phase"]): r for r in a["table"]}
        speeds = [1, 2, 4, 8]
        for mode, label in (("req", "请求级（开环）"), ("dep", "依赖感知（闭环）")):
            rs = [by.get(f"replay_{mode}_speed{s}") for s in speeds]
            axes[0].plot(speeds, [r and r.get("e2e_s_median") for r in rs], marker="o", label=label)
        axes[0].set_xscale("log", base=2)
        axes[0].set_xlabel("回放加速倍数")
        axes[0].set_ylabel("请求延迟中位数（秒）")
        axes[0].set_title("同一份 Agent trace 的两种回放（f4）")
        axes[0].legend(fontsize=7)
        pc = a.get("prefix_cache") or {}
        order = [p["phase"] for p in a["result"]["phases"]] if a.get("result") else list(pc)
        names = [p for p in order if p in pc]
        axes[1].bar(range(len(names)), [100 * (pc[n]["cached_fraction"] or 0) for n in names],
                    color=["tab:orange" if i == 0 else "tab:gray" for i in range(len(names))])
        axes[1].set_xticks(range(len(names)), [n.replace("replay_", "").replace("_speed", "×") for n in names], rotation=45, fontsize=7)
        axes[1].set_ylim(80, 100)
        axes[1].set_ylabel("前缀缓存命中的输入 token（%）")
        axes[1].set_title("按运行顺序：只有第一段是冷缓存")
    fig.suptitle("回放方式和回放顺序都会改变测量结果（RTX A6000，MiniCPM-o 4.5）", fontsize=9)
    fig.tight_layout()
    p = out / "replay_modes_and_cache.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data_root, out = Path(a.data_root).expanduser(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (placement, saturation, replay):
        print(f(data_root, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
