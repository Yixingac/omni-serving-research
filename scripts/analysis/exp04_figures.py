#!/usr/bin/env python3
"""实验 4 的图：从 exp04_report.py 写出的 results/summary.json 画。

    python scripts/analysis/exp04_figures.py --out research/01_性能瓶颈/实验04_每批处理多少任务最好

Every point is one closed-loop level of one run (concurrency 1-256); every configuration ran once.
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

COLOR = {"s2x6": "#0072B2", "s1x7": "#56B4E9", "p4": "#999999", "r8": "#000000"}


def finish(fig, path: Path, title: str) -> None:  # noqa: ANN001
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


def _xy(c: dict[str, Any], role: str, field: str, unit: str | None = None) -> tuple[list[float], list[float]]:
    pts = [((lv["eff_batch"] or {}).get(role), lv["unit"][unit] if unit else lv[field]) for lv in c["levels"]]
    pts = [(x, y) for x, y in pts if x and y]
    return [x for x, _ in pts], [y for _, y in pts]


def fig1(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.4))
    for k, c in s["curves"].items():
        x, y = _xy(c, "thinker", "", "thinker_ms_per_token")
        a.plot(x, y, marker="o", color=COLOR[k], label=c["label"])
        x2, y2 = _xy(c, "thinker", "thinker_tokens_per_s_per_replica")
        b.plot(x2, y2, marker="o", color=COLOR[k], label=c["label"])
    a.axvline(s["config_limits"]["thinker_max_running_per_replica"], color="#CC0000", linestyle="--", linewidth=0.8)
    a.text(s["config_limits"]["thinker_max_running_per_replica"] * 0.97, 30, "每个副本最多 64 个", fontsize=7, color="#CC0000", ha="right")
    a.set_ylabel("thinker 每生成一个 token 的时间（毫秒，中位数）")
    a.set_title("（a）batch 变大，每个 token 只慢一点（8 个完整副本是同卡争用）", fontsize=9, loc="left")
    b.set_ylabel("每个 thinker 副本每秒生成的 token（推算）")
    b.set_title("（b）所以每个副本的产出几乎跟着 batch 成比例增长", fontsize=9, loc="left")
    for ax in (a, b):
        ax.set_xscale("log")
        ax.set_xlabel("thinker 的有效 batch（每个副本平均同时在算的请求数）")
        ax.legend(fontsize=7, frameon=False)
    finish(fig, path, "图 1　thinker：每个点是一个并发（1–256）。数据：results/summary.json")
    plt.close(fig)


def fig2(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.4))
    for k, c in s["curves"].items():
        x, y = _xy(c, "talker", "", "talker_ms_per_audio_s")
        a.plot(x, y, marker="o", color=COLOR[k], label=c["label"])
        x2, y2 = _xy(c, "talker", "talker_audio_s_per_s_per_replica")
        b.plot(x2, y2, marker="o", color=COLOR[k], label=c["label"])
    a.set_ylabel("talker 生成 1 秒语音的时间（毫秒，中位数）")
    a.set_title("（a）batch 变大，每秒语音要的时间涨到约 3 倍（含同卡 code2wav 的争用）", fontsize=9, loc="left")
    b.set_ylabel("每个 talker 副本每秒生成的语音秒数（推算）")
    b.set_title("（b）每个副本的产出仍然在涨，只是涨得越来越慢", fontsize=9, loc="left")
    for ax in (a, b):
        ax.set_xscale("log")
        ax.set_xlabel("talker 的有效 batch")
        ax.legend(fontsize=7, frameon=False)
    finish(fig, path, "图 2　talker：每个点是一个并发（1–256）。数据：results/summary.json")
    plt.close(fig)


def fig3(plt, s: dict[str, Any], path: Path) -> None:  # noqa: ANN001
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.4))
    cap = s["config_limits"]["code2wav_max_batch_size"]
    for k, c in s["curves"].items():
        lv = [x for x in c["levels"] if x.get("code2wav_batches", {}).get("mean_size")]
        a.plot([x["concurrency"] for x in lv], [x["code2wav_batches"]["mean_size"] for x in lv], marker="o", color=COLOR[k], label=f"{c['label']}：平均")
        a.plot([x["concurrency"] for x in lv], [x["code2wav_batches"]["max_size"] for x in lv], marker=".", linestyle=":", color=COLOR[k], alpha=0.7)
        g = [x for x in c["levels"] if x.get("code2wav_ms_per_request_from_gaps")]
        if g:
            b.plot([x["code2wav_batches"]["mean_size"] for x in g], [x["code2wav_ms_per_request_from_gaps"]["median_of_replica_medians_ms"] for x in g],
                   marker="o", color=COLOR[k], label=f"{c['label']}：从相邻两批的间隔推算")
    a.axhline(cap, color="#CC0000", linestyle="--", linewidth=0.8)
    a.text(1.1, cap + 0.4, f"上限 {cap} 个", fontsize=7, color="#CC0000")
    a.set_xscale("log", base=2)
    a.set_xticks([1, 4, 16, 64, 256], ["1", "4", "16", "64", "256"])
    a.set_xlabel("闭环并发")
    a.set_ylabel("code2wav 每批的请求数（实线平均，点线最大）")
    a.set_title("（a）code2wav 实际凑成的批（服务日志）", fontsize=9, loc="left")
    low = [lv["code2wav_service_ms_p50"] for c in s["curves"].values() for lv in c["levels"]
           if lv["concurrency"] <= 8 and lv.get("code2wav_service_ms_p50")]
    b.axhspan(min(low), max(low), color="#DDDDDD", alpha=0.7,
              label=f"并发 ≤ 8（几乎每批 1 个）时一个请求的 code2wav 时间：{min(low):.0f}–{max(low):.0f} 毫秒")
    b.set_xlabel("每批平均请求数")
    b.set_ylabel("折合每个请求的时间（毫秒）")
    b.set_ylim(0, 320)
    b.set_title("（b）批再大，每个请求的时间也不降", fontsize=9, loc="left")
    for ax in (a, b):
        ax.legend(fontsize=7, frameon=False)
    finish(fig, path, "图 3　code2wav：合批不省时间（（b）只画 code2wav 一直有活干的档位；2+6 的 6 个 code2wav 写在同一个日志里，算不了间隔）。数据：results/summary.json")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    s = json.loads((out / "results" / "summary.json").read_text())
    plt = setup_matplotlib()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    for fn, name in ((fig1, "fig1_thinker"), (fig2, "fig2_talker"), (fig3, "fig3_code2wav")):
        fn(plt, s, out / "figures" / f"{name}.png")
        print(f"figures/{name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
