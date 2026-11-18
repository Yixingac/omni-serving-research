"""Figures for one experiment. Every figure is drawn from measured records and names its source."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from . import metrics as M


# Chinese-capable fonts, in order of preference: Spark/Ubuntu (Noto), macOS, generic fallbacks.
CJK_FONTS = ["Noto Sans CJK SC", "PingFang SC", "Hiragino Sans GB", "Heiti SC", "STHeiti", "Droid Sans Fallback",
             "WenQuanYi Micro Hei", "AR PL UMing CN"]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    cjk = [f for f in CJK_FONTS if f in available]
    plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9,
                         "font.sans-serif": cjk[:1] + ["DejaVu Sans"], "axes.unicode_minus": False})
    return plt


def latency_vs_concurrency(table: list[dict[str, Any]], out: Path, title: str) -> list[Path]:
    """Median e2e/TTFT/TTFA (with p90 where the sample size allows) and throughput per phase."""
    plt = _plt()
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in table:
        if r["level_kind"] == "concurrency":
            by_phase[r["phase"]].append(r)
    paths = []
    for phase, rows in by_phase.items():
        rows.sort(key=lambda r: float(r["level"]))
        if len(rows) < 2:
            continue
        x = [float(r["level"]) for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        for metric, label in (("e2e_s", "端到端"), ("ttft_s", "首 token"), ("ttfa_s", "首个音频块")):
            ys = [r.get(f"{metric}_median") for r in rows]
            if all(y is None for y in ys):
                continue
            axes[0].plot(x, ys, marker="o", label=f"{label} 中位数")
            p90 = [r.get(f"{metric}_p90") for r in rows]
            if any(v is not None for v in p90):
                axes[0].plot(x, p90, ls="--", marker=".", alpha=0.6, label=f"{label} p90")
        axes[0].set_xscale("log", base=2)
        axes[0].set_xlabel("并发数")
        axes[0].set_ylabel("秒")
        axes[0].set_title("延迟")
        axes[0].legend(fontsize=7)
        tps = [r.get("tp_requests_per_s") for r in rows]
        axes[1].plot(x, tps, marker="o", color="tab:green", label="请求/秒")
        toks = [r.get("tp_output_tokens_per_s") for r in rows]
        ax2 = axes[1].twinx()
        ax2.plot(x, toks, marker="s", color="tab:purple", alpha=0.6, label="生成 token/秒")
        ax2.set_ylabel("token/秒")
        axes[1].set_xscale("log", base=2)
        axes[1].set_xlabel("并发数")
        axes[1].set_ylabel("请求/秒")
        axes[1].set_title("吞吐")
        fig.suptitle(f"{title} · {phase}（每点 n 见报告表格；p90 仅在 n≥50 时画出）", fontsize=9)
        fig.tight_layout()
        p = out / f"latency_throughput_{phase}.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)
    return paths


def gpu_timeline(samples: list[dict[str, Any]], phases: list[dict[str, Any]], out: Path, title: str) -> Path | None:
    if not samples:
        return None
    plt = _plt()
    t0 = min(s["t"] for s in samples)
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True)
    by_gpu: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        by_gpu[s["gpu"]].append(s)
    any_mem = False
    for gpu, ss in sorted(by_gpu.items()):
        ss.sort(key=lambda s: s["t"])
        axes[0].plot([s["t"] - t0 for s in ss], [s["util"] for s in ss], lw=0.7, label=f"GPU {gpu}")
        mem = [(s["t"] - t0, s["mem_mib"] / 1024) for s in ss if s.get("mem_mib") is not None]
        if mem:
            any_mem = True
            axes[1].plot([m[0] for m in mem], [m[1] for m in mem], lw=0.8, label=f"GPU {gpu}")
    if not any_mem:
        axes[1].text(0.5, 0.5, "nvidia-smi 不报告显存（统一内存，例如 GB10）", ha="center", va="center", transform=axes[1].transAxes)
    for ph in phases:
        if ph.get("started") and ph.get("ended"):
            for ax in axes:
                ax.axvspan(ph["started"] - t0, ph["ended"] - t0, alpha=0.08)
            axes[0].text(ph["started"] - t0, 102, ph.get("phase", ""), fontsize=6, rotation=20)
    axes[0].set_ylabel("采样利用率 %")
    axes[0].set_ylim(0, 115)
    axes[1].set_ylabel("显存 GiB")
    axes[1].set_xlabel("秒（相对第一次采样）")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"{title}：nvidia-smi 采样（每 200 ms；“利用率”= 采样周期内有 kernel 运行的时间比例）", fontsize=9)
    fig.tight_layout()
    p = out / "gpu_timeline.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def request_gantt(records: list[dict[str, Any]], out: Path, title: str, phase: str, limit: int = 40) -> Path | None:
    """Client-side request timeline: send -> first token -> first audio -> done, per request."""
    recs = [r for r in records if r.get("phase") == phase and r.get("sent_ns") and r.get("done_ns") and not r.get("warmup")
            and not r.get("crash_affected")]
    if not recs:
        return None
    recs.sort(key=lambda r: r["sent_ns"])
    recs = recs[:limit]
    t0 = recs[0]["sent_ns"]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 0.18 * len(recs) + 1.2))
    for i, r in enumerate(recs):
        s = (r["sent_ns"] - t0) / 1e9
        e = (r["done_ns"] - t0) / 1e9
        ax.barh(i, e - s, left=s, color="tab:blue" if r.get("status") == "ok" else "tab:red", alpha=0.35)
        if r.get("first_text_ns"):
            ax.plot((r["first_text_ns"] - t0) / 1e9, i, "k|", ms=6)
        if r.get("first_audio_ns"):
            ax.plot((r["first_audio_ns"] - t0) / 1e9, i, "m|", ms=8)
    ax.invert_yaxis()
    ax.set_xlabel("秒")
    ax.set_ylabel("请求（按发出时间）")
    ax.set_title(f"{title} · {phase}：请求时间线（蓝条=发出到完成，黑线=首 token，紫线=首个音频块；红=失败）", fontsize=8)
    fig.tight_layout()
    p = out / f"request_timeline_{phase}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def stage_breakdown_bars(stages: dict[str, Any], out: Path, title: str) -> Path | None:
    rows = []
    for phase, st in stages.items():
        for label, s in st["summary_ms"].items():
            if s.get("n"):
                rows.append((phase, label, s.get("median"), s.get("n")))
    if not rows:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(rows) + 1.2))
    names = [f"{p} · {lab} (n={n})" for p, lab, _, n in rows]
    ax.barh(range(len(rows)), [m for _, _, m, _ in rows], color="tab:orange", alpha=0.7)
    ax.set_yticks(range(len(rows)), names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("毫秒（中位数）")
    ax.set_title(f"{title}：服务端阶段耗时（SGLang-Omni request event recorder）", fontsize=9)
    fig.tight_layout()
    p = out / "stage_breakdown.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def agent_time_split(agents: dict[str, Any], out: Path, title: str) -> Path | None:
    if not agents:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(agents) + 1.2))
    keys = ["model", "tool", "think", "other"]
    labels = {"model": "等模型", "tool": "工具", "think": "用户思考（合成）", "other": "Agent 处理与间隙"}
    for i, (name, a) in enumerate(agents.items()):
        split = a["time_split_s"]
        total = split["session"] or 1
        left = 0.0
        for k in keys:
            w = split[k] / total
            ax.barh(i, w, left=left, label=labels[k] if i == 0 else None, alpha=0.8)
            left += w
    ax.set_yticks(range(len(agents)), list(agents), fontsize=7)
    ax.set_xlabel("占会话总时长的比例")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"{title}：Agent 会话时间拆分（区间并集）", fontsize=9)
    fig.tight_layout()
    p = out / "agent_time_split.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def make_all(analysis: dict[str, Any], exp_dir: Path, records: list[dict[str, Any]], samples: list[dict[str, Any]]) -> list[Path]:
    out = exp_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    title = analysis["config"].get("experiment_id", exp_dir.name)
    paths: list[Path] = []
    paths += latency_vs_concurrency(analysis["table"], out, title)
    g = gpu_timeline(samples, analysis["result"].get("phases", []), out, title)
    if g:
        paths.append(g)
    for ph in analysis["result"].get("phases", []):
        if ph.get("kind") in ("closed_loop", "open_loop", "replay"):
            p = request_gantt(records, out, title, ph["phase"])
            if p:
                paths.append(p)
    s = stage_breakdown_bars(analysis["stages"], out, title)
    if s:
        paths.append(s)
    a = agent_time_split(analysis["agents"], out, title)
    if a:
        paths.append(a)
    _ = M
    return paths
