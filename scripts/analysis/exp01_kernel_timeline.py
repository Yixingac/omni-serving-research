#!/usr/bin/env python3
"""实验 1（kernel 级）：从 torch profiler trace 统计每张 GPU、每个阶段的忙闲和空隙，画 kernel 时间线。

    python scripts/analysis/exp01_kernel_timeline.py --data-root ~/research-staging/unicorn \
        --run A=20261117_exp01_prof_r8_run01 --run B=20261117_exp01_prof_p4_run01 --run C=20261117_exp01_prof_s2x6_run01 \
        --out research/01_性能瓶颈/实验01_8张GPU在做什么

Each trace file under profiling/torch/<phase>/<server>/ is one process: "preprocessing*" is the
pipeline process that runs the thinker, then "talker*" and "code2wav*". GPU activity = kernel,
gpu_memcpy and gpu_memset events. A trace's ts (microseconds) plus its baseTimeNanoseconds is unix
time in ns, the clock the server events use. The device in a trace is the process's CUDA device
index; a server sees only its own GPUs, so the physical GPU is the server's gpus[device].

Per phase and GPU: the share of the window with any GPU activity (busy), per stage, the share with
two or more processes in flight at once (contention: without MPS their kernels take turns), and the
idle time split by gap length. Per thinker process: idle gaps longer than 1 ms, and for each such gap
whether the thinker had requests in its model path at the time (server events) -- idle with work
waiting points at the CPU side; idle without work means nothing to do.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp01_gpu_timeline import load_util  # noqa: E402
from omni_research.analysis.experiment_analysis import read_jsonl  # noqa: E402

DEVICE_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
GAP_BINS_MS = [(0, 0.01, "< 10 微秒"), (0.01, 0.1, "10–100 微秒"), (0.1, 1, "0.1–1 毫秒"), (1, 10, "1–10 毫秒"), (10, 1e12, "> 10 毫秒")]
STAGE_COLORS = {"thinker": "#d94801", "talker": "#41ab5d", "code2wav": "#6a51a3"}
SCHEME_NAMES = {"r8": "8 个完整副本", "p4": "4 条两卡流水线", "s2x6": "2 张 thinker 卡 + 6 张 talker 卡"}


def stage_of(name: str) -> tuple[str, str]:
    """'preprocessing@r1_trace_pid1_rank0.trace.json.gz' -> ('thinker', 'r1'); 'talker_trace...' -> ('talker', '')."""
    head = name.split("_trace")[0]
    stage, _, rep = head.partition("@")
    # The pipeline process runs preprocessing, the thinker and text decode; its file is named after
    # whichever of those stages started first.
    return ("thinker" if stage in ("preprocessing", "decode", "thinker", "audio_encoder", "image_encoder") else stage), rep


def union(iv: np.ndarray) -> np.ndarray:
    """Sorted, merged [start, end] rows."""
    if len(iv) == 0:
        return iv.reshape(0, 2)
    iv = iv[np.argsort(iv[:, 0])]
    out = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return np.array(out)


def clip(iv: np.ndarray, a: float, b: float) -> np.ndarray:
    if len(iv) == 0:
        return iv
    iv = iv[(iv[:, 1] > a) & (iv[:, 0] < b)].copy()
    iv[:, 0] = np.maximum(iv[:, 0], a)
    iv[:, 1] = np.minimum(iv[:, 1], b)
    return iv


def total(iv: np.ndarray) -> float:
    return float((iv[:, 1] - iv[:, 0]).sum()) if len(iv) else 0.0


def intersect(a: np.ndarray, b: np.ndarray) -> float:
    """Total overlap of two unions."""
    i = j = 0
    t = 0.0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i, 0], b[j, 0]), min(a[i, 1], b[j, 1])
        if hi > lo:
            t += hi - lo
        if a[i, 1] < b[j, 1]:
            i += 1
        else:
            j += 1
    return t


def load_process(path: Path, server_gpus: list[int]) -> dict[str, Any]:
    with gzip.open(path, "rt") as fh:
        tr = json.load(fh)
    base = int(tr.get("baseTimeNanoseconds") or 0)
    by_dev: dict[int, list[tuple[float, float]]] = defaultdict(list)
    t_first = t_last = None
    for ev in tr.get("traceEvents", []):
        if ev.get("ph") == "X" and "ts" in ev:
            s0 = base + float(ev["ts"]) * 1000.0
            e0 = s0 + float(ev.get("dur", 0)) * 1000.0
            t_first = s0 if t_first is None else min(t_first, s0)
            t_last = e0 if t_last is None else max(t_last, e0)
        if ev.get("ph") != "X" or ev.get("cat") not in DEVICE_CATS:
            continue
        dev = (ev.get("args") or {}).get("device")
        if dev is None:
            continue
        s = base + float(ev["ts"]) * 1000.0
        by_dev[int(dev)].append((s, s + float(ev.get("dur", 0)) * 1000.0))
    stage, rep = stage_of(path.name)
    gpus = {}
    for dev, iv in by_dev.items():
        phys = server_gpus[dev] if dev < len(server_gpus) else dev
        gpus[phys] = union(np.array(iv))
    return {"file": path.name, "server": path.parent.name, "stage": stage, "replica": rep, "gpus": gpus, "span": (t_first, t_last)}


def server_gpus(exp: Path) -> dict[str, list[int]]:
    cfg = json.load(open(exp / "config.json"))
    return {s["name"]: list(s["gpus"]) for s in cfg["servers"]}


def thinker_inflight(exp: Path, phase: str) -> dict[tuple[str, str], np.ndarray]:
    """Per (server, thinker replica): [start, end] of every request's thinker model path (unix ns)."""
    base = exp / "raw" / "server_events" / phase
    out: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for sdir in sorted(p for p in base.iterdir() if p.is_dir()) if base.exists() else []:
        first: dict[tuple[str, str], int] = {}
        last: dict[tuple[str, str], int] = {}
        for f in sdir.glob("*.jsonl"):
            for e in read_jsonl(f):
                st = e.get("stage") or ""
                if not st.startswith("thinker"):
                    continue
                key = (e.get("request_id"), st)
                t = e.get("timestamp_ns")
                if e.get("event_name") == "model_path_start":
                    first[key] = min(first.get(key, t), t)
                elif e.get("event_name") == "model_path_end":
                    last[key] = max(last.get(key, t), t)
        for (rid, st), t0 in first.items():
            if (rid, st) in last:
                rep = st.partition("@")[2]
                out[(sdir.name, rep)].append((t0, last[(rid, st)]))
    return {k: np.array(sorted(v)) for k, v in out.items()}


WORK_EVENTS = {"thinker": ("model_path_start", "model_path_end"), "talker": ("model_path_start", "model_path_end"),
               "code2wav": ("stage_dispatch", "stage_complete")}


def stage_work(exp: Path, phase: str) -> dict[tuple[str, str, str], np.ndarray]:
    """(server, stage, replica) -> union of the intervals during which that stage had a request in
    progress: thinker and talker from model path start to end, code2wav from dispatch to completion."""
    base = exp / "raw" / "server_events" / phase
    out: dict[tuple[str, str, str], np.ndarray] = {}
    for sdir in sorted(p for p in base.iterdir() if p.is_dir()) if base.exists() else []:
        first: dict[tuple[str, str], int] = {}
        last: dict[tuple[str, str], int] = {}
        for f in sdir.glob("*.jsonl"):
            for e in read_jsonl(f):
                st = e.get("stage") or ""
                stage, _, rep = st.partition("@")
                if stage not in WORK_EVENTS:
                    continue
                a_ev, b_ev = WORK_EVENTS[stage]
                k = (e.get("request_id"), st)
                t = e.get("timestamp_ns")
                if e.get("event_name") == a_ev:
                    first[k] = min(first.get(k, t), t)
                elif e.get("event_name") == b_ev:
                    last[k] = max(last.get(k, t), t)
        per: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
        for (rid, st), t0 in first.items():
            if (rid, st) in last and last[(rid, st)] >= t0:
                stage, _, rep = st.partition("@")
                per[(sdir.name, stage, rep)].append((t0, last[(rid, st)]))
        for k, v in per.items():
            out[k] = union(np.array(v, dtype=float))
    return out


def subtract_total(a_iv: np.ndarray, b_iv: np.ndarray) -> float:
    """Total length of a minus b (both unions)."""
    return total(a_iv) - intersect(a_iv, b_iv)


def complement(iv: np.ndarray, a: float, b: float) -> np.ndarray:
    """Gaps of a union within [a, b]."""
    edges = np.concatenate([[a], iv.ravel(), [b]]) if len(iv) else np.array([a, b])
    g = edges.reshape(-1, 2)
    return g[g[:, 1] > g[:, 0]]


def busy_count_at(iv: np.ndarray, t: float) -> int:
    return int(((iv[:, 0] <= t) & (iv[:, 1] >= t)).sum()) if len(iv) else 0


def analyze_phase(exp: Path, phase: str) -> dict[str, Any]:
    sg = server_gpus(exp)
    # A process with several stages (the pipeline process) can write one file per stage name; only one
    # of them holds the recording, the others cover a few milliseconds. Keep the largest file per pid.
    files: dict[tuple[str, str], Path] = {}
    for f in sorted((exp / "profiling" / "torch" / phase).rglob("*.trace.json*")):
        m = re.search(r"_pid(\d+)_", f.name)
        key = (f.parent.name, m.group(1) if m else f.name)
        if key not in files or f.stat().st_size > files[key].stat().st_size:
            files[key] = f
    procs = [load_process(f, sg.get(f.parent.name, [])) for f in files.values()]
    # The recording window every process covers (all events, not just GPU work: a process that has
    # nothing to do at the end of the window is idle, not outside it).
    a = max(p["span"][0] for p in procs if p["span"][0] is not None)
    b = min(p["span"][1] for p in procs if p["span"][1] is not None)
    win = b - a
    per_gpu: dict[int, dict[str, Any]] = {}
    gpus = sorted({g for p in procs for g in p["gpus"]})
    work = stage_work(exp, phase)
    gpu_work: dict[int, list[np.ndarray]] = defaultdict(list)
    for p in procs:
        w = work.get((p["server"], p["stage"], p["replica"]))
        if w is not None:
            for g in p["gpus"]:
                gpu_work[g].append(clip(w, a, b))
    for g in gpus:
        stage_iv: dict[str, list[np.ndarray]] = defaultdict(list)
        proc_iv = []
        for p in procs:
            if g in p["gpus"]:
                c = clip(p["gpus"][g], a, b)
                stage_iv[p["stage"]].append(c)
                proc_iv.append(c)
        allu = union(np.vstack(proc_iv)) if proc_iv else np.zeros((0, 2))
        stages = {s: union(np.vstack(v)) for s, v in stage_iv.items()}
        # Contention: time with >= 2 processes in flight on this GPU.
        multi = 0.0
        for i in range(len(proc_iv)):
            for j in range(i + 1, len(proc_iv)):
                multi += intersect(proc_iv[i], proc_iv[j])
        gaps = np.diff(np.concatenate([[a], allu.ravel(), [b]]))[::2] / 1e6 if len(allu) else np.array([win / 1e6])
        gap_bins = {lab: float(gaps[(gaps >= lo) & (gaps < hi)].sum()) / (win / 1e6) for lo, hi, lab in GAP_BINS_MS}
        # Idle time while a stage on this GPU had a request in progress (the GPU waited on the CPU side:
        # scheduling, launches, copies, synchronization) versus idle with nothing to do.
        idle = complement(allu, a, b)
        w = union(np.vstack(gpu_work[g])) if gpu_work.get(g) else np.zeros((0, 2))
        idle_with_work = intersect(idle, w) if len(w) else 0.0
        per_gpu[g] = {"busy": total(allu) / win, "by_stage": {s: total(v) / win for s, v in stages.items()},
                      "multi_process": multi / win, "idle_by_gap_length": gap_bins,
                      "idle_with_work": idle_with_work / win, "idle_without_work": (total(idle) - idle_with_work) / win,
                      "contended": min(multi, total(allu)) / win}
    # Thinker processes: long gaps (> 1 ms) and whether requests were waiting in the thinker meanwhile.
    inflight = thinker_inflight(exp, phase)
    thinker_gaps = {}
    for p in procs:
        if p["stage"] != "thinker":
            continue
        for g, iv in p["gpus"].items():
            u = clip(iv, a, b)
            gaps = np.column_stack([u[:-1, 1], u[1:, 0]])
            long = gaps[(gaps[:, 1] - gaps[:, 0]) > 1e6]
            work = inflight.get((p["server"], p["replica"]), np.zeros((0, 2)))
            with_work = sum((e - s) for s, e in long if busy_count_at(work, (s + e) / 2) > 0)
            no_work = sum((e - s) for s, e in long if busy_count_at(work, (s + e) / 2) == 0)
            n_inflight = [busy_count_at(work, t) for t in np.linspace(a, b, 400)] if len(work) else []
            thinker_gaps[f"{p['server']}/{p['replica'] or 'thinker'}@GPU{g}"] = {
                "gpu": g, "busy": total(u) / win, "long_gaps": int(len(long)),
                "long_gap_ms_p50": float(np.median((long[:, 1] - long[:, 0]) / 1e6)) if len(long) else None,
                "idle_long_with_work": with_work / win, "idle_long_without_work": no_work / win,
                "requests_in_thinker_mean": float(np.mean(n_inflight)) if n_inflight else None}
    # nvidia-smi over exactly the same window, to compare the two ways of measuring "busy".
    util = load_util(exp)
    for g in per_gpu:
        v = [u for t, u, _ in util.get(g, []) if a / 1e9 <= t <= b / 1e9]
        per_gpu[g]["nvidia_smi_util"] = sum(v) / len(v) / 100 if v else None
    return {"window_s": win / 1e9, "window_unix_ns": [a, b], "gpus": per_gpu, "thinker": thinker_gaps,
            "processes": [{"server": p["server"], "stage": p["stage"], "replica": p["replica"], "gpus": sorted(p["gpus"])} for p in procs]}, procs


STEP_MARK = "create_flashinfer_kv_indices"  # one launch per thinker forward pass (attention metadata)


def thinker_steps(path: Path) -> dict[str, Any] | None:
    """Per forward pass of a thinker process: period (one step marker to the next), GPU time inside it
    and idle time inside it; and the kernels (or copies) that most idle time follows."""
    with gzip.open(path, "rt") as fh:
        tr = json.load(fh)
    k = sorted((e for e in tr.get("traceEvents", []) if e.get("ph") == "X" and e.get("cat") in DEVICE_CATS), key=lambda e: e["ts"])
    if not k:
        return None
    graph = sum(1 for e in k if (e.get("args") or {}).get("graph id", 0))
    marks = [i for i, e in enumerate(k) if e["name"].startswith(STEP_MARK)]
    period, busy, idle = [], [], []
    after: dict[str, float] = defaultdict(float)
    for i0, i1 in zip(marks, marks[1:]):
        seg = k[i0:i1]
        period.append(k[i1]["ts"] - k[i0]["ts"])
        busy.append(sum(e["dur"] for e in seg))
        idle.append(period[-1] - busy[-1])
        for j, e in enumerate(seg):
            nxt = seg[j + 1]["ts"] if j + 1 < len(seg) else k[i1]["ts"]
            gap = nxt - (e["ts"] + e["dur"])
            if gap > 100:  # microseconds
                name = e["name"]
                kind = ("拷贝：GPU→CPU" if "DtoH" in name else "拷贝：CPU→GPU" if "HtoD" in name else "拷贝：GPU 内部" if "DtoD" in name
                        else "准备注意力索引" if name.startswith(STEP_MARK) else "graph 之外的小 kernel（输入准备、采样等）"
                        if not (e.get("args") or {}).get("graph id", 0) else "CUDA graph 里的 kernel")
                after[kind] += gap
    if not period:
        return None
    med = lambda v: float(np.median(v))  # noqa: E731
    tot = sum(after.values()) or 1.0
    return {"steps": len(period), "kernels": len(k), "graph_kernel_share": graph / len(k),
            "period_ms_p50": med(period) / 1e3, "busy_ms_p50": med(busy) / 1e3, "idle_ms_p50": med(idle) / 1e3,
            "idle_share_p50": med([i / p for i, p in zip(idle, period)]),
            "idle_after_share": {kk: v / tot for kk, v in sorted(after.items(), key=lambda x: -x[1])}}


def fig_steps(plt, rows: list[tuple[str, dict[str, Any]]], path: Path) -> None:  # noqa: ANN001
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(rows) + 1.5))
    for i, (label, st) in enumerate(rows):
        ax.barh(i, st["busy_ms_p50"], color="#d94801", label="GPU 在算" if i == 0 else None)
        ax.barh(i, st["idle_ms_p50"], left=st["busy_ms_p50"], color="#fdd0a2", label="GPU 空着（等 CPU 那边）" if i == 0 else None)
        ax.text(st["busy_ms_p50"] + st["idle_ms_p50"] + 0.3, i, f"{st['busy_ms_p50']:.1f} + {st['idle_ms_p50']:.1f} 毫秒", va="center", fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("thinker 每一步（生成一个 token）的时间，毫秒，中位数", fontsize=8)
    ax.set_xlim(0, max(r[1]["busy_ms_p50"] + r[1]["idle_ms_p50"] for r in rows) * 1.25)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("图 9　thinker 每生成一个 token：GPU 算了多久，又空了多久", fontsize=10, loc="left")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def role_label(stages: set[str]) -> str:
    if {"thinker", "talker", "code2wav"} <= stages:
        return "全部\n三段"
    if stages == {"talker", "code2wav"}:
        return "talker\n+c2w"
    return "+".join(sorted(stages))


def setup_matplotlib():  # noqa: ANN201
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Hiragino Sans GB", "Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    return plt


def fig_kernel_timeline(plt, panels: list[tuple[str, list[dict], tuple[float, float]]], path: Path, span_ms: float = 300.0) -> None:  # noqa: ANN001
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 2.4 * len(panels)), squeeze=False)
    for ax, (title, procs, (a, b)) in zip(axes[:, 0], panels):
        t0 = a + (b - a) / 2 - span_ms * 1e6 / 2  # a window in the middle of the phase
        t1 = t0 + span_ms * 1e6
        for p in procs:
            for g, iv in p["gpus"].items():
                c = clip(iv, t0, t1)
                if len(c):
                    off = {"thinker": -0.25, "talker": 0.0, "code2wav": 0.25}.get(p["stage"], 0)
                    ax.broken_barh([((s - t0) / 1e6, (e - s) / 1e6) for s, e in c], (g + off - 0.11, 0.22),
                                   facecolors=STAGE_COLORS.get(p["stage"], "#999999"), linewidth=0)
        ax.set_ylim(7.6, -0.6)
        ax.set_xlim(0, span_ms)
        ax.set_yticks(range(8))
        ax.set_yticklabels([f"GPU {g}" for g in range(8)], fontsize=8)
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(axis="x", alpha=0.3)
    axes[-1, 0].set_xlabel("时间（毫秒，取这一段中间的 300 毫秒）", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in STAGE_COLORS.values()]
    fig.legend(handles, ["thinker 的 kernel", "talker 的 kernel", "code2wav 的 kernel"], loc="upper right", fontsize=8, ncol=3)
    fig.suptitle("图 6　每张 GPU 上的 kernel（有颜色 = 正在执行；白色 = GPU 空着）", fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)


def fig_gpu_breakdown(plt, results: dict[str, dict[str, Any]], roles: dict[str, dict[int, str]], path: Path, title: str) -> None:  # noqa: ANN001
    parts = [("独自在算", "#2171b5"), ("和别的进程抢着算", "#9ecae1"), ("有活却空着", "#fd8d3c"), ("没活，空着", "#e0e0e0")]
    keys = list(results)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.8 * len(keys), 3.9), squeeze=False, sharey=True)
    for ax, key in zip(axes[0], keys):
        r = results[key]
        gpus = sorted(r["gpus"])
        vals = {
            "独自在算": [(r["gpus"][g]["busy"] - r["gpus"][g]["contended"]) * 100 for g in gpus],
            "和别的进程抢着算": [r["gpus"][g]["contended"] * 100 for g in gpus],
            "有活却空着": [r["gpus"][g]["idle_with_work"] * 100 for g in gpus],
            "没活，空着": [r["gpus"][g]["idle_without_work"] * 100 for g in gpus],
        }
        bottom = np.zeros(len(gpus))
        for name, color in parts:
            ax.bar(gpus, vals[name], bottom=bottom, color=color, label=name)
            bottom += np.array(vals[name])
        ax.set_xticks(gpus)
        ax.set_xticklabels([f"{g}\n{roles[key].get(g, '')}" for g in gpus], fontsize=7)
        ax.set_ylim(0, 100)
        ax.set_title(key, fontsize=9)
    axes[0][0].set_ylabel("占这段时间的百分比 %")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=4, loc="lower center")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def fig_gaps(plt, results: dict[str, dict[str, Any]], path: Path) -> None:  # noqa: ANN001
    labels = [lab for _, _, lab in GAP_BINS_MS]
    colors = ["#deebf7", "#9ecae1", "#4292c6", "#08519c", "#08306b"]
    rows = []
    for key, r in results.items():
        for g in sorted(r["gpus"]):
            rows.append((f"{key} · GPU {g}", r["gpus"][g]["idle_by_gap_length"]))
    fig, ax = plt.subplots(figsize=(11, 0.22 * len(rows) + 1.4))
    for i, (lab, bins) in enumerate(rows):
        x = 0.0
        for j, b in enumerate(labels):
            v = bins.get(b, 0) * 100
            ax.barh(i, v, left=x, color=colors[j], label=b if i == 0 else None)
            x += v
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("GPU 空着的时间占比 %（按每段空隙的长短分颜色）", fontsize=8)
    ax.legend(title="单段空隙的长度", fontsize=7, title_fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.06))
    ax.set_title("图 8　GPU 空着的时间，是很多很短的空隙，还是少数很长的空档？", fontsize=10, loc="left")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", action="append", required=True, help="KEY=RUN_DIR (phases w3b_prof_c8 and w3b_prof_c64)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib()
    summary: dict[str, Any] = {"generated_by": "scripts/analysis/exp01_kernel_timeline.py", "runs": {}}
    panels, by_level, roles, step_rows = [], defaultdict(dict), {}, []
    for spec in a.run:
        key, run = spec.split("=", 1)
        exp = root / run
        eid = json.load(open(exp / "result.json"))["experiment_id"]
        layout = re.sub(r"^exp01_prof_", "", eid)
        name = f"方案 {key}：{SCHEME_NAMES.get(layout, layout)}"
        summary["runs"][key] = {"run": run, "experiment_id": eid, "name": name, "phases": {}}
        for phase in ("w3b_prof_c8", "w3b_prof_c64"):
            res, procs = analyze_phase(exp, phase)
            c = phase.rsplit("_c", 1)[1]
            steps = {}
            for p in procs:
                if p["stage"] == "thinker":
                    st = thinker_steps(exp / "profiling" / "torch" / phase / p["server"] / p["file"])
                    if st:
                        steps[f"{p['server']}/{p['replica'] or 'thinker'}"] = st
            res["thinker_steps"] = steps
            if steps:
                med = {k2: float(np.median([v[k2] for v in steps.values()])) for k2 in ("period_ms_p50", "busy_ms_p50", "idle_ms_p50")}
                step_rows.append((f"{name}，并发 {c}（{len(steps)} 个 thinker 的中位数）", med))
            summary["runs"][key]["phases"][phase] = res
            by_level[c][f"{name}（并发 {c}）"] = res
            roles[f"{name}（并发 {c}）"] = {g: role_label({p["stage"] for p in procs if g in p["gpus"]}) for g in range(8)}
            a0, b0 = res["window_unix_ns"]
            focus = a0 + (b0 - a0) * (0.5 if c == "8" else 0.3)  # the c64 phase is one burst: look while it is busy
            panels.append((f"{name}，并发 {c}", procs, (focus - (b0 - a0) / 2, focus + (b0 - a0) / 2)))
    fig_kernel_timeline(plt, [p for p in panels if p[0].endswith("64") or "方案 C" in p[0]], out / "figures" / "fig6_kernel_timeline.png")
    for c, res in by_level.items():
        fig_gpu_breakdown(plt, res, roles, out / "figures" / f"fig7_gpu_time_c{c}.png",
                          f"图 7　并发 {c}：每张卡的时间去哪了（kernel 级，开着 profiler 的那一段）")
    fig_gaps(plt, {k: v for c in ("8", "64") for k, v in by_level[c].items()}, out / "figures" / "fig8_idle_gaps.png")
    if step_rows:
        fig_steps(plt, step_rows, out / "figures" / "fig9_thinker_step.png")
    (out / "results" / "kernel_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
