#!/usr/bin/env python3
"""实验 11 的实测时间线：从每组的 trace 段里按固定规则截一小段 thinker 的时间线，存成小的 JSON，再画图。

    # on the cluster, where the raw traces stay:
    python scripts/analysis/exp11_timeline.py extract --data-root <results root> --run A=<run> --run B=<run> --run C=<run> \
        --phase w3b_ns_c64_C --out research/01_性能瓶颈/实验11_Thinker异步优化
    # anywhere, from results/timeline_windows.json:
    python scripts/analysis/exp11_timeline.py draw --out research/01_性能瓶颈/实验11_Thinker异步优化

Window rule, fixed before any 实验 11 trace exists: in a thinker process's trace, the first
thinker.request_finished_copy annotation that starts at least 1 s after the trace's first event; the window runs from
60 ms before it to 90 ms after it. The first process (sorted trace files) with such an annotation is used. Times in the
JSON are milliseconds from the window start; no file names or host names are kept.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp11_steps as st  # noqa: E402

AFTER_US, BEFORE_US, SPAN_US = 1_000_000, 60_000, 150_000


def pick_window(trace: dict[str, Any]) -> tuple[float, float] | None:
    ev = st.events_of(trace)
    if not ev:
        return None
    t0 = min(e["ts"] for e in ev)
    copies = sorted(e["ts"] for e in ev if e.get("cat") == "user_annotation" and e.get("name") == st.COPY_ANNOTATION
                    and e["ts"] >= t0 + AFTER_US)
    return (copies[0] - BEFORE_US, copies[0] - BEFORE_US + SPAN_US) if copies else None


def window_events(trace: dict[str, Any], start: float, end: float) -> dict[str, Any]:
    ev = st.events_of(trace)
    ms = lambda t: round((t - start) / 1000, 3)  # noqa: E731
    steps = [{"index": s["index"], "mode": s["mode"], "launch": s["launch"], "graph": s["graph"], "gpu_start_ms": ms(s["gpu_start"]),
              "gpu_end_ms": ms(s["gpu_end"]), "busy_ms": round(s["busy"] / 1000, 3)}
             for s in st.steps(trace) if s["gpu_end"] > start and s["gpu_start"] < end]
    copies = [{"start_ms": ms(e["ts"]), "dur_ms": round(e.get("dur", 0) / 1000, 3)} for e in ev
              if e.get("cat") == "user_annotation" and e.get("name") == st.COPY_ANNOTATION and start <= e["ts"] < end]
    gpu_bytes = {(e.get("args") or {}).get("correlation"): (e.get("args") or {}).get("bytes") for e in ev if e.get("cat") == "gpu_memcpy"}
    blocking = []
    for e in ev:
        if e.get("cat") not in ("cuda_runtime", "cuda_driver") or e.get("dur", 0) <= st.BLOCKING_US or not start <= e["ts"] < end:
            continue
        if e["name"] in st.SYNC:
            kind = "sync"
        elif e["name"].startswith("cudaMemcpy"):
            n = gpu_bytes.get((e.get("args") or {}).get("correlation"))
            kind = "unknown" if n is None else "small" if n <= st.SMALL_BYTES else "large"
        else:
            continue
        blocking.append({"name": e["name"], "kind": kind, "start_ms": ms(e["ts"]), "dur_ms": round(e["dur"] / 1000, 3)})
    busy = [(max(e["ts"], start), min(e["ts"] + e.get("dur", 0), end)) for e in ev
            if e.get("cat") in st.GPU_CATS and e["ts"] + e.get("dur", 0) > start and e["ts"] < end]
    merged: list[list[float]] = []
    for a, b in sorted(busy):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return {"steps": steps, "copies": copies, "blocking": blocking, "gpu_busy_ms": [[ms(a), ms(b)] for a, b in merged],
            "gpu_idle_share": round(1 - sum(b - a for a, b in merged) / (end - start), 4)}


def extract(data_root: Path, runs: dict[str, str], phase: str) -> dict[str, Any]:
    out: dict[str, Any] = {"generated_by": "scripts/analysis/exp11_timeline.py", "phase": phase,
                           "rule": "first thinker.request_finished_copy at least 1 s into the trace; 60 ms before to 90 ms after",
                           "arms": {}}
    for arm, run in runs.items():
        files = sorted((data_root / "experiments" / run / "profiling" / "torch" / phase).rglob("*.trace.json*"))
        for i, f in enumerate(files):
            tr = st.load_trace(f)
            w = pick_window(tr)
            if w:
                out["arms"][arm] = {"run": run, "process": i, **window_events(tr, *w)}
                break
        else:
            out["arms"][arm] = {"run": run, "process": None, "note": "no window found"}
    return out


ARM_NAMES = {"A": "A：同步执行", "B": "B：提前发射 + 快照", "C": "C：B + 旁路复制"}


def draw(data: dict[str, Any], path: Path) -> None:
    from matplotlib.patches import Rectangle

    from exp01_gpu_timeline import setup_matplotlib

    plt = setup_matplotlib()
    arms = [a for a in "ABC" if a in data["arms"] and data["arms"][a].get("process") is not None]
    fig, axes = plt.subplots(len(arms), 1, figsize=(13, 2.3 * len(arms) + 0.6), sharex=True, squeeze=False)
    for ax, arm in zip(axes[:, 0], arms):
        w = data["arms"][arm]
        for a, b in w["gpu_busy_ms"]:
            ax.add_patch(Rectangle((a, 1.3), b - a, 0.7, facecolor="#BBBBBB", edgecolor="none"))
        for s in w["steps"]:
            color = "#0072B2" if s["graph"] else "#E69F00"
            ax.add_patch(Rectangle((s["gpu_start_ms"], 2.1), s["gpu_end_ms"] - s["gpu_start_ms"], 0.5, facecolor=color, edgecolor="white", lw=0.5))
            ax.text((s["gpu_start_ms"] + s["gpu_end_ms"]) / 2, 2.35, f"{s['index']}", ha="center", va="center", fontsize=5.5, color="white")
        for c in w["copies"]:
            ax.add_patch(Rectangle((c["start_ms"], 0.5), max(c["dur_ms"], 0.3), 0.6, facecolor="#D55E00", edgecolor="none"))
        for b in w["blocking"]:
            ax.add_patch(Rectangle((b["start_ms"], 0.0), max(b["dur_ms"], 0.3), 0.4, facecolor="black" if b["kind"] == "large" else "#666666",
                                   edgecolor="none"))
        ax.set_ylim(-0.1, 2.8)
        ax.set_yticks([0.2, 0.8, 1.65, 2.35])
        ax.set_yticklabels(["阻塞调用 > 1 ms", "请求结束拷贝", "GPU 有活", "调度步（序号）"], fontsize=7)
        ax.set_title(f"{ARM_NAMES[arm]}　这段窗口里 GPU 空着的比例 {w['gpu_idle_share']:.0%}", fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
    axes[-1, 0].set_xlim(0, SPAN_US / 1000)
    axes[-1, 0].set_xlabel("毫秒（窗口从“请求结束拷贝”之前 60 毫秒开始）", fontsize=8)
    fig.suptitle(f"图 10　实测时间线（{data['phase']}，每组一个 thinker 进程的 150 毫秒；蓝 = 走 CUDA graph 的步，橙 = 不走 graph 的步）",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--data-root", required=True)
    ex.add_argument("--run", action="append", required=True, help="ARM=run_dir")
    ex.add_argument("--phase", default="w3b_ns_c64_C")
    ex.add_argument("--out", required=True)
    dr = sub.add_parser("draw")
    dr.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if a.cmd == "extract":
        runs = dict(spec.split("=", 1) for spec in a.run)
        data = extract(Path(a.data_root).expanduser(), runs, a.phase)
        (out / "results").mkdir(parents=True, exist_ok=True)
        (out / "results" / "timeline_windows.json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
        print(out / "results" / "timeline_windows.json")
    else:
        data = json.loads((out / "results" / "timeline_windows.json").read_text())
        (out / "figures").mkdir(parents=True, exist_ok=True)
        draw(data, out / "figures" / "fig10_measured_timeline.png")
        print(out / "figures" / "fig10_measured_timeline.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
