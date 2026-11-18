#!/usr/bin/env python3
"""实验 5 第二版：thinker、talker 每生成一步时，CPU 和 GPU 各在干什么（从 torch profiler trace 拆空档）。

    python scripts/analysis/exp05_step_sync.py --data-root ~/research-staging/unicorn \
        --run A=20261117_exp01_prof_r8_run01 --run B=20261117_exp01_prof_p4_run01 \
        --run C=20261117_exp01_prof_s2x6_run01 --run C128=20261117_exp05_talker_s2x6_run01 \
        --out research/01_性能瓶颈/实验05_GPU为什么会等待

Each decode step of the thinker and talker is one cudaGraphLaunch on the CPU thread and the graph's kernels
on the GPU (linked by the CUDA correlation id). Per step, from one launch to the next:

- GPU busy: union of the process's kernels / copies from this graph's first kernel to the next one's;
- idle: the rest of that span;
- lead: the next launch's return minus this graph's last kernel end, negative when the GPU had nothing
  queued and waited for the CPU;
- sync: CPU time inside cudaStreamSynchronize / cudaEventSynchronize / cudaDeviceSynchronize; the step
  is "waited" when the last sync of the step returns only after this step's graph has finished, i.e. the
  CPU blocked on the step it had just launched and prepared the next one afterwards;
- host after sync: from that sync's return to the next launch (CPU work the GPU waits through).

Writes results/step_sync.json and figures fig10 (a zoomed CPU / GPU timeline of a few thinker steps)
and fig11 (per layout and load: GPU time, idle while the CPU prepares the next step, other idle).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp01_kernel_timeline import DEVICE_CATS, setup_matplotlib, stage_of, union  # noqa: E402

SYNC = {"cudaStreamSynchronize", "cudaEventSynchronize", "cudaDeviceSynchronize"}
PHASES = {"A": ["w3b_prof_c8", "w3b_prof_c64"], "B": ["w3b_prof_c8", "w3b_prof_c64"], "C": ["w3b_prof_c8", "w3b_prof_c64"],
          "C128": ["w3b_prof_c128"]}
LABEL = {"A": "8 个完整副本", "B": "4 对两卡流水线", "C": "2+6", "C128": "2+6"}
MAX_STEP_US = 200_000  # a launch-to-launch span longer than this is a pause between requests, not a step


def load_trace(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def steps_of(tr: dict[str, Any]) -> list[dict[str, float]]:
    """One row per pair of consecutive graph launches (times in microseconds of the trace clock)."""
    ev = tr.get("traceEvents", [])
    rt = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver")), key=lambda e: e["ts"])
    dev = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") in DEVICE_CATS), key=lambda e: e["ts"])
    if not rt or not dev:
        return []
    by_corr: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for e in dev:
        c = (e.get("args") or {}).get("correlation")
        if c is not None:
            by_corr[c].append(e)
    launches = []
    for e in rt:
        if e["name"] != "cudaGraphLaunch":
            continue
        ks = by_corr.get((e.get("args") or {}).get("correlation"), [])
        if ks:
            launches.append((float(e["ts"]), float(e["dur"]), min(k["ts"] for k in ks), max(k["ts"] + k["dur"] for k in ks)))
    launches.sort()
    syncs = np.array([(e["ts"], e["ts"] + e["dur"]) for e in rt if e["name"] in SYNC], dtype=float).reshape(-1, 2)
    dv = np.array([(e["ts"], e["ts"] + e["dur"]) for e in dev], dtype=float)
    starts = dv[:, 0]
    rows = []
    for (a, ad, gs, ge), (a2, ad2, gs2, _) in zip(launches, launches[1:]):
        span = gs2 - gs
        if span <= 0 or span > MAX_STEP_US:
            continue
        i0, i1 = np.searchsorted(starts, gs), np.searchsorted(starts, gs2)
        seg = dv[i0:i1].copy()
        seg[:, 1] = np.minimum(seg[:, 1], gs2)
        busy = float((union(seg)[:, 1] - union(seg)[:, 0]).sum()) if len(seg) else 0.0
        s = syncs[(syncs[:, 0] >= a) & (syncs[:, 0] < a2)] if len(syncs) else syncs
        last_end = float(s[:, 1].max()) if len(s) else None
        waited = last_end is not None and last_end >= ge - 50
        rows.append({"launch_period": a2 - a, "span": span, "graph": ge - gs, "busy": busy, "idle": span - busy,
                     "lead": ge - (a2 + ad2), "sync": float((s[:, 1] - s[:, 0]).sum()) if len(s) else 0.0,
                     "waited": waited, "host_after_sync": (a2 - last_end) if waited else None})
    return rows


def summarize(rows: list[dict[str, float]]) -> dict[str, float] | None:
    if not rows:
        return None
    med = lambda k: float(np.median([r[k] for r in rows])) / 1e3  # noqa: E731  (ms)
    waited = [r for r in rows if r["waited"]]
    span = sum(r["span"] for r in rows)
    idle_w = sum(r["idle"] for r in waited)
    return {"steps": len(rows), "span_ms_p50": med("span"), "busy_ms_p50": med("busy"), "idle_ms_p50": med("idle"),
            "graph_ms_p50": med("graph"), "launch_period_ms_p50": med("launch_period"), "lead_ms_p50": med("lead"),
            "sync_share_of_period": sum(r["sync"] for r in rows) / sum(r["launch_period"] for r in rows),
            "waited_share": len(waited) / len(rows),
            "host_after_sync_ms_p50": float(np.median([r["host_after_sync"] for r in waited])) / 1e3 if waited else None,
            "idle_share": sum(r["idle"] for r in rows) / span,
            "idle_share_in_waited_steps": idle_w / span,
            "late_launch_share": sum(1 for r in rows if r["lead"] < 0) / len(rows)}


def procs_of(phase_dir: Path, stage: str) -> list[Path]:
    """Trace files of one stage, the largest per pid (a process can write a near-empty file per stage name)."""
    best: dict[str, Path] = {}
    for f in phase_dir.rglob("*.trace.json*"):
        if stage_of(f.name)[0] != stage:
            continue
        m = re.search(r"_pid(\d+)_", f.name)
        pid = f"{f.parent.name}/{m.group(1) if m else f.name}"
        if pid not in best or f.stat().st_size > best[pid].stat().st_size:
            best[pid] = f
    return sorted(best.values())


def zoom(tr: dict[str, Any], n_steps: int = 4) -> dict[str, Any]:
    """A few consecutive thinker steps from the middle of the trace, for the CPU / GPU picture (ms from the first launch)."""
    ev = tr.get("traceEvents", [])
    rt = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver")), key=lambda e: e["ts"])
    launches = [e for e in rt if e["name"] == "cudaGraphLaunch"]
    if len(launches) < n_steps + 2:
        return {}
    mid = len(launches) // 2
    t0, t1 = launches[mid]["ts"] - 2000, launches[mid + n_steps]["ts"]
    gpu = [((e["ts"] - t0) / 1e3, e["dur"] / 1e3, bool((e.get("args") or {}).get("graph id", 0)))
           for e in ev if e.get("ph") == "X" and e.get("cat") in DEVICE_CATS and t0 <= e["ts"] <= t1]
    cpu = [((e["ts"] - t0) / 1e3, e["dur"] / 1e3, "sync" if e["name"] in SYNC else "launch" if e["name"] == "cudaGraphLaunch" else "api")
           for e in rt if t0 <= e["ts"] <= t1]
    return {"span_ms": (t1 - t0) / 1e3, "gpu": gpu, "cpu": cpu}


def fig_zoom(plt, z: dict[str, Any], title: str, path: Path) -> None:  # noqa: ANN001
    fig, ax = plt.subplots(figsize=(12, 3.2))
    for s, d, g in z["gpu"]:
        ax.broken_barh([(s, max(d, 0.02))], (1.55, 0.7), color="#d94801" if g else "#fd8d3c", lw=0)
    # The CPU lane: blocked in sync (red), CUDA API calls (gray), launches (blue); white = Python and the scheduler.
    ax.broken_barh([(0, z["span_ms"])], (0.15, 0.7), color="#fdd0a2", lw=0)
    for s, d, k in z["cpu"]:
        color = {"sync": "#cb181d", "launch": "#2171b5", "api": "#969696"}[k]
        ax.broken_barh([(s, max(d, 0.05 if k == "launch" else 0.01))], (0.15, 0.7), color=color, lw=0)
    ax.set_yticks([0.5, 1.9])
    ax.set_yticklabels(["CPU（thinker 进程的调度线程）", "GPU（thinker 的卡）"], fontsize=8)
    ax.set_xlim(0, z["span_ms"])
    ax.set_xlabel("时间（毫秒）", fontsize=8)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color="#d94801", label="GPU：CUDA graph 里的 kernel（一步的主体）"),
                       Patch(color="#fd8d3c", label="GPU：graph 之外的小 kernel 和拷贝"),
                       Patch(color="#cb181d", label="CPU：卡在同步上，等 GPU 算完这一步"),
                       Patch(color="#fdd0a2", label="CPU：Python 和调度（处理结果、准备下一步）"),
                       Patch(color="#2171b5", label="CPU：发出下一步（cudaGraphLaunch）")],
              fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)
    ax.set_title(title, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_split(plt, rows: list[tuple[str, dict[str, float]]], path: Path) -> None:  # noqa: ANN001
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 1.6))
    for i, (lab, s) in enumerate(rows):
        busy = s["busy_ms_p50"]
        wait = s["idle_ms_p50"] * (s["idle_share_in_waited_steps"] / s["idle_share"] if s["idle_share"] else 0)
        rest = s["idle_ms_p50"] - wait
        ax.barh(i, busy, color="#d94801", label="GPU 在算" if i == 0 else None)
        ax.barh(i, wait, left=busy, color="#cb181d", alpha=0.45, label="GPU 空着：CPU 同步等完这一步之后才准备下一步" if i == 0 else None)
        ax.barh(i, rest, left=busy + wait, color="#bdbdbd", label="GPU 空着：其他" if i == 0 else None)
        ax.text(busy + wait + rest + 0.2, i, f"{busy:.1f} + {s['idle_ms_p50']:.1f} 毫秒；{s['waited_share'] * 100:.0f}% 的步是同步的",
                va="center", fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(r[1]["busy_ms_p50"] + r[1]["idle_ms_p50"] for r in rows) * 1.6)
    ax.set_xlabel("每一步的时间（毫秒，中位数）", fontsize=8)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("图 11　thinker 和 talker 每一步：GPU 空着的时候 CPU 在干什么", fontsize=10, loc="left")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", action="append", default=[], help="KEY=run directory (KEY in A, B, C, C128)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib()
    result: dict[str, Any] = {"generated_by": "scripts/analysis/exp05_step_sync.py", "runs": {}}
    zoom_data = None
    for spec in a.run:
        key, run = spec.split("=", 1)
        result["runs"][key] = {"run": run, "phases": {}}
        for ph in PHASES[key]:
            pdir = root / run / "profiling" / "torch" / ph
            per: dict[str, Any] = {}
            for stage in ("thinker", "talker"):
                files = procs_of(pdir, stage)
                if stage == "talker":
                    files = files[:2]  # a talker trace is 20-40 MB; two per phase show the pattern
                for f in files:
                    tr = load_trace(f)
                    s = summarize(steps_of(tr))
                    if s:
                        per[f"{stage}/{f.parent.name}/{f.name.split('_trace')[0]}"] = s
                    if zoom_data is None and key == "C" and ph == "w3b_prof_c8" and stage == "thinker":
                        zoom_data = zoom(tr)
                    print(f"{key} {ph} {f.name}: {s and round(s['idle_ms_p50'], 2)} ms idle", file=sys.stderr)
            result["runs"][key]["phases"][ph] = per
    rows = []
    for key in ("C", "B", "A", "C128"):
        r = result["runs"].get(key)
        if not r:
            continue
        for ph, per in r["phases"].items():
            for stage in ("thinker", "talker"):
                ss = [v for k, v in per.items() if k.startswith(stage + "/")]
                if not ss:
                    continue
                agg = {k: float(np.median([s[k] for s in ss if s.get(k) is not None])) for k in ss[0] if any(s.get(k) is not None for s in ss)}
                result["runs"][key].setdefault("median_over_processes", {}).setdefault(ph, {})[stage] = agg
                rows.append((f"{LABEL[key]}，并发 {ph.rsplit('_c', 1)[1]}：{stage}", agg))
    if zoom_data:
        fig_zoom(plt, zoom_data, "图 10　放大看 thinker 的 4 步（2+6，并发 8）：CPU 先等 GPU 算完这一步，再花几毫秒准备下一步，GPU 就空着等",
                 out / "figures" / "fig10_thinker_step_zoom.png")
    fig_split(plt, rows, out / "figures" / "fig11_step_idle_split.png")
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "results" / "step_sync.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(out / "results" / "step_sync.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
