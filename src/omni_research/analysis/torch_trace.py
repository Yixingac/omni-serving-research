"""GPU busy/idle intervals from PyTorch-profiler Chrome traces (``*.trace.json[.gz]``).

SGLang-Omni's torch profiler (``POST /start_profile`` with ``enable_torch``) writes one Chrome
trace per process. Device-side events carry ``cat`` = ``kernel`` / ``gpu_memcpy`` /
``gpu_memset`` with ``ts``/``dur`` in microseconds and the device in ``args.device``. Busy time is
the union of those intervals per device (see ``nsys.busy_idle_report``).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from . import intervals as iv
from .nsys import busy_idle_report

DEVICE_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def load_chrome_trace(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return json.load(fh)


def device_activity(trace: dict[str, Any], include_memcpy: bool = True) -> dict[int, list[tuple[float, float]]]:
    """Per-device intervals in nanoseconds."""
    out: dict[int, list[tuple[float, float]]] = {}
    cats = DEVICE_CATS if include_memcpy else {"kernel"}
    for ev in trace.get("traceEvents", []):
        if ev.get("ph") != "X" or ev.get("cat") not in cats:
            continue
        args = ev.get("args") or {}
        dev = args.get("device", args.get("Device"))
        if dev is None:
            continue
        start = float(ev["ts"]) * 1000.0
        out.setdefault(int(dev), []).append((start, start + float(ev.get("dur", 0)) * 1000.0))
    return out


def kernel_names(trace: dict[str, Any], top: int = 15) -> list[tuple[str, int, float]]:
    """Most time-consuming kernels: (name, count, total_ms) — naive sum per kernel name."""
    agg: dict[str, list[float]] = {}
    for ev in trace.get("traceEvents", []):
        if ev.get("ph") == "X" and ev.get("cat") == "kernel":
            agg.setdefault(ev.get("name", "?"), []).append(float(ev.get("dur", 0)) / 1000.0)
    rows = [(n, len(v), sum(v)) for n, v in agg.items()]
    rows.sort(key=lambda r: -r[2])
    return rows[:top]


def analyze_files(paths: list[Path]) -> dict[str, Any]:
    """Merge device activity from several per-process traces of the same window.

    Each file is one process (MiniCPM-o: the pipeline process that runs the thinker, and the talker
    process that also runs code2wav), so each file gets its own busy time per device. Per device,
    ``multi_process_ns`` is the time during which two or more processes had device work in flight.
    Without MPS, kernels of different processes take turns on the GPU instead of running together,
    so this overlap is time spent contending for the GPU, not parallel execution.
    """
    merged: dict[int, list[tuple[float, float]]] = {}
    unions: dict[int, list[list[tuple[float, float]]]] = {}
    files = []
    for p in paths:
        tr = load_chrome_trace(p)
        act = device_activity(tr)
        busy = {}
        for dev, iv_list in act.items():
            merged.setdefault(dev, []).extend(iv_list)
            u = iv.normalize(iv_list)
            unions.setdefault(dev, []).append(u)
            busy[dev] = iv.total(u)
        files.append({"file": p.name, "devices": sorted(act), "busy_ns": busy, "kernels_top": kernel_names(tr, 8)})
    report = busy_idle_report(merged)
    for dev, us in unions.items():
        if dev in report and len(us) > 1:
            both = [x for i in range(len(us)) for j in range(i + 1, len(us)) for x in iv.intersect(us[i], us[j])]
            report[dev]["multi_process_ns"] = iv.total(both)
    return {"files": files, "per_device": report}
