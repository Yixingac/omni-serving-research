"""GPU busy/idle intervals from an Nsight Systems SQLite export (``nsys export --type sqlite``).

Busy time per device is the *union* of kernel (and optionally memcpy/memset) intervals, so
kernels that overlap on different streams are not double-counted. Idle gaps are the complement
inside the analysis window. NVTX ranges, when present, are returned so gaps can be attributed to
code regions.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from . import intervals as iv


def _tables(con: sqlite3.Connection) -> set[str]:
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def device_activity(db: str | Path, include_memcpy: bool = True) -> dict[int, list[tuple[float, float]]]:
    """Kernel (+ memcpy/memset) intervals per device, in nanoseconds on the nsys clock."""
    con = sqlite3.connect(str(db))
    try:
        tabs = _tables(con)
        out: dict[int, list[tuple[float, float]]] = {}
        sources = ["CUPTI_ACTIVITY_KIND_KERNEL"]
        if include_memcpy:
            sources += ["CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"]
        for t in sources:
            if t not in tabs:
                continue
            for dev, start, end in con.execute(f"SELECT deviceId, start, end FROM {t}"):
                out.setdefault(int(dev), []).append((float(start), float(end)))
        return out
    finally:
        con.close()


def nvtx_ranges(db: str | Path) -> list[dict[str, Any]]:
    con = sqlite3.connect(str(db))
    try:
        if "NVTX_EVENTS" not in _tables(con):
            return []
        rows = con.execute(
            "SELECT start, end, COALESCE(text, (SELECT value FROM StringIds WHERE id = textId)) FROM NVTX_EVENTS "
            "WHERE end IS NOT NULL AND end > start"
        ).fetchall()
        return [{"start": s, "end": e, "name": n} for s, e, n in rows]
    finally:
        con.close()


def busy_idle_report(activity: dict[int, list[tuple[float, float]]], window: tuple[float, float] | None = None,
                     gap_edges_ns: tuple[float, ...] = (1e4, 1e5, 1e6, 1e7, 1e8)) -> dict[int, dict[str, Any]]:
    """Busy fraction, idle total and a histogram of idle-gap lengths for each device."""
    report: dict[int, dict[str, Any]] = {}
    for dev, acts in activity.items():
        u = iv.normalize(acts)
        if not u:
            continue
        w = window or (u[0][0], u[-1][1])
        gaps = iv.gaps(u, w)
        report[dev] = {
            "window_ns": w,
            "busy_ns": iv.total(iv.clip(u, w)),
            "busy_fraction": iv.coverage(u, w),
            "idle_ns": sum(e - s for s, e in gaps),
            "gap_count": len(gaps),
            "longest_gap_ns": max((e - s for s, e in gaps), default=0.0),
            "gap_histogram": iv.gap_breakdown(gaps, gap_edges_ns),
            "kernel_sum_ns": sum(e - s for s, e in acts),  # naive sum, shown only to expose overlap
        }
    return report
