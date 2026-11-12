"""Interval arithmetic for busy/idle analysis.

GPU kernels on different streams, concurrent requests and overlapping stage executions all
overlap in time; summing their durations double-counts. Everything here works on the *union*
of half-open intervals [start, end).
"""

from __future__ import annotations

from typing import Iterable, Sequence

Interval = tuple[float, float]


def normalize(intervals: Iterable[Sequence[float]]) -> list[Interval]:
    """Sort, drop empty/inverted intervals, and merge overlapping or touching ones."""
    items = sorted((float(s), float(e)) for s, e in intervals if e > s)
    merged: list[Interval] = []
    for s, e in items:
        if merged and s <= merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def total(intervals: Iterable[Sequence[float]]) -> float:
    """Length of the union."""
    return sum(e - s for s, e in normalize(intervals))


def clip(intervals: Iterable[Sequence[float]], window: Sequence[float]) -> list[Interval]:
    lo, hi = window
    return normalize((max(s, lo), min(e, hi)) for s, e in intervals if e > lo and s < hi)


def gaps(intervals: Iterable[Sequence[float]], window: Sequence[float]) -> list[Interval]:
    """Idle intervals: the complement of the union inside ``window``."""
    lo, hi = window
    out: list[Interval] = []
    cur = lo
    for s, e in clip(intervals, window):
        if s > cur:
            out.append((cur, s))
        cur = max(cur, e)
    if cur < hi:
        out.append((cur, hi))
    return out


def intersect(a: Iterable[Sequence[float]], b: Iterable[Sequence[float]]) -> list[Interval]:
    a_n, b_n = normalize(a), normalize(b)
    i = j = 0
    out: list[Interval] = []
    while i < len(a_n) and j < len(b_n):
        s = max(a_n[i][0], b_n[j][0])
        e = min(a_n[i][1], b_n[j][1])
        if e > s:
            out.append((s, e))
        if a_n[i][1] < b_n[j][1]:
            i += 1
        else:
            j += 1
    return out


def subtract(a: Iterable[Sequence[float]], b: Iterable[Sequence[float]]) -> list[Interval]:
    """Parts of ``a`` not covered by ``b``."""
    a_n = normalize(a)
    if not a_n:
        return []
    window = (a_n[0][0], a_n[-1][1])
    return intersect(a_n, gaps(b, window))


def coverage(intervals: Iterable[Sequence[float]], window: Sequence[float]) -> float:
    """Busy fraction of ``window`` (0..1)."""
    lo, hi = window
    if hi <= lo:
        return 0.0
    return total(clip(intervals, window)) / (hi - lo)


def concurrency_profile(intervals: Iterable[Sequence[float]]) -> list[tuple[float, int]]:
    """Step function (time, number of active intervals from this time on)."""
    edges: list[tuple[float, int]] = []
    for s, e in intervals:
        if e > s:
            edges.append((float(s), 1))
            edges.append((float(e), -1))
    edges.sort(key=lambda x: (x[0], x[1]))  # ends before starts at equal times (half-open)
    out: list[tuple[float, int]] = []
    level = 0
    for t, d in edges:
        level += d
        if out and out[-1][0] == t:
            out[-1] = (t, level)
        else:
            out.append((t, level))
    return out


def time_weighted_mean_level(profile: list[tuple[float, int]], window: Sequence[float]) -> float:
    """Average concurrency over ``window`` for a profile from ``concurrency_profile``."""
    lo, hi = window
    if hi <= lo or not profile:
        return 0.0
    acc = 0.0
    level = 0
    prev = lo
    for t, lvl in profile:
        if t > hi:
            break
        if t > prev:
            acc += level * (min(t, hi) - prev)
            prev = t
        level = lvl
    if hi > prev:
        acc += level * (hi - prev)
    return acc / (hi - lo)


def gap_breakdown(gap_list: Iterable[Sequence[float]], edges: Sequence[float]) -> dict[str, dict[str, float]]:
    """Bucket idle gaps by length; returns count and total idle time per bucket."""
    edges = sorted(edges)
    labels = [f"<{edges[0]:g}"] + [f"{a:g}-{b:g}" for a, b in zip(edges, edges[1:])] + [f">={edges[-1]:g}"]
    out = {lab: {"count": 0, "total": 0.0} for lab in labels}
    for s, e in gap_list:
        d = e - s
        idx = 0
        while idx < len(edges) and d >= edges[idx]:
            idx += 1
        out[labels[idx]]["count"] += 1
        out[labels[idx]]["total"] += d
    return out
