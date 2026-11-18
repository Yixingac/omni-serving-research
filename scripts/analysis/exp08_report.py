#!/usr/bin/env python3
"""实验 8 第一阶段的汇总和判定：合并主区组和补测区组，按预先登记的标准判 H8a–H8d 和第三阶段的进入条件。

    python scripts/analysis/exp08_report.py --main <summary of the main blocks> --supp <summary of the supplement> \
        --out research/01_性能瓶颈/实验08_Thinker_Lookahead

Both inputs are written by exp08_lookahead.py (one entry per block in per_run, blocks in order). Rule of the
amendment of 16:35 (hypotheses.md): the C-phase cells of the text conditions at concurrency 32 and 64 come only
from the supplement; every other cell from the main blocks. A cell with a missing block is "数据不足".
Writes results/phase1.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

LEVELS = (8, 32, 64)
CONDITIONS = ("w3t", "w3t_ns", "w3b", "w3b_ns")
SUPP_CELLS = {(w, c) for w in ("w3t", "w3t_ns") for c in (32, 64)}
IDLE = "graph_steps.idle_ms_p50"
BATCH = "steps.batch_size_decode.p50"
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
CELL_METRICS = {"C": [IDLE, "graph_steps.busy_ms_p50", "graph_steps.steps", "graph_steps.waited_share"],
                "B": ["steps.decode_lookahead_share", BATCH, "thinker_ms_per_token.p50", "client.throughput_rps",
                      "client.e2e_s.p50", "client.e2e_s.p95", "client.ttft_s.p50", "flush_share"]}


def get(d: dict[str, Any], path: str) -> Any:
    if path == "flush_share":
        steps = d.get("steps") or {}
        total = (steps.get("iterations") or 0) + (steps.get("flush_only_iterations") or 0)
        return sum((steps.get("flush_reasons") or {}).values()) / total if total else None
    for part in path.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def cell_runs(main: dict[str, Any], supp: dict[str, Any], cond: str, c: int, mode: str, metric: str) -> list[Any]:
    src = supp if mode == "C" and (cond, c) in SUPP_CELLS else main
    return [get(r["phases"].get(f"{cond}_c{c}_{mode}", {}), metric) for r in src.get("per_run", [])]


def complete(runs: list[Any]) -> bool:
    return bool(runs) and all(x is not None for x in runs)


def interval(runs: list[Any]) -> dict[str, Any]:
    v = [x for x in runs if x is not None]
    out: dict[str, Any] = {"runs": runs, "n": len(v), "mean": float(np.mean(v)) if v else None, "ci95": None}
    if len(v) >= 2:
        half = T975.get(len(v) - 1, 2.0) * float(np.std(v, ddof=1)) / math.sqrt(len(v))
        out["ci95"] = [out["mean"] - half, out["mean"] + half]
    return out


def nearest_level(main: dict[str, Any], cond: str, batch_size: float) -> int:
    sizes = {c: np.mean(cell_runs(main, {}, cond, c, "B", BATCH)) for c in LEVELS if complete(cell_runs(main, {}, cond, c, "B", BATCH))}
    return min(sizes, key=lambda c: (abs(sizes[c] - batch_size), c))


def overall(levels: dict[str, dict[str, Any]], yes: str, no: str) -> str:
    verdicts = [x["verdict"] for x in levels.values()]
    if no in verdicts:
        return no
    if "数据不足" in verdicts:
        return "数据不足"
    return yes if all(v == yes for v in verdicts) else "不确定"


def verdicts(main: dict[str, Any], supp: dict[str, Any]) -> dict[str, Any]:
    idle = lambda cond, c: cell_runs(main, supp, cond, c, "C", IDLE)  # noqa: E731
    out: dict[str, Any] = {}

    # H8c: text only, the seed is the only difference. Supported when the no-seed idle time is at most half at
    # every level with all blocks in that direction; refuted when the difference is < 1 ms or directions differ.
    levels = {}
    for c in LEVELS:
        t, tn = idle("w3t", c), idle("w3t_ns", c)
        if not (complete(t) and complete(tn)):
            levels[str(c)] = {"verdict": "数据不足"}
            continue
        diffs = [b - a for a, b in zip(t, tn)]
        same_sign = all(d < 0 for d in diffs) or all(d > 0 for d in diffs)
        if abs(np.mean(diffs)) < 1 or not same_sign:
            v = "推翻"
        elif np.mean(tn) <= np.mean(t) / 2 and all(d < 0 for d in diffs):
            v = "支持"
        else:
            v = "不确定"
        levels[str(c)] = {"verdict": v, "w3t": interval(t), "w3t_ns": interval(tn), "diff": interval(diffs)}
    out["H8c"] = {"verdict": overall(levels, "支持", "推翻"), "by_level": levels}

    # H8d (negative control): speech with and without the seed do not differ (< 1 ms, or no consistent direction).
    levels = {}
    for c in LEVELS:
        b, bn = idle("w3b", c), idle("w3b_ns", c)
        if not (complete(b) and complete(bn)):
            levels[str(c)] = {"verdict": "数据不足"}
            continue
        diffs = [y - x for x, y in zip(b, bn)]
        same_sign = all(d < 0 for d in diffs) or all(d > 0 for d in diffs)
        v = "支持" if abs(np.mean(diffs)) < 1 or not same_sign else "不支持"
        levels[str(c)] = {"verdict": v, "w3b": interval(b), "w3b_ns": interval(bn), "diff": interval(diffs)}
    out["H8d"] = {"verdict": overall(levels, "支持", "不支持"), "by_level": levels}

    # H8a: text vs speech, both seeded, at the text level of the nearest decode batch size. Supported when the
    # text idle time is at most half the speech one; refuted when they are about the same (< 1 ms or mixed signs).
    levels = {}
    for c in LEVELS:
        b = idle("w3b", c)
        bs = cell_runs(main, supp, "w3b", c, "B", BATCH)
        if not (complete(b) and complete(bs)):
            levels[str(c)] = {"verdict": "数据不足"}
            continue
        ct = nearest_level(main, "w3t", float(np.mean(bs)))
        t = idle("w3t", ct)
        if not complete(t):
            levels[str(c)] = {"verdict": "数据不足", "text_level": ct}
            continue
        diffs = [x - y for x, y in zip(t, b)]
        same_sign = all(d < 0 for d in diffs) or all(d > 0 for d in diffs)
        if np.mean(t) <= np.mean(b) / 2 and all(d < 0 for d in diffs):
            v = "支持"
        elif abs(np.mean(diffs)) < 1 or not same_sign:
            v = "推翻"
        else:
            v = "不确定"
        levels[str(c)] = {"verdict": v, "text_level": ct, "w3t": interval(t), "w3b": interval(b), "diff": interval(diffs)}
    out["H8a"] = {"verdict": overall(levels, "支持", "推翻"), "by_level": levels}

    # H8b: in the text condition that does run ahead (w3t_ns), the share of iterations that flush rises with concurrency.
    shares = {c: cell_runs(main, supp, "w3t_ns", c, "B", "flush_share") for c in LEVELS}
    if all(complete(v) for v in shares.values()):
        means = [float(np.mean(shares[c])) for c in LEVELS]
        v = "支持" if all(a < b for a, b in zip(means, means[1:])) else "推翻"
    else:
        v = "数据不足"
    out["H8b"] = {"verdict": v, "flush_share": {str(c): interval(shares[c]) for c in LEVELS}}

    # Entry to phase 3: H8c supported and, at every speech level, speech idle minus the idle of text that runs
    # ahead at the nearest batch size is at least 2 ms.
    by_level = {}
    for c in LEVELS:
        bs = cell_runs(main, supp, "w3b_ns", c, "B", BATCH)
        bn = idle("w3b_ns", c)
        if not (complete(bs) and complete(bn)):
            by_level[str(c)] = {"saving_ms": None}
            continue
        ct = nearest_level(main, "w3t_ns", float(np.mean(bs)))
        tn = idle("w3t_ns", ct)
        by_level[str(c)] = {"text_level": ct, "speech_batch": float(np.mean(bs)),
                            "saving_ms": float(np.mean(bn) - np.mean(tn)) if complete(tn) else None}
    savings = [x["saving_ms"] for x in by_level.values()]
    out["phase3_entry"] = {"met": out["H8c"]["verdict"] == "支持" and all(s is not None and s >= 2 for s in savings),
                           "by_level": by_level}
    return out


def cells(main: dict[str, Any], supp: dict[str, Any]) -> dict[str, Any]:
    return {cond: {str(c): {m: interval(cell_runs(main, supp, cond, c, mode, m)) for mode, ms in CELL_METRICS.items() for m in ms}
                   for c in LEVELS} for cond in CONDITIONS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--main", required=True)
    ap.add_argument("--supp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m, s = json.loads(Path(a.main).read_text()), json.loads(Path(a.supp).read_text())
    summary = {"generated_by": "scripts/analysis/exp08_report.py", "main_runs": m.get("runs"), "supp_runs": s.get("runs"),
               "rule": "w3t、w3t_ns 在并发 32、64 的 C 段指标只用补测区组（补充登记二）", "verdicts": verdicts(m, s), "cells": cells(m, s)}
    out = Path(a.out) / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float) + "\n")
    print(out / "phase1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
