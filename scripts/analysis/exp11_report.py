#!/usr/bin/env python3
"""实验 11 性能对比（A 同步、B 快照、C 快照 + 旁路复制）的汇总和判定。

    python scripts/analysis/exp11_report.py --data-root <results root> --run 20261118_exp11_perf_block1_A_run01 ... \
        --out research/01_性能瓶颈/实验11_Thinker异步优化

Per run and level: A phase (no events): throughput, end-to-end percentiles (failures as infinite), TTFT, thinker
cards' memory; B phase (thinker events): lookahead and mixed shares, ms per token, mean output tokens, and the TTFT
split of exp11_ttft.py (means, which add up); C phase (thinker-only torch trace): exp11_steps.py's step-aligned gaps,
idle shares and the blocking copies inside or outside the end-of-request annotation. Verdicts H11a-H11e as
registered in hypotheses.md; contrasts are paired within a block.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RUN_RE = re.compile(r"exp11_perf_block(\d+)_([ABC])")
PHASE_RE = re.compile(r"^(w3b_ns|w3l_ns|w3xl_ns)_c(\d+)_([ABC])$")
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
THINKER_GPUS = [0, 1]


def interval(values: list[Any]) -> dict[str, Any]:
    v = [x for x in values if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
    out: dict[str, Any] = {"runs": values, "n": len(v), "mean": float(np.mean(v)) if v else None, "ci95": None}
    if len(v) >= 2:
        half = T975.get(len(v) - 1, 2.0) * float(np.std(v, ddof=1)) / math.sqrt(len(v))
        out["ci95"] = [out["mean"] - half, out["mean"] + half]
    return out


def val(runs: list[dict[str, Any]], arm: str, block: int, w: str, c: int, mode: str, key: str) -> Any:
    for r in runs:
        if r["arm"] == arm and r["block"] == block:
            return (((r["levels"].get(w) or {}).get(c) or {}).get(mode) or {}).get(key)
    return None


def blocks_of(runs: list[dict[str, Any]]) -> list[int]:
    return sorted({r["block"] for r in runs})


def rule(values: list[Any], support, refute) -> str:  # noqa: ANN001
    if not values or any(v is None for v in values):
        return "数据不足"
    if all(support(v) for v in values):
        return "支持"
    if all(refute(v) for v in values):
        return "反驳"
    return "不确定"


def verdicts(runs: list[dict[str, Any]]) -> dict[str, Any]:
    bl = blocks_of(runs)
    out: dict[str, Any] = {}

    parts, shares_all = {}, []
    for arm in "BC":
        for c in (8, 32):
            shares = []
            for b in bl:
                ta, tx = val(runs, "A", b, "w3b_ns", c, "B", "ttft_mean_ms"), val(runs, arm, b, "w3b_ns", c, "B", "ttft_mean_ms")
                qa, qx = val(runs, "A", b, "w3b_ns", c, "B", "queue_wait_mean_ms"), val(runs, arm, b, "w3b_ns", c, "B", "queue_wait_mean_ms")
                shares.append((qx - qa) / (tx - ta) if None not in (ta, tx, qa, qx) and tx - ta > 0 else None)
            parts[f"{arm}_vs_A_c{c}"] = {"verdict": rule(shares, lambda s: s >= 0.8, lambda s: s < 0.5), "queue_share_of_ttft_increase": shares}
            shares_all.append(parts[f"{arm}_vs_A_c{c}"]["verdict"])
    out["H11a"] = {"verdict": "支持" if all(v == "支持" for v in shares_all) else "反驳" if all(v == "反驳" for v in shares_all)
                   else "数据不足" if "数据不足" in shares_all else "不确定", "parts": parts}

    inside = []
    for b in bl:
        n = sum(val(runs, "B", b, "w3b_ns", c, "C", "large_copies") or 0 for c in (8, 32, 64))
        k = sum(val(runs, "B", b, "w3b_ns", c, "C", "large_inside") or 0 for c in (8, 32, 64))
        inside.append(k / n if n else None)
    out["H11b"] = {"verdict": rule(inside, lambda s: s >= 0.9, lambda s: s < 0.5), "share_inside_annotation": inside}

    copies, idle = {}, []
    for c in (32, 64):
        red = []
        for b in bl:
            nb, sb = val(runs, "B", b, "w3b_ns", c, "C", "large_copies"), val(runs, "B", b, "w3b_ns", c, "C", "steps")
            nc, sc = val(runs, "C", b, "w3b_ns", c, "C", "large_copies"), val(runs, "C", b, "w3b_ns", c, "C", "steps")
            red.append(1 - (nc / sc) / (nb / sb) if None not in (nb, sb, nc, sc) and nb and sb and sc else None)
        copies[c] = red
    for b in bl:
        ib, ic = val(runs, "B", b, "w3b_ns", 64, "C", "between_idle_share"), val(runs, "C", b, "w3b_ns", 64, "C", "between_idle_share")
        idle.append(ib - ic if None not in (ib, ic) else None)
    all_red = copies[32] + copies[64]
    if any(v is None for v in all_red + idle) or not idle:
        v = "数据不足"
    elif all(r >= 0.9 for r in all_red) and all(d >= 0.02 for d in idle):
        v = "支持"
    elif all(d <= 0 for d in idle):
        v = "不支持"
    else:
        v = "不确定"
    out["H11c"] = {"verdict": v, "large_copy_reduction_per_step": copies, "idle_share_drop_c64": idle}

    h11d = {}
    for arm, base in (("C", "B"), ("C", "A")):
        rel, by_level = [], {}
        for c in (32, 64):
            rows = []
            for b in bl:
                x, y = val(runs, arm, b, "w3b_ns", c, "A", "e2e_p50"), val(runs, base, b, "w3b_ns", c, "A", "e2e_p50")
                rows.append(1 - x / y if None not in (x, y) and y else None)
            by_level[c] = rows
            rel += rows
        if not rel or any(r is None for r in rel):
            v = "数据不足"
        elif all(r >= 0.02 for r in rel):
            v = "支持"
        elif all(r <= 0 for r in rel):
            v = "不支持"
        else:
            v = "不确定"
        h11d[f"{arm}_vs_{base}"] = {"verdict": v, "relative_p50_improvement": by_level}
    out["H11d"] = h11d

    saving, gain = {}, {}
    for w in ("w3b_ns", "w3l_ns", "w3xl_ns"):
        s_rows, g_rows = [], []
        for b in bl:
            ma, mc = val(runs, "A", b, w, 32, "B", "ms_per_token"), val(runs, "C", b, w, 32, "B", "ms_per_token")
            tok = val(runs, "C", b, w, 32, "B", "tokens_mean")
            s_rows.append((ma - mc) * tok if None not in (ma, mc, tok) else None)
            ea, ec = val(runs, "A", b, w, 32, "A", "e2e_p50"), val(runs, "C", b, w, 32, "A", "e2e_p50")
            g_rows.append(1 - ec / ea if None not in (ea, ec) and ea else None)
        saving[w], gain[w] = s_rows, g_rows
    rows = list(zip(saving["w3b_ns"], saving["w3l_ns"], saving["w3xl_ns"], gain["w3b_ns"], gain["w3xl_ns"]))
    if not rows or any(x is None for r in rows for x in r):
        v = "数据不足"
    elif all(sl > sb and sx > sb and gx > gb for sb, sl, sx, gb, gx in rows):
        v = "支持"
    elif all(gx <= gb for _, _, _, gb, gx in rows):
        v = "不支持"
    else:
        v = "不确定"
    out["H11e"] = {"verdict": v, "thinker_saving_ms_per_request": saving, "relative_e2e_p50_gain": gain}
    return out


# ---------------------------------------------------------------- reading runs

def analyze_run(exp: Path) -> dict[str, Any]:
    import exp08_lookahead as la
    import exp11_steps as st
    import exp11_ttft as tt
    from exp01_gpu_timeline import load_util
    from exp06_report import client_metrics, phase_window
    from exp09_report import censored_percentiles
    from omni_research.analysis.experiment_analysis import clock_anchor, read_jsonl

    res = json.loads((exp / "result.json").read_text())
    m = RUN_RE.search(res["experiment_id"])
    run: dict[str, Any] = {"run": exp.name, "block": int(m.group(1)), "arm": m.group(2), "restarts": res.get("server_restarts") or [], "levels": {}}
    util, anchor = load_util(exp), clock_anchor(exp)
    for ph in res.get("phases", []):
        pm = PHASE_RE.match(ph["phase"])
        if not pm:
            continue
        w, c, mode = pm.group(1), int(pm.group(2)), pm.group(3)
        recs = read_jsonl(exp / "raw" / f"requests_{ph['phase']}.jsonl")
        meas = [r for r in recs if not r.get("warmup")]
        cm = client_metrics(recs)
        row: dict[str, Any] = {"client": cm}
        if mode == "A":
            cen = censored_percentiles(meas)
            win = phase_window(meas, anchor)
            mem = [mm for g in THINKER_GPUS for t, _, mm in util.get(g, []) if win and win[0] <= t <= win[1]]
            row.update(e2e_p50=cen["p50"], e2e_p95=cen["p95"], e2e_p99=cen["p99"], failed=cen["failed"],
                       throughput_rps=cm.get("throughput_rps"), ttft_p50=(cm.get("ttft_s") or {}).get("p50"),
                       thinker_mem_max_mib=float(max(mem)) if mem else None)
        elif mode == "B":
            pdir = exp / "raw" / "server_events" / ph["phase"]
            events = la.read_thinker_events(pdir)
            steps = la.step_summary(events)
            row["steps"] = steps
            row["ms_per_token"] = (la.thinker_ms_per_token(events, {r["request_id"]: r for r in recs if r.get("request_id")}) or {}).get("p50")
            toks = [r.get("completion_tokens") for r in meas if r.get("status") == "ok" and r.get("completion_tokens")]
            row["tokens_mean"] = float(np.mean(toks)) if toks else None
            tev = tt.read_events(pdir)
            off = tt.clock_offset(tev)
            if off["offset_ns"] is not None:
                pts = tt.points(tev, off["offset_ns"])
                segs = [tt.segments(pts.get(r["request_id"], {}), r) for r in meas if r.get("status") == "ok" and r.get("request_id")]
                means = {k: (float(np.mean([s[k] for s in segs if s.get(k) is not None])) if any(s.get(k) is not None for s in segs) else None)
                         for k in [n for n, _, _ in tt.SEGMENTS] + ["ttft"]}
                row.update(ttft_mean_ms=means["ttft"], queue_wait_mean_ms=means["queue_wait"], ttft_segments_mean_ms=means, clock=off)
        else:
            procs = []
            for f in sorted((exp / "profiling" / "torch" / ph["phase"]).rglob("*.trace.json*")):
                tr = st.load_trace(f)
                s = st.steps(tr)
                if not s:
                    continue
                gpu = [(e["ts"], e["ts"] + e.get("dur", 0)) for e in st.events_of(tr) if e.get("cat") in st.GPU_CATS]
                g = st.gaps(s, gpu)
                g.pop("steps")
                procs.append({**g, "n_steps": len(s), "blocking": st.blocking_attribution(tr)})
            if procs:
                row["procs"] = procs
                row["steps"] = sum(p["n_steps"] for p in procs)
                row["large_copies"] = sum(p["blocking"]["large"]["n"] for p in procs)
                row["large_inside"] = sum(p["blocking"]["large"]["inside"] for p in procs)
                row["between_idle_share"] = float(np.mean([p["between_step_idle_share"] for p in procs if p["between_step_idle_share"] is not None]))
                row["total_idle_share"] = float(np.mean([p["total_idle_share"] for p in procs if p["total_idle_share"] is not None]))
                row["gap_p50_ms"] = float(np.median([p["gap_p50"] for p in procs if p["gap_p50"] is not None])) / 1000
                row["gap_p99_ms"] = float(np.median([p["gap_p99"] for p in procs if p["gap_p99"] is not None])) / 1000
        run["levels"].setdefault(w, {}).setdefault(c, {})[mode] = row
    return run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    runs = [analyze_run(root / n) for n in a.run]
    summary = {"generated_by": "scripts/analysis/exp11_report.py", "runs": runs, "verdicts": verdicts(runs)}
    out = Path(a.out) / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "performance.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=lambda x: None) + "\n")
    print(out / "performance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
