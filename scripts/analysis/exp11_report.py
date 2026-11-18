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
    """One value of one run's phase; None when it is missing or the phase is unusable (errors + timeouts > 5 %)."""
    for r in runs:
        if r["arm"] == arm and r["block"] == block:
            row = ((r["levels"].get(w) or {}).get(c) or {}).get(mode) or {}
            return None if row.get("usable") is False else row.get(key)
    return None


def blocks_of(runs: list[dict[str, Any]]) -> list[int]:
    return sorted({r["block"] for r in runs})


def finite(x: Any) -> bool:
    return x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def contrast(runs: list[dict[str, Any]], w: str, c: int, mode: str, key: str, arm: str, base: str) -> dict[str, Any]:
    """Both arms per block and their paired difference (arm - base)."""
    bl = blocks_of(runs)
    a = [val(runs, arm, b, w, c, mode, key) for b in bl]
    o = [val(runs, base, b, w, c, mode, key) for b in bl]
    return {base: interval(o), arm: interval(a), "diff": interval([x - y if finite(x) and finite(y) else None for x, y in zip(a, o)])}


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
            shares, in_steps = [], []
            for b in bl:
                ta, tx = val(runs, "A", b, "w3b_ns", c, "B", "ttft_mean_ms"), val(runs, arm, b, "w3b_ns", c, "B", "ttft_mean_ms")
                qa, qx = val(runs, "A", b, "w3b_ns", c, "B", "queue_wait_mean_ms"), val(runs, arm, b, "w3b_ns", c, "B", "queue_wait_mean_ms")
                shares.append((qx - qa) / (tx - ta) if None not in (ta, tx, qa, qx) and tx - ta > 0 else None)
                step = val(runs, arm, b, "w3b_ns", c, "C", "decode_busy_ms_p50")  # reported only (supplementary registration 1)
                in_steps.append((qx - qa) / step if None not in (qa, qx) and step else None)
            parts[f"{arm}_vs_A_c{c}"] = {"verdict": rule(shares, lambda s: s >= 0.8, lambda s: s < 0.5), "queue_share_of_ttft_increase": shares,
                                         "queue_increase_in_decode_steps": in_steps}
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


def recommend(runs: list[dict[str, Any]], v: dict[str, Any]) -> dict[str, Any]:
    """The registered default-on rule: H11d supported for C vs A, and the TTFT increase (client P50 in the unprofiled A
    phase, C - A paired by block, mean over blocks) below 10 ms at W3b 8, 32 and 64 (supplementary registration 1)."""
    inc = {c: contrast(runs, "w3b_ns", c, "A", "ttft_p50_ms", "C", "A")["diff"] for c in (8, 32, 64)}
    n = len(blocks_of(runs))
    small = all(d["n"] == n and d["mean"] is not None and d["mean"] < 10 for d in inc.values())
    h11d = v["H11d"]["C_vs_A"]["verdict"]
    on = h11d == "支持" and small
    return {"default_on": on, "h11d_C_vs_A": h11d, "ttft_increase_ms": inc, "ttft_increase_below_10ms": small,
            "advice": "建议默认开启 C" if on else "只建议作为可选开关，并说明代价"}


TABLE = [("A", "throughput_rps"), ("A", "e2e_p50"), ("A", "e2e_p95"), ("A", "e2e_p99"), ("A", "ttft_p50_ms"), ("A", "ttft_p95_ms"),
         ("A", "thinker_mem_max_mib"), ("B", "lookahead_share"), ("B", "mixed_share"), ("B", "ms_per_token"), ("B", "ttft_mean_ms"),
         ("B", "queue_wait_mean_ms"), ("C", "between_idle_share"), ("C", "total_idle_share"), ("C", "gap_p50_ms"), ("C", "gap_p95_ms"),
         ("C", "gap_p99_ms"), ("C", "large_copies_per_1k_steps"), ("C", "decode_busy_ms_p50")]


def table(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    levels = sorted({(w, c) for r in runs for w, lv in r["levels"].items() for c in lv})
    for w, c in levels:
        for mode, key in TABLE:
            for arm, base in (("B", "A"), ("C", "A"), ("C", "B")):
                out[f"{w}|{c}|{mode}|{key}|{arm}-{base}"] = contrast(runs, w, c, mode, key, arm, base)
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
    restart_phases = {x.get("where", "").split()[-1] for x in run["restarts"] if x.get("where")}
    after_restart = False
    for ph in res.get("phases", []):
        after_restart = after_restart or ph["phase"] in restart_phases
        pm = PHASE_RE.match(ph["phase"])
        if not pm:
            continue
        w, c, mode = pm.group(1), int(pm.group(2)), pm.group(3)
        recs = read_jsonl(exp / "raw" / f"requests_{ph['phase']}.jsonl")
        meas = [r for r in recs if not r.get("warmup")]
        cm = client_metrics(recs)
        bad = (cm.get("errors") or 0) + (cm.get("timeouts") or 0)
        row: dict[str, Any] = {"client": cm, "usable": not (cm.get("measured") and bad / cm["measured"] > 0.05), "after_restart": after_restart}
        if mode == "A":
            cen = censored_percentiles(meas)
            win = phase_window(meas, anchor)
            mem = [mm for g in THINKER_GPUS for t, _, mm in util.get(g, []) if win and win[0] <= t <= win[1]]
            ttft = cm.get("ttft_s") or {}
            row.update(e2e_p50=cen["p50"], e2e_p95=cen["p95"], e2e_p99=cen["p99"], failed=cen["failed"],
                       throughput_rps=cm.get("throughput_rps"),
                       ttft_p50_ms=ttft["p50"] * 1000 if ttft.get("p50") is not None else None,
                       ttft_p95_ms=ttft["p95"] * 1000 if ttft.get("p95") is not None else None,
                       thinker_mem_max_mib=float(max(mem)) if mem else None)
        elif mode == "B":
            pdir = exp / "raw" / "server_events" / ph["phase"]
            events = la.read_thinker_events(pdir)
            steps = la.step_summary(events)
            row["steps"] = steps
            row["lookahead_share"] = steps.get("lookahead_share")
            row["mixed_share"] = (steps.get("mode_share") or {}).get("MIXED", 0.0) if steps.get("iterations") else None
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
            procs, decode_busy = [], []
            for f in sorted((exp / "profiling" / "torch" / ph["phase"]).rglob("*.trace.json*")):
                tr = st.load_trace(f)
                s = st.steps(tr)
                if not s:
                    continue
                decode_busy += [x["busy"] for x in s if x["mode"] == "DECODE"]
                gpu = [(e["ts"], e["ts"] + e.get("dur", 0)) for e in st.events_of(tr) if e.get("cat") in st.GPU_CATS]
                g = st.gaps(s, gpu)
                g.pop("steps")
                procs.append({**g, "n_steps": len(s), "blocking": st.blocking_attribution(tr)})
            if procs:
                row["procs"] = procs  # trace times are microseconds
                row["steps"] = sum(p["n_steps"] for p in procs)
                row["large_copies"] = sum(p["blocking"]["large"]["n"] for p in procs)
                row["large_inside"] = sum(p["blocking"]["large"]["inside"] for p in procs)
                row["large_copies_per_1k_steps"] = 1000 * row["large_copies"] / row["steps"] if row["steps"] else None
                row["decode_busy_ms_p50"] = float(np.median(decode_busy)) / 1000 if decode_busy else None
                for key, src in (("between_idle_share", "between_step_idle_share"), ("total_idle_share", "total_idle_share")):
                    v = [p[src] for p in procs if p[src] is not None]
                    row[key] = float(np.mean(v)) if v else None
                for q in ("p50", "p95", "p99"):
                    v = [p[f"gap_{q}"] for p in procs if p[f"gap_{q}"] is not None]
                    row[f"gap_{q}_ms"] = float(np.median(v)) / 1000 if v else None
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
    v = verdicts(runs)
    summary = {"generated_by": "scripts/analysis/exp11_report.py", "runs": runs, "verdicts": v, "recommendation": recommend(runs, v),
               "table": table(runs),
               "flags": {"unusable": [f"{r['arm']} block {r['block']} {w} c{c} {m}" for r in runs for w, lv in r["levels"].items()
                                      for c, ms in lv.items() for m, row in ms.items() if row.get("usable") is False],
                         "after_restart": [f"{r['arm']} block {r['block']} {w} c{c} {m}" for r in runs for w, lv in r["levels"].items()
                                           for c, ms in lv.items() for m, row in ms.items() if row.get("after_restart")]}}
    out = Path(a.out) / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "performance.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=lambda x: None) + "\n")
    print(out / "performance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
