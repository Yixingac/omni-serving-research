#!/usr/bin/env python3
"""实验 3：同时处理多少请求最好。只分析 plan v2 已有的运行，没有新的 GPU 运行。

    python scripts/analysis/exp03_report.py --data-root <results root> \
        --curve s2x6:20261116_v2b_sb_s2x6_run01:w3b_small --curve s2x6:20261116_v2b_a_s2x6_c03_run01:w3b_closed ... \
        --open s2x6:20261116_v2b_aoh_s2x6_run01 ... --out research/01_性能瓶颈/实验03_同时处理多少请求最好

Closed-loop curves join the small-batch runs (concurrency 1-8) with the closed-loop sweeps (16-256) of the same split;
both have server events, which stage_by_level.breakdown turns into per-level stage times. Open-loop runs offer the
same absolute rates (16-40 req/s) to every split and have no server events. Workload W3b (text question, spoken
answer). SLO (plan v2): success, TTFT <= 1 s and e2e / output-audio seconds <= 0.5. Percentiles are
linear-interpolated. Little's law: requests in the system = throughput x mean time in the system.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from omni_research.analysis.metrics import percentile, request_metrics  # noqa: E402

SLO_FRACTION = 0.95
STAGES = (("thinker", "queue"), ("thinker", "model_path"), ("talker", "queue"), ("talker", "model_path"), ("code2wav", "service"))


def pct(values: list[float | None], q: float) -> float | None:
    v = [x for x in values if x is not None]
    return percentile(v, q) if v else None


def level_summary(row: dict[str, Any]) -> dict[str, Any]:
    """One closed-loop level from a stage_by_level.breakdown row."""
    e2e = row["e2e_s"]
    st = {f"{r}.{m}": pct(row["stage_ms"].get((r, m), []), 0.5) for r, m in STAGES}
    e2e_p50 = pct(e2e, 0.5)
    known = sum(v for v in st.values() if v is not None)
    split = {"other_ms": e2e_p50 * 1e3 - known if e2e_p50 is not None else None,
             "code2wav_share": (st["code2wav.service"] or 0.0) / (e2e_p50 * 1e3) if e2e_p50 else None,
             "thinker_share": ((st["thinker.queue"] or 0.0) + (st["thinker.model_path"] or 0.0)) / (e2e_p50 * 1e3) if e2e_p50 else None,
             "talker_share": ((st["talker.queue"] or 0.0) + (st["talker.model_path"] or 0.0)) / (e2e_p50 * 1e3) if e2e_p50 else None}
    return {"concurrency": row["level"], "n": row["n"], "errors": row["errors"], "rps": row["tp_rps"], "slo": row["slo"],
            "ttft_p50_s": pct(row["ttft_s"], 0.5), "ttft_p90_s": pct(row["ttft_s"], 0.9), "e2e_p50_s": e2e_p50,
            "e2e_p90_s": pct(e2e, 0.9), "e2e_mean_s": statistics.fmean(e2e) if e2e else None,
            "e2e_per_audio_p90": pct(row["e2e_per_audio_s"], 0.9),
            "in_flight_little": row["tp_rps"] * statistics.fmean(e2e) if e2e and row["tp_rps"] else None,
            "stages_ms": st, "split": split, "eff_batch": row.get("eff_batch"), "gpu_util": row.get("gpu_util")}


def slo_capacity(levels: list[tuple[float, float | None]], threshold: float = SLO_FRACTION) -> dict[str, Any]:
    """Highest offered rate before the first level under the threshold; censored when no level failed."""
    best = None
    for rate, frac in sorted(levels):
        if frac is None or frac < threshold:
            return {"capacity": best, "censored": False}
        best = rate
    return {"capacity": best, "censored": True}


def doubling_gain(rps: dict[int, float]) -> dict[int, float]:
    """Throughput gain when concurrency doubles: rps(c) / rps(c/2) - 1."""
    return {c: rps[c] / rps[c // 2] - 1 for c in sorted(rps) if c % 2 == 0 and c // 2 in rps and rps[c // 2]}


def full_replica_verdict(curves: dict[str, dict[str, Any]], tol: float = 0.03, per_gpu_limit: int = 16) -> dict[str, Any]:
    """Plan v2 A-H1, second half: 8 full replicas are best only at <= 2 requests per GPU (concurrency <= 16 on 8 GPUs).
    "Best" = e2e median within tol of the lowest at that concurrency, among the curves that have the level.
    支持 when the replicas are near-best at every concurrency <= 16 and at none above; 不支持 otherwise."""
    by_c: dict[float, dict[str, float]] = {}
    for layout, c in curves.items():
        for lv in c["levels"]:
            if lv.get("e2e_p50_s") is not None:
                by_c.setdefault(lv["concurrency"], {})[layout] = lv["e2e_p50_s"]
    near = {}
    for conc, vals in sorted(by_c.items()):
        if "r8" in vals and len(vals) > 1:
            near[int(conc)] = vals["r8"] <= min(vals.values()) * (1 + tol) + 1e-12
    low = [v for c, v in near.items() if c <= per_gpu_limit]
    high = [v for c, v in near.items() if c > per_gpu_limit]
    ok = bool(low) and all(low) and not any(high)
    return {"verdict": "支持" if ok else "不支持", "r8_near_best": near, "tolerance": tol}


def slo_ok(rec: dict[str, Any]) -> bool:
    if rec.get("status") != "ok":
        return False
    m = request_metrics(rec)
    if m.get("ttft_s") is None or m["ttft_s"] > 1.0:
        return False
    return m.get("e2e_per_audio_s") is not None and m["e2e_per_audio_s"] <= 0.5


def scan_open(exp: Path, phase: str = "w3b_open_abs") -> list[dict[str, Any]]:
    from omni_research.analysis.experiment_analysis import read_jsonl

    recs = [r for r in read_jsonl(exp / "raw" / f"requests_{phase}.jsonl") if not r.get("warmup")]
    out = []
    for rate in sorted({r["rate_rps"] for r in recs if r.get("rate_rps") is not None}):
        rs = [r for r in recs if r.get("rate_rps") == rate]
        ok = [r for r in rs if r.get("status") == "ok" and r.get("done_ns")]
        ms = [request_metrics(r) for r in ok]
        t0, t1 = min(r["sent_ns"] for r in rs), max((r["done_ns"] for r in ok), default=None)
        errors: dict[str, int] = {}
        for r in rs:
            if r.get("status") != "ok":
                e = str(r.get("error") or r.get("status"))
                k = "client fd limit" if "Too many open files" in e else ("CUDA out of memory" if "out of memory" in e else e[:60])
                errors[k] = errors.get(k, 0) + 1
        out.append({"rate_rps": rate, "n": len(rs), "ok": len(ok), "errors": errors,
                    "rps": len(ok) / ((t1 - t0) / 1e9) if t1 and t1 > t0 else None, "slo": sum(map(slo_ok, rs)) / len(rs),
                    "ttft_p90_s": pct([m.get("ttft_s") for m in ms], 0.9), "e2e_p50_s": pct([m.get("e2e_s") for m in ms], 0.5),
                    "e2e_p90_s": pct([m.get("e2e_s") for m in ms], 0.9), "e2e_mean_s": statistics.fmean([m["e2e_s"] for m in ms]) if ms else None,
                    "e2e_per_audio_p90": pct([m.get("e2e_per_audio_s") for m in ms], 0.9)})
    return out


def label(layout: str) -> str:
    if layout.startswith("s"):
        t, k = layout[1:].split("x")
        return f"{t}+{k}"
    return "4 组 1+1" if layout == "p4" else "8 个完整副本"


def build(curves: list[tuple[str, str, str]], opens: list[tuple[str, str]], root: Path) -> dict[str, Any]:
    from stage_by_level import breakdown

    out_curves: dict[str, Any] = {}
    for layout, run, phase in curves:
        rows = [level_summary(r) for r in breakdown(root / run, phase)]
        node = "第一批节点" if run.startswith("20261115") else "compute-03"
        c = out_curves.setdefault(layout, {"label": label(layout), "levels": [], "runs": []})
        c["runs"].append({"run": run, "phase": phase, "node": node})
        for r in rows:
            r["run"], r["node"] = run, node
        c["levels"] = sorted(c["levels"] + rows, key=lambda r: r["concurrency"])
    for c in out_curves.values():
        rps = {int(r["concurrency"]): r["rps"] for r in c["levels"] if r["rps"]}
        c["doubling_gain"] = doubling_gain(rps)
        ok = [r for r in c["levels"] if r["slo"] is not None and r["slo"] >= SLO_FRACTION]
        c["highest_concurrency_within_slo"] = max((r["concurrency"] for r in ok), default=None)
    out_open = {}
    for layout, run in opens:
        lv = scan_open(root / run)
        cap = slo_capacity([(r["rate_rps"], r["slo"]) for r in lv])
        at = next((r for r in lv if r["rate_rps"] == cap["capacity"]), None)
        out_open[layout] = {"label": label(layout), "run": run, "levels": lv, **cap,
                            "in_flight_at_capacity": at["rps"] * at["e2e_mean_s"] if at and at["rps"] and at["e2e_mean_s"] else None}
    return {"curves": out_curves, "open": out_open, "verdicts": {"A-H1_second_half": full_replica_verdict(out_curves)}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--curve", action="append", required=True, help="LAYOUT:RUN_DIR:PHASE")
    ap.add_argument("--open", action="append", default=[], help="LAYOUT:RUN_DIR (open loop at absolute rates)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    curves = [tuple(x.split(":")) for x in a.curve]
    opens = [tuple(x.split(":")) for x in a.open]
    summary = {"generated_by": "scripts/analysis/exp03_report.py", "rule": {"slo_fraction": SLO_FRACTION}, **build(curves, opens, root)}
    out = Path(a.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "results" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    for k, c in summary["curves"].items():
        print(k, "highest c within SLO", c["highest_concurrency_within_slo"], {int(r["concurrency"]): round(r["rps"], 2) for r in c["levels"]})
    for k, o in summary["open"].items():
        print("open", k, "capacity", o["capacity"], "censored", o["censored"], "in flight", o["in_flight_at_capacity"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
