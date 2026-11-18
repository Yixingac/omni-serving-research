#!/usr/bin/env python3
"""实验 14：视频请求能不能靠加副本提速。只分析 plan v2 已有的运行（F1 组和事件补测），没有新的 GPU 运行。

    python scripts/analysis/exp14_report.py --data-root <results root> \
        --run 20261116_v2b_f1_r1_run01 ... --events 20261117_v2b_f1_r1_events_run01 \
        --out research/01_性能瓶颈/实验14_视频请求加副本

Each replica is one GPU running the whole MiniCPM-o pipeline with --text-only; the client spreads requests round robin.
Workload W4: a synthetic video of moving circles (8 fps, 640x360, 10 s or 20 s) with the question how many circles
there are; the answer is one character. Per closed-loop level: throughput over the level's makespan (first send to
last completion, as summarize_plan_v2.py) and, less sensitive to the ramp and the tail of these short levels,
concurrency / mean time in the system (Little's law, no think time). CPU cores are the server process groups of
the 1-s CPU sampler; GPU busy is the nvidia-smi sampled utilisation of the run's GPUs. Stage times come from the
server events of the events run. The verdict rules for F-H1 are in the experiment's hypotheses.md; they were set
after the data had been seen, so the verdicts are exploratory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from omni_research.analysis.metrics import percentile  # noqa: E402

LINEAR = 0.8  # scaling efficiency counted as "approximately linear"
GPU_PTS = 5.0
PHASES = ("w4_video_10s", "w4_video_20s")
STAGES = ("preprocessing.service", "hop.preprocessing->image_encoder", "image_encoder.service", "hop.image_encoder->thinker",
          "thinker.queue", "thinker.prefill", "thinker.service", "hop.thinker->decode", "decode.service",
          "request.admission_to_last_terminal")


def pct(values: list[float | None], q: float) -> float | None:
    v = [x for x in values if x is not None]
    return percentile(v, q) if v else None


def level_stats(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by: dict[int, list[dict[str, Any]]] = {}
    for r in records:
        if not r.get("warmup") and r.get("concurrency") is not None:
            by.setdefault(int(r["concurrency"]), []).append(r)
    out = {}
    for c, recs in sorted(by.items()):
        ok = [r for r in recs if r.get("status") == "ok" and r.get("done_ns")]
        if not ok:
            out[c] = {"n": len(recs), "ok": 0}
            continue
        t0 = min(r["sent_ns"] for r in recs)
        t1 = max(r["done_ns"] for r in ok)
        e2e = [(r["done_ns"] - r["sent_ns"]) / 1e9 for r in ok]
        mean = sum(e2e) / len(e2e)
        out[c] = {"n": len(recs), "ok": len(ok), "span_s": (t1 - t0) / 1e9, "rps": len(ok) / ((t1 - t0) / 1e9),
                  "e2e_mean_s": mean, "e2e_p50_s": pct(e2e, 0.5), "e2e_p90_s": pct(e2e, 0.9), "little_rps": c / mean,
                  "window_ns": [t0, t1]}
    return out


def efficiency(best: dict[int, float]) -> dict[int, float]:
    """Throughput with n replicas / (n x throughput with one replica)."""
    one = best.get(1)
    return {n: round(v / (n * one), 6) for n, v in sorted(best.items())} if one else {}


def linear_verdict(eff_at_max: dict[str, float], threshold: float = LINEAR) -> dict[str, Any]:
    ok = [v >= threshold for v in eff_at_max.values()]
    v = "支持" if ok and all(ok) else ("不支持" if not any(ok) else "不确定")
    return {"verdict": v, "efficiency": eff_at_max, "threshold": threshold}


def gpu_flat_verdict(busy: dict[int, float], pts: float = GPU_PTS) -> dict[str, Any]:
    vals = [v for v in busy.values() if v is not None]
    spread = max(vals) - min(vals) if vals else None
    return {"verdict": "不确定" if spread is None else ("支持" if spread <= pts else "不支持"), "busy": busy, "spread_pts": spread}


def combine(verdicts: list[str]) -> str:
    """One verdict for both video lengths: agreeing verdicts stand, anything else is 不确定."""
    return verdicts[0] if verdicts and all(v == verdicts[0] for v in verdicts) and verdicts[0] in ("支持", "不支持") else "不确定"


def phase_gpu_busy(ana: dict[str, Any], phase: str, gpus: list[int]) -> float | None:
    """Mean sampled busy % over a whole phase (all its levels) of the GPUs the run used (analyze_experiment.py's gpu_by_phase)."""
    v = [g["sampled_util_mean"] for g in ana.get("gpu_by_phase", []) if g["phase"] == phase and int(g["gpu"]) in gpus
         and g.get("sampled_util_mean") is not None]
    return sum(v) / len(v) if v else None


def window_cores(samples: list[dict[str, Any]], a_ns: int, b_ns: int, skip: str = "client") -> float | None:
    """Mean CPU cores of the server process groups between two wall-clock times (ns); client-only samples are skipped."""
    vals = [sum(g["cpu_percent"] for k, g in s["groups"].items() if k != skip) / 100
            for s in samples if a_ns <= s["t_wall_ns"] <= b_ns and len(s["groups"]) > 1]
    return sum(vals) / len(vals) if vals else None


def stage_medians(per_request: dict[str, dict[str, float]], rids: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {k: pct([per_request[r].get(k) for r in rids if r in per_request], 0.5) for k in STAGES}
    total = out["request.admission_to_last_terminal"]
    for k in STAGES[:-1]:
        out[f"share.{k}"] = out[k] / total if out[k] is not None and total else None
    return out


# ------------------------------------------------------------------------------------------
# runs


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(p)] if p.exists() else []


def scan_run(exp: Path) -> dict[str, Any]:
    from exp01_gpu_timeline import load_util
    from omni_research.analysis.experiment_analysis import clock_anchor

    res = json.loads((exp / "result.json").read_text())
    ana = json.loads((exp / "analysis" / "analysis.json").read_text())
    cfg = json.loads((exp / "config.json").read_text())
    gpus = sorted({g for s in cfg.get("servers", []) for g in s.get("gpus", [])})
    n = len(cfg.get("servers", [])) or 1
    anchor = clock_anchor(exp)
    off = anchor[1] - anchor[0] if anchor else None
    util = load_util(exp)
    samples = _read_jsonl(exp / "raw" / "cpu_samples.jsonl")
    phases = {}
    for ph in PHASES:
        recs = _read_jsonl(exp / "raw" / f"requests_{ph}.jsonl")
        if not recs:
            continue
        levels = level_stats(recs)
        for c, lv in levels.items():
            if not lv.get("ok"):
                continue
            a, b = lv.pop("window_ns")
            if off is not None:
                cores = window_cores(samples, a + off, b + off)
                lv["cpu_cores_per_replica"] = cores / n if cores is not None else None
                busy = [x for g in gpus for t, x, _ in util.get(g, []) if (a + off) / 1e9 <= t < (b + off) / 1e9]
                lv["gpu_busy_mean"] = sum(busy) / len(busy) if busy else None
            lv["per_replica_concurrency"] = c / n
            lv["rps_per_replica"] = lv["rps"] / n
        phases[ph] = {str(c): lv for c, lv in levels.items()}
    return {"run": exp.name, "experiment_id": res["experiment_id"], "replicas": n, "gpus": gpus, "phases": phases,
            "phase_gpu_busy": {ph: phase_gpu_busy(ana, ph, gpus) for ph in PHASES if ph in phases}}


def scan_events(exp: Path) -> dict[str, Any]:
    from omni_research.analysis.experiment_analysis import read_jsonl, stage_breakdown

    out = {}
    for ph in PHASES:
        base = exp / "raw" / "server_events" / ph
        if not base.exists():
            continue
        recs = {r["request_id"]: r for r in read_jsonl(exp / "raw" / f"requests_{ph}.jsonl") if not r.get("warmup")}
        ev = [e for f in sorted(base.rglob("*.jsonl")) for e in read_jsonl(f)]
        per = stage_breakdown(ev)["per_request"]
        by: dict[int, list[str]] = {}
        for rid in per:
            if rid in recs:
                by.setdefault(int(recs[rid]["concurrency"]), []).append(rid)
        out[ph] = {str(c): {"requests": len(rids), **stage_medians(per, rids)} for c, rids in sorted(by.items())}
    return {"run": exp.name, "phases": out}


def build(runs: list[dict[str, Any]], events: dict[str, Any] | None) -> dict[str, Any]:
    scaling = {}
    for ph in PHASES:
        best = {r["replicas"]: max((lv["rps"] for lv in r["phases"].get(ph, {}).values() if lv.get("ok")), default=None) for r in runs}
        best_little = {r["replicas"]: max((lv["little_rps"] for lv in r["phases"].get(ph, {}).values() if lv.get("ok")), default=None) for r in runs}
        busy = {}
        for r in runs:  # GPU busy at the level with the highest makespan throughput
            lvs = [lv for lv in r["phases"].get(ph, {}).values() if lv.get("ok")]
            if lvs:
                busy[r["replicas"]] = max(lvs, key=lambda lv: lv["rps"]).get("gpu_busy_mean")
        best = {k: v for k, v in best.items() if v is not None}
        best_little = {k: v for k, v in best_little.items() if v is not None}
        scaling[ph] = {"best_rps": best, "efficiency": efficiency(best), "best_little_rps": best_little,
                       "efficiency_little": efficiency(best_little), "gpu_busy_at_best": busy,
                       "gpu_busy_phase": {r["replicas"]: r["phase_gpu_busy"].get(ph) for r in runs if r["phase_gpu_busy"].get(ph) is not None}}
    top = max(r["replicas"] for r in runs)
    eff = {ph.removeprefix("w4_video_"): scaling[ph]["efficiency"].get(top) for ph in PHASES if scaling[ph]["efficiency"]}
    eff_little = {ph.removeprefix("w4_video_"): scaling[ph]["efficiency_little"].get(top) for ph in PHASES if scaling[ph]["efficiency_little"]}
    verdicts = {
        "linear": {**linear_verdict(eff), "replicas": top, "metric": "各副本数下最高的吞吐（首个请求发出到最后完成）"},
        "linear_little": {**linear_verdict(eff_little), "replicas": top, "metric": "并发 ÷ 平均耗时（利特尔法则）"},
        "gpu_flat": {ph.removeprefix("w4_video_"): gpu_flat_verdict(scaling[ph]["gpu_busy_phase"]) for ph in PHASES},
        "gpu_flat_best_level": {ph.removeprefix("w4_video_"): gpu_flat_verdict(scaling[ph]["gpu_busy_at_best"]) for ph in PHASES},
    }
    verdicts["linear_overall"] = combine([verdicts["linear"]["verdict"]])
    verdicts["gpu_flat_overall"] = combine([x["verdict"] for x in verdicts["gpu_flat"].values()])
    return {"runs": runs, "events": events, "scaling": scaling, "verdicts": verdicts}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--events")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    runs = [scan_run(root / r) for r in a.run]
    ev = scan_events(root / a.events) if a.events else None
    ev_levels = scan_run(root / a.events) if a.events else None
    summary = {"generated_by": "scripts/analysis/exp14_report.py", "rule": {"linear_efficiency": LINEAR, "gpu_flat_pts": GPU_PTS},
               **build(runs, ev), "events_run_levels": ev_levels}
    out = Path(a.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "results" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    v = summary["verdicts"]
    print("linear", v["linear"]["verdict"], v["linear"]["efficiency"], "| little", v["linear_little"]["verdict"], v["linear_little"]["efficiency"],
          "| gpu (phase)", {k: (x["verdict"], round(x["spread_pts"], 1)) for k, x in v["gpu_flat"].items()}, v["gpu_flat_overall"],
          "| gpu (best level)", {k: (x["verdict"], round(x["spread_pts"], 1)) for k, x in v["gpu_flat_best_level"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
