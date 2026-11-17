#!/usr/bin/env python3
"""实验 8 第三阶段的性能对比：同一个原型副本，SGLANG_OMNI_SPEECH_LOOKAHEAD 关（off）和开（on），W3b 不带种子。

    python scripts/analysis/exp08_proto_perf.py --data-root <results root> \
        --run 20261117_exp08_proto_perf_block1_off_run01 --run ... --out research/01_性能瓶颈/实验08_Thinker_Lookahead

Phases are w3b_ns_c<concurrency>_<A|B|C>. A: client metrics without events (throughput, end-to-end percentiles,
TTFT, errors) and the thinker cards' memory; B: the thinker's per-iteration events (lookahead share, token time);
C: a 3 s torch trace of the thinker processes only (idle per step, sync calls per step). Off and on are paired
within a block. Verdict registered in hypotheses.md (amendment of 16:35): on is lower than off in every block and
the mean idle per step drops by at least 2 ms. Writes results/phase3.json.
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

import exp05_step_sync as ss  # noqa: E402

PHASE_RE = re.compile(r"^w3b_ns_c(\d+)_([ABC])$")
RUN_RE = re.compile(r"^exp08_proto_perf_block(\d+)_(off|on)$")
THINKER_GPUS = [0, 1]
BLOCKING_COPY_US = 1000
T975 = {1: 12.706, 2: 4.303, 3: 3.182}
METRICS = {"A": ["client.throughput_rps", "client.e2e_s.p50", "client.e2e_s.p95", "client.e2e_s.p99", "client.ttft_s.p50",
                 "client.first_audio_s.p50", "client.errors", "thinker_mem_max_mib"],
           "B": ["steps.decode_lookahead_share", "steps.batch_size_decode.p50", "thinker_ms_per_token.p50", "client.throughput_rps"],
           "C": ["graph_steps.idle_ms_p50", "graph_steps.busy_ms_p50", "graph_steps.steps", "graph_steps.waited_share", "sync_calls_per_step"]}


def sync_calls_per_step(trace: dict[str, Any]) -> float | None:
    """Calls where the CPU waits for the GPU (synchronize calls, and copies that blocked over 1 ms) per graph launch."""
    calls = [e for e in trace.get("traceEvents", []) if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver")]
    launches = sum(1 for e in calls if e["name"] == "cudaGraphLaunch")
    waits = sum(1 for e in calls if e["name"] in ss.SYNC or (e["name"].startswith("cudaMemcpy") and e.get("dur", 0) > BLOCKING_COPY_US))
    return waits / launches if launches else None


def get(d: dict[str, Any] | None, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def interval(values: list[Any]) -> dict[str, Any]:
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    out: dict[str, Any] = {"runs": values, "n": len(v), "mean": float(np.mean(v)) if v else None, "ci95": None}
    if len(v) >= 2:
        half = T975.get(len(v) - 1, 2.0) * float(np.std(v, ddof=1)) / math.sqrt(len(v))
        out["ci95"] = [out["mean"] - half, out["mean"] + half]
    return out


def paired(runs: list[dict[str, Any]], level: str, mode: str, path: str) -> dict[str, Any]:
    by = {(r["arm"], r["block"]): r for r in runs}
    blocks = sorted({r["block"] for r in runs})
    off = [get(by.get(("off", b), {}).get("levels", {}).get(level, {}).get(mode), path) for b in blocks]
    on = [get(by.get(("on", b), {}).get("levels", {}).get(level, {}).get(mode), path) for b in blocks]
    diff = [y - x if x is not None and y is not None else None for x, y in zip(off, on)]
    rel = [(y - x) / x if x not in (None, 0) and y is not None else None for x, y in zip(off, on)]
    return {"off": interval(off), "on": interval(on), "diff": interval(diff), "relative": interval(rel)}


def verdicts(runs: list[dict[str, Any]]) -> dict[str, Any]:
    levels = sorted({k for r in runs for k in r["levels"]}, key=int)
    idle = {}
    for lv in levels:
        p = paired(runs, lv, "C", "graph_steps.idle_ms_p50")
        d = p["diff"]["runs"]
        if not d or any(x is None for x in d):
            v = "数据不足"
        elif all(x < 0 for x in d) and -float(np.mean(d)) >= 2:
            v = "支持"
        else:
            v = "不支持"
        idle[lv] = {"verdict": v, **p}
    verdict_list = [x["verdict"] for x in idle.values()]
    overall = "支持" if verdict_list and all(v == "支持" for v in verdict_list) else "数据不足" if "数据不足" in verdict_list and "支持" not in verdict_list \
        else "部分支持" if "支持" in verdict_list else "不支持"
    return {"idle": idle, "idle_verdict": overall}


def analyze_run(exp: Path) -> dict[str, Any]:
    import exp08_lookahead as la
    from exp01_gpu_timeline import load_util
    from exp06_report import client_metrics, phase_window
    from omni_research.analysis.experiment_analysis import clock_anchor, read_jsonl

    res = json.load(open(exp / "result.json"))
    m = RUN_RE.match(res["experiment_id"])
    run: dict[str, Any] = {"run": exp.name, "block": int(m.group(1)), "arm": m.group(2), "levels": {},
                           "restarts": res.get("server_restarts") or []}
    util, anchor = load_util(exp), clock_anchor(exp)
    for ph in res["phases"]:
        pm = PHASE_RE.match(ph["phase"])
        if not pm:
            continue
        c, mode = pm.group(1), pm.group(2)
        recs = read_jsonl(exp / "raw" / f"requests_{ph['phase']}.jsonl")
        row: dict[str, Any] = {"client": client_metrics(recs)}
        if mode == "A":
            win = phase_window([r for r in recs if not r.get("warmup")], anchor)
            mem = [mm for g in THINKER_GPUS for t, _, mm in util.get(g, []) if win and win[0] <= t <= win[1]]
            row["thinker_mem_max_mib"] = float(max(mem)) if mem else None
        elif mode == "B":
            events = la.read_thinker_events(exp / "raw" / "server_events" / ph["phase"])
            row["steps"] = la.step_summary(events)
            row["thinker_ms_per_token"] = la.thinker_ms_per_token(events, {r["request_id"]: r for r in recs if r.get("request_id")})
        else:
            per_proc, syncs = [], []
            for f in ss.procs_of(exp / "profiling" / "torch" / ph["phase"], "thinker"):
                tr = ss.load_trace(f)
                s = ss.summarize(ss.steps_of(tr))
                if s:
                    per_proc.append(s)
                syncs.append(sync_calls_per_step(tr))
            row["graph_steps"] = {k: float(np.median([p[k] for p in per_proc if p.get(k) is not None]))
                                  for k in ("steps", "busy_ms_p50", "idle_ms_p50", "waited_share", "lead_ms_p50")} if per_proc else {}
            row["sync_calls_per_step"] = float(np.median([x for x in syncs if x is not None])) if any(x is not None for x in syncs) else None
        run["levels"].setdefault(c, {})[mode] = row
    return run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    runs = [analyze_run(root / name) for name in a.run]
    levels = sorted({k for r in runs for k in r["levels"]}, key=int)
    contrasts = {lv: {mode: {p: paired(runs, lv, mode, p) for p in paths} for mode, paths in METRICS.items()} for lv in levels}
    summary = {"generated_by": "scripts/analysis/exp08_proto_perf.py", "runs": [r["run"] for r in runs], "per_run": runs,
               "contrasts": contrasts, "verdicts": verdicts(runs)}
    out = Path(a.out) / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase3.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float) + "\n")
    print(out / "phase3.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
