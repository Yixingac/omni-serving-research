#!/usr/bin/env python3
"""实验 8 第一阶段：thinker 什么时候走提前一步发射、被什么挡住，以及每一步 GPU 空了多久。

    python scripts/analysis/exp08_lookahead.py --data-root <results root> \
        --run 20261117_exp08_lookahead_block1_run01 --run ... --out research/01_性能瓶颈/实验08_Thinker_Lookahead

Phases are named <condition>_c<concurrency>_<B|C>; conditions are w3t (text only, seed 0), w3t_ns (text only,
no seed), w3b (text + speech, seed 0), w3b_ns (text + speech, no seed). B phases carry the measurement patch's
exp_thinker_step events (one per scheduler iteration of each thinker); C phases a 3 s kernel trace, from which
exp05_step_sync.py measures every decode step's GPU busy and idle time. Unit of replication: the run (block).
Writes results/summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

PHASE_RE = re.compile(r"^(w3[tb](?:_ns)?)_c(\d+)_([BC])$")
CONDITIONS = ["w3t", "w3t_ns", "w3b", "w3b_ns"]
CPU_PARTS = ("launch", "resolve", "drain", "run", "process", "top_flush")
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def stats(values: list[float]) -> dict[str, Any]:
    v = np.array([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    return {"n": int(v.size), "mean": float(v.mean()), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95))}


def step_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Shares of lookahead / blockers / flushes over a phase's thinker iterations (see the module docstring)."""
    steps = [e["metadata"] for e in events if e.get("event_name") == "exp_thinker_step"]
    with_batch = [m for m in steps if "batch_size" in m]
    decode = [m for m in with_batch if m.get("forward_mode") == "DECODE"]
    runner_blocked = [m for m in with_batch if m.get("blocked_by") == "runner_ineligible"]
    n = len(with_batch)
    out: dict[str, Any] = {
        "iterations": n, "flush_only_iterations": len(steps) - n,
        "lookahead_share": sum(1 for m in with_batch if m.get("use_lookahead")) / n if n else None,
        "decode_iterations": len(decode),
        "decode_lookahead_share": sum(1 for m in decode if m.get("use_lookahead")) / len(decode) if decode else None,
        "blocked_by": dict(Counter(m["blocked_by"] for m in with_batch if m.get("blocked_by") not in (None, "none"))),
        "runner_blockers": {kind: sum(1 for m in runner_blocked if m.get(f"blocker_{kind}", 0) > 0) for kind in ("speech", "sampling", "logprob")
                            if any(m.get(f"blocker_{kind}", 0) > 0 for m in runner_blocked)},
        "flush_reasons": dict(Counter(m["flush_reason"] for m in steps if m.get("flush_reason") not in (None, "none"))),
        "batch_size_decode": stats([m["batch_size"] for m in decode]),
        "mode_share": {mode: k / n for mode, k in Counter(m.get("forward_mode") for m in with_batch).items()} if n else {},
        "cpu_ms": {part: stats([m[f"{part}_ns"] / 1e6 for m in steps if m.get(f"{part}_ns")]) for part in CPU_PARTS},
    }
    return out


def read_thinker_events(phase_dir: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted(phase_dir.rglob("*.jsonl")):
        if not re.search(r"events_(preprocessing|decode|thinker|coordinator)", path.name):
            continue
        with open(path) as fh:
            for line in fh:
                if "stream_chunk" in line or "coordinator_stream_received" in line:
                    continue
                if "exp_thinker_step" in line or "scheduler_prefill_end" in line or "model_path_end" in line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return out


def thinker_ms_per_token(events: list[dict[str, Any]], recs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    first: dict[tuple[str, str], int] = {}
    for e in events:
        if not (e.get("stage") or "").startswith("thinker"):
            continue
        key = (e.get("request_id"), e.get("event_name"))
        if e.get("event_name") == "model_path_end":
            first[key] = max(first.get(key, 0), e["timestamp_ns"])
        elif key not in first:
            first[key] = e["timestamp_ns"]
    vals = []
    for rid, r in recs.items():
        a, b = first.get((rid, "scheduler_prefill_end")), first.get((rid, "model_path_end"))
        toks = r.get("completion_tokens")
        if a and b and toks and toks > 1 and r.get("status") == "ok" and not r.get("warmup"):
            vals.append((b - a) / 1e6 / (toks - 1))
    return stats(vals)


def client_metrics(recs: list[dict[str, Any]]) -> dict[str, Any]:
    from omni_research.analysis.metrics import request_metrics

    ok = [r for r in recs if not r.get("warmup") and r.get("status") == "ok" and r.get("sent_ns") and r.get("done_ns")]
    m = [request_metrics(r) for r in ok]
    span = (max(r["done_ns"] for r in ok) - min(r["sent_ns"] for r in ok)) / 1e9 if ok else None
    return {"ok": len(ok), "errors": sum(1 for r in recs if not r.get("warmup") and r.get("status") != "ok"),
            "throughput_rps": len(ok) / span if span else None, "e2e_s": stats([x.get("e2e_s") for x in m]),
            "ttft_s": stats([x.get("ttft_s") for x in m]), "completion_tokens": stats([r.get("completion_tokens") for r in ok])}


def analyze_run(exp: Path) -> dict[str, Any]:
    import exp05_step_sync as ss
    from omni_research.analysis.experiment_analysis import read_jsonl

    res = json.load(open(exp / "result.json"))
    run: dict[str, Any] = {"run": exp.name, "experiment_id": res["experiment_id"], "phases": {}}
    for ph in res["phases"]:
        m = PHASE_RE.match(ph["phase"])
        if not m:
            continue
        cond, c, mode = m.group(1), int(m.group(2)), m.group(3)
        recs = read_jsonl(exp / "raw" / f"requests_{ph['phase']}.jsonl")
        row: dict[str, Any] = {"condition": cond, "concurrency": c, "mode": mode, "client": client_metrics(recs)}
        if mode == "B":
            events = read_thinker_events(exp / "raw" / "server_events" / ph["phase"])
            row["steps"] = step_summary(events)
            row["thinker_ms_per_token"] = thinker_ms_per_token(events, {r["request_id"]: r for r in recs if r.get("request_id")})
        else:
            per_proc = []
            for f in ss.procs_of(exp / "profiling" / "torch" / ph["phase"], "thinker"):
                s = ss.summarize(ss.steps_of(ss.load_trace(f)))
                if s:
                    per_proc.append(s)
            row["graph_steps"] = {k: float(np.median([p[k] for p in per_proc if p.get(k) is not None])) for k in
                                  ("steps", "span_ms_p50", "busy_ms_p50", "idle_ms_p50", "lead_ms_p50", "waited_share", "late_launch_share",
                                   "sync_share_of_period")} if per_proc else {}
        run["phases"][f"{cond}_c{c}_{mode}"] = row
    return run


def get(d: dict[str, Any], path: str) -> Any:
    for part in path.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def interval(values: list[float]) -> dict[str, Any]:
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    out: dict[str, Any] = {"runs": v, "n": len(v), "mean": float(np.mean(v)) if v else None, "ci95": None}
    if len(v) >= 2:
        half = T975.get(len(v) - 1, 2.0) * float(np.std(v, ddof=1)) / math.sqrt(len(v))
        out["ci95"] = [out["mean"] - half, out["mean"] + half]
    return out


METRICS = {"B": ["steps.decode_lookahead_share", "steps.lookahead_share", "steps.batch_size_decode.p50", "steps.cpu_ms.run.p50",
                 "steps.cpu_ms.resolve.p50", "steps.cpu_ms.launch.p50", "thinker_ms_per_token.p50", "client.throughput_rps",
                 "client.e2e_s.p50", "client.e2e_s.p95", "client.ttft_s.p50"],
           "C": ["graph_steps.idle_ms_p50", "graph_steps.busy_ms_p50", "graph_steps.span_ms_p50", "graph_steps.waited_share",
                 "graph_steps.lead_ms_p50", "graph_steps.steps", "client.throughput_rps"]}
CONTRASTS = [("w3t_ns", "w3t", "H8c：只差种子（文字）"), ("w3b_ns", "w3b", "H8d：只差种子（语音，阴性对照）"), ("w3t", "w3b", "H8a：文字对语音（都带种子）")]


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    levels = sorted({int(k.split("_c")[1].split("_")[0]) for r in runs for k in r["phases"]})
    out: dict[str, Any] = {"by_condition": {}, "contrasts": {}}
    for cond in CONDITIONS:
        for c in levels:
            for mode, metrics in METRICS.items():
                key = f"{cond}_c{c}_{mode}"
                out["by_condition"][key] = {mp: interval([get(r["phases"].get(key, {}), mp) for r in runs]) for mp in metrics}
    for a, b, label in CONTRASTS:
        for c in levels:
            for mode, metrics in METRICS.items():
                for mp in metrics:
                    diffs = []
                    for r in runs:
                        x, y = get(r["phases"].get(f"{a}_c{c}_{mode}", {}), mp), get(r["phases"].get(f"{b}_c{c}_{mode}", {}), mp)
                        diffs.append(x - y if x is not None and y is not None else None)
                    out["contrasts"].setdefault(f"{a}-{b}", {"label": label}).setdefault(f"c{c}", {})[f"{mode}:{mp}"] = interval(diffs)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    out = Path(a.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    runs = [analyze_run(root / name) for name in a.run]
    summary = {"generated_by": "scripts/analysis/exp08_lookahead.py", "runs": [r["run"] for r in runs], "per_run": runs, **summarize_runs(runs)}
    (out / "results" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float) + "\n")
    print(out / "results" / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
