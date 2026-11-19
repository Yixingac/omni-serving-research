#!/usr/bin/env python3
"""P0 of the pre-presentation validation: is the talker's admission cap of 32 what limits 2+1+5? Summary and verdicts.

    python scripts/analysis/ppv_p0_report.py --data-root <results root> --run 20261118_ppv_p0_cap32_block1_run01 ... \
        --out research/next_steps/presentation_validation/results/p0_summary.json

Rules registered in research/next_steps/p0_p3_experiment_plan.md (P0) and the pre-run addendum in
research/next_steps/presentation_validation/preregistration.md; algorithms reused from 实验 9 (exp09_report.py):
- capacity: completed requests / (last completion - first send) at the top closed-loop level without events
  (W3b 256: phase w3b_closed_A_c256; W3L 128: w3l_closed_A_c128); a level with errors + timeouts > 5 % or a server
  restart in its phase gives a missing value;
- open loop: steady when the least-squares slope of the in-flight count over the second half of the sending window is
  <= 0.05 x the offered rate; stable capacity = the highest steady rate with errors + timeouts <= 5 %;
- goodput: requests with TTFT <= 1 s and end-to-end / audio seconds <= 0.5 (docs/04_experiments/metrics.md);
- stage residence and the bottleneck stage from the events phase (w3b_closed_B_c256);
- talker scheduler state from SGLang's decode log lines (next_talker_slots.py heuristic: tokens per running request
  above 200 marks a talker line), restricted to each level's wall-clock window.
H0a: B's W3b capacity >= +5 % over A in all 3 blocks and B's talker running median > 32 at W3b 256 in every block ->
支持; < +2 % in all blocks -> 反驳; else 不确定. H0c: B - reference >= 0 in all blocks -> 支持; < 0 in all -> 反驳.
H0b (only if H0a is not 反驳): classify B's W3b 256 events window. Addendum G: admission effect = G vs A, graph-coverage
effect = B vs G; >= +5 % in all blocks "有贡献", within +-2 % in all blocks "没有可测的贡献", else "不确定".
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "scripts" / "slurm"))

RUN_RE = re.compile(r"^ppv_p0_(cap32|cap64|cap64_graph32|ref_t2_k2_c4|cap64_kv262k)_block(\d+)(?:_a\d+)?$")
ARM_OF = {"cap32": "A", "cap64": "B", "cap64_graph32": "G", "ref_t2_k2_c4": "R", "cap64_kv262k": "C"}
PLACEMENT = {"A": "t2_k1_c5", "B": "t2_k1_c5", "G": "t2_k1_c5", "C": "t2_k1_c5", "R": "t2_k2_c4"}
CAP = {"A": 32, "B": 64, "G": 64, "C": 64, "R": 32}
GRAPH_MAX = {"A": 32, "B": 64, "G": 32, "C": 64, "R": 32}
BLOCKS = (1, 2, 3)
CAPACITY_PHASE = {"w3b": "w3b_closed_A_c256", "w3l": "w3l_closed_A_c128"}
EVENTS_PHASE = "w3b_closed_B_c256"
SLO = {"ttft_s": 1.0, "e2e_per_audio_s": 0.5}
STAGES = ("thinker", "talker", "code2wav")
DECODE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(?P<ms>\d{3}) .*Decode batch, #running-req: (?P<run>\d+), "
    r"#token: (?P<tok>\d+), token usage: (?P<usage>[0-9.]+),.*cuda graph: (?P<graph>True|False), "
    r"gen throughput \(token/s\): (?P<gen>[0-9.]+), #queue-req: (?P<queue>\d+)")
CAPTURE = re.compile(r"Capture target decode CUDA graph begin\..*bs=\[(?P<bs>[0-9, ]+)\]")
EVENT_FILE = re.compile(r"^events_(?P<stage>[a-z0-9_]+?)(?:@r\d+)?_(?P<pid>\d+)\.jsonl$")
TALKER_MIN_TOKENS_PER_REQ = 200.0


# ---------------------------------------------------------------- pure helpers (tested)

def parse_decode(lines, utc_offset_s: float) -> list[dict[str, Any]]:
    """Decode-batch lines; the log's naive local time is converted with the run's UTC offset (manifest)."""
    out = []
    for line in lines:
        m = DECODE.match(line)
        if not m:
            continue
        naive = dt.datetime.strptime(m.group("date"), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
        out.append({"ts": naive.timestamp() - utc_offset_s + int(m.group("ms")) / 1e3, "running": int(m.group("run")),
                    "tokens": int(m.group("tok")), "usage": float(m.group("usage")), "graph": m.group("graph") == "True",
                    "gen_tok_s": float(m.group("gen")), "queue": int(m.group("queue"))})
    return out


def is_talker(row: dict[str, Any]) -> bool:
    return row["running"] > 0 and row["tokens"] / row["running"] > TALKER_MIN_TOKENS_PER_REQ


def capture_max_bs(lines) -> list[int]:
    """Largest captured decode batch of every CUDA-graph capture in the log (one per SGLang engine process)."""
    out = []
    for line in lines:
        m = CAPTURE.search(line)
        if m:
            out.append(max(int(x) for x in m.group("bs").split(",") if x.strip()))
    return sorted(out)


def talker_window(rows: list[dict[str, Any]], cap: int, graph_max: int) -> dict[str, Any]:
    """Scheduler state of the talker lines in one window: admission cap, queue, KV usage, CUDA-graph use by batch size."""
    n = len(rows)
    if not n:
        return {"lines": 0}
    small = [r for r in rows if r["running"] <= graph_max]
    big = [r for r in rows if r["running"] > graph_max]
    return {
        "lines": n,
        "running_p50": statistics.median(r["running"] for r in rows),
        "running_max": max(r["running"] for r in rows),
        "above_32_share": sum(r["running"] > 32 for r in rows) / n,
        "at_cap_share": sum(r["running"] >= cap for r in rows) / n,
        "at_cap_with_queue_share": sum(r["running"] >= cap and r["queue"] > 0 for r in rows) / n,
        "queue_p50": statistics.median(r["queue"] for r in rows),
        "queue_max": max(r["queue"] for r in rows),
        "usage_p50": statistics.median(r["usage"] for r in rows),
        "usage_max": max(r["usage"] for r in rows),
        # registered trigger of arm C: running below the cap, requests waiting, KV usage >= 0.9
        "kv_bound_share": sum(r["running"] < cap and r["queue"] > 0 and r["usage"] >= 0.9 for r in rows) / n,
        "graph_share_le_graph_max": sum(r["graph"] for r in small) / len(small) if small else None,
        "graph_share_above_graph_max": sum(r["graph"] for r in big) / len(big) if big else None,
        "lines_above_graph_max": len(big),
    }


def step_curve(rows: list[dict[str, Any]], min_lines: int = 20, min_gen: float = 50.0) -> list[dict[str, Any]]:
    """Median decode throughput per running batch size and the step time R / G; idle-gap lines (G < min_gen) dropped."""
    by: dict[int, list[float]] = {}
    for r in rows:
        if r["gen_tok_s"] >= min_gen:
            by.setdefault(r["running"], []).append(r["gen_tok_s"])
    out = []
    for bs in sorted(by):
        if len(by[bs]) >= min_lines:
            med = statistics.median(by[bs])
            out.append({"batch": bs, "lines": len(by[bs]), "gen_tok_s_median": med, "step_ms": 1000.0 * bs / med})
    return out


def goodput(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """Measured requests meeting TTFT <= 1 s and e2e / audio s <= 0.5, as a share and per second of the level."""
    meas = [r for r in recs if not r.get("warmup")]
    met = 0
    for r in meas:
        if r.get("status") != "ok" or r.get("sent_ns") is None or r.get("done_ns") is None or not r.get("output_audio_s"):
            continue
        ttft = (r["first_text_ns"] - r["sent_ns"]) / 1e9 if r.get("first_text_ns") is not None else None
        e2e = (r["done_ns"] - r["sent_ns"]) / 1e9
        if ttft is not None and ttft <= SLO["ttft_s"] and e2e / r["output_audio_s"] <= SLO["e2e_per_audio_s"]:
            met += 1
    sent = [r["sent_ns"] for r in meas if r.get("sent_ns") is not None]
    done = [r["done_ns"] for r in meas if r.get("done_ns") is not None]
    span = (max(done) - min(sent)) / 1e9 if sent and done else None
    return {"met": met, "measured": len(meas), "share": met / len(meas) if meas else None,
            "goodput_rps": met / span if span else None}


def rel_by_block(caps: dict[tuple[str, int], float], arm: str, base: str) -> list[float | None]:
    return [caps[(arm, b)] / caps[(base, b)] - 1 if (arm, b) in caps and (base, b) in caps and caps[(base, b)] else None
            for b in BLOCKS]


def t_interval(values: list[float | None]) -> dict[str, Any]:
    v = [x for x in values if x is not None]
    out: dict[str, Any] = {"by_block": values, "n": len(v), "mean": float(np.mean(v)) if v else None, "ci95": None}
    if len(v) >= 2:
        t = {1: 12.706, 2: 4.303, 3: 3.182}.get(len(v) - 1, 2.0)
        half = t * float(np.std(v, ddof=1)) / math.sqrt(len(v))
        out["ci95"] = [out["mean"] - half, out["mean"] + half]
    return out


def verdict_h0a(rel: list[float | None], running_p50: list[float | None]) -> str:
    if all(x is not None and x >= 0.05 for x in rel) and all(r is not None and r > 32 for r in running_p50):
        return "支持"
    if all(x is not None and x < 0.02 for x in rel):
        return "反驳"
    return "不确定"


def verdict_h0c(rel: list[float | None]) -> str:
    if all(x is not None and x >= 0 for x in rel):
        return "支持"
    if all(x is not None and x < 0 for x in rel):
        return "反驳"
    return "不确定"


def effect_label(rel: list[float | None]) -> str:
    if all(x is not None and x >= 0.05 for x in rel):
        return "有贡献"
    if all(x is not None and abs(x) <= 0.02 for x in rel):
        return "没有可测的贡献"
    return "不确定"


def classify_h0b(window: dict[str, Any], c_triggered: bool) -> list[str]:
    """Registered H0b classes for B's W3b 256 events window (several can hold at once)."""
    out = []
    bn = window.get("bottleneck") or {}
    talker_gpu = (window.get("gpu") or {}).get("talker", {}).get("busy")
    talker_cpu = window.get("talker_cpu_percent")
    if bn.get("stage") == "talker" and talker_cpu is not None and talker_cpu >= 95 and talker_gpu is not None and talker_gpu < 0.80:
        out.append("CPU/调度受限")
    if talker_gpu is not None and talker_gpu >= 0.90:
        out.append("GPU 受限")
    if c_triggered:
        out.append("KV 预留受限")
    if bn.get("stage") == "code2wav":
        out.append("瓶颈转移到 code2wav")
    return out or ["未归类"]


# ---------------------------------------------------------------- reading one run

def stage_pids(exp: Path) -> dict[str, list[int]]:
    """Stage -> process ids, from the server-event file names (events_<stage>[@rK]_<pid>.jsonl)."""
    out: dict[str, set[int]] = {}
    for f in (exp / "raw" / "server_events").rglob("events_*.jsonl"):
        m = EVENT_FILE.match(f.name)
        if m:
            out.setdefault(m.group("stage"), set()).add(int(m.group("pid")))
    return {k: sorted(v) for k, v in out.items()}


def cpu_of(samples: list[dict[str, Any]], pids: list[int], win: tuple[float, float] | None) -> float | None:
    """Mean CPU % (100 = one core) of the busiest of the given processes within the window."""
    if not win or not pids:
        return None
    per: dict[int, list[float]] = {p: [] for p in pids}
    for s in samples:
        t = s["t_wall_ns"] / 1e9
        if not (win[0] <= t <= win[1]):
            continue
        by_pid = {}
        for g in (s.get("groups") or {}).values():
            if isinstance(g, dict):
                by_pid.update(g.get("by_pid") or {})
        for p in pids:
            per[p].append(float(by_pid.get(str(p), 0.0)))
    means = [float(np.mean(v)) for v in per.values() if v]
    return max(means) if means else None


def analyze_run(exp: Path) -> dict[str, Any]:
    import exp06_c2w_breakdown as bd
    from exp01_gpu_timeline import load_util, utc_offset_s
    from exp06_report import phase_window, read_events
    from exp07_report import load_clocks, talker_times
    from exp09_report import bottleneck, gpu_window, level_client, queue_growth, stage_residence, usable
    from make_plan_ppv_p0 import PLACEMENT_GPUS
    from run_records import clean, restart_flags
    from omni_research.analysis.experiment_analysis import clock_anchor, read_jsonl

    res = json.load(open(exp / "result.json"))
    m = RUN_RE.match(res["experiment_id"])
    if not m:
        raise SystemExit(f"not a P0 run: {res['experiment_id']}")
    arm, block = ARM_OF[m.group(1)], int(m.group(2))
    thinker, talker_gpus, c2w = PLACEMENT_GPUS[PLACEMENT[arm]]
    roles = {"thinker": thinker, "talker": talker_gpus, "code2wav": c2w}
    util, clocks, anchor, off = load_util(exp), load_clocks(exp), clock_anchor(exp), utc_offset_s(exp)
    with open(exp / "logs" / "server_s0.log", errors="ignore") as fh:
        log_lines = fh.readlines()
    decode = [r for r in parse_decode(log_lines, off) if is_talker(r)]
    cpu = read_jsonl(exp / "raw" / "cpu_samples.jsonl") if (exp / "raw" / "cpu_samples.jsonl").exists() else []
    pids = stage_pids(exp)
    talker_pids = pids.get("talker", [])
    run: dict[str, Any] = {"run": exp.name, "experiment_id": res["experiment_id"], "arm": arm, "block": block,
                           "status": res.get("status"), "restarts": res.get("server_restarts") or [],
                           "graph_capture_max_bs": capture_max_bs(log_lines), "talker_pids": talker_pids,
                           "levels": [], "capacity": {}, "step_curve": step_curve(decode)}
    flags = restart_flags([ph["phase"] for ph in res.get("phases", [])], run["restarts"])
    for ph in res.get("phases", []):
        phase = ph["phase"]
        path = exp / "raw" / f"requests_{phase}.jsonl"
        if phase.startswith("smoke") or not path.exists():
            continue
        restart_here, _ = flags[phase]
        recs, _ = clean(read_jsonl(path))
        events = read_events(exp / "raw" / "server_events" / phase) if ph.get("profiled") == "request_events" else None
        key = "concurrency" if ph.get("kind") == "closed_loop" else "rate_rps"
        for level in sorted({r.get(key) for r in recs if r.get(key) is not None}):
            lv = [r for r in recs if r.get(key) == level]
            meas = [r for r in lv if not r.get("warmup")]
            win = phase_window(meas, anchor)
            row = {"phase": phase, "workload": phase.split("_")[0], "kind": ph.get("kind"), key: level,
                   **level_client(lv), "goodput": goodput(lv), "gpu": gpu_window(util, clocks, roles, win),
                   "talker": talker_window([r for r in decode if win and win[0] <= r["ts"] <= win[1]],
                                           CAP[arm], GRAPH_MAX[arm]),
                   "talker_cpu_percent": cpu_of(cpu, talker_pids, win), "restart_here": restart_here}
            row["usable"] = usable(row["client"]) and not restart_here
            if key == "rate_rps":
                row["growth"] = queue_growth(meas, float(level))
            if events is not None:
                ids = {r["request_id"] for r in meas if r.get("request_id")}
                row["residence_ms"] = stage_residence(events, ids)
                row["bottleneck"] = bottleneck(row["residence_ms"])
                tt = talker_times(events)
                row["talker_queue_ms"] = bd.stats([tt[i]["queue_ms"] for i in ids if i in tt])
                _, batches, _ = bd.breakdown(events)
                row["code2wav_batch_size"] = bd.stats([b.get("size") for b in batches.values()])
            run["levels"].append(row)
    for w, phase in CAPACITY_PHASE.items():
        row = next((x for x in run["levels"] if x["phase"] == phase), None)
        if row is not None:
            run["capacity"][w] = row["client"].get("throughput_rps") if row["usable"] else None
    run["stable_open_capacity"] = {}
    for w in ("w3b", "w3l"):
        steady = [x["rate_rps"] for x in run["levels"] if x["workload"] == w and x["kind"] == "open_loop"
                  and x.get("growth", {}).get("steady") and x["usable"]]
        tested = [x["rate_rps"] for x in run["levels"] if x["workload"] == w and x["kind"] == "open_loop"]
        if tested:
            run["stable_open_capacity"][w] = {"rate_rps": max(steady) if steady else None, "lowest_tested": min(tested),
                                              "highest_tested": max(tested)}
    return run


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    caps = {w: {(r["arm"], r["block"]): r["capacity"].get(w) for r in runs if r["capacity"].get(w) is not None}
            for w in CAPACITY_PHASE}
    out: dict[str, Any] = {"capacity": {w: {f"{a}{b}": v for (a, b), v in c.items()} for w, c in caps.items()}}
    w = "w3b"
    rel_ba = rel_by_block(caps[w], "B", "A")
    running_b = []
    for b in BLOCKS:
        row = next((x for r in runs if r["arm"] == "B" and r["block"] == b for x in r["levels"]
                    if x["phase"] == CAPACITY_PHASE[w]), None)
        running_b.append(row["talker"].get("running_p50") if row and row["talker"].get("lines") else None)
    out["H0a"] = {"rule": "B/A - 1 >= 0.05 in all blocks and B talker running median > 32 at W3b 256",
                  "relative": t_interval(rel_ba), "b_running_p50": running_b, "verdict": verdict_h0a(rel_ba, running_b)}
    rel_br = rel_by_block(caps[w], "B", "R")
    out["H0c"] = {"rule": "B/R - 1 >= 0 in all blocks", "relative": t_interval(rel_br), "verdict": verdict_h0c(rel_br)}
    rel_ga, rel_bg = rel_by_block(caps[w], "G", "A"), rel_by_block(caps[w], "B", "G")
    out["G_decomposition"] = {"admission_G_vs_A": {**t_interval(rel_ga), "label": effect_label(rel_ga)},
                              "graph_B_vs_G": {**t_interval(rel_bg), "label": effect_label(rel_bg)}}
    high = [x for r in runs if r["arm"] == "B" for x in r["levels"]
            if x["workload"] == "w3b" and (x["phase"] in (CAPACITY_PHASE["w3b"], EVENTS_PHASE)
                                           or (x["kind"] == "open_loop" and x.get("rate_rps", 0) >= 26))]
    lines = sum(x["talker"].get("lines", 0) for x in high)
    kv = sum(x["talker"].get("kv_bound_share", 0) * x["talker"].get("lines", 0) for x in high)
    out["C_trigger"] = {"rule": ">= 10 % of B's high-load talker lines with running < 64, queue > 0, usage >= 0.9",
                        "lines": lines, "share": kv / lines if lines else None,
                        "triggered": bool(lines and kv / lines >= 0.10)}
    if out["H0a"]["verdict"] != "反驳":
        cls = []
        for b in BLOCKS:
            row = next((x for r in runs if r["arm"] == "B" and r["block"] == b for x in r["levels"] if x["phase"] == EVENTS_PHASE), None)
            cls.append(classify_h0b(row, out["C_trigger"]["triggered"]) if row else None)
        out["H0b"] = {"by_block": cls}
    for wl in CAPACITY_PHASE:
        out[f"wl_{wl}_relative"] = {"B_vs_A": t_interval(rel_by_block(caps[wl], "B", "A"))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    runs = [analyze_run(root / r) for r in a.run]
    doc = {"generated_by": "scripts/analysis/ppv_p0_report.py", "runs": runs, "summary": summarize(runs)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(doc["summary"], ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
