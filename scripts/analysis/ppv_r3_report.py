#!/usr/bin/env python3
"""R3 (pre-presentation validation): where does continuous camera input stop on the request-response path?

    python scripts/analysis/ppv_r3_report.py --data-root <PROJECT_DATA_ROOT> --run <run dir> [--run ...] \
        --out research/next_steps/presentation_validation/results/r3_camera.json

Per camera phase (sessions.camera plugin, one 1-GPU text-only replica, runtime unicorn_ppv_kv):
- client: drop ratio (ticks skipped because the session's previous request was still running), late ratio (answer
  after the next tick), first-response and completion percentiles, counting accuracy;
- server events (request events on): per request preprocessing wait (stage_input_received -> stage_dispatch) and
  service, image-encoder service, thinker queue and prefill; with the patch's exp_preprocess event also the split
  of preprocessing into media load (decode, frame sampling), cache keys, video frames, and the processor call
  (slicing/resize on the GPU, tokenization);
- preprocessing slot utilisation = summed preprocessing service time / (window x max_concurrency);
- CPU of the process that hosts preprocessing (per-pid sampler) and GPU busy of the replica's GPU (nvidia-smi).
Classification (registered in r3_r4_design.md, applied per cell):
  saturated = drop ratio >= 0.10 or late ratio >= 0.20;
  preprocessing-bound = saturated and preprocessing (wait + service) >= 50 % of the server time and slot
    utilisation >= 0.8;
  gpu-bound = saturated and GPU busy >= 80 % and slot utilisation < 0.8; otherwise other/unclear.
Improvement of max_concurrency 4 over 1 (paired cells): drop ratio lower by >= 5 points and >= 20 % relative, or
first-response P95 lower by >= 20 %.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SATURATED_DROP, SATURATED_LATE = 0.10, 0.20
PREPROC_SHARE, SLOT_UTIL, GPU_BUSY = 0.5, 0.8, 80.0
IMPROVE_DROP_PTS, IMPROVE_REL, IMPROVE_P95 = 0.05, 0.20, 0.20


def pct(values: list[float], q: float) -> float | None:
    v = sorted(x for x in values if x is not None)
    return v[min(len(v) - 1, int(q * len(v)))] if v else None


def client_stats(ticks: list[dict[str, Any]], recs: list[dict[str, Any]]) -> dict[str, Any]:
    meas = [t for t in ticks if not t.get("warmup")]
    sent = [r for r in recs if not r.get("warmup")]
    ok = [r for r in sent if r.get("status") == "ok" and r.get("first_text_ns") and r.get("sent_ns")]
    first = [(r["first_text_ns"] - r["sent_ns"]) / 1e9 for r in ok]
    done = [(r["done_ns"] - r["sent_ns"]) / 1e9 for r in ok if r.get("done_ns")]
    n = len(meas)
    dropped = sum(1 for t in meas if t.get("dropped"))
    return {"ticks": n, "sent": len(sent), "dropped": dropped, "drop_ratio": dropped / n if n else None,
            "late_ratio": sum(1 for r in sent if r.get("late")) / len(sent) if sent else None,
            "errors": sum(1 for r in sent if r.get("status") != "ok"),
            "first_response_s": {"p50": pct(first, 0.5), "p95": pct(first, 0.95), "p99": pct(first, 0.99) if len(first) >= 100 else None,
                                 "n": len(first)},
            "completion_s": {"p50": pct(done, 0.5), "p95": pct(done, 0.95), "n": len(done)},
            "answer_correct_share": sum(1 for r in ok if r.get("answer_correct")) / len(ok) if ok else None,
            "prompt_tokens_p50": pct([r.get("prompt_tokens") for r in ok], 0.5)}


def server_stats(events: list[dict[str, Any]], ids: set[str], window_s: float, max_concurrency: int) -> dict[str, Any]:
    """Medians over the phase's measured requests, from raw request events (ms)."""
    from omni_research.analysis.experiment_analysis import stage_breakdown

    per = stage_breakdown(events)["per_request"]
    first: dict[tuple[str, str, str], int] = {}
    pre: dict[str, dict[str, Any]] = {}
    for e in events:
        rid, st, name, t = e.get("request_id"), e.get("stage"), e.get("event_name"), e.get("timestamp_ns")
        if rid not in ids or t is None:
            continue
        k = (rid, st, name)
        first[k] = min(first.get(k, t), t)
        if name == "exp_preprocess":
            pre[rid] = e.get("metadata") or {}
    wait = {rid: (first[(rid, "preprocessing", "stage_dispatch")] - first[(rid, "preprocessing", "stage_input_received")]) / 1e6
            for rid in ids if (rid, "preprocessing", "stage_dispatch") in first
            and (rid, "preprocessing", "stage_input_received") in first}
    rows = {rid: per[rid] for rid in ids if rid in per}

    def med(key: str) -> float | None:
        return pct([d.get(key) for d in rows.values()], 0.5)

    service = [d.get("preprocessing.service") for d in rows.values() if d.get("preprocessing.service") is not None]
    total = med("request.admission_to_first_terminal")
    pre_med = {k: pct([m.get(k) / 1e6 for m in pre.values() if m.get(k) is not None], 0.5)
               for k in ("media_load_ns", "cache_key_ns", "video_frames_ns", "processor_ns", "inline_media_load_ns", "total_ns")}
    share = None
    if total and wait and service:
        share = ((pct(list(wait.values()), 0.5) or 0.0) + (pct(service, 0.5) or 0.0)) / total
    return {"requests_with_events": len(rows), "preprocessing_wait_ms_p50": pct(list(wait.values()), 0.5),
            "preprocessing_wait_ms_p95": pct(list(wait.values()), 0.95), "preprocessing_service_ms_p50": pct(service, 0.5),
            "image_encoder_service_ms_p50": med("image_encoder.service"), "thinker_queue_ms_p50": med("thinker.queue"),
            "thinker_prefill_ms_p50": med("thinker.prefill"), "server_first_terminal_ms_p50": total,
            "preprocessing_share_of_server_time": share,
            "preprocessing_slot_utilisation": (sum(service) / 1e3) / (window_s * max_concurrency) if window_s and service else None,
            "preprocess_split_ms_p50": {k.removesuffix("_ns"): v for k, v in pre_med.items() if v is not None},
            "preprocess_inflight_max": max((m.get("inflight_at_start") or 0 for m in pre.values()), default=None)}


def classify(client: dict[str, Any], server: dict[str, Any], gpu_busy: float | None) -> str:
    drop, late = client.get("drop_ratio") or 0.0, client.get("late_ratio") or 0.0
    if drop < SATURATED_DROP and late < SATURATED_LATE:
        return "not_saturated"
    share, util = server.get("preprocessing_share_of_server_time"), server.get("preprocessing_slot_utilisation")
    if share is not None and util is not None and share >= PREPROC_SHARE and util >= SLOT_UTIL:
        return "preprocessing_bound"
    if gpu_busy is not None and gpu_busy >= GPU_BUSY and (util is None or util < SLOT_UTIL):
        return "gpu_bound"
    return "saturated_other"


def improved(c1: dict[str, Any], c4: dict[str, Any]) -> bool:
    d1, d4 = c1.get("drop_ratio"), c4.get("drop_ratio")
    if d1 is not None and d4 is not None and d1 - d4 >= IMPROVE_DROP_PTS and (d1 == 0 or (d1 - d4) / d1 >= IMPROVE_REL):
        return True
    p1, p4 = (c1.get("first_response_s") or {}).get("p95"), (c4.get("first_response_s") or {}).get("p95")
    return bool(p1 and p4 is not None and (p1 - p4) / p1 >= IMPROVE_P95)


def confirm_cells(cells: dict[str, dict[str, dict[str, Any]]], limit: int = 6) -> list[str]:
    """Registered selection for the 3-block confirmation: cells where concurrency 1 is saturated or 4 improves on it,
    most-dropping first (ties: more sessions), at most ``limit``."""
    chosen = []
    for key, by in cells.items():
        c1, c4 = by.get("1"), by.get("4")
        if not c1:
            continue
        sat = (c1.get("drop_ratio") or 0) >= SATURATED_DROP or (c1.get("late_ratio") or 0) >= SATURATED_LATE
        if sat or (c4 and improved(c1, c4)):
            chosen.append((c1.get("drop_ratio") or 0.0, c1.get("sessions", 0), key))
    chosen.sort(reverse=True)
    return [k for *_, k in chosen[:limit]]


CONFIRM_RE = re.compile(r"ppv_r3_confirm_seq_c(\d+)_block(\d+)")
BLOCKS = (1, 2, 3)


def cell_key(c: dict[str, Any]) -> str:
    return f"{c['shape']}_s{c['sessions']}_f{c['fps']:g}"


def confirm_verdicts(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Registered (r3_r4_design.md 2.4): per confirmation cell, concurrency 4 is compared with 1 inside each block;
    支持 when it improves in all 3 blocks, 不支持 when in none, 不确定 otherwise; a missing block or arm is 数据不足.
    Also, per block, concurrency 1's class and both arms' drop ratio and first-response P95 (descriptive)."""
    by: dict[str, dict[int, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for run in runs:
        m = CONFIRM_RE.search(run.get("experiment_id") or "")
        if not m:
            continue
        for c in run["cells"]:
            by[cell_key(c)][int(m.group(2))][str(c["max_concurrency"])] = c
    out: dict[str, Any] = {}
    for key, blocks in sorted(by.items()):
        imp = {str(b): (improved(blocks[b]["1"], blocks[b]["4"]) if "1" in blocks.get(b, {}) and "4" in blocks.get(b, {}) else None)
               for b in BLOCKS}
        vals = list(imp.values())
        verdict = ("数据不足" if any(v is None for v in vals) else "支持" if all(vals) else "不支持" if not any(vals) else "不确定")
        col = lambda conc, f: {str(b): (f(blocks[b][conc]) if conc in blocks.get(b, {}) else None) for b in BLOCKS}  # noqa: E731
        out[key] = {"verdict": verdict, "improved_by_block": imp,
                    "class_c1_by_block": col("1", lambda c: c.get("class")),
                    "drop_ratio": {"c1": col("1", lambda c: c.get("drop_ratio")), "c4": col("4", lambda c: c.get("drop_ratio"))},
                    "first_response_p95_s": {"c1": col("1", lambda c: (c.get("first_response_s") or {}).get("p95")),
                                             "c4": col("4", lambda c: (c.get("first_response_s") or {}).get("p95"))},
                    "sent_per_s": {conc: col(str(n), lambda c: c.get("sent") / 60.0 if c.get("sent") is not None else None)
                                   for conc, n in (("c1", 1), ("c4", 4))},
                    "offered_per_s": next((c["sessions"] * c["fps"] for b in blocks.values() for c in b.values()), None),
                    "gpu_power_mean_w_c1": col("1", lambda c: c.get("gpu_power_mean_w"))}
    return out


def gpu_evidence(win: dict[str, dict[str, Any]], gpus: list[int]) -> dict[str, Any]:
    """Descriptive GPU columns next to gpu_busy_mean (the registered class still uses only gpu_busy_mean): nvidia-smi
    utilisation aliases under periodic per-session loads (seen in R0), so mean power and the share of 0 % readings are
    reported, and the cell is flagged when utilisation is unusable (duplex.report.UTIL_SUSPECT_*)."""
    g = [win[str(i)] for i in gpus if str(i) in win]
    if not g:
        return {}
    power = [x["power_mean_w"] for x in g if x.get("power_mean_w") is not None]
    return {"gpu_power_mean_w": statistics.fmean(power) if power else None,
            "gpu_util_zero_share": statistics.fmean(x["util_zero_share"] for x in g),
            "gpu_util_suspect": any(x["util_suspect"] for x in g)}


def analyze_run(exp: Path) -> dict[str, Any]:
    from exp01_gpu_timeline import load_util
    from omni_research.analysis.experiment_analysis import clock_anchor, read_jsonl
    from omni_research.duplex.report import gpu_window, load_gpu_samples_utc

    cfg = json.loads((exp / "config.json").read_text())
    res = json.loads((exp / "result.json").read_text())
    gpus = sorted({g for s in cfg.get("servers", []) for g in s.get("gpus", [])})
    anchor = clock_anchor(exp)
    off = (anchor[1] - anchor[0]) / 1e9 if anchor else None
    util = load_util(exp) if (exp / "raw" / "gpu_samples.csv").exists() else {}
    samples = load_gpu_samples_utc(exp) if (exp / "raw" / "gpu_samples.csv").exists() else []
    cpu = read_jsonl(exp / "raw" / "cpu_samples.jsonl") if (exp / "raw" / "cpu_samples.jsonl").exists() else []
    phases_cfg = {p.get("name"): p for p in cfg.get("phases", [])}
    out = {"run": exp.name, "experiment_id": res.get("experiment_id"), "cells": []}
    for ph in res.get("phases", []):
        name = ph["phase"]
        tick_path = exp / "raw" / f"camera_{name}.jsonl"
        if not tick_path.exists():
            continue
        pc = phases_cfg.get(name, {})
        ticks, recs = read_jsonl(tick_path), read_jsonl(exp / "raw" / f"requests_{name}.jsonl")
        client = client_stats(ticks, recs)
        meas = [r for r in recs if not r.get("warmup") and r.get("sent_ns")]
        cell = {"phase": name, "sessions": pc.get("sessions"), "fps": pc.get("fps"), "shape": pc.get("shape", "frame"),
                "max_concurrency": int(cfg.get("ppv_preprocessing_concurrency", 1)), **client}
        if meas and off is not None:
            a = min(r["sent_ns"] for r in meas) / 1e9 + off
            b = max(r.get("done_ns") or r["sent_ns"] for r in meas) / 1e9 + off
            busy = [u for g in gpus for t, u, _ in util.get(g, []) if a <= t <= b]
            cell["gpu_busy_mean"] = statistics.fmean(busy) if busy else None
            cell.update(gpu_evidence(gpu_window(samples, a, b), gpus))
            base = exp / "raw" / "server_events" / name
            events = [e for f in sorted(base.rglob("*.jsonl")) for e in read_jsonl(f)] if base.exists() else []
            pre_pids = {f.stem.rsplit("_", 1)[-1] for f in base.rglob("events_preprocessing_*.jsonl")} if base.exists() else set()
            cores = [sum(v for g in s.get("groups", {}).values() if isinstance(g, dict)
                         for pid, v in (g.get("by_pid") or {}).items() if pid in pre_pids) / 100
                     for s in cpu if a <= s["t_wall_ns"] / 1e9 <= b]
            cell["preprocessing_process_cores_mean"] = statistics.fmean(cores) if cores and pre_pids else None
            if events:
                cell["server"] = server_stats(events, {r["request_id"] for r in meas}, b - a, cell["max_concurrency"])
        cell["class"] = classify(client, cell.get("server") or {}, cell.get("gpu_busy_mean"))
        out["cells"].append(cell)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = [analyze_run(Path(a.data_root).expanduser() / "experiments" / r) for r in a.run]
    cells: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for run in runs:
        for c in run["cells"]:
            cells[f"{c['shape']}_s{c['sessions']}_f{c['fps']:g}"][str(c["max_concurrency"])] = c
    if any(CONFIRM_RE.search(r.get("experiment_id") or "") for r in runs):
        # confirmation runs: never pool blocks; the registered verdict is per cell over the three blocks
        out = {"generated_by": "scripts/analysis/ppv_r3_report.py", "runs": runs, "confirmation": confirm_verdicts(runs)}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(json.dumps({k: v["verdict"] for k, v in out["confirmation"].items()}, ensure_ascii=False))
        return 0
    out = {"generated_by": "scripts/analysis/ppv_r3_report.py", "runs": runs,
           "improved_by_concurrency_4": {k: improved(v["1"], v["4"]) for k, v in cells.items() if "1" in v and "4" in v},
           "confirm_cells": confirm_cells(cells)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({"confirm_cells": out["confirm_cells"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
