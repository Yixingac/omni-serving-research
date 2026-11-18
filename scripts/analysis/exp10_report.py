#!/usr/bin/env python3
"""实验 10（code2wav 的批大小和组批方式）的汇总和判定。

    python scripts/analysis/exp10_report.py --data-root <results root> --run 20261118_exp10_graph_block1_run01 ... \
        --check-run 20261118_exp10_check_run01 --out research/01_性能瓶颈/实验10_Code2Wav_Batching

Per measured phase (<workload>_<setting>, the setting written by the preceding control phase): completed audio
seconds per second (client), code2wav GPU time per effective audio second (sum of flow + HiFT CUDA event spans over
the batches / their requests' audio seconds; audio = codec tokens x 0.04 s), frame-weighted padding of the flow
(prompt + codec tokens padded to the longest row), batch sizes, CUDA graph share, queue waits by length class,
whether every batch ran with the phase's setting, and whether every request got the audio length of its own codec
tokens. Verdicts H10a-H10e as registered in hypotheses.md. The correctness run pairs requests with the same codec
token hash across phases and compares waveform statistics.
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

SECONDS_PER_CODEC_TOKEN = 0.04  # 24 kHz, 960 samples per codec token
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
PHASE_RE = re.compile(r"^(w3b|mix)_(fifo\d+|length\d+)$")
RUN_RE = re.compile(r"exp10_(graph|nograph)_block(\d+)")


# ---------------------------------------------------------------- per-batch measures

def padding_ratio(batches: list[dict[str, Any]]) -> float | None:
    padded = effective = 0.0
    for b in batches:
        toks = b.get("token_lengths") or []
        if not toks:
            continue
        prompts = b.get("prompt_token_lengths") or [0] * len(toks)
        rows = [t + (p or 0) for t, p in zip(toks, prompts)]
        padded += len(rows) * max(rows)
        effective += sum(rows)
    return 1 - effective / padded if padded else None


def gpu_ms_per_audio_s(batches: list[dict[str, Any]]) -> float | None:
    gpu = audio = 0.0
    for b in batches:
        if b.get("gpu_flow_ms") is None or b.get("gpu_hift_ms") is None or not b.get("token_lengths"):
            continue
        gpu += b["gpu_flow_ms"] + b["gpu_hift_ms"]
        audio += sum(b["token_lengths"]) * SECONDS_PER_CODEC_TOKEN
    return gpu / audio if audio else None


def graph_share(batches: list[dict[str, Any]]) -> float | None:
    paths = [b.get("flow_solve_path") for b in batches if b.get("flow_solve_path") not in (None, "none")]
    return sum(p.endswith("_graph") for p in paths) / len(paths) if paths else None


def control_check(batches: list[dict[str, Any]], cap: int, policy: str) -> dict[str, Any]:
    seen = [b for b in batches if b.get("batch_cap") is not None]
    bad = sum(1 for b in seen if b["batch_cap"] != cap or b.get("batch_policy") != policy)
    return {"batches": len(seen), "mismatched": bad, "ok": bool(seen) and bad == 0}


def correspondence(batches: list[dict[str, Any]], client: dict[str, dict[str, Any]]) -> dict[str, int]:
    checked = mismatched = 0
    for b in batches:
        for rid, tokens in zip(b.get("request_ids", []), b.get("request_codec_tokens") or b.get("token_lengths") or []):
            rec = client.get(rid)
            if not rec or rec.get("status") != "ok" or rec.get("output_audio_s") is None:
                continue
            checked += 1
            if abs(rec["output_audio_s"] - tokens * SECONDS_PER_CODEC_TOKEN) > SECONDS_PER_CODEC_TOKEN + 1e-9:
                mismatched += 1
    return {"checked": checked, "mismatched": mismatched}


def wait_by_length(reqs: dict[str, dict[str, Any]], audio: dict[str, float]) -> dict[str, Any]:
    ids = [r for r in reqs if r in audio and reqs[r].get("queue_ms") is not None]
    if not ids:
        return {"split_s": None, "short": None, "long": None}
    split = float(np.median([audio[r] for r in ids]))

    def summary(sel: list[str]) -> dict[str, Any] | None:
        v = [reqs[r]["queue_ms"] for r in sel]
        return {"n": len(v), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "max": float(max(v))} if v else None

    return {"split_s": split, "short": summary([r for r in ids if audio[r] < split]), "long": summary([r for r in ids if audio[r] >= split])}


def compare_waves(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pair requests with the same codec-token hash; registered measure sqrt(|ss_a - ss_b|) / sqrt(ss_b)."""
    first_b: dict[str, list[float]] = {}
    for x in b.values():
        first_b.setdefault(x["sha"], x["stats"])
    pairs, mismatch, diffs, rms = 0, 0, [], []
    seen: set[str] = set()
    for x in a.values():
        if x["sha"] in seen or x["sha"] not in first_b:
            continue
        seen.add(x["sha"])
        sa, sb = x["stats"], first_b[x["sha"]]
        pairs += 1
        if sa[0] != sb[0]:
            mismatch += 1
            continue
        diffs.append(math.sqrt(abs(sa[2] - sb[2])) / math.sqrt(sb[2]) if sb[2] > 0 else 0.0)
        rms.append(abs(math.sqrt(sa[2]) - math.sqrt(sb[2])) / math.sqrt(sb[2]) if sb[2] > 0 else 0.0)
    return {"pairs": pairs, "sample_mismatch": mismatch, "rel_diff_max": max(diffs) if diffs else None,
            "rel_diff_p95": float(np.percentile(diffs, 95)) if diffs else None, "rms_rel_change_max": max(rms) if rms else None}


# ---------------------------------------------------------------- verdicts

def _per_block(runs: list[dict[str, Any]], graph: bool, w: str, setting: str, key: str) -> dict[int, float]:
    return {r["block"]: r["cells"][(w, setting)][key] for r in runs if r["graph"] == graph and (w, setting) in r["cells"]
            and r["cells"][(w, setting)].get(key) is not None}


def verdicts(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    blocks = sorted({r["block"] for r in runs})

    e1, e16 = _per_block(runs, False, "w3b", "fifo1", "gpu_ms_per_audio_s"), _per_block(runs, False, "w3b", "fifo16", "gpu_ms_per_audio_s")
    ratio = [e16[b] / e1[b] for b in blocks if b in e1 and b in e16]
    v = "数据不足" if len(ratio) < len(blocks) else "支持" if all(x >= 0.9 for x in ratio) else "反驳" if all(x <= 0.8 for x in ratio) else "不确定"
    out["H10a"] = {"verdict": v, "e16_over_e1_graph_off": ratio}

    pm, pw = _per_block(runs, False, "mix", "fifo16", "padding"), _per_block(runs, False, "w3b", "fifo16", "padding")
    both = [(pm[b], pw[b]) for b in blocks if b in pm and b in pw]
    if len(both) < len(blocks):
        v = "数据不足"
    elif all(m >= 0.15 and m >= w + 0.05 for m, w in both):
        v = "支持"
    elif all(m < 0.05 for m, _ in both):
        v = "反驳"
    else:
        v = "不确定"
    out["H10b"] = {"verdict": v, "padding_mix_w3b_graph_off": both}

    el, ef = _per_block(runs, True, "mix", "length16", "gpu_ms_per_audio_s"), _per_block(runs, True, "mix", "fifo16", "gpu_ms_per_audio_s")
    pl, pf = _per_block(runs, True, "mix", "length16", "padding"), _per_block(runs, True, "mix", "fifo16", "padding")
    rows = [(el[b], ef[b], pl.get(b), pf.get(b)) for b in blocks if b in el and b in ef]
    if len(rows) < len(blocks):
        v = "数据不足"
    elif all(lp is not None and fp is not None and lp < fp and le <= 0.95 * fe for le, fe, lp, fp in rows):
        v = "支持"
    elif all(le >= fe for le, fe, _, _ in rows):
        v = "反驳"
    else:
        v = "不确定"
    out["H10c"] = {"verdict": v, "length16_vs_fifo16_mix_graph_on": [{"e_length": a, "e_fifo": b, "pad_length": c, "pad_fifo": d} for a, b, c, d in rows]}

    ratios = []
    for b in blocks:
        cl, cf = (next((r["cells"].get(("mix", s)) for r in runs if r["graph"] and r["block"] == b), None) for s in ("length16", "fifo16"))
        if cl and cf and cf.get("wait_long_p95") and cf.get("wait_long_max"):
            ratios.append((cl["wait_long_p95"] / cf["wait_long_p95"], cl["wait_long_max"] / cf["wait_long_max"]))
    if len(ratios) < len(blocks):
        v = "数据不足"
    elif all(p >= 1.2 or m >= 1.2 for p, m in ratios):
        v = "支持"
    elif all(p < 1.2 and m < 1.2 for p, m in ratios):
        v = "不支持"
    else:
        v = "不确定"
    out["H10d"] = {"verdict": v, "long_wait_ratio_p95_max": ratios}

    gain = {}
    for s in ("fifo1", "fifo16"):
        g0, g1 = _per_block(runs, False, "w3b", s, "gpu_ms_per_audio_s"), _per_block(runs, True, "w3b", s, "gpu_ms_per_audio_s")
        gain[s] = [g0[b] / g1[b] - 1 for b in blocks if b in g0 and b in g1]
    if min(len(x) for x in gain.values()) < len(blocks):
        v = "数据不足"
    elif all(x >= 0.10 for x in gain["fifo1"]) and all(x <= 0.05 for x in gain["fifo16"]):
        v = "支持"
    elif all(x < 0.05 for x in gain["fifo1"]):
        v = "反驳"
    else:
        v = "不确定"
    out["H10e"] = {"verdict": v, "graph_gain": gain}
    return out


def interval(values: list[Any]) -> dict[str, Any]:
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    out: dict[str, Any] = {"runs": values, "n": len(v), "mean": float(np.mean(v)) if v else None, "ci95": None}
    if len(v) >= 2:
        half = T975.get(len(v) - 1, 2.0) * float(np.std(v, ddof=1)) / math.sqrt(len(v))
        out["ci95"] = [out["mean"] - half, out["mean"] + half]
    return out


# ---------------------------------------------------------------- reading runs

def attach_extra(events: list[dict[str, Any]], batches: dict[tuple, dict[str, Any]]) -> None:
    """Fields the 实验 10 patch adds: per batch the control setting and window, prompt lengths, digests."""
    for e in events:
        md = e.get("metadata") or {}
        name = e.get("event_name")
        if name not in ("exp_batch_formed", "exp_vocode") or md.get("batch_id") is None:
            continue
        b = batches.get((e.get("pid"), md["batch_id"]))
        if b is None:
            continue
        if name == "exp_batch_formed":
            for k in ("batch_cap", "batch_policy", "window_size"):
                if k in md:
                    b[k] = md[k]
        else:
            for k in ("prompt_token_lengths", "request_codec_tokens", "request_codec_sha256", "request_wave_stats"):
                if k in md:
                    b[k] = md[k]
            if "request_ids" in md:
                b["vocode_request_ids"] = md["request_ids"]


def phase_settings(res: dict[str, Any]) -> dict[str, tuple[str, int]]:
    out = {}
    for ph in res.get("phases", []):
        if ph.get("kind") == "control" or ph["phase"].startswith("ctl_"):
            c = ph.get("control") or {}
            out[ph["phase"].removeprefix("ctl_")] = (c.get("policy"), c.get("max_batch_size"))
    return out


def analyze_phase(exp: Path, phase: str, setting: tuple[str, int] | None) -> dict[str, Any]:
    import exp06_c2w_breakdown as bd
    from exp06_report import client_metrics, read_events
    from omni_research.analysis.experiment_analysis import read_jsonl

    recs = read_jsonl(exp / "raw" / f"requests_{phase}.jsonl")
    meas = [r for r in recs if not r.get("warmup")]
    events = read_events(exp / "raw" / "server_events" / phase)
    reqs, batches, issues = bd.breakdown(events)
    attach_extra(events, batches)
    ids = {r["request_id"] for r in meas if r.get("request_id")}
    mine = [b for b in batches.values() if 2 * sum(x in ids for x in b.get("request_ids", [])) > len(b.get("request_ids", []))]
    client = {r["request_id"]: r for r in meas if r.get("request_id")}
    audio = {rid: r["output_audio_s"] for rid, r in client.items() if r.get("status") == "ok" and r.get("output_audio_s")}
    sizes = [b["size"] for b in mine if b.get("size")]
    cell = {"client": client_metrics(recs), "issues": issues, "batches": len(mine),
            "batch_size": {"mean": float(np.mean(sizes)) if sizes else None, "p50": float(np.median(sizes)) if sizes else None,
                           "at_cap": (sum(s == setting[1] for s in sizes) / len(sizes)) if sizes and setting else None},
            "gpu_ms_per_audio_s": gpu_ms_per_audio_s(mine), "padding": padding_ratio(mine), "graph_share": graph_share(mine),
            "control": control_check(mine, setting[1], setting[0]) if setting else None,
            "correspondence": correspondence(mine, client),
            "wait": wait_by_length({k: v for k, v in reqs.items() if k in ids}, audio),
            "queue_ms": bd.stats([reqs[i]["queue_ms"] for i in ids if i in reqs and reqs[i].get("queue_ms") is not None])}
    cell["audio_s_per_s"] = cell["client"].get("audio_s_per_s")
    cell["wait_long_p95"] = (cell["wait"].get("long") or {}).get("p95")
    cell["wait_long_max"] = (cell["wait"].get("long") or {}).get("max")
    return cell


def analyze_run(exp: Path) -> dict[str, Any]:
    res = json.loads((exp / "result.json").read_text())
    m = RUN_RE.search(res["experiment_id"])
    settings = phase_settings(res)
    run = {"run": exp.name, "graph": m.group(1) == "graph", "block": int(m.group(2)), "restarts": res.get("server_restarts") or [], "cells": {}}
    for ph in res.get("phases", []):
        pm = PHASE_RE.match(ph["phase"])
        if pm:
            run["cells"][(pm.group(1), pm.group(2))] = analyze_phase(exp, ph["phase"], settings.get(ph["phase"]))
    return run


def analyze_check(exp: Path) -> dict[str, Any]:
    from exp06_report import read_events

    res = json.loads((exp / "result.json").read_text())
    phases = {}
    for ph in res.get("phases", []):
        if not ph["phase"].startswith("chk_"):
            continue
        waves = {}
        for e in read_events(exp / "raw" / "server_events" / ph["phase"]):
            md = e.get("metadata") or {}
            if e.get("event_name") != "exp_vocode" or "request_codec_sha256" not in md:
                continue
            for rid, sha, st in zip(md["request_ids"], md["request_codec_sha256"], md["request_wave_stats"]):
                waves[rid] = {"sha": sha, "stats": st}
        phases[ph["phase"]] = waves
    ref = phases.get("chk_fifo1", {})
    out = {name: compare_waves(w, ref) for name, w in phases.items() if name != "chk_fifo1"}
    if "chk_fifo16" in phases and "chk_fifo16b" in phases:
        out["chk_fifo16b_vs_chk_fifo16"] = compare_waves(phases["chk_fifo16b"], phases["chk_fifo16"])
    return {"run": exp.name, "requests_with_digest": {k: len(v) for k, v in phases.items()}, "vs_fifo1": out}


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({k for r in runs for k in r["cells"]})
    out: dict[str, Any] = {}
    for graph in (True, False):
        for w, s in keys:
            rows = sorted((r for r in runs if r["graph"] == graph and (w, s) in r["cells"]), key=lambda r: r["block"])
            cells = [r["cells"][(w, s)] for r in rows]
            out[f"{'graph' if graph else 'nograph'}|{w}|{s}"] = {
                k: interval([c.get(k) for c in cells]) for k in ("audio_s_per_s", "gpu_ms_per_audio_s", "padding", "graph_share",
                                                                    "wait_long_p95", "wait_long_max")} | {
                "batch_size_mean": interval([c["batch_size"]["mean"] for c in cells]),
                "throughput_rps": interval([c["client"].get("throughput_rps") for c in cells]),
                "e2e_p50_s": interval([(c["client"].get("e2e_s") or {}).get("p50") for c in cells]),
                "e2e_p95_s": interval([(c["client"].get("e2e_s") or {}).get("p95") for c in cells]),
                "control_ok": [c["control"]["ok"] if c.get("control") else None for c in cells],
                "audio_mismatch": [c["correspondence"]["mismatched"] for c in cells], "blocks": [r["block"] for r in rows]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--check-run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    runs = [analyze_run(root / n) for n in a.run]
    summary: dict[str, Any] = {"generated_by": "scripts/analysis/exp10_report.py",
                               "runs": [{**r, "cells": {f"{w}|{s}": c for (w, s), c in r["cells"].items()}} for r in runs]}
    if runs:
        summary["by_setting"] = summarize(runs)
        summary["verdicts"] = verdicts(runs)
    if a.check_run:
        summary["check"] = analyze_check(root / a.check_run)
    out = Path(a.out) / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str) + "\n")
    print(out / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
