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

from run_records import clean, restart_flags  # noqa: E402

SECONDS_PER_CODEC_TOKEN = 0.04  # 24 kHz, 960 samples per codec token
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
PHASE_RE = re.compile(r"^(w3b|mix)_(fifo\d+|length\d+)$")
RUN_RE = re.compile(r"exp10_(graph|nograph)_block(\d+)")
BLOCKS = (1, 2, 3)  # registered: three blocks; a block that is missing gives 数据不足, never a verdict on fewer
CHECK_COMPARISONS = ("chk_fifo16", "chk_length16", "chk_length8", "chk_fifo16b", "chk_fifo16b_vs_chk_fifo16")


# ---------------------------------------------------------------- per-batch measures

def padding_ratio(batches: list[dict[str, Any]]) -> float | None:
    padded = effective = 0.0
    for b in batches:
        toks = b.get("token_lengths") or []
        if not toks:
            continue
        prompts = b.get("prompt_token_lengths")
        if not prompts or len(prompts) != len(toks) or any(p is None for p in prompts):
            return None  # the registered ratio counts the reference audio; without its length it is unknown
        rows = [t + p for t, p in zip(toks, prompts)]
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
        # the vocode event lists the requests in the order of its codec token counts; the breakdown may have dropped some
        ids = b.get("vocode_request_ids") or b.get("request_ids", [])
        for rid, tokens in zip(ids, b.get("request_codec_tokens") or b.get("token_lengths") or []):
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
    """Every request of a against the first request of b with the same codec-token hash; registered measure
    sqrt(|ss_a - ss_b|) / sqrt(ss_b)."""
    first_b: dict[str, list[float]] = {}
    for x in b.values():
        first_b.setdefault(x["sha"], x["stats"])
    pairs, mismatch, diffs, rms = 0, 0, [], []
    for x in a.values():
        if x["sha"] not in first_b:
            continue
        sa, sb = x["stats"], first_b[x["sha"]]
        pairs += 1
        if sa[0] != sb[0]:
            mismatch += 1
            continue
        diffs.append(math.sqrt(abs(sa[2] - sb[2])) / math.sqrt(sb[2]) if sb[2] > 0 else 0.0)
        rms.append(abs(math.sqrt(sa[2]) - math.sqrt(sb[2])) / math.sqrt(sb[2]) if sb[2] > 0 else 0.0)
    return {"pairs": pairs, "sample_mismatch": mismatch, "rel_diff_max": max(diffs) if diffs else None,
            "rel_diff_p95": float(np.percentile(diffs, 95)) if diffs else None, "rms_rel_change_max": max(rms) if rms else None}


def length_consistency(phases: dict[str, dict[str, dict[str, Any]]]) -> dict[str, int]:
    """Registered: requests with the same codec-token hash must get waveforms of the same length, within a phase and
    across phases."""
    lengths: dict[str, set[float]] = {}
    n = 0
    for waves in phases.values():
        for x in waves.values():
            lengths.setdefault(x["sha"], set()).add(x["stats"][0])
            n += 1
    return {"hashes": len(lengths), "requests": n, "inconsistent_hashes": sum(len(v) > 1 for v in lengths.values())}


# ---------------------------------------------------------------- verdicts

def valid(cell: dict[str, Any] | None) -> bool:
    """Registered: a phase with errors + timeouts > 5 % is unusable; a phase whose batches did not follow the control
    file is invalid (a missing control record counts as not followed). A phase in which the server restarted is
    invalid too: the restarted server records no events (supplementary registration 3). All are reported, but left
    out of the verdicts and intervals."""
    return (bool(cell) and cell.get("usable") is not False and (cell.get("control") or {}).get("ok") is True
            and not cell.get("restart_here"))


def _per_block(runs: list[dict[str, Any]], graph: bool, w: str, setting: str, key: str) -> dict[int, float]:
    return {r["block"]: r["cells"][(w, setting)][key] for r in runs if r["graph"] == graph and valid(r["cells"].get((w, setting)))
            and r["cells"][(w, setting)].get(key) is not None}


def validation_candidates(runs: list[dict[str, Any]], correctness_ok: bool | None = True) -> dict[str, Any]:
    """Registered trigger for the full-service validation: a setting whose completed audio seconds per second, graph on,
    W3b or MIX, is at least 5 % above fifo16 in every block. Length-aware settings count only when the correctness
    rules hold. Of several, the one with the largest mean gain, and of equal gains the smaller batch cap
    (supplementary registration 1)."""
    out = []
    for w in ("w3b", "mix"):
        base = _per_block(runs, True, w, "fifo16", "audio_s_per_s")
        for s in sorted({s for r in runs for ww, s in r["cells"] if ww == w and s != "fifo16"}):
            if s.startswith("length") and correctness_ok is not True:
                continue
            x = _per_block(runs, True, w, s, "audio_s_per_s")
            gain = [x[b] / base[b] - 1 for b in BLOCKS if b in x and b in base and base[b]]
            if len(gain) == len(BLOCKS) and all(g >= 0.05 for g in gain):
                out.append({"workload": w, "setting": s, "gain": gain, "mean_gain": float(np.mean(gain))})
    cap = lambda c: int(re.sub(r"\D", "", c["setting"]))  # noqa: E731
    chosen = max(out, key=lambda c: (round(c["mean_gain"], 12), -cap(c))) if out else None
    return {"candidates": out, "chosen": chosen}


def check_unique(runs: list[dict[str, Any]]) -> None:
    keys = [(r["graph"], r["block"]) for r in runs]
    if len(keys) != len(set(keys)):
        raise ValueError(f"more than one run for the same graph setting and block: {sorted(keys)}")


def verdicts(runs: list[dict[str, Any]], correctness_ok: bool | None = True) -> dict[str, Any]:
    check_unique(runs)
    out: dict[str, Any] = {}
    blocks = list(BLOCKS)

    e1, e16 = _per_block(runs, False, "w3b", "fifo1", "gpu_ms_per_audio_s"), _per_block(runs, False, "w3b", "fifo16", "gpu_ms_per_audio_s")
    ratio = [e16[b] / e1[b] for b in blocks if b in e1 and b in e16]
    v = "数据不足" if len(ratio) < len(blocks) else "支持" if all(x >= 0.9 for x in ratio) else "反驳" if all(x <= 0.8 for x in ratio) else "不确定"
    g1, g16 = _per_block(runs, True, "w3b", "fifo1", "gpu_ms_per_audio_s"), _per_block(runs, True, "w3b", "fifo16", "gpu_ms_per_audio_s")
    out["H10a"] = {"verdict": v, "e16_over_e1_graph_off": ratio,
                   "e16_over_e1_graph_on": [g16[b] / g1[b] for b in blocks if b in g1 and b in g16]}  # reported, not judged

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
    also = {}
    for graph, w in ((False, "mix"), (True, "w3b"), (False, "w3b")):  # reported, not judged
        e_l, e_f = _per_block(runs, graph, w, "length16", "gpu_ms_per_audio_s"), _per_block(runs, graph, w, "fifo16", "gpu_ms_per_audio_s")
        p_l, p_f = _per_block(runs, graph, w, "length16", "padding"), _per_block(runs, graph, w, "fifo16", "padding")
        also[f"{'graph' if graph else 'nograph'}|{w}"] = [{"block": b, "e_length_over_fifo": e_l[b] / e_f[b] if b in e_l and b in e_f else None,
                                                            "pad_length": p_l.get(b), "pad_fifo": p_f.get(b)} for b in blocks]
    out["H10c"] = {"verdict": v, "length16_vs_fifo16_mix_graph_on": [{"e_length": a, "e_fifo": b, "pad_length": c, "pad_fifo": d} for a, b, c, d in rows],
                   "also_reported": also}

    ratios = []
    for b in blocks:
        cl, cf = (next((r["cells"].get(("mix", s)) for r in runs if r["graph"] and r["block"] == b), None) for s in ("length16", "fifo16"))
        if valid(cl) and valid(cf) and cl.get("wait_long_p95") is not None and cl.get("wait_long_max") is not None \
                and cf.get("wait_long_p95") and cf.get("wait_long_max"):
            ratios.append((cl["wait_long_p95"] / cf["wait_long_p95"], cl["wait_long_max"] / cf["wait_long_max"]))
    if len(ratios) < len(blocks):
        v = "数据不足"
    elif all(p >= 1.2 for p, _ in ratios) or all(m >= 1.2 for _, m in ratios):  # P95 in every block, or the maximum in every block
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
    # Registered: if any request's audio does not match its codec tokens, or waveforms differ in length, stop the
    # length-aware performance conclusions and report only the failure evidence.
    if correctness_ok is False:
        for h in ("H10c", "H10d"):
            out[h] = {"verdict": "停止（正确性规则未通过）"}
    elif correctness_ok is None:
        for h in ("H10c", "H10d"):
            out[h]["verdict"] = "数据不足（缺正确性检查）"
    return out


VALIDATION_RATES = (21.0, 23.5)  # the two highest W3b rates of 实验 9 (supplementary registration 1)
VAL_RE = re.compile(r"exp10_validation_(t2_[a-z0-9_]+?)_((?:fifo|length)\d+)_block(\d+)")


def p50_reduction(cand: float, base: float) -> float:
    """1 - cand / base for censored P50s: an infinite candidate never counts as lower, an infinite base with a finite
    candidate counts as fully lower, two infinite values as no change."""
    if math.isinf(cand):
        return 0.0 if math.isinf(base) else -math.inf
    if math.isinf(base):
        return 1.0
    return 1 - cand / base


def validation_verdict(runs: list[dict[str, Any]], blocks: tuple[int, ...] = (1, 2, 3)) -> dict[str, Any]:
    """Supplementary registration 1: at 21 and 23.5 requests/s, the candidate's P50 at least 5 % below fifo16 in every
    block and its completed requests/s at least 98 % of fifo16 -> improved; P50 never lower -> not improved."""
    rows = []
    for b in blocks:
        r = next((x for x in runs if x["block"] == b), None)
        if r is None:
            return {"verdict": "数据不足", "rows": rows, "missing_block": b}
        control = r.get("control") or {}
        if not all((control.get(k) or {}).get("ok") is True for k in ("fifo16", r["setting"])):
            return {"verdict": "无效（控制文件没有生效）", "rows": rows, "block": b, "control": control}
        if r.get("restarts"):
            return {"verdict": "数据不足（验证运行里服务重启）", "rows": rows, "block": b, "restarts": r["restarts"]}
        base, cand = r["phases"].get("fifo16") or {}, r["phases"].get(r["setting"]) or {}
        for rate in VALIDATION_RATES:
            x, y = cand.get(rate) or {}, base.get(rate) or {}
            if None in (x.get("e2e_p50"), y.get("e2e_p50"), x.get("throughput_rps")) or not y.get("throughput_rps"):
                return {"verdict": "数据不足", "rows": rows, "missing": f"block {b} rate {rate:g}"}
            rows.append({"block": b, "rate": rate, "p50_reduction": p50_reduction(x["e2e_p50"], y["e2e_p50"]),
                         "throughput_ratio": x["throughput_rps"] / y["throughput_rps"],
                         "steady": {"candidate": x.get("steady"), "fifo16": y.get("steady")}})
    if all(r["p50_reduction"] >= 0.05 and r["throughput_ratio"] >= 0.98 for r in rows):
        v = "完整服务上有改善"
    elif all(r["p50_reduction"] <= 0 for r in rows):
        v = "完整服务上没有改善"
    else:
        v = "不确定"
    return {"verdict": v, "rows": rows}


def analyze_validation_run(exp: Path) -> dict[str, Any]:
    from exp06_report import client_metrics
    from exp09_report import censored_percentiles, queue_growth
    from omni_research.analysis.experiment_analysis import read_jsonl

    res = json.loads((exp / "result.json").read_text())
    m = VAL_RE.search(res["experiment_id"])
    run: dict[str, Any] = {"run": exp.name, "placement": m.group(1), "setting": m.group(2), "block": int(m.group(3)),
                           "restarts": res.get("server_restarts") or [], "phases": {}, "control": {}}
    settings = phase_settings(res)
    for ph in res.get("phases", []):
        if not ph["phase"].startswith("val_"):
            continue
        name = ph["phase"].removeprefix("val_")
        recs, _ = clean(read_jsonl(exp / "raw" / f"requests_{ph['phase']}.jsonl"))
        levels = {}
        for rate in sorted({r.get("rate_rps") for r in recs if r.get("rate_rps") is not None}):
            meas = [r for r in recs if r.get("rate_rps") == rate and not r.get("warmup")]
            cen, cm, growth = censored_percentiles(meas), client_metrics(meas), queue_growth(meas, float(rate))
            levels[float(rate)] = {"e2e_p50": cen["p50"], "e2e_p95": cen["p95"], "e2e_p99": cen["p99"], "failed": cen["failed"],
                                   "throughput_rps": cm.get("throughput_rps"), "audio_s_per_s": cm.get("audio_s_per_s"),
                                   "steady": growth["steady"], "growth_slope_rps": growth["slope_rps"]}
        run["phases"][name] = levels
        setting = settings.get(ph["phase"])
        if setting and (exp / "raw" / "server_events" / ph["phase"]).exists():
            cell = analyze_phase(exp, ph["phase"], setting)
            run["control"][name] = cell["control"]
            run.setdefault("correspondence", {})[name] = cell["correspondence"]
    return run


def correctness(runs: list[dict[str, Any]], check: dict[str, Any] | None,
                validation_runs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Registered correctness rules, with evidence required: every comparison of the check run has paired requests,
    no waveform-length mismatch for one codec hash, every batch of the check run followed its setting, and no request
    in any phase with events (check, performance, validation) got audio of another length than its codec tokens."""
    audio = sum((c.get("correspondence") or {}).get("mismatched", 0) for r in runs for c in r["cells"].values())
    audio += sum(x.get("mismatched", 0) for v in (validation_runs or []) for x in (v.get("correspondence") or {}).values())
    if check is None:
        return {"ok": None, "audio_codec_mismatch": audio, "reasons": ["no correctness run"]}
    comps = check.get("vs_fifo1") or {}
    audio += (check.get("correspondence") or {}).get("mismatched", 0)
    samples = sum((x or {}).get("sample_mismatch", 0) for x in comps.values())
    reasons = [f"{k}: no paired requests" for k in CHECK_COMPARISONS if not (comps.get(k) or {}).get("pairs")]
    if samples:
        reasons.append(f"{samples} paired requests with different waveform lengths")
    if (check.get("lengths") or {}).get("inconsistent_hashes", 1):
        reasons.append(f"codec hashes with more than one waveform length: {(check.get('lengths') or {}).get('inconsistent_hashes')}")
    if check.get("control_ok") is not True:
        reasons.append("the control file was not in effect in every batch of the check run")
    if check.get("restarts"):
        reasons.append("a server restarted during the check run")
    if audio:
        reasons.append(f"{audio} requests whose audio length does not match their codec tokens")
    return {"ok": not reasons, "reasons": reasons, "audio_codec_mismatch": audio, "waveform_length_mismatch": samples,
            "check_run": check.get("run")}


CONTRAST_METRICS = ("gpu_ms_per_audio_s", "audio_s_per_s", "padding", "wait_long_p95")


def contrasts(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Registered statistics: comparisons paired within a block. Each setting against fifo16 in the same run (relative
    change; padding as a difference in points), and graph off against graph on for the same setting and block."""
    out: dict[str, Any] = {}
    settings = sorted({s for r in runs for _, s in r["cells"]})
    for graph in (True, False):
        for w in ("w3b", "mix"):
            for s in (x for x in settings if x != "fifo16"):
                for k in CONTRAST_METRICS:
                    x, y = _per_block(runs, graph, w, s, k), _per_block(runs, graph, w, "fifo16", k)
                    vals = [None if b not in x or b not in y else x[b] - y[b] if k == "padding" else (x[b] / y[b] - 1 if y[b] else None)
                            for b in BLOCKS]
                    out[f"{'graph' if graph else 'nograph'}|{w}|{s}_vs_fifo16|{k}"] = interval(vals)
    for w in ("w3b", "mix"):
        for s in settings:
            x, y = _per_block(runs, False, w, s, "gpu_ms_per_audio_s"), _per_block(runs, True, w, s, "gpu_ms_per_audio_s")
            out[f"nograph_vs_graph|{w}|{s}|gpu_ms_per_audio_s"] = interval([x[b] / y[b] - 1 if b in x and b in y and y[b] else None for b in BLOCKS])
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

    recs, dropped = clean(read_jsonl(exp / "raw" / f"requests_{phase}.jsonl"))
    meas = [r for r in recs if not r.get("warmup")]
    events = read_events(exp / "raw" / "server_events" / phase)
    reqs, batches, issues = bd.breakdown(events)
    attach_extra(events, batches)
    ids = {r["request_id"] for r in meas if r.get("request_id")}
    every = list(batches.values())  # registered: all batches of the phase, the warm-up wave included
    client = {r["request_id"]: r for r in recs if r.get("request_id")}
    audio = {rid: r["output_audio_s"] for rid, r in client.items() if rid in ids and r.get("status") == "ok" and r.get("output_audio_s")}
    sizes = [b["size"] for b in every if b.get("size")]
    tokens = [t for b in every for t in (b.get("token_lengths") or [])]
    cell = {"client": client_metrics(recs), "issues": issues, "batches": len(every), "crash_affected_dropped": dropped,
            "batch_size": {"mean": float(np.mean(sizes)) if sizes else None, "p50": float(np.median(sizes)) if sizes else None,
                           "at_cap": (sum(s == setting[1] for s in sizes) / len(sizes)) if sizes and setting else None},
            "codec_tokens": {"n": len(tokens), "p10": float(np.percentile(tokens, 10)) if tokens else None,
                             "p50": float(np.median(tokens)) if tokens else None, "p90": float(np.percentile(tokens, 90)) if tokens else None},
            "gpu_ms_per_audio_s": gpu_ms_per_audio_s(every), "padding": padding_ratio(every), "graph_share": graph_share(every),
            "control": control_check(every, setting[1], setting[0]) if setting else None,
            "correspondence": correspondence(every, client),
            "wait": wait_by_length({k: v for k, v in reqs.items() if k in ids}, audio),
            "queue_ms": bd.stats([reqs[i]["queue_ms"] for i in ids if i in reqs and reqs[i].get("queue_ms") is not None])}
    bad = (cell["client"].get("errors") or 0) + (cell["client"].get("timeouts") or 0)
    cell["usable"] = not (cell["client"].get("measured") and bad / cell["client"]["measured"] > 0.05)
    cell["audio_s_per_s"] = cell["client"].get("audio_s_per_s")
    cell["wait_long_p95"] = (cell["wait"].get("long") or {}).get("p95")
    cell["wait_long_max"] = (cell["wait"].get("long") or {}).get("max")
    return cell


def analyze_run(exp: Path) -> dict[str, Any]:
    res = json.loads((exp / "result.json").read_text())
    m = RUN_RE.search(res["experiment_id"])
    settings = phase_settings(res)
    run = {"run": exp.name, "graph": m.group(1) == "graph", "block": int(m.group(2)), "restarts": res.get("server_restarts") or [], "cells": {}}
    flags = restart_flags([ph["phase"] for ph in res.get("phases", [])], run["restarts"])
    for ph in res.get("phases", []):
        pm = PHASE_RE.match(ph["phase"])
        if pm:
            cell = analyze_phase(exp, ph["phase"], settings.get(ph["phase"]))
            cell["restart_here"], cell["after_restart"] = flags[ph["phase"]]  # flagged; restart_here also makes it invalid
            run["cells"][(pm.group(1), pm.group(2))] = cell
    return run


def analyze_check(exp: Path) -> dict[str, Any]:
    from exp06_report import read_events

    res = json.loads((exp / "result.json").read_text())
    settings = phase_settings(res)
    phases, cells = {}, {}
    for ph in res.get("phases", []):
        if not ph["phase"].startswith("chk_"):
            continue
        cells[ph["phase"]] = analyze_phase(exp, ph["phase"], settings.get(ph["phase"]))
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
    return {"run": exp.name, "restarts": res.get("server_restarts") or [], "requests_with_digest": {k: len(v) for k, v in phases.items()},
            "vs_fifo1": out, "lengths": length_consistency(phases),
            "control": {k: c["control"] for k, c in cells.items()},
            "control_ok": bool(cells) and all((c["control"] or {}).get("ok") is True for c in cells.values()),
            "correspondence": {"checked": sum(c["correspondence"]["checked"] for c in cells.values()),
                               "mismatched": sum(c["correspondence"]["mismatched"] for c in cells.values())}}


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({k for r in runs for k in r["cells"]})
    out: dict[str, Any] = {}
    for graph in (True, False):
        for w, s in keys:
            rows = sorted((r for r in runs if r["graph"] == graph and (w, s) in r["cells"]), key=lambda r: r["block"])
            # invalid cells (unusable or control not applied) stay in runs[] and flags, but not in the intervals
            cells = [r["cells"][(w, s)] if valid(r["cells"][(w, s)]) else {"client": {}, "batch_size": {}} for r in rows]
            out[f"{'graph' if graph else 'nograph'}|{w}|{s}"] = {
                k: interval([c.get(k) for c in cells]) for k in ("audio_s_per_s", "gpu_ms_per_audio_s", "padding", "graph_share",
                                                                    "wait_long_p95", "wait_long_max")} | {
                "batch_size_mean": interval([c["batch_size"].get("mean") for c in cells]),
                "throughput_rps": interval([c["client"].get("throughput_rps") for c in cells]),
                "e2e_p50_s": interval([(c["client"].get("e2e_s") or {}).get("p50") for c in cells]),
                "e2e_p95_s": interval([(c["client"].get("e2e_s") or {}).get("p95") for c in cells]),
                "control_ok": [(r["cells"][(w, s)].get("control") or {}).get("ok") for r in rows],
                "usable": [r["cells"][(w, s)].get("usable") for r in rows],
                "audio_mismatch": [(r["cells"][(w, s)].get("correspondence") or {}).get("mismatched") for r in rows],
                "blocks": [r["block"] for r in rows]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--check-run")
    ap.add_argument("--validation-run", action="append", default=[], help="the conditional full-service validation runs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    runs = [analyze_run(root / n) for n in a.run]
    summary: dict[str, Any] = {"generated_by": "scripts/analysis/exp10_report.py",
                               "runs": [{**r, "cells": {f"{w}|{s}": c for (w, s), c in r["cells"].items()}} for r in runs]}
    check = analyze_check(root / a.check_run) if a.check_run else None
    if check:
        summary["check"] = check
    vruns = [analyze_validation_run(root / n) for n in a.validation_run]
    if runs:
        check_unique(runs)
        corr = correctness(runs, check, vruns)
        cands = validation_candidates(runs, corr["ok"])
        label = lambda r, w, st: f"{'graph' if r['graph'] else 'nograph'} block {r['block']} {w}|{st}"  # noqa: E731
        summary["by_setting"] = summarize(runs)
        summary["contrasts"] = contrasts(runs)
        summary["correctness"] = corr
        summary["verdicts"] = verdicts(runs, corr["ok"])
        summary["validation"] = {**cands, "decision": "在实验 9 最好的放置上做完整服务验证" if cands["chosen"] else "没有找到值得验证的设置"}
        summary["flags"] = {
            "unusable": [label(r, w, st) for r in runs for (w, st), c in r["cells"].items() if c.get("usable") is False],
            "control_invalid": [label(r, w, st) for r in runs for (w, st), c in r["cells"].items() if (c.get("control") or {}).get("ok") is not True],
            "restart_here": [label(r, w, st) for r in runs for (w, st), c in r["cells"].items() if c.get("restart_here")],
            "after_restart": [label(r, w, st) for r in runs for (w, st), c in r["cells"].items() if c.get("after_restart")],
            # G0 must not use the flow CUDA graph at all
            "graph_not_off": [label(r, w, st) for r in runs if not r["graph"] for (w, st), c in r["cells"].items() if c.get("graph_share")],
        }
    if vruns:
        summary["validation_runs"] = vruns
        summary.setdefault("validation", {})["verdict"] = validation_verdict(vruns)
    out = Path(a.out) / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str) + "\n")
    print(out / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
