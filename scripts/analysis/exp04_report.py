#!/usr/bin/env python3
"""实验 4：每批处理多少任务最好。只分析 plan v2 已有的运行，没有新的 GPU 运行。

    python scripts/analysis/exp04_report.py --data-root <results root> \
        --curve s2x6:20261116_v2b_sb_s2x6_run01:w3b_small --curve s2x6:20261116_v2b_a_s2x6_c03_run01:w3b_closed ... \
        --out research/01_性能瓶颈/实验04_每批处理多少任务最好

The same closed-loop runs as 实验 3 (W3b, concurrency 1-256, server events). Per level, stage_by_level.breakdown gives
the effective batch of each stage (time-averaged requests inside one replica's model path, Little's law) and the
per-unit time: thinker ms per output token (model path minus prefill, over tokens - 1), talker and code2wav ms per
second of output audio. A replica's output rate is effective batch x 1000 / per-unit ms (Derived; it ignores the
prefill inside the thinker's model path). code2wav batches come from the server log lines
"minicpm_code2wav_batch size=N" (sglang-omni 520b49ee caps a batch at 16 and does not wait to fill it,
models/minicpm_o/config.py:144-145); where a log holds one code2wav replica (4 pairs, 8 full replicas), the gap
between back-to-back batch starts divided by the earlier batch's size estimates the time per request. Configured
running limits: thinker 64 and talker 32 requests per replica (models/minicpm_o/stages.py:361 and :144).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from omni_research.analysis.metrics import percentile  # noqa: E402

C2W_CAP = 16
THINKER_SLOTS = 64
TALKER_SLOTS = 32
MAX_GAP_S = 10.0  # back-to-back batches: the next batch started within this many seconds
BACKLOG_MEAN = 2.0  # mean batch size from which code2wav counts as backlogged
LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) .*minicpm_code2wav_batch size=(\d+) max_codec_tokens=(\d+)")


def pct(values: list[float | None], q: float) -> float | None:
    v = [x for x in values if x is not None]
    return percentile(v, q) if v else None


def parse_c2w_log(lines: list[str] | Any, utc_offset_s: int) -> list[dict[str, Any]]:
    """code2wav batch records of one server log; t is unix seconds (the log prints the node's local time)."""
    out = []
    for line in lines:
        m = LINE.match(line)
        if m:
            local = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() + int(m.group(2)) / 1e3
            out.append({"t": local - utc_offset_s, "size": int(m.group(3)), "max_codec_tokens": int(m.group(4))})
    return out


def batch_stats(batches: list[dict[str, Any]], start: float, end: float, cap: int = C2W_CAP) -> dict[str, Any]:
    sizes = [b["size"] for b in batches if start <= b["t"] <= end]
    if not sizes:
        return {"batches": 0, "requests": 0, "mean_size": None, "max_size": None, "share_at_cap": None}
    return {"batches": len(sizes), "requests": sum(sizes), "mean_size": sum(sizes) / len(sizes), "max_size": max(sizes),
            "share_at_cap": sum(s >= cap for s in sizes) / len(sizes)}


def per_request_from_gaps(batches: list[dict[str, Any]], max_gap_s: float = MAX_GAP_S) -> dict[str, Any]:
    """One replica's batches: gap to the next batch start / this batch's size, for back-to-back batches."""
    b = sorted(batches, key=lambda x: x["t"])
    vals = [(b[i + 1]["t"] - b[i]["t"]) / b[i]["size"] * 1e3 for i in range(len(b) - 1) if b[i + 1]["t"] - b[i]["t"] <= max_gap_s]
    return {"pairs": len(vals), "median_ms": pct(vals, 0.5), "p90_ms": pct(vals, 0.9)}


def backlogged(stats: dict[str, Any], min_mean: float = BACKLOG_MEAN) -> bool:
    """code2wav had work waiting most of the time: its batches averaged at least min_mean requests. Only then do gaps
    between batch starts measure service time rather than waiting for the next request."""
    return stats.get("mean_size") is not None and stats["mean_size"] >= min_mean


def rate_per_replica(eff_batch: float | None, ms_per_unit: float | None) -> float | None:
    """Units (tokens, or seconds of audio) a replica turns out per second: requests in flight x their own rate."""
    return eff_batch * 1000.0 / ms_per_unit if eff_batch is not None and ms_per_unit else None


# ------------------------------------------------------------------------------------------
# runs


def label(layout: str) -> str:
    if layout.startswith("s"):
        t, k = layout[1:].split("x")
        return f"{t}+{k}"
    return "4 组 1+1" if layout == "p4" else "8 个完整副本"


def level_windows(exp: Path, phase: str) -> dict[float, tuple[float, float]]:
    """Unix-time window of each level (first send to last completion, warmup included)."""
    from omni_research.analysis.experiment_analysis import clock_anchor, read_jsonl

    anchor = clock_anchor(exp)
    if anchor is None:
        return {}
    mono0, wall0 = anchor
    wall = lambda ns: (ns - mono0 + wall0) / 1e9  # noqa: E731
    by: dict[float, list[dict[str, Any]]] = {}
    for r in read_jsonl(exp / "raw" / f"requests_{phase}.jsonl"):
        if r.get("concurrency") is not None and r.get("sent_ns"):
            by.setdefault(float(r["concurrency"]), []).append(r)
    return {c: (wall(min(r["sent_ns"] for r in rs)), wall(max((r.get("done_ns") or r["sent_ns"]) for r in rs))) for c, rs in by.items()}


def scan(layout: str, exp: Path, phase: str) -> list[dict[str, Any]]:
    from exp01_gpu_timeline import utc_offset_s
    from stage_by_level import breakdown

    off = utc_offset_s(exp)
    logs = {p.name: parse_c2w_log(open(p, errors="replace"), off) for p in sorted((exp / "logs").glob("server_*.log"))}
    one_per_log = layout.startswith(("p", "r"))
    windows = level_windows(exp, phase)
    rows = []
    for row in breakdown(exp, phase):
        c = float(row["level"])
        st = row["stage_ms"]
        unit = {"thinker_ms_per_token": pct(st.get(("thinker", "ms_per_token"), []), 0.5),
                "talker_ms_per_audio_s": pct(st.get(("talker", "ms_per_audio_s"), []), 0.5),
                "code2wav_ms_per_audio_s": pct(st.get(("code2wav", "ms_per_audio_s"), []), 0.5)}
        eff = row.get("eff_batch") or {}
        out = {"concurrency": c, "run": exp.name, "eff_batch": eff, "replicas": row.get("replicas"), "unit": unit,
               "gpu_util": row.get("gpu_util"),
               "thinker_tokens_per_s_per_replica": rate_per_replica(eff.get("thinker"), unit["thinker_ms_per_token"]),
               "talker_audio_s_per_s_per_replica": rate_per_replica(eff.get("talker"), unit["talker_ms_per_audio_s"]),
               "output_tokens_p50": pct(st.get(("output", "tokens"), []), 0.5), "output_audio_s_p50": pct(st.get(("output", "audio_s"), []), 0.5),
               "code2wav_service_ms_p50": pct(st.get(("code2wav", "service"), []), 0.5)}
        if c in windows:
            a, b = windows[c]
            out["code2wav_batches"] = batch_stats([x for v in logs.values() for x in v], a, b)
            if one_per_log and backlogged(out["code2wav_batches"]):
                est = [per_request_from_gaps([x for x in v if a <= x["t"] <= b]) for v in logs.values()]
                vals = [e["median_ms"] for e in est if e["median_ms"] is not None]
                out["code2wav_ms_per_request_from_gaps"] = {"replicas": len(vals), "pairs": sum(e["pairs"] for e in est),
                                                           "median_of_replica_medians_ms": pct(vals, 0.5)}
        rows.append(out)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--curve", action="append", required=True, help="LAYOUT:RUN_DIR:PHASE")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    curves: dict[str, Any] = {}
    for spec in a.curve:
        layout, run, phase = spec.split(":")
        c = curves.setdefault(layout, {"label": label(layout), "levels": []})
        c["levels"] = sorted(c["levels"] + scan(layout, root / run, phase), key=lambda r: r["concurrency"])
    summary = {"generated_by": "scripts/analysis/exp04_report.py",
               "config_limits": {"thinker_max_running_per_replica": THINKER_SLOTS, "talker_max_running_per_replica": TALKER_SLOTS,
                                 "code2wav_max_batch_size": C2W_CAP, "code2wav_max_batch_wait_ms": 0},
               "rule": {"back_to_back_max_gap_s": MAX_GAP_S, "backlogged_mean_batch": BACKLOG_MEAN}, "curves": curves}
    out = Path(a.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "results" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    for k, c in curves.items():
        for r in c["levels"]:
            e, u, cb = r["eff_batch"], r["unit"], r.get("code2wav_batches") or {}
            g = r.get("code2wav_ms_per_request_from_gaps") or {}
            print(f"{c['label']:8s} c{int(r['concurrency']):>3} effT={e.get('thinker') or 0:5.1f} msTok={u['thinker_ms_per_token'] or 0:5.1f} tok/s={r['thinker_tokens_per_s_per_replica'] or 0:6.0f} "
                  f"effK={e.get('talker') or 0:4.1f} msAud={u['talker_ms_per_audio_s'] or 0:5.0f} aud/s={r['talker_audio_s_per_s_per_replica'] or 0:5.1f} "
                  f"effC={e.get('code2wav') or 0:4.1f} c2wAud={u['code2wav_ms_per_audio_s'] or 0:4.0f} batch={cb.get('mean_size') or 0:4.1f} max={cb.get('max_size')} cap%={100 * (cb.get('share_at_cap') or 0):3.0f} gap/req={g.get('median_of_replica_medians_ms') or 0:4.0f} gpu={ {k2: round(v) for k2, v in (r['gpu_util'] or {}).items()} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
