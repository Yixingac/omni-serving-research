#!/usr/bin/env python3
"""实验 12：回放能不能代替真实负载。只分析 plan v2 已有的回放运行（D 组），没有新的 GPU 运行。

    python scripts/analysis/exp12_report.py --data-root <results root> \
        --live 20261116_v2b_b2l_r8_run01:lo_text_c256 --replay 20261116_v2b_d_dep_orig_1x_run01 ... \
        --out research/01_性能瓶颈/实验12_回放能不能代替真实负载

The live run is the text-agent level whose trace every replay re-sends, on 8 full replicas (R8). Each replay ran on a
freshly started R8 (empty caches). Only model requests count; tool calls and waits are reproduced by the replay
engine (src/omni_research/replay/engine.py) and are not requests to the server. A run whose client ran out of file
descriptors is kept in the summary but marked invalid and left out of every comparison. Percentiles are
linear-interpolated. The verdict rules for D-H1 are in the experiment's hypotheses.md; they were set after the data
had been seen, so the verdicts are exploratory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from omni_research.analysis.metrics import percentile  # noqa: E402

TOL = 0.10  # D-H1's own tolerance ("相差不到 10%")
PAYLOAD_ZH = {"auto": "原文", "prefix_surrogate": "保留前缀的替代", "surrogate": "完全替代"}
MODE_ZH = {"dependency": "依赖感知", "request": "请求级"}


def pct(values: list[float | None], q: float) -> float | None:
    v = [x for x in values if x is not None]
    return percentile(v, q) if v else None


def rel(a: float | None, b: float | None) -> float | None:
    return None if a is None or not b else a / b - 1


def error_class(e: Any) -> str:
    s = str(e or "")
    if "Too many open files" in s:
        return "client fd limit"
    if "out of memory" in s:
        return "CUDA out of memory"
    return s[:80] or "unknown"


def is_model_request(r: dict[str, Any]) -> bool:
    return r.get("kind") in (None, "model_request")


def replay_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Client-side numbers of a live agent level or a replay, model requests only."""
    mr = [r for r in records if is_model_request(r)]
    ok = [r for r in mr if r.get("status") == "ok" and r.get("done_ns") and r.get("sent_ns") is not None]
    errors: dict[str, int] = {}
    for r in mr:
        if r.get("status") not in ("ok", "skipped"):
            k = error_class(r.get("error"))
            errors[k] = errors.get(k, 0) + 1
    out: dict[str, Any] = {"model_requests": len(mr), "ok": len(ok), "skipped": sum(r.get("status") == "skipped" for r in mr),
                           "errors": errors, "valid": errors.get("client fd limit", 0) == 0}
    if not ok:
        return out
    t0 = min(r["sent_ns"] for r in mr if r.get("sent_ns") is not None)
    span = (max(r["done_ns"] for r in ok) - t0) / 1e9
    e2e = [(r["done_ns"] - r["sent_ns"]) / 1e9 for r in ok]
    ttft = [(r["first_text_ns"] - r["sent_ns"]) / 1e9 for r in ok if r.get("first_text_ns")]
    sess: dict[str, list[int]] = {}
    for r in ok:
        s = sess.setdefault(r.get("session_id") or r["request_id"], [r["sent_ns"], r["done_ns"]])
        s[0], s[1] = min(s[0], r["sent_ns"]), max(s[1], r["done_ns"])
    edges = sorted([(r["sent_ns"], 1) for r in ok] + [(r["done_ns"], -1) for r in ok], key=lambda x: (x[0], x[1]))
    cur = peak = 0
    for _, step in edges:
        cur += step
        peak = max(peak, cur)
    finish: dict[str, int] = {}
    for r in ok:
        finish[r.get("finish_reason") or "?"] = finish.get(r.get("finish_reason") or "?", 0) + 1
    sess_s = [(b - a) / 1e9 for a, b in sess.values()]
    out.update({
        "span_s": span, "requests_per_s": len(ok) / span, "mean_in_flight": sum(e2e) / span, "peak_in_flight": peak,
        "e2e_p50_s": pct(e2e, 0.5), "e2e_p90_s": pct(e2e, 0.9), "ttft_p50_s": pct(ttft, 0.5), "ttft_p90_s": pct(ttft, 0.9),
        "sessions": len(sess), "session_p50_s": pct(sess_s, 0.5), "session_p90_s": pct(sess_s, 0.9),
        "output_tokens_mean": (sum(r["completion_tokens"] for r in ok if r.get("completion_tokens") is not None)
                               / max(1, sum(r.get("completion_tokens") is not None for r in ok))),
        "prompt_tokens_p50": pct([r.get("prompt_tokens") for r in ok], 0.5), "finish": finish})
    return out


def in_flight_series(records: list[dict[str, Any]], bin_s: float = 1.0) -> list[float]:
    """Mean number of successful model requests in flight per time bin, from the first send."""
    ok = [r for r in records if is_model_request(r) and r.get("status") == "ok" and r.get("done_ns") and r.get("sent_ns") is not None]
    if not ok:
        return []
    t0 = min(r["sent_ns"] for r in ok) / 1e9
    end = max(r["done_ns"] for r in ok) / 1e9
    n = max(1, math.ceil((end - t0) / bin_s - 1e-9))
    acc = [0.0] * n
    for r in ok:
        a, b = r["sent_ns"] / 1e9 - t0, r["done_ns"] / 1e9 - t0
        i = int(a // bin_s)
        while i < n and i * bin_s < b:
            lo, hi = max(a, i * bin_s), min(b, (i + 1) * bin_s)
            acc[i] += max(0.0, hi - lo)
            i += 1
    return [x / bin_s for x in acc]


def copy_verdict(e2e_rel: float | None, tol: float = TOL) -> dict[str, Any]:
    """Copied sessions overestimate capacity: the copied replay looks >= tol faster than the unique-session one."""
    v = "不确定" if e2e_rel is None else ("支持" if e2e_rel <= -tol else ("不支持" if e2e_rel >= tol else "不确定"))
    return {"verdict": v, "e2e_p50_rel": e2e_rel}


def within_verdict(rels: dict[str, float | None], tol: float = TOL) -> dict[str, Any]:
    """Within tol of the live run on every listed metric."""
    known = {k: v for k, v in rels.items() if v is not None}
    v = "不确定" if not known else ("支持" if all(abs(x) < tol for x in known.values()) else "不支持")
    return {"verdict": v, "rel": rels}


def underestimate_verdict(e2e_rel: float | None, tol: float = TOL) -> dict[str, Any]:
    """A replay underestimates capacity when it looks >= tol slower than the live run."""
    v = "不确定" if e2e_rel is None else ("支持" if e2e_rel >= tol else ("不支持" if e2e_rel <= -tol else "不确定"))
    return {"verdict": v, "e2e_p50_rel": e2e_rel}


# ------------------------------------------------------------------------------------------
# runs


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(p)] if p.exists() else []


def _gpu_mean(ana: dict, phase: str) -> float | None:
    v = [g["sampled_util_mean"] for g in ana.get("gpu_by_phase", []) if g["phase"] == phase and g.get("sampled_util_mean") is not None]
    return sum(v) / len(v) if v else None


def scan_live(exp: Path, phase: str) -> dict[str, Any]:
    res = json.loads((exp / "result.json").read_text())
    ana = json.loads((exp / "analysis" / "analysis.json").read_text())
    recs = _read_jsonl(exp / "raw" / f"requests_{phase}.jsonl")
    return {"run": exp.name, "experiment_id": res["experiment_id"], "phase": phase, "label": "实时运行", "mode": "live",
            "payload": "auto", "speed": 1.0, "copies": 1, **replay_summary(recs),
            "prefix_hit": ((ana.get("prefix_cache") or {}).get(phase) or {}).get("cached_fraction"), "gpu_util_mean": _gpu_mean(ana, phase),
            "series": in_flight_series(recs)}


def scan_replay(exp: Path) -> dict[str, Any]:
    res = json.loads((exp / "result.json").read_text())
    ana = json.loads((exp / "analysis" / "analysis.json").read_text())
    cfg = json.loads((exp / "config.json").read_text())
    ph = next(p for p in res["phases"] if p.get("kind") == "replay")
    cfg_ph = next(p for p in cfg.get("phases", []) if p.get("kind") == "replay")
    rp = ph.get("replay") or {}
    parts = Path(ph.get("source_trace") or "").parts
    source = None
    if "latest" in parts:  # latest/<experiment_id>/raw/agent_trace_<phase>.jsonl; keep the ids, never the path
        source = f"{parts[parts.index('latest') + 1]} {Path(parts[-1]).stem.removeprefix('agent_trace_')}"
    recs = _read_jsonl(exp / "raw" / "requests_replay.jsonl")
    payload = cfg_ph.get("payload", "auto")
    copies = rp.get("session_copies") or 1
    eid = res["experiment_id"]
    return {"run": exp.name, "experiment_id": eid, "label": eid.removeprefix("v2b_d_"), "mode": rp.get("mode"), "payload": payload,
            "speed": float(rp.get("speed", 1)), "copies": copies, "max_sessions": cfg_ph.get("max_sessions"), "source": source,
            "engine_status": rp.get("status_counts"), **replay_summary(recs),
            "prefix_hit": ((ana.get("prefix_cache") or {}).get("replay") or {}).get("cached_fraction"),
            "gpu_util_mean": _gpu_mean(ana, "replay"), "series": in_flight_series(recs)}


METRICS = ("requests_per_s", "e2e_p50_s", "e2e_p90_s", "ttft_p50_s", "ttft_p90_s", "session_p50_s", "session_p90_s",
           "output_tokens_mean", "prompt_tokens_p50", "prefix_hit", "mean_in_flight")


def compare(a: dict[str, Any], b: dict[str, Any] | None) -> dict[str, float | None] | None:
    return None if b is None else {m: rel(a.get(m), b.get(m)) for m in METRICS}


def build(live: dict[str, Any], replays: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in replays if r["valid"]]

    def orig_at(speed: float) -> dict[str, Any] | None:
        return next((r for r in valid if r["mode"] == "dependency" and r["payload"] == "auto" and r["copies"] == 1 and r["speed"] == speed), None)

    for r in replays:
        r["vs_live"] = compare(r, live) if r["valid"] else None
        ref = orig_at(r["speed"])
        r["vs_dep_orig_same_speed"] = compare(r, ref) if r["valid"] and ref is not None and ref is not r else None
    find = lambda **kw: next((r for r in valid if all(r.get(k) == v for k, v in kw.items())), None)  # noqa: E731
    copy4 = find(mode="dependency", payload="auto", speed=1.0, copies=4)
    prefix = find(mode="dependency", payload="prefix_surrogate", speed=1.0, copies=1)
    sur = find(mode="dependency", payload="surrogate", speed=1.0, copies=1)
    verdicts = {
        "part1_copies_overestimate": {**copy_verdict((copy4 or {}).get("vs_dep_orig_same_speed", {}).get("e2e_p50_s") if copy4 else None),
                                      "compared": "4 份复制对 2048 个不重复会话（都是依赖感知、原文、1 倍速）",
                                      "prefix_hit": [copy4 and copy4["prefix_hit"], orig_at(1.0) and orig_at(1.0)["prefix_hit"]]},
        "part2_prefix_within_10pct": {**within_verdict({k: prefix["vs_live"][k] for k in ("e2e_p50_s", "e2e_p90_s", "session_p50_s")} if prefix else {}),
                                      "compared": "保留前缀的替代（依赖感知、1 倍速）对实时运行"},
        "part3_surrogate_underestimates": {**underestimate_verdict(sur["vs_live"]["e2e_p50_s"] if sur else None),
                                           "compared": "完全替代（依赖感知、1 倍速）对实时运行"},
    }
    return {"live": live, "replays": replays, "verdicts": verdicts}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--live", required=True, help="RUN_DIR:PHASE of the live agent level that was recorded")
    ap.add_argument("--replay", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    run, phase = a.live.split(":")
    live = scan_live(root / run, phase)
    replays = [scan_replay(root / r) for r in a.replay]
    summary = {"generated_by": "scripts/analysis/exp12_report.py", "rule": {"tolerance": TOL}, **build(live, replays)}
    out = Path(a.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "results" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    rows = [summary["live"]] + summary["replays"]
    keys = ["label", "mode", "payload", "speed", "copies", "valid", "model_requests", "ok", "skipped", *METRICS, "peak_in_flight", "gpu_util_mean"]
    lines = [",".join(keys)] + [",".join("" if r.get(k) is None else (f"{r[k]:.4f}" if isinstance(r.get(k), float) else str(r[k])) for k in keys) for r in rows]
    (out / "results" / "runs.csv").write_text("\n".join(lines) + "\n")
    for k, v in summary["verdicts"].items():
        print(k, v["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
