#!/usr/bin/env python3
"""实验 19：负载类型会不会改变最佳分工。只分析 plan v2 两批已有的运行，没有新的 GPU 运行。

    python scripts/analysis/exp19_report.py --data-root <results root> \
        --mix 20261115_v2_b1_r8_run01 ... --agent 20261116_v2b_b2l_r8_run01 ... \
        --agent-short 20261115_v2_b2_r8_run01 ... --w3b 20261115_v2_a_r8_run1_run01 ... \
        --out research/01_性能瓶颈/实验19_负载类型和最佳分工

Mixed workload (B1): 40% W1 text chat, 40% W3b text question with a spoken answer, 20% W2 spoken question with a
text answer, drawn per request; closed loop at concurrency 16-256, then open loop at 0.5-1.2 x the run's own
closed-loop maximum. A request meets the plan v2 SLO when it succeeded, its TTFT is <= 1 s and, for a spoken answer,
e2e / output-audio seconds <= 0.5. Agent workload (B2 short levels in batch 1, B2l in batch 2): sessions of up to
5 model requests with in-process tools; text agents, and voice agents whose last step answers in speech.
Throughput of a closed-loop level is analyze_experiment.py's tp_requests_per_s. Percentiles are linear-interpolated
(omni_research.analysis.metrics.percentile); agent request and session percentiles are analyze_experiment.py's.
The verdict rules are in the experiment's hypotheses.md. They were fixed after the batch-1 results had been seen,
so every verdict here is exploratory.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from omni_research.analysis.metrics import percentile, request_metrics  # noqa: E402

TOL = 0.03  # most repeats of the same layout in plan v2 differ by < 3 % (W3b closed maximum, both nodes)
TOL_WIDE = 0.06  # the largest repeat difference (4+4: 14.40 and 15.23 req/s, 5.6 %); verdicts are recomputed with it
W3B_REFERENCE_THINKER_GPUS = 2  # 实验 2: the best split for W3b alone is 2+6 (1+7 ties on SLO capacity, the smaller count).
# Fixed, not re-picked with the tolerance: 8 full replicas share each GPU between thinker and talker, so they are not a
# point on the thinker:talker axis, and counting them as 8 thinker GPUs made "more thinker" impossible at 6 %.
HIT_PTS = 3.0
TTFT_REL = 0.05
TTFT_S = 1.0
SPEECH_E2E_PER_AUDIO_S = 0.5
TYPES = ("W1", "W2", "W3b")
NODE = {"20261115": "第一批节点", "20261116": "compute-03", "20261117": "compute-03"}


def thinker_gpus(layout: str) -> int:
    """GPUs that run a thinker: s<t>x<k> -> t; p<n> -> n (one thinker GPU per pair); r<n> -> n (every GPU)."""
    if layout.startswith("s"):
        return int(layout[1:].split("x")[0])
    return int(layout[1:])


def layout_label(layout: str) -> str:
    if layout.startswith("s"):
        t, k = layout[1:].split("x")
        return f"{t}+{k}"
    if layout.startswith("p"):
        return f"{layout[1:]} 组 1+1"
    return f"{layout[1:]} 个完整副本"


def layout_roles(layout: str) -> dict[str, list[int]]:
    """GPU indices per role (plan v2 configs: thinkers first in a pooled split; thinker on the even GPU of a pair)."""
    if layout.startswith("s"):
        t, k = (int(x) for x in layout[1:].split("x"))
        return {"thinker": list(range(t)), "talker": list(range(t, t + k))}
    if layout.startswith("p"):
        n = int(layout[1:])
        return {"thinker": [2 * i for i in range(n)], "talker": [2 * i + 1 for i in range(n)]}
    return {"full": list(range(int(layout[1:])))}


def eid_layout(eid: str) -> str:
    """v2_b1_p4 -> p4; v2b_b2l_s2x6 -> s2x6; v2_a_s2x6_run1 -> s2x6."""
    return next(p for p in eid.split("_")[2:] if p[:1] in "rps" and any(c.isdigit() for c in p))


def pct(values: list[float | None], q: float) -> float | None:
    v = [x for x in values if x is not None]
    return percentile(v, q) if v else None


def near_best(values: dict[str, float], tol: float = TOL, higher_is_better: bool = True) -> list[str]:
    """Names whose value is within ``tol`` (relative) of the best, best first."""
    vals = {k: v for k, v in values.items() if v is not None}
    if not vals:
        return []
    if higher_is_better:
        best = max(vals.values())
        keep = [k for k, v in vals.items() if v >= best * (1 - tol) - 1e-12]
        return sorted(keep, key=lambda k: (-vals[k], k))
    best = min(vals.values())
    keep = [k for k, v in vals.items() if v <= best * (1 + tol) + 1e-12]
    return sorted(keep, key=lambda k: (vals[k], k))


def more_thinker_verdict(best_set: list[str], reference: int) -> dict[str, Any]:
    """支持 when every near-best layout has more thinker GPUs than the reference, 不支持 when none has."""
    gpus = {b: thinker_gpus(b) for b in best_set}
    more = [b for b, g in gpus.items() if g > reference]
    verdict = "支持" if best_set and len(more) == len(best_set) else ("不支持" if not more else "不确定")
    return {"verdict": verdict, "near_best": best_set, "thinker_gpus": gpus, "reference_thinker_gpus": reference}


def slo_ok(rec: dict[str, Any]) -> bool:
    if rec.get("status") != "ok":
        return False
    m = request_metrics(rec)
    if m.get("ttft_s") is None or m["ttft_s"] > TTFT_S:
        return False
    if "audio" in (rec.get("output_modalities") or []):
        return m.get("e2e_per_audio_s") is not None and m["e2e_per_audio_s"] <= SPEECH_E2E_PER_AUDIO_S
    return True


def type_of(rec: dict[str, Any]) -> str:
    return (rec.get("workload_id") or "?").split("_")[0]


def _type_stats(recs: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in recs if r.get("status") == "ok"]
    ms = [request_metrics(r) for r in ok]
    return {"n": len(recs), "ok": len(ok), "errors": len(recs) - len(ok), "slo": sum(map(slo_ok, recs)) / len(recs) if recs else None,
            "ttft_p50_s": pct([m.get("ttft_s") for m in ms], 0.5), "ttft_p90_s": pct([m.get("ttft_s") for m in ms], 0.9),
            "e2e_p50_s": pct([m.get("e2e_s") for m in ms], 0.5), "e2e_p90_s": pct([m.get("e2e_s") for m in ms], 0.9),
            "e2e_per_audio_p50": pct([m.get("e2e_per_audio_s") for m in ms], 0.5),
            "e2e_per_audio_p90": pct([m.get("e2e_per_audio_s") for m in ms], 0.9)}


def mix_level_stats(records: list[dict[str, Any]], key: str = "concurrency") -> dict[Any, dict[str, Any]]:
    """Per load level (non-warmup requests): SLO share of all requests and per request type."""
    by: dict[Any, list[dict[str, Any]]] = {}
    for r in records:
        if not r.get("warmup") and r.get(key) is not None:
            by.setdefault(r[key], []).append(r)
    out = {}
    for lv, recs in sorted(by.items()):
        types = {t: _type_stats([r for r in recs if type_of(r) == t]) for t in TYPES if any(type_of(r) == t for r in recs)}
        out[lv] = {"n": len(recs), "ok": sum(r.get("status") == "ok" for r in recs), "slo_all": sum(map(slo_ok, recs)) / len(recs),
                   "by_type": types}
    return out


def ttft_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    v = [(r["first_text_ns"] - r["sent_ns"]) / 1e9 for r in records
         if r.get("status") == "ok" and r.get("first_text_ns") and r.get("sent_ns") is not None]
    return {"n": len(v), "p50_s": pct(v, 0.5), "p90_s": pct(v, 0.9)}


def _conc(phase: str) -> int:
    m = re.search(r"c(\d+)$", phase)
    return int(m.group(1)) if m else 0


def routing_comparison(lo: dict[str, dict[str, Any]], sa: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Per level present under both routings: session-affinity minus least-outstanding."""
    out = []
    for lv in sorted(set(lo) & set(sa), key=lambda p: (p.split("_c")[0], _conc(p))):
        a, b = lo[lv], sa[lv]
        row = {"level": lv, "hit_gain_pts": (b["prefix_hit"] - a["prefix_hit"]) * 100}
        for k in ("ttft_p50_s", "ttft_p90_s", "request_p50_s", "session_p50_s"):
            if a.get(k) and b.get(k) is not None:
                row[k.replace("_s", "_rel")] = b[k] / a[k] - 1
        out.append(row)
    return out


def routing_verdict(cmp: list[dict[str, Any]], hit_pts: float = HIT_PTS, ttft_rel: float = TTFT_REL) -> dict[str, Any]:
    gains = [c["hit_gain_pts"] for c in cmp]
    rels = [c["ttft_p50_rel"] for c in cmp if c.get("ttft_p50_rel") is not None]
    hit = ("小（< 3 个百分点）" if gains and max(gains) < hit_pts else "恢复（每档 ≥ 3 个百分点）" if gains and min(gains) >= hit_pts else "不一致")
    faster = [x <= -ttft_rel for x in rels]
    ttft = "支持" if rels and all(faster) else ("不支持" if not any(faster) else "不确定")
    return {"hit": hit, "ttft": ttft, "hit_gain_pts": [min(gains, default=None), max(gains, default=None)],
            "ttft_p50_rel": [min(rels, default=None), max(rels, default=None)]}


def role_busy(util: dict[int, list[tuple[float, float, float]]], roles: dict[str, list[int]], start: float, end: float) -> dict[str, float | None]:
    """Mean sampled nvidia-smi busy % of each role's GPUs in [start, end)."""
    out = {}
    for role, gpus in roles.items():
        v = [b for g in gpus for t, b, _ in util.get(g, []) if start <= t < end]
        out[role] = sum(v) / len(v) if v else None
    return out


# ------------------------------------------------------------------------------------------
# runs


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(p)] if p.exists() else []


def _load(exp: Path) -> tuple[dict, dict]:
    return json.loads((exp / "result.json").read_text()), json.loads((exp / "analysis" / "analysis.json").read_text())


def _window_fn(exp: Path):  # noqa: ANN202
    from omni_research.analysis.experiment_analysis import clock_anchor

    anchor = clock_anchor(exp)
    if anchor is None:
        return None
    mono0, wall0 = anchor
    return lambda ns: (ns - mono0 + wall0) / 1e9


def _closed_tp(ana: dict, phase: str) -> dict[int, float]:
    return {int(r["level"]): r["tp_requests_per_s"] for r in ana["table"] if r["phase"] == phase and r.get("tp_requests_per_s") is not None}


def _errors(recs: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in recs:
        if r.get("status") != "ok":
            e = str(r.get("error") or r.get("status"))
            k = "client fd limit" if "Too many open files" in e else ("CUDA out of memory" if "out of memory" in e else e[:80])
            out[k] = out.get(k, 0) + 1
    return out


def scan_mix(exp: Path) -> dict[str, Any]:
    from exp01_gpu_timeline import load_util

    res, ana = _load(exp)
    layout = eid_layout(res["experiment_id"])
    roles = layout_roles(layout)
    util, wall = load_util(exp), _window_fn(exp)
    closed = _read_jsonl(exp / "raw" / "requests_mix_closed.jsonl")
    levels = mix_level_stats(closed)
    tp = _closed_tp(ana, "mix_closed")
    rows = []
    for c, lv in levels.items():
        recs = [r for r in closed if r.get("concurrency") == c and not r.get("warmup")]
        busy = None
        if wall and recs:
            busy = role_busy(util, roles, wall(min(r["sent_ns"] for r in recs)), wall(max(r["done_ns"] or r["sent_ns"] for r in recs)))
        rows.append({"concurrency": c, "rps": tp.get(c), **lv, "role_busy": busy})
    cap = max(tp.values()) if tp else None
    opened = mix_level_stats(_read_jsonl(exp / "raw" / "requests_mix_open.jsonl"), key="rate_rps")
    open_rows = [{"rate_rps": r, "of_closed_max": r / cap if cap else None, **lv} for r, lv in opened.items()]
    return {"run": exp.name, "experiment_id": res["experiment_id"], "layout": layout, "label": layout_label(layout),
            "thinker_gpus": thinker_gpus(layout), "node": NODE.get(exp.name[:8], "?"), "closed": rows, "closed_max_rps": cap,
            "closed_max_at": max(tp, key=tp.get) if tp else None, "open": open_rows, "errors": _errors(closed)}


def _gpu_phase_busy(ana: dict, phase: str, roles: dict[str, list[int]]) -> dict[str, float | None]:
    by = {int(g["gpu"]): g["sampled_util_mean"] for g in ana.get("gpu_by_phase", []) if g["phase"] == phase and g.get("sampled_util_mean") is not None}
    return {role: (sum(by[g] for g in gpus if g in by) / len([g for g in gpus if g in by]) if any(g in by for g in gpus) else None)
            for role, gpus in roles.items()}


def scan_agent(exp: Path) -> dict[str, Any]:
    res, ana = _load(exp)
    layout = eid_layout(res["experiment_id"])
    roles = layout_roles(layout)
    phases = {}
    for ph in res["phases"]:
        if ph.get("kind") != "agent":
            continue
        name = ph["phase"]
        a = (ana.get("agents") or {}).get(name) or {}
        recs = _read_jsonl(exp / "raw" / f"requests_{name}.jsonl")
        ok = [r for r in recs if r.get("status") == "ok" and r.get("done_ns")]
        span = (max(r["done_ns"] for r in ok) - min(r["sent_ns"] for r in recs)) / 1e9 if ok else None
        mr, ss = a.get("model_request_s") or {}, a.get("session_s") or {}
        tt = ttft_stats(recs)
        phases[name] = {
            "concurrency": _conc(name), "agent": "voice" if "voice" in name else "text",
            "routing": "session_affinity" if name.startswith("sa_") else "least_outstanding",
            "tool_latency_s": 0.5 if "toollat0p5" in name else (2.0 if "toollat2" in name else 0.0),
            "sessions": a.get("sessions"), "task_success": a.get("task_success"), "model_requests": a.get("model_requests"),
            "requests_per_session": (a.get("requests_per_session") or {}).get("mean"),
            "request_p50_s": mr.get("p50"), "request_p90_s": mr.get("p90"), "session_p50_s": ss.get("p50"), "session_p90_s": ss.get("p90"),
            "ttft_p50_s": tt["p50_s"], "ttft_p90_s": tt["p90_s"],
            "prefix_hit": ((ana.get("prefix_cache") or {}).get(name) or {}).get("cached_fraction"),
            "span_s": span, "requests_per_s": len(ok) / span if span else None,
            "mean_in_flight": sum((r["done_ns"] - r["sent_ns"]) / 1e9 for r in ok) / span if span else None,
            "tool_s_total": (a.get("time_split_s") or {}).get("tool"), "role_busy": _gpu_phase_busy(ana, name, roles),
            "errors": _errors(recs)}
    return {"run": exp.name, "experiment_id": res["experiment_id"], "layout": layout, "label": layout_label(layout),
            "thinker_gpus": thinker_gpus(layout), "node": NODE.get(exp.name[:8], "?"), "phases": phases}


def scan_w3b(exp: Path) -> dict[str, Any]:
    res, ana = _load(exp)
    layout = eid_layout(res["experiment_id"])
    tp = _closed_tp(ana, "w3b_closed")
    return {"run": exp.name, "experiment_id": res["experiment_id"], "layout": layout, "label": layout_label(layout),
            "thinker_gpus": thinker_gpus(layout), "closed_max_rps": max(tp.values()) if tp else None,
            "closed_max_at": max(tp, key=tp.get) if tp else None}


# ------------------------------------------------------------------------------------------
# verdicts


def _agent_key(phases: dict[str, Any], agent: str, conc: int) -> dict[str, Any] | None:
    """The default-routing, no-tool-latency phase of an agent type at a concurrency."""
    for name, p in phases.items():
        if p["agent"] == agent and p["concurrency"] == conc and p["routing"] == "least_outstanding" and p["tool_latency_s"] == 0.0:
            return p
    return None


def verdicts(w3b: list[dict], mix: list[dict], agent: list[dict], tol: float = TOL) -> dict[str, Any]:
    w3b_best: dict[str, float] = {}
    for r in w3b:  # best of the repeats per layout
        if r["closed_max_rps"] is not None:
            w3b_best[r["layout"]] = max(w3b_best.get(r["layout"], 0.0), r["closed_max_rps"])
    w3b_near = near_best(w3b_best, tol)
    ref = W3B_REFERENCE_THINKER_GPUS
    out: dict[str, Any] = {"w3b_reference": {"closed_max_rps": w3b_best, "near_best": w3b_near, "thinker_gpus": ref}}
    mix_max = {m["layout"]: m["closed_max_rps"] for m in mix}
    mix_c256 = {m["layout"]: next((x["slo_all"] for x in m["closed"] if x["concurrency"] == 256), None) for m in mix}
    out["B-H1"] = {**more_thinker_verdict(near_best(mix_max, tol), ref), "metric": "闭环最高吞吐（请求/秒）", "values": mix_max,
                   "secondary": {**more_thinker_verdict(near_best(mix_c256, tol), ref), "metric": "闭环并发 256 时满足 SLO 的比例", "values": mix_c256}}
    per = {}
    for agent_type in ("text", "voice"):
        for conc in (256, 512):
            vals = {a["layout"]: (_agent_key(a["phases"], agent_type, conc) or {}).get("session_p50_s") for a in agent}
            per[f"{agent_type}_c{conc}"] = {**more_thinker_verdict(near_best(vals, tol, higher_is_better=False), ref),
                                           "metric": "会话时长中位数（秒）", "values": vals}
    text = [per["text_c256"]["verdict"], per["text_c512"]["verdict"]]
    out["B-H2"] = {"verdict": "支持" if all(v == "支持" for v in text) else ("不支持" if all(v == "不支持" for v in text) else "不确定"),
                   "rule": "文本 Agent 并发 256 和 512 两档都支持才算支持", "levels": per}
    r8 = next((a for a in agent if a["layout"] == "r8"), None)
    if r8:
        lo = {k.removeprefix("lo_"): v for k, v in r8["phases"].items() if k.startswith("lo_")}
        sa = {k.removeprefix("sa_"): v for k, v in r8["phases"].items() if k.startswith("sa_")}
        cmp = routing_comparison(lo, sa)
        out["B-H3"] = {"part_a": {"verdict": "无法检验", "why": "plan v2 没有同一种 Agent 负载的单卡运行"},
                       "part_b": routing_verdict(cmp), "comparison": cmp}
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--mix", action="append", required=True)
    ap.add_argument("--agent", action="append", required=True)
    ap.add_argument("--agent-short", action="append", default=[])
    ap.add_argument("--w3b", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    mix = [scan_mix(root / r) for r in a.mix]
    agent = [scan_agent(root / r) for r in a.agent]
    short = [scan_agent(root / r) for r in a.agent_short]
    w3b = [scan_w3b(root / r) for r in a.w3b]
    out = Path(a.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    summary = {"generated_by": "scripts/analysis/exp19_report.py",
               "rule": {"near_best_tolerance": TOL, "near_best_tolerance_wide": TOL_WIDE, "hit_gain_pts": HIT_PTS, "ttft_rel": TTFT_REL, "slo_ttft_s": TTFT_S,
                        "slo_speech_e2e_per_audio_s": SPEECH_E2E_PER_AUDIO_S},
               "w3b": w3b, "mix": mix, "agent": agent, "agent_short": short, "verdicts": verdicts(w3b, mix, agent),
               "verdicts_tol_wide": verdicts(w3b, mix, agent, TOL_WIDE)}
    (out / "results" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    write_csv(out / "results" / "mix_closed.csv", [
        {"layout": m["label"], "node": m["node"], "concurrency": x["concurrency"], "rps": x["rps"], "slo_all": x["slo_all"],
         **{f"{t}_{k}": x["by_type"].get(t, {}).get(k) for t in TYPES for k in ("slo", "ttft_p90_s", "e2e_p50_s")}}
        for m in mix for x in m["closed"]])
    write_csv(out / "results" / "agent_phases.csv", [
        {"layout": g["label"], "batch": batch, "phase": name, **{k: v for k, v in p.items() if k not in ("role_busy", "errors")},
         **{f"busy_{k}": v for k, v in (p["role_busy"] or {}).items()}}
        for batch, group in (("第二批（加长）", agent), ("第一批（每档太短）", short)) for g in group for name, p in g["phases"].items()])
    for key in ("verdicts", "verdicts_tol_wide"):
        v = summary[key]
        print(key, "B-H1", v["B-H1"]["verdict"], v["B-H1"]["near_best"], v["B-H1"]["secondary"]["verdict"],
              "| B-H2", v["B-H2"]["verdict"], {k: x["near_best"] for k, x in v["B-H2"]["levels"].items()}, "| B-H3b", v.get("B-H3", {}).get("part_b"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
