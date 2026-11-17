#!/usr/bin/env python3
"""Per-load-level server-side stage breakdown, effective batch size and GPU utilization by role.

    python scripts/analysis/stage_by_level.py EXP_DIR [EXP_DIR ...] [--phase w3b_small] --out stages.md

For every phase with server events (raw/server_events/<phase>/s<N>/), joins each request's
server-side intervals (experiment_analysis.stage_breakdown) with its client record and groups them
by load level (closed-loop concurrency or open-loop rate). Roles are stage names without the
replica suffix (thinker@r1 -> thinker). Warmup requests are left out of the latency columns.

Measured, per level: client TTFT / e2e, and per role the median and p90 of queue (enqueued ->
prefill start), prefill, model path (prefill + all decode steps), code2wav service, and the hop
from thinker to talker. Derived, per level:
- effective batch: the time-averaged number of requests inside one replica's model path (code2wav:
  service), i.e. the sum of those intervals over the level window divided by window x replicas
  (Little's law; warmup requests included because they occupy the replica too);
- sampled GPU utilization of each role's GPUs over the same window (nvidia-smi: share of the sample
  period with a kernel running, not SM occupancy).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from omni_research.analysis.experiment_analysis import load_gpu_samples, read_jsonl, stage_breakdown  # noqa: E402
from omni_research.analysis.metrics import request_metrics  # noqa: E402
from summarize_plan_v2 import eid_layout, f, layout_roles, pct, slo_ok  # noqa: E402

ROLES = ("thinker", "talker", "code2wav")
OCCUPANCY_METRIC = {"thinker": "model_path", "talker": "model_path", "code2wav": "service"}


def role_of(stage: str) -> str:
    """'s3/thinker@r1' -> 'thinker'."""
    return re.sub(r"@r\d+$", "", stage.split("/")[-1])


def label_role(label: str) -> tuple[str, str]:
    """'s0/talker@r2.queue' -> ('talker', 'queue'); 'hop.s0/thinker@r0->s0/talker@r2' -> ('hop', 'thinker->talker')."""
    if label.startswith("hop."):
        a, b = label[4:].split("->")
        return "hop", f"{role_of(a)}->{role_of(b)}"
    stage, metric = label.rsplit(".", 1)
    return role_of(stage), metric


def load_phase_events(exp: Path, phase: str) -> list[dict[str, Any]]:
    """Server events of one phase; stage names get the server directory as prefix so the replicas of
    separate servers (R8: eight servers that each call their thinker 'thinker') stay apart."""
    base = exp / "raw" / "server_events" / phase
    evs: list[dict[str, Any]] = []
    for sdir in sorted(p for p in base.iterdir() if p.is_dir()) if base.exists() else []:
        for fp in sorted(sdir.rglob("*.jsonl")):
            for e in read_jsonl(fp):
                if e.get("stage"):
                    e["stage"] = f"{sdir.name}/{e['stage']}"
                evs.append(e)
    return evs


def level_of(rec: dict[str, Any]) -> float | None:
    for k in ("concurrency", "rate_rps"):
        if rec.get(k) is not None:
            return float(rec[k])
    return None


def breakdown(exp: Path, phase: str) -> list[dict[str, Any]]:
    """One row per load level of ``phase`` (see the module docstring)."""
    evs = load_phase_events(exp, phase)
    if not evs:
        return []
    recs = {r["request_id"]: r for r in read_jsonl(exp / "raw" / f"requests_{phase}.jsonl")}
    per_req = stage_breakdown(evs)["per_request"]
    # Request window on the server clock, and the replicas each role has.
    span: dict[str, list[int]] = {}
    replicas: dict[str, set[str]] = defaultdict(set)
    for e in evs:
        rid, t, st = e.get("request_id"), e.get("timestamp_ns"), e.get("stage")
        if st and role_of(st) in ROLES:
            replicas[role_of(st)].add(st)
        if rid in recs and t is not None:
            s = span.setdefault(rid, [t, t])
            s[0], s[1] = min(s[0], t), max(s[1], t)
    gpu = load_gpu_samples(exp)
    try:
        eid = json.load(open(exp / "result.json"))["experiment_id"]
        roles_gpus = layout_roles(eid_layout(eid))
    except (OSError, KeyError, StopIteration, ValueError):
        roles_gpus = {}
    by_level: dict[float, list[str]] = defaultdict(list)
    for rid, r in recs.items():
        lv = level_of(r)
        if lv is not None and rid in span:
            by_level[lv].append(rid)
    rows = []
    for lv, rids in sorted(by_level.items()):
        t0 = min(span[r][0] for r in rids)
        t1 = max(span[r][1] for r in rids)
        window_s = (t1 - t0) / 1e9
        measured = [r for r in rids if not recs[r].get("warmup")]
        vals: dict[tuple[str, str], list[float]] = defaultdict(list)
        for r in measured:
            got: dict[tuple[str, str], float] = {}
            for label, ms in (per_req.get(r) or {}).items():
                vals[label_role(label)].append(ms)
                got[label_role(label)] = ms
            # Normalized by output length: answers differ in length between levels (plan v2's c=1
            # sample drew shorter ones), so per-token and per-audio-second times compare across levels.
            rec = recs[r]
            toks, audio = rec.get("completion_tokens"), rec.get("output_audio_s")
            if ("thinker", "model_path") in got and toks and toks > 1:
                vals[("thinker", "ms_per_token")].append((got[("thinker", "model_path")] - got.get(("thinker", "prefill"), 0)) / (toks - 1))
            for role, metric in (("talker", "model_path"), ("code2wav", "service")):
                if (role, metric) in got and audio:
                    vals[(role, "ms_per_audio_s")].append(got[(role, metric)] / audio)
            if toks:
                vals[("output", "tokens")].append(toks)
            if audio:
                vals[("output", "audio_s")].append(audio)
        occ_sum: dict[str, float] = defaultdict(float)
        for r in rids:
            for label, ms in (per_req.get(r) or {}).items():
                role, metric = label_role(label)
                if OCCUPANCY_METRIC.get(role) == metric:
                    occ_sum[role] += ms / 1e3
        ms_ = [request_metrics(recs[r]) for r in measured if recs[r].get("status") == "ok"]
        sent = [recs[r]["sent_ns"] for r in measured if recs[r].get("sent_ns")]
        done = [recs[r]["done_ns"] for r in measured if recs[r].get("done_ns")]
        row: dict[str, Any] = {
            "level": lv, "n": len(measured), "errors": sum(1 for r in measured if recs[r].get("status") != "ok"),
            "tp_rps": len(done) / ((max(done) - min(sent)) / 1e9) if done and sent and max(done) > min(sent) else None,
            "ttft_s": [m["ttft_s"] for m in ms_ if m.get("ttft_s") is not None],
            "e2e_s": [m["e2e_s"] for m in ms_ if m.get("e2e_s") is not None],
            "e2e_per_audio_s": [m["e2e_per_audio_s"] for m in ms_ if m.get("e2e_per_audio_s") is not None],
            "slo": sum(1 for r in measured if slo_ok(recs[r])) / len(measured) if measured else None,
            "stage_ms": dict(vals), "window_s": window_s,
            "eff_batch": {role: occ_sum[role] / (window_s * len(replicas[role]))
                          for role in ROLES if replicas.get(role) and window_s > 0},
            "replicas": {role: len(replicas[role]) for role in ROLES if replicas.get(role)},
        }
        util: dict[str, list[float]] = defaultdict(list)
        for s in gpu:
            if t0 / 1e9 <= s["t"] <= t1 / 1e9:
                util[s["gpu"]].append(s["util"])
        row["gpu_util"] = {role: sum(sum(util[str(g)]) / len(util[str(g)]) for g in gs if util.get(str(g))) /
                           max(1, sum(1 for g in gs if util.get(str(g)))) for role, gs in roles_gpus.items()
                           if any(util.get(str(g)) for g in gs)}
        rows.append(row)
    return rows


def p50_p90(v: list[float] | None, scale: float = 1.0, nd: int = 2) -> str:
    if not v:
        return "—"
    return f"{f(pct(v, 0.5) * scale, nd)} / {f(pct(v, 0.9) * scale, nd)}"


def section(exp: Path, phase: str, rows: list[dict[str, Any]]) -> list[str]:
    name = exp.name
    roles_util = sorted({r for row in rows for r in row["gpu_util"]})
    L = [f"### {name} · {phase}", "",
         "延迟（秒，中位数 / p90；客户端测量）和服务端各阶段（毫秒，中位数 / p90）。", "",
         "| 档 | n | 吞吐 请求/秒 | TTFT | e2e | e2e/语音秒 | 满足 SLO | thinker 排队 | thinker prefill | thinker 模型 | thinker→talker | talker 排队 | talker 模型 | code2wav |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        sm = row["stage_ms"]
        L.append(f"| {row['level']:g} | {row['n']} | {f(row['tp_rps'])} | {p50_p90(row['ttft_s'], nd=3)} | {p50_p90(row['e2e_s'])} | "
                 f"{p50_p90(row['e2e_per_audio_s'], nd=3)} | {f(row['slo'])} | {p50_p90(sm.get(('thinker', 'queue')), nd=0)} | "
                 f"{p50_p90(sm.get(('thinker', 'prefill')), nd=0)} | {p50_p90(sm.get(('thinker', 'model_path')), nd=0)} | "
                 f"{p50_p90(sm.get(('hop', 'thinker->talker')), nd=1)} | {p50_p90(sm.get(('talker', 'queue')), nd=0)} | "
                 f"{p50_p90(sm.get(('talker', 'model_path')), nd=0)} | {p50_p90(sm.get(('code2wav', 'service')), nd=0)} |")
    L += ["", "按输出长度归一化（中位数 / p90）：thinker 每个解码 token 的毫秒数（模型路径减 prefill，除以输出 token 数减 1），"
          "talker 和 code2wav 每秒输出语音用的毫秒数。", "",
          "| 档 | 输出 token | 输出语音 s | thinker ms/token | talker ms/语音秒 | code2wav ms/语音秒 |", "|---|---|---|---|---|---|"]
    for row in rows:
        sm = row["stage_ms"]
        L.append(f"| {row['level']:g} | {p50_p90(sm.get(('output', 'tokens')), nd=0)} | {p50_p90(sm.get(('output', 'audio_s')), nd=1)} | "
                 f"{p50_p90(sm.get(('thinker', 'ms_per_token')), nd=1)} | {p50_p90(sm.get(('talker', 'ms_per_audio_s')), nd=0)} | "
                 f"{p50_p90(sm.get(('code2wav', 'ms_per_audio_s')), nd=0)} |")
    L += ["", "有效 batch（每个副本模型路径里平均的请求数，Derived）和采样 GPU 利用率（%，Measured）。", "",
          "| 档 | 窗口 s | " + " | ".join(f"{r} 有效 batch（{rows[0]['replicas'].get(r, '?')} 个副本）" for r in ROLES if r in rows[0]["replicas"]) +
          " | " + " | ".join(f"GPU {r}" for r in roles_util) + " |",
          "|---|---|" + "---|" * (sum(1 for r in ROLES if r in rows[0]["replicas"]) + len(roles_util))]
    for row in rows:
        L.append(f"| {row['level']:g} | {row['window_s']:.0f} | " +
                 " | ".join(f(row["eff_batch"].get(r)) for r in ROLES if r in rows[0]["replicas"]) + " | " +
                 " | ".join(f(row["gpu_util"].get(r), 0) for r in roles_util) + " |")
    return L + [""]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--phase", action="append", help="phase name (repeatable); default: every phase with server events")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", help="also write the rows (without per-request lists) as JSON")
    a = ap.parse_args()
    L = ["# 按负载档的阶段分解", "", "由 `scripts/analysis/stage_by_level.py` 生成。", ""]
    dump = {}
    for e in a.experiments:
        exp = Path(e).expanduser()
        base = exp / "raw" / "server_events"
        phases = a.phase or (sorted(p.name for p in base.iterdir() if p.is_dir()) if base.exists() else [])
        for ph in phases:
            rows = breakdown(exp, ph)
            if rows:
                L += section(exp, ph, rows)
                dump[f"{exp.name}/{ph}"] = [{**{k: v for k, v in r.items() if k not in ("ttft_s", "e2e_s", "e2e_per_audio_s", "stage_ms")},
                                              "ttft_s_p50": pct(r["ttft_s"], 0.5), "ttft_s_p90": pct(r["ttft_s"], 0.9),
                                              "e2e_s_p50": pct(r["e2e_s"], 0.5), "e2e_s_p90": pct(r["e2e_s"], 0.9),
                                              "e2e_per_audio_s_p90": pct(r["e2e_per_audio_s"], 0.9),
                                              "stage_ms_p50": {f"{k[0]}.{k[1]}": pct(v, 0.5) for k, v in r["stage_ms"].items()},
                                              "stage_ms_p90": {f"{k[0]}.{k[1]}": pct(v, 0.9) for k, v in r["stage_ms"].items()}}
                                             for r in rows]
    Path(a.out).expanduser().write_text("\n".join(L) + "\n")
    if a.json:
        Path(a.json).expanduser().write_text(json.dumps(dump, indent=1, ensure_ascii=False))
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
