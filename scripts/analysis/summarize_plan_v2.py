#!/usr/bin/env python3
"""Summary tables for plan v2 (docs/04_experiments/plan_8gpu_v2.md) across its experiments.

    python scripts/analysis/summarize_plan_v2.py --data-root ~/research/data/unicorn \
        --prefix 20261115_v2_ --out summary.md

Reads each run directory's result.json and analysis/analysis.json (run analyze_experiment.py first;
for the W3b experiments with --slo ttft_s=1.0,e2e_per_audio_s=0.5) and, for mixed workloads,
raw/requests_mix_*.jsonl. All numbers are Measured except the SLO capacity, which is Derived: the
highest offered open-loop rate up to which every level kept >= 95% of requests within the SLO
(">" when even the highest tested rate passed, so the capacity is above the tested range).

Plan v2 SLOs (request level): speech output e2e / output-audio seconds <= 0.5 and TTFT <= 1 s;
text output TTFT <= 1 s.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from omni_research.analysis.metrics import request_metrics  # noqa: E402

SLO_FRACTION = 0.95
SPEECH_E2E_PER_AUDIO_S = 0.5
TTFT_S = 1.0


def slo_ok(rec: dict[str, Any]) -> bool:
    """Plan v2 request-level SLO; speech when the request asked for audio output."""
    if rec.get("status") != "ok":
        return False
    m = request_metrics(rec)
    if m.get("ttft_s") is None or m["ttft_s"] > TTFT_S:
        return False
    if "audio" in (rec.get("output_modalities") or []):
        return m.get("e2e_per_audio_s") is not None and m["e2e_per_audio_s"] <= SPEECH_E2E_PER_AUDIO_S
    return True


def slo_capacity(levels: list[tuple[float, float]], threshold: float = SLO_FRACTION) -> tuple[float | None, bool]:
    """(highest offered rate before the first level under ``threshold``, whether such a level exists)."""
    best = None
    for rate, frac in sorted(levels):
        if frac is None or frac < threshold:
            return best, True
        best = rate
    return best, False


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def f(v: Any, nd: int = 2) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def load(d: Path) -> tuple[dict, dict] | None:
    """(result.json, analysis/analysis.json with the run's config.json under "config")."""
    try:
        res, ana = json.load(open(d / "result.json")), json.load(open(d / "analysis" / "analysis.json"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        ana["config"] = json.load(open(d / "config.json"))
    except (OSError, json.JSONDecodeError):
        ana["config"] = {}
    return res, ana


def layout_roles(layout: str) -> dict[str, list[int]]:
    """GPU indices per role for plan v2's layout names (r8, p4, s2x6, ...)."""
    if layout.startswith("s"):
        t, k = (int(x) for x in layout[1:].split("x"))
        return {"thinker": list(range(t)), "talker": list(range(t, t + k))}
    if layout.startswith("p"):
        n = int(layout[1:])
        return {"thinker": [2 * i for i in range(n)], "talker": [2 * i + 1 for i in range(n)]}
    return {"full replica": list(range(int(layout[1:])))}


def eid_layout(eid: str) -> str:
    """v2_a_s2x6_run1 -> s2x6; v2b_aoh_r8 -> r8; v2_b1_p4 -> p4; v2b_a_s2x6_c03 -> s2x6."""
    parts = eid.split("_")
    return next(p for p in parts[2:] if p[:1] in "rps" and any(c.isdigit() for c in p))


# ------------------------------------------------------------------------------------------
# sections


def gate_section(runs: list[tuple[Path, dict, dict]]) -> list[str]:
    L = ["## 稳定性门槛（Y0s）", "",
         "| 实验 | 开环档数 × 每档 | 服务退出（重启）次数 | 受影响的档 | 失败请求 / 总请求 | 每小时崩溃 |", "|---|---|---|---|---|---|"]
    for d, res, _ in runs:
        if "_y0s_" not in d.name:
            continue
        soak = [p for p in res["phases"] if p["phase"] == "w3b_soak"]
        if not soak:
            continue
        rates = soak[0]["rates"]
        minutes = sum(lv["wall_s"] for lv in rates) / 60
        n_req = sum(lv["requests"] for lv in rates)
        n_err = sum(lv["errors"] for lv in rates)
        crashes = len(res.get("server_restarts") or [])
        affected = sum(1 for lv in rates if lv.get("crash_affected"))
        L.append(f"| {res['experiment_id']} | {len(rates)} × {rates[0]['wall_s'] / 60:.1f} 分钟（共 {minutes:.0f} 分钟） | {crashes} | {affected} | "
                 f"{n_err} / {n_req} | {crashes / (minutes / 60):.1f} |")
    return L + [""]


def a_section(runs: list[tuple[Path, dict, dict]], title: str) -> list[str]:
    L = [f"## {title}", "",
         "闭环：每档 max(96, 2×并发) 个请求。开环：每档 2 分钟（第一批按各自闭环容量的 50%–120%），或 90 秒（第二批绝对速率）。", "",
         "| 实验 | 闭环最高吞吐（请求/秒 @并发） | 并发 64 e2e 中位数 s | 开环最高满足 SLO 的速率（请求/秒） | 开环最高 goodput（请求/秒） | 客户端发送延迟 p90 最大值 s |",
         "|---|---|---|---|---|---|"]
    detail: list[str] = []
    for d, res, ana in runs:
        closed = [r for r in ana["table"] if r["phase"] == "w3b_closed"]
        opens = [r for r in ana["table"] if r["phase"].startswith("w3b_open")]
        best = max(closed, key=lambda r: r.get("tp_requests_per_s") or 0, default=None)
        c64 = next((r for r in closed if r["level"] == "64"), None)
        cap, reached = slo_capacity([(float(r["level"]), r.get("goodput_fraction")) for r in opens])
        good = max((r.get("goodput_req_per_s") or 0 for r in opens), default=None)
        late = lateness_p90(d, [p for p in res["phases"] if p["phase"].startswith("w3b_open")])
        L.append(f"| {res['experiment_id']} | " + (f"{best['tp_requests_per_s']:.2f} @{best['level']}" if best else "—") + " | "
                 f"{f(c64 and c64.get('e2e_s_median'), 1)} | " + ("—" if cap is None else (f"{cap:.1f}" if reached else f"> {cap:.1f}")) +
                 f" | {f(good)} | {f(late)} |")
        if opens:
            detail += [f"**{res['experiment_id']}** 开环", "",
                       "| 提供速率 | 实际吞吐 | n | 失败 | e2e 中位数 | e2e p90 | TTFT p90 | e2e/语音秒 p90 | 满足 SLO 比例 |", "|---|---|---|---|---|---|---|---|---|"]
            for r in opens:
                detail.append(f"| {float(r['level']):.2f} | {f(r.get('tp_requests_per_s'))} | {r['n']} | {r['errors']} | {f(r.get('e2e_s_median'))} | "
                              f"{f(r.get('e2e_s_p90'))} | {f(r.get('ttft_s_p90'), 3)} | {f(r.get('e2e_per_audio_s_p90'))} | {f(r.get('goodput_fraction'))} |")
            detail.append("")
    return L + ["", *detail]


def lateness_p90(d: Path, phases: list[dict]) -> float | None:
    """Worst per-level p90 of open-loop send lateness (scheduled -> sent): the client falling behind."""
    worst = None
    for ph in phases:
        by: dict[Any, list[float]] = {}
        try:
            for line in open(d / "raw" / f"requests_{ph['phase']}.jsonl"):
                rec = json.loads(line)
                if rec.get("lateness_s") is not None:
                    by.setdefault(rec.get("rate_rps"), []).append(rec["lateness_s"])
        except OSError:
            continue
        for vals in by.values():
            p = pct(vals, 0.9)
            worst = p if worst is None or (p is not None and p > worst) else worst
    return worst


def role_util_section(runs: list[tuple[Path, dict, dict]], phase_prefix: str) -> list[str]:
    L = ["## GPU 采样利用率按角色（开环阶段的均值，%）", "",
         "“采样利用率”是 nvidia-smi 采样周期内有 kernel 在跑的时间比例，不是 SM 利用率。", "",
         "| 实验 | 角色 | GPU | 均值 | 最低卡 | 最高卡 |", "|---|---|---|---|---|---|"]
    for d, res, ana in runs:
        layout = eid_layout(res["experiment_id"])
        util = {}
        for g in ana["gpu_by_phase"]:
            if g["phase"].startswith(phase_prefix):
                util.setdefault(int(g["gpu"]), []).append(g["sampled_util_mean"])
        if not util:
            continue
        for role, gpus in layout_roles(layout).items():
            vals = [sum(util[g]) / len(util[g]) for g in gpus if g in util]
            if vals:
                L.append(f"| {res['experiment_id']} | {role} | {','.join(map(str, gpus))} | {sum(vals) / len(vals):.0f} | {min(vals):.0f} | {max(vals):.0f} |")
    return L + [""]


def mix_section(runs: list[tuple[Path, dict, dict]]) -> list[str]:
    L = ["## B1 混合负载，按请求类型", "",
         "W1 = 文本对话（文本输出），W3b = 文本提问语音回答，W2 = 语音提问文本回答。SLO 按输出类型判断（见文件头）。", ""]
    for d, res, _ in runs:
        L += [f"**{res['experiment_id']}**", "", "| 阶段 | 负载 | 类型 | n | e2e 中位数 s | e2e p90 s | TTFT p90 s | 满足 SLO 比例 |", "|---|---|---|---|---|---|---|---|"]
        for ph in ("mix_closed", "mix_open"):
            groups: dict[tuple[Any, str], list[dict]] = {}
            try:
                for line in open(d / "raw" / f"requests_{ph}.jsonl"):
                    rec = json.loads(line)
                    if rec.get("warmup"):
                        continue
                    lv = rec.get("concurrency") if ph == "mix_closed" else rec.get("rate_rps")
                    groups.setdefault((lv, rec.get("workload_id", "?")), []).append(rec)
            except OSError:
                continue
            for (lv, wid), recs in sorted(groups.items(), key=lambda kv: (float(kv[0][0] or 0), kv[0][1])):
                ms = [request_metrics(r) for r in recs if r.get("status") == "ok"]
                e2e = [m["e2e_s"] for m in ms if m.get("e2e_s") is not None]
                ttft = [m["ttft_s"] for m in ms if m.get("ttft_s") is not None]
                ok = sum(1 for r in recs if slo_ok(r)) / len(recs)
                L.append(f"| {ph} | {lv} | {wid.split('_')[0]} | {len(recs)} | {f(pct(e2e, 0.5))} | {f(pct(e2e, 0.9))} | {f(pct(ttft, 0.9), 3)} | {ok:.2f} |")
        L.append("")
    return L


def agent_section(runs: list[tuple[Path, dict, dict]]) -> list[str]:
    L = ["## B2 Agent", "",
         "| 实验 | 阶段 | 用时 s | 会话 | 任务成功 | 模型请求 | 单次请求中位数 / p90 s | 会话中位数 / p90 s | 前缀缓存命中 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for d, res, ana in runs:
        walls = {p["phase"]: p["ended"] - p["started"] for p in res["phases"] if p.get("ended") and p.get("started")}
        for name, a in (ana.get("agents") or {}).items():
            mr, ss = a.get("model_request_s") or {}, a.get("session_s") or {}
            pc = (ana.get("prefix_cache") or {}).get(name) or {}
            L.append(f"| {res['experiment_id']} | {name} | {f(walls.get(name), 0)} | {a.get('sessions')} | {a.get('task_success')} | {a.get('model_requests')} | "
                     f"{f(mr.get('p50'))} / {f(mr.get('p90'))} | {f(ss.get('p50'))} / {f(ss.get('p90'))} | {f(pc.get('cached_fraction'))} |")
    return L + [""]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--prefix", action="append", required=True, help="run directory prefix, e.g. 20261115_v2_ (repeatable)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    runs = []
    for prefix in a.prefix:
        for d in sorted(root.glob(f"{prefix}*")):
            loaded = load(d)
            if loaded:
                runs.append((d, *loaded))
    by = lambda key: [r for r in runs if key in r[1]["experiment_id"]]  # noqa: E731
    w3b8 = [r for r in runs if ("_a_" in r[1]["experiment_id"] or "_aoh_" in r[1]["experiment_id"]) and r[1]["experiment_id"] != "v2_a_r1"
            and r[2]["config"].get("hardware_label") == "a6000x8"]
    w3b_small = [r for r in runs if "_a_" in r[1]["experiment_id"] and r[2]["config"].get("hardware_label") in ("a6000x1", "a6000x2", "a6000x4")]
    L = [f"# plan v2 汇总（{', '.join(a.prefix)}）", "", f"由 `scripts/analysis/summarize_plan_v2.py` 生成，{len(runs)} 个运行目录。", ""]
    L += gate_section(runs)
    L += a_section(w3b8, "A：8 卡 W3b")
    L += a_section(w3b_small, "A：1、2、4 卡 W3b")
    L += role_util_section(w3b8, "w3b_open")
    L += mix_section(by("_b1_"))
    L += agent_section([r for r in runs if "_b2" in r[1]["experiment_id"]])
    Path(a.out).expanduser().write_text("\n".join(L) + "\n")
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
