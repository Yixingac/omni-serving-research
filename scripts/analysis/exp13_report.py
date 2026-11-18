#!/usr/bin/env python3
"""实验 13：thinker 和 talker 之间的 GPU 连接快慢有没有影响。只分析 plan v2 已有的运行（G1 组和拓扑测量），没有新的 GPU 运行。

    python scripts/analysis/exp13_report.py --data-root <results root> --topology 20261116_v2b_y0_env_topology_run01 \
        --run 20261116_v2b_g1a_links_run01 --run 20261116_v2b_g1b_links_run01 \
        --out research/01_性能瓶颈/实验13_GPU间连接快慢

Each G1 run serves W3b with four 2-GPU split services at once (thinker on the first GPU, talker and code2wav on the
second); the client hands requests to the services in turn. A service's name carries the nvidia-smi topo link of its
GPU pair: NV4 = four NVLinks, PIX = one PCIe switch, NODE = PCIe host bridges within one NUMA node, SYS = across the
CPU sockets. The hand-off time is the next stage's input-received event minus the previous stage's hop-sent event
(server events). The topology run measured peer-to-peer copy bandwidth for every GPU pair and NCCL all-reduce on one
NVLink pair. The verdict rule for G-H1 is in the experiment's hypotheses.md; it was set after the data had been
seen, so the verdict is exploratory.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from omni_research.analysis.metrics import percentile, request_metrics  # noqa: E402

TOL = 0.05  # G-H1's own "差别不到 5%"
TTFT_S = 1.0
SPEECH_E2E_PER_AUDIO_S = 0.5
PHASES = ("w3b_closed", "w3b_open_abs")


def pct(values: list[float | None], q: float) -> float | None:
    v = [x for x in values if x is not None]
    return percentile(v, q) if v else None


def link_of(service: str) -> str:
    """g_nv4a -> NV4, g_pix -> PIX, g_sysb -> SYS (the trailing a/b only tells two services of one link apart); a run
    after '#' (g_pix#g1b_links) is ignored."""
    return service.split("#")[0].removeprefix("g_").rstrip("ab").upper()


def bandwidth_by_link(topo: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Peer-to-peer copy bandwidth of every measured GPU pair, both directions, grouped by link type."""
    by: dict[str, list[float]] = {}
    pairs: dict[str, int] = {}
    for p in topo.get("pairs", []):
        a, b = p["pair"]
        vals = [p.get(f"{a}->{b}_GBps"), p.get(f"{b}->{a}_GBps")]
        by.setdefault(p["link"], []).extend(v for v in vals if v is not None)
        pairs[p["link"]] = pairs.get(p["link"], 0) + 1
    return {k: {"pairs": pairs[k], "median_GBps": statistics.median(v), "min_GBps": min(v), "max_GBps": max(v)} for k, v in sorted(by.items())}


def link_verdict(hop_ms: dict[str, float], e2e_s: dict[str, float], tol: float = TOL) -> dict[str, Any]:
    """G-H1: the hand-off differs by less than tol of the end-to-end time, and the services on slower links are within
    tol of the NVLink services. A larger latency gap counts against G-H1 only when it exceeds the gap between two
    services on the same kind of link (otherwise 不确定)."""
    med = statistics.median(e2e_s.values())
    spread = (max(hop_ms.values()) - min(hop_ms.values())) / 1e3
    nv = [v for k, v in e2e_s.items() if link_of(k) == "NV4"]
    other = [v for k, v in e2e_s.items() if link_of(k) != "NV4"]
    rel = (sum(other) / len(other)) / (sum(nv) / len(nv)) - 1 if nv and other else None
    same: list[float] = []
    for link in {link_of(k) for k in e2e_s}:
        vals = [v for k, v in e2e_s.items() if link_of(k) == link]
        if len(vals) > 1:
            same.append(max(vals) / min(vals) - 1)
    same_spread = max(same) if same else None
    if spread / med >= tol or rel is None:
        v = "不确定"
    elif abs(rel) < tol:
        v = "支持"
    else:
        v = "不支持" if same_spread is None or abs(rel) > same_spread else "不确定"
    return {"verdict": v, "hop_spread_ms": spread * 1e3, "hop_spread_share_of_e2e": spread / med, "e2e_rel_non_nvlink": rel,
            "same_link_spread": same_spread, "e2e_median_s": med}


def slo_ok(rec: dict[str, Any]) -> bool:
    if rec.get("status") != "ok":
        return False
    m = request_metrics(rec)
    if m.get("ttft_s") is None or m["ttft_s"] > TTFT_S:
        return False
    return m.get("e2e_per_audio_s") is not None and m["e2e_per_audio_s"] <= SPEECH_E2E_PER_AUDIO_S


def _level(r: dict[str, Any]) -> Any:
    return r.get("concurrency") if r.get("concurrency") is not None else r.get("rate_rps")


def scan_topology(exp: Path) -> dict[str, Any]:
    topo = json.loads((exp / "raw" / "p2p.json").read_text())
    return {"run": exp.name, "links": topo.get("links"), "bandwidth_by_link": bandwidth_by_link(topo),
            "pairs": [{"pair": p["pair"], "link": p["link"], "GBps": [p.get(f"{p['pair'][0]}->{p['pair'][1]}_GBps"), p.get(f"{p['pair'][1]}->{p['pair'][0]}_GBps")],
                       "peer_access": p.get("peer_access")} for p in topo.get("pairs", [])],
            "allreduce": [{k: a.get(k) for k in ("link", "devices", "MiB", "algbw_GBps", "busbw_GBps", "p2p_disabled")} for a in topo.get("allreduce", [])]}


def scan_run(exp: Path) -> dict[str, Any]:
    from omni_research.analysis.experiment_analysis import read_jsonl, stage_breakdown

    res = json.loads((exp / "result.json").read_text())
    cfg = json.loads((exp / "config.json").read_text())
    gpus = {s["name"]: s["gpus"] for s in cfg.get("servers", [])}
    phases: dict[str, Any] = {}
    for ph in PHASES:
        recs = [r for r in read_jsonl(exp / "raw" / f"requests_{ph}.jsonl") if not r.get("warmup")]
        base = exp / "raw" / "server_events" / ph
        services = {}
        for name in sorted(gpus):
            # A request belongs to the service whose server events carry its id (the request record only has a URL).
            sdir = base / name
            per = stage_breakdown([e for f in sorted(sdir.glob("*.jsonl")) for e in read_jsonl(f)])["per_request"] if sdir.exists() else {}
            mine = [r for r in recs if r["request_id"] in per]
            hop_t2k = [per[r["request_id"]].get("hop.thinker->talker") for r in mine]
            hop_k2c = [per[r["request_id"]].get("hop.talker->code2wav") for r in mine]
            levels = {}
            for lv in sorted({_level(r) for r in mine}):
                rs = [r for r in mine if _level(r) == lv]
                e2e = [request_metrics(r).get("e2e_s") for r in rs if r.get("status") == "ok"]
                levels[str(lv)] = {"n": len(rs), "ok": sum(r.get("status") == "ok" for r in rs), "e2e_p50_s": pct(e2e, 0.5),
                                   "e2e_p90_s": pct(e2e, 0.9), "slo": sum(map(slo_ok, rs)) / len(rs) if rs else None}
            services[name] = {"gpus": gpus[name], "link": link_of(name), "requests": len(mine), "hop_events": sum(x is not None for x in hop_t2k),
                              "hop_thinker_talker_p50_ms": pct(hop_t2k, 0.5), "hop_thinker_talker_p90_ms": pct(hop_t2k, 0.9),
                              "hop_talker_code2wav_p50_ms": pct(hop_k2c, 0.5), "levels": levels}
        phases[ph] = services
    return {"run": exp.name, "experiment_id": res["experiment_id"], "phases": phases}


def build(topology: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts = {}
    for ph in PHASES:
        hop: dict[str, list[float]] = {}
        e2e: dict[str, float] = {}
        for r in runs:
            for name, s in r["phases"][ph].items():
                if s["hop_thinker_talker_p50_ms"] is not None:
                    hop.setdefault(s["link"], []).append(s["hop_thinker_talker_p50_ms"])
                top = max(s["levels"], key=float)
                e2e[f"{name}#{r['experiment_id'].removeprefix('v2b_')}"] = s["levels"][top]["e2e_p50_s"]
        hop_by_link = {k: statistics.median(v) for k, v in hop.items()}
        verdicts[ph] = {**link_verdict(hop_by_link, e2e), "hop_by_link_p50_ms": hop_by_link,
                        "e2e_top_level_p50_s": e2e}
    both = [verdicts[ph]["verdict"] for ph in PHASES]
    overall = both[0] if all(v == both[0] for v in both) else "不确定"
    return {"topology": topology, "runs": runs, "verdicts": verdicts, "verdict_overall": overall}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--topology", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    summary = {"generated_by": "scripts/analysis/exp13_report.py", "rule": {"tolerance": TOL},
               **build(scan_topology(root / a.topology), [scan_run(root / r) for r in a.run])}
    out = Path(a.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "results" / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    for ph, v in summary["verdicts"].items():
        print(ph, v["verdict"], {k: round(x, 2) for k, x in v["hop_by_link_p50_ms"].items()}, "rel %.3f" % v["e2e_rel_non_nvlink"],
              "same-link %.3f" % (v["same_link_spread"] or 0), "hop share %.5f" % v["hop_spread_share_of_e2e"])
    print("overall", summary["verdict_overall"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
