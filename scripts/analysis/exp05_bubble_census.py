#!/usr/bin/env python3
"""实验 5 第二版：两批 8 卡 Job 里所有实验的 GPU 空闲时间，按原因拆开（bubble 普查）。

    python scripts/analysis/exp05_bubble_census.py --data-root ~/research-staging/unicorn \
        --out research/01_性能瓶颈/实验05_GPU为什么会等待

Every run of plan v2 (experiment ids v2_*, v2b_*, exp01_*, exp05_*) under <data-root>/experiments is read.
Per phase and load level (closed-loop concurrency, open-loop rate, or the whole phase):

- every GPU's sampled busy share (nvidia-smi utilization.gpu, every 200 ms; all runs);
- with server events (phases that recorded them): a 1 ms timeline of where each request is --
  CPU preprocessing, thinker queue / model path, the hop to the talker, talker queue / model path, the hop
  to code2wav, code2wav queue / batch, the tail to the last response -- and from it, per GPU:
  * work: a stage on this GPU has the request queued or in its model path / batch;
  * idle with work (estimated): work share minus nvidia-smi busy share over the level, when positive;
  * no work, by where the server's requests are at that moment, in this order:
    type 1  upstream GPU stage (a talker GPU while requests are still in the thinker);
    type 5  in a hop between stages (thinker -> talker, talker -> code2wav);
    type 3  in CPU preprocessing (video decoding, prompt building);
    imbalance  the same stage on another GPU has requests;
    type 2  only downstream of this GPU (closed loop: they hold the client's concurrency slots);
    type 7  no request in the server (for a talker GPU: no request that needs speech);
  * both_have_work: two or more of this GPU's stage processes have requests in their model path / batch
    at the same time -- an upper bound on contention, not contention itself (kernel traces measure that);
- each measured request's critical path (median per segment).

Writes results/bubble_census.json. Busy and idle-with-work are estimates (nvidia-smi samples a driver
window, not the exact 200 ms); the no-work split and the critical path are measured from server events.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp01_gpu_timeline import load_util  # noqa: E402
from omni_research.analysis.experiment_analysis import clock_anchor, read_jsonl  # noqa: E402
from omni_research.analysis.metrics import request_metrics  # noqa: E402

KINDS = ("thinker", "talker", "code2wav")
# Places a request passes through, in pipeline order. (kind, part) for GPU stages.
PLACES = ["pre", "th_queue", "th_svc", "hop1", "tk_queue", "tk_svc", "hop2", "cw_queue", "cw_svc", "tail"]
PLACE_KIND = {"th_queue": "thinker", "th_svc": "thinker", "tk_queue": "talker", "tk_svc": "talker", "cw_queue": "code2wav", "cw_svc": "code2wav"}
NOWORK = ["type1", "type5", "type3", "imbalance", "type2", "type7"]
WANTED = {
    "coordinator": {"request_admission", "terminal_response"},
    "thinker": {"scheduler_queue_enter", "scheduler_prefill_start", "scheduler_prefill_end", "model_path_end"},
    "talker": {"stage_input_received", "scheduler_queue_enter", "scheduler_prefill_start", "model_path_end"},
    "code2wav": {"stage_input_received", "stage_dispatch", "stage_complete"},
}
SKIP = ("stream_chunk", "coordinator_stream_received")
SEGMENTS = [  # critical path: (key, start event, end event)
    ("pre", ("coordinator", "request_admission"), ("thinker", "scheduler_queue_enter")),
    ("thinker_queue", ("thinker", "scheduler_queue_enter"), ("thinker", "scheduler_prefill_start")),
    ("thinker_prefill", ("thinker", "scheduler_prefill_start"), ("thinker", "scheduler_prefill_end")),
    ("thinker_decode", ("thinker", "scheduler_prefill_end"), ("thinker", "model_path_end")),
    ("hop1", ("thinker", "model_path_end"), ("talker", "stage_input_received")),
    ("talker_queue", ("talker", "stage_input_received"), ("talker", "scheduler_prefill_start")),
    ("talker_gen", ("talker", "scheduler_prefill_start"), ("talker", "model_path_end")),
    ("hop2", ("talker", "model_path_end"), ("code2wav", "stage_input_received")),
    ("code2wav_wait", ("code2wav", "stage_input_received"), ("code2wav", "stage_dispatch")),
    ("code2wav_run", ("code2wav", "stage_dispatch"), ("code2wav", "stage_complete")),
    ("tail", ("code2wav", "stage_complete"), ("coordinator", "terminal_response")),
]


# ------------------------------------------------------------------------------------------
# runs and layouts


def batch_of(eid: str) -> dict[str, str]:
    if eid.startswith("v2_"):
        return {"job": "625796", "node": "第一批节点（共享 gpu 分区的 8×A6000，驱动 570）"}
    return {"job": "454290", "node": "compute-03（8×A6000，驱动 580）"}


def family_of(eid: str) -> str:
    """Which question a run belongs to (used to group the census)."""
    for key, fam in (("_y0s_", "soak"), ("_aoh_", "W3b 开环过载"), ("_sb_", "W3b 低并发"), ("_b1_", "混合负载"),
                     ("_b2l_", "Agent 负载"), ("_b2_", "Agent 负载"), ("_d_", "回放"), ("_f1_", "视频"), ("_g1", "卡间连接"),
                     ("exp01_", "W3b kernel 补测"), ("exp05_", "W3b 并发 128")):
        if key in eid or eid.startswith(key):
            return fam
    return "W3b 扫描"


def arg_after(args: list[str], flag: str) -> str | None:
    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else None


def stage_gpus(cfg: dict[str, Any], repo: Path = ROOT) -> dict[str, dict[tuple[str, str], int]]:
    """Per server: (kind, replica) -> physical GPU index of the job.

    A ``--config`` YAML gives replica_devices per process (pipeline = thinker); otherwise the thinker is on
    the server's first GPU and ``--talker.gpu`` / ``--code2wav.gpu`` pick the others; ``--text-only`` has
    no talker or code2wav."""
    import yaml

    out: dict[str, dict[tuple[str, str], int]] = {}
    for s in cfg["servers"]:
        gpus = list(s["gpus"])
        args = [str(a) for a in s.get("args", [])]
        m: dict[tuple[str, str], int] = {}
        conf = arg_after(args, "--config")
        if conf:
            y = yaml.safe_load(open(conf.replace("{research_repo}", str(repo))))
            procs = y.get("processes") or {}
            for kind, proc in (("thinker", "pipeline"), ("talker", "talker"), ("code2wav", "code2wav")):
                devs = (procs.get(proc) or {}).get("replica_devices") or [0]
                for i, d in enumerate(devs):
                    m[(kind, f"r{i}")] = gpus[int(d)]
        else:
            m[("thinker", "r0")] = gpus[0]
            if "--text-only" not in args:
                tk = int(arg_after(args, "--talker.gpu") or 0)
                m[("talker", "r0")] = gpus[tk]
                m[("code2wav", "r0")] = gpus[int(arg_after(args, "--code2wav.gpu") or tk)]
        out[s["name"]] = m
    return out


def gpu_role(kinds: set[str]) -> str:
    if kinds >= set(KINDS):
        return "完整副本"
    if kinds == {"talker", "code2wav"}:
        return "talker 卡"
    if kinds == {"thinker"}:
        return "thinker 卡"
    return "+".join(sorted(kinds))


# ------------------------------------------------------------------------------------------
# events -> per-request timestamps


def stage_parts(stage: str) -> tuple[str, str]:
    """'thinker@r1' -> ('thinker', 'r1'); 'preprocessing' -> ('preprocessing', 'r0')."""
    kind, _, rep = stage.partition("@")
    return kind, rep or "r0"


def load_requests(exp: Path, phase: str) -> dict[str, dict[str, Any]]:
    """request id -> {'server', 'reps': {kind: replica}, 't': {(kind, event): ns}} (first time of each event,
    last time of model_path_end / stage_complete / terminal_response)."""
    base = exp / "raw" / "server_events" / phase
    reqs: dict[str, dict[str, Any]] = {}
    if not base.exists():
        return reqs
    last_wins = {"model_path_end", "stage_complete", "terminal_response"}
    for sdir in sorted(p for p in base.iterdir() if p.is_dir()):
        for fp in sorted(sdir.glob("*.jsonl")):
            with open(fp) as fh:
                for line in fh:
                    if any(k in line for k in SKIP):
                        continue
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    kind, rep = stage_parts(e.get("stage") or "")
                    name, rid, t = e.get("event_name"), e.get("request_id"), e.get("timestamp_ns")
                    if name not in WANTED.get(kind, ()) or rid is None or t is None:
                        continue
                    r = reqs.setdefault(rid, {"server": sdir.name, "reps": {}, "t": {}})
                    if kind in KINDS:
                        r["reps"][kind] = rep
                    k = (kind, name)
                    old = r["t"].get(k)
                    if old is None or (t > old if name in last_wins else t < old):
                        r["t"][k] = t
    return reqs


def place_intervals(t: dict[tuple[str, str], int]) -> dict[str, tuple[int, int]]:
    """Where a request is, as [start, end) in ns per place; places it never reached are left out."""
    g = t.get
    adm, term = g(("coordinator", "request_admission")), g(("coordinator", "terminal_response"))
    th_q, th_ps, th_end = g(("thinker", "scheduler_queue_enter")), g(("thinker", "scheduler_prefill_start")), g(("thinker", "model_path_end"))
    tk_in, tk_ps, tk_end = g(("talker", "stage_input_received")), g(("talker", "scheduler_prefill_start")), g(("talker", "model_path_end"))
    cw_in, cw_d, cw_end = g(("code2wav", "stage_input_received")), g(("code2wav", "stage_dispatch")), g(("code2wav", "stage_complete"))
    out: dict[str, tuple[int, int]] = {}

    def put(name: str, a: int | None, b: int | None) -> None:
        if a is not None and b is not None and b > a:
            out[name] = (a, b)

    put("pre", adm, th_q)
    put("th_queue", th_q, th_ps)
    put("th_svc", th_ps, th_end)
    if tk_in is not None:
        put("hop1", th_end, tk_in)
        put("tk_queue", tk_in, tk_ps)
        put("tk_svc", tk_ps, tk_end)
        put("hop2", tk_end, cw_in)
        put("cw_queue", cw_in, cw_d)
        put("cw_svc", cw_d, cw_end)
        put("tail", cw_end, term)
    else:  # text only: the thinker's answer goes straight back
        put("tail", th_end, term)
    return out


def segments(t: dict[tuple[str, str], int]) -> dict[str, float]:
    """Critical-path segments of one request, in ms."""
    out = {}
    for key, a, b in SEGMENTS:
        if a in t and b in t and t[b] >= t[a]:
            out[key] = (t[b] - t[a]) / 1e6
    if ("coordinator", "request_admission") in t and ("coordinator", "terminal_response") in t:
        out["total"] = (t[("coordinator", "terminal_response")] - t[("coordinator", "request_admission")]) / 1e6
    return out


# ------------------------------------------------------------------------------------------
# 1 ms timelines


class Grid:
    """Counts of requests per (server, place) and per (server, kind, replica, queue|svc) on a 1 ms grid."""

    def __init__(self, t0_ns: int, t1_ns: int):
        self.t0 = t0_ns
        self.n = int((t1_ns - t0_ns) // 1_000_000) + 2
        self.diff: dict[tuple, np.ndarray] = {}

    def add(self, key: tuple, a_ns: int, b_ns: int) -> None:
        d = self.diff.get(key)
        if d is None:
            d = self.diff[key] = np.zeros(self.n + 1, dtype=np.int32)
        i = min(max(int((a_ns - self.t0) // 1_000_000), 0), self.n)
        j = min(max(int((b_ns - self.t0) // 1_000_000), 0), self.n)
        if j > i:
            d[i] += 1
            d[j] -= 1

    def counts(self, key: tuple) -> np.ndarray:
        d = self.diff.get(key)
        return np.zeros(self.n, dtype=np.int32) if d is None else np.cumsum(d[:-1])

    def idx(self, t_s: float) -> int:
        return min(max(int((t_s * 1e9 - self.t0) // 1_000_000), 0), self.n)


def classify_gpu(grid: Grid, server: str, on_gpu: list[tuple[str, str]], same_kind_elsewhere: list[tuple[str, str]],
                 places_present: dict[str, np.ndarray], speech_present: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Boolean 1 ms series for one GPU: work, svc per kind, and the no-work categories (exclusive).

    For a GPU whose first stage is the talker, only requests that go on to the talker count: a text-only
    answer never reaches it, so a talker GPU idling during text requests has no work coming (type 7),
    not a slow upstream."""
    kinds = {k for k, _ in on_gpu}
    svc = {k: np.zeros(grid.n, dtype=bool) for k in kinds}
    work = np.zeros(grid.n, dtype=bool)
    for kind, rep in on_gpu:
        s = grid.counts((server, kind, rep, "svc")) > 0
        q = grid.counts((server, kind, rep, "queue")) > 0
        svc[kind] |= s
        work |= s | q
    order = [p for p in PLACES]
    first = min(order.index(p) for p, k in PLACE_KIND.items() if k in kinds)
    last = max(order.index(p) for p, k in PLACE_KIND.items() if k in kinds)
    upstream = speech_present if first >= order.index("tk_queue") else places_present  # also used downstream
    up_gpu = np.zeros(grid.n, dtype=bool)
    hop = np.zeros(grid.n, dtype=bool)
    cpu = upstream["pre"].copy()
    down = np.zeros(grid.n, dtype=bool)
    for i, p in enumerate(order):
        if p == "pre":
            continue
        if p.startswith("hop") and i < last:
            hop |= upstream[p]
        elif i < first and p in PLACE_KIND:
            up_gpu |= upstream[p]
        elif i > last:
            down |= upstream[p]
    other = np.zeros(grid.n, dtype=bool)
    for kind, rep in same_kind_elsewhere:
        other |= (grid.counts((server, kind, rep, "svc")) > 0) | (grid.counts((server, kind, rep, "queue")) > 0)
    free = ~work
    out = {"work": work}
    for name, cond in (("type1", up_gpu), ("type5", hop), ("type3", cpu), ("imbalance", other), ("type2", down)):
        out[name] = free & cond
        free = free & ~cond
    out["type7"] = free
    nsvc = sum(v.astype(np.int8) for v in svc.values())
    out["both_have_work"] = nsvc >= 2 if len(svc) > 1 else np.zeros(grid.n, dtype=bool)
    for k, v in svc.items():
        out[f"svc_{k}"] = v
    return out


# ------------------------------------------------------------------------------------------
# one run


def level_windows(exp: Path, phase: str, anchor: tuple[int, int] | None) -> list[dict[str, Any]]:
    """Client requests grouped by load level (or the whole phase), with the level's wall-clock window
    (warmup requests included: they occupy the GPUs) and latency / throughput of the measured ones."""
    recs = read_jsonl(exp / "raw" / f"requests_{phase}.jsonl")
    if not recs or anchor is None:
        return []
    mono0, wall0 = anchor
    m = re.search(r"_c(\d+)$", phase)
    by: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        lv = r.get("concurrency") if r.get("concurrency") is not None else r.get("rate_rps")
        kind = "closed" if r.get("concurrency") is not None else "open" if r.get("rate_rps") is not None else "phase"
        if lv is None and m:
            lv, kind = int(m.group(1)), "closed"
        by[(kind, lv)].append(r)
    out = []
    for (kind, lv), rs in sorted(by.items(), key=lambda x: (x[0][0], x[0][1] if x[0][1] is not None else -1)):
        ts = [(r["sent_ns"] - mono0 + wall0) / 1e9 for r in rs if r.get("sent_ns")]
        te = [(r["done_ns"] - mono0 + wall0) / 1e9 for r in rs if r.get("done_ns")]
        if not ts or not te:
            continue
        meas = [r for r in rs if not r.get("warmup") and r.get("status") == "ok" and r.get("sent_ns") and r.get("done_ns")]
        mets = [request_metrics(r) for r in meas]
        e2e = sorted(x["e2e_s"] for x in mets if x.get("e2e_s") is not None)
        ttft = sorted(x["ttft_s"] for x in mets if x.get("ttft_s") is not None)
        span = (max(r["done_ns"] for r in meas) - min(r["sent_ns"] for r in meas)) / 1e9 if meas else None
        out.append({"kind": kind, "level": lv, "t0": min(ts), "t1": max(te), "n": len(meas),
                     "errors": sum(1 for r in rs if r.get("status") != "ok"),
                     "speech_share": sum(1 for r in meas if (r.get("output_audio_s") or 0) > 0) / len(meas) if meas else None,
                     "tp_rps": len(meas) / span if span else None,
                     "e2e_p50": e2e[len(e2e) // 2] if e2e else None, "e2e_p90": e2e[int(len(e2e) * 0.9)] if e2e else None,
                     "ttft_p50": ttft[len(ttft) // 2] if ttft else None,
                     "rids": {r.get("request_id") for r in rs}, "measured": {r.get("request_id"): r for r in meas}})
    return out


def analyze_run(exp: Path) -> dict[str, Any] | None:
    try:
        res = json.load(open(exp / "result.json"))
        cfg = json.load(open(exp / "config.json"))
    except (OSError, json.JSONDecodeError):
        return None
    eid = res["experiment_id"]
    layout = stage_gpus(cfg)
    gpu_kinds: dict[int, set[str]] = defaultdict(set)
    for m in layout.values():
        for (kind, _), g in m.items():
            gpu_kinds[g].add(kind)
    util = load_util(exp)
    anchor = clock_anchor(exp)
    # A phase that records server events only inside a short profiler window has partial request
    # timelines; its kernel-level analysis (exp05_talker.py) covers it instead.
    windowed = {p["name"] for p in cfg.get("phases", []) if p.get("torch_profile_window")}
    run: dict[str, Any] = {"run": exp.name, "experiment_id": eid, "family": family_of(eid), **batch_of(eid),
                           "gpus_used": len(gpu_kinds), "roles": {g: gpu_role(k) for g, k in sorted(gpu_kinds.items())},
                           "phases": {}}
    for ph in res["phases"]:
        phase = ph["phase"]
        if phase.startswith("smoke"):
            continue
        wins = level_windows(exp, phase, anchor)
        if not wins:
            continue
        reqs = {} if phase in windowed else load_requests(exp, phase)
        has_events = bool(reqs)
        grid = series = None
        if has_events:
            t_lo = min(min(r["t"].values()) for r in reqs.values())
            t_hi = max(max(r["t"].values()) for r in reqs.values())
            grid = Grid(min(t_lo, int(min(w["t0"] for w in wins) * 1e9)), max(t_hi, int(max(w["t1"] for w in wins) * 1e9)))
            for rid, r in reqs.items():
                s = r["server"]
                speech = ("talker", "stage_input_received") in r["t"]
                for place, (a, b) in place_intervals(r["t"]).items():
                    grid.add((s, place), a, b)
                    if speech:
                        grid.add((s, place, "speech"), a, b)
                    kind = PLACE_KIND.get(place)
                    if kind:
                        grid.add((s, kind, r["reps"].get(kind, "r0"), "svc" if place.endswith("svc") else "queue"), a, b)
            series = {}
            for s, m in layout.items():
                present = {p: grid.counts((s, p)) > 0 for p in PLACES}
                speech = {p: grid.counts((s, p, "speech")) > 0 for p in PLACES}
                for g in sorted({g for g in m.values()}):
                    on = [k for k, gg in m.items() if gg == g]
                    kinds_here = {k for k, _ in on}
                    elsewhere = [k for k, gg in m.items() if gg != g and k[0] in kinds_here]
                    series[g] = classify_gpu(grid, s, on, elsewhere, present, speech)
        levels = []
        for w in wins:
            row = {k: v for k, v in w.items() if k not in ("rids", "measured")}
            row["window_s"] = w["t1"] - w["t0"]
            gpus: dict[int, dict[str, float]] = {}
            for g in sorted(gpu_kinds):
                smp = [u / 100.0 for t, u, _ in util.get(g, []) if w["t0"] <= t <= w["t1"]]
                d: dict[str, float] = {"busy": float(np.mean(smp)) if smp else float("nan")}
                if series is not None and g in series:
                    sr = series[g]
                    i0, i1 = grid.idx(w["t0"]), grid.idx(w["t1"])
                    if i1 > i0:
                        for k, v in sr.items():
                            d[k] = float(v[i0:i1].mean())
                        if smp:
                            # Window averages, not per sample: nvidia-smi's driver window does not line up
                            # with the 200 ms query interval, and per-sample matching overstated idle-with-work
                            # at low load (checked against the kernel traces, see exp05_step_sync.py).
                            d["idle_with_work"] = max(0.0, d["work"] - d["busy"])
                            d["busy_beyond_work"] = max(0.0, d["busy"] - d["work"])
                gpus[g] = d
            row["gpus"] = gpus
            if has_events:
                segs: dict[str, list[float]] = defaultdict(list)
                for rid, rec in w["measured"].items():
                    r = reqs.get(rid)
                    if not r:
                        continue
                    sg = segments(r["t"])
                    for k, v in sg.items():
                        segs[k].append(v)
                    # Per output token / audio second: answer lengths differ between levels.
                    toks, audio = rec.get("completion_tokens"), rec.get("output_audio_s")
                    if "thinker_decode" in sg and toks and toks > 1:
                        segs["thinker_ms_per_token"].append(sg["thinker_decode"] / (toks - 1))
                    if audio:
                        for k in ("talker_gen", "code2wav_run"):
                            if k in sg:
                                segs[f"{k}_ms_per_audio_s"].append(sg[k] / audio)
                row["path_ms_p50"] = {k: float(np.median(v)) for k, v in segs.items()}
                row["path_ms_p90"] = {k: float(np.percentile(v, 90)) for k, v in segs.items()}
                row["path_n"] = len(segs.get("total", []))
            levels.append(row)
        run["phases"][phase] = {"events": has_events, "profiled": "prof" in phase or "torch_profile" in phase, "levels": levels}
    return run


# ------------------------------------------------------------------------------------------
# role summaries


def role_summary(run: dict[str, Any], row: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Mean of every per-GPU number over the GPUs of each role."""
    by: dict[str, list[dict[str, float]]] = defaultdict(list)
    for g, d in row["gpus"].items():
        by[run["roles"][g]].append(d)
    out = {}
    for role, ds in by.items():
        keys = set().union(*ds)
        out[role] = {k: float(np.nanmean([d[k] for d in ds if k in d])) for k in keys}
        out[role]["gpus"] = len(ds)
    return out


def rounded(o: Any, digits: int = 4) -> Any:
    """Floats rounded for the summary file (shares to 0.01%, times to 0.1 us or better)."""
    if isinstance(o, float):
        return round(o, digits) if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: rounded(v, digits) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [rounded(v, digits) for v in o]
    return o


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="regex on the run directory name (for a quick look)")
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    out = Path(a.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    runs = {}
    for exp in sorted(root.iterdir()):
        if not re.match(r"2026111[5-7]_(v2_|v2b_|exp0)", exp.name) or "_y0_env" in exp.name:
            continue
        if a.only and not re.search(a.only, exp.name):
            continue
        r = analyze_run(exp)
        if r is None:
            continue
        for ph in r["phases"].values():
            for row in ph["levels"]:
                row["roles"] = role_summary(r, row)
        runs[r["experiment_id"]] = r
        print(f"{exp.name}: {sum(len(p['levels']) for p in r['phases'].values())} levels", file=sys.stderr)
    doc = {"generated_by": "scripts/analysis/exp05_bubble_census.py", "runs": runs}
    (out / "results" / "bubble_census.json").write_text(json.dumps(rounded(doc), ensure_ascii=False, indent=0, default=float) + "\n")
    print(out / "results" / "bubble_census.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
