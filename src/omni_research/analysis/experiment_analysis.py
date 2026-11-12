"""Turn one experiment directory into tables, figures and a Chinese report.

Inputs (written by orchestration.experiment): raw/requests_*.jsonl, raw/gpu_samples.csv,
raw/cpu_samples.jsonl, raw/server_events/<phase>/**.jsonl, raw/*trace*.jsonl, result.json.

Clock handling: client records carry ``time.monotonic_ns`` of the client process; the
benchmark/agent trace headers in the same directory hold that process's monotonic->wall anchor,
which is used to place client times on the wall clock that SGLang-Omni stamps on its events
(``time.time_ns`` on the same host). nvidia-smi timestamps are local wall-clock strings.
"""

from __future__ import annotations

import csv
import json
import re
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import intervals as iv
from . import metrics as M

# Per-stage intervals computed for every stage that emits the events (no model-specific names):
#   queue        scheduler_queue_enter   -> scheduler_prefill_start
#   prefill      scheduler_prefill_start -> scheduler_prefill_end
#   first_emit   scheduler_prefill_start -> scheduler_first_emit
#   model_path   model_path_start        -> model_path_end      (the stage's model execution, all steps)
#   service      stage_dispatch          -> stage_complete      (whole stage, including waiting inside it)
#   preprocess / encoder from their own start/end events.
# Hops between stages: destination stage_input_received minus the latest source stage_hop_sent
# before it. Only events present in the data produce numbers.
STAGE_EVENT_PAIRS = [
    ("queue", "scheduler_queue_enter", "scheduler_prefill_start"),
    ("prefill", "scheduler_prefill_start", "scheduler_prefill_end"),
    ("first_emit", "scheduler_prefill_start", "scheduler_first_emit"),
    ("model_path", "model_path_start", "model_path_end"),
    ("service", "stage_dispatch", "stage_complete"),
    ("preprocess", "preprocess_start", "preprocess_end"),
    ("encoder", "encoder_start", "encoder_end"),
]


# ------------------------------------------------------------------------------------------
# Loading


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def load_requests(exp: Path) -> list[dict[str, Any]]:
    recs = []
    for f in sorted((exp / "raw").glob("requests_*.jsonl")):
        for r in read_jsonl(f):
            # Each phase writes its own requests_<phase>.jsonl, so the file name is the reliable phase
            # label (agent records carry a generic "agent" label in their own phase field).
            r["phase"] = f.stem.removeprefix("requests_")
            recs.append(r)
    return recs


def clock_anchor(exp: Path) -> tuple[int, int] | None:
    """(mono_anchor_ns, wall_anchor_ns) of the client process, from any trace header."""
    for f in sorted((exp / "raw").glob("*trace*.jsonl")):
        with open(f) as fh:
            first = fh.readline()
        try:
            h = json.loads(first)
            c = h["clock"]
            return int(c["mono_anchor_ns"]), int(c["wall_anchor_ns"])
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return None


def load_gpu_samples(exp: Path) -> list[dict[str, Any]]:
    p = exp / "raw" / "gpu_samples.csv"
    if not p.exists():
        return []
    rows = []
    with open(p) as fh:
        reader = csv.reader(fh)
        header = [h.strip() for h in next(reader, [])]
        for row in reader:
            if len(row) != len(header):
                continue
            d = dict(zip(header, (x.strip() for x in row)))
            try:
                t = datetime.strptime(d["timestamp"], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                util = float(d["utilization.gpu"])
            except (KeyError, ValueError):
                continue
            # GB10 (unified memory) reports memory as [N/A]; keep the utilization sample anyway.
            rows.append({"t": t, "gpu": d["index"], "util": util, "mem_mib": _f(d.get("memory.used")),
                         "power_w": _f(d.get("power.draw"))})
    return rows


def _f(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_server_events(exp: Path, phase: str) -> list[dict[str, Any]]:
    base = exp / "raw" / "server_events" / phase
    evs: list[dict[str, Any]] = []
    for f in sorted(base.rglob("*.jsonl")) if base.exists() else []:
        evs.extend(read_jsonl(f))
    return evs


# ------------------------------------------------------------------------------------------
# Tables


def _level_key(r: dict[str, Any]) -> tuple[str, str]:
    for k in ("concurrency", "rate_rps"):
        if r.get(k) is not None:
            return k, str(r[k])
    if r.get("mode"):
        return "replay", f"{r['mode']}"
    return "all", "-"


def phase_level_table(records: list[dict[str, Any]], slo: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """One row per (phase, load level): sample sizes, errors, latency summaries, throughput."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r.get("warmup") or r.get("crash_affected") or r.get("kind") not in (None, "model_request"):
            continue
        kname, kval = _level_key(r)
        groups[(r.get("phase", "?"), kname, kval)].append(r)
    rows = []
    for (phase, kname, kval), recs in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], _num(x[0][2]))):
        ok = [r for r in recs if r.get("status") == "ok"]
        ms = [M.request_metrics(r) for r in ok]
        row = {"phase": phase, "level_kind": kname, "level": kval, "n": len(recs), "ok": len(ok),
               "errors": len(recs) - len(ok)}
        for metric in ("e2e_s", "ttft_s", "ttfa_s", "audio_rtf", "e2e_per_audio_s", "e2e_per_token_s"):
            s = M.summarize([m.get(metric) for m in ms], ci=False)
            row[f"{metric}_median"] = s.get("median")
            row[f"{metric}_p90"] = s.get("p90")
            row[f"{metric}_p95"] = s.get("p95")
            row[f"{metric}_mean"] = s.get("mean")
            row[f"{metric}_n"] = s.get("n")
        tok = [r.get("completion_tokens") for r in ok if r.get("completion_tokens") is not None]
        row["output_tokens_median"] = M.summarize(tok, ci=False).get("median") if tok else None
        aud = [r.get("output_audio_s") for r in ok if r.get("output_audio_s")]
        row["output_audio_s_median"] = M.summarize(aud, ci=False).get("median") if aud else None
        row.update({f"tp_{k}": v for k, v in M.throughput(recs).items()})
        if slo:
            g = M.goodput(recs, slo)
            row["goodput_fraction"] = g["fraction"]
            row["goodput_req_per_s"] = g["goodput_req_per_s"]
        rows.append(row)
    return rows


def _num(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return float("inf")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in keys})


def _fmt(v: Any) -> Any:
    return round(v, 4) if isinstance(v, float) else v


# ------------------------------------------------------------------------------------------
# GPU samples by phase


def gpu_by_phase(samples: list[dict[str, Any]], phases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for ph in phases:
        t0, t1 = ph.get("started"), ph.get("ended")
        if not (t0 and t1):
            continue
        per: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for s in samples:
            if t0 <= s["t"] <= t1:
                per[s["gpu"]].append(s)
        for gpu, ss in sorted(per.items()):
            utils = [s["util"] for s in ss]
            out.append({"phase": ph.get("phase"), "gpu": gpu, "samples": len(ss),
                        "sampled_util_mean": sum(utils) / len(utils) if utils else None,
                        "sampled_util_zero_fraction": sum(1 for u in utils if u == 0) / len(utils) if utils else None,
                        "mem_max_mib": max((s["mem_mib"] for s in ss if s["mem_mib"] is not None), default=None)})
    return out


# ------------------------------------------------------------------------------------------
# Server-side stage breakdown


def stage_breakdown(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-request server-side intervals (ms) for every stage, hops between stages, and totals."""
    first: dict[tuple[str, str, str], int] = {}
    last: dict[tuple[str, str, str], int] = {}
    hops_sent: dict[tuple[str, str], list[int]] = defaultdict(list)
    for e in events:
        rid, st, name, t = e.get("request_id"), e.get("stage"), e.get("event_name"), e.get("timestamp_ns")
        if t is None or not rid:
            continue
        k = (rid, st, name)
        first[k] = min(first.get(k, t), t)
        last[k] = max(last.get(k, t), t)
        if name == "stage_hop_sent":
            hops_sent[(rid, st)].append(t)
    by_req: dict[str, dict[tuple[str, str], tuple[int, int]]] = defaultdict(dict)
    for (rid, st, name), t in first.items():
        by_req[rid][(st, name)] = (t, last[(rid, st, name)])
    per_request: dict[str, dict[str, float]] = {}
    for rid, evs in by_req.items():
        d: dict[str, float] = {}
        stages = {st for st, _ in evs}
        for st in stages:
            for label, a, b in STAGE_EVENT_PAIRS:
                if (st, a) in evs and (st, b) in evs:
                    ta, tb = evs[(st, a)][0], evs[(st, b)][1] if b in ("stage_complete", "model_path_end") else evs[(st, b)][0]
                    if tb >= ta:
                        d[f"{st}.{label}"] = (tb - ta) / 1e6
        # Hops: for each stage's first input, the latest hop sent by another stage before it.
        for st in stages:
            if (st, "stage_input_received") not in evs:
                continue
            t_in = evs[(st, "stage_input_received")][0]
            best = None
            for (r2, src), ts in hops_sent.items():
                if r2 != rid or src == st:
                    continue
                cands = [t for t in ts if t <= t_in]
                if cands and (best is None or max(cands) > best[1]):
                    best = (src, max(cands))
            if best:
                d[f"hop.{best[0]}->{st}"] = (t_in - best[1]) / 1e6
        adm = [v[0] for (st, n), v in evs.items() if n == "request_admission"]
        term = [v for (st, n), v in evs.items() if n == "terminal_response"]
        if adm and term:
            d["request.admission_to_first_terminal"] = (min(v[0] for v in term) - min(adm)) / 1e6
            d["request.admission_to_last_terminal"] = (max(v[1] for v in term) - min(adm)) / 1e6
        audio = [v[0] for (st, n), v in evs.items() if n == "code2wav_first_audio"]
        if adm and audio:
            d["request.admission_to_first_audio"] = (min(audio) - min(adm)) / 1e6
        per_request[rid] = d
    labels = sorted({k for d in per_request.values() for k in d})
    summary = {lab: M.summarize([d.get(lab) for d in per_request.values()], ci=False) for lab in labels}
    return {"requests": len(per_request), "per_request": per_request, "summary_ms": summary}


# ------------------------------------------------------------------------------------------
# Prefix cache (SGLang "Prefill batch" log lines)

_PREFILL_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?:,(\d{3}))? .*?Prefill batch, #new-seq: (\d+), "
                         r"#new-token: (\d+), #cached-token: (\d+)")
_ISO_OFFSET_RE = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?([+-]\d\d):?(\d\d)")


def _offset_from_text(z: str) -> timedelta | None:
    m = re.fullmatch(r"([+-])(\d\d):?(\d\d)", z.strip())
    if not m:
        return None
    return timedelta(minutes=(1 if m.group(1) == "+" else -1) * (int(m.group(2)) * 60 + int(m.group(3))))


def log_utc_offset(exp: Path) -> timedelta | None:
    """UTC offset of the host that ran the experiment. Server logs carry local wall-clock times
    without a zone. Newer runs record the offset in the manifest; Slurm runs have ``date -Is`` output
    in the step log. Otherwise the server log's first line (local time at server start) is compared
    with the manifest's start time, rounded to a quarter hour; that works when the server started
    within a few minutes of the runner."""
    mf = exp / "manifest.json"
    manifest = json.loads(mf.read_text()) if mf.exists() else {}
    if manifest.get("utc_offset"):
        off = _offset_from_text(str(manifest["utc_offset"]))
        if off is not None:
            return off
    for name in ("step.log", "srun.log", "logs/client.log"):
        f = exp / name
        if f.exists():
            m = _ISO_OFFSET_RE.search(f.read_text(errors="replace")[:20000])
            if m:
                sign = -1 if m.group(1).startswith("-") else 1
                return timedelta(minutes=sign * (abs(int(m.group(1))) * 60 + int(m.group(2))))
    started = manifest.get("started")
    logs = sorted((exp / "logs").glob("server_*.log")) if (exp / "logs").exists() else []
    if started and logs:
        m = re.match(r"# (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)", logs[0].read_text(errors="replace")[:200])
        if m:
            naive = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC).timestamp()
            diff = naive - float(started)
            quarter = round(diff / 900) * 900
            if abs(diff - quarter) < 600:
                return timedelta(seconds=quarter)
    return None


def prefix_cache_by_phase(exp: Path, phases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Prompt tokens computed vs served from SGLang's prefix cache, per phase.

    Sums every SGLang engine stage that logs prefill batches into the server logs (for text-output
    phases that is the thinker alone; with speech output the talker's prefills are included).
    Returns {} when the time zone of the logs cannot be determined.
    """
    offset = log_utc_offset(exp)
    logs = sorted((exp / "logs").glob("server_*.log")) if (exp / "logs").exists() else []
    if offset is None or not logs or not phases:
        return {}
    tz = timezone(offset)
    windows = [(ph.get("phase", ""), ph.get("started"), ph.get("ended")) for ph in phases]
    out: dict[str, dict[str, Any]] = {}
    for log in logs:
        for line in log.read_text(errors="replace").splitlines():
            m = _PREFILL_RE.match(line)
            if not m:
                continue
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz).timestamp() + int(m.group(2) or 0) / 1000
            for name, t0, t1 in windows:
                if t0 is not None and t1 is not None and t0 <= t <= t1:
                    d = out.setdefault(name, {"prefill_batches": 0, "new_seqs": 0, "new_tokens": 0, "cached_tokens": 0})
                    d["prefill_batches"] += 1
                    d["new_seqs"] += int(m.group(3))
                    d["new_tokens"] += int(m.group(4))
                    d["cached_tokens"] += int(m.group(5))
                    break
    for d in out.values():
        tot = d["new_tokens"] + d["cached_tokens"]
        d["cached_fraction"] = d["cached_tokens"] / tot if tot else None
    return out


# Server-side failures that do not stop the runner (the stage process dies, the server exits, and
# a runner without a liveness check records the run as ok; f2, x7c and f5 on 2026-11-11).
_FAILURE_PATTERNS = (
    ("illegal_memory_access", re.compile(r"illegal memory access")),
    ("dead_stage_process", re.compile(r"Dead stage process\(es\) detected: ([^\n]+)")),
    ("cuda_oom", re.compile(r"CUDA out of memory|OutOfMemoryError")),
    ("cufft_error", re.compile(r"cuFFT error: \w+")),
)
_LOG_TIME_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def server_failures(exp: Path, phases: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """First occurrence of each failure pattern in the server logs, with the phase it fell in."""
    logs = sorted((exp / "logs").glob("server_*.log")) if (exp / "logs").exists() else []
    offset = log_utc_offset(exp)
    out: list[dict[str, Any]] = []
    for log in logs:
        last_t = None
        seen: set[str] = set()
        for i, line in enumerate(log.read_text(errors="replace").splitlines(), 1):
            m = _LOG_TIME_RE.match(line)
            if m:
                last_t = m.group(1)
            for kind, pat in _FAILURE_PATTERNS:
                if kind in seen:
                    continue
                hit = pat.search(line)
                if not hit:
                    continue
                seen.add(kind)
                d: dict[str, Any] = {"log": log.name, "line": i, "kind": kind, "text": line.strip()[:300], "local_time": last_t}
                if last_t and offset is not None:
                    t = datetime.strptime(last_t, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone(offset)).timestamp()
                    d["time"] = t
                    for ph in phases or []:
                        if ph.get("started") is not None and ph.get("ended") is not None and ph["started"] <= t <= ph["ended"]:
                            d["phase"] = ph.get("phase")
                            break
                out.append(d)
    return out


# ------------------------------------------------------------------------------------------
# Agent traces


def agent_trace_summary(path: Path) -> dict[str, Any]:
    from ..tracing.io import load_trace
    from ..tracing.validate import validate_trace

    tr = load_trace(path)
    rep = validate_trace(tr)
    reqs = tr.by_kind("model_request")
    tools = tr.by_kind("tool_call")
    waits = tr.by_kind("wait")
    sessions = tr.by_kind("session")
    dur = lambda s: (s.end_ns - s.start_ns) / 1e9 if s.end_ns else None  # noqa: E731
    # Within each session: how much of the wall time was spent waiting on the model, on tools,
    # on (synthetic) user think time, and in agent-side gaps between those.
    split = {"model": 0.0, "tool": 0.0, "think": 0.0, "other": 0.0, "session": 0.0}
    for s in sessions:
        if s.end_ns is None:
            continue
        win = (s.start_ns, s.end_ns)
        mine = lambda kind: [(x.start_ns, x.end_ns) for x in tr.by_kind(kind) if x.session_id == s.session_id and x.end_ns]  # noqa: E731
        model = iv.clip(mine("model_request"), win)
        tool = iv.subtract(iv.clip(mine("tool_call"), win), model)
        think = iv.subtract(iv.clip(mine("wait"), win), model + tool)
        total = win[1] - win[0]
        split["session"] += total / 1e9
        split["model"] += iv.total(model) / 1e9
        split["tool"] += iv.total(tool) / 1e9
        split["think"] += iv.total(think) / 1e9
        split["other"] += (total - iv.total(model + tool + think)) / 1e9
    succ = [s.attributes.get("success") for s in sessions]
    return {
        "valid": rep.ok,
        "validation_errors": rep.errors[:10],
        "source": tr.header.source,
        "sessions": len(sessions),
        "task_success": sum(1 for x in succ if x),
        "model_requests": len(reqs),
        "tool_calls": len(tools),
        "waits": len(waits),
        "requests_per_session": M.summarize([sum(1 for r in reqs if r.session_id == s.session_id) for s in sessions], ci=False),
        "model_request_s": M.summarize([dur(r) for r in reqs], ci=False),
        "tool_call_s": M.summarize([dur(t) for t in tools], ci=False),
        "session_s": M.summarize([dur(s) for s in sessions], ci=False),
        "time_split_s": split,
    }


# ------------------------------------------------------------------------------------------
# Whole experiment


def analyze_experiment(exp: Path, slo: dict[str, float] | None = None) -> dict[str, Any]:
    exp = Path(exp)
    result = json.loads((exp / "result.json").read_text()) if (exp / "result.json").exists() else {}
    config = json.loads((exp / "config.json").read_text()) if (exp / "config.json").exists() else {}
    manifest = json.loads((exp / "manifest.json").read_text()) if (exp / "manifest.json").exists() else {}
    records = load_requests(exp)
    table = phase_level_table(records, slo)
    samples = load_gpu_samples(exp)
    phases = result.get("phases", [])
    gpu_rows = gpu_by_phase(samples, phases)
    stages = {}
    for ph in phases:
        evs = load_server_events(exp, ph.get("phase", ""))
        if evs:
            stages[ph["phase"]] = stage_breakdown(evs)
    prefix_cache = prefix_cache_by_phase(exp, phases)
    failures = server_failures(exp, phases)
    agents = {}
    for f in sorted((exp / "raw").glob("agent_trace_*.jsonl")):
        agents[f.stem.removeprefix("agent_trace_")] = agent_trace_summary(f)
    gpu_traces = {}
    torch_root = exp / "profiling" / "torch"
    if torch_root.exists():
        from .torch_trace import analyze_files

        for phase_dir in sorted(p for p in torch_root.iterdir() if p.is_dir()):
            files = sorted(phase_dir.rglob("*.trace.json*"))
            if files:
                try:
                    r = analyze_files(files)
                    gpu_traces[phase_dir.name] = {"files": r["files"], "per_device": {
                        str(dev): {k: v for k, v in d.items() if k != "window_ns"} | {"window_s": (d["window_ns"][1] - d["window_ns"][0]) / 1e9}
                        for dev, d in r["per_device"].items()}}
                except (OSError, ValueError, KeyError) as exc:
                    gpu_traces[phase_dir.name] = {"error": str(exc)}
    return {"experiment": exp.name, "status": result.get("status"), "error_class": result.get("error_class"),
            "error": (result.get("error") or "")[:1500], "config": config, "manifest": manifest, "result": result,
            "table": table, "gpu_by_phase": gpu_rows, "stages": stages, "agents": agents, "gpu_traces": gpu_traces,
            "prefix_cache": prefix_cache, "server_failures": failures,
            "fake_server": bool(config.get("fake_server")), "analyzed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
