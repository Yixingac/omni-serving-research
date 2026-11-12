"""Load drivers for baseline experiments: closed-loop concurrency sweeps and open-loop arrivals.

Every request is also recorded as a ``model_request`` span (one synthetic "benchmark session"
per worker), so benchmark traffic can be joined with server-side stage events by request_id
exactly like agent traffic.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any

from ..tracing.collector import TraceCollector
from ..tracing.schema import EV_COMPLETE, EV_FIRST_AUDIO, EV_FIRST_TEXT, EV_REQUEST_SENT, SpanKind, new_id
from .backends import Backend, ModelRequest, RequestResult
from .payloads import AssetPool, Workload, WorkloadMix, build_request


class JsonlSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self.records: list[dict[str, Any]] = []

    def write(self, rec: dict[str, Any]) -> None:
        self.records.append(rec)
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def trace_request(collector: TraceCollector | None, req: ModelRequest, res: RequestResult, session_id: str,
                  name: str = "benchmark_request", extra: dict[str, Any] | None = None) -> None:
    """Record a finished request as a model_request span with its streaming milestones."""
    if collector is None or res.sent_ns is None:
        return
    h = collector.start_span(
        SpanKind.MODEL_REQUEST,
        name,
        start_ns=res.sent_ns,
        session_id=session_id,
        request_id=req.request_id,
        attributes={
            "input_modalities": req.input_modalities,
            "output_modalities": req.output_modalities,
            "endpoint": "chat.completions",
            "max_tokens": req.max_tokens,
            "payload": req.metadata.get("payload"),
            **(extra or {}),
        },
    )
    h.event(EV_REQUEST_SENT, time_ns=res.sent_ns)
    if res.first_text_ns:
        h.event(EV_FIRST_TEXT, time_ns=res.first_text_ns)
    if res.first_audio_ns:
        h.event(EV_FIRST_AUDIO, time_ns=res.first_audio_ns)
    end = res.done_ns or collector.now()
    h.event(EV_COMPLETE, time_ns=end, status=res.status)
    status = {"ok": "ok", "timeout": "timeout", "cancelled": "cancelled"}.get(res.status, "error")
    h.end(status, end_ns=max(end, res.sent_ns), completion_tokens=res.completion_tokens,
          output_audio_s=res.output_audio_s, output_text_chars=res.text_chars, error=res.error)


def _record(req: ModelRequest, res: RequestResult, **extra: Any) -> dict[str, Any]:
    rec = res.to_dict()
    rec.update(extra)
    rec["input_modalities"] = req.input_modalities
    rec["output_modalities"] = req.output_modalities
    rec["payload"] = req.metadata.get("payload")
    rec["max_tokens"] = req.max_tokens
    return rec


async def closed_loop(
    backend: Backend,
    wl: Workload | WorkloadMix,
    pool: AssetPool | None,
    *,
    concurrency: int,
    requests: int,
    warmup: int,
    sink: JsonlSink,
    collector: TraceCollector | None = None,
    seed: int = 0,
    phase: str = "closed_loop",
    keep_text: bool = False,
) -> dict[str, Any]:
    """``concurrency`` workers send back-to-back requests until ``requests`` measured ones are done.

    The first ``warmup`` requests (sent with the same concurrency) are recorded with
    ``warmup=True`` and excluded from statistics by the analysis code.
    """
    rng = random.Random(seed)
    total = warmup + requests
    counter = {"next": 0}
    t_start = time.monotonic_ns()

    async def worker(wid: int) -> None:
        session = f"bench-{phase}-c{concurrency}-w{wid}-{new_id()[:8]}"
        while True:
            i = counter["next"]
            if i >= total:
                return
            counter["next"] = i + 1
            rid = f"{phase}-c{concurrency}-{i:05d}-{new_id()[:8]}"
            req_rng = random.Random(rng.random())
            w = wl.pick(req_rng) if isinstance(wl, WorkloadMix) else wl
            req = build_request(w, req_rng, rid, pool, metadata={"session_id": session})
            res = await backend.run(req)
            sink.write(_record(req, res, phase=phase, concurrency=concurrency, index=i, worker=wid,
                               warmup=i < warmup, workload_id=w.workload_id, text=res.text if keep_text else None))
            trace_request(collector, req, res, session, extra={"phase": phase, "concurrency": concurrency, "warmup": i < warmup})

    await asyncio.gather(*(worker(w) for w in range(concurrency)))
    return {"phase": phase, "concurrency": concurrency, "requests": requests, "warmup": warmup,
            "wall_s": (time.monotonic_ns() - t_start) / 1e9}


async def open_loop(
    backend: Backend,
    wl: Workload | WorkloadMix,
    pool: AssetPool | None,
    *,
    rate_rps: float,
    requests: int,
    sink: JsonlSink,
    collector: TraceCollector | None = None,
    seed: int = 0,
    phase: str = "open_loop",
) -> dict[str, Any]:
    """Poisson arrivals at ``rate_rps`` regardless of completions (offered load is fixed)."""
    rng = random.Random(seed)
    t0 = time.monotonic_ns()
    offsets = []
    t = 0.0
    for _ in range(requests):
        offsets.append(t)
        t += rng.expovariate(rate_rps)

    async def one(i: int, at_s: float) -> None:
        scheduled = t0 + int(at_s * 1e9)
        delay = (scheduled - time.monotonic_ns()) / 1e9
        if delay > 0:
            await asyncio.sleep(delay)
        rid = f"{phase}-r{rate_rps:g}-{i:05d}-{new_id()[:8]}"
        req_rng = random.Random(seed * 100003 + i)
        w = wl.pick(req_rng) if isinstance(wl, WorkloadMix) else wl
        req = build_request(w, req_rng, rid, pool, metadata={"session_id": f"{phase}-{i}"})
        res = await backend.run(req)
        sink.write(_record(req, res, phase=phase, rate_rps=rate_rps, index=i, scheduled_ns=scheduled,
                           lateness_s=((res.sent_ns or scheduled) - scheduled) / 1e9, workload_id=w.workload_id, warmup=False))
        trace_request(collector, req, res, f"{phase}-{i}", extra={"phase": phase, "rate_rps": rate_rps})

    await asyncio.gather(*(one(i, o) for i, o in enumerate(offsets)))
    return {"phase": phase, "rate_rps": rate_rps, "requests": requests, "wall_s": (time.monotonic_ns() - t0) / 1e9}
