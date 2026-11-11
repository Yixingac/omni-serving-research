"""Trace replay.

Mode A, ``request`` (open loop): every model request of the trace is sent at its recorded
arrival offset divided by ``speed``, regardless of whether earlier requests have finished.
Use it to hold the offered load fixed.

Mode B, ``dependency`` (closed loop): a node (model request, tool call or wait) starts only after
all of its dependencies finished, plus the client-side gap that separated them in the original
trace (agent processing, tool time, user think time), divided by ``speed``. Model requests go to
the real backend; tool calls and waits are reproduced by sleeping for their recorded duration
(``tool_mode="sleep"``). The serving system's latency therefore feeds back into when the next
request is sent, as it does for a real agent.

Both modes can clone sessions (``session_copies``) and re-time session starts
(``session_arrival``) to scale concurrency beyond the original trace. Every replayed request
is written to a results JSONL with the original span id, so the replay is auditable.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..tracing.collector import TraceCollector
from ..tracing.io import Trace
from ..tracing.schema import EV_COMPLETE, EV_FIRST_AUDIO, EV_FIRST_TEXT, EV_REQUEST_SENT, SourceKind, SpanKind
from .backends import Backend, ModelRequest, RequestResult
from .payloads import AssetPool, surrogate_request

REPLAYABLE = (SpanKind.MODEL_REQUEST.value, SpanKind.TOOL_CALL.value, SpanKind.WAIT.value)
_TOL_NS = 1_000_000


@dataclass
class ReplayNode:
    span_id: str
    kind: str
    session_id: str
    start_ns: int
    end_ns: int
    deps: list[str] = field(default_factory=list)
    gap_ns: int = 0  # recorded client-side delay between (last dependency end | session start) and this start
    attrs: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    step_id: str | None = None
    interaction_id: str | None = None
    request_id: str | None = None

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass
class ReplayPlan:
    trace_id: str
    source_kind: str
    nodes: dict[str, ReplayNode]
    sessions: dict[str, list[str]]  # session_id -> node ids in start order
    session_offset_ns: dict[str, int]  # session start relative to the trace start
    dependency_source: str  # "explicit" | "inferred"
    t0_ns: int

    @property
    def n_requests(self) -> int:
        return sum(1 for n in self.nodes.values() if n.kind == SpanKind.MODEL_REQUEST.value)


def build_plan(trace: Trace) -> ReplayPlan:
    spans = [s for s in trace.spans.values() if s.kind in REPLAYABLE and s.end_ns is not None and s.session_id]
    if not spans:
        raise ValueError("trace has no replayable spans (model_request/tool_call/wait with a session)")
    t0 = trace.t0_ns
    replayable_ids = {s.span_id for s in spans}
    explicit = any(s.depends_on for s in spans)
    nodes: dict[str, ReplayNode] = {}
    by_session: dict[str, list[str]] = {}
    for s in sorted(spans, key=lambda s: s.start_ns):
        nodes[s.span_id] = ReplayNode(
            span_id=s.span_id,
            kind=s.kind,
            session_id=s.session_id,
            start_ns=s.start_ns,
            end_ns=s.end_ns,
            attrs=dict(s.attributes),
            name=s.name,
            step_id=s.step_id,
            interaction_id=s.interaction_id,
            request_id=s.request_id,
        )
        by_session.setdefault(s.session_id, []).append(s.span_id)

    def resolve(dep: str, seen: set[str]) -> list[str]:
        """Map a dependency onto replayable nodes: a non-replayable span (e.g. an agent_step)
        stands for the replayable spans it contains."""
        if dep in replayable_ids:
            return [dep]
        if dep in seen or dep not in trace.spans:
            return []
        seen.add(dep)
        inner = [c.span_id for c in trace.spans.values() if c.parent_span_id == dep]
        out: list[str] = []
        for c in inner:
            out.extend(resolve(c, seen))
        return out

    session_start: dict[str, int] = {}
    for sid, ids in by_session.items():
        sess_spans = [s for s in trace.spans.values() if s.session_id == sid and s.kind == SpanKind.SESSION.value]
        session_start[sid] = min([nodes[i].start_ns for i in ids] + [s.start_ns for s in sess_spans])
        for i in ids:
            n = nodes[i]
            if explicit:
                deps: list[str] = []
                for d in trace.spans[i].depends_on:
                    deps.extend(resolve(d, set()))
                n.deps = sorted(set(d for d in deps if d != i and d in nodes))
            else:
                earlier = [nodes[j] for j in ids if nodes[j].end_ns <= n.start_ns + _TOL_NS and j != i]
                if earlier:
                    n.deps = [max(earlier, key=lambda x: x.end_ns).span_id]
            ref = max((nodes[d].end_ns for d in n.deps), default=session_start[sid])
            n.gap_ns = max(0, n.start_ns - ref)
    return ReplayPlan(
        trace_id=trace.header.trace_id,
        source_kind=trace.header.source.get("kind", "unknown"),
        nodes=nodes,
        sessions=by_session,
        session_offset_ns={sid: session_start[sid] - t0 for sid in by_session},
        dependency_source="explicit" if explicit else "inferred",
        t0_ns=t0,
    )


class _ResultSink:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._fh = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")
        self.records: list[dict[str, Any]] = []

    def write(self, rec: dict[str, Any]) -> None:
        self.records.append(rec)
        if self._fh:
            self._fh.write(json.dumps(rec, default=str) + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.close()


@dataclass
class ReplayConfig:
    mode: str = "dependency"  # "request" | "dependency"
    speed: float = 1.0  # >1 compresses client-side gaps and (mode A) arrival offsets
    session_copies: int = 1
    session_arrival: str = "trace"  # "trace" | "all_at_once" | "stagger:<seconds>" | "poisson:<sessions_per_s>"
    on_failure: str = "skip_dependents"  # or "continue"
    tool_mode: str = "sleep"
    request_timeout_s: float | None = None
    seed: int = 0
    max_inflight: int | None = None  # optional client-side cap (recorded; changes open-loop semantics)

    def __post_init__(self) -> None:
        if self.mode not in ("request", "dependency"):
            raise ValueError(f"mode must be 'request' or 'dependency', not {self.mode!r}")
        if self.speed <= 0:
            raise ValueError("speed must be > 0")
        if self.session_copies < 1:
            raise ValueError("session_copies must be >= 1")


class ReplayEngine:
    def __init__(
        self,
        plan: ReplayPlan,
        backend: Backend,
        config: ReplayConfig,
        *,
        pool: AssetPool | None = None,
        results_path: str | Path | None = None,
        collector: TraceCollector | None = None,
    ) -> None:
        self.plan = plan
        self.backend = backend
        self.cfg = config
        self.pool = pool
        self.sink = _ResultSink(results_path)
        self.collector = collector
        self._rng = random.Random(config.seed)
        self._sem = asyncio.Semaphore(config.max_inflight) if config.max_inflight else None
        self.status: dict[tuple[int, str], str] = {}  # (copy, node) -> ok|error|timeout|skipped|cancelled
        self._t_start = 0

    # -- session timing ------------------------------------------------------------------------
    def _session_starts(self) -> dict[tuple[int, str], float]:
        """Start offset in seconds for each (copy, session)."""
        cfg = self.cfg
        keys = [(c, sid) for c in range(cfg.session_copies) for sid in sorted(self.plan.sessions, key=lambda s: self.plan.session_offset_ns[s])]
        out: dict[tuple[int, str], float] = {}
        arrival = cfg.session_arrival
        if arrival == "trace":
            span_s = max(self.plan.session_offset_ns.values(), default=0) / 1e9
            for c, sid in keys:
                # Copies repeat the recorded schedule back-to-back, shifted by one trace span.
                out[(c, sid)] = (self.plan.session_offset_ns[sid] / 1e9 + c * span_s) / cfg.speed
        elif arrival == "all_at_once":
            out = {k: 0.0 for k in keys}
        elif arrival.startswith("stagger:"):
            step = float(arrival.split(":", 1)[1])
            out = {k: i * step for i, k in enumerate(keys)}
        elif arrival.startswith("poisson:"):
            rate = float(arrival.split(":", 1)[1])
            t = 0.0
            for k in keys:
                out[k] = t
                t += self._rng.expovariate(rate)
        else:
            raise ValueError(f"unknown session_arrival {arrival!r}")
        return out

    # -- execution -----------------------------------------------------------------------------
    async def run(self) -> dict[str, Any]:
        self._t_start = time.monotonic_ns()
        starts = self._session_starts()
        if self.cfg.mode == "request":
            tasks = [
                asyncio.create_task(self._mode_a_request(copy, nid, starts[(copy, self.plan.nodes[nid].session_id)]))
                for copy in range(self.cfg.session_copies)
                for nid, n in self.plan.nodes.items()
                if n.kind == SpanKind.MODEL_REQUEST.value
            ]
        else:
            tasks = [
                asyncio.create_task(self._mode_b_session(copy, sid, starts[(copy, sid)]))
                for (copy, sid) in starts
            ]
        try:
            await asyncio.gather(*tasks)
        finally:
            self.sink.close()
        return self.summary()

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for st in self.status.values():
            counts[st] = counts.get(st, 0) + 1
        recs = [r for r in self.sink.records if r.get("kind") == SpanKind.MODEL_REQUEST.value]
        lateness = [r["lateness_s"] for r in recs if r.get("lateness_s") is not None]
        return {
            "mode": self.cfg.mode,
            "speed": self.cfg.speed,
            "session_copies": self.cfg.session_copies,
            "session_arrival": self.cfg.session_arrival,
            "dependency_source": self.plan.dependency_source,
            "trace_id": self.plan.trace_id,
            "trace_source_kind": self.plan.source_kind,
            "backend": getattr(self.backend, "name", "?"),
            "status_counts": counts,
            "requests": len(recs),
            "max_lateness_s": max(lateness) if lateness else None,
            "wall_s": (time.monotonic_ns() - self._t_start) / 1e9,
        }

    def _new_request(self, copy: int, node: ReplayNode) -> ModelRequest:
        rid = f"rp-{copy}-{node.request_id or node.span_id}-{self._rng.getrandbits(32):08x}"
        req = surrogate_request(
            node.attrs,
            self._rng,
            rid,
            self.pool,
            metadata={
                "session_id": f"{node.session_id}#{copy}",
                "replay_of_span": node.span_id,
                "replay_of_request": node.request_id,
                "copy": copy,
            },
        )
        if self.cfg.request_timeout_s:
            req.timeout_s = self.cfg.request_timeout_s
        return req

    async def _send(self, copy: int, node: ReplayNode, scheduled_ns: int | None) -> RequestResult:
        req = self._new_request(copy, node)
        dispatched = time.monotonic_ns()
        if self._sem:
            async with self._sem:
                res = await self.backend.run(req)
        else:
            res = await self.backend.run(req)
        rec = res.to_dict()
        rec.update(
            kind=SpanKind.MODEL_REQUEST.value,
            mode=self.cfg.mode,
            copy=copy,
            replay_of_span=node.span_id,
            session_id=req.metadata["session_id"],
            scheduled_ns=scheduled_ns,
            dispatched_ns=dispatched,
            lateness_s=None if scheduled_ns is None else (dispatched - scheduled_ns) / 1e9,
            recorded_duration_s=node.duration_ns / 1e9,
            input_modalities=req.input_modalities,
            output_modalities=req.output_modalities,
            payload=req.metadata.get("payload"),
        )
        self.sink.write(rec)
        self._trace_request(copy, node, req, res)
        return res

    def _trace_request(self, copy: int, node: ReplayNode, req: ModelRequest, res: RequestResult) -> None:
        c = self.collector
        if c is None or res.sent_ns is None:
            return
        h = c.start_span(
            SpanKind.MODEL_REQUEST,
            node.name or "model_request",
            start_ns=res.sent_ns,
            session_id=req.metadata["session_id"],
            request_id=req.request_id,
            step_id=node.step_id,
            attributes={
                "input_modalities": req.input_modalities,
                "output_modalities": req.output_modalities,
                "endpoint": "chat.completions",
                "replay_of_span": node.span_id,
                "replay_mode": self.cfg.mode,
                "payload": req.metadata.get("payload"),
            },
        )
        h.event(EV_REQUEST_SENT, time_ns=res.sent_ns)
        if res.first_text_ns:
            h.event(EV_FIRST_TEXT, time_ns=res.first_text_ns)
        if res.first_audio_ns:
            h.event(EV_FIRST_AUDIO, time_ns=res.first_audio_ns)
        h.event(EV_COMPLETE, time_ns=res.done_ns or c.now(), status=res.status)
        status = {"ok": "ok", "timeout": "timeout", "cancelled": "cancelled"}.get(res.status, "error")
        h.end(status, end_ns=max(res.done_ns or c.now(), res.sent_ns), completion_tokens=res.completion_tokens, output_audio_s=res.output_audio_s)

    async def _mode_a_request(self, copy: int, nid: str, session_start_s: float) -> None:
        node = self.plan.nodes[nid]
        # Offset of this request inside its session, scaled; sessions are placed by _session_starts.
        within = (node.start_ns - self.plan.t0_ns - self.plan.session_offset_ns[node.session_id]) / 1e9 / self.cfg.speed
        at = self._t_start + int((session_start_s + within) * 1e9)
        delay = (at - time.monotonic_ns()) / 1e9
        if delay > 0:
            await asyncio.sleep(delay)
        res = await self._send(copy, node, scheduled_ns=at)
        self.status[(copy, nid)] = res.status

    async def _mode_b_session(self, copy: int, sid: str, start_s: float) -> None:
        delay = (self._t_start + int(start_s * 1e9) - time.monotonic_ns()) / 1e9
        if delay > 0:
            await asyncio.sleep(delay)
        done: dict[str, asyncio.Event] = {nid: asyncio.Event() for nid in self.plan.sessions[sid]}

        async def run_node(nid: str) -> None:
            node = self.plan.nodes[nid]
            try:
                for d in node.deps:
                    if d in done:
                        await done[d].wait()
                failed = [d for d in node.deps if self.status.get((copy, d)) not in ("ok", None)]
                if failed and self.cfg.on_failure == "skip_dependents":
                    self.status[(copy, nid)] = "skipped"
                    self.sink.write({"kind": node.kind, "mode": "dependency", "copy": copy, "replay_of_span": nid,
                                     "status": "skipped", "reason": f"dependency failed: {failed}"})
                    return
                await asyncio.sleep(node.gap_ns / 1e9 / self.cfg.speed)
                if node.kind == SpanKind.MODEL_REQUEST.value:
                    res = await self._send(copy, node, scheduled_ns=None)
                    self.status[(copy, nid)] = res.status
                else:
                    t0 = time.monotonic_ns()
                    await asyncio.sleep(node.duration_ns / 1e9 / self.cfg.speed)
                    self.status[(copy, nid)] = "ok"
                    self.sink.write({"kind": node.kind, "mode": "dependency", "copy": copy, "replay_of_span": nid,
                                     "status": "ok", "tool_mode": self.cfg.tool_mode, "start_ns": t0,
                                     "end_ns": time.monotonic_ns(), "recorded_duration_s": node.duration_ns / 1e9})
            finally:
                done[nid].set()

        await asyncio.gather(*(run_node(nid) for nid in self.plan.sessions[sid]))


def replay_collector(path: str | Path, plan: ReplayPlan, cfg: ReplayConfig, backend_name: str) -> TraceCollector:
    """Collector for the trace recorded during a replay (source kind ``replay``)."""
    return TraceCollector(
        path,
        source_kind=SourceKind.REPLAY,
        generator="omni_research.replay",
        seed=cfg.seed,
        extra_source={"replay_of_trace": plan.trace_id, "mode": cfg.mode, "speed": cfg.speed, "backend": backend_name},
    )
