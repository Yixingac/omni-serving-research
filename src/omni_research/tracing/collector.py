"""Append-only JSONL trace writer and a span-based collector for agent traces.

Usage::

    tc = TraceCollector(path, source_kind="observed", generator="voice_agent", seed=0)
    with tc.span("session", "math-agent", session_id=sid) as session:
        with session.child("agent_step", "step-1", step_id="s1") as step:
            with step.child("model_request", "plan", request_id=rid,
                            attributes={...}) as req:
                req.event("first_text_token")
    tc.close()

Spans are written once, when they end (with their final status). Events are written as they
happen. Anything still open at ``close()`` is written with status ``cancelled`` and
``attributes.incomplete = True`` so a crash never silently drops a started span.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from .schema import (
    REQUIRED_IDS,
    ClockDomain,
    ContentPolicy,
    Event,
    SourceKind,
    Span,
    SpanKind,
    Status,
    TraceFooter,
    TraceHeader,
    new_id,
)

_INHERITED_IDS = ("session_id", "interaction_id", "step_id")


def read_clock_anchor(samples: int = 5) -> tuple[int, int, int]:
    """Return (mono_ns, wall_ns, uncertainty_ns) from the tightest of several back-to-back reads."""
    best = None
    for _ in range(samples):
        m0 = time.monotonic_ns()
        w = time.time_ns()
        m1 = time.monotonic_ns()
        cand = (m1 - m0, (m0 + m1) // 2, w)
        if best is None or cand[0] < best[0]:
            best = cand
    assert best is not None
    gap, mono, wall = best
    return mono, wall, gap // 2 + 1


class TraceWriter:
    """Thread-safe append-only JSONL writer. Each record is one line; ``flush`` fsyncs."""

    def __init__(self, path: str | os.PathLike[str], flush_every: int = 64) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()
        self._since_flush = 0
        self._flush_every = flush_every
        self.records = 0

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, separators=(",", ":"), sort_keys=False, default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self.records += 1
            self._since_flush += 1
            if self._since_flush >= self._flush_every:
                self._flush_locked()

    def _flush_locked(self) -> None:
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._since_flush = 0

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._flush_locked()
                self._fh.close()


class SpanHandle:
    """A started span. Use as a (sync or async) context manager or call ``end`` explicitly."""

    def __init__(self, collector: "TraceCollector", span: Span) -> None:
        self._c = collector
        self.span = span
        self._ended = False

    # ids -------------------------------------------------------------------------------------
    @property
    def span_id(self) -> str:
        return self.span.span_id

    @property
    def ended(self) -> bool:
        return self._ended

    # children & events -----------------------------------------------------------------------
    def child(self, kind: str | SpanKind, name: str, *, depends_on: Iterable[str] = (), **kw: Any) -> "SpanHandle":
        for key in _INHERITED_IDS:
            kw.setdefault(key, getattr(self.span, key))
        return self._c.start_span(kind, name, parent=self, depends_on=depends_on, **kw)

    def event(self, name: str, time_ns: int | None = None, **attributes: Any) -> None:
        self._c._write_event(
            Event(
                trace_id=self.span.trace_id,
                span_id=self.span.span_id,
                name=name,
                time_ns=self._c.now() if time_ns is None else time_ns,
                attributes=attributes,
            )
        )

    def set(self, **attributes: Any) -> None:
        self.span.attributes.update(attributes)

    def end(self, status: str | Status = Status.OK, end_ns: int | None = None, **attributes: Any) -> None:
        if self._ended:
            return
        self._ended = True
        self.span.end_ns = self._c.now() if end_ns is None else end_ns
        self.span.status = Status(status).value
        self.span.attributes.update(attributes)
        self._c._finish_span(self)

    # context managers -----------------------------------------------------------------------
    def _status_for(self, exc: BaseException | None) -> tuple[Status, dict[str, Any]]:
        if exc is None:
            return Status.OK, {}
        if isinstance(exc, asyncio.CancelledError):
            return Status.CANCELLED, {"error_type": "CancelledError"}
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
            return Status.TIMEOUT, {"error_type": type(exc).__name__}
        return Status.ERROR, {"error_type": type(exc).__name__, "error": str(exc)[:500]}

    def __enter__(self) -> "SpanHandle":
        return self

    def __exit__(self, et, exc, tb) -> None:  # noqa: ANN001
        status, extra = self._status_for(exc)
        if not self._ended:
            self.end(status, **extra)

    async def __aenter__(self) -> "SpanHandle":
        return self

    async def __aexit__(self, et, exc, tb) -> None:  # noqa: ANN001
        self.__exit__(et, exc, tb)


class TraceCollector:
    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        source_kind: str | SourceKind,
        generator: str,
        seed: int | None = None,
        content_policy: str | ContentPolicy = ContentPolicy.NONE,
        workload: dict[str, Any] | None = None,
        notes: str = "",
        trace_id: str | None = None,
        extra_source: dict[str, Any] | None = None,
    ) -> None:
        mono, wall, unc = read_clock_anchor()
        self.trace_id = trace_id or new_id("tr-")
        self.clock = ClockDomain(
            domain="monotonic_ns",
            host=socket.gethostname(),
            pid=os.getpid(),
            mono_anchor_ns=mono,
            wall_anchor_ns=wall,
            anchor_uncertainty_ns=unc,
        )
        source = {"kind": SourceKind(source_kind).value, "generator": generator, "seed": seed}
        source.update(extra_source or {})
        self.header = TraceHeader(
            trace_id=self.trace_id,
            clock=self.clock,
            source=source,
            content_policy=ContentPolicy(content_policy).value,
            workload=workload or {},
            notes=notes,
        )
        self.writer = TraceWriter(path)
        self.writer.write(self.header.to_dict())
        self._open: dict[str, SpanHandle] = {}
        self._lock = threading.Lock()
        self.span_count = 0
        self.event_count = 0
        self._closed = False

    @staticmethod
    def now() -> int:
        return time.monotonic_ns()

    def start_span(
        self,
        kind: str | SpanKind,
        name: str,
        *,
        parent: SpanHandle | None = None,
        depends_on: Iterable[str] = (),
        start_ns: int | None = None,
        attributes: dict[str, Any] | None = None,
        span_id: str | None = None,
        **ids: Any,
    ) -> SpanHandle:
        kind = SpanKind(kind)
        missing = [k for k in REQUIRED_IDS[kind] if not ids.get(k)]
        if missing:
            raise ValueError(f"{kind.value} span {name!r} needs ids {missing}")
        span = Span(
            trace_id=self.trace_id,
            span_id=span_id or new_id("sp-"),
            kind=kind.value,
            name=name,
            start_ns=self.now() if start_ns is None else start_ns,
            parent_span_id=parent.span_id if parent else None,
            depends_on=list(depends_on),
            attributes=dict(attributes or {}),
            **{k: ids.get(k) for k in ("session_id", "interaction_id", "step_id", "request_id")},
        )
        handle = SpanHandle(self, span)
        with self._lock:
            self._open[span.span_id] = handle
        return handle

    # convenience alias used by agents
    def span(self, kind: str | SpanKind, name: str, **kw: Any) -> SpanHandle:
        return self.start_span(kind, name, **kw)

    def _write_event(self, ev: Event) -> None:
        self.writer.write(ev.to_dict())
        with self._lock:
            self.event_count += 1

    def _finish_span(self, handle: SpanHandle) -> None:
        self.writer.write(handle.span.to_dict())
        with self._lock:
            self._open.pop(handle.span_id, None)
            self.span_count += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._lock:
            leftovers = list(self._open.values())
        for h in leftovers:
            h.end(Status.CANCELLED, incomplete=True)
        footer = TraceFooter(
            trace_id=self.trace_id,
            span_count=self.span_count,
            event_count=self.event_count,
            closed_ns=self.now(),
        )
        self.writer.write(footer.to_dict())
        self.writer.close()

    def __enter__(self) -> "TraceCollector":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
