"""Structural validation of agent traces.

Errors make a trace unusable for analysis or replay; warnings flag things a reader should know
(e.g. a child span that outlives its parent because it was cancelled late).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io import Trace, load_trace
from .schema import MODALITIES, REQUIRED_ATTRS, REQUIRED_IDS, SCHEMA_VERSION, SourceKind, SpanKind, Status

# Allowed slack when comparing times of related records (monotonic reads on one host are
# ordered, but writers may stamp a child end a hair after reading the parent end).
TOLERANCE_NS = 1_000_000  # 1 ms


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings, "stats": self.stats}


def _find_cycle(graph: dict[str, list[str]]) -> list[str] | None:
    """Return one cycle in a dependency graph (node -> prerequisites), or None."""
    WHITE, GREY, BLACK = 0, 1, 2
    color = {n: WHITE for n in graph}
    stack: list[str] = []

    def visit(n: str) -> list[str] | None:
        color[n] = GREY
        stack.append(n)
        for m in graph.get(n, []):
            if color.get(m, WHITE) == GREY:
                return stack[stack.index(m) :] + [m]
            if color.get(m, WHITE) == WHITE and m in graph:
                found = visit(m)
                if found:
                    return found
        stack.pop()
        color[n] = BLACK
        return None

    for n in list(graph):
        if color[n] == WHITE:
            found = visit(n)
            if found:
                return found
    return None


def validate_trace(trace: Trace) -> ValidationReport:
    rep = ValidationReport()
    E, W = rep.errors.append, rep.warnings.append
    h = trace.header
    if h.schema != SCHEMA_VERSION:
        E(f"header.schema is {h.schema!r}, expected {SCHEMA_VERSION!r}")
    try:
        kind = SourceKind(h.source.get("kind"))
    except ValueError:
        E(f"header.source.kind {h.source.get('kind')!r} is not one of {[k.value for k in SourceKind]}")
        kind = None
    if kind == SourceKind.SYNTHETIC and not h.source.get("generator"):
        E("synthetic trace must name its generator in header.source.generator")
    if h.clock.domain not in ("monotonic_ns", "synthetic_ns"):
        E(f"unknown clock domain {h.clock.domain!r}")
    for msg in trace.parse_errors:
        E(f"parse: {msg}")
    for sid in trace.duplicate_span_ids:
        E(f"duplicate span_id {sid}")

    spans = trace.spans
    counts: dict[str, int] = {}
    for s in spans.values():
        counts[s.kind] = counts.get(s.kind, 0) + 1
        where = f"span {s.span_id} ({s.kind} {s.name!r})"
        if s.trace_id != h.trace_id:
            E(f"{where}: trace_id {s.trace_id} != header {h.trace_id}")
        try:
            k = SpanKind(s.kind)
        except ValueError:
            E(f"{where}: unknown kind")
            continue
        try:
            Status(s.status)
        except ValueError:
            E(f"{where}: unknown status {s.status!r}")
        for idname in REQUIRED_IDS[k]:
            if not getattr(s, idname):
                E(f"{where}: missing {idname}")
        for attr in REQUIRED_ATTRS[k]:
            if attr not in s.attributes:
                E(f"{where}: missing attribute {attr!r}")
        if s.end_ns is None:
            E(f"{where}: no end_ns")
        elif s.end_ns < s.start_ns:
            E(f"{where}: ends before it starts ({s.end_ns} < {s.start_ns})")
        if k == SpanKind.MODEL_REQUEST:
            for key in ("input_modalities", "output_modalities"):
                bad = [m for m in s.attributes.get(key, []) if m not in MODALITIES]
                if bad:
                    E(f"{where}: {key} has unknown modalities {bad}")
        if s.parent_span_id:
            p = spans.get(s.parent_span_id)
            if p is None:
                E(f"{where}: parent {s.parent_span_id} not in trace")
            else:
                if s.start_ns + TOLERANCE_NS < p.start_ns:
                    E(f"{where}: starts before its parent {p.span_id}")
                if s.end_ns is not None and p.end_ns is not None and s.end_ns > p.end_ns + TOLERANCE_NS:
                    W(f"{where}: ends after its parent {p.span_id} by {(s.end_ns - p.end_ns) / 1e6:.3f} ms")
                if s.session_id and p.session_id and s.session_id != p.session_id:
                    E(f"{where}: session_id differs from parent's")

    # Dependency DAG: references exist, no cycles, and causality holds for observed traces.
    graph = {sid: list(s.depends_on) for sid, s in spans.items()}
    for sid, deps in graph.items():
        for d in deps:
            if d not in spans:
                E(f"span {sid}: depends_on {d} not in trace")
            elif d == sid:
                E(f"span {sid}: depends on itself")
            else:
                dep, me = spans[d], spans[sid]
                if dep.end_ns is not None and dep.end_ns > me.start_ns + TOLERANCE_NS:
                    msg = f"span {sid} starts {(dep.end_ns - me.start_ns) / 1e6:.3f} ms before dependency {d} ends"
                    (E if kind == SourceKind.OBSERVED else W)(msg)
    cycle = _find_cycle(graph)
    if cycle:
        E(f"dependency cycle: {' -> '.join(cycle)}")

    n_events = 0
    for span_id, evs in trace.events.items():
        s = spans.get(span_id)
        for e in evs:
            n_events += 1
            if s is None:
                E(f"event {e.name!r} refers to missing span {span_id}")
                continue
            if e.time_ns + TOLERANCE_NS < s.start_ns or (s.end_ns is not None and e.time_ns > s.end_ns + TOLERANCE_NS):
                W(f"event {e.name!r} at {e.time_ns} outside span {span_id} [{s.start_ns}, {s.end_ns}]")

    if trace.footer:
        if trace.footer.get("span_count") != len(spans):
            E(f"footer span_count {trace.footer.get('span_count')} != {len(spans)} spans read")
        if trace.footer.get("event_count") != n_events:
            E(f"footer event_count {trace.footer.get('event_count')} != {n_events} events read")
    else:
        W("no trace_footer: the writer may not have closed cleanly")

    rep.stats = {"spans": len(spans), "events": n_events, "by_kind": counts, "source": h.source.get("kind")}
    return rep


def validate_file(path: str | Path) -> ValidationReport:
    try:
        trace = load_trace(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return ValidationReport(errors=[f"cannot load {path}: {exc}"])
    return validate_trace(trace)
