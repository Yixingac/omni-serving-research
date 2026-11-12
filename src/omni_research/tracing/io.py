"""Load agent traces into memory with a few graph helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .schema import Event, Span, TraceHeader, event_from_dict, header_from_dict, span_from_dict


@dataclass
class Trace:
    header: TraceHeader
    spans: dict[str, Span] = field(default_factory=dict)
    events: dict[str, list[Event]] = field(default_factory=lambda: defaultdict(list))
    footer: dict[str, Any] | None = None
    unknown_records: list[dict[str, Any]] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    duplicate_span_ids: list[str] = field(default_factory=list)

    # -- selection ---------------------------------------------------------------------------
    def by_kind(self, kind: str) -> list[Span]:
        return sorted((s for s in self.spans.values() if s.kind == kind), key=lambda s: s.start_ns)

    def children(self, span_id: str) -> list[Span]:
        return sorted((s for s in self.spans.values() if s.parent_span_id == span_id), key=lambda s: s.start_ns)

    def sessions(self) -> dict[str, list[Span]]:
        out: dict[str, list[Span]] = defaultdict(list)
        for s in self.spans.values():
            if s.session_id:
                out[s.session_id].append(s)
        for v in out.values():
            v.sort(key=lambda s: s.start_ns)
        return dict(out)

    def first_event(self, span_id: str, name: str) -> Event | None:
        evs = [e for e in self.events.get(span_id, []) if e.name == name]
        return min(evs, key=lambda e: e.time_ns) if evs else None

    @property
    def t0_ns(self) -> int:
        return min((s.start_ns for s in self.spans.values()), default=0)

    def iter_records(self) -> Iterator[dict[str, Any]]:
        yield self.header.to_dict()
        for s in sorted(self.spans.values(), key=lambda s: s.start_ns):
            yield s.to_dict()
        for evs in self.events.values():
            for e in evs:
                yield e.to_dict()


def load_trace(path: str | Path) -> Trace:
    path = Path(path)
    header = None
    trace: Trace | None = None
    pending: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                if trace is not None:
                    trace.parse_errors.append(f"line {lineno}: {exc}")
                else:
                    pending.append({"__parse_error__": f"line {lineno}: {exc}"})
                continue
            rt = rec.get("record_type")
            if rt == "trace_header" and header is None:
                header = header_from_dict(rec)
                trace = Trace(header=header)
                for p in pending:
                    trace.parse_errors.append(p.get("__parse_error__", "record before header"))
                continue
            if trace is None:
                pending.append({"__parse_error__": f"line {lineno}: {rt!r} before trace_header"})
                continue
            if rt == "span":
                s = span_from_dict(rec)
                if s.span_id in trace.spans:
                    trace.duplicate_span_ids.append(s.span_id)
                trace.spans[s.span_id] = s
            elif rt == "event":
                e = event_from_dict(rec)
                trace.events[e.span_id].append(e)
            elif rt == "trace_footer":
                trace.footer = rec
            else:
                trace.unknown_records.append(rec)
    if trace is None:
        raise ValueError(f"{path}: no trace_header record")
    return trace
