"""Application/agent trace layer: schema, collector, loader, validator."""

from .collector import SpanHandle, TraceCollector, TraceWriter, read_clock_anchor
from .io import Trace, load_trace
from .schema import SCHEMA_VERSION, ClockDomain, ContentPolicy, SourceKind, SpanKind, Status, new_id
from .validate import ValidationReport, validate_file, validate_trace

__all__ = [
    "SCHEMA_VERSION",
    "ClockDomain",
    "ContentPolicy",
    "SourceKind",
    "SpanHandle",
    "SpanKind",
    "Status",
    "Trace",
    "TraceCollector",
    "TraceWriter",
    "ValidationReport",
    "load_trace",
    "new_id",
    "read_clock_anchor",
    "validate_file",
    "validate_trace",
]
