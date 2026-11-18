import json

import pytest

from omni_research.tracing import TraceCollector, load_trace, validate_file, validate_trace
from omni_research.tracing.schema import SCHEMA_VERSION


def _agent_trace(path, *, close=True):
    tc = TraceCollector(path, source_kind="observed", generator="test-agent", seed=1)
    sess = tc.span("session", "s", session_id="sess-1")
    step1 = sess.child("agent_step", "plan", step_id="st-1")
    req1 = step1.child(
        "model_request",
        "plan-call",
        request_id="r-1",
        attributes={"input_modalities": ["audio", "text"], "output_modalities": ["text"], "endpoint": "chat"},
    )
    req1.event("first_text_token")
    req1.end()
    tool = step1.child("tool_call", "calc", depends_on=[req1.span_id], attributes={"tool_name": "calculator"})
    tool.end()
    step1.end()
    step2 = sess.child("agent_step", "answer", step_id="st-2", depends_on=[step1.span_id])
    req2 = step2.child(
        "model_request",
        "answer-call",
        request_id="r-2",
        depends_on=[tool.span_id],
        attributes={"input_modalities": ["text"], "output_modalities": ["text", "audio"], "endpoint": "chat"},
    )
    req2.event("first_audio_chunk")
    req2.end()
    step2.end()
    sess.end()
    if close:
        tc.close()
    return tc, (req1, tool, req2)


def test_roundtrip_and_valid(tmp_path):
    p = tmp_path / "t.jsonl"
    tc, (req1, tool, req2) = _agent_trace(p)
    trace = load_trace(p)
    assert trace.header.schema == SCHEMA_VERSION
    assert trace.header.source["kind"] == "observed"
    assert len(trace.spans) == 6
    assert trace.spans[req2.span_id].depends_on == [tool.span_id]
    assert trace.spans[req1.span_id].session_id == "sess-1"  # inherited from the session span
    assert trace.spans[req1.span_id].step_id == "st-1"
    rep = validate_trace(trace)
    assert rep.ok, rep.errors
    assert rep.stats["by_kind"]["model_request"] == 2
    # Clock anchor converts monotonic span times to wall time on this host.
    wall = trace.header.clock.to_wall_ns(trace.spans[req1.span_id].start_ns)
    assert abs(trace.header.clock.from_wall_ns(wall) - trace.spans[req1.span_id].start_ns) == 0


def test_required_ids_enforced_at_creation(tmp_path):
    tc = TraceCollector(tmp_path / "t.jsonl", source_kind="synthetic", generator="g")
    with pytest.raises(ValueError):
        tc.span("model_request", "no-ids", attributes={})
    tc.close()


def test_open_spans_closed_as_cancelled(tmp_path):
    p = tmp_path / "t.jsonl"
    tc = TraceCollector(p, source_kind="observed", generator="g")
    s = tc.span("session", "s", session_id="x")
    s.child("wait", "think", attributes={"reason": "user_think"})
    tc.close()  # neither span ended explicitly
    trace = load_trace(p)
    assert {sp.status for sp in trace.spans.values()} == {"cancelled"}
    assert all(sp.attributes.get("incomplete") for sp in trace.spans.values())
    assert validate_trace(trace).ok


def test_context_manager_marks_errors(tmp_path):
    p = tmp_path / "t.jsonl"
    tc = TraceCollector(p, source_kind="observed", generator="g")
    with pytest.raises(RuntimeError):
        with tc.span("session", "s", session_id="x"):
            raise RuntimeError("boom")
    tc.close()
    (span,) = load_trace(p).spans.values()
    assert span.status == "error" and span.attributes["error_type"] == "RuntimeError"


def _rewrite(path, mutate):
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    lines = mutate(lines)
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")


def test_validator_detects_cycle_and_dangling(tmp_path):
    p = tmp_path / "t.jsonl"
    tc, (req1, tool, req2) = _agent_trace(p)

    def mutate(lines):
        for rec in lines:
            if rec.get("record_type") != "span":
                continue
            if rec.get("span_id") == req1.span_id:
                rec["depends_on"] = [req2.span_id]  # req1 <- req2 <- tool <- req1: a cycle
            if rec.get("span_id") == tool.span_id:
                rec["parent_span_id"] = "sp-missing"
        return lines

    _rewrite(p, mutate)
    rep = validate_file(p)
    assert not rep.ok
    assert any("cycle" in e for e in rep.errors)
    assert any("parent sp-missing" in e for e in rep.errors)


def test_validator_flags_causality_violation_in_observed_trace(tmp_path):
    p = tmp_path / "t.jsonl"
    tc, (req1, tool, req2) = _agent_trace(p)

    def mutate(lines):
        for rec in lines:
            if rec.get("record_type") == "span" and rec.get("span_id") == req2.span_id:
                rec["start_ns"] = rec["start_ns"] - 10_000_000_000  # starts long before the tool ended
        return lines

    _rewrite(p, mutate)
    rep = validate_file(p)
    assert any("before dependency" in e for e in rep.errors)


def test_validator_bad_modality_and_footer_mismatch(tmp_path):
    p = tmp_path / "t.jsonl"
    tc, (req1, tool, req2) = _agent_trace(p)

    def mutate(lines):
        out = []
        for rec in lines:
            if rec.get("record_type") == "span" and rec.get("span_id") == req1.span_id:
                rec["attributes"]["input_modalities"] = ["smell"]
            if rec.get("record_type") == "event" and rec.get("name") == "first_audio_chunk":
                continue  # drop one event so the footer count no longer matches
            out.append(rec)
        return out

    _rewrite(p, mutate)
    rep = validate_file(p)
    assert any("unknown modalities" in e for e in rep.errors)
    assert any("event_count" in e for e in rep.errors)


def test_missing_footer_is_warning_only(tmp_path):
    p = tmp_path / "t.jsonl"
    _agent_trace(p)
    _rewrite(p, lambda lines: [r for r in lines if r.get("record_type") != "trace_footer"])
    rep = validate_file(p)
    assert rep.ok and any("no trace_footer" in w for w in rep.warnings)


def test_synthetic_trace_requires_generator(tmp_path):
    p = tmp_path / "t.jsonl"
    tc = TraceCollector(p, source_kind="synthetic", generator="")
    tc.close()
    assert any("generator" in e for e in validate_file(p).errors)
