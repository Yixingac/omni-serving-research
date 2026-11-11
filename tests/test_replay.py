import asyncio
import json
import time

import pytest

from omni_research.replay import (
    BackendPool,
    MockBackend,
    MockServiceModel,
    ModelRequest,
    ReplayConfig,
    ReplayEngine,
    build_plan,
    replay_collector,
)
from omni_research.replay.backends import OpenAIChatBackend, RequestResult, silence_wav, wav_duration
from omni_research.tracing import TraceCollector, load_trace, validate_trace
from omni_research.tracing.schema import ClockDomain, Span, TraceHeader
from omni_research.tracing.io import Trace

MS = 1_000_000


def _synthetic_trace():
    """Two sessions. In s1: r1 -> tool(50 ms) -> r2, with a 20 ms gap before the tool."""
    h = TraceHeader(
        trace_id="tr-x",
        clock=ClockDomain("synthetic_ns", "h", 1, 0, 0, 0),
        source={"kind": "synthetic", "generator": "test"},
    )
    attrs = {"input_modalities": ["text"], "output_modalities": ["text"], "endpoint": "chat", "input_text_chars": 50}
    spans = [
        Span("tr-x", "s1", "session", "s1", 0, 400 * MS, session_id="A"),
        Span("tr-x", "r1", "model_request", "r1", 10 * MS, 60 * MS, parent_span_id="s1", session_id="A", request_id="q1", attributes=dict(attrs)),
        Span("tr-x", "t1", "tool_call", "t1", 80 * MS, 130 * MS, parent_span_id="s1", session_id="A", depends_on=["r1"], attributes={"tool_name": "calc"}),
        Span("tr-x", "r2", "model_request", "r2", 140 * MS, 190 * MS, parent_span_id="s1", session_id="A", request_id="q2", depends_on=["t1"], attributes=dict(attrs)),
        Span("tr-x", "s2", "session", "s2", 100 * MS, 300 * MS, session_id="B"),
        Span("tr-x", "r3", "model_request", "r3", 100 * MS, 150 * MS, parent_span_id="s2", session_id="B", request_id="q3", attributes=dict(attrs)),
    ]
    t = Trace(header=h)
    for s in spans:
        t.spans[s.span_id] = s
    return t


def test_build_plan_explicit_dependencies_and_gaps():
    plan = build_plan(_synthetic_trace())
    assert plan.dependency_source == "explicit"
    assert plan.n_requests == 3
    assert plan.nodes["t1"].deps == ["r1"] and plan.nodes["t1"].gap_ns == 20 * MS
    assert plan.nodes["r2"].deps == ["t1"] and plan.nodes["r2"].gap_ns == 10 * MS
    assert plan.nodes["r1"].gap_ns == 10 * MS  # from session start
    assert plan.session_offset_ns == {"A": 0, "B": 100 * MS}


def test_build_plan_infers_sequential_dependencies():
    t = _synthetic_trace()
    for s in t.spans.values():
        s.depends_on = []
    plan = build_plan(t)
    assert plan.dependency_source == "inferred"
    assert plan.nodes["r2"].deps == ["t1"]
    assert plan.nodes["t1"].deps == ["r1"]
    assert plan.nodes["r3"].deps == []


async def test_dependency_mode_respects_order(tmp_path):
    plan = build_plan(_synthetic_trace())
    backend = MockBackend(MockServiceModel(prefill_s=0.03, per_token_s=0.0, output_tokens=1))
    eng = ReplayEngine(plan, backend, ReplayConfig(mode="dependency"), results_path=tmp_path / "r.jsonl")
    summary = await eng.run()
    assert summary["status_counts"] == {"ok": 4}
    recs = {r["replay_of_span"]: r for r in (json.loads(x) for x in (tmp_path / "r.jsonl").read_text().splitlines())}
    # r2 cannot be sent before the replayed tool call (which follows r1's completion) has ended.
    assert recs["r2"]["sent_ns"] >= recs["t1"]["end_ns"]
    assert recs["t1"]["start_ns"] >= recs["r1"]["done_ns"]
    # The tool sleep reproduced the recorded 50 ms (scheduler slack allowed).
    assert recs["t1"]["end_ns"] - recs["t1"]["start_ns"] >= 45 * MS


async def test_dependency_mode_feedback_from_slow_server(tmp_path):
    """A slower server delays every downstream request: the closed-loop property."""
    plan = build_plan(_synthetic_trace())

    async def run(prefill):
        eng = ReplayEngine(plan, MockBackend(MockServiceModel(prefill_s=prefill, per_token_s=0, output_tokens=1)), ReplayConfig())
        await eng.run()
        r = {x["replay_of_span"]: x for x in eng.sink.records}
        return (r["r2"]["sent_ns"] - r["r1"]["sent_ns"]) / 1e9

    fast, slow = await run(0.01), await run(0.15)
    assert slow - fast > 0.1


async def test_request_mode_is_open_loop(tmp_path):
    plan = build_plan(_synthetic_trace())
    backend = MockBackend(MockServiceModel(prefill_s=0.3, per_token_s=0, output_tokens=1))
    eng = ReplayEngine(plan, backend, ReplayConfig(mode="request", speed=2.0))
    summary = await eng.run()
    assert summary["requests"] == 3
    r = {x["replay_of_span"]: x for x in eng.sink.records}
    # Open loop: r2 goes out at its recorded offset (140 ms / speed 2 = 70 ms) although r1
    # (300 ms prefill) has not finished.
    assert r["r2"]["sent_ns"] < r["r1"]["done_ns"]
    assert abs((r["r2"]["scheduled_ns"] - r["r1"]["scheduled_ns"]) / 1e9 - 0.065) < 0.01
    assert summary["max_lateness_s"] < 0.05


async def test_failure_skips_dependents():
    plan = build_plan(_synthetic_trace())
    backend = MockBackend(MockServiceModel(error_rate=1.0, prefill_s=0.001))
    eng = ReplayEngine(plan, backend, ReplayConfig())
    s = await eng.run()
    assert s["status_counts"].get("skipped") == 2  # t1 and r2 of session A
    assert s["status_counts"].get("error") == 2  # r1 and r3


async def test_session_copies_and_poisson_arrivals():
    plan = build_plan(_synthetic_trace())
    eng = ReplayEngine(plan, MockBackend(MockServiceModel(prefill_s=0.001, per_token_s=0, output_tokens=1)),
                       ReplayConfig(session_copies=3, session_arrival="poisson:50", seed=3))
    s = await eng.run()
    assert s["requests"] == 9
    assert len({r["session_id"] for r in eng.sink.records if r.get("kind") == "model_request"}) == 6


async def test_replay_trace_is_valid(tmp_path):
    plan = build_plan(_synthetic_trace())
    cfg = ReplayConfig()
    tc = replay_collector(tmp_path / "replay.jsonl", plan, cfg, "mock")
    eng = ReplayEngine(plan, MockBackend(MockServiceModel(prefill_s=0.001)), cfg, collector=tc)
    await eng.run()
    tc.close()
    t = load_trace(tmp_path / "replay.jsonl")
    assert t.header.source["kind"] == "replay" and t.header.source["replay_of_trace"] == "tr-x"
    rep = validate_trace(t)
    assert rep.ok, rep.errors
    assert rep.stats["by_kind"] == {"model_request": 3}


async def test_backend_pool_policies():
    backs = [MockBackend(MockServiceModel(prefill_s=0.02, per_token_s=0, output_tokens=1)) for _ in range(2)]
    pool = BackendPool(backs, policy="least_outstanding")
    reqs = [ModelRequest(f"r{i}", [{"role": "user", "content": "x"}], ["text"], ["text"], metadata={}) for i in range(6)]
    await asyncio.gather(*(pool.run(r) for r in reqs))
    assert pool.routed == [3, 3]
    aff = BackendPool([MockBackend(), MockBackend()], policy="session_affinity")
    for i in range(4):
        res = await aff.run(ModelRequest(f"a{i}", [], ["text"], ["text"], metadata={"session_id": "same"}))
    assert sorted(aff.routed) == [0, 4]
    with pytest.raises(ValueError):
        BackendPool(backs, policy="random")


def test_wav_duration_pcm_float_and_streaming_header():
    wav = silence_wav(1.5, rate=24000)
    assert wav_duration(wav) == (1.5, 24000)
    # Float32 WAV (format 3) as SGLang-Omni may emit; `wave` cannot read these.
    import struct

    n = 2400
    data = b"\x00" * (n * 4)
    f32 = (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE" + b"fmt " + struct.pack("<IHHIIHH", 16, 3, 1, 24000, 96000, 4, 32)
           + b"data" + struct.pack("<I", len(data)) + data)
    assert wav_duration(f32) == (0.1, 24000)
    streaming = f32[:40] + struct.pack("<I", 0xFFFFFFFF) + data
    assert wav_duration(streaming)[0] == pytest.approx(0.1)
    assert wav_duration(b"not a wav") == (0.0, None)


def test_sse_chunk_parsing():
    b = OpenAIChatBackend("http://x")
    res = RequestResult("r", "openai")
    parts = []
    import base64

    b._apply_chunk({"choices": [{"delta": {"role": "assistant"}}]}, 1, res, parts)
    b._apply_chunk({"choices": [{"delta": {"content": "Hel"}}]}, 2, res, parts)
    b._apply_chunk({"choices": [{"delta": {"content": "lo"}}]}, 3, res, parts)
    b._apply_chunk({"choices": [{"delta": {"audio": {"id": "a", "data": base64.b64encode(silence_wav(0.5)).decode()}}}]}, 4, res, parts)
    b._apply_chunk({"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 7, "completion_tokens": 2}}, 5, res, parts)
    assert "".join(parts) == "Hello" and res.first_text_ns == 2 and res.text_chunks == 2
    assert res.first_audio_ns == 4 and res.output_audio_s == pytest.approx(0.5)
    assert res.finish_reason == "stop" and res.completion_tokens == 2
    b._apply_chunk({"error": {"message": "bad input", "code": 400}}, 6, res, parts)
    assert res.status == "error" and res.error == "bad input"


async def test_openai_backend_against_local_sse_server():
    """End-to-end SSE client test against a tiny in-process aiohttp server (no GPU)."""
    from aiohttp import web
    import base64

    async def handler(request):
        body = await request.json()
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        rid = body["request_id"]
        for piece in ["a", "b", "c"]:
            await resp.write(f'data: {json.dumps({"id": rid, "choices": [{"delta": {"content": piece}}]})}\n\n'.encode())
            await asyncio.sleep(0.01)
        audio = base64.b64encode(silence_wav(0.25)).decode()
        await resp.write(f'data: {json.dumps({"id": rid, "choices": [{"delta": {"audio": {"id": "x", "data": audio}}}]})}\n\n'.encode())
        await resp.write(f'data: {json.dumps({"id": rid, "choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 3}})}\n\n'.encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        be = OpenAIChatBackend(f"http://127.0.0.1:{port}", model="m", keep_text=True)
        res = await be.run(ModelRequest("rid-1", [{"role": "user", "content": "hi"}], ["text"], ["text", "audio"]))
        await be.close()
    finally:
        await runner.cleanup()
    assert res.status == "ok", res.error
    assert res.text == "abc" and res.completion_tokens == 3
    assert res.output_audio_s == pytest.approx(0.25)
    assert res.sent_ns < res.first_text_ns < res.first_audio_ns <= res.done_ns


def test_agent_parser_handles_real_model_replies():
    """Replies recorded from MiniCPM-o 4.5 on 2026-11-11 (Spark smoke test s2)."""
    from omni_research.tracing.agent_harness import _normalize_args, parse_actions

    a = parse_actions('{"tool": "calculator", "args": "18*22-4"} {"final": "392"}')
    assert a["calls"] == [("calculator", "18*22-4")] and a["final"] == "392"
    assert _normalize_args("calculator", "18*22-4") == {"expression": "18*22-4"}
    b = parse_actions('{"inventory": {"item": "steel bolts", "warehouse": "east"}, '
                      '"inventory": {"item": "steel bolts", "warehouse": "south"}}')
    assert [c[1]["warehouse"] for c in b["calls"]] == ["east", "south"]  # duplicate keys kept
    c = parse_actions('{"inventory": "steel bolts, east", "inventory": "steel bolts, south"}')
    assert _normalize_args("inventory", c["calls"][1][1]) == {"item": "steel bolts", "warehouse": "south"}
    d = parse_actions('{"tool": "convert", "args": {"value": 97, "from_unit": "miles", "to_unit": "km"}}')
    assert d["calls"][0][0] == "convert" and d["final"] is None
    assert parse_actions("no json here") == {"calls": [], "final": None}
    with pytest.raises(ValueError):
        _normalize_args("convert", "97 miles")


async def test_agent_parallel_tool_calls_are_traced(tmp_path):
    from omni_research.replay.backends import RequestResult
    from omni_research.tracing.agent_harness import AgentConfig, AgentTask, ToolBox, make_knowledge_base, run_agent_session
    import random as _r

    replies = iter(['{"inventory": {"item": "oak planks", "warehouse": "north"}, "inventory": {"item": "oak planks", "warehouse": "east"}}',
                    '{"final": "The total is __ANS__"}'])
    kb = make_knowledge_base(7)
    ans = kb["oak planks"]["north"] + kb["oak planks"]["east"]

    class Scripted:
        name = "scripted"

        async def run(self, req):
            t = time.monotonic_ns()
            return RequestResult(req.request_id, "scripted", status="ok", sent_ns=t, first_text_ns=t + 1, done_ns=t + 2,
                                 text=next(replies).replace("__ANS__", str(ans)))

        async def close(self):
            pass

    tc = TraceCollector(tmp_path / "t.jsonl", source_kind="observed", generator="test", content_policy="public")
    task = AgentTask("t1", "inventory_sum", "How many oak planks in north and east?", float(ans))
    out = await run_agent_session(Scripted(), tc, task, ToolBox(kb), AgentConfig(), _r.Random(0), [])
    tc.close()
    assert out["success"] and out["tool_calls"] == 2
    tr = load_trace(tmp_path / "t.jsonl")
    assert validate_trace(tr).ok
    reqs = tr.by_kind("model_request")
    tools_ = tr.by_kind("tool_call")
    assert len(tools_) == 2 and sorted(reqs[1].depends_on) == sorted(t.span_id for t in tools_)
