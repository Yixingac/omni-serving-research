"""Duplex client and plugin phase against the CPU fake server (protocol flow, admission, failures, files)."""

import json
import types

import numpy as np
from aiohttp.test_utils import TestServer

from omni_research.duplex.audio import write_wav
from omni_research.duplex.client import run_session
from omni_research.duplex.fake_server import build_app
from omni_research.duplex.metrics import session_metrics
from omni_research.duplex.phase import run_phase
from omni_research.duplex.script import SAMPLE_RATE, Clip, build_script

FAST = 0.05  # 80 ms packets every 4 ms: protocol tests only, never a measurement


def _clip():
    rng = np.random.default_rng(1)
    return Clip("q0", (rng.standard_normal(int(1.5 * SAMPLE_RATE)) * 2000).astype(np.int16), "a question")


async def _server(**kw):
    srv = TestServer(build_app(unit_delay_s=0.01, **kw))
    await srv.start_server()
    return srv


async def test_one_session_runs_the_protocol_and_gets_a_reply(tmp_path):
    srv = await _server()
    try:
        script = build_script([_clip()], duration_s=5.0, seed=3, lead_in_s=1.0, answer_window_s=2.0)
        res = await run_session(str(srv.make_url("/v1/realtime")), script, trace_path=tmp_path / "s.jsonl",
                                session_update={"output_modalities": ["audio"]}, audio_path=tmp_path / "out.wav",
                                pace_scale=FAST)
        assert res.status == "ok", res.reason
        assert res.packets_sent == res.packets_planned == 63 and res.closed_reason == "client_closed"
        recs = [json.loads(x) for x in (tmp_path / "s.jsonl").read_text().splitlines()]
        types_seen = {r.get("type") for r in recs if r["d"] == "recv"}
        assert {"session.created", "session.updated", "sglang.input_audio.accepted", "sglang.unit.done",
                "response.created", "response.output_audio.delta", "response.done", "sglang.input_audio.drained",
                "session.closed"} <= types_seen
        assert all("delta" not in r.get("ev", {}) for r in recs if r.get("type") == "response.output_audio.delta")
        m = session_metrics(recs, script.to_json())
        assert m["units"]["planned"] == 5 and m["units"]["done"] == 5 and m["units"]["empty_end_units"] == 1
        assert m["turns"]["answered"] == 1 and m["units"]["decisions"]["speak"] >= 1
        assert (tmp_path / "out.wav").stat().st_size > 44
    finally:
        await srv.close()


async def test_a_full_server_denies_with_503_and_the_client_reports_it(tmp_path):
    srv = await _server(max_sessions=1)
    try:
        url = str(srv.make_url("/v1/realtime"))
        long = build_script([_clip()], duration_s=8.0, seed=1, lead_in_s=1.0, answer_window_s=2.0)
        import asyncio

        first = asyncio.create_task(run_session(url, long, trace_path=tmp_path / "a.jsonl", session_update={},
                                                pace_scale=FAST))
        await asyncio.sleep(0.1)
        second = await run_session(url, long, trace_path=tmp_path / "b.jsonl", session_update={}, pace_scale=FAST,
                                   connect_retries=1, connect_backoff_s=0.01)
        assert second.status == "denied" and "503" in second.reason and second.admission_attempts == 2
        assert (await first).status == "ok"
    finally:
        await srv.close()


async def test_a_fatal_server_error_fails_the_session_and_is_recorded(tmp_path):
    srv = await _server(fatal_after_units=2)
    try:
        script = build_script([_clip()], duration_s=6.0, seed=1, lead_in_s=1.0, answer_window_s=2.0)
        res = await run_session(str(srv.make_url("/v1/realtime")), script, trace_path=tmp_path / "s.jsonl",
                                session_update={}, pace_scale=FAST, drain_timeout_s=5, close_timeout_s=2)
        assert res.status == "failed" and res.errors and res.errors[0]["fatal"]
        assert res.packets_sent < res.packets_planned or res.reason
    finally:
        await srv.close()


async def test_a_refused_append_stops_the_input_instead_of_piling_up_errors(tmp_path):
    srv = await _server(reject_after_packets=20)
    try:
        script = build_script([_clip()], duration_s=6.0, seed=1, lead_in_s=1.0, answer_window_s=2.0)
        res = await run_session(str(srv.make_url("/v1/realtime")), script, trace_path=tmp_path / "s.jsonl",
                                session_update={}, pace_scale=FAST, drain_timeout_s=3, close_timeout_s=3)
        assert res.status == "failed" and "buffer_overflow" in res.reason
        assert res.packets_sent <= 22 and len(res.errors) <= 3
    finally:
        await srv.close()


async def test_plugin_phase_writes_traces_metrics_and_skips_after_a_failure(tmp_path):
    srv = await _server(max_sessions=2)
    pool = tmp_path / "pool"
    (pool / "audio16k").mkdir(parents=True)
    write_wav(pool / "audio16k" / "q0.wav", _clip().samples, SAMPLE_RATE)
    (pool / "manifest.jsonl").write_text(json.dumps({"asset_id": "q0", "kind": "audio", "path": "audio16k/q0.wav",
                                                     "tags": ["agent_question"], "transcript": "a question"}) + "\n")
    try:
        base = str(srv.make_url("")).rstrip("/")
        runner = types.SimpleNamespace(phase_summaries=[], data_root=tmp_path)
        ctx = {"phase": "duplex_s2", "seed": 7, "raw_dir": tmp_path / "raw", "exp_dir": tmp_path,
               "servers": {"s0": base}, "runner": runner}
        ph = {"kind": "plugin", "module": "duplex.phase", "sessions": 2, "duration_s": 5.0, "lead_in_s": 1.0,
              "answer_window_s": 2.0, "pace_scale": FAST, "stagger_s": 0.05, "save_audio_sessions": 1,
              "asset_pool": "pool"}
        (tmp_path / "assets").mkdir()
        (tmp_path / "assets" / "pool").symlink_to(pool)
        out = await run_phase(ctx, ph)
        d = out["duplex"]
        assert d["sessions"] == 2 and d["sessions_failed"] == 0 and d["session_status"] == ["ok", "ok"]
        assert d["unit_lag_ms"]["n"] == 10 and d["ttfa_ms"]["p99"] is None  # too few samples for a P99
        assert out["capabilities_status"] == 200
        files = {p.name for p in (tmp_path / "raw" / "duplex_duplex_s2").iterdir()}
        assert {"session_00.jsonl", "session_01.jsonl", "scripts.json", "metrics.json", "capabilities.json",
                "session_00_output.wav"} <= files and "session_01_output.wav" not in files
        runner.phase_summaries.append({"phase": "duplex_s2", "duplex": {"sessions_failed": 1}})
        skipped = await run_phase({**ctx, "phase": "duplex_s4"}, {**ph, "sessions": 4, "skip_if_failed": ["duplex_s2"]})
        assert skipped["skipped"] is True and "duplex_s2" in skipped["reason"]
        runner.phase_summaries[-1] = {"phase": "duplex_s2", "duplex": {"sessions_failed": 0, "unit_miss_rate": 0.02}}
        assert (await run_phase({**ctx, "phase": "duplex_s4"}, {**ph, "sessions": 4, "skip_if_failed": ["duplex_s2"],
                                                                "stop_if_unit_miss_above": 0.01}))["skipped"] is True
    finally:
        await srv.close()


async def test_report_recomputes_metrics_from_raw_traces_and_marks_missing_server_events(tmp_path):
    from omni_research.duplex.report import pool_by_phase, summarize_run

    srv = await _server(max_sessions=2)
    exp = tmp_path / "exp"
    pool_dir = tmp_path / "assets" / "duplex-v1"
    (pool_dir / "audio16k").mkdir(parents=True)
    write_wav(pool_dir / "audio16k" / "q0.wav", _clip().samples, SAMPLE_RATE)
    (pool_dir / "manifest.jsonl").write_text(json.dumps({"asset_id": "q0", "kind": "audio", "path": "audio16k/q0.wav",
                                                         "tags": ["agent_question"]}) + "\n")
    try:
        runner = types.SimpleNamespace(phase_summaries=[], data_root=tmp_path)
        ctx = {"phase": "duplex_n2", "seed": 1, "raw_dir": exp / "raw", "exp_dir": exp,
               "servers": {"s0": str(srv.make_url("")).rstrip("/")}, "runner": runner}
        import time

        t0 = time.time()
        await run_phase(ctx, {"kind": "plugin", "module": "duplex.phase", "sessions": 2, "duration_s": 5.0,
                              "lead_in_s": 1.0, "answer_window_s": 2.0, "pace_scale": FAST, "stagger_s": 0.0})
        (exp / "analysis").mkdir(parents=True)
        (exp / "analysis" / "phases.json").write_text(json.dumps([{"phase": "duplex_n2", "kind": "plugin",
                                                                    "started": t0, "ended": time.time() + 1}]))
        (exp / "raw" / "cpu_samples.jsonl").write_text(json.dumps({"t_wall_ns": int((t0 + 0.1) * 1e9), "system_cpu_percent": 12.0,
                                                                   "groups": {"s0": {"cpu_percent": 150.0, "by_pid": {"1": 99.0}},
                                                                              "client": {"cpu_percent": 20.0}}}) + "\n")
        run = summarize_run(exp)
        ph = run["phases"]["duplex_n2"]
        assert ph["duplex"]["sessions"] == 2 and ph["duplex"]["unit_lag_ms"]["n"] == 10
        assert ph["server_units"]["available"] is False and "without request events" in ph["server_units"]["reason"]
        assert ph["cpu"]["busiest_server_process_cpu_percent"]["max"] == 99.0 and ph["gpu"] == []
        pooled = pool_by_phase([run], [exp, exp])  # the same run twice: counts double
        assert pooled["duplex_n2"]["sessions"] == 4
    finally:
        await srv.close()


async def test_camera_frames_go_out_just_before_the_audio_packet_that_covers_them(tmp_path):
    from omni_research.duplex.phase import synthetic_frames

    srv = await _server()
    try:
        frames = synthetic_frames(3.0, 1.0, seed=4)
        assert [t for t, _ in frames] == [0.0, 1000.0, 2000.0] and all(b[:2] == b"\xff\xd8" for _, b in frames)
        assert synthetic_frames(3.0, 1.0, seed=4) == frames  # deterministic
        script = build_script([_clip()], duration_s=4.0, seed=1, lead_in_s=0.5, answer_window_s=1.0)
        res = await run_session(str(srv.make_url("/v1/realtime")), script, trace_path=tmp_path / "s.jsonl",
                                session_update={}, frames=frames, pace_scale=FAST)
        assert res.status == "ok"
        sends = [json.loads(x) for x in (tmp_path / "s.jsonl").read_text().splitlines() if '"d": "send"' in x]
        order = [(s["type"], s.get("t_ms", s.get("t_start_ms"))) for s in sends
                 if s["type"] in ("sglang.input_image.append", "input_audio_buffer.append")]
        for i, (typ, t) in enumerate(order):
            if typ == "sglang.input_image.append":
                nxt = order[i + 1]
                assert nxt[0] == "input_audio_buffer.append" and nxt[1] <= t < nxt[1] + 80
        recv = [json.loads(x) for x in (tmp_path / "s.jsonl").read_text().splitlines() if '"d": "recv"' in x]
        assert sum(1 for r in recv if r["type"] == "sglang.input_image.accepted") == 3
    finally:
        await srv.close()
