"""R1 report on real structures: traces from the duplex client against the CPU fake server, request records
from the replay driver's own record builder, unit reports shaped like sglang-omni's duplex_metrics.unit_report.
No GPU, no network; none of these numbers is a measurement."""

import importlib.util
import json
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
from aiohttp.test_utils import TestServer

from omni_research.duplex.audio import write_wav
from omni_research.duplex.fake_server import build_app
from omni_research.duplex.phase import run_phase
from omni_research.duplex.report import duplex_phase_metrics
from omni_research.replay.backends import RequestResult
from omni_research.replay.drivers import _record
from omni_research.replay.payloads import AssetPool, Workload, build_request

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ppv_r1_report", ROOT / "scripts/analysis/ppv_r1_report.py")
r1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1)
FAST = 0.05
STAGES = ("perception", "thinker", "talker", "speech")


def unit_report(session_id, seq, timings, outputs):
    """One unit as sglang_omni.profiler.duplex_metrics.unit_report returns it (the keys the analysis reads)."""
    stages = []
    for name in STAGES:
        t = timings.get(name)
        if t is None:
            stages.append({"stage": name, "timing_status": "bypassed", "queue_ms": None, "service_ms": None,
                           "handoff_from_previous_ms": None, "bypass_reason": "generation_bypassed"})
        else:
            stages.append({"stage": name, "timing_status": "observed", "queue_ms": t[0], "service_ms": t[1],
                           "handoff_from_previous_ms": None if name == "perception" else 1.0, "bypass_reason": None})
    covered = sum(t[0] + t[1] for t in timings.values() if t) + sum(1.0 for n in STAGES[1:] if timings.get(n))
    return {"request_id": f"req-{seq}", "session_id": session_id, "session_open_index": 0, "input_seq": seq,
            "ready_to_finished_ms": covered + 2.0, "ready_to_first_output_ms": None, "stages": stages,
            "outputs": [{"emitted_timestamp_ns": 1, "kind": "data"}] * outputs}


def fake_duplex_metrics(monkeypatch):
    """Stand-in for sglang_omni.profiler.duplex_metrics: the test writes finished unit reports as events."""
    mod = types.ModuleType("sglang_omni.profiler.duplex_metrics")
    mod.build_duplex_session_metrics = lambda events: {"units": [e["unit"] for e in events if "unit" in e],
                                                       "sessions": [], "summary": {"stages": {}}}
    for name, m in (("sglang_omni", types.ModuleType("sglang_omni")),
                    ("sglang_omni.profiler", types.ModuleType("sglang_omni.profiler")),
                    ("sglang_omni.profiler.duplex_metrics", mod)):
        monkeypatch.setitem(sys.modules, name, m)


def _pool(root):
    pool = root / "assets" / "duplex-v1"
    (pool / "audio16k").mkdir(parents=True)
    rng = np.random.default_rng(1)
    rows = []
    for i in range(2):
        write_wav(pool / "audio16k" / f"q{i}.wav", (rng.standard_normal(24000) * 2000).astype(np.int16), 16000)
        rows.append({"asset_id": f"q{i}", "kind": "audio", "path": f"audio16k/q{i}.wav", "tags": ["agent_question"]})
    (pool / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    syn = root / "assets" / "synthetic-v1"
    (syn / "audio").mkdir(parents=True)
    for i in range(2):
        write_wav(syn / "audio" / f"q{i}.wav", np.zeros(2400, np.int16), 24000)
    (syn / "manifest.jsonl").write_text("".join(json.dumps({"asset_id": f"q{i}", "kind": "audio", "path": f"audio/q{i}.wav",
                                                            "duration_s": 0.1, "tags": ["agent_question"]}) + "\n" for i in range(2)))


async def _duplex_run(root, base_url, block, run="run01", status="ok"):
    exp = root / "experiments" / f"20261118_ppv_r1_duplex_block{block}_{run}"
    (exp / "analysis").mkdir(parents=True)
    eid = f"ppv_r1_duplex_block{block}"
    (exp / "result.json").write_text(json.dumps({"experiment_id": eid, "status": status}))
    runner = types.SimpleNamespace(phase_summaries=[], data_root=root)
    summaries = []
    for name, events in (("duplex_r1_A", False), ("duplex_r1_B", True)):
        ctx = {"phase": name, "seed": 3, "raw_dir": exp / "raw", "exp_dir": exp, "servers": {"s0": base_url}, "runner": runner}
        t0 = time.time()
        out = await run_phase(ctx, {"kind": "plugin", "module": "duplex.phase", "sessions": 1, "duration_s": 12.0,
                                    "lead_in_s": 1.0, "answer_window_s": 2.0, "barge_in_every": 2, "barge_in_offset_s": 0.5,
                                    "pace_scale": FAST})
        summaries.append({"phase": name, "kind": "plugin", "started": t0, "ended": time.time(), **out,
                          **({"profiled": "request_events"} if events else {})})
    (exp / "analysis" / "phases.json").write_text(json.dumps(summaries))
    s = duplex_phase_metrics(exp / "raw" / "duplex_duplex_r1_B")["sessions"][0]
    speak = set(s["speak_units"])
    units = [unit_report(s["session_id"], k, {"perception": (1.0, 5.0), "thinker": (2.0, 40.0),
                                              "talker": (1.0, 20.0) if k in speak else None,
                                              "speech": (1.0, 60.0) if k in speak else None}, 1 if k in speak else 0)
             for k in range(s["units"]["planned"])]
    ev = exp / "raw" / "server_events" / "duplex_r1_B" / "s0"
    ev.mkdir(parents=True)
    (ev / "events_thinker_111.jsonl").write_text("".join(json.dumps({"unit": u}) + "\n" for u in units))
    return exp, speak


def _chat_run(root, block, ttfa_ms):
    exp = root / "experiments" / f"20261118_ppv_r1_chat_ref_block{block}_run01"
    (exp / "raw").mkdir(parents=True)
    (exp / "result.json").write_text(json.dumps({"experiment_id": f"ppv_r1_chat_ref_block{block}", "status": "ok"}))
    wl = Workload.from_file(ROOT / "configs/workloads/w3_audio_speech.json")
    pool = AssetPool(root / "assets" / "synthetic-v1" / "manifest.jsonl")
    for phase in ("w3_c1_A", "w3_c1_B"):
        lines = []
        for i in range(5):
            rid = f"{phase}-{i}"
            req = build_request(wl, random.Random(i), rid, pool)
            t0 = 10_000_000_000 + i * 5_000_000_000
            res = RequestResult(request_id=rid, backend="s0", sent_ns=t0, first_text_ns=t0 + 300_000_000,
                                first_audio_ns=t0 + int(ttfa_ms * 1e6), last_audio_ns=t0 + int(ttfa_ms * 1e6),
                                done_ns=t0 + int((ttfa_ms + 50) * 1e6), text_chars=40 + i, output_audio_s=3.0,
                                completion_tokens=12, audio_chunks=[(t0 + int(ttfa_ms * 1e6), 3.0, 144000)])
            lines.append(json.dumps(_record(req, res, phase=phase, concurrency=1, index=i, worker=0, warmup=i < 1,
                                            workload_id=wl.workload_id), default=str))
        (exp / "raw" / f"requests_{phase}.jsonl").write_text("\n".join(lines) + "\n")
    return exp


async def _build_all(tmp_path):
    _pool(tmp_path)
    srv = TestServer(build_app(unit_delay_s=0.01))
    await srv.start_server()
    try:
        base = str(srv.make_url("")).rstrip("/")
        failed, _ = await _duplex_run(tmp_path, base, 1, run="run00", status="failed")  # an earlier failed attempt
        runs = [await _duplex_run(tmp_path, base, b) for b in (1, 2, 3)]
    finally:
        await srv.close()
    for b, t in ((1, 2000.0), (2, 2200.0), (3, 2400.0)):
        _chat_run(tmp_path, b, t)
    return failed, runs


async def test_r1_report_blocks_pooling_ttfa_parts_stages_and_reference(tmp_path, monkeypatch):
    failed, runs = await _build_all(tmp_path)
    fake_duplex_metrics(monkeypatch)
    rep = r1.build(r1.discover_runs(tmp_path, []))
    assert rep["runs"]["duplex_block1"] == runs[0][0].name and rep["runs"]["chat_ref_block3"].endswith("block3_run01")
    assert any(e["run"] == failed.name and e["status"] == "failed" for e in rep["excluded_runs"])
    a = rep["duplex"]["A"]
    assert set(a["blocks"]) == {"1", "2", "3"} and a["pooled"]["sessions"] == 3
    assert a["pooled"]["turns"] == sum(v["turns"] for v in a["blocks"].values())
    assert a["pooled"]["ttfa_ms"]["n"] == a["pooled"]["answered"] and a["pooled"]["ttfa_ms"]["p99"] is None  # N < 100
    assert a["block_p50"]["ttfa_ms"]["n"] == 3 and a["block_p50"]["ttfa_ms"]["lo"] is not None
    assert a["pooled"]["barge_in"]["scripted"] >= 3 and set(a["pooled"]["playback"]) == {"0ms", "80ms", "250ms"}
    for row in rep["ttfa_turns"]:
        if row["k"] is not None:
            assert abs(row["a_wait_unit_end_ms"] + row["b_listen_ms"] + row["c_service_ms"] - row["ttfa_ms"]) < 1e-6
    st = rep["server_stages"]["pooled"]
    assert st["available"] and st["speaking_from"] == "client audio deltas (joined)"
    assert st["top_stage_speaking_units"] == "speech" and st["top_stage_all_units"] == "thinker"
    assert st["speaking_units"]["units"] == sum(len(s) for _, s in runs)
    assert st["speaking_units"]["stages"]["talker"]["bypassed"] == 0 and st["all_units"]["stages"]["talker"]["bypassed"] > 0
    assert rep["events_overhead"]["relative_difference"] is not None and isinstance(rep["events_overhead"]["flag"], bool)
    c = rep["chat_reference"]["A"]
    assert c["pooled"]["ok"] == 12 and c["pooled"]["requests"] == 12  # 4 measured per block, the warmup left out
    assert abs(c["blocks"]["2"]["ttfa_ms"]["p50"] - 2200.0) < 1e-6 and c["pooled"]["audio_s"]["p50"] == 3.0
    assert {q["asset_id"] for q in rep["side_by_side"]["by_question"]} >= {"q0"}
    assert rep["side_by_side"]["request_response"]["text_chars"]["n"] == 12
    d1 = rep["d1"]["A"]
    assert set(d1["per_block"]) == {"1", "2", "3"} and d1["block1_vs_rest"]["ttfa_p50_ms"]["blocks_2_3_mean"] is not None
    assert "1" in rep["block_notes"] and len(rep["semantics"]) == 3
    assert "TTFA parts" in r1.markdown(rep)


async def test_r1_server_stages_are_unavailable_without_sglang_omni_never_zero(tmp_path, monkeypatch):
    await _build_all(tmp_path)
    monkeypatch.setitem(sys.modules, "sglang_omni.profiler.duplex_metrics", None)  # import fails like on the Mac
    rep = r1.build(r1.discover_runs(tmp_path, []))
    st = rep["server_stages"]["pooled"]
    assert st["available"] is False and "not importable" in st["reason"] and "speaking_units" not in st
    assert all(v["available"] is False for v in rep["server_stages"]["per_block"].values())
