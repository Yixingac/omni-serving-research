"""R2 verdicts on real structures: traces from the duplex client against the CPU fake server (units made late by
shifting their sglang.unit.done records where a test needs misses), phase summaries as the runner writes them,
unit reports shaped like sglang-omni's duplex_metrics.unit_report. No GPU, no network, no measurement."""

import importlib.util
import json
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import numpy as np
from aiohttp.test_utils import TestServer

from omni_research.duplex.audio import write_wav
from omni_research.duplex.fake_server import build_app
from omni_research.duplex.phase import run_phase
from omni_research.duplex.report import duplex_phase_metrics

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ppv_r2_report", ROOT / "scripts/analysis/ppv_r2_report.py")
r2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r2)
FAST = 0.05
STAGES = ("perception", "thinker", "talker", "speech")
PIDS = {"thinker": "201", "speech": "202"}


def unit_report(session_id, seq, thinker_queue_ms):
    timings = {"perception": (1.0, 5.0), "thinker": (thinker_queue_ms, 40.0), "talker": (1.0, 20.0), "speech": (1.0, 60.0)}
    stages = [{"stage": n, "timing_status": "observed", "queue_ms": timings[n][0], "service_ms": timings[n][1],
               "handoff_from_previous_ms": None if n == "perception" else 1.0, "bypass_reason": None} for n in STAGES]
    total = sum(q + s for q, s in timings.values()) + 3.0
    return {"request_id": f"req-{session_id}-{seq}", "session_id": session_id, "session_open_index": 0, "input_seq": seq,
            "ready_to_finished_ms": total, "stages": stages, "outputs": []}


def fake_duplex_metrics(monkeypatch):
    mod = types.ModuleType("sglang_omni.profiler.duplex_metrics")
    mod.build_duplex_session_metrics = lambda events: {"units": [e["unit"] for e in events if "unit" in e],
                                                       "sessions": [], "summary": {}}
    for name, m in (("sglang_omni", types.ModuleType("sglang_omni")),
                    ("sglang_omni.profiler", types.ModuleType("sglang_omni.profiler")),
                    ("sglang_omni.profiler.duplex_metrics", mod)):
        monkeypatch.setitem(sys.modules, name, m)


def _pool(root):
    pool = root / "assets" / "duplex-v1"
    (pool / "audio16k").mkdir(parents=True)
    write_wav(pool / "audio16k" / "q0.wav", (np.random.default_rng(1).standard_normal(24000) * 2000).astype(np.int16), 16000)
    (pool / "manifest.jsonl").write_text(json.dumps({"asset_id": "q0", "kind": "audio", "path": "audio16k/q0.wav",
                                                     "tags": ["agent_question"]}) + "\n")


def _shift_unit_done(phase_dir, ms):
    for t in phase_dir.glob("session_*.jsonl"):
        recs = [json.loads(x) for x in t.read_text().splitlines()]
        for r in recs:
            if r.get("d") == "recv" and r.get("type") == "sglang.unit.done":
                r["t_mono_ns"] += int(ms * 1e6)
        t.write_text("".join(json.dumps(r) + "\n" for r in recs))


async def _arm_run(root, base_url, arm, block, levels, *, slow=(), skip=(), denial=(), gpu_util=95.0, server_args=None,
                   power_w=200.0, oom_level=None):
    exp = root / "experiments" / f"20261119_ppv_r2_{arm}_block{block}_run01"
    (exp / "analysis").mkdir(parents=True)
    (exp / "raw").mkdir(parents=True)
    (exp / "result.json").write_text(json.dumps({"experiment_id": f"ppv_r2_{arm}_block{block}", "status": "ok",
                                                 "server_ready_s": {"s0": 1.0}}))
    (exp / "config.json").write_text(json.dumps({"servers": [{"args": server_args or ["--variant", "session"]}]}))
    runner = types.SimpleNamespace(phase_summaries=[], data_root=root)
    summaries, cpu, gpu, log_lines = [], [], [], ["2026-11-19 00:00:00,000 [INFO] start"]
    for lv in levels:
        name = f"duplex_n{lv}"
        if lv in skip:
            summaries.append({"phase": name, "kind": "plugin", "skipped": True, "reason": "earlier phase failed"})
            continue
        ctx = {"phase": name, "seed": 5, "raw_dir": exp / "raw", "exp_dir": exp, "servers": {"s0": base_url}, "runner": runner}
        t0 = time.time()
        out = await run_phase(ctx, {"kind": "plugin", "module": "duplex.phase", "sessions": lv, "duration_s": 5.0,
                                    "lead_in_s": 1.0, "answer_window_s": 2.0, "pace_scale": FAST, "stagger_s": 0.0,
                                    "save_audio_sessions": 0})
        t1 = time.time()
        d = exp / "raw" / f"duplex_{name}"
        if lv in slow:
            _shift_unit_done(d, 1500.0)
        if lv in denial:
            out["duplex"]["sessions_with_admission_denials"] = 1
        summaries.append({"phase": name, "kind": "plugin", "started": t0, "ended": t1, "profiled": "request_events", **out})
        ev = exp / "raw" / "server_events" / name / "s0"
        ev.mkdir(parents=True)
        units = [unit_report(s["session_id"], k, 2.0 * lv) for s in duplex_phase_metrics(d)["sessions"]
                 for k in range(s["units"]["planned"])]
        (ev / f"events_thinker_{PIDS['thinker']}.jsonl").write_text("".join(json.dumps({"unit": u}) + "\n" for u in units))
        (ev / f"events_speech_{PIDS['speech']}.jsonl").write_text("")
        for k in range(3):
            ts = t0 + (t1 - t0) * (k + 1) / 4  # inside this level's window only
            cpu.append({"t_wall_ns": int(ts * 1e9), "system_cpu_percent": 20.0,
                        "groups": {"s0": {"cpu_percent": 130.0, "by_pid": {PIDS["thinker"]: 99.0, PIDS["speech"]: 30.0}},
                                   "client": {"cpu_percent": 5.0}}})
            for g in range(len([a for a in (server_args or []) if a == "--thinker.gpu"]) + 1):
                stamp = datetime.fromtimestamp(ts).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
                gpu.append(f"{stamp}, {3 + g}, {gpu_util if g == 0 else 40.0}, 30000, {power_w}")
        if lv == oom_level:  # one OOM inside this level's window, logged the way the server logs it
            mid = datetime.fromtimestamp((t0 + t1) / 2)
            log_lines.append(mid.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3] + " [ERROR] sglang_omni.serve.realtime.adapters: "
                             "Realtime session output reader failed")
            log_lines.append("RuntimeError: CUDA out of memory. Tried to allocate 114.00 MiB. GPU 0 has a total capacity of 47.40 GiB")
    (exp / "analysis" / "phases.json").write_text(json.dumps(summaries))
    (exp / "raw" / "cpu_samples.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cpu))
    (exp / "raw" / "gpu_samples.csv").write_text("timestamp, index, utilization.gpu, memory.used, power.draw\n" + "\n".join(gpu) + "\n")
    (exp / "logs").mkdir(exist_ok=True)
    (exp / "logs" / "server_s0.log").write_text("\n".join(log_lines) + "\n")
    return exp


async def _server(**kw):
    srv = TestServer(build_app(unit_delay_s=0.005, **kw))
    await srv.start_server()
    return srv, str(srv.make_url("")).rstrip("/")


async def test_capacity_stage_verdict_intervals_and_p99_rule(tmp_path, monkeypatch):
    _pool(tmp_path)
    srv, base = await _server(max_sessions=4)
    try:
        for b in (1, 2, 3):
            await _arm_run(tmp_path, base, "ms4_1gpu", b, [1, 2, 4], slow={4})
    finally:
        await srv.close()
    fake_duplex_metrics(monkeypatch)
    rep = r2.build(r2.discover_runs(tmp_path, []))
    a = rep["arms"]["ms4_1gpu"]
    by = {e["level"]: e for e in a["levels"]}
    assert by[1]["meets_serving_target"] and by[2]["meets_serving_target"] and not by[4]["meets_serving_target"]
    assert by[4]["unit_miss_rate"] == 1.0 and by[4]["all_blocks_ran"]
    assert a["serving_capacity"] == {"capacity": 2, "bounded": True, "levels_meeting": [1, 2], "statement": "2"}
    assert a["deadline_capacity"]["1s"]["capacity"] == 4 and not a["deadline_capacity"]["1s"]["bounded"]
    fl = a["first_limiting_stage"]
    assert fl["verdict"] == "thinker" and fl["base_level"] == 1 and fl["target_level"] == 4
    assert abs(fl["increase_queue_plus_service_p50_ms"]["thinker"] - 6.0) < 1e-9
    assert fl["increase_queue_plus_service_p50_ms"]["speech"] == 0.0
    assert fl["resource"] == "GPU-bound" and fl["stage_gpu_util_mean"] == 95.0 and fl["stage_process_cpu_p90"] is None
    assert by[4]["pooled"]["ttfa_ms"]["p99"] is None and by[4]["pooled"]["unit_lag_ms"]["n"] < 100
    assert by[2]["block_p50"]["unit_lag_ms"]["n"] == 3 and by[2]["block_p50"]["unit_lag_ms"]["lo"] is not None
    assert a["stop_rule_skips"] == {} and rep["placement"]["expected_arms"] is None
    assert "Serving capacity: 2" in r2.markdown(rep)
    d = by[1]["descriptive"]  # mix-controlled views: speaking / listening unit lags, TTFA parts
    assert d["unit_lag_speaking_ms"]["n"] + d["unit_lag_listening_ms"]["n"] == by[1]["pooled"]["unit_lag_ms"]["n"]
    assert 0 < d["speak_share"] < 1 and d["ttfa_parts"]["answered"] == by[1]["pooled"]["answered"]
    assert set(a["first_reply_after_start"]) == {"1", "2", "3"}
    assert fl["gpu_util_suspect"] is False and fl["resource_input_note"] is None
    assert abs(fl["descriptive_increase_speaking_units_only_ms"]["thinker"] - 6.0) < 1e-9
    assert by[1]["pooled"]["oom_events"] == 0


async def test_capacity_is_the_unbroken_run_from_the_lowest_level(tmp_path, monkeypatch):
    """Mutation target: a level above a failing one never counts, even if it meets the target itself."""
    _pool(tmp_path)
    srv, base = await _server(max_sessions=4)
    try:
        for b in (1, 2, 3):
            await _arm_run(tmp_path, base, "ms4_1gpu", b, [1, 2, 4], denial={2} if b == 2 else set())
    finally:
        await srv.close()
    fake_duplex_metrics(monkeypatch)
    a = r2.build(r2.discover_runs(tmp_path, []))["arms"]["ms4_1gpu"]
    by = {e["level"]: e for e in a["levels"]}
    assert not by[2]["meets_serving_target"] and by[2]["per_block"]["2"]["denials"] == 1 and by[4]["meets_serving_target"]
    assert a["serving_capacity"]["capacity"] == 1 and a["serving_capacity"]["levels_meeting"] == [1]


async def test_a_level_skipped_by_the_stop_rule_in_one_block_never_meets_the_target(tmp_path, monkeypatch):
    _pool(tmp_path)
    srv, base = await _server(max_sessions=8)
    try:
        for b in (1, 2, 3):
            await _arm_run(tmp_path, base, "ms8_2gpu", b, [4, 8], skip={8} if b == 3 else set(), gpu_util=50.0,
                           server_args=["--variant", "session", "--thinker.gpu", "1"])
    finally:
        await srv.close()
    fake_duplex_metrics(monkeypatch)
    a = r2.build(r2.discover_runs(tmp_path, []))["arms"]["ms8_2gpu"]
    by = {e["level"]: e for e in a["levels"]}
    assert by[8]["blocks_ran"] == [1, 2] and not by[8]["meets_serving_target"]
    assert a["stop_rule_skips"] == {"8": {"3": "earlier phase failed"}}
    assert a["serving_capacity"]["statement"] == "4"
    fl = a["first_limiting_stage"]
    assert fl["base_level"] == 4 and fl["target_level"] == 8 and fl["stage_gpu_position"] == 1
    assert fl["stage_gpu_index_per_block"] == ["4", "4"] and fl["resource"] == "neither saturated (uncertain)"


async def test_smoke_runs_decide_the_placement_and_failures_are_classified(tmp_path):
    _pool(tmp_path)
    bad = tmp_path / "experiments" / "20261119_ppv_r2_smoke_ms8_1gpu_run01"
    (bad / "logs").mkdir(parents=True)
    (bad / "result.json").write_text(json.dumps({"experiment_id": "ppv_r2_smoke_ms8_1gpu", "status": "failed",
                                                 "error": "server s0 not healthy"}))
    (bad / "logs" / "server_s0.log").write_text("loading\nValueError: max_sessions needs 10.12 GiB of thinker KV, "
                                                "more than gpu_memory_fraction=0.52 of the card\n")
    good = tmp_path / "experiments" / "20261119_ppv_r2_smoke_ms8_2gpu_run01"
    (good / "analysis").mkdir(parents=True)
    (good / "result.json").write_text(json.dumps({"experiment_id": "ppv_r2_smoke_ms8_2gpu", "status": "ok",
                                                  "server_ready_s": {"s0": 300.0}}))
    srv, base = await _server(max_sessions=8)
    try:
        runner = types.SimpleNamespace(phase_summaries=[], data_root=tmp_path)
        ctx = {"phase": "duplex_smoke_n8", "seed": 1, "raw_dir": good / "raw", "exp_dir": good, "servers": {"s0": base},
               "runner": runner}
        out = await run_phase(ctx, {"kind": "plugin", "module": "duplex.phase", "sessions": 8, "duration_s": 4.0,
                                    "lead_in_s": 1.0, "answer_window_s": 1.0, "pace_scale": FAST, "stagger_s": 0.0,
                                    "save_audio_sessions": 0})
        (good / "analysis" / "phases.json").write_text(json.dumps([{"phase": "duplex_smoke_n8", "kind": "plugin", **out}]))
        await _arm_run(tmp_path, base, "ms4_1gpu", 1, [1])
    finally:
        await srv.close()
    rep = r2.build(r2.discover_runs(tmp_path, []))
    assert rep["smoke"]["ms8_1gpu"]["passed"] is False and rep["smoke"]["ms8_1gpu"]["failure"]["class"] == "memory"
    assert "max_sessions needs" in rep["smoke"]["ms8_1gpu"]["failure"]["evidence"]
    assert rep["smoke"]["ms8_2gpu"]["passed"] is True and rep["smoke"]["ms8_2gpu"]["sessions"] == 8
    assert rep["placement"]["expected_arms"] == ["ms4_1gpu", "ms8_2gpu"] and rep["placement"]["consistent"] is False
    a = rep["arms"]["ms4_1gpu"]  # one block only: nothing can meet the target
    assert a["serving_capacity"]["statement"] == "below the lowest level"
    assert {e["level"]: e["all_blocks_ran"] for e in a["levels"]} == {1: False, 2: False, 4: False}
    assert a["first_limiting_stage"]["verdict"] is None


async def test_deadline_misses_from_listening_are_attributed_to_the_model(tmp_path):
    _pool(tmp_path)
    srv, base = await _server(max_sessions=8, reply_units=0)  # the toy model never speaks
    try:
        await _arm_run(tmp_path, base, "ms8_1gpu", 1, [1])
    finally:
        await srv.close()
    a = r2.build(r2.discover_runs(tmp_path, []))["arms"]["ms8_1gpu"]
    mb = a["model_behaviour_at_lowest_level"]
    assert mb["level"] == 1 and mb["1s"]["missed"] == mb["1s"]["turns"] > 0 and mb["1s"]["share"] == 1.0
    assert mb["1s"]["statement"] == "主要是模型的轮次决策，不是服务容量"


def test_gpu_window_flags_zero_utilisation_while_the_card_draws_power():
    from omni_research.duplex.report import gpu_window

    busy = [{"t": 10.0 + k, "gpu": "0", "util": 0.0 if k < 8 else 30.0, "mem_mib": 1.0, "power_w": 150.0} for k in range(10)]
    w = gpu_window(busy, 0.0, 100.0)["0"]
    assert w["util_zero_share"] == 0.8 and w["power_mean_w"] == 150.0 and w["util_suspect"] is True
    idle = [{**x, "power_w": 40.0} for x in busy]  # zero readings at idle power are plausible
    assert gpu_window(idle, 0.0, 100.0)["0"]["util_suspect"] is False
    steady = [{**x, "util": 60.0} for x in busy]
    assert gpu_window(steady, 0.0, 100.0)["0"]["util_suspect"] is False


def test_gpu_and_log_stamps_use_the_runs_utc_offset_not_this_machines(tmp_path, monkeypatch):
    import os

    from omni_research.duplex.report import load_gpu_samples_utc, oom_events

    exp = tmp_path / "run"
    (exp / "raw").mkdir(parents=True)
    (exp / "logs").mkdir()
    (exp / "manifest.json").write_text(json.dumps({"utc_offset": "-0500"}))
    (exp / "raw" / "gpu_samples.csv").write_text("timestamp, index, utilization.gpu, memory.used, power.draw\n"
                                                 "2026/11/19 00:35:00.000, 0, 50, 1000, 120\n")
    (exp / "logs" / "server_s0.log").write_text("2026-11-19 00:35:00,000 [ERROR] x\nRuntimeError: CUDA out of memory.\n"
                                                "[W1119 00:36:00.123456789 CUDACachingAllocator.cpp:508] expandable_segments: "
                                                "memory mapping failed with OOM on device 0\n")
    want = datetime(2026, 11, 19, 5, 35, 0, tzinfo=__import__("datetime").timezone.utc).timestamp()
    monkeypatch.setenv("TZ", "Asia/Tokyo")  # the analysis machine is somewhere else
    time.tzset()
    try:
        assert load_gpu_samples_utc(exp)[0]["t"] == want
        ev = oom_events(exp)
        assert [e["t"] for e in ev] == [want, want + 60.123456] and "OOM" in ev[1]["line"]
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()
    assert os.environ.get("TZ") is None


async def test_suspect_gpu_utilisation_is_flagged_and_ooms_are_counted_per_level(tmp_path, monkeypatch):
    _pool(tmp_path)
    srv, base = await _server(max_sessions=4)
    try:
        for b in (1, 2, 3):
            await _arm_run(tmp_path, base, "ms4_1gpu", b, [1, 2, 4], slow={4}, gpu_util=0.0, power_w=150.0, oom_level=4)
    finally:
        await srv.close()
    fake_duplex_metrics(monkeypatch)
    a = r2.build(r2.discover_runs(tmp_path, []))["arms"]["ms4_1gpu"]
    by = {e["level"]: e for e in a["levels"]}
    assert all(by[4]["per_block"][str(b)]["oom_events"] == 1 for b in (1, 2, 3)) and by[4]["pooled"]["oom_events"] == 3
    assert by[2]["pooled"]["oom_events"] == 0 and "CUDA out of memory" in by[4]["per_block"]["1"]["oom_first"]
    fl = a["first_limiting_stage"]
    assert fl["resource"] == "neither saturated (uncertain)"  # the registered test, unchanged
    assert fl["gpu_util_suspect"] is True and "unreliable input" in fl["resource_input_note"]
    assert fl["stage_gpu_power_mean_w"] == [150.0, 150.0, 150.0]
