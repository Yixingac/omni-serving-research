import json
import sqlite3
import sys
from pathlib import Path

import pytest

from omni_research.analysis import experiment_analysis as EA
from omni_research.analysis.nsys import busy_idle_report, device_activity, nvtx_ranges


def make_nsys_db(path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (deviceId INT, start INT, end INT)")
    con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (deviceId INT, start INT, end INT)")
    con.execute("CREATE TABLE StringIds (id INT, value TEXT)")
    con.execute("CREATE TABLE NVTX_EVENTS (start INT, end INT, text TEXT, textId INT)")
    # Device 0: two kernels overlapping on different streams (0-500 and 300-800) + a copy 900-1000.
    con.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?)", [(0, 0, 500), (0, 300, 800), (1, 0, 100)])
    con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (0, 900, 1000)")
    con.execute("INSERT INTO StringIds VALUES (1, 'decode_step')")
    con.execute("INSERT INTO NVTX_EVENTS VALUES (0, 1000, NULL, 1)")
    con.commit()
    con.close()


def test_nsys_union_not_sum(tmp_path):
    db = tmp_path / "t.sqlite"
    make_nsys_db(db)
    act = device_activity(db)
    rep = busy_idle_report(act, window=(0, 1000))
    assert rep[0]["busy_ns"] == 900  # union of 0-800 and 900-1000
    assert rep[0]["kernel_sum_ns"] == 1100  # naive sum double-counts the overlap
    assert rep[0]["gap_count"] == 1 and rep[0]["longest_gap_ns"] == 100
    assert rep[0]["busy_fraction"] == pytest.approx(0.9)
    assert nvtx_ranges(db) == [{"start": 0, "end": 1000, "name": "decode_step"}]


def test_stage_breakdown_from_server_events():
    ns = 1_000_000
    evs = [
        {"request_id": "r1", "stage": "preprocessing", "event_name": "preprocess_start", "timestamp_ns": 0},
        {"request_id": "r1", "stage": "preprocessing", "event_name": "preprocess_end", "timestamp_ns": 30 * ns},
        {"request_id": "r1", "stage": "coordinator", "event_name": "request_admission", "timestamp_ns": 0},
        {"request_id": "r1", "stage": "thinker", "event_name": "scheduler_queue_enter", "timestamp_ns": 40 * ns},
        {"request_id": "r1", "stage": "thinker", "event_name": "scheduler_prefill_start", "timestamp_ns": 55 * ns},
        {"request_id": "r1", "stage": "thinker", "event_name": "stage_first_stream_chunk_sent", "timestamp_ns": 100 * ns},
        {"request_id": "r1", "stage": "talker", "event_name": "scheduler_queue_enter", "timestamp_ns": 110 * ns},
        {"request_id": "r1", "stage": "talker", "event_name": "scheduler_prefill_start", "timestamp_ns": 300 * ns},
        {"request_id": "r1", "stage": "code2wav", "event_name": "code2wav_first_audio", "timestamp_ns": 450 * ns},
        {"request_id": "r1", "stage": "coordinator", "event_name": "terminal_response", "timestamp_ns": 900 * ns},
    ]
    out = EA.stage_breakdown(evs)
    d = out["per_request"]["r1"]
    assert d["preprocess"] == 30 and d["thinker_queue"] == 15 and d["thinker_prefill_to_first_chunk"] == 45
    assert d["talker_queue"] == 190 and d["admission_to_first_audio"] == 450 and d["admission_to_terminal"] == 900
    assert "encoder" not in d  # no encoder events -> no encoder interval is invented


def test_analyze_fake_experiment(tmp_path, monkeypatch):
    """Analysis of a real runner output (fake server): tables, agent summary, stage breakdown."""
    from test_experiment_runner import REPO, free_port, write_workload
    from omni_research.orchestration.experiment import ExperimentRunner

    monkeypatch.setenv("OMNI_REPO", str(REPO))
    monkeypatch.setenv("OMNI_PORT", str(free_port()))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    wl = write_workload(tmp_path, "w3", output_modalities=["text", "audio"])
    cfg = {"experiment_id": "fake", "data_root": str(tmp_path / "d"),
           "servers": [{"name": "s0", "command": f"{sys.executable} -m omni_research.replay.fake_server --port {{port}} --startup-s 1 --agent-json"}],
           "runtime": {"env": {"PYTHONPATH": str(REPO / "src")}, "collect_versions": False},
           "phases": [{"kind": "closed_loop", "name": "sweep", "workload": wl, "concurrency": [1, 4], "requests_per_level": 12,
                       "warmup": 2, "server_events": True},
                      {"kind": "agent", "name": "ag", "sessions": 4, "concurrency": 2}]}
    out = tmp_path / "exp"
    assert ExperimentRunner(cfg, out).run()["status"] == "ok"
    a = EA.analyze_experiment(out, slo={"ttft_s": 5.0})
    rows = {(r["phase"], r["level"]): r for r in a["table"]}
    assert rows[("sweep", "1")]["n"] == 12 and rows[("sweep", "4")]["errors"] == 0
    assert rows[("sweep", "4")]["ttfa_s_median"] is not None and rows[("sweep", "4")]["goodput_fraction"] == 1.0
    assert a["agents"]["ag"]["valid"] and a["agents"]["ag"]["sessions"] == 4
    split = a["agents"]["ag"]["time_split_s"]
    assert split["model"] > 0 and abs(sum(v for k, v in split.items() if k != "session") - split["session"]) < 1e-6
    assert a["stages"]["sweep"]["requests"] >= 24
