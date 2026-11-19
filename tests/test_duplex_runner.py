"""The experiment runner drives duplex plugin phases end to end against the CPU fake duplex server (no GPU)."""

import json
import socket
import sys
from pathlib import Path

import numpy as np

from omni_research.duplex.audio import write_wav
from omni_research.orchestration.experiment import ExperimentRunner

REPO = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_runner_runs_duplex_phases_with_events_and_skips_after_a_failed_level(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNI_REPO", str(REPO))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("OMNI_PORT", str(free_port()))
    data = tmp_path / "data"
    pool = data / "assets" / "duplex-v1"
    (pool / "audio16k").mkdir(parents=True)
    rng = np.random.default_rng(0)
    write_wav(pool / "audio16k" / "q0.wav", (rng.standard_normal(24000) * 2000).astype(np.int16), 16000)
    (pool / "manifest.jsonl").write_text(json.dumps({"asset_id": "q0", "kind": "audio", "path": "audio16k/q0.wav",
                                                     "tags": ["agent_question"]}) + "\n")
    phase = {"kind": "plugin", "module": "duplex.phase", "duration_s": 5.0, "lead_in_s": 1.0, "answer_window_s": 2.0,
             "pace_scale": 0.05, "stagger_s": 0.0, "save_audio_sessions": 0}
    cfg = {"experiment_id": "test-duplex", "data_root": str(data),
           "servers": [{"name": "s0", "command": f"{sys.executable} -m omni_research.duplex.fake_server --port {{port}} "
                                                 "--max-sessions 2 --unit-delay-s 0.01", "ready_timeout_s": 60}],
           "runtime": {"env": {"PYTHONPATH": str(REPO / "src")}, "collect_versions": False},
           "phases": [{**phase, "name": "duplex_n1", "sessions": 1, "server_events": True},
                      {**phase, "name": "duplex_n4", "sessions": 4, "connect_retries": 0},  # max_sessions 2: 2 refused
                      {**phase, "name": "duplex_n8", "sessions": 8, "skip_if_failed": ["duplex_n4"]}]}
    out = tmp_path / "exp"
    res = ExperimentRunner(cfg, out).run()
    assert res["status"] == "ok", res.get("error")
    phases = {p["phase"]: p for p in res["phases"]}
    one = phases["duplex_n1"]
    assert one["profiled"] == "request_events" and one["duplex"]["session_status"] == ["ok"]
    assert list((out / "raw" / "server_events" / "duplex_n1").rglob("*.jsonl"))
    four = phases["duplex_n4"]["duplex"]
    assert four["sessions_failed"] == 2 and sorted(four["session_status"]).count("denied") == 2
    assert one["duplex"]["mean_open_sessions"] == 1.0 and one["duplex"]["sessions_with_admission_denials"] == 0
    assert phases["duplex_n8"]["skipped"] is True
    assert (out / "raw" / "duplex_duplex_n1" / "metrics.json").exists()
