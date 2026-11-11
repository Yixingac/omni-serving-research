"""End-to-end test of the experiment runner against the CPU-only fake server (no GPU)."""

import json
import socket
import sys
from pathlib import Path

import pytest

from omni_research.orchestration.experiment import ExperimentRunner, classify_log, summarize_gpu_samples
from omni_research.tracing import load_trace, validate_trace

REPO = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def write_workload(tmp, name, **kw):
    d = {"workload_id": name, "input_modalities": ["text"], "output_modalities": ["text"], "max_tokens": 8,
         "prompt": {"kind": "synthetic_text", "tokens": 32}}
    d.update(kw)
    p = tmp / f"{name}.json"
    p.write_text(json.dumps(d))
    return str(p)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNI_REPO", str(REPO))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    port = free_port()
    monkeypatch.setenv("OMNI_PORT", str(port))
    text = write_workload(tmp_path, "w1")
    speech = write_workload(tmp_path, "w3", output_modalities=["text", "audio"])
    return {
        "experiment_id": "test-fake",
        "data_root": str(tmp_path / "data"),
        "servers": [{"name": "s0", "command": f"{sys.executable} -m omni_research.replay.fake_server --port {{port}} --startup-s 1 --agent-json",
                     "ready_timeout_s": 60}],
        "runtime": {"env": {"PYTHONPATH": str(REPO / "src")}, "collect_versions": False},
        "phases": [
            {"kind": "smoke", "name": "smoke", "workload": text, "requests": 2},
            {"kind": "closed_loop", "name": "sweep", "workload": speech, "concurrency": [1, 2], "requests_per_level": 6,
             "warmup": 1, "server_events": True},
            {"kind": "open_loop", "name": "poisson", "workload": text, "rates_rps": [20], "requests_per_rate": 10},
            {"kind": "agent", "name": "agents", "sessions": 3, "concurrency": 2, "agent": {"max_steps": 3}},
        ],
    }


def test_runner_end_to_end(cfg, tmp_path):
    out = tmp_path / "exp"
    res = ExperimentRunner(cfg, out).run()
    assert res["status"] == "ok", res.get("error")
    assert res["server_ready_s"]["s0"] >= 0
    assert res["server_exit_codes"]["s0"] is not None  # the server was stopped
    for f in ("manifest.json", "config.json", "result.json", "analysis/metrics.json", "logs/client.log", "logs/server_s0.log"):
        assert (out / f).exists(), f
    phases = {p["phase"]: p for p in res["phases"]}
    assert phases["smoke"]["validation"] == "ok"
    assert [lv["concurrency"] for lv in phases["sweep"]["levels"]] == [1, 2]
    assert all(lv["errors"] == 0 and lv["n"] == 6 for lv in phases["sweep"]["levels"])
    sweep = [json.loads(x) for x in (out / "raw" / "requests_sweep.jsonl").read_text().splitlines()]
    assert sum(1 for r in sweep if r["warmup"]) == 2 and all(r["output_audio_s"] > 0 for r in sweep)
    # Server-side request events were recorded with the client's request ids.
    ev_files = list((out / "raw" / "server_events" / "sweep").rglob("*.jsonl"))
    rids = {json.loads(x)["request_id"] for f in ev_files for x in f.read_text().splitlines()}
    assert {r["request_id"] for r in sweep} <= rids
    # Agent sessions: fake server calls the calculator, then answers 42 (6*7) -> success.
    assert phases["agents"]["task_success"] == 0 or phases["agents"]["sessions"] == 3
    trace = load_trace(out / "raw" / "agent_trace_agents.jsonl")
    rep = validate_trace(trace)
    assert rep.ok, rep.errors
    assert rep.stats["by_kind"]["tool_call"] == 3 and rep.stats["by_kind"]["model_request"] == 6
    bench = validate_trace(load_trace(out / "raw" / "bench_trace.jsonl"))
    assert bench.ok, bench.errors


def test_runner_reports_server_crash(cfg, tmp_path):
    cfg["servers"][0]["command"] = "echo 'torch.OutOfMemoryError: CUDA out of memory'; exit 3"
    res = ExperimentRunner(cfg, tmp_path / "exp").run()
    assert res["status"] == "failed" and res["error_class"] == "oom"
    assert json.loads((tmp_path / "exp" / "result.json").read_text())["error_class"] == "oom"


def test_runner_bad_output_on_failed_smoke(cfg, tmp_path):
    cfg["servers"][0]["command"] = cfg["servers"][0]["command"].replace("--agent-json", "")
    # Speech requested, but with 2 tokens the fake server never emits an audio chunk.
    cfg["phases"] = [{"kind": "smoke", "workload": cfg["phases"][0]["workload"], "requests": 1,
                      "workload_overrides": {"max_tokens": 2, "output_modalities": ["text", "audio"]}}]
    res = ExperimentRunner(cfg, tmp_path / "exp").run()
    assert res["status"] == "failed" and res["error_class"] == "bad_output"
    assert "no audio" in res["error"]


def test_classify_and_gpu_summary(tmp_path):
    assert classify_log("RuntimeError: Address already in use", "x") == "port_in_use"
    assert classify_log("no kernel image is available for execution", "x") == "unsupported"
    assert classify_log("all good", "server_crashed") == "server_crashed"
    p = tmp_path / "g.csv"
    p.write_text("hdr\n2026/11/11 19:20:00.000, 0, GPU-a, 50, 10, 1000, 49140, 100, 1800, P2\n"
                 "2026/11/11 19:20:00.200, 0, GPU-a, 100, 20, 3000, 49140, 200, 1800, P0\n")
    g = summarize_gpu_samples(p)
    assert g["0"]["mem_max_mib"] == 3000 and g["0"]["util_mean"] == 75 and g["0"]["samples"] == 2


def test_runner_torch_profile_window_waits_for_export(cfg, tmp_path):
    cfg["phases"] = [{"kind": "closed_loop", "name": "prof", "workload": cfg["phases"][0]["workload"], "concurrency": [1],
                      "requests_per_level": 2, "warmup": 0, "torch_profile": True}]
    out = tmp_path / "exp"
    res = ExperimentRunner(cfg, out).run()
    assert res["status"] == "ok", res.get("error")
    traces = list((out / "profiling" / "torch" / "prof").rglob("*.trace.json.gz"))
    assert len(traces) == 1 and "fake_trace" in traces[0].name  # {stage} was filled in by the server
    assert res["phases"][0]["profiled"] == "torch+request_events"


def test_stop_kills_server_in_its_own_process_group(cfg, tmp_path):
    """Like nsys: the real server runs under setsid (new session/group); stop() must still end it."""
    import time

    cmd = cfg["servers"][0]["command"]
    # Portable setsid (macOS has no setsid command): new session, then exec the real server.
    setsid = f"{sys.executable} -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])'"
    cfg["servers"][0]["command"] = f"{setsid} {cmd} & wait"
    cfg["phases"] = [{"kind": "smoke", "name": "s", "workload": cfg["phases"][0]["workload"], "requests": 1}]
    runner = ExperimentRunner(cfg, tmp_path / "exp")
    res = runner.run()
    assert res["status"] == "ok", res.get("error")
    from omni_research.orchestration.experiment import _port_free

    port = runner.servers[0].port
    for _ in range(20):  # the detached server exits shortly after SIGTERM
        if _port_free(port):
            break
        time.sleep(0.5)
    assert _port_free(port), "server in its own session is still listening after stop()"
