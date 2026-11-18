"""Runner: a control phase writes a JSON file into the run directory, and server env values may name that directory
({exp_dir}); 实验 10 uses this to switch code2wav's batching between phases without restarting the server."""

import json
import sys

from omni_research.orchestration.experiment import ExperimentRunner

from test_experiment_runner import REPO, free_port, write_workload


def test_control_phase_writes_json_and_the_server_env_names_the_run_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNI_REPO", str(REPO))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("OMNI_PORT", str(free_port()))
    text = write_workload(tmp_path, "w1")
    probe = tmp_path / "env_seen.txt"
    cfg = {"experiment_id": "test-control", "data_root": str(tmp_path / "data"),
           "servers": [{"name": "s0", "env": {"C2W_CONTROL": "{exp_dir}/c2w_control.json"}, "ready_timeout_s": 60,
                        "command": f'printf %s "$C2W_CONTROL" > {probe} && exec {sys.executable} -m omni_research.replay.fake_server '
                                   "--port {port} --startup-s 1"}],
           "runtime": {"env": {"PYTHONPATH": str(REPO / "src")}, "collect_versions": False},
           "phases": [{"kind": "control", "name": "cap4", "path": "{exp_dir}/c2w_control.json",
                       "content": {"max_batch_size": 4, "policy": "fifo"}, "settle_s": 0.1},
                      {"kind": "smoke", "name": "smoke", "workload": text, "requests": 1}]}
    out = tmp_path / "exp"
    res = ExperimentRunner(cfg, out).run()
    assert res["status"] == "ok", res.get("error")
    assert json.loads((out / "c2w_control.json").read_text()) == {"max_batch_size": 4, "policy": "fifo"}
    assert probe.read_text() == str(out / "c2w_control.json")
    assert res["phases"][0]["control"] == {"max_batch_size": 4, "policy": "fifo"}
    assert not list(out.glob("*.tmp"))
