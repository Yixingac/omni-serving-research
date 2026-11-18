"""Run inventory for metadata.json: ids, status, start and end, restarts and versions, never host names or paths."""

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("run_inventory", HERE / "run_inventory.py")
ri = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ri)


def test_one_run(tmp_path):
    exp = tmp_path / "20261118_exp09_t2_k2_c4_block1_run02"
    exp.mkdir()
    (exp / "result.json").write_text(json.dumps({"experiment_id": "exp09_t2_k2_c4_block1", "status": "ok", "wall_s": 1500.5,
                                                 "server_restarts": [{"server": "s0", "exit_code": -11, "where": "w3b_open rate=23.5",
                                                                      "log_tail": "/users/someone/x"}],
                                                 "phases": [{"phase": "smoke_w3b"}, {"phase": "w3b_closed_A"}]}))
    (exp / "step.log").write_text("2026-11-18T00:39:25-05:00 experiment exp09_t2_k2_c4_block1.json on node-host CUDA_VISIBLE_DEVICES=0\n"
                                  "2026-11-18T00:44:01 starting server s0\n2026-11-18T01:06:57 phase w3l_open done\n")
    (exp / "manifest.json").write_text(json.dumps({"environment": {"research_repo_commit": "af62bb5d3c", "research_repo_dirty": True,
                                                                   "sglang_omni_commit": "520b49ee69", "hostname": "node-host",
                                                                   "packages": {"torch": "2.13.0", "sglang": "0.5.21", "numpy": "2.0"}}}))
    r = ri.inventory_run(exp)
    assert r == {"run_dir": exp.name, "experiment_id": "exp09_t2_k2_c4_block1", "status": "ok", "start": "2026-11-18 00:39:25",
                 "end": "2026-11-18 01:06:57", "wall_s": 1500.5, "phases": ["smoke_w3b", "w3b_closed_A"],
                 "restarts": [{"server": "s0", "exit_code": -11, "where": "w3b_open rate=23.5"}],
                 "research_repo_commit": "af62bb5d3c", "research_repo_dirty": True, "sglang_omni_commit": "520b49ee69",
                 "packages": {"torch": "2.13.0", "sglang": "0.5.21"}}
    assert "node-host" not in json.dumps(r) and "/users/" not in json.dumps(r)


def test_a_run_that_failed_before_writing_its_result(tmp_path):
    exp = tmp_path / "20261118_exp09_t2_k2_c4_block1_run01"
    exp.mkdir()
    (exp / "step.log").write_text("2026-11-18T00:29:58-05:00 experiment x\n2026-11-18T00:38:56 FAILED (port_in_use): server s0 exited\n")
    r = ri.inventory_run(exp)
    assert r["status"] is None and r["start"] == "2026-11-18 00:29:58" and r["end"] == "2026-11-18 00:38:56"
