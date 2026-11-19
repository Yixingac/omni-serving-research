"""Experiment manifest: provenance from the orchestrator state and run files, no host paths in the output."""

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("ppv_manifest", HERE / "ppv_manifest.py")
pm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pm)


def _state(out_dir):
    return {"experiments": {
        "exp11_diag_B2": {"status": "SUCCEEDED", "started_at": 1.0},
        "ppv_p0_cap32_block1": {"status": "FAILED", "attempts": 1, "failures": [{"detail": "exit code 1"}],
                                "started_at": 1795059039.98, "ended_at": 1795059056.63, "assigned": {"gpus": 8},
                                "output_dir": "/somewhere/results/experiments/20261118_ppv_p0_cap32_block1_run01",
                                "gpu_seconds": 133.1},
        "ppv_p0_cap32_block1_a2": {"status": "SUCCEEDED", "attempts": 1, "failures": [], "started_at": 1795059400.0,
                                   "ended_at": 1795061200.0, "assigned": {"gpus": 8},
                                   "output_dir": str(out_dir), "gpu_seconds": 14400.0},
    }}


def test_build_reads_state_and_run_files_and_keeps_failed_attempts(tmp_path):
    root = tmp_path / "results"
    run = root / "experiments" / "20261118_ppv_p0_cap32_block1_a2_run01"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"status": "ok", "environment": {
        "research_repo_commit": "bce117a0123456789", "research_repo_dirty": False, "sglang_omni_commit": "520b49ee69385d4a",
        "models": {"s0": {"path": "~/m", "revision": "fb09628e"}}, "slurm": {"SLURM_JOB_ID": "454290", "SLURM_STEP_ID": "131"}}}))
    (run / "result.json").write_text(json.dumps({"status": "ok", "server_restarts": [{"t": 1}]}))
    (run / "config.json").write_text(json.dumps({"runtime_profile": "unicorn_instr"}))
    inc = {"ppv_p0_cap32_block1_a2": {"included": True, "reason": "registered arm A, block 1"}}
    rows = pm.build(_state(f"/nfs/someone/results/experiments/{run.name}"), root, inc)
    assert [r["experiment_id"] for r in rows] == ["ppv_p0_cap32_block1", "ppv_p0_cap32_block1_a2"]  # start order, ppv only
    failed, ok = rows
    assert failed["orchestrator_status"] == "FAILED" and failed["failure"] == "exit code 1" and failed["family"] == "P0"
    assert failed["run_dir"] == "20261118_ppv_p0_cap32_block1_run01" and failed["included"] is None
    assert ok["gpu_hours_allocated"] == 4.0 and ok["slurm_step"] == "131" and ok["research_repo_commit"] == "bce117a012"
    assert ok["model_revision"] == "fb09628e" and ok["server_restarts"] == 1 and ok["included"] is True
    assert ok["runtime_profile"] == "unicorn_instr"
    text = json.dumps(rows) + pm.to_markdown(rows)
    assert "/nfs/" not in text and "/somewhere/" not in text
    assert "Allocated GPU-hours, all attempts: 4.04" in pm.to_markdown(rows)


def test_markdown_adds_manual_rows_to_the_total():
    rows = [{"experiment_id": "ppv_p0_x", "orchestrator_status": "SUCCEEDED", "gpu_hours_allocated": 1.5}]
    md = pm.to_markdown(rows, [{"experiment_id": "ppv_r0_manual", "orchestrator_status": "MANUAL", "gpu_hours_allocated": 0.5}])
    assert "ppv_r0_manual" in md and "2.00 (MANUAL 0.50, SUCCEEDED 1.50)" in md


def test_microbenchmark_run_log_gives_the_commits(tmp_path):
    root = tmp_path / "results"
    run = root / "experiments" / "20261118_ppv_c1_c2_microbench_run01"
    run.mkdir(parents=True)
    (run / "run.log").write_text("2026-11-18T23:40:52-05:00 repo 99e044240c4974f27c0a38f044e59d05b65d0fb5 sglang-omni "
                                 "520b49ee69385d4a58912b9ce7a4712e181bf38a gpus c1=0 c2=1\n")
    state = {"experiments": {"ppv_c1_c2_microbench": {"status": "SUCCEEDED", "started_at": 1.0, "ended_at": 2.0,
                                                       "assigned": {"gpus": 2}, "output_dir": str(run), "gpu_seconds": 7200.0}}}
    row = pm.build(state, root, {})[0]
    assert row["research_repo_commit"] == "99e044240c" and row["sglang_omni_commit"] == "520b49ee69"
    assert row["gpu_hours_allocated"] == 2.0 and row["family"] == "C1"
