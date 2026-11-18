"""scripts/slurm/run_all.sh hold mode and the STOP file after a requeue (no Slurm needed: squeue is
missing, so the remaining time reads as unlimited)."""

import os
import subprocess
import time
from pathlib import Path

RUN_ALL = Path(__file__).resolve().parents[1] / "scripts/slurm/run_all.sh"


def start(tmp_path, stop_text, restart_count):
    repo = tmp_path / "repo"
    (repo / "scripts/slurm").mkdir(parents=True)
    for name in ("preflight.sh", "cleanup.sh"):
        (repo / "scripts/slurm" / name).write_text("#!/usr/bin/env bash\nexit 0\n")
    run = tmp_path / "run"
    run.mkdir()
    if stop_text is not None:
        (run / "STOP").write_text(stop_text)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.update(OMNI_REPO=str(repo), OMNI_RUN_DIR=str(run), OMNI_HOLD_POLL_S="0.2", PATH="/usr/bin:/bin")
    if restart_count is not None:
        env["SLURM_RESTART_COUNT"] = str(restart_count)
    proc = subprocess.Popen(["bash", str(RUN_ALL), "hold"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc, run


def wait_for(pred, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.1)
    return False


def test_requeued_job_removes_a_stop_written_by_the_signal_and_holds(tmp_path):
    proc, run = start(tmp_path, "signal 2026-11-15T10:40:29-05:00\n", restart_count=1)
    try:
        assert wait_for(lambda: "removing the STOP" in (run / "run_all.log").read_text() if (run / "run_all.log").exists() else False)
        assert not (run / "STOP").exists()
        time.sleep(0.5)
        assert proc.poll() is None, "the requeued job should keep holding"
        (run / "STOP").touch()  # a person releases it
        assert proc.wait(timeout=10) == 0
    finally:
        proc.kill() if proc.poll() is None else None


def test_a_stop_made_by_a_person_survives_the_requeue(tmp_path):
    proc, run = start(tmp_path, "", restart_count=1)
    assert proc.wait(timeout=10) == 0
    assert (run / "STOP").exists()


def test_without_a_requeue_the_signal_stop_still_ends_the_job(tmp_path):
    proc, run = start(tmp_path, "signal 2026-11-15T10:40:29-05:00\n", restart_count=None)
    assert proc.wait(timeout=10) == 0
    assert (run / "STOP").exists()


def test_stop_signal_writes_its_reason_into_the_stop_file(tmp_path):
    proc, run = start(tmp_path, None, restart_count=None)
    try:
        assert wait_for(lambda: "preflight done" in (run / "run_all.log").read_text() if (run / "run_all.log").exists() else False)
        proc.terminate()
        assert proc.wait(timeout=10) == 0
        assert (run / "STOP").read_text().startswith("signal ")
    finally:
        proc.kill() if proc.poll() is None else None
