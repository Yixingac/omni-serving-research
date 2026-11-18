"""Shared reading rules for run records: which phase a server restart belongs to, and crash-affected requests."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("run_records", HERE / "run_records.py")
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)


def test_restart_phase_reads_every_format_the_runner_writes():
    # src/omni_research/orchestration/experiment.py: after a closed-loop level, after an open-loop rate, at phase end
    assert rr.restart_phase("w3b_closed_A c=256 rep=0") == "w3b_closed_A"
    assert rr.restart_phase("w3b_open rate=23.5") == "w3b_open"
    assert rr.restart_phase("phase w3l_closed_B") == "w3l_closed_B"
    assert rr.restart_phase("") is None and rr.restart_phase(None) is None


def test_restart_flags_by_phase_in_run_order():
    phases = ["smoke_w3b", "w3b_closed_A", "w3b_closed_B", "w3b_open"]
    restarts = [{"server": "s0", "exit_code": -11, "where": "w3b_closed_B c=128 rep=0"}]
    flags = rr.restart_flags(phases, restarts)
    assert flags["w3b_closed_A"] == (False, False)
    assert flags["w3b_closed_B"] == (True, True)  # (restart here, at or after a restart)
    assert flags["w3b_open"] == (False, True)


def test_clean_drops_crash_affected_requests_only():
    recs = [{"request_id": "a", "status": "ok"}, {"request_id": "b", "status": "error", "crash_affected": True},
            {"request_id": "c", "status": "ok", "warmup": True}]
    kept, dropped = rr.clean(recs)
    assert [r["request_id"] for r in kept] == ["a", "c"] and dropped == 1
