"""The CPU sampler's pressure (PSI) and process-state fields, added for locating pauses (P3)."""

from __future__ import annotations

from omni_research.orchestration.experiment import process_states, read_pressure


def test_read_pressure_parses_some_and_full_lines(tmp_path):
    (tmp_path / "cpu").write_text("some avg10=1.50 avg60=0.80 avg300=0.10 total=123456\n"
                                  "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n")
    (tmp_path / "io").write_text("some avg10=42.00 avg60=3.00 avg300=1.00 total=999\n"
                                 "full avg10=40.10 avg60=2.90 avg300=0.90 total=888\n")
    psi = read_pressure(str(tmp_path))
    assert psi["cpu"] == {"some_avg10": 1.5, "some_total": 123456.0, "full_avg10": 0.0, "full_total": 0.0}
    assert psi["io"]["full_avg10"] == 40.1 and psi["io"]["some_total"] == 999.0
    assert "memory" not in psi  # a missing file is skipped, not an error


def test_read_pressure_is_empty_without_psi(tmp_path):
    assert read_pressure(str(tmp_path / "does-not-exist")) == {}


class _Proc:
    def __init__(self, status):
        self._status = status

    def status(self):
        if isinstance(self._status, Exception):
            raise self._status
        return self._status


def test_process_states_counts_states_and_skips_exited_processes():
    tree = [_Proc("running"), _Proc("sleeping"), _Proc("sleeping"), _Proc("disk-sleep"), _Proc(ProcessLookupError())]
    assert process_states(tree) == {"running": 1, "sleeping": 2, "disk-sleep": 1}
