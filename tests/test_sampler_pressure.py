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


def test_plugin_phase_imports_the_module_and_passes_the_context(tmp_path, monkeypatch):
    import asyncio
    import sys
    import types

    from omni_research.orchestration.experiment import ExperimentRunner

    seen = {}

    async def run_phase(ctx, ph):
        seen.update(ctx=ctx, ph=ph)
        return {"plugin_ok": True}

    mod = types.ModuleType("omni_research.fake_plugin_phase")
    mod.run_phase = run_phase
    monkeypatch.setitem(sys.modules, "omni_research.fake_plugin_phase", mod)

    class _Srv:
        name, base_url = "s0", "http://127.0.0.1:1234"

    fake = types.SimpleNamespace(out=tmp_path, servers=[_Srv()])
    ph = {"kind": "plugin", "name": "duplex_1", "module": "fake_plugin_phase", "sessions": 2}
    out = asyncio.run(ExperimentRunner._phase_plugin(fake, ph, "duplex_1", 7))
    assert out == {"plugin_ok": True}
    assert seen["ctx"]["servers"] == {"s0": "http://127.0.0.1:1234"} and seen["ctx"]["seed"] == 7
    assert seen["ctx"]["raw_dir"] == tmp_path / "raw" and seen["ph"]["sessions"] == 2
