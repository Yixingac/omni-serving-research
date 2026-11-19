"""R0 verdict: the five registered pass conditions (no GPU, synthetic report)."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("ppv_r0_verdict", HERE / "ppv_r0_verdict.py")
rv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rv)

CAPS = {"status": 200, "body": {"native_full_duplex": True}}


def _session(status="ok", fatal=False, with_audio=2, audio_s=5.0, silent=0, chunks=10, planned=300, done=300, lags=300):
    return {"status": status, "errors": [{"code": "x", "fatal": fatal}] if fatal else [],
            "replies": {"with_audio": with_audio, "audio_s_total": audio_s, "silent_chunks": silent},
            "reply_rows": [{"chunks": chunks}], "units": {"planned": planned, "done": done, "lag_ms": {"n": lags}}}


def _report(s1=None, s2=None, restarts=None, miss=(0.0, 0.0)):
    phases = {}
    for name, sessions, m in (("duplex_s1", s1, miss[0]), ("duplex_s2", s2, miss[1])):
        if sessions is not None:
            # the pooled block as omni_research.duplex.metrics.pool writes it (unit_miss_rate at top level)
            phases[name] = {"duplex_sessions": sessions, "duplex": {"unit_miss_rate": m, "units_planned": 300}}
    return {"runs": [{"run": "r", "server_restarts": restarts or [], "phases": phases}]}


def test_all_conditions_pass_and_realtime():
    out = rv.verdict(_report([_session()], [_session(), _session()]), CAPS)
    assert all(out["conditions"].values()) and out["verdict"] == "通过"


def test_pass_but_not_realtime_when_units_miss_more_than_one_percent():
    out = rv.verdict(_report([_session()], [_session(), _session()], miss=(0.0, 0.05)), CAPS)
    assert out["verdict"].startswith("通过（能跑但不实时")


def test_each_condition_can_fail_the_verdict():
    cases = {
        "1_server_started": (_report([_session()], [_session(), _session()]), {"status": 200, "body": {}}),
        "2_sessions_completed": (_report([_session()], [_session(status="failed"), _session()]), CAPS),
        "3_no_fatal_error": (_report([_session(fatal=True)], [_session(), _session()]), CAPS),
        "4_valid_audio": (_report([_session(with_audio=0)], [_session(), _session()]), CAPS),
        "5_unit_latency_extractable": (_report([_session(done=200, lags=200)], [_session(), _session()]), CAPS),
    }
    for cond, (report, caps) in cases.items():
        out = rv.verdict(report, caps)
        assert out["conditions"][cond] is False, cond
        assert out["verdict"] == "未通过", cond


def test_mostly_silent_audio_and_missing_phase_or_restart_fail():
    assert rv.verdict(_report([_session(silent=8, chunks=10)], [_session(), _session()]), CAPS)["conditions"]["4_valid_audio"] is False
    out = rv.verdict(_report([_session()], None), CAPS)
    assert out["phases"]["duplex_s2"] == {"present": False} and out["verdict"] == "未通过"
    assert rv.verdict(_report([_session()], [_session(), _session()], restarts=[{"t": 1}]), CAPS)["conditions"]["3_no_fatal_error"] is False


def test_reads_the_miss_rate_where_the_real_pool_puts_it():
    from omni_research.duplex.metrics import pool

    assert "unit_miss_rate" in pool([])  # top-level key of the pooled block the verdict reads
