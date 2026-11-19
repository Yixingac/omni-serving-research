"""P2 gate on synthetic verify events: pass, a bit mismatch, unverified requests, log warnings, the row erratum."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("ppv_p2_report", HERE / "ppv_p2_report.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)


def ev(rid, **over):
    md = {"is_equal_reference": True, "is_equal_resync": True, "is_rows_expected": True, "is_aligned": True,
          "rows": 40, "reference_rows": 40, "expected_rows": 40, "first_bad_row": -1}
    md.update(over)
    return {"request_id": rid, "metadata": md}


def cl(rid, status="ok", audio=6.0):
    return {"request_id": rid, "status": status, "output_audio_s": audio}


def test_all_bit_exact_passes():
    g = rp.gate({"chk_c8": [ev("a"), ev("b")]}, {"chk_c8": [cl("a"), cl("b"), cl("t", audio=0.0)]}, [], [])
    assert g["gate"] == "通过" and g["requests_verified"] == 2 and g["per_phase"]["chk_c8"]["unverified"] == 0


def test_one_mismatch_fails_with_the_evidence():
    g = rp.gate({"chk_c32": [ev("a"), ev("b", is_equal_reference=False, first_bad_row=7, side_sha256="x", reference_sha256="y")]},
                {"chk_c32": [cl("a"), cl("b")]}, [], [])
    assert g["gate"] == "不通过" and g["mismatch_count"] == 1
    m = g["mismatches"][0]
    assert m["request_id"] == "b" and m["first_bad_row"] == 7 and m["failed"] == ["is_equal_reference"]


def test_unverified_speech_requests_and_log_warnings_fail():
    g = rp.gate({"chk_c8": [ev("a")]}, {"chk_c8": [cl("a"), cl("b")]}, [], [])
    assert g["gate"] == "不通过" and any("without a verify event" in r for r in g["reasons"])
    g = rp.gate({"chk_c8": [ev("a")]}, {"chk_c8": [cl("a")]}, ["... exp_hidden_verify mismatch request_id=z {...}"], [])
    assert g["gate"] == "不通过" and g["log_warnings"] == 1


def test_client_timeouts_count_only_outside_the_abort_phase():
    g = rp.gate({"chk_timeout": [ev("a")], "chk_after": [ev("b")]},
                {"chk_timeout": [cl("a"), cl("x", status="timeout", audio=0)], "chk_after": [cl("b")]}, [], [])
    assert g["gate"] == "通过"
    g = rp.gate({"chk_after": [ev("b")]}, {"chk_after": [cl("b"), cl("y", status="timeout", audio=0)]}, [], [])
    assert g["gate"] == "不通过"


def test_expected_rows_erratum_is_reported_but_never_rescues_a_bit_mismatch():
    # the reference itself has one row more than len(output_ids): the expected count was defined wrongly
    rows_def = rp.gate({"chk_c8": [ev("a", rows=41, reference_rows=41, expected_rows=40, is_rows_expected=False)]},
                       {"chk_c8": [cl("a")]}, [], [])
    assert rows_def["gate"] == "通过" and rows_def["erratum_count"] == 1
    bits = rp.gate({"chk_c8": [ev("a", rows=41, reference_rows=41, expected_rows=40, is_rows_expected=False, is_equal_reference=False)]},
                   {"chk_c8": [cl("a")]}, [], [])
    assert bits["gate"] == "不通过"


def test_restart_or_no_events_fail():
    assert rp.gate({"chk_c8": [ev("a")]}, {"chk_c8": [cl("a")]}, [], [{"where": "chk_c8"}])["gate"] == "不通过"
    assert rp.gate({"chk_c8": []}, {"chk_c8": []}, [], [])["gate"] == "不通过"
