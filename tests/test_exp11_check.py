"""实验 11 correctness gate: pairing requests across runs and the registered pass rules (synthetic data only)."""

import importlib.util
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("exp11_check", HERE / "exp11_check.py")
ck = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ck)


def fp(norms, scale=1.0):
    # a projection on a random unit-variance direction of a 4096-wide row is about norm / 64; half of that here
    return {"norm": list(norms), "proj": [[n * scale / 128] * 4 for n in norms]}


def req(i, out="t", rows=3, tokens=3, fpx=None, hidden="h", codec="c", status="ok", text=None, speech=True):
    return {"index": i, "status": status, "tokens": tokens, "rows": rows, "out_sha": f"{out}{i}", "hidden": f"{hidden}{i}",
            "codec": f"{codec}{i}", "fp": fpx or fp([10.0, 10.0, 10.0]), "text": text, "speech": speech}


def test_fingerprint_deviation_is_relative_to_the_row_norm():
    assert ck.fp_deviation(fp([10.0, 10.0]), fp([10.0, 10.0])) == 0.0
    assert math.isclose(ck.fp_deviation(fp([10.0, 10.5]), fp([10.0, 10.0])), 0.05)
    assert math.isinf(ck.fp_deviation(fp([10.0]), fp([10.0, 10.0])))  # different row counts never match


def test_exact_comparison_at_concurrency_one():
    a = [req(i, text="x") for i in range(3)]
    c = [req(i, text="x") for i in range(3)]
    assert ck.exact(a, c) == {"pairs": 3, "text": 3, "hidden": 3, "codec": 3}
    c[1]["hidden"] = "other"
    assert ck.exact(a, c)["hidden"] == 2


def test_fingerprint_comparison_only_on_identical_text():
    a = [req(i) for i in range(4)]
    b = [req(i, fpx=fp([10.0, 10.0, 12.0])) if i == 0 else req(i) for i in range(4)]
    b[3]["out_sha"] = "different text"
    r = ck.fingerprints(a, b)
    assert r["pairs"] == 3 and r["over_tolerance"] == 1 and math.isclose(r["max_deviation"], 0.2)


def test_rows_must_equal_tokens_for_finished_speech_requests():
    reqs = [req(0), req(1, rows=2), req(2, status="timeout", rows=None), req(3, rows=None, tokens=5, speech=False)]  # 3: text only
    assert ck.rows_check(reqs) == {"checked": 2, "mismatched": 1}
    assert ck.rows_check([req(0, rows=None)]) == {"checked": 1, "mismatched": 1}  # a speech request without hidden states


def run_set(c_dev=0.0, a2_dev=0.0, c_err=0, c1_same=True):
    def phases(dev, err):
        bad = fp([10.0, 10.0, 10.0 * (1 + dev)])
        out = {"chk_c1": [req(i, text="x") for i in range(18)]}
        for name in ck.EXPECTED_PHASES[1:]:
            out[name] = [req(i, fpx=bad if i < 10 else None) for i in range(40)]
        out["chk_timeout"] = [req(i, status="timeout", rows=None) for i in range(40)]
        out["chk_after"][0]["status"] = "error" if err else "ok"
        return out
    a = phases(0.0, 0)
    c = phases(c_dev, c_err)
    if not c1_same:
        c["chk_c1"][5]["codec"] = "x"
    return {"A": a, "C": c, "A2": phases(a2_dev, 0)}


def test_gate_passes_and_fails_by_the_registered_rules():
    assert ck.gate(run_set())["pass"] is True
    g = ck.gate(run_set(c_dev=0.5))  # 10 of 40 over tolerance for C, none for A2: +25 points
    assert g["pass"] is False and any("chk_c8" in r for r in g["reasons"])
    assert ck.gate(run_set(c_dev=0.5, a2_dev=0.5))["pass"] is True  # A repeats itself just as badly
    assert ck.gate(run_set(c_err=1))["pass"] is False
    assert ck.gate(run_set(c1_same=False))["pass"] is False


def test_projection_differences_count_at_their_natural_scale():
    a = {"norm": [10.0], "proj": [[0.10, 0.0, 0.0, 0.0]]}
    b = {"norm": [10.0], "proj": [[0.10 + 0.22, 0.0, 0.0, 0.0]]}  # another row of the same norm: off by 0.022 x norm
    assert math.isclose(ck.fp_deviation(b, a), 0.22 / (10.0 / 64))  # 1.4, far over the tolerance


def test_gate_needs_every_phase_a2_and_evidence():
    rs = run_set()
    del rs["C"]["chk_after"]
    assert ck.gate(rs)["pass"] is False
    rs = run_set()
    del rs["A2"]
    assert ck.gate(rs)["pass"] is False
    rs = run_set()
    for r in rs["C"]["chk_c8"]:
        r["out_sha"] = "never the same text"
    g = ck.gate(rs)
    assert g["pass"] is False and any("chk_c8" in x and "pair" in x for x in g["reasons"])


def test_gate_timeout_phase_exempts_only_timeouts():
    rs = run_set()
    rs["C"]["chk_timeout"][3]["status"] = "error"
    assert ck.gate(rs)["pass"] is False


def test_gate_fails_on_a_missing_hidden_state_event():
    rs = run_set()
    rs["C"]["chk_c32"][20]["rows"] = None
    assert ck.gate(rs)["pass"] is False


def test_gate_needs_all_eighteen_requests_at_concurrency_one():
    rs = run_set()
    rs["C"]["chk_c1"] = rs["C"]["chk_c1"][:17]
    assert ck.gate(rs)["pass"] is False


def test_gate_fails_when_a_run_restarted_or_failed():
    meta = {k: {"status": "ok", "restarts": []} for k in ("A", "C", "A2")}
    assert ck.gate(run_set(), meta)["pass"] is True
    meta["C"]["restarts"] = [{"where": "chk_c32 c=32 rep=0"}]
    assert ck.gate(run_set(), meta)["pass"] is False
    meta = {k: {"status": "ok", "restarts": []} for k in ("A", "C", "A2")}
    meta["A2"]["status"] = "failed"
    assert ck.gate(run_set(), meta)["pass"] is False
