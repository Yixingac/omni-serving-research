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
    return {"norm": list(norms), "proj": [[n * 0.1 * scale] * 4 for n in norms]}


def req(i, out="t", rows=3, tokens=3, fpx=None, hidden="h", codec="c", status="ok", text=None):
    return {"index": i, "status": status, "tokens": tokens, "rows": rows, "out_sha": f"{out}{i}", "hidden": f"{hidden}{i}",
            "codec": f"{codec}{i}", "fp": fpx or fp([10.0, 10.0, 10.0]), "text": text}


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
    reqs = [req(0), req(1, rows=2), req(2, status="timeout", rows=None), req(3, rows=None, tokens=5)]  # 3 is text-only
    assert ck.rows_check(reqs) == {"checked": 2, "mismatched": 1}


def run_set(c_dev=0.0, a2_dev=0.0, c_err=0, c1_same=True):
    def phases(dev, err):
        bad = fp([10.0, 10.0, 10.0 * (1 + dev)])
        out = {"chk_c1": [req(i, text="x") for i in range(18)]}
        for name in ("chk_c8", "chk_c32", "chk_mix_long", "chk_after"):
            out[name] = [req(i, fpx=bad if i < 10 else None) for i in range(40)]
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
