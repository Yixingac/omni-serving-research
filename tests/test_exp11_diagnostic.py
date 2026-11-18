"""实验 11 diagnostic (supplementary registration 五): per-phase shares and pooled shares of fingerprints over the
tolerance for each comparison, the largest deviation, and the registered reading."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("exp11_diagnostic", HERE / "exp11_diagnostic.py")
dg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dg)


def gate(rows):
    """rows: phase -> (C pairs, C over, C max, A2 pairs, A2 over, A2 max), the layout exp11_check.py writes."""
    phases = {}
    for name, (cp, co, cm, ap, ao, am) in rows.items():
        phases[name] = {"C_vs_A": {"pairs": cp, "over_tolerance": co, "share_over": co / cp, "max_deviation": cm},
                        "A2_vs_A": {"pairs": ap, "over_tolerance": ao, "share_over": ao / ap, "max_deviation": am}}
    return {"gate": {"phases": phases, "pass": False}}


def test_table_and_pooled_shares():
    side = gate({"chk_c32": (20, 4, 0.07, 20, 3, 0.08), "chk_c64": (30, 9, 0.06, 30, 9, 0.07)})
    t = dg.comparison(side, "C 对 B", "B2 对 B")
    assert t["phases"]["chk_c32"] == {"test": 0.2, "reference": 0.15, "test_max": 0.07, "reference_max": 0.08,
                                      "test_pairs": 20, "reference_pairs": 20}
    assert t["pooled"] == {"test": 13 / 50, "reference": 12 / 50}
    assert t["phases_over_margin"] == [] and t["max_deviation"] == 0.08


def test_registered_reading():
    ok = dg.comparison(gate({"chk_c32": (20, 4, 0.07, 20, 3, 0.08)}), "C 对 B", "B2 对 B")
    assert dg.reading(ok) == "多出的偏差和提前发射有关，没有证据指向旁路复制"
    over = dg.comparison(gate({"chk_c32": (20, 8, 0.07, 20, 3, 0.08)}), "C 对 B", "B2 对 B")  # 0.40 vs 0.15
    assert over["phases_over_margin"] == ["chk_c32"]
    assert dg.reading(over) == "不能排除旁路复制的问题"
    big = dg.comparison(gate({"chk_c32": (20, 4, 0.7, 20, 3, 0.08)}), "C 对 B", "B2 对 B")  # one row read wrongly
    assert dg.reading(big) == "不能排除旁路复制的问题"
