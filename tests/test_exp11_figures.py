"""实验 11 figures: values are aligned by block, so a missing block never shifts a pairing (synthetic data)."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("exp11_figures", HERE / "exp11_figures.py")
fg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fg)


def test_per_block_is_aligned_to_the_three_blocks():
    s = {"runs": [{"arm": "C", "block": b, "levels": {"w3b_ns": {"32": {"A": {"e2e_p50": float(b)}}}}} for b in (1, 3)]}
    assert fg.per_block(s, "C", "w3b_ns", 32, "A", "e2e_p50") == [1.0, None, 3.0]
