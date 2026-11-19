"""Presentation figures: every figure whose committed input exists renders to PNG and SVG; a missing input is skipped,
never drawn as zeros (no GPU, no network)."""

import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("ppv_figures", HERE / "ppv_figures.py")
pf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pf)


@pytest.mark.parametrize("name", sorted(pf.FIGURES))
def test_each_figure_renders_or_reports_its_missing_input(name, tmp_path):
    try:
        paths = pf.FIGURES[name](tmp_path)
    except FileNotFoundError as exc:  # main() reports these as skipped
        assert exc.filename and not Path(exc.filename).exists()
        return
    assert [p.suffix for p in paths] == [".png", ".svg"]
    assert all(p.exists() and p.stat().st_size > 1000 for p in paths)


def test_main_skips_figures_without_input(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pf, "P0_SUMMARY", tmp_path / "missing.json")
    monkeypatch.setattr(sys, "argv", ["ppv_figures.py", "--out", str(tmp_path / "f"), "--only", "p0_admission"])
    assert pf.main() == 0
    assert "skipped (missing input" in capsys.readouterr().out
    assert not (tmp_path / "f" / "fig02_talker_admission.png").exists()
