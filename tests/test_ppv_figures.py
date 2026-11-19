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


def test_p1_figure_draws_from_a_summary_and_never_draws_unbounded_means(tmp_path, monkeypatch):
    import json

    def iv(m):
        return {"runs": [m, m], "n": 2, "mean": m, "ci95": [m * 0.9, m * 1.1]}

    summ = {}
    for w in ("mix", "w3b"):
        for st, *_ in pf.P1_SETTINGS:
            for k, tag in enumerate(("x090", "x100", "x110")):
                summ[f"{w}|{st}|{tag}"] = {"offered_rps": iv(13.0 + 1.5 * k), "goodput_rps": iv(10.0 - k), "e2e_p95": iv(3.0 + k),
                                           "short_queue_p99_ms": iv(800.0 + 100 * k), "steady": [True, k < 2, True]}
    summ["w3b|fifo16|x110"]["e2e_p95"] = {"runs": ["inf"], "n": 1, "mean": "inf", "ci95": None}
    p = tmp_path / "p1.json"
    p.write_text(json.dumps({"summary": summ, "verdicts": {"H1a": {"verdict": "不确定"}, "H1b": {"verdict": "支持"}, "H1c": {"verdict": "数据不足"}}}))
    monkeypatch.setattr(pf, "P1_SUMMARY", p)
    paths = pf.fig_p1_batching(tmp_path / "out")
    assert [q.suffix for q in paths] == [".png", ".svg"] and all(q.stat().st_size > 1000 for q in paths)
    assert "inconclusive" in paths[1].read_text() and "w3b fifo16 x110" in paths[1].read_text()


def test_r4_figure_draws_blocks_from_a_summary(tmp_path, monkeypatch):
    import json

    def level(n, k):
        return {"phase": f"r4_n{n}", "sessions": n, "logical_ratio_peak": {"a": 0.01 * n / 8 + k * 0.001},
                "pool": {"a": {"resident_peak_ratio": 0.02 * n / 8}},
                "attribution": {"lost_share_of_prefill": {"eviction": 0.0, "routing": 0.02 + k * 0.01}}}

    runs = {f"20261119_ppv_r4_block{k}_run01": {"levels": [level(n, k) for n in (8, 16, 32, 64, 128)] + [
        {"phase": "r4_n32_media2", "sessions": 32}], "verdict": {"conclusion": "x"}} for k in (1, 2, 3)}
    runs["20261119_ppv_r4_smoke_run01"] = {"smoke": {}}
    p = tmp_path / "r4.json"
    p.write_text(json.dumps({"runs": runs, "combined": {"conclusion": "x"}}))
    monkeypatch.setattr(pf, "R4_SUMMARY", p)
    paths = pf.fig_r4_working_set(tmp_path / "out")
    assert [q.suffix for q in paths] == [".png", ".svg"] and all(q.stat().st_size > 1000 for q in paths)
    assert "block 3" in paths[1].read_text()


def test_duplex_sessions_figure_overlays_r2b_only_when_it_exists(tmp_path, monkeypatch):
    import json

    r2 = json.loads(pf.R2_SUMMARY.read_text())
    lv = [dict(x) for x in r2["arms"]["ms8_1gpu"]["levels"] if x["level"] in (4, 8)]
    p = tmp_path / "r2b.json"
    p.write_text(json.dumps({"runs": {"block1": "x", "block2": "y", "block3": "z"}, "verdict_h_r2b": {"verdict": "支持"},
                             "arms": {"ms8_2gpu": {"levels": lv}}}))
    monkeypatch.setattr(pf, "R2B_SUMMARY", p)
    svg = pf.fig_duplex_sessions(tmp_path / "with")[1].read_text()
    assert "R2b" in svg and "supported" in svg
    monkeypatch.setattr(pf, "R2B_SUMMARY", tmp_path / "missing.json")
    assert "R2b" not in pf.fig_duplex_sessions(tmp_path / "without")[1].read_text()


def test_r3_figure_overlays_the_confirmation_only_when_it_exists(tmp_path, monkeypatch):
    import json

    conf = {"frame_s16_f1": {"verdict": "不支持", "offered_per_s": 16.0,
                             "sent_per_s": {"c1": {"1": 7.4, "2": 7.6, "3": 7.5}, "c4": {"1": 7.5, "2": 7.5, "3": None}}}}
    p = tmp_path / "conf.json"
    p.write_text(json.dumps({"confirmation": conf}))
    monkeypatch.setattr(pf, "R3_CONFIRM", p)
    svg = pf.fig_r3_camera(tmp_path / "with")[1].read_text()
    assert "3-block confirmation" in svg and "not supported" in svg
    monkeypatch.setattr(pf, "R3_CONFIRM", tmp_path / "missing.json")
    assert "3-block confirmation" not in pf.fig_r3_camera(tmp_path / "without")[1].read_text()
