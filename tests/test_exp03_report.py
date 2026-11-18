"""实验 3: throughput and latency against concurrency and arrival rate, and where the extra time goes."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("exp03_report", HERE / "exp03_report.py")
r03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r03)


def breakdown_row(level=64.0):
    return {"level": level, "n": 4, "errors": 0, "tp_rps": 10.0, "window_s": 20.0, "slo": 0.75,
            "ttft_s": [0.1, 0.1, 0.2, 0.3], "e2e_s": [2.0, 3.0, 4.0, 5.0], "e2e_per_audio_s": [0.2, 0.3, 0.4, 0.6],
            "stage_ms": {("thinker", "queue"): [10.0, 30.0], ("thinker", "model_path"): [1000.0, 1200.0],
                         ("talker", "queue"): [5.0, 5.0], ("talker", "model_path"): [500.0, 700.0],
                         ("code2wav", "service"): [200.0, 1800.0]},
            "eff_batch": {"thinker": 3.0}, "replicas": {"thinker": 2}, "gpu_util": {"thinker": 70.0}}


def test_level_summary_percentiles_and_littles_law():
    s = r03.level_summary(breakdown_row())
    assert s["concurrency"] == 64.0 and s["slo"] == 0.75
    assert s["e2e_p50_s"] == 3.5 and abs(s["e2e_p90_s"] - 4.7) < 1e-9
    assert abs(s["in_flight_little"] - 10.0 * 3.5) < 1e-9  # throughput x mean time in the system


def test_level_summary_stage_medians_and_split():
    s = r03.level_summary(breakdown_row())
    st = s["stages_ms"]
    assert st["thinker.queue"] == 20.0 and st["thinker.model_path"] == 1100.0 and st["code2wav.service"] == 1000.0
    # split of the median end-to-end time into the stage medians and the rest
    total = 20.0 + 1100.0 + 5.0 + 600.0 + 1000.0
    assert abs(s["split"]["other_ms"] - (3500.0 - total)) < 1e-9
    assert abs(s["split"]["code2wav_share"] - 1000.0 / 3500.0) < 1e-9


def test_slo_capacity_stops_at_the_first_failing_rate():
    assert r03.slo_capacity([(16, 1.0), (20, 0.99), (24, 0.96), (28, 0.83), (32, 0.97)]) == {"capacity": 24, "censored": False}
    assert r03.slo_capacity([(16, 1.0), (20, 0.99)]) == {"capacity": 20, "censored": True}  # never failed: a lower bound
    assert r03.slo_capacity([(16, 0.5)]) == {"capacity": None, "censored": False}


def test_doubling_gain():
    assert r03.doubling_gain({1: 0.5, 2: 1.0, 4: 1.5, 8: 1.6}) == {2: 1.0, 4: 0.5, 8: 1.6 / 1.5 - 1}


def _curve(e2e_by_c):
    return {"levels": [{"concurrency": float(c), "e2e_p50_s": v} for c, v in e2e_by_c.items()]}


def test_full_replicas_best_only_at_low_concurrency_verdict():
    # A-H1 second half: 8 full replicas are best only while each GPU has <= 2 requests (concurrency <= 16).
    good = {"r8": _curve({1: 1.0, 16: 2.0, 64: 4.0}), "s2x6": _curve({1: 1.02, 16: 2.05, 64: 3.0})}
    v = r03.full_replica_verdict(good, tol=0.03)
    assert v["verdict"] == "支持" and v["r8_near_best"] == {1: True, 16: True, 64: False}
    bad = {"r8": _curve({1: 1.0, 16: 2.8, 64: 4.0}), "s2x6": _curve({1: 1.02, 16: 2.2, 64: 3.0})}
    assert r03.full_replica_verdict(bad, tol=0.03)["verdict"] == "不支持"
