"""实验 14: video throughput as pipeline replicas are added, and where a video request spends its time."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("exp14_report", HERE / "exp14_report.py")
r14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r14)

S = 1_000_000_000


def rec(i, conc, sent, done, status="ok", warmup=False):
    return {"request_id": f"v{i}", "concurrency": conc, "sent_ns": int(sent * S), "done_ns": int(done * S), "status": status, "warmup": warmup}


def test_level_stats_makespan_and_littles_law():
    recs = [rec(0, 2, 0.0, 1.0), rec(1, 2, 0.0, 1.0), rec(2, 2, 1.0, 2.0), rec(3, 2, 1.0, 3.0),
            rec(4, 2, -5.0, -4.0, warmup=True), rec(5, 2, 1.5, 2.5, status="error")]
    lv = r14.level_stats(recs)[2]
    assert lv["n"] == 5 and lv["ok"] == 4
    assert abs(lv["span_s"] - 3.0) < 1e-9 and abs(lv["rps"] - 4 / 3.0) < 1e-9
    assert abs(lv["e2e_mean_s"] - 1.25) < 1e-9
    assert abs(lv["little_rps"] - 2 / 1.25) < 1e-9  # concurrency / mean time in the system


def test_efficiency_against_linear_scaling():
    eff = r14.efficiency({1: 2.0, 2: 3.8, 4: 7.2, 8: 12.8})
    assert eff == {1: 1.0, 2: 0.95, 4: 0.9, 8: 0.8}


def test_linear_verdict_needs_both_video_lengths():
    assert r14.linear_verdict({"10s": 0.9, "20s": 0.85})["verdict"] == "支持"
    assert r14.linear_verdict({"10s": 0.9, "20s": 0.78})["verdict"] == "不确定"
    assert r14.linear_verdict({"10s": 0.7, "20s": 0.6})["verdict"] == "不支持"


def test_gpu_flat_verdict_within_five_points():
    assert r14.gpu_flat_verdict({1: 14.0, 2: 15.5, 4: 16.0, 8: 18.9})["verdict"] == "支持"
    assert r14.gpu_flat_verdict({1: 14.0, 8: 25.0})["verdict"] == "不支持"


def test_window_cores_skips_the_client_and_single_group_samples():
    samples = [{"t_wall_ns": 1 * S, "groups": {"client": {"cpu_percent": 50.0}, "r0": {"cpu_percent": 100.0}, "r1": {"cpu_percent": 140.0}}},
               {"t_wall_ns": 2 * S, "groups": {"client": {"cpu_percent": 50.0}, "r0": {"cpu_percent": 120.0}, "r1": {"cpu_percent": 120.0}}},
               {"t_wall_ns": 3 * S, "groups": {"client": {"cpu_percent": 50.0}}},
               {"t_wall_ns": 9 * S, "groups": {"client": {"cpu_percent": 50.0}, "r0": {"cpu_percent": 900.0}}}]
    assert abs(r14.window_cores(samples, 0, 5 * S) - 2.4) < 1e-9  # (2.4 + 2.4) / 2; the client-only sample is skipped


def test_stage_medians_and_shares():
    per = {"a": {"preprocessing.service": 500.0, "thinker.service": 70.0, "request.admission_to_last_terminal": 600.0},
           "b": {"preprocessing.service": 700.0, "thinker.service": 74.0, "request.admission_to_last_terminal": 800.0}}
    st = r14.stage_medians(per, ["a", "b"])
    assert st["preprocessing.service"] == 600.0 and st["request.admission_to_last_terminal"] == 700.0
    assert abs(st["share.preprocessing.service"] - 600.0 / 700.0) < 1e-9


def test_combine_needs_agreement():
    assert r14.combine(["支持", "支持"]) == "支持"
    assert r14.combine(["不支持", "不支持"]) == "不支持"
    assert r14.combine(["支持", "不支持"]) == "不确定"
    assert r14.combine(["支持", "不确定"]) == "不确定"


def test_phase_gpu_busy_averages_the_runs_gpus_only():
    ana = {"gpu_by_phase": [{"phase": "w4_video_10s", "gpu": "0", "sampled_util_mean": 10.0},
                            {"phase": "w4_video_10s", "gpu": "1", "sampled_util_mean": 14.0},
                            {"phase": "w4_video_10s", "gpu": "5", "sampled_util_mean": 0.0},   # not used by this run
                            {"phase": "w4_video_20s", "gpu": "0", "sampled_util_mean": 99.0}]}
    assert r14.phase_gpu_busy(ana, "w4_video_10s", [0, 1]) == 12.0
