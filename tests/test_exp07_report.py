"""实验 7 report: splitting a run into levels, talker and code2wav processing metrics, kernel overlap and the
pre-registered verdicts (synthetic data only)."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("exp07_report", HERE / "exp07_report.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)


def ev(rid, stage, name, t_ms):
    return {"request_id": rid, "stage": stage, "event_name": name, "timestamp_ns": int(t_ms * 1e6), "metadata": {}}


def test_talker_times_use_only_talker_stage_events():
    events = [ev("r1", "talker@r0", "scheduler_queue_enter", 0), ev("r1", "talker@r0", "scheduler_prefill_start", 30),
              ev("r1", "talker@r0", "model_path_end", 830),
              # the thinker has the same event names; they must not leak into the talker numbers
              ev("r1", "thinker@r0", "scheduler_prefill_start", 5), ev("r1", "thinker@r0", "model_path_end", 900),
              ev("r2", "talker@r1", "scheduler_queue_enter", 0)]  # no prefill: incomplete, left out
    t = rp.talker_times(events)
    assert t == {"r1": {"queue_ms": 30.0, "gen_ms": 800.0}}


def test_batches_go_to_the_level_of_most_of_their_requests():
    batches = {(1, 1): {"request_ids": ["a", "b", "x"]}, (1, 2): {"request_ids": ["x", "y"]}, (2, 1): {"request_ids": ["z"]}}
    assert set(rp.batches_of(batches, {"a", "b"})) == {(1, 1)}
    assert set(rp.batches_of(batches, {"x", "y", "z"})) == {(1, 2), (2, 1)}


def test_flow_time_per_frame_uses_batches_of_one_only():
    batches = {(1, i): {"size": 1, "gpu_flow_ms": 100.0 + i, "flow_solve_frames": 1000} for i in range(3)}
    batches[(1, 9)] = {"size": 4, "gpu_flow_ms": 900.0, "flow_solve_frames": 1000}
    batches[(1, 10)] = {"size": 1, "gpu_flow_ms": None, "flow_solve_frames": 1000}
    assert rp.flow_us_per_frame_b1(batches) == {"n": 3, "p50": 101.0}


def test_overlap_of_two_kernel_sets():
    talker = [(0, 10), (5, 20), (40, 50)]  # union 0-20, 40-50
    c2w = [(15, 45)]
    assert rp.overlap_ms(talker, c2w, scale=1.0) == 10.0  # 15-20 and 40-45


def run(arm, block, tp256, p95_95=4.0, flow=None, talker=None, tp8=4.0, p50_55=2.0, p50_95=3.0, errors=0):
    closed = {}
    for c, tp in ((8, tp8), (128, tp256), (256, tp256)):
        closed[str(c)] = {"client": {"throughput_rps": tp, "measured": 100, "errors": errors, "timeouts": 0},
                          "c2w_flow_us_per_frame_b1": {"p50": (flow or {}).get(c)},
                          "talker_gen_ms_per_audio_s": {"p50": (talker or {}).get(c)}}
    open_ = {"5.5": {"client": {"measured": 100, "errors": 0, "timeouts": 0, "e2e_s": {"p50": p50_55, "p95": 3.0}}},
             "9.5": {"client": {"measured": 100, "errors": 0, "timeouts": 0, "e2e_s": {"p50": p50_95, "p95": p95_95}}}}
    return {"arm": arm, "block": block, "closed": closed, "open": open_}


def world(shared_flow=160.0, c3_flow=110.0, shared_talker=50.0, c3_talker=40.0, k4c2_tp=26.0, k4c2_p95=3.5, c3_tp=None):
    runs = []
    for b in (1, 2, 3):
        runs.append(run("t2_k6_shared", b, 22.0))
        runs.append(run("t2_k3_shared", b, 15.0, flow={128: shared_flow, 256: shared_flow}, talker={128: shared_talker, 256: shared_talker}))
        runs.append(run("t2_k3_c3", b, c3_tp or 20.0, flow={128: c3_flow, 256: c3_flow}, talker={128: c3_talker, 256: c3_talker}))
        runs.append(run("t2_k4_c2", b, k4c2_tp, p95_95=k4c2_p95))
        runs.append(run("t2_k2_c4", b, 18.0, p95_95=6.0))
    return runs


def test_h7a_supported_when_both_processing_times_fall_by_more_than_ten_percent():
    v = rp.verdicts(world())
    assert v["H7a"]["verdict"] == "支持"
    assert v["H7a"]["by_metric"]["c2w_flow_us_per_frame_b1@128"]["relative"]["mean"] == (110 - 160) / 160


def test_h7a_refuted_when_a_metric_moves_less_than_five_percent():
    v = rp.verdicts(world(c3_talker=49.0))
    assert v["H7a"]["verdict"] == "反驳"


def test_h7b_needs_throughput_and_p95_from_the_same_isolated_arm():
    v = rp.verdicts(world())
    assert v["H7b"]["verdict"] == "支持" and v["H7b"]["by_arm"]["t2_k4_c2"]["meets"] is True
    # More throughput but a worse P95 at 9.5 req/s: not supported, and not refuted either, since the registered
    # refutation needs every isolated arm to be no faster than 2+6 (or faster only in some blocks).
    v = rp.verdicts(world(k4c2_p95=4.5))
    assert v["H7b"]["verdict"] == "不确定" and v["H7b"]["by_arm"]["t2_k4_c2"]["meets"] is False
    v = rp.verdicts(world(k4c2_tp=21.0))  # every isolated arm below the baseline
    assert v["H7b"]["verdict"] == "反驳"


def test_levels_with_more_than_five_percent_errors_are_unusable():
    runs = world()
    runs[3]["closed"]["256"]["client"]["errors"] = 6  # t2_k4_c2 block 1: 6 % errors at 256
    v = rp.verdicts(runs)
    assert v["H7b"]["by_arm"]["t2_k4_c2"]["throughput_256"]["unusable_blocks"] == [1]
    assert v["H7b"]["by_arm"]["t2_k4_c2"]["meets"] is False


def test_h7c_low_load_differences():
    v = rp.verdicts(world())
    assert v["H7c"]["low_load_small"] is True
    assert set(v["H7c"]["by_arm"]) == {"t2_k4_c2", "t2_k3_c3", "t2_k2_c4"}
