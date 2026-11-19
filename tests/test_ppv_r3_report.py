"""R3 analysis on synthetic camera records and server events (no GPU, no data files)."""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
spec = importlib.util.spec_from_file_location("ppv_r3_report", HERE / "ppv_r3_report.py")
r3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r3)


def _ticks(n, dropped_every=0):
    out = []
    for k in range(n):
        drop = bool(dropped_every) and k % dropped_every == 0
        out.append({"session": 0, "tick": k, "warmup": k < 2, "sent": not drop, "dropped": drop})
    return out


def _recs(ticks, latency_s=0.2, late_every=0):
    out = []
    for t in ticks:
        if not t["sent"]:
            continue
        sent = 1_000_000_000 * t["tick"]
        out.append({"request_id": f"c-s000-k{t['tick']:05d}", "warmup": t["warmup"], "status": "ok", "sent_ns": sent,
                    "first_text_ns": sent + int(latency_s * 1e9), "done_ns": sent + int(latency_s * 1.2e9),
                    "late": bool(late_every) and t["tick"] % late_every == 0, "answer_correct": True, "prompt_tokens": 250})
    return out


def test_client_stats_count_only_measured_ticks():
    ticks = _ticks(12, dropped_every=4)  # ticks 0, 4, 8 dropped; 0 and 1 are warmup
    st = r3.client_stats(ticks, _recs(ticks))
    assert st["ticks"] == 10 and st["dropped"] == 2 and abs(st["drop_ratio"] - 0.2) < 1e-9
    assert st["first_response_s"]["p50"] == 0.2 and st["first_response_s"]["p99"] is None  # fewer than 100 samples
    assert st["answer_correct_share"] == 1.0 and st["prompt_tokens_p50"] == 250


def _events(rid, t0, wait_ms, service_ms, total_ms, media_ms=None):
    ms = 1_000_000
    ev = [{"request_id": rid, "stage": "coordinator", "event_name": "request_admission", "timestamp_ns": t0},
          {"request_id": rid, "stage": "preprocessing", "event_name": "stage_input_received", "timestamp_ns": t0 + ms},
          {"request_id": rid, "stage": "preprocessing", "event_name": "stage_dispatch", "timestamp_ns": t0 + ms + wait_ms * ms},
          {"request_id": rid, "stage": "preprocessing", "event_name": "stage_complete",
           "timestamp_ns": t0 + ms + (wait_ms + service_ms) * ms},
          {"request_id": rid, "stage": "decode", "event_name": "terminal_response", "timestamp_ns": t0 + total_ms * ms}]
    if media_ms is not None:
        ev.append({"request_id": rid, "stage": "preprocessing", "event_name": "exp_preprocess", "timestamp_ns": t0,
                   "metadata": {"media_load_ns": media_ms * ms, "processor_ns": 5 * ms, "total_ns": service_ms * ms,
                                "inflight_at_start": 3}})
    return ev


def test_server_stats_split_preprocessing_and_utilisation():
    ids = {f"r{i}" for i in range(4)}
    events = [e for i in range(4) for e in _events(f"r{i}", i * 10**9, wait_ms=300, service_ms=200, total_ms=600, media_ms=150)]
    st = r3.server_stats(events, ids, window_s=1.0, max_concurrency=1)
    assert st["preprocessing_wait_ms_p50"] == 300 and st["preprocessing_service_ms_p50"] == 200
    assert abs(st["preprocessing_share_of_server_time"] - 500 / 600) < 1e-9
    assert abs(st["preprocessing_slot_utilisation"] - 0.8) < 1e-9  # 4 x 200 ms in a 1 s window, one slot
    assert st["preprocess_split_ms_p50"]["media_load"] == 150 and st["preprocess_inflight_max"] == 3


def test_classify_and_improvement_rules():
    sat = {"drop_ratio": 0.3, "late_ratio": 0.5}
    assert r3.classify({"drop_ratio": 0.01, "late_ratio": 0.05}, {}, 95.0) == "not_saturated"
    assert r3.classify(sat, {"preprocessing_share_of_server_time": 0.7, "preprocessing_slot_utilisation": 0.95}, 20.0) == "preprocessing_bound"
    assert r3.classify(sat, {"preprocessing_share_of_server_time": 0.2, "preprocessing_slot_utilisation": 0.3}, 92.0) == "gpu_bound"
    assert r3.classify(sat, {}, None) == "saturated_other"
    assert r3.improved({"drop_ratio": 0.30}, {"drop_ratio": 0.10})
    assert not r3.improved({"drop_ratio": 0.30}, {"drop_ratio": 0.27})  # 3 points: below the 5-point bar
    assert r3.improved({"drop_ratio": 0.0, "first_response_s": {"p95": 2.0}}, {"drop_ratio": 0.0, "first_response_s": {"p95": 1.5}})


def test_confirm_cells_picks_saturated_or_improved_cells_most_dropping_first():
    cells = {
        "frame_s16_f2": {"1": {"drop_ratio": 0.6, "late_ratio": 0.9, "sessions": 16}, "4": {"drop_ratio": 0.2}},
        "frame_s4_f2": {"1": {"drop_ratio": 0.2, "late_ratio": 0.4, "sessions": 4}, "4": {"drop_ratio": 0.2}},
        "frame_s1_f0.5": {"1": {"drop_ratio": 0.0, "late_ratio": 0.0, "sessions": 1}, "4": {"drop_ratio": 0.0}},
        "clip_s1_f1": {"1": {"drop_ratio": 0.0, "late_ratio": 0.1, "sessions": 1, "first_response_s": {"p95": 1.0}},
                       "4": {"drop_ratio": 0.0, "first_response_s": {"p95": 0.5}}},
    }
    assert r3.confirm_cells(cells) == ["frame_s16_f2", "frame_s4_f2", "clip_s1_f1"]
    assert r3.confirm_cells(cells, limit=1) == ["frame_s16_f2"]


def test_gpu_evidence_reports_power_and_flags_aliased_utilisation():
    win = {"0": {"samples": 300, "util_mean": 0.0, "util_zero_share": 1.0, "power_mean_w": 180.0, "util_suspect": True},
           "1": {"samples": 300, "util_mean": 85.0, "util_zero_share": 0.0, "power_mean_w": 290.0, "util_suspect": False}}
    assert r3.gpu_evidence(win, [0]) == {"gpu_power_mean_w": 180.0, "gpu_util_zero_share": 1.0, "gpu_util_suspect": True}
    assert r3.gpu_evidence(win, [1])["gpu_util_suspect"] is False
    assert r3.gpu_evidence(win, [7]) == {}


def test_confirmation_verdict_is_per_cell_over_three_paired_blocks():
    def cell(conc, drop, p95, shape="frame", sessions=16, fps=1.0):
        return {"shape": shape, "sessions": sessions, "fps": fps, "max_concurrency": conc, "drop_ratio": drop,
                "first_response_s": {"p95": p95}, "class": "gpu_bound", "sent": 450}

    def run(conc, block, cells):
        return {"experiment_id": f"ppv_r3_confirm_seq_c{conc}_block{block}", "cells": cells}

    runs = []
    for b in (1, 2, 3):
        runs.append(run(1, b, [cell(1, 0.50, 2.0), cell(1, 0.60, 2.0, shape="clip")]))
        # frame: no change in any block; clip: concurrency 4 drops far fewer frames in blocks 1-2 only
        runs.append(run(4, b, [cell(4, 0.50, 2.0), cell(4, 0.30 if b < 3 else 0.60, 2.0, shape="clip")]))
    v = r3.confirm_verdicts(runs)
    assert v["frame_s16_f1"]["verdict"] == "不支持" and v["clip_s16_f1"]["verdict"] == "不确定"
    assert v["clip_s16_f1"]["improved_by_block"] == {"1": True, "2": True, "3": False}
    assert r3.confirm_verdicts(runs[:-1])["clip_s16_f1"]["verdict"] == "数据不足"
    better = [run(1, b, [cell(1, 0.5, 2.0)]) for b in (1, 2, 3)] + [run(4, b, [cell(4, 0.2, 2.0)]) for b in (1, 2, 3)]
    assert r3.confirm_verdicts(better)["frame_s16_f1"]["verdict"] == "支持"
    assert r3.confirm_verdicts([{"experiment_id": "ppv_r3_screen_c1_a2", "cells": [cell(1, 0.5, 2.0)]}]) == {}
