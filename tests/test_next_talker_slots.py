"""Next-step research P0: talker admission-cap evidence from SGLang decode log lines (synthetic logs only)."""

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("next_talker_slots", HERE / "next_talker_slots.py")
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)

PREFIX = "[INFO] sglang.srt.managers.scheduler_components.metrics_reporter: "


def line(t: dt.datetime, running: int, tokens: int, graph: bool, gen: float, queue: int) -> str:
    stamp = t.strftime("%Y-%m-%d %H:%M:%S") + f",{t.microsecond // 1000:03d}"
    return (f"{stamp} {PREFIX}Decode batch, #running-req: {running}, #token: {tokens}, token usage: 0.00, "
            f"cuda graph: {graph}, gen throughput (token/s): {gen:.2f}, #queue-req: {queue}\n")


T = dt.datetime(2026, 11, 17, 23, 50, 0)


def test_parse_keeps_decode_lines_and_their_fields():
    lines = [line(T, 32, 9000, True, 7982.0, 180),
             f"{T:%Y-%m-%d %H:%M:%S},000 {PREFIX}Prefill batch, #new-seq: 1, #new-token: 26, #cached-token: 0\n",
             "Capturing batches (bs=32 avail_mem=38.32 GB)\n"]
    rows = ts.parse_decode_lines(lines)
    assert len(rows) == 1
    r = rows[0]
    assert (r["running"], r["tokens"], r["graph"], r["gen_tok_s"], r["queue"]) == (32, 9000, True, 7982.0, 180)
    assert abs(r["ts"] - T.timestamp()) < 1e-6


def test_talker_filter_uses_tokens_per_request():
    rows = ts.parse_decode_lines([
        line(T, 64, 64 * 80, True, 2000.0, 0),    # thinker: short prompt + text tokens
        line(T, 16, 16 * 400, True, 6000.0, 0),   # talker: hidden-state rows + audio codes
        line(T, 0, 0, True, 0.0, 5),              # idle line
    ])
    kept = ts.talker_lines(rows, 200)
    assert [r["running"] for r in kept] == [16]


def test_window_stats_counts_cap_queue_and_graph():
    rows = ts.parse_decode_lines([line(T, 32, 32 * 400, True, 8000.0, 180)] * 3
                                 + [line(T, 20, 20 * 400, False, 6000.0, 0)])
    s = ts.window_stats(rows, cap=32)
    assert s["lines"] == 4 and s["at_cap_share"] == 0.75 and s["at_cap_with_queue_share"] == 0.75
    assert s["above_cap_lines"] == 0 and s["queue_max"] == 180 and s["graph_share"] == 0.75
    assert ts.window_stats([], cap=32) == {"lines": 0}


def test_decode_curve_medians_and_step_time():
    rows = ts.parse_decode_lines([line(T, 16, 6400, True, g, 0) for g in (6000.0, 6100.0, 6200.0)]
                                 + [line(T, 32, 12800, True, 8000.0, 0)] * 2
                                 + [line(T, 8, 3200, True, 10.0, 0)] * 5)   # idle-gap lines are dropped
    curve = ts.decode_curve(rows, min_lines=2)
    assert [c["batch"] for c in curve] == [16, 32]
    assert curve[0]["gen_tok_s_median"] == 6100.0
    assert abs(curve[1]["step_ms"] - 4.0) < 1e-9   # 32 requests / 8000 tokens per second


def test_main_last_level_window(tmp_path, monkeypatch):
    run = "20990101_exp09_t2_k1_c5_block1_run01"
    exp = tmp_path / "experiments" / run
    (exp / "analysis").mkdir(parents=True)
    (exp / "logs").mkdir()
    t_end = T + dt.timedelta(seconds=100)
    phases = [{"phase": "w3b_closed_A", "started": T.timestamp(), "ended": t_end.timestamp(),
               "levels": [{"concurrency": 128, "wall_s": 50.0}, {"concurrency": 256, "wall_s": 40.0}]}]
    (exp / "analysis" / "phases.json").write_text(json.dumps(phases))
    log = [line(T + dt.timedelta(seconds=10), 20, 8000, True, 6000.0, 0),   # before the last level
           line(T + dt.timedelta(seconds=70), 32, 12800, True, 8000.0, 150),
           line(T + dt.timedelta(seconds=90), 32, 12800, True, 8000.0, 170)]
    (exp / "logs" / "server_s0.log").write_text("".join(log))
    out = tmp_path / "o.json"
    monkeypatch.setattr(sys, "argv", ["x", "--data-root", str(tmp_path), "--run", run, "--phase", "w3b_closed_A",
                                      "--last-level", "--cap", "32", "--curve", "--out", str(out)])
    assert ts.main() == 0
    res = json.loads(out.read_text())["runs"][run]
    assert res["lines"] == 2 and res["at_cap_with_queue_share"] == 1.0 and res["queue_p50"] == 170
    assert abs(res["window_s"] - 40.0) < 1e-6
