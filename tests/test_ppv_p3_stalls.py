"""P3 pause scan: log gaps inside measured phases, PSI stall time per second, sparse-log flag (no GPU, no data)."""

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("ppv_p3_stalls", HERE / "ppv_p3_stalls.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)

T0 = 1795070000.0  # 2026-11-19 02:13:20 UTC


def _log_line(t_epoch: float, offset_s: float = -18000) -> str:
    import datetime as dt

    local = dt.datetime.fromtimestamp(t_epoch + offset_s, dt.timezone.utc)
    return local.strftime("%Y-%m-%d %H:%M:%S") + f",{int(round((t_epoch % 1) * 1000)) % 1000:03d} [INFO] x: line\n"


def _sample(t: float, io_full_total_us: float, d_state: int = 0) -> dict:
    return {"t_wall_ns": int(t * 1e9), "groups": {"s0": {"states": {"sleeping": 3, "disk-sleep": d_state}}},
            "psi": {"io": {"some_total": io_full_total_us, "full_total": io_full_total_us},
                    "cpu": {"some_total": 0.0, "full_total": 0.0}, "memory": {"some_total": 0.0, "full_total": 0.0}}}


def _run(tmp_path: Path, line_times: list[float], samples: list[dict]) -> Path:
    exp = tmp_path / "20261119_ppv_x_run01"
    (exp / "analysis").mkdir(parents=True)
    (exp / "logs").mkdir()
    (exp / "raw").mkdir()
    (exp / "manifest.json").write_text(json.dumps({"utc_offset": "-0500"}))
    (exp / "analysis" / "phases.json").write_text(json.dumps([
        {"phase": "smoke", "kind": "smoke", "started": T0 - 5, "ended": T0},
        {"phase": "open", "kind": "open_loop", "started": T0, "ended": T0 + 20}]))
    (exp / "logs" / "server_s0.log").write_text("".join(_log_line(t) for t in line_times))
    (exp / "raw" / "cpu_samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    return exp


def test_dense_log_gap_is_found_with_its_io_stall_and_d_state(tmp_path):
    lines = [T0 + 0.1 * i for i in range(0, 80)] + [T0 + 11.2 + 0.1 * i for i in range(0, 80)]  # 3.3 s gap at 7.9 s
    total, samples = 0.0, []
    for i in range(21):
        stall = 700_000.0 if i in (8, 9, 10) else 1_000.0  # 700 ms of full I/O stall per second during the gap
        total += stall
        samples.append(_sample(T0 + i, total, d_state=2 if i in (9, 10) else 0))
    r = ps.analyze_run(_run(tmp_path, lines, samples))
    assert r["log_dense"] and len(r["log_gaps"]) == 1
    g = r["log_gaps"][0]
    assert abs(g["gap_s"] - 3.3) < 0.01 and g["io_full_ms_per_s_max"] >= 699 and g["d_state_max"] == 2
    assert r["baseline"]["io_full_ms_per_s_median"] < 10
    s = ps.summarize([r])
    assert s["log_gaps_over_1s"] == 1 and s["log_gaps_over_2s"] == 1 and s["gaps_with_io_stall"] == 1


def test_sparse_logs_are_not_counted_and_sampler_gaps_are(tmp_path):
    lines = [T0 + 2.0 * i for i in range(10)]  # one line every 2 s, as a quiet duplex server
    samples = [_sample(T0 + t, 0.0) for t in (0, 1, 2, 5, 6, 7)]  # the sampler missed 2 beats (3 s spacing)
    r = ps.analyze_run(_run(tmp_path, lines, samples))
    assert not r["log_dense"] and r["log_gaps"] == [] and len(r["log_gaps_sparse_log"]) == 9
    assert len(r["sampler_gaps"]) == 1 and r["sampler_gaps"][0]["gap_s"] == 3.0
    assert ps.summarize([r])["log_gaps_over_1s"] == 0


def test_utc_offset_parsing():
    assert ps.utc_offset_s({"utc_offset": "-0500"}) == -18000
    assert ps.utc_offset_s({"utc_offset": "+0530"}) == 19800


def test_duplex_runs_never_count_log_gaps(tmp_path):
    lines = [T0 + 0.1 * i for i in range(0, 80)] + [T0 + 11.2 + 0.1 * i for i in range(0, 80)]
    exp = _run(tmp_path, lines, [_sample(T0 + i, 0.0) for i in range(21)])
    (exp / "raw" / "duplex_duplex_s1").mkdir()
    r = ps.analyze_run(exp)
    assert r["duplex"] and not r["log_dense"] and r["log_gaps"] == [] and len(r["log_gaps_sparse_log"]) == 1


def test_gen2_gc_is_counted_in_measured_phases_and_matched_to_an_overlapping_gap(tmp_path):
    # lines until 7.9 s, then from 10.5 s; a 1.5 s collection starts at 8.0 s and its log line is written when it ends
    lines = [T0 + 0.1 * i for i in range(0, 80)] + [T0 + 10.5 + 0.1 * i for i in range(0, 80)]
    exp = _run(tmp_path, lines, [_sample(T0 + i, 0.0) for i in range(21)])
    gc_in_gap = _log_line(T0 + 9.5).rstrip("\n").replace("x: line", f"exp_trace: exp_gc generation=2 pid=4242 stage= start_ns={int((T0 + 8.0) * 1e9)} duration_ms=1500.0 collected=7")
    gc_startup = _log_line(T0 - 3).rstrip("\n").replace("x: line", f"exp_trace: exp_gc generation=2 pid=4242 stage= start_ns={int((T0 - 3) * 1e9)} duration_ms=900.0 collected=7")
    with open(exp / "logs" / "server_s0.log", "a") as fh:
        fh.write(gc_in_gap + "\n" + gc_startup + "\n")
    (exp / "raw" / "server_events" / "open").mkdir(parents=True)
    (exp / "raw" / "server_events" / "open" / "events_code2wav@r1_4242.jsonl").write_text("")
    r = ps.analyze_run(exp)
    assert r["gc_gen2"]["count"] == 1 and r["gc_gen2"]["duration_ms_max"] == 1500.0  # the start-up one is outside
    assert r["gc_gen2"]["by_stage"] == {"code2wav": 1}
    gap = next(g for g in r["log_gaps"] if g["gap_s"] > 1.5)
    assert gap["gc_overlap"] == [{"pid": 4242, "stage": "code2wav", "duration_ms": 1500.0}]
    assert ps.summarize([r])["gaps_with_gc_overlap"] == 1
