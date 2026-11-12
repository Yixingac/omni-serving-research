import json
import sqlite3
import sys
from pathlib import Path

import pytest

from omni_research.analysis import experiment_analysis as EA
from omni_research.analysis.nsys import busy_idle_report, device_activity, nvtx_ranges


def make_nsys_db(path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (deviceId INT, start INT, end INT)")
    con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (deviceId INT, start INT, end INT)")
    con.execute("CREATE TABLE StringIds (id INT, value TEXT)")
    con.execute("CREATE TABLE NVTX_EVENTS (start INT, end INT, text TEXT, textId INT)")
    con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (deviceId INT, start INT, end INT, graphId INT)")
    # Device 0: two kernels overlapping on different streams (0-500 and 300-800) + a copy 900-1000.
    con.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?)", [(0, 0, 500), (0, 300, 800), (1, 0, 100)])
    con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (0, 900, 1000)")
    con.execute("INSERT INTO StringIds VALUES (1, 'decode_step')")
    con.execute("INSERT INTO NVTX_EVENTS VALUES (0, 1000, NULL, 1)")
    con.commit()
    con.close()


def test_nsys_union_not_sum(tmp_path):
    db = tmp_path / "t.sqlite"
    make_nsys_db(db)
    act = device_activity(db)
    rep = busy_idle_report(act, window=(0, 1000))
    assert rep[0]["busy_ns"] == 900  # union of 0-800 and 900-1000
    assert rep[0]["kernel_sum_ns"] == 1100  # naive sum double-counts the overlap
    assert rep[0]["gap_count"] == 1 and rep[0]["longest_gap_ns"] == 100
    assert rep[0]["busy_fraction"] == pytest.approx(0.9)
    assert nvtx_ranges(db) == [{"start": 0, "end": 1000, "name": "decode_step"}]


def test_nsys_cuda_graph_executions_count_as_busy(tmp_path):
    db = tmp_path / "g.sqlite"
    make_nsys_db(db)
    con = sqlite3.connect(db)
    # A decode step replayed as one CUDA graph from 2000 to 2700 ns (no KERNEL rows for it).
    con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (0, 2000, 2700, 7)")
    con.commit()
    con.close()
    rep = busy_idle_report(device_activity(db), window=(0, 3000))
    assert rep[0]["busy_ns"] == 900 + 700


def test_stage_breakdown_from_server_events():
    """Shape taken from a real MiniCPM-o speech request (Spark s2): thinker -> talker -> code2wav in sequence."""
    ms = 1_000_000
    ev = lambda st, name, t: {"request_id": "r1", "stage": st, "event_name": name, "timestamp_ns": int(t * ms)}  # noqa: E731
    evs = [
        ev("coordinator", "request_admission", 0),
        ev("thinker", "stage_input_received", 2.0), ev("thinker", "stage_dispatch", 2.0),
        ev("thinker", "scheduler_queue_enter", 2.5), ev("thinker", "scheduler_prefill_start", 3.7),
        ev("thinker", "scheduler_prefill_end", 72.9), ev("thinker", "scheduler_first_emit", 73.0),
        ev("thinker", "model_path_start", 3.6), ev("thinker", "model_path_end", 3670.9),
        ev("thinker", "stage_complete", 3671.4), ev("thinker", "stage_hop_sent", 3672.4),
        ev("coordinator", "terminal_response", 3677.1),
        ev("talker", "stage_input_received", 3676.8), ev("talker", "stage_dispatch", 3676.9),
        ev("talker", "scheduler_queue_enter", 3678.1), ev("talker", "scheduler_prefill_start", 3678.7),
        ev("talker", "model_path_start", 3678.7), ev("talker", "model_path_end", 4826.8),
        ev("talker", "stage_complete", 4828.0), ev("talker", "stage_hop_sent", 4828.0),
        ev("code2wav", "stage_input_received", 4828.0), ev("code2wav", "stage_dispatch", 4828.1),
        ev("code2wav", "stage_complete", 5020.8), ev("coordinator", "terminal_response", 5021.0),
    ]
    d = EA.stage_breakdown(evs)["per_request"]["r1"]
    assert d["thinker.prefill"] == pytest.approx(69.2) and d["thinker.queue"] == pytest.approx(1.2)
    assert d["thinker.model_path"] == pytest.approx(3667.3)
    assert d["talker.model_path"] == pytest.approx(1148.1)
    assert d["code2wav.service"] == pytest.approx(192.7)
    assert d["hop.thinker->talker"] == pytest.approx(4.4)
    assert d["request.admission_to_first_terminal"] == pytest.approx(3677.1)
    assert d["request.admission_to_last_terminal"] == pytest.approx(5021.0)
    assert not any(k.startswith("encoder") or ".encoder" in k for k in d)  # nothing invented


def test_analyze_fake_experiment(tmp_path, monkeypatch):
    """Analysis of a real runner output (fake server): tables, agent summary, stage breakdown."""
    from test_experiment_runner import REPO, free_port, write_workload
    from omni_research.orchestration.experiment import ExperimentRunner

    monkeypatch.setenv("OMNI_REPO", str(REPO))
    monkeypatch.setenv("OMNI_PORT", str(free_port()))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    wl = write_workload(tmp_path, "w3", output_modalities=["text", "audio"])
    cfg = {"experiment_id": "fake", "data_root": str(tmp_path / "d"),
           "servers": [{"name": "s0", "command": f"{sys.executable} -m omni_research.replay.fake_server --port {{port}} --startup-s 1 --agent-json"}],
           "runtime": {"env": {"PYTHONPATH": str(REPO / "src")}, "collect_versions": False},
           "phases": [{"kind": "closed_loop", "name": "sweep", "workload": wl, "concurrency": [1, 4], "requests_per_level": 12,
                       "warmup": 2, "server_events": True},
                      {"kind": "agent", "name": "ag", "sessions": 4, "concurrency": 2}]}
    out = tmp_path / "exp"
    assert ExperimentRunner(cfg, out).run()["status"] == "ok"
    a = EA.analyze_experiment(out, slo={"ttft_s": 5.0})
    rows = {(r["phase"], r["level"]): r for r in a["table"]}
    assert rows[("sweep", "1")]["n"] == 12 and rows[("sweep", "4")]["errors"] == 0
    assert rows[("sweep", "4")]["ttfa_s_median"] is not None and rows[("sweep", "4")]["goodput_fraction"] == 1.0
    assert a["agents"]["ag"]["valid"] and a["agents"]["ag"]["sessions"] == 4
    split = a["agents"]["ag"]["time_split_s"]
    assert split["model"] > 0 and abs(sum(v for k, v in split.items() if k != "session") - split["session"]) < 1e-6
    assert a["stages"]["sweep"]["requests"] >= 24


def test_torch_chrome_trace_busy_idle(tmp_path):
    import gzip

    from omni_research.analysis.torch_trace import analyze_files

    tr = {"traceEvents": [
        {"ph": "X", "cat": "kernel", "name": "gemm", "ts": 0, "dur": 500, "args": {"device": 0}},
        {"ph": "X", "cat": "kernel", "name": "attn", "ts": 300, "dur": 500, "args": {"device": 0}},
        {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy HtoD", "ts": 900, "dur": 100, "args": {"device": 0}},
        {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 0, "dur": 2000, "args": {}},
    ]}
    p = tmp_path / "trace_rank0.trace.json.gz"
    with gzip.open(p, "wt") as fh:
        json.dump(tr, fh)
    out = analyze_files([p])
    d0 = out["per_device"][0]
    assert d0["busy_ns"] == 900_000 and d0["kernel_sum_ns"] == 1_100_000  # union vs naive sum, in ns
    assert out["files"][0]["kernels_top"][0][0] in ("gemm", "attn")


def test_gpu_timeline_without_memory_values(tmp_path):
    """GB10 reports memory as [N/A]: the plot must still render utilization."""
    from omni_research.analysis.plots import gpu_timeline

    samples = [{"t": 100.0 + i * 0.2, "gpu": "0", "util": float(i % 100), "mem_mib": None, "power_w": None} for i in range(20)]
    p = gpu_timeline(samples, [{"phase": "p", "started": 100.5, "ended": 102.0}], tmp_path, "t")
    assert p is not None and p.exists()


def test_compare_mixes_numeric_and_named_levels(tmp_path, monkeypatch):
    """Replay phases use the level "dependency"; sorting levels must not assume numbers (crashed 2026-11-11)."""
    from omni_research.analysis import compare as C

    def rows(e2e):
        return [{"phase": "p", "level": lv, "n": 12, "errors": 0, "e2e_s_median": e2e, "tp_requests_per_s": 1 / e2e}
                for lv in ("16", "dependency", "2")]

    tables = {"a": rows(1.0), "b": rows(1.5)}
    for name in tables:
        (tmp_path / name).mkdir()
        (tmp_path / name / "result.json").write_text("{}")
    monkeypatch.setattr(C.EA, "analyze_experiment",
                        lambda d: {"table": tables[Path(d).name], "status": "ok", "config": {}, "stages": {}})
    defn = {"name": "mixed", "phase": "p",
            "experiments": [{"label": "A", "dir": str(tmp_path / "a")}, {"label": "B", "dir": str(tmp_path / "b")}]}
    cmp = C.compare(defn, tmp_path)
    assert cmp["levels"] == ["2", "16", "dependency"]
    assert "负载水平 dependency" in C.render(cmp)
    assert C.figure(cmp, tmp_path / "fig").exists()


def test_torch_traces_per_process_busy_and_overlap(tmp_path):
    """One trace per process: per-process busy time and the time both processes had work in flight."""
    import gzip

    from omni_research.analysis.torch_trace import analyze_files

    def write(name, kernels):
        p = tmp_path / name
        with gzip.open(p, "wt") as fh:
            json.dump({"traceEvents": [{"ph": "X", "cat": "kernel", "name": "k", "ts": ts, "dur": dur, "args": {"device": 0}}
                                       for ts, dur in kernels]}, fh)
        return p

    # Process a is busy 0-600 us, process b 400-1000 us: 200 us with work from both.
    out = analyze_files([write("a.trace.json.gz", [(0, 300), (300, 300)]), write("b.trace.json.gz", [(400, 600)])])
    assert [f["busy_ns"][0] for f in out["files"]] == [600_000, 600_000]
    d0 = out["per_device"][0]
    assert d0["busy_ns"] == 1_000_000 and d0["multi_process_ns"] == 200_000
    single = analyze_files([tmp_path / "a.trace.json.gz"])
    assert "multi_process_ns" not in single["per_device"][0]


def test_prefix_cache_by_phase_from_server_log(tmp_path):
    """Prefill batch lines (local time, no zone) are assigned to phases via the step log's UTC offset."""
    from datetime import datetime, timedelta, timezone

    exp = tmp_path / "exp"
    (exp / "logs").mkdir(parents=True)
    (exp / "step.log").write_text("2026-11-11T17:00:00-05:00 starting\n")
    tz = timezone(timedelta(hours=-5))
    t0 = datetime(2026, 11, 11, 17, 5, 0, tzinfo=tz).timestamp()
    line = ("2026-11-11 17:05:{s:02d},500 [INFO] sglang.srt.managers.scheduler_components.metrics_reporter: "
            "Prefill batch, #new-seq: 1, #new-token: {n}, #cached-token: {c}, token usage: 0.01, #running-req: 0")
    (exp / "logs" / "server_s0.log").write_text("\n".join([
        line.format(s=1, n=382, c=0), line.format(s=2, n=54, c=378),  # phase a
        "unrelated line", line.format(s=30, n=10, c=90),               # phase b
        line.format(s=59, n=1, c=1)]) + "\n")                          # outside both phases
    phases = [{"phase": "a", "started": t0, "ended": t0 + 10}, {"phase": "b", "started": t0 + 20, "ended": t0 + 40}]
    pc = EA.prefix_cache_by_phase(exp, phases)
    assert pc["a"]["new_tokens"] == 436 and pc["a"]["cached_tokens"] == 378 and pc["a"]["prefill_batches"] == 2
    assert pc["b"]["cached_fraction"] == pytest.approx(0.9)
    (exp / "step.log").write_text("no timestamps\n")
    assert EA.prefix_cache_by_phase(exp, phases) == {}


def test_log_utc_offset_sources(tmp_path):
    """Manifest offset first; else a date -Is line; else server log header vs manifest start time."""
    from datetime import datetime, timedelta, timezone

    exp = tmp_path / "e"
    (exp / "logs").mkdir(parents=True)
    tz = timezone(timedelta(hours=-5))
    start = datetime(2026, 11, 11, 17, 28, 5, tzinfo=tz)
    (exp / "logs" / "server_s0.log").write_text("# 2026-11-11T17:28:11 CUDA_VISIBLE_DEVICES=0\n")
    (exp / "manifest.json").write_text(json.dumps({"started": start.timestamp()}))
    assert EA.log_utc_offset(exp) == timedelta(hours=-5)  # header 6 s after the runner started
    (exp / "manifest.json").write_text(json.dumps({"started": start.timestamp(), "utc_offset": "+0530"}))
    assert EA.log_utc_offset(exp) == timedelta(hours=5, minutes=30)


def test_server_failures_found_in_log_and_placed_in_phase(tmp_path):
    """A stage-process crash recorded only in the server log is reported with the phase it hit."""
    from datetime import datetime, timedelta, timezone

    exp = tmp_path / "e"
    (exp / "logs").mkdir(parents=True)
    (exp / "manifest.json").write_text(json.dumps({"utc_offset": "-0500"}))
    tz = timezone(timedelta(hours=-5))
    t0 = datetime(2026, 11, 11, 17, 26, 0, tzinfo=tz).timestamp()
    (exp / "logs" / "server_s0.log").write_text("\n".join([
        "2026-11-11 17:26:59,649 [INFO] Decode batch, #running-req: 2",
        "2026-11-11 17:27:00,523 [ERROR] sglang_omni.scheduling.simple_scheduler: compute_fn failed",
        "Traceback (most recent call last):",
        "RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered",
        "2026-11-11 17:27:01,000 [ERROR] Dead stage process(es) detected: talker (pid=614961, exit=-6)",
        "RuntimeError: CUDA error: an illegal memory access was encountered",
    ]) + "\n")
    phases = [{"phase": "w3b_c32_64", "started": t0 - 600, "ended": t0 - 10}, {"phase": "w3b_open", "started": t0 - 10, "ended": t0 + 120}]
    fails = EA.server_failures(exp, phases)
    kinds = {f["kind"]: f for f in fails}
    assert set(kinds) == {"illegal_memory_access", "dead_stage_process"}  # each pattern reported once
    assert kinds["illegal_memory_access"]["line"] == 4 and kinds["illegal_memory_access"]["phase"] == "w3b_open"
    assert kinds["illegal_memory_access"]["local_time"] == "2026-11-11 17:27:00"
