"""Audit of finished runs: run, phase, request and profiler status are kept apart (synthetic run dirs only)."""

import gzip
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("audit_runs", HERE / "audit_runs.py")
au = importlib.util.module_from_spec(spec)
spec.loader.exec_module(au)


def make_run(root: Path) -> Path:
    exp = root / "20261117_demo_block1_run01"
    (exp / "raw").mkdir(parents=True)
    result = {"status": "ok", "experiment_id": "demo_block1", "wall_s": 100.0,
              "server_restarts": [{"server": "s0", "exit_code": 1, "where": "phase w_c64_C"}],
              "phases": [{"phase": "w_c64_B", "kind": "closed_loop"}, {"phase": "w_c64_C", "kind": "closed_loop"},
                         {"phase": "w_c128_C", "kind": "closed_loop"}]}
    (exp / "result.json").write_text(json.dumps(result))
    rows = [{"request_id": "a", "status": "ok", "warmup": True}, {"request_id": "b", "status": "ok"},
            {"request_id": "c", "status": "error"}, {"request_id": "d", "status": "timeout"},
            {"request_id": "e", "status": "ok", "crash_affected": True}]
    (exp / "raw" / "requests_w_c64_C.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (exp / "raw" / "requests_w_c64_B.jsonl").write_text(json.dumps({"request_id": "f", "status": "ok"}) + "\n")
    for phase, procs in (("w_c64_C", ["talker@r0", "pipeline@r0"]), ("w_c128_C", ["talker@r0", "pipeline@r0", "code2wav@r0"])):
        d = exp / "profiling" / "torch" / phase / "s0"
        d.mkdir(parents=True)
        for p in procs:
            with gzip.open(d / f"{p}_trace_pid1_rank0.trace.json.gz", "wt") as fh:
                fh.write("{}")
    return exp


def test_phase_requests_restarts_and_missing_traces_are_reported_separately(tmp_path):
    a = au.audit_run(make_run(tmp_path))
    assert a["run_status"] == "ok" and a["restarts"] == ["s0 exit 1 in phase w_c64_C"]
    c64 = a["phases"]["w_c64_C"]
    assert c64["requests"] == {"measured": 4, "ok": 2, "error": 1, "timeout": 1, "crash_affected": 1}
    assert c64["restart_here"] is True
    # the largest set of traced processes in this run is the reference; code2wav@r0 is missing in w_c64_C
    assert c64["traces"] == {"files": 2, "missing": ["code2wav@r0"]}
    assert a["phases"]["w_c128_C"]["traces"]["missing"] == []
    assert a["phases"]["w_c64_B"]["traces"] is None and a["phases"]["w_c64_B"]["restart_here"] is False
    assert a["phases"]["w_c128_C"]["requests"] is None  # no request file: unknown, not zero


def test_summary_counts_runs_with_problems(tmp_path):
    a = au.audit_run(make_run(tmp_path))
    s = au.summarize([a])
    assert s == {"runs": 1, "runs_ok": 1, "runs_with_restarts": 1, "phases_with_request_errors": 1,
                 "measured_requests": 5, "ok_requests": 3, "error_requests": 1, "timeout_requests": 1,
                 "phases_with_missing_traces": 1}


def test_pipeline_stage_names_are_one_process_role(tmp_path):
    # the pipeline process names its trace after whichever stage it saw first (preprocessing, thinker, decode, ...)
    exp = tmp_path / "20261117_demo2_block1_run01"
    (exp / "raw").mkdir(parents=True)
    (exp / "result.json").write_text(json.dumps({"status": "ok", "phases": [{"phase": "a_C"}, {"phase": "b_C"}]}))
    for phase, procs in (("a_C", ["preprocessing@r0", "code2wav@r1"]), ("b_C", ["decode@r0", "thinker@r0"])):
        d = exp / "profiling" / "torch" / phase / "s0"
        d.mkdir(parents=True)
        for p in procs:
            with gzip.open(d / f"{p}_trace_pid7_rank0.trace.json.gz", "wt") as fh:
                fh.write("{}")
    a = au.audit_run(exp)
    assert a["phases"]["a_C"]["traces"]["missing"] == []
    assert a["phases"]["b_C"]["traces"]["missing"] == ["code2wav@r1"]


def test_a_restart_after_a_closed_loop_level_is_attributed_to_its_phase(tmp_path):
    exp = make_run(tmp_path)
    res = json.loads((exp / "result.json").read_text())
    res["server_restarts"] = [{"server": "s0", "exit_code": -11, "where": "w_c64_C c=64 rep=0"}]  # the runner's format
    (exp / "result.json").write_text(json.dumps(res))
    a = au.audit_run(exp)
    assert a["phases"]["w_c64_C"]["restart_here"] is True and a["phases"]["w_c64_B"]["restart_here"] is False
