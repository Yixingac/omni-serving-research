"""Stall watcher (P3) with a fake log, a fake /proc and a fake signal function (no GPU, no real processes)."""

from __future__ import annotations

import json
import signal

from omni_research.orchestration.stall_watch import StallWatcher, proc_state, registered_stack_pids


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def fake_proc(root, pid, state="S", starttime=5000, wchan="futex_wait_queue"):
    d = root / str(pid)
    d.mkdir(parents=True, exist_ok=True)
    rest = [state] + ["0"] * 18 + [str(starttime)] + ["0"] * 10
    (d / "stat").write_text(f"{pid} (python3 (stage)) " + " ".join(rest) + "\n")
    (d / "wchan").write_text(wchan)


def test_proc_state_reads_state_wchan_and_starttime(tmp_path):
    fake_proc(tmp_path, 42, state="D", starttime=777, wchan="nfs_wait_bit_killable")
    s = proc_state(42, str(tmp_path))
    assert s == {"pid": 42, "state": "D", "starttime": 777, "wchan": "nfs_wait_bit_killable"}
    assert proc_state(43, str(tmp_path)) == {}


def test_only_processes_whose_start_time_matches_their_registration_are_signalled(tmp_path):
    proc, stacks = tmp_path / "proc", tmp_path / "stacks"
    stacks.mkdir()
    fake_proc(proc, 10, starttime=111)
    fake_proc(proc, 11, starttime=222)
    (stacks / "stack_10.txt").write_text("registered pid=10 starttime=111\n")
    (stacks / "stack_11.txt").write_text("registered pid=11 starttime=999\n")  # pid reused by another process
    (stacks / "stack_12.txt").write_text("registered pid=12 starttime=1\n")    # gone
    (stacks / "stack_13.txt").write_text("garbage\n")
    assert set(registered_stack_pids(stacks, str(proc))) == {10}


def test_a_pause_is_recorded_once_with_a_snapshot_and_its_end(tmp_path):
    log, out, proc, stacks = tmp_path / "server_s0.log", tmp_path / "stalls.jsonl", tmp_path / "proc", tmp_path / "stacks"
    stacks.mkdir()
    for pid in (10, 11):
        fake_proc(proc, pid, starttime=pid * 10)
    (stacks / "stack_10.txt").write_text("registered pid=10 starttime=100\n")
    sent = []
    clock = Clock()
    w = StallWatcher(log, lambda: [10, 11], out, stack_dir=stacks, gap_s=1.0, proc_root=str(proc),
                     signal_fn=lambda pid, sig: sent.append((pid, sig)), clock=clock)
    log.write_text("a\n")
    w.poll_once()
    for _ in range(5):  # 0.5 s of silence: not yet a pause
        clock.t += 0.1
        w.poll_once()
    assert not w.records
    for _ in range(10):  # 1.5 s of silence: one pause, one snapshot
        clock.t += 0.1
        w.poll_once()
    kinds = [r["event"] for r in w.records]
    assert kinds == ["pause_start", "pause_snapshot"]
    snap = w.records[1]
    assert [p["pid"] for p in snap["processes"]] == [10, 11] and snap["processes"][0]["wchan"] == "futex_wait_queue"
    assert sent == [(10, signal.SIGUSR2)] and snap["stack_signals"] == [{"pid": 10, "file": "stack_10.txt"}]
    log.write_text("a\nb\n")  # the log grows again
    clock.t += 0.1
    w.poll_once()
    assert w.records[-1]["event"] == "pause_end" and w.records[-1]["silent_s"] >= 1.5
    assert [json.loads(x)["event"] for x in out.read_text().splitlines()] == ["pause_start", "pause_snapshot", "pause_end"]


def test_snapshots_are_rate_limited_and_capped(tmp_path):
    log, out = tmp_path / "server_s0.log", tmp_path / "stalls.jsonl"
    clock = Clock()
    w = StallWatcher(log, lambda: [], out, gap_s=1.0, min_interval_s=30.0, max_dumps=2, proc_root=str(tmp_path), clock=clock)
    text = ""
    for k in range(6):  # six pauses 10 s apart
        text += f"{k}\n"
        log.write_text(text)
        w.poll_once()
        clock.t += 2.0
        w.poll_once()
        clock.t += 8.0
    starts = [r for r in w.records if r["event"] == "pause_start"]
    snaps = [r for r in w.records if r["event"] == "pause_snapshot"]
    assert len(starts) == 6 and len(snaps) == 2


def test_slow_stat_is_recorded(tmp_path):
    log, out = tmp_path / "server_s0.log", tmp_path / "stalls.jsonl"
    log.write_text("a\n")
    ticks = iter([0.0, 0.0, 0.5])  # construction, before the stat, after it: the stat takes 0.5 s

    w = StallWatcher(log, lambda: [], out, slow_stat_s=0.2, clock=lambda: next(ticks))
    w.poll_once()
    assert w.records and w.records[0]["event"] == "slow_stat" and w.records[0]["stat_s"] == 0.5


def test_py_spy_is_skipped_with_the_reason_when_ptrace_is_restricted(tmp_path, monkeypatch):
    import omni_research.orchestration.stall_watch as sw

    monkeypatch.setattr(sw, "ptrace_scope", lambda path="": 1)
    w = StallWatcher(tmp_path / "log", lambda: [1], tmp_path / "out.jsonl", py_spy="/usr/bin/py-spy", proc_root=str(tmp_path),
                     run_fn=lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")))
    assert "ptrace_scope=1" in w.snapshot(2.0)["py_spy"]
    monkeypatch.setattr(sw, "ptrace_scope", lambda path="": 0)

    class Res:
        stdout, stderr = "Thread 1 (idle)\n  wait (threading.py:300)\n", ""

    w.run_fn = lambda *a, **k: Res()
    assert w.snapshot(2.0)["py_spy"] == {"1": Res.stdout}


def test_tree_pids_includes_children_of_live_processes():
    import os
    import subprocess
    import sys

    from omni_research.orchestration.stall_watch import tree_pids

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        pids = tree_pids([os.getpid(), 999999999])
        assert os.getpid() in pids and child.pid in pids and 999999999 not in pids
    finally:
        child.kill()
        child.wait()


def test_a_disarmed_watcher_records_no_pause(tmp_path):
    log, out = tmp_path / "server_s0.log", tmp_path / "stalls.jsonl"
    clock = Clock()
    w = StallWatcher(log, lambda: [], out, gap_s=1.0, proc_root=str(tmp_path), clock=clock)
    w.disarm()
    log.write_text("a\n")
    w.poll_once()
    clock.t += 5.0
    w.poll_once()
    assert not w.records
    w.arm()  # silence is counted from the start of the load phase
    w.poll_once()
    assert not w.records
    clock.t += 1.5
    w.poll_once()
    assert [r["event"] for r in w.records] == ["pause_start", "pause_snapshot"]
    w.disarm()
    assert w.armed is False


def test_runner_hook_starts_arms_and_stops_the_watcher_only_when_enabled(tmp_path, monkeypatch):
    """experiment.py: watchers exist only with "stall_watch": true, are armed only while load is offered."""
    import json as _json
    from pathlib import Path
    import socket
    import sys as _sys

    from omni_research.orchestration import stall_watch as sw
    from omni_research.orchestration.experiment import ExperimentRunner

    calls = []

    class FakeWatcher:
        def __init__(self, log_path, pids_fn, out_path, **kw):
            calls.append(("init", Path(out_path).name))

        def start(self):
            calls.append(("start",))

        def arm(self):
            calls.append(("arm",))

        def disarm(self):
            calls.append(("disarm",))

        def stop(self):
            calls.append(("stop",))

    monkeypatch.setattr(sw, "StallWatcher", FakeWatcher)
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("OMNI_REPO", str(repo))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        monkeypatch.setenv("OMNI_PORT", str(s.getsockname()[1]))
    wl = tmp_path / "w1.json"
    wl.write_text(_json.dumps({"workload_id": "w1", "input_modalities": ["text"], "output_modalities": ["text"],
                               "max_tokens": 8, "prompt": {"kind": "synthetic_text", "tokens": 32}}))
    cfg = {"experiment_id": "test-stall-hook", "data_root": str(tmp_path / "data"),
           "servers": [{"name": "s0", "command": f"{_sys.executable} -m omni_research.replay.fake_server --port {{port}} --startup-s 1",
                        "ready_timeout_s": 60}],
           "runtime": {"env": {"PYTHONPATH": str(repo / "src")}, "collect_versions": False},
           "phases": [{"kind": "smoke", "name": "smoke", "workload": str(wl), "requests": 1},
                      {"kind": "closed_loop", "name": "c", "workload": str(wl), "concurrency": [1], "requests_per_level": 2}]}
    res = ExperimentRunner(dict(cfg, stall_watch=True), tmp_path / "on").run()
    assert res["status"] == "ok", res.get("error")
    names = [c[0] for c in calls]
    assert names[:3] == ["init", "disarm", "start"] and calls[0][1] == "stalls_s0.jsonl"
    assert names.count("arm") == 1 and names[-1] == "stop"  # armed for the closed loop only, not the smoke phase
    calls.clear()
    res = ExperimentRunner(cfg, tmp_path / "off").run()
    assert res["status"] == "ok" and calls == []
