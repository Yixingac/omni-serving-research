"""Scheduler tests with a fake launcher and a virtual clock (no processes, no Slurm)."""

import json
from pathlib import Path

import pytest

from omni_research.orchestration.launcher import Assignment, Handle, SlurmStepLauncher
from omni_research.orchestration.plan import Experiment, Plan, load_plan, validate_plan
from omni_research.orchestration.resources import Allocation, NodeResources, parse_scontrol_job
from omni_research.orchestration.scheduler import Orchestrator
from omni_research.orchestration.state import StateStore


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeProc:
    pid = 0


class FakeLauncher:
    """Each experiment 'runs' for its scripted duration and exits with its scripted outcome.

    outcomes: experiment_id -> list of (duration_s, exit_code, result_json_or_None), one per attempt.
    """

    def __init__(self, clock, outcomes):
        self.clock = clock
        self.outcomes = {k: list(v) for k, v in outcomes.items()}
        self.launched = []  # (time, experiment_id, gpus)
        self.terminated = []
        self.max_gpus_in_use = 0
        self._running = {}

    def launch(self, exp, a, out_dir, env):
        dur, rc, result = self.outcomes[exp.experiment_id].pop(0)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.launched.append((self.clock(), exp.experiment_id, a.gpus))
        h = Handle(exp.experiment_id, FakeProc(), self.clock(), out_dir, a, step_name=f"fake:{exp.experiment_id}")
        self._running[exp.experiment_id] = (self.clock() + dur, rc, result)
        self.max_gpus_in_use = max(self.max_gpus_in_use, sum(x.assignment.gpus for x in self._handles(h)))
        return h

    def _handles(self, new):
        return [new]

    def poll(self, h):
        end, rc, result = self._running[h.experiment_id]
        if self.clock() >= end:
            if result is not None:
                (h.out_dir / "result.json").write_text(json.dumps(result))
            return rc
        return None

    def terminate(self, h, grace_s=60.0):
        self.terminated.append(h.experiment_id)
        return -15


def make(tmp_path, exps, outcomes, gpus=4, cpus=32, mem=256, time_left=1e9, margin=100):
    clock = Clock()
    plan = Plan("p", {e.experiment_id: e for e in exps}, [e.experiment_id for e in exps])
    validate_plan(plan)
    alloc = Allocation("local", None, [NodeResources("n0", gpus, cpus, mem)])
    launcher = FakeLauncher(clock, outcomes)
    tl = time_left if callable(time_left) else (lambda: time_left - clock())
    orch = Orchestrator(plan, StateStore(tmp_path / "run"), alloc, launcher, time_left_fn=tl, stop_margin_s=margin,
                        poll_s=1.0, clock=clock, sleep=clock.sleep)
    return orch, launcher, clock


def E(eid, gpus=1, prio=0, deps=(), dur=10, **kw):
    return Experiment(experiment_id=eid, command="true", required_gpus=gpus, required_cpus=1, required_memory_gb=1,
                      priority=prio, dependencies=list(deps), estimated_duration_s=dur, **kw)


def test_dependencies_and_priority_order(tmp_path):
    exps = [E("a", prio=1), E("b", prio=5, deps=["a"]), E("c", prio=9)]
    orch, launcher, clock = make(tmp_path, exps, {k: [(10, 0, {"status": "ok"})] for k in "abc"}, gpus=1)
    st = orch.run()
    assert [x[1] for x in launcher.launched] == ["c", "a", "b"]  # c has top priority; b waits for a
    assert {k: v.status for k, v in st.experiments.items()} == {"a": "SUCCEEDED", "b": "SUCCEEDED", "c": "SUCCEEDED"}
    assert st.experiments["b"].gpu_seconds == pytest.approx(10, abs=1.5)


def test_packing_respects_gpu_budget(tmp_path):
    exps = [E(f"e{i}", gpus=2) for i in range(4)]
    orch, launcher, clock = make(tmp_path, exps, {f"e{i}": [(10, 0, None)] for i in range(4)}, gpus=4)
    orch.run()
    starts = sorted(t for t, _, _ in launcher.launched)
    assert starts[:2] == [0, 0] and starts[2] >= 10  # only two 2-GPU experiments fit at once


def test_failure_blocks_dependents_and_no_identical_retry(tmp_path):
    exps = [E("a", max_retries=2), E("b", deps=["a"]), E("c", deps=["b"])]
    outcomes = {"a": [(5, 1, {"status": "failed", "error_class": "oom", "error": "CUDA OOM"})]}
    orch, launcher, clock = make(tmp_path, exps, outcomes)
    st = orch.run()
    assert st.experiments["a"].status == "FAILED" and st.experiments["a"].attempts == 1  # oom is not retried
    assert st.experiments["b"].status == "BLOCKED" and st.experiments["c"].status == "BLOCKED"
    assert "dependency a is FAILED" in st.experiments["b"].reason


def test_transient_failure_is_retried_up_to_limit(tmp_path):
    exps = [E("a", max_retries=1)]
    fail = (5, 1, {"status": "failed", "error_class": "server_start_timeout", "error": "no /health"})
    orch, launcher, clock = make(tmp_path, exps, {"a": [fail, fail]})
    st = orch.run()
    assert st.experiments["a"].attempts == 2 and st.experiments["a"].status == "FAILED"
    assert [f["class"] for f in st.experiments["a"].failures] == ["server_start_timeout"] * 2


def test_insufficient_time_and_oversized_are_skipped(tmp_path):
    exps = [E("long", dur=500), E("huge", gpus=16), E("ok", dur=10)]
    orch, launcher, clock = make(tmp_path, exps, {"ok": [(10, 0, None)]}, time_left=400, margin=100)
    st = orch.run()
    assert st.experiments["long"].status == "SKIPPED" and "insufficient_time" in st.experiments["long"].reason
    assert st.experiments["huge"].status == "SKIPPED"
    assert st.experiments["ok"].status == "SUCCEEDED"


def test_backfill_does_not_starve_big_experiment(tmp_path):
    # 4 GPUs. "small1" (1 GPU, 100 s) runs; "big" (4 GPUs, high priority) must wait for it. A short
    # 1-GPU job may backfill only if it finishes before big can start.
    exps = [E("small1", gpus=1, prio=10, dur=100), E("big", gpus=4, prio=5, dur=50),
            E("short", gpus=1, prio=1, dur=20), E("long", gpus=1, prio=1, dur=500)]
    outcomes = {"small1": [(100, 0, None)], "big": [(50, 0, None)], "short": [(20, 0, None)], "long": [(500, 0, None)]}
    orch, launcher, clock = make(tmp_path, exps, outcomes, gpus=4)
    orch.run()
    t = {eid: when for when, eid, _ in launcher.launched}
    assert t["short"] == 0  # fits before the reservation at t=100
    assert t["big"] == pytest.approx(100, abs=1.5)
    assert t["long"] >= t["big"]  # would have delayed big, so it waited


def test_timeout_kills_and_records(tmp_path):
    exps = [E("slow", timeout_s=30)]
    orch, launcher, clock = make(tmp_path, exps, {"slow": [(1000, 0, None)]})
    st = orch.run()
    assert launcher.terminated == ["slow"]
    assert st.experiments["slow"].status == "FAILED" and st.experiments["slow"].failures[0]["class"] == "timeout"


def test_walltime_margin_triggers_shutdown(tmp_path):
    exps = [E("a", dur=10), E("b", dur=10, deps=["a"])]
    outcomes = {"a": [(1000, 0, None)], "b": [(10, 0, None)]}
    orch, launcher, clock = make(tmp_path, exps, outcomes, time_left=300, margin=100)
    st = orch.run()
    assert st.shutdown_reason == "walltime margin reached"
    assert launcher.terminated == ["a"]
    assert st.experiments["a"].failures[-1]["class"] == "interrupted"
    assert clock() <= 205  # stopped at the margin, not at the end of the allocation


def test_exit_zero_with_failed_result_is_failure(tmp_path):
    exps = [E("a")]
    orch, launcher, clock = make(tmp_path, exps, {"a": [(5, 0, {"status": "failed", "error_class": "bad_output", "error": "empty audio"})]})
    st = orch.run()
    assert st.experiments["a"].status == "FAILED" and st.experiments["a"].failures[0]["class"] == "bad_output"


def test_plan_additions_are_picked_up(tmp_path):
    exps = [E("a", dur=10)]
    orch, launcher, clock = make(tmp_path, exps, {"a": [(10, 0, None)], "follow": [(5, 0, None)]})
    add = tmp_path / "run" / "plan.d"
    add.mkdir(parents=True)
    (add / "001-follow.json").write_text(json.dumps({"experiments": [
        {"experiment_id": "follow", "command": "true", "dependencies": ["a"], "estimated_duration_s": 5,
         "required_gpus": 1, "required_cpus": 1, "required_memory_gb": 1}]}))
    st = orch.run()
    assert st.experiments["follow"].status == "SUCCEEDED"


def test_keep_alive_runs_late_additions_until_stop(tmp_path):
    """Hold mode: the queue drains, a plan.d entry added later still runs, and the stop file ends the run."""
    orch, launcher, clock = make(tmp_path, [E("a", dur=10)], {"a": [(10, 0, None)], "late": [(5, 0, None)]})
    stop = tmp_path / "STOP"
    orch.keep_alive, orch.stop_file = True, stop
    add = tmp_path / "run" / "plan.d"

    def sleep(s):
        clock.sleep(s)
        if clock() >= 30 and not add.exists():
            add.mkdir(parents=True)
            (add / "001-late.json").write_text(json.dumps({"experiments": [
                {"experiment_id": "late", "command": "true", "estimated_duration_s": 5,
                 "required_gpus": 1, "required_cpus": 1, "required_memory_gb": 1}]}))
        if clock() >= 60:
            stop.touch()

    orch.sleep = sleep
    st = orch.run()
    assert st.experiments["a"].status == "SUCCEEDED" and st.experiments["late"].status == "SUCCEEDED"
    assert [x[1] for x in launcher.launched] == ["a", "late"] and clock() >= 60


def test_stop_file_lets_running_finish_and_skips_the_rest(tmp_path):
    orch, launcher, clock = make(tmp_path, [E("a", dur=10), E("b", gpus=4, dur=10)],
                                 {"a": [(10, 0, None)], "b": [(10, 0, None)]})
    stop = tmp_path / "STOP"
    orch.stop_file = stop

    def sleep(s):
        clock.sleep(s)
        if clock() >= 3:
            stop.touch()

    orch.sleep = sleep
    st = orch.run()
    assert st.experiments["a"].status == "SUCCEEDED"  # running when STOP appeared: finished normally
    assert st.experiments["b"].status == "SKIPPED" and "stop file" in st.experiments["b"].reason
    assert [x[1] for x in launcher.launched] == ["a"]


def test_plan_validation_errors(tmp_path):
    with pytest.raises(ValueError, match="unknown experiment"):
        validate_plan(Plan("p", {"a": E("a", deps=["zz"])}, ["a"]))
    with pytest.raises(ValueError, match="cycle"):
        validate_plan(Plan("p", {"a": E("a", deps=["b"]), "b": E("b", deps=["a"])}, ["a", "b"]))
    p = tmp_path / "plan.json"
    p.write_text(json.dumps({"plan_id": "x", "defaults": {"required_cpus": 3},
                             "experiments": [{"experiment_id": "q", "command": "echo hi"}]}))
    plan = load_plan(p)
    assert plan.experiments["q"].required_cpus == 3 and plan.experiments["q"].output_path == "q"
    p.write_text(json.dumps({"experiments": [{"experiment_id": "q", "command": "x", "gpus": 1}]}))
    with pytest.raises(ValueError, match="unknown fields"):
        load_plan(p)


def test_srun_argv_requests_exact_resources():
    lz = SlurmStepLauncher("123", "/repo", gpu_type="nvidia_rtx_a6000")
    a = Assignment(node="nodeA", gpus=2, cpus=12, mem_gb=96, port=21000)
    argv = lz.srun_argv(E("x", gpus=2), a, Path("/out"))
    assert "--jobid=123" in argv and "--exact" in argv and "--gres=gpu:nvidia_rtx_a6000:2" in argv
    assert "--cpus-per-task=12" in argv and "--mem=98304M" in argv and "--nodelist=nodeA" in argv
    cpu_only = lz.srun_argv(E("y", gpus=0), Assignment("nodeA", 0, 2, 4, 21010), Path("/out"))
    assert "--gres=none" in cpu_only  # otherwise the step would inherit all of the job's GPUs


def test_parse_scontrol_job_detail():
    text = """JobId=409712 JobName=omni-research-hold
   JobState=RUNNING Reason=None Dependency=(null)
   StartTime=2026-11-11T19:19:38 EndTime=2026-11-12T05:19:38 Deadline=N/A
   NodeList=node-a
     Nodes=node-a CPU_IDs=0-11,16-27 Mem=307200 GRES=gpu:nvidia_rtx_a6000:3(IDX:0-2)
"""
    info = parse_scontrol_job(text)
    assert info["state"] == "RUNNING" and info["end_time_str"] == "2026-11-12T05:19:38"
    assert info["nodes"] == [{"nodes": "node-a", "cpus": 24, "mem_mb": 307200, "gpus": 3,
                              "gpu_type": "nvidia_rtx_a6000"}]
