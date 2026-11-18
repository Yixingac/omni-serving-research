"""Recovery: a restarted orchestrator must not redo finished work and must retry interrupted work."""

import json
import os

from omni_research.orchestration.state import RunState, StateStore, atomic_write_json
from test_orchestrator import E, make


def test_atomic_write_leaves_no_partial_file(tmp_path):
    p = tmp_path / "state.json"
    atomic_write_json(p, {"a": 1})
    atomic_write_json(p, {"a": 2})
    assert json.loads(p.read_text()) == {"a": 2}
    assert [f for f in os.listdir(tmp_path) if f.startswith(".state.json.")] == []


def test_resume_skips_succeeded_and_retries_interrupted(tmp_path):
    exps = [E("done", dur=5), E("was_running", dur=5, max_retries=1), E("todo", dur=5)]
    store = StateStore(tmp_path / "run")
    # Simulate a previous orchestrator that died: one experiment finished, one was mid-run.
    prev = RunState(plan_id="p", run_dir=str(tmp_path / "run"))
    from omni_research.orchestration.state import ExperimentState

    prev.experiments = {
        "done": ExperimentState(status="SUCCEEDED", attempts=1),
        "was_running": ExperimentState(status="RUNNING", attempts=1),
        "todo": ExperimentState(status="PENDING"),
    }
    store.save(prev)
    orch, launcher, clock = make(tmp_path, exps, {"was_running": [(5, 0, None)], "todo": [(5, 0, None)]})
    st = orch.run()
    launched = [eid for _, eid, _ in launcher.launched]
    assert "done" not in launched  # never re-run a finished experiment
    assert sorted(launched) == ["todo", "was_running"]
    assert st.experiments["was_running"].attempts == 2
    assert st.experiments["was_running"].failures[0]["class"] == "interrupted"
    assert all(es.status == "SUCCEEDED" for es in st.experiments.values())
    events = [json.loads(line)["event"] for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    assert "resume" in events and events[-1] == "run_finished"


def test_interrupted_without_retry_budget_fails(tmp_path):
    exps = [E("x", dur=5, max_retries=0)]
    store = StateStore(tmp_path / "run")
    from omni_research.orchestration.state import ExperimentState

    prev = RunState(plan_id="p", run_dir=str(tmp_path / "run"))
    prev.experiments = {"x": ExperimentState(status="RUNNING", attempts=1)}
    store.save(prev)
    orch, launcher, clock = make(tmp_path, exps, {})
    st = orch.run()
    assert st.experiments["x"].status == "FAILED" and launcher.launched == []


def test_state_roundtrip(tmp_path):
    store = StateStore(tmp_path)
    exps = [E("a")]
    orch, launcher, clock = make(tmp_path, exps, {"a": [(3, 0, None)]})
    orch.run()
    loaded = StateStore(tmp_path / "run").load()
    assert loaded.experiments["a"].status == "SUCCEEDED"
    assert loaded.allocation["mode"] == "local"
