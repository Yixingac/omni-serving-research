"""Plan v2 runner features against the CPU-only fake server: restart after a server exits, open-loop
rates derived from measured capacity, per-level request counts, mixed workloads, unique agent tasks
and a router that skips a replica that is down. No GPU, no network beyond localhost."""

import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import pytest

from omni_research.orchestration.experiment import ExperimentRunner, Server, mark_crash_affected
from omni_research.replay.backends import BackendPool, ModelRequest, RequestResult
from omni_research.replay.payloads import Workload, WorkloadMix
from omni_research.tracing.agent_harness import make_knowledge_base, make_tasks

from test_experiment_runner import REPO, free_port, write_workload


@pytest.fixture
def base(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNI_REPO", str(REPO))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("OMNI_PORT", str(free_port()))
    text = write_workload(tmp_path, "w1")
    speech = write_workload(tmp_path, "w3", output_modalities=["text", "audio"])
    cfg = {
        "experiment_id": "test-v2",
        "data_root": str(tmp_path / "data"),
        "servers": [{"name": "s0", "command": f"{sys.executable} -m omni_research.replay.fake_server --port {{port}} --startup-s 1 --agent-json",
                     "ready_timeout_s": 60}],
        "runtime": {"env": {"PYTHONPATH": str(REPO / "src")}, "collect_versions": False},
        "phases": [],
    }
    return cfg, text, speech


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def test_restart_marks_and_reruns_the_crashed_level(base, tmp_path):
    cfg, text, speech = base
    cfg["servers"][0]["command"] += f" --die-after 4 --die-marker {tmp_path / 'died'}"
    cfg["restart_dead_servers"] = True
    cfg["phases"] = [
        {"kind": "smoke", "name": "smoke", "workload": text, "requests": 2},
        {"kind": "closed_loop", "name": "sweep", "workload": speech, "concurrency": [1, 2], "requests_per_level": 6, "warmup": 1},
        {"kind": "open_loop", "name": "poisson", "workload": text, "rates_rps": [20], "requests_per_rate": 5},
    ]
    out = tmp_path / "exp"
    res = ExperimentRunner(cfg, out).run()
    assert res["status"] == "ok", res.get("error")
    assert len(res["server_restarts"]) == 1
    ev = res["server_restarts"][0]
    assert ev["server"] == "s0" and ev["exit_code"] == 134 and ev["where"].startswith("sweep c=1")
    levels = res["phases"][1]["levels"]
    assert [(lv["concurrency"], bool(lv.get("crash_affected")), bool(lv.get("rerun_after_restart"))) for lv in levels] == [
        (1, True, False), (1, False, True), (2, False, False)]
    assert levels[1]["errors"] == 0 and levels[2]["errors"] == 0
    recs = read_jsonl(out / "raw" / "requests_sweep.jsonl")
    flagged = [r for r in recs if r.get("crash_affected")]
    assert len(flagged) == 7  # warmup + 6 measured requests of the first c=1 attempt
    assert all(not r.get("crash_affected") for r in recs if r["concurrency"] == 2)
    assert json.loads((out / "analysis" / "server_restarts.json").read_text())[0]["log_tail"]
    assert res["phases"][2]["rates"][0]["errors"] == 0  # the restarted server serves the next phase


def test_without_restart_a_crash_still_fails_the_experiment(base, tmp_path):
    cfg, text, speech = base
    cfg["servers"][0]["command"] += " --die-after 3"
    cfg["phases"] = [{"kind": "closed_loop", "name": "sweep", "workload": speech, "concurrency": [1, 2],
                      "requests_per_level": 6, "warmup": 1}]
    res = ExperimentRunner(cfg, tmp_path / "exp").run()
    assert res["status"] == "failed" and res["error_class"] == "server_died"
    levels = res["phases"][0]["levels"]
    assert len(levels) == 1 and levels[0]["crash_affected"] and res["phases"][0]["stopped_reason"].startswith("server exited")
    assert res["server_restarts"] == []


def test_restart_budget_is_enforced(base, tmp_path):
    cfg, text, speech = base
    cfg["servers"][0]["command"] += " --die-after 2"  # dies again after every restart
    cfg["restart_dead_servers"] = True
    cfg["max_server_restarts"] = 2
    cfg["phases"] = [{"kind": "closed_loop", "name": "sweep", "workload": speech, "concurrency": [1, 1, 1],
                      "requests_per_level": 4, "warmup": 0}]
    res = ExperimentRunner(cfg, tmp_path / "exp").run()
    assert res["status"] == "failed" and res["error_class"] == "server_died"
    assert "restart budget of 2" in res["error"]
    assert len(res["server_restarts"]) == 2


def test_capacity_rates_and_requests_per_concurrency(base, tmp_path):
    cfg, text, speech = base
    cfg["phases"] = [
        {"kind": "closed_loop", "name": "sweep", "workload": text, "concurrency": [2, 8], "requests_per_level": 4,
         "requests_per_concurrency": 1, "warmup": 0},
        {"kind": "open_loop", "name": "open", "workload": text, "capacity_from": "sweep", "capacity_fractions": [0.25, 0.5],
         "duration_per_rate_s": 1, "min_requests_per_rate": 3},
        {"kind": "open_loop", "name": "open_missing", "workload": text, "capacity_from": "nope", "capacity_fractions": [0.5]},
    ]
    res = ExperimentRunner(cfg, tmp_path / "exp").run()
    assert res["status"] == "ok", res.get("error")
    assert [lv["n"] for lv in res["phases"][0]["levels"]] == [4, 8]
    cap = max(lv["throughput"]["requests_per_s"] for lv in res["phases"][0]["levels"])
    op = res["phases"][1]
    assert op["capacity_rps"] == pytest.approx(cap)
    assert [r["rate_rps"] for r in op["rates"]] == [round(cap * 0.25, 3), round(cap * 0.5, 3)]
    for r in op["rates"]:
        assert r["requests"] == max(3, -(-int(r["rate_rps"] * 1000) // 1000))
    assert res["phases"][2]["rates"] == [] and "no clean throughput" in res["phases"][2]["skipped"]


def test_workload_mix_records_each_component(base, tmp_path):
    cfg, text, speech = base
    cfg["phases"] = [{"kind": "closed_loop", "name": "mix", "concurrency": [4], "requests_per_level": 40, "warmup": 0,
                      "workload_mix": [{"workload": text, "weight": 0.5}, {"workload": speech, "weight": 0.5}]}]
    out = tmp_path / "exp"
    res = ExperimentRunner(cfg, out).run()
    assert res["status"] == "ok", res.get("error")
    ids = [r["workload_id"] for r in read_jsonl(out / "raw" / "requests_mix.jsonl")]
    assert set(ids) == {"w1", "w3"}
    audio = [r for r in read_jsonl(out / "raw" / "requests_mix.jsonl") if r["workload_id"] == "w3"]
    assert all(r["output_audio_s"] > 0 for r in audio)


def test_workload_mix_weights_and_determinism():
    a = Workload(workload_id="a", input_modalities=["text"], output_modalities=["text"])
    b = Workload(workload_id="b", input_modalities=["text"], output_modalities=["text"])
    mix = WorkloadMix([(a, 0.8), (b, 0.2)])
    picks = [mix.pick(random.Random(i)).workload_id for i in range(5000)]
    assert 0.77 < picks.count("a") / len(picks) < 0.83
    assert [mix.pick(random.Random(i)).workload_id for i in range(50)] == picks[:50]
    with pytest.raises(ValueError):
        WorkloadMix([(a, 0.0)])


def test_ticket_tasks_are_unique_and_keep_answers():
    kb = make_knowledge_base(7)
    plain = make_tasks(600, 5, kb)
    tick = make_tasks(600, 5, kb, ticket=True)
    assert len({t.question for t in plain}) < 600  # the bare question space repeats at this size
    assert len({t.question for t in tick}) == 600
    for p, t in zip(plain, tick):
        assert t.answer == p.answer and t.question.endswith(p.question)
        code = t.question.split(".")[0].removeprefix("Ticket ")
        assert len(code) == 6 and code.isalpha()


class _Fake:
    def __init__(self, name, fail=False):
        self.name, self.fail, self.calls = name, fail, 0

    async def run(self, req):
        self.calls += 1
        res = RequestResult(request_id=req.request_id, backend=self.name)
        if self.fail:
            res.status, res.error = "error", "ClientConnectorError: refused"
        return res

    async def close(self):
        pass


@pytest.mark.parametrize("policy", ["least_outstanding", "round_robin", "session_affinity"])
def test_pool_skips_a_replica_that_refused_connections(policy):
    bad, good = _Fake("bad", fail=True), _Fake("good")
    pool = BackendPool([bad, good], policy=policy, down_cooldown_s=60)

    def req(i):
        return ModelRequest(request_id=f"r{i}", messages=[{"role": "user", "content": "x"}], input_modalities=["text"],
                            output_modalities=["text"], metadata={"session_id": "s0"})

    async def go():
        # session "s0" may hash to either replica; send until the bad one has been tried once
        for i in range(4):
            await pool.run(req(i))
        n_bad = bad.calls
        for i in range(4, 20):
            await pool.run(req(i))
        return n_bad

    n_bad = asyncio.run(go())
    assert bad.calls == n_bad <= 1  # after one refused connection the replica is skipped
    assert good.calls == 20 - bad.calls


def test_stop_kills_stage_processes_left_by_a_crashed_server(tmp_path):
    pidfile = tmp_path / "child.pid"
    spec = {"name": "x", "command": f"sleep 300 & echo $! > {pidfile}; exit 3"}
    srv = Server(spec, {}, [], free_port(), tmp_path / "logs", 10)
    srv.start()
    srv.proc.wait(timeout=10)
    pid = int(pidfile.read_text())
    os.kill(pid, 0)  # the "stage process" outlived the main process
    srv.stop(grace_s=5)
    t_end = time.monotonic() + 10
    while time.monotonic() < t_end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.2)
    else:
        pytest.fail("orphaned group member still alive after stop()")


def test_mark_crash_affected_rewrites_only_given_ids(tmp_path):
    p = tmp_path / "r.jsonl"
    p.write_text("".join(json.dumps({"request_id": f"q{i}", "status": "ok"}) + "\n" for i in range(4)))
    assert mark_crash_affected(p, {"q1", "q3"}) == 2
    assert [bool(r.get("crash_affected")) for r in read_jsonl(p)] == [False, True, False, True]
    assert mark_crash_affected(p, {"q1"}) == 0
