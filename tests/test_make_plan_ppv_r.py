"""R0-R2 plan generator (full duplex): pinned stock runtime, registered phases and GPU counts (no GPU, no network)."""

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/slurm"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("make_plan_ppv_r", HERE / "make_plan_ppv_r.py")
mp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mp)


def _phases(c):
    return [(p["name"], p.get("sessions"), p.get("duration_s"), p.get("server_events", False)) for p in c["phases"]]


def test_r0_is_the_registered_functional_check_on_one_gpu_with_the_stock_runtime():
    (c, meta), = mp.r0_entries()
    assert c["experiment_id"] == "ppv_r0_duplex_functional" and c["hardware_label"] == "a6000x1"
    assert c["runtime_profile"] == "unicorn"  # 520b49ee without a research patch
    assert c["servers"][0]["args"] == ["--variant", "session", "--enable-realtime"] and c["servers"][0]["gpus"] == [0]
    assert _phases(c) == [("duplex_smoke", 1, 30.0, False), ("duplex_s1", 1, 300.0, False), ("duplex_s2", 2, 300.0, False)]
    assert all(p["module"] == "duplex.phase" and p["session_seed"] == mp.SEED for p in c["phases"])
    assert c["phases"][2]["skip_if_failed"] == ["duplex_smoke"] and meta["timeout"] <= 3600


def test_r1_alternates_event_order_and_has_a_same_gpu_chat_reference():
    entries = mp.r1_entries()
    duplex = [c for c, _ in entries if "_duplex_" in c["experiment_id"]]
    chat = [c for c, _ in entries if "_chat_ref_" in c["experiment_id"]]
    assert len(duplex) == len(chat) == 3
    for b, c in enumerate(duplex, start=1):
        names = [p["name"] for p in c["phases"]]
        assert names == ["duplex_smoke"] + (["duplex_r1_B", "duplex_r1_A"] if b == 2 else ["duplex_r1_A", "duplex_r1_B"])
        a, bb = (next(p for p in c["phases"] if p["name"] == n) for n in ("duplex_r1_A", "duplex_r1_B"))
        assert a["barge_in_every"] == bb["barge_in_every"] == 3 and not a.get("server_events") and bb["server_events"]
    for c in chat:
        assert c["servers"][0]["args"] == ["--code2wav.process", "code2wav"] and c["hardware_label"] == "a6000x1"
        loops = [p for p in c["phases"] if p["kind"] == "closed_loop"]
        assert all(p["workload"] == "configs/workloads/w3_audio_speech.json" and p["concurrency"] == [1] for p in loops)
        assert sorted(p.get("server_events", False) for p in loops) == [False, True]


def test_r2_levels_ascend_stop_after_a_failure_and_use_the_registered_placements():
    for arm, (ms, gpus, levels) in mp.R2_ARMS.items():
        entries = mp.r2_entries(arm)
        assert len(entries) == 3
        for c, meta in entries:
            assert c["hardware_label"] == f"a6000x{gpus}" and c["servers"][0]["gpus"] == list(range(gpus))
            args = c["servers"][0]["args"]
            assert args[:3] == ["--variant", "session", "--enable-realtime"] and args[args.index("--max_sessions") + 1] == str(ms)
            assert ("--thinker.gpu" in args) == (gpus == 2)
            assert [p["sessions"] for p in c["phases"]] == levels and all(p["duration_s"] == 180.0 for p in c["phases"])
            assert all(p["server_events"] for p in c["phases"])
            for prev, p in zip(c["phases"], c["phases"][1:]):
                assert p["skip_if_failed"] == [prev["name"]] and p["stop_if_unit_miss_above"] == 0.01
            assert "stop_if_unit_miss_above" not in c["phases"][0]
            assert max(levels) <= ms and meta["timeout"] <= 3600


def test_rendered_plans_run_from_the_worktree():
    files = mp.render()
    names = {Path(p).name for p in files if p.startswith("configs/experiments/plan_")}
    assert names == {"plan_ppv_r0.json", "plan_ppv_r1.json", "plan_ppv_r2_smoke.json", "plan_ppv_r2_ms4_1gpu.json",
                     "plan_ppv_r2_ms8_2gpu.json", "plan_ppv_r2_ms8_1gpu.json"}
    for p, text in files.items():
        if Path(p).name.startswith("plan_"):
            for e in json.loads(text)["experiments"]:
                assert e["command"] == ("cd ~/workspace/omni-ppv && OMNI_REPO=\"$PWD\" bash scripts/slurm/run_experiment.sh "
                                        f"configs/experiments/ppv/{e['experiment_id']}.json")
                assert f"configs/experiments/ppv/{e['experiment_id']}.json" in files
                assert e["required_gpus"] in (1, 2)


def test_runs_of_a_plan_are_chained_so_latency_runs_never_overlap():
    import json as _json

    files = mp.render()
    for path, text in files.items():
        if not path.startswith("configs/experiments/plan_"):
            continue
        exps = _json.loads(text)["experiments"]
        assert exps[0]["dependencies"] == []
        for prev, cur in zip(exps, exps[1:]):
            assert cur["dependencies"] == [prev["experiment_id"]]
