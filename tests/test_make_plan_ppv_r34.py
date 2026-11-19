"""R3/R4 plan generator: cells, runtimes, commands and the registered confirmation rule (no GPU, no network)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1] / "scripts/slurm"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("make_plan_ppv_r34", HERE / "make_plan_ppv_r34.py")
mp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mp)


def test_r3_screening_runs_share_one_seeded_cell_order_and_differ_only_in_concurrency():
    (c1, m1), (c4, m4) = mp.r3_screen_entries()
    assert c1["experiment_id"] == "ppv_r3_screen_c1_a2" and c4["experiment_id"] == "ppv_r3_screen_c4_a2"
    cams1 = [p for p in c1["phases"] if p["kind"] == "plugin"]
    cams4 = [p for p in c4["phases"] if p["kind"] == "plugin"]
    assert cams1 == cams4 and len(cams1) == 18
    assert {(p["shape"], p["sessions"], p["fps"]) for p in cams1} == {(s, n, f) for s in ("frame", "clip") for n in (1, 4, 16)
                                                                       for f in (0.5, 1.0, 2.0)}
    assert all(p["module"] == "sessions.camera" and p["server_events"] and p["policy"] == "drop_if_busy" for p in cams1)
    assert c1["servers"][0]["args"] == ["--text-only"]
    assert c4["servers"][0]["args"] == ["--text-only", "--preprocessing.factory.max_concurrency", "4"]
    assert c1["runtime_profile"] == c4["runtime_profile"] == "unicorn_ppv_kv" and c1["hardware_label"] == "a6000x1"
    assert (c1["ppv_preprocessing_concurrency"], c4["ppv_preprocessing_concurrency"]) == (1, 4)
    assert cams1 != [mp.camera_phase(*c) for c in sorted(mp.r3_cells())]  # shuffled, not in sorted order


def test_r3_confirmation_needs_known_cells_and_pairs_both_concurrencies_per_block():
    entries = mp.r3_confirm_entries(["frame_s16_f2", "clip_s4_f1"])
    assert len(entries) == 6
    for block in (1, 2, 3):
        pair = [c for c, _ in entries if c["experiment_id"].endswith(f"_block{block}")]
        assert len(pair) == 2
        assert [p for p in pair[0]["phases"] if p["kind"] == "plugin"] == [p for p in pair[1]["phases"] if p["kind"] == "plugin"]
    with pytest.raises(ValueError):
        mp.r3_confirm_entries(["frame_s99_f2"])
    with pytest.raises(ValueError):
        mp.r3_confirm_entries([])


def test_r4_blocks_sweep_sessions_with_a_media_contrast():
    entries = mp.r4_entries()
    assert [c["experiment_id"] for c, _ in entries] == ["ppv_r4_block1", "ppv_r4_block2", "ppv_r4_block3"]
    for cfg, meta in entries:
        plugin = [p for p in cfg["phases"] if p["kind"] == "plugin"]
        assert [p["sessions"] for p in plugin] == [8, 16, 32, 64, 128, 32]
        assert [p["history_media"] for p in plugin] == [0, 0, 0, 0, 0, 2]
        assert len({p["name"] for p in plugin}) == 6
        assert all(p["module"] == "sessions.robot_sessions" and p["speak_final"] and p["server_events"] for p in plugin)
        assert cfg["servers"][0]["args"][-1].endswith("minicpm_o_4_5_exp_t2_k2_c4.yaml") and cfg["hardware_label"] == "a6000x8"
        assert meta["timeout"] <= 5400
    smoke, _ = mp.r4_smoke_entry()
    plug = [p for p in smoke["phases"] if p["kind"] == "plugin"]
    assert [p["sessions"] for p in plug] == [4, 4, 4] and [p["history_media"] for p in plug] == [0, 2, 0]
    assert (plug[2]["image"], plug[2]["audio"]) == (False, False) and plug[0]["turns"] == plug[2]["turns"]


def test_rendered_plans_run_from_the_worktree():
    files = mp.render(confirm=["frame_s16_f2"])
    for name, gpus in (("plan_ppv_r3_screen", 1), ("plan_ppv_r4_smoke", 8), ("plan_ppv_r4", 8), ("plan_ppv_r3_confirm", 1)):
        plan = json.loads(files[f"configs/experiments/{name}.json"])
        for e in plan["experiments"]:
            assert e["command"] == ("cd ~/workspace/omni-ppv && OMNI_REPO=\"$PWD\" bash scripts/slurm/run_experiment.sh "
                                    f"configs/experiments/ppv/{e['experiment_id']}.json")
            assert e["required_gpus"] == gpus
            assert f"configs/experiments/ppv/{e['experiment_id']}.json" in files
    assert "configs/experiments/plan_ppv_r3_confirm.json" not in mp.render()


def test_r3_r4_configs_turn_the_stall_watcher_on():
    import json as _json

    files = mp.render()
    cfgs = [_json.loads(t) for p, t in files.items() if p.startswith("configs/experiments/ppv/")]
    assert cfgs and all(c.get("stall_watch") is True for c in cfgs)


def test_r3_confirmation_blocks_are_chained_and_never_overlap():
    import json as _json

    plan = _json.loads(mp.render(confirm=["frame_s16_f2", "clip_s16_f2"])["configs/experiments/plan_ppv_r3_confirm.json"])
    deps = {e["experiment_id"]: e["dependencies"] for e in plan["experiments"]}
    assert deps["ppv_r3_confirm_seq_c1_block1"] == [] and deps["ppv_r3_confirm_seq_c4_block1"] == []
    for b in (2, 3):
        for c in (1, 4):
            assert sorted(deps[f"ppv_r3_confirm_seq_c{c}_block{b}"]) == [f"ppv_r3_confirm_seq_c1_block{b - 1}",
                                                                           f"ppv_r3_confirm_seq_c4_block{b - 1}"]
