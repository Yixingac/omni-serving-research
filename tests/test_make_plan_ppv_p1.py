"""P1 plan generator (length-aware batching + max wait on the full service): settings, cells, seeds (no GPU, no network)."""

import pytest
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/slurm"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("make_plan_ppv_p1", HERE / "make_plan_ppv_p1.py")
mp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mp)

X = {"w3b": 29.3, "w3l": 9.1, "mix": 14.2}


def test_settings_write_every_control_key():
    assert set(mp.SETTINGS) == {"fifo16", "length8", "length8_w500", "length8_w1000", "length8_w2000"}
    for content in mp.SETTINGS.values():
        assert set(content) == {"max_batch_size", "policy", "window", "max_wait_ms"}
    assert mp.SETTINGS["fifo16"]["policy"] == "fifo" and mp.SETTINGS["fifo16"]["max_batch_size"] == 16
    assert [mp.SETTINGS[s]["max_wait_ms"] for s in ("length8", "length8_w500", "length8_w1000", "length8_w2000")] == [None, 500, 1000, 2000]


def test_rates_follow_the_rule():
    r = mp.rates(X)
    assert r["w3b"] == {"x090": 26.5, "x100": 29.5, "x110": 32.0}
    assert r["w3l"] == {"x090": 8.0, "x100": 9.0, "x110": 10.0}
    assert mp.round_half(8.25) == 8.5 and mp.round_half(8.24) == 8.0


def test_prerun_is_fifo16_closed_loop_without_events():
    cfg, meta = mp.prerun_entry()
    env = cfg["servers"][0]["env"]
    assert cfg["runtime_profile"] == "unicorn_ppv_maxwait" and env["SGLANG_OMNI_C2W_CONTROL"] == "{exp_dir}/c2w_control.json"
    assert env["SGLANG_OMNI_EXP_STACK_DIR"] == "{exp_dir}/stacks"
    ctl = [p for p in cfg["phases"] if p["kind"] == "control"]
    assert len(ctl) == 1 and ctl[0]["content"] == mp.SETTINGS["fifo16"]
    closed = [p for p in cfg["phases"] if p["kind"] == "closed_loop"]
    assert [(p["name"], p["concurrency"]) for p in closed] == [("pre_w3b_c256", [256]), ("pre_w3l_c128", [128]), ("pre_mix_c192", [192])]
    assert not any(p.get("server_events") for p in closed)
    assert "workload_mix" in closed[2] and "workload" not in closed[2]


def test_main_blocks_hold_every_cell_once_with_a_control_before_each():
    entries = mp.main_entries(X)
    assert [c["experiment_id"] for c, _ in entries] == ["ppv_p1_block1", "ppv_p1_block2", "ppv_p1_block3"]
    r = mp.rates(X)
    orders = []
    for block, (cfg, meta) in enumerate(entries, start=1):
        phases = cfg["phases"][1:]
        assert len(phases) == 2 * 45
        cells = []
        for ctl, ph in zip(phases[::2], phases[1::2]):
            assert ctl["kind"] == "control" and ph["kind"] == "open_loop" and ctl["name"] == f"ctl_{ph['name']}"
            w, rest = ph["name"].split("_", 1)
            setting, tag = rest.rsplit("_", 1)
            assert ctl["content"] == mp.SETTINGS[setting]
            assert ph["rates_rps"] == [r[w][tag]] and ph["server_events"] is True and ph["seed"] == mp.SEED + block
            assert ph["duration_per_rate_s"] == mp.WARMUP_S[w] + mp.MEASURED_S[w] and ph["warmup_s"] == mp.WARMUP_S[w]
            cells.append((w, setting, tag))
        assert sorted(cells) == sorted((w, s, t) for w in mp.WORKLOADS for s in mp.SETTINGS for t in mp.FRACTIONS)
        workload_order = list(dict.fromkeys(w for w, _, _ in cells))
        assert workload_order == list(mp.WORKLOADS[block - 1:] + mp.WORKLOADS[:block - 1])
        orders.append([c for c in cells if c[0] == "w3b"])
        assert meta["timeout"] >= 2 * meta["est"] - 1
    assert orders[0] != orders[1] != orders[2]  # random order per block
    assert entries == mp.main_entries(X)  # deterministic


def test_render_writes_the_main_plan_only_with_x_and_uses_the_worktree_command(monkeypatch):
    monkeypatch.setattr(mp, "P1_X", None)  # before the pre-run: no X, no main plan
    files = mp.render(None)
    assert "configs/experiments/plan_ppv_p1_prerun.json" in files and "configs/experiments/plan_ppv_p1.json" not in files
    files = mp.render(X)
    plan = json.loads(files["configs/experiments/plan_ppv_p1.json"])
    for e in plan["experiments"]:
        assert e["command"] == ("cd ~/workspace/omni-ppv && OMNI_REPO=\"$PWD\" bash scripts/slurm/run_experiment.sh "
                                f"configs/experiments/ppv/{e['experiment_id']}.json")
        assert e["required_gpus"] == 8
    pre = json.loads(files["configs/experiments/plan_ppv_p1_prerun.json"])["experiments"][0]
    assert pre["priority"] > max(e["priority"] for e in plan["experiments"])  # the pre-run first


def test_interaction_variant_pairs_dense_and_packed_on_the_same_cells():
    yaml = mp.packed_yaml()
    assert "enable_flow_variable_length: true" in yaml and "enable_flow_block_compile: true" in yaml
    assert yaml.count("process: code2wav") == 1
    entries = mp.interaction_entries(X)
    assert len(entries) == 6
    by_block = {}
    for cfg, _ in entries:
        variant, block = cfg["experiment_id"].split("_")[2], int(cfg["experiment_id"][-1])
        names = [p["name"] for p in cfg["phases"] if p["kind"] == "open_loop"]
        by_block.setdefault(block, {})[variant] = names
        assert cfg["servers"][0]["args"][-1].endswith("packed.yaml" if variant == "packed" else "t2_k2_c4.yaml")
        assert len(names) == 8 and all(n.split("_", 1)[1].rsplit("_", 1)[0] in ("fifo16", "length8_w1000") for n in names)
    for block, v in by_block.items():
        assert v["dense"] == v["packed"]


def test_p1_configs_turn_the_stall_watcher_on():
    import json as _json

    files = mp.render()
    cfgs = [_json.loads(t) for p, t in files.items() if p.startswith("configs/experiments/ppv/")]
    assert cfgs and all(c.get("stall_watch") is True for c in cfgs)


def test_block2_rerun_is_block2_without_the_stack_hook():
    if mp.P1_X is None:
        pytest.skip("X not filled in")
    (cfg, meta), = mp.rerun_entries(mp.P1_X)
    orig = next(c for c, _ in mp.main_entries(mp.P1_X) if c["experiment_id"] == "ppv_p1_block2")
    assert cfg["experiment_id"] == "ppv_p1_block2_a2" and cfg["phases"] == orig["phases"]
    assert -631 < meta["priority"] < -613  # right after block 3, before the R3 confirmation runs
    env = cfg["servers"][0]["env"]
    parts = env["SGLANG_OMNI_EXP_TRACE"].split(",")
    assert "stack" not in parts and "gc" in parts and "SGLANG_OMNI_EXP_STACK_DIR" not in env
