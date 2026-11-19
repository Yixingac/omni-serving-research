"""P0 plan generator (talker admission cap): arms differ only in the talker engine overrides (no GPU, no network)."""

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/slurm"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("make_plan_ppv_p0", HERE / "make_plan_ppv_p0.py")
mp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mp)


def test_arm_yamls_change_only_the_talker_engine_block():
    base = (Path(__file__).resolve().parents[1] / "configs/models/minicpm_o_4_5_exp_t2_k1_c5.yaml").read_text()
    body = lambda text: [ln for ln in text.splitlines() if ln and not ln.startswith("#")]  # noqa: E731
    for arm, extra in (("B", ["max_running_requests: 64"]), ("G", ["max_running_requests: 64", "cuda_graph_max_bs: 32"]),
                       ("C", ["max_running_requests: 64", "max_total_tokens: 262144"])):
        got = body(mp.arm_yaml(arm))
        i = got.index("    engine:")
        assert [ln.strip() for ln in got[i + 1:i + 1 + len(extra)]] == extra
        assert got[:i] + got[i + 1 + len(extra):] == body(base)  # everything else as in 实验 9's arm 3
    assert mp.yaml_path("A") == "configs/models/minicpm_o_4_5_exp_t2_k1_c5.yaml"
    assert mp.yaml_path("R") == "configs/models/minicpm_o_4_5_exp_t2_k2_c4.yaml"


def test_blocks_hold_each_arm_once_in_a_seeded_order_and_conditional_arm_is_separate():
    entries = mp.entries()
    assert len(entries) == 4 * mp.BLOCKS
    orders = []
    for block in range(1, mp.BLOCKS + 1):
        ids = [c["experiment_id"] for c, _ in entries if c["experiment_id"].endswith(f"_block{block}_a2")]
        assert sorted(ids) == sorted(mp.experiment_id(a, block) for a in mp.ARMS)
        orders.append(ids)
    assert entries == mp.entries()  # deterministic
    assert len({tuple(x.split("_block")[0] for x in o) for o in orders}) > 1  # the order changes between blocks
    assert all(i.startswith("ppv_p0_") for o in orders for i in o)
    prios = [m["priority"] for _, m in entries]
    assert prios == sorted(prios, reverse=True)  # blocks and positions run in the listed order
    assert all("_cap64_kv262k_" not in c["experiment_id"] for c, _ in entries)


def test_phases_per_level_rates_and_runtime():
    for arm in mp.ARMS:
        cfg = next(c for c, _ in mp.entries() if c["experiment_id"] == mp.experiment_id(arm, 1))
        assert cfg["runtime_profile"] == "unicorn_instr" and cfg["hardware_label"] == "a6000x8"
        names = [p["name"] for p in cfg["phases"]]
        assert names[:5] == ["smoke_w3b", "w3b_closed_A_c64", "w3b_closed_A_c128", "w3b_closed_A_c256", "w3b_closed_B_c256"]
        closed = [p for p in cfg["phases"] if p["kind"] == "closed_loop"]
        assert all(len(p["concurrency"]) == 1 for p in closed)
        assert [p.get("server_events", False) for p in closed][:4] == [False, False, False, True]
        w3b_open = next(p for p in cfg["phases"] if p["name"] == "w3b_open")
        assert w3b_open["rates_rps"] == [15.5, 18.0, 21.0, 23.5, 26.0, 28.5, 31.0] and w3b_open["duration_per_rate_s"] == 60
        has_w3l = "w3l_open" in names
        assert has_w3l == (arm in ("A", "B"))
        if has_w3l:
            assert next(p for p in cfg["phases"] if p["name"] == "w3l_open")["rates_rps"] == [8.0, 9.5, 10.5, 12.0]
        assert cfg["servers"][0]["args"][-1] == "{research_repo}/" + mp.yaml_path(arm)


def test_rendered_plans_run_from_the_worktree_and_need_all_eight_gpus():
    files = mp.render()
    for b in (1, 2, 3):
        plan = json.loads(files[f"configs/experiments/plan_ppv_p0_block{b}_a2.json"])
        assert len(plan["experiments"]) == 4
        for e in plan["experiments"]:
            # the orchestrator exports OMNI_REPO=<its checkout>; the command must point it at this worktree
            assert e["command"] == ("cd ~/workspace/omni-ppv && OMNI_REPO=\"$PWD\" bash scripts/slurm/run_experiment.sh "
                                    f"configs/experiments/ppv/{e['experiment_id']}.json")
            assert e["required_gpus"] == 8 and e["timeout_s"] == 5400
            assert f"configs/experiments/ppv/{e['experiment_id']}.json" in files
    assert len(json.loads(files["configs/experiments/plan_ppv_p0_armC_a2.json"])["experiments"]) == 3


def test_blocks_2_and_3_without_arm_g_keep_the_seeded_order_of_the_others():
    files = mp.render()
    for b in (2, 3):
        full = [e["experiment_id"] for e in json.loads(files[f"configs/experiments/plan_ppv_p0_block{b}_a2.json"])["experiments"]]
        without = [e["experiment_id"] for e in json.loads(files[f"configs/experiments/plan_ppv_p0_block{b}_a2_without_g.json"])["experiments"]]
        assert without == [i for i in full if "_cap64_graph32_" not in i] and len(without) == 3
