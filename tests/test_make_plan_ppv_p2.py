"""P2 plan generator (bit-exact hidden-state copy check): the verify run and the gated performance runs (no GPU)."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "scripts/slurm"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("make_plan_ppv_p2", HERE / "make_plan_ppv_p2.py")
mp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mp)


def test_verify_run_uses_the_exp11_check_phases_with_the_verify_mode():
    cfg, meta = mp.verify_entry()
    exp11 = json.loads((ROOT / "configs/experiments/exp09_11/exp11_check_C.json").read_text())
    assert cfg["phases"] == exp11["phases"]  # the same segments as 实验 11's gate
    assert cfg["servers"][0]["args"] == exp11["servers"][0]["args"]  # 2+6
    env = cfg["servers"][0]["env"]
    assert env["SGLANG_OMNI_SPEECH_LOOKAHEAD"] == "sidecopy_verify" and env["SGLANG_OMNI_EXP_TRACE"] == "thinker,c2w,digest"
    assert cfg["runtime_profile"] == "unicorn_ppv_verify" and meta["timeout"] <= 3600
    assert all(p.get("server_events") for p in cfg["phases"][1:])  # every check phase records events


def test_perf_runs_are_the_registered_exp11_runs_with_new_ids():
    perf = mp.perf_entries()
    assert len(perf) == 9
    for cfg, _ in perf:
        old = json.loads((ROOT / "configs/experiments/exp09_11" / (cfg["experiment_id"].replace("ppv_p2_perf_", "exp11_perf_") + ".json")).read_text())
        assert cfg["experiment_id"].startswith("ppv_p2_perf_block")
        assert {k: v for k, v in cfg.items() if k not in ("experiment_id", "description")} == \
               {k: v for k, v in old.items() if k not in ("experiment_id", "description")}
        assert cfg["runtime_profile"] == "unicorn_async"


def test_rendered_plans_are_separate_and_run_from_the_worktree():
    files = mp.render()
    verify = json.loads(files["configs/experiments/plan_ppv_p2_verify.json"])["experiments"]
    perf = json.loads(files["configs/experiments/plan_ppv_p2_perf.json"])["experiments"]
    assert [e["experiment_id"] for e in verify] == ["ppv_p2_verify"] and len(perf) == 9
    for e in verify + perf:
        assert e["command"].startswith("cd ~/workspace/omni-ppv && OMNI_REPO=\"$PWD\" bash scripts/slurm/run_experiment.sh ")
        assert f"configs/experiments/ppv/{e['experiment_id']}.json" in files
    assert verify[0]["priority"] > max(e["priority"] for e in perf)
    assert verify[0]["priority"] < -713  # queued behind the R4 blocks
