"""P3 conditional plan (run writes on /scratch vs NFS): ABBA order, identical phases, the scratch command (no GPU)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/slurm"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("make_plan_ppv_p3", HERE / "make_plan_ppv_p3.py")
mp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mp)


def test_abba_runs_share_every_phase():
    es = mp.entries()
    assert [m["arm"] for _, m in es] == ["nfs", "scratch", "scratch", "nfs"]
    phases = [c["phases"] for c, _ in es]
    assert all(p == phases[0] for p in phases)
    opens = [p for p in phases[0] if p["kind"] == "open_loop"]
    assert len(opens) == 6 and all(p["rates_rps"] == [21.0] and p["duration_per_rate_s"] == 300 and p["server_events"] for p in opens)
    for cfg, _ in es:
        env = cfg["servers"][0]["env"]
        assert "SGLANG_OMNI_C2W_CONTROL" not in env and env["SGLANG_OMNI_EXP_STACK_DIR"] == "{exp_dir}/stacks"
        assert cfg["runtime_profile"] == "unicorn_ppv_maxwait"


def test_scratch_command_runs_there_and_copies_back(tmp_path):
    files = mp.render()
    plan = json.loads(files["configs/experiments/plan_ppv_p3.json"])["experiments"]
    scratch = [e for e in plan if "_scratch_" in e["experiment_id"]]
    nfs = [e for e in plan if "_nfs_" in e["experiment_id"]]
    assert len(scratch) == 2 and len(nfs) == 2
    assert all("OMNI_EXP_DIR=\"$SCR\"" not in e["command"] for e in nfs)
    cmd = scratch[0]["command"]
    assert "rsync -a \"$SCR\"/ \"$OMNI_EXP_DIR\"/ && rm -rf \"$SCR\"; exit $rc" in cmd
    # the shell logic with stand-ins: the run writes into $SCR, the result ends up in the orchestrator's directory
    fake = cmd.replace("cd ~/workspace/omni-ppv && ", "").replace("/scratch/$USER/omni_ppv", str(tmp_path / "scratch")).replace(
        "bash scripts/slurm/run_experiment.sh configs/experiments/ppv/" + scratch[0]["experiment_id"] + ".json",
        "sh -c 'echo done > \"$OMNI_EXP_DIR/result.json\"; exit 3'")
    out = tmp_path / "nfs_out"
    out.mkdir()
    rc = subprocess.run(["bash", "-c", fake], env={"OMNI_EXP_ID": "x1", "OMNI_EXP_DIR": str(out), "PATH": "/usr/bin:/bin"}).returncode
    assert rc == 3 and (out / "result.json").read_text().strip() == "done" and not (tmp_path / "scratch" / "x1").exists()
