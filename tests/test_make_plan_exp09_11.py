"""Plan generator for 实验 9–11: the conditional 实验 10 full-service validation runs (no GPU, no network)."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/slurm"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("make_plan_exp09_11", HERE / "make_plan_exp09_11.py")
mp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mp)


def test_exp10_validation_alternates_the_order_and_switches_by_control_file():
    entries = mp.exp10_validation_entries("t2_k2_c4", "length8")
    assert len(entries) == 3
    for block, (cfg, meta) in enumerate(entries, start=1):
        assert cfg["experiment_id"] == f"exp10_validation_t2_k2_c4_length8_block{block}"
        assert cfg["runtime_profile"] == mp.BATCH_RUNTIME
        srv = cfg["servers"][0]
        assert srv["gpus"] == list(range(8)) and srv["env"] == {"SGLANG_OMNI_C2W_CONTROL": "{exp_dir}/c2w_control.json",
                                                              "SGLANG_OMNI_PORT_CLAIM_DIR": "{exp_dir}/port_claims"}
        assert srv["args"][-1].endswith("minicpm_o_4_5_exp_t2_k2_c4.yaml")
        opens = [p for p in cfg["phases"] if p["kind"] == "open_loop"]
        assert [p["name"] for p in opens] == (["val_length8", "val_fifo16"] if block == 2 else ["val_fifo16", "val_length8"])
        for i, p in enumerate(cfg["phases"]):
            if p["kind"] != "open_loop":
                continue
            ctl = cfg["phases"][i - 1]
            assert ctl["kind"] == "control"
            assert ctl["content"] == {"max_batch_size": 8 if p["name"] == "val_length8" else 16,
                                      "policy": "length" if p["name"] == "val_length8" else "fifo", "window": 64}
            assert p["rates_rps"] == mp.EXP09_RATES_W3B and p["duration_per_rate_s"] == 60 and p["server_events"] is True
        assert meta["timeout"] <= 3600


def test_exp10_validation_needs_a_known_placement_and_setting():
    for placement, setting in (("t9_k9", "length16"), ("t2_k2_c4", "fifo16"), ("t2_k2_c4", "length32")):
        try:
            mp.exp10_validation_entries(placement, setting)
        except ValueError:
            continue
        raise AssertionError(f"accepted {placement} {setting}")


def test_exp10_and_exp11_servers_claim_ports_in_a_fresh_directory_per_run():
    """Stale claim files left by a killed server let two starting processes take the same port (实验 9, 2026-11-18
    00:38); a directory inside the run directory never holds claims of an earlier run. 实验 9 keeps its setting."""
    entries = [mp.exp10_check_entry(), *mp.exp10_entries(), *mp.exp11_check_entries(), *mp.exp11_entries(),
               *mp.exp10_validation_entries("t2_k2_c4", "length8")]
    for cfg, _ in entries:
        for srv in cfg["servers"]:
            assert srv["env"]["SGLANG_OMNI_PORT_CLAIM_DIR"] == "{exp_dir}/port_claims", cfg["experiment_id"]
    for cfg, _ in mp.exp09_entries():
        assert all("SGLANG_OMNI_PORT_CLAIM_DIR" not in (srv.get("env") or {}) for srv in cfg["servers"])


def test_exp11_diagnostic_runs_b_twice_with_the_check_phases():
    """Supplementary registration 五 of 实验 11: after the gate failed, B (lookahead + snapshot, no side copy) and a
    repeat B2 run the same check phases, to tell lookahead scheduling from the side copy as the source of the extra
    fingerprint deviations. Written only on demand (--exp11-diagnostic)."""
    check_a = next(cfg for cfg, _ in mp.exp11_check_entries() if cfg["experiment_id"] == "exp11_check_A")
    diag = mp.exp11_diag_entries()
    assert [cfg["experiment_id"] for cfg, _ in diag] == ["exp11_diag_B", "exp11_diag_B2"]
    for cfg, meta in diag:
        env = cfg["servers"][0]["env"]
        assert env["SGLANG_OMNI_SPEECH_LOOKAHEAD"] == "1" and env["SGLANG_OMNI_EXP_TRACE"] == "thinker,c2w,digest"
        assert env["SGLANG_OMNI_PORT_CLAIM_DIR"] == "{exp_dir}/port_claims"
        assert cfg["phases"] == check_a["phases"] and cfg["runtime_profile"] == "unicorn_async"
        assert cfg["servers"][0]["args"] == check_a["servers"][0]["args"]
    assert [meta["priority"] for _, meta in diag] == [-290, -291]
    files = mp.render_exp11_diag()
    assert mp.PLAN_EXP11_DIAG in files and len(files) == 3
