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
        assert srv["gpus"] == list(range(8)) and srv["env"] == {"SGLANG_OMNI_C2W_CONTROL": "{exp_dir}/c2w_control.json"}
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
