"""Phase and server descriptions of scripts/analysis/list_plans.py (no cluster data needed)."""

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("list_plans", Path(__file__).resolve().parents[1] / "scripts/analysis/list_plans.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_closed_loop_line_states_requests_warmup_stops_and_profiling():
    ph = {"kind": "closed_loop", "name": "w3b_closed", "workload": "configs/workloads/w3b_text_speech.json",
          "concurrency": [16, 32], "requests_per_level": 96, "requests_per_concurrency": 2, "server_events": True,
          "stop_if_p50_e2e_above_s": 90}
    line = mod.phase_line(ph)
    assert "并发 16, 32" in line and "max(96, 2×并发)" in line and "预热请求数与并发数相同" in line
    assert "e2e 中位数 > 90 s" in line and "错误率 > 0.2" in line and "服务端事件" in line


def test_open_loop_line_from_capacity_and_duration():
    ph = {"kind": "open_loop", "name": "w3b_open", "workload": "configs/workloads/w3b_text_speech.json", "capacity_from": "w3b_closed",
          "capacity_fractions": [0.5, 1.2], "duration_per_rate_s": 120, "min_requests_per_rate": 96}
    line = mod.phase_line(ph)
    assert "`w3b_closed` 阶段的闭环最高吞吐 × 0.5, 1.2" in line and "max(96, 速率×时长)" in line


def test_planned_levels_and_server_prefix_hides_paths():
    assert mod.planned_levels({"rates_rps": [1.5] * 8}) == 8
    assert mod.planned_levels({"kind": "idle"}) is None
    s = mod.server_line({"name": "s0", "model": "m", "gpus": [0], "ready_timeout_s": 60,
                         "prefix": "nsys profile --duration=40 -o {exp_dir}/profiling/nsys_s0"})
    assert "{exp_dir}" not in s and "<实验目录>" in s
