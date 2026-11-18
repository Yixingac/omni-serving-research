"""实验 8 phase-3 performance analysis: sync calls per step and the paired off/on verdict (synthetic data only)."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("exp08_proto_perf", HERE / "exp08_proto_perf.py")
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)


def call(name, ts, dur):
    return {"ph": "X", "cat": "cuda_runtime", "name": name, "ts": ts, "dur": dur}


def test_sync_calls_per_step_count_syncs_and_blocking_copies():
    trace = {"traceEvents": [call("cudaGraphLaunch", 0, 5), call("cudaGraphLaunch", 100, 5), call("cudaGraphLaunch", 200, 5),
                             call("cudaStreamSynchronize", 10, 20), call("cudaStreamSynchronize", 110, 20),
                             call("cudaMemcpyAsync", 210, 2000),  # waits for the step: counts
                             call("cudaMemcpyAsync", 50, 10)]}  # a quick copy: does not
    assert pp.sync_calls_per_step(trace) == 1.0


def runs(idle_off, idle_on, level=8):
    out = []
    for b, (x, y) in enumerate(zip(idle_off, idle_on), start=1):
        for arm, v in (("off", x), ("on", y)):
            out.append({"arm": arm, "block": b, "levels": {str(level): {"C": {"graph_steps": {"idle_ms_p50": v}}}}})
    return out


def test_verdict_needs_every_block_lower_and_a_mean_drop_of_two_ms():
    v = pp.verdicts(runs([4.0, 4.5, 3.8], [0.1, 0.2, 0.1]))
    assert v["idle"]["8"]["verdict"] == "支持" and v["idle_verdict"] == "支持"
    v = pp.verdicts(runs([4.0, 4.5, 3.8], [3.5, 4.6, 3.0]))  # one block goes up
    assert v["idle"]["8"]["verdict"] == "不支持"
    v = pp.verdicts(runs([4.0, 4.5, 3.8], [3.0, 3.2, 2.9]))  # lower in every block but by < 2 ms
    assert v["idle"]["8"]["verdict"] == "不支持"


def test_missing_block_is_insufficient():
    r = runs([4.0, 4.5, 3.8], [0.1, 0.2, 0.1])
    r[-1]["levels"]["8"]["C"]["graph_steps"]["idle_ms_p50"] = None
    assert pp.verdicts(r)["idle"]["8"]["verdict"] == "数据不足"


def test_time_weighted_idle_share_is_kept_besides_the_median():
    # the median idle per step hides the steps that still wait (flushes); the time-weighted share does not
    assert "idle_share" in pp.GRAPH_KEYS and "graph_steps.idle_share" in pp.METRICS["C"]
