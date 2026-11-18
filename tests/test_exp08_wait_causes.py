"""实验 8 phase 3: which CPU calls block the thinker (synthetic traces only)."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("exp08_wait_causes", HERE / "exp08_wait_causes.py")
wc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wc)


def rt(name, ts, dur, corr=None):
    return {"ph": "X", "cat": "cuda_runtime", "name": name, "ts": ts, "dur": dur, "args": {"correlation": corr} if corr else {}}


def copy(ts, nbytes, corr):
    return {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy DtoH (Device -> Pageable)", "ts": ts, "dur": 2,
            "args": {"correlation": corr, "bytes": nbytes}}


def test_blocking_calls_keep_slow_syncs_and_copies_with_their_sizes():
    trace = {"traceEvents": [
        rt("cudaMemcpyAsync", 0, 20000, 1), copy(19990, 368640, 1),   # hidden states at request end: large
        rt("cudaMemcpyAsync", 30000, 1800, 2), copy(31790, 16, 2),    # token ids: small
        rt("cudaMemcpyAsync", 40000, 10, 3), copy(40005, 16, 3),      # quick copy: not blocking
        rt("cudaMemcpyAsync", 45000, 2500, 9),                        # no GPU record: size unknown
        rt("cudaEventSynchronize", 50000, 15000),
        rt("cudaGraphLaunch", 66000, 30)]}
    calls = wc.blocking_calls(trace)
    assert [(c["kind"], round(c["ms"], 1)) for c in calls] == [
        ("copy_large", 20.0), ("copy_small", 1.8), ("copy_unknown", 2.5), ("sync", 15.0)]
    assert calls[0]["bytes"] == 368640


def test_summary_per_kind_is_normalised_by_graph_launches():
    calls = [{"kind": "copy_large", "ms": 20.0, "bytes": 1}, {"kind": "copy_large", "ms": 22.0, "bytes": 1},
             {"kind": "sync", "ms": 15.0, "bytes": None}]
    s = wc.summarize(calls, launches=10)
    assert s["copy_large"] == {"n": 2, "total_ms": 42.0, "per_launch": 0.2, "ms_p50": 21.0}
    assert s["sync"]["per_launch"] == 0.1 and s["copy_small"]["n"] == 0
