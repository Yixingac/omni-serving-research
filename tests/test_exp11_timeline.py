"""实验 11 measured timeline windows from a thinker torch trace (synthetic trace, no GPU)."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("exp11_timeline", HERE / "exp11_timeline.py")
tl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tl)


def x(name, cat, ts, dur, tid=1, corr=None, **args):
    a = dict(args)
    if corr is not None:
        a["correlation"] = corr
    return {"ph": "X", "name": name, "cat": cat, "ts": ts, "dur": dur, "tid": tid, "args": a}


def trace():
    ev = []
    t = 0.0
    for i in range(1, 80):  # one step every 30 ms (24 ms of kernels), steps 1..79
        ev.append(x(f"thinker.step{i}.DECODE.la", "user_annotation", t, 1000))
        ev.append(x("cudaGraphLaunch", "cuda_runtime", t + 200, 100, corr=i))
        ev.append(x("graph_kernel", "kernel", t + 1000, 24000, tid=7, corr=i))
        t += 30000
    # the first request-finished copy before 1 s does not count; the one at 1.2 s does
    ev.append(x("thinker.request_finished_copy", "user_annotation", 500000, 23000))
    ev.append(x("thinker.request_finished_copy", "user_annotation", 1200000, 23000))
    ev.append(x("cudaMemcpyAsync", "cuda_runtime", 1200500, 22000, corr=9001))
    ev.append(x("Memcpy DtoH", "gpu_memcpy", 1221000, 400, tid=8, corr=9001, bytes=401408))
    return {"traceEvents": ev}


def test_window_starts_at_the_first_copy_after_one_second():
    w = tl.pick_window(trace())
    assert w == (1200000 - 60000, 1200000 + 90000)
    assert tl.pick_window({"traceEvents": []}) is None


def test_window_events_are_relative_milliseconds():
    tr = trace()
    out = tl.window_events(tr, *tl.pick_window(tr))
    assert [c["start_ms"] for c in out["copies"]] == [60.0] and out["copies"][0]["dur_ms"] == 23.0
    steps = out["steps"]
    assert steps and all(-1 <= s["gpu_start_ms"] <= 150 for s in steps)  # steps whose GPU work overlaps the window
    assert all(s["mode"] == "DECODE" and s["launch"] == "la" and s["graph"] for s in steps)
    big = [b for b in out["blocking"] if b["kind"] == "large"]
    assert len(big) == 1 and big[0]["start_ms"] == 60.5 and big[0]["dur_ms"] == 22.0
