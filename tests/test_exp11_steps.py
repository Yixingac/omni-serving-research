"""实验 11 step-aligned GPU gaps and blocking-copy attribution in a thinker torch trace (synthetic traces only)."""

import importlib.util
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("exp11_steps", HERE / "exp11_steps.py")
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)


def x(cat, name, ts, dur, tid=1, corr=None, **args):
    a = dict(args)
    if corr is not None:
        a["correlation"] = corr
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur, "tid": tid, "args": a}


def trace():
    ev = [
        # step 1: lookahead decode through a CUDA graph, GPU 100-120
        x("user_annotation", "thinker.step1.DECODE.la", 90, 5),
        x("cuda_runtime", "cudaGraphLaunch", 91, 2, corr=1), x("kernel", "k", 100, 20, tid=7, corr=1),
        # step 2: a mixed step run synchronously and eagerly, GPU 125-130 and 131-150 (1 us idle inside the step)
        x("user_annotation", "thinker.step2.MIXED.sync", 121, 4),  # launch ranges follow one another
        x("cuda_runtime", "cudaLaunchKernel", 122, 1, corr=2), x("kernel", "k", 125, 5, tid=7, corr=2),
        x("cuda_runtime", "cudaLaunchKernel", 123, 1, corr=3), x("kernel", "k", 131, 19, tid=7, corr=3),
        # step 3: lookahead decode, GPU 150.5-170
        x("user_annotation", "thinker.step3.DECODE.la", 140, 5),
        x("cuda_runtime", "cudaGraphLaunch", 141, 2, corr=4), x("kernel", "k", 150.5, 19.5, tid=7, corr=4),
        # a blocking copy inside the request-finished annotation, and one outside it
        x("user_annotation", "thinker.request_finished_copy", 200, 30),
        x("cuda_runtime", "cudaMemcpyAsync", 201, 25000, corr=5), x("gpu_memcpy", "Memcpy DtoH", 24000, 3, tid=8, corr=5, bytes=368640),
        x("cuda_runtime", "cudaMemcpyAsync", 30000, 1500, corr=6), x("gpu_memcpy", "Memcpy DtoH", 31400, 1, tid=8, corr=6, bytes=16),
    ]
    return {"traceEvents": ev}


def test_steps_are_tied_to_their_gpu_work_and_typed():
    steps = st.steps(trace())
    assert [s["index"] for s in steps] == [1, 2, 3]
    assert [(s["mode"], s["launch"], s["graph"]) for s in steps] == [("DECODE", "la", True), ("MIXED", "sync", False), ("DECODE", "la", True)]
    assert (steps[1]["gpu_start"], steps[1]["gpu_end"]) == (125, 150)
    assert math.isclose(steps[1]["busy"], 24.0)  # 5 + 19, the 1 us inside the step is not busy


def test_gaps_between_steps_and_idle_shares():
    g = st.gaps(st.steps(trace()))
    assert [s["gap_before"] for s in g["steps"]] == [None, 5.0, 0.5]
    assert math.isclose(g["window"], 70.0)
    assert math.isclose(g["between_step_idle_share"], 5.5 / 70)
    assert math.isclose(g["total_idle_share"], (70 - (20 + 24 + 19.5)) / 70)  # also counts the 1 us inside step 2
    assert g["by_type"]["MIXED.sync"]["gap_sum"] == 5.0 and g["by_type"]["DECODE.la"]["gap_sum"] == 0.5


def test_a_long_gap_is_flagged_when_step_indices_jump():
    tr = trace()
    step3 = next(e for e in tr["traceEvents"] if e["name"] == "thinker.step3.DECODE.la")
    step3["name"] = "thinker.step9.DECODE.la"  # steps 3-8 were not traced: the gap is not one step's
    g = st.gaps(st.steps(tr))
    assert g["steps"][-1]["index_jump"] is True and g["index_jumps"] == 1


def test_blocking_copies_inside_the_request_finished_annotation():
    a = st.blocking_attribution(trace())
    assert a["large"] == {"n": 1, "inside": 1, "share_inside": 1.0}
    assert a["small"]["n"] == 1 and a["small"]["inside"] == 0
