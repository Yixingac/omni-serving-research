"""Pure helpers of the 实验 5 bubble scripts (scripts/analysis/exp05_*.py); synthetic data only."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


census = load("exp05_bubble_census")
steps = load("exp05_step_sync")
figs = load("exp05_bubble_figures")

MS = 1_000_000


def speech_request(t0_ms=0):
    """Events of one speech request (ms from t0): thinker 0-100, hop 100-110, talker 110-200, code2wav 200-250."""
    t = {("coordinator", "request_admission"): 0, ("thinker", "scheduler_queue_enter"): 2,
         ("thinker", "scheduler_prefill_start"): 3, ("thinker", "scheduler_prefill_end"): 10,
         ("thinker", "model_path_end"): 100, ("talker", "stage_input_received"): 110,
         ("talker", "scheduler_prefill_start"): 112, ("talker", "model_path_end"): 200,
         ("code2wav", "stage_input_received"): 201, ("code2wav", "stage_dispatch"): 201,
         ("code2wav", "stage_complete"): 250, ("coordinator", "terminal_response"): 255}
    return {k: (v + t0_ms) * MS for k, v in t.items()}


def test_places_and_critical_path_of_a_speech_and_a_text_request():
    t = speech_request()
    p = census.place_intervals(t)
    assert p["th_svc"] == (3 * MS, 100 * MS) and p["hop1"] == (100 * MS, 110 * MS)
    assert p["tk_queue"] == (110 * MS, 112 * MS) and p["cw_svc"] == (201 * MS, 250 * MS) and p["tail"] == (250 * MS, 255 * MS)
    seg = census.segments(t)
    assert seg["thinker_decode"] == 90 and seg["hop1"] == 10 and seg["talker_gen"] == 88 and seg["total"] == 255
    text = {k: v for k, v in t.items() if k[0] in ("coordinator", "thinker")}
    p = census.place_intervals(text)
    assert "hop1" not in p and p["tail"] == (100 * MS, 255 * MS)


def test_stage_gpus_from_yaml_flags_and_text_only(tmp_path):
    y = tmp_path / "s1x3.yaml"
    y.write_text("processes:\n  pipeline: {replica_devices: [0]}\n  talker: {replica_devices: [1, 2, 3]}\n  code2wav: {replica_devices: [1, 2, 3]}\n")
    cfg = {"servers": [{"name": "s0", "gpus": [4, 5, 6, 7], "args": ["--config", str(y)]},
                       {"name": "p0", "gpus": [0, 1], "args": ["--talker.gpu", "1", "--code2wav.gpu", "1"]},
                       {"name": "r2", "gpus": [2], "args": ["--text-only"]},
                       {"name": "r3", "gpus": [3], "args": ["--code2wav.process", "code2wav"]}]}
    m = census.stage_gpus(cfg)
    assert m["s0"][("thinker", "r0")] == 4 and m["s0"][("talker", "r2")] == 7 and m["s0"][("code2wav", "r0")] == 5
    assert m["p0"] == {("thinker", "r0"): 0, ("talker", "r0"): 1, ("code2wav", "r0"): 1}
    assert m["r2"] == {("thinker", "r0"): 2}
    assert set(m["r3"].values()) == {3}
    assert census.gpu_role({"thinker", "talker", "code2wav"}) == "完整副本" and census.gpu_role({"talker", "code2wav"}) == "talker 卡"


def classify(reqs):
    """GPU 0 = thinker, GPU 1 = talker + code2wav of one server; returns {gpu: {name: bool array}}."""
    grid = census.Grid(0, 300 * MS)
    for t, speech in reqs:
        for place, (a, b) in census.place_intervals(t).items():
            grid.add(("s", place), a, b)
            if speech:
                grid.add(("s", place, "speech"), a, b)
            kind = census.PLACE_KIND.get(place)
            if kind:
                grid.add(("s", kind, "r0", "svc" if place.endswith("svc") else "queue"), a, b)
    present = {p: grid.counts(("s", p)) > 0 for p in census.PLACES}
    speech = {p: grid.counts(("s", p, "speech")) > 0 for p in census.PLACES}
    g0 = census.classify_gpu(grid, "s", [("thinker", "r0")], [], present, speech)
    g1 = census.classify_gpu(grid, "s", [("talker", "r0"), ("code2wav", "r0")], [], present, speech)
    return g0, g1


def test_no_work_time_is_split_by_where_the_request_is():
    g0, g1 = classify([(speech_request(), True)])
    # Talker GPU: the request is in CPU preprocessing (type 3), then in the thinker (type 1), then in the hop
    # (type 5); the talker works; then the tail is downstream.
    assert g1["type3"][:2].all() and g1["type1"][2:100].all() and g1["type5"][100:110].all()
    assert g1["work"][110:200].all() and g1["type5"][200] and g1["work"][201:250].all() and g1["type2"][250:255].all()
    assert g1["type7"][255:].all()
    # Thinker GPU: works, then the request is downstream (type 2 in a closed loop).
    assert g0["work"][2:100].all() and g0["type2"][100:255].all() and not g0["type1"].any()
    # The categories never overlap.
    for g in (g0, g1):
        total = sum(g[k].astype(int) for k in ("work", *census.NOWORK))
        assert (total == 1).all()


def test_a_text_request_leaves_the_talker_gpu_with_no_work_coming():
    text = {k: v for k, v in speech_request().items() if k[0] in ("coordinator", "thinker")}
    _, g1 = classify([(text, False)])
    assert not g1["type1"].any() and not g1["type2"].any() and g1["type7"].all()


def trace(steps_spec):
    """A torch trace with one graph launch per step: (launch_us, graph_start_us, graph_end_us, sync_end_us or None)."""
    ev = []
    for i, (a, gs, ge, se) in enumerate(steps_spec):
        ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch", "ts": a, "dur": 20, "args": {"correlation": i}})
        ev.append({"ph": "X", "cat": "kernel", "name": "k", "ts": gs, "dur": ge - gs, "args": {"correlation": i, "graph id": 1}})
        if se is not None:
            ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaStreamSynchronize", "ts": a + 30, "dur": se - a - 30})
    return {"traceEvents": ev}


def test_steps_that_wait_for_their_own_graph_are_marked_synchronous():
    # Synchronous: each launch waits until its graph ends (+10 us), then 4 ms later launches the next one.
    sync = trace([(0, 50, 24_000, 24_010), (28_010, 28_060, 52_000, 52_010), (56_010, 56_060, 80_000, 80_010)])
    rows = steps.steps_of(sync)
    assert len(rows) == 2 and all(r["waited"] for r in rows)
    assert abs(rows[0]["idle"] - 4_060) < 1 and rows[0]["lead"] < 0 and abs(rows[0]["host_after_sync"] - 4_000) < 1
    # Lookahead: the next launch is issued while the current graph still runs, no sync on it.
    ahead = trace([(0, 50, 24_000, None), (20_000, 24_000, 48_000, None), (44_000, 48_000, 72_000, None)])
    s = steps.summarize(steps.steps_of(ahead))
    assert s["waited_share"] == 0 and s["idle_ms_p50"] == 0 and s["lead_ms_p50"] > 0


def test_stack_adds_up_and_names_the_largest_idle_part():
    v = {"busy": 0.6, "work": 0.8, "type1": 0.0, "type5": 0.0, "type3": 0.0, "imbalance": 0.05, "type2": 0.15, "type7": 0.0}
    s = figs.stack(v)
    assert abs(sum(s.values()) - 1.0) < 1e-9 and abs(s["idle_with_work"] - 0.2) < 1e-9
    assert figs.dominant(v) == "有活空着"
    # nvidia-smi busier than the event-based work share: the no-work parts shrink so the stack still adds up to 1.
    s = figs.stack({**v, "busy": 0.9})
    assert s["idle_with_work"] == 0 and abs(sum(s.values()) - 1.0) < 1e-9
    assert figs.stack({"busy": 0.5}) == {"busy": 0.5}
    assert np.isclose(figs.stack({**v, "busy": 0.2, "work": 0.2, "type7": 0.8, "type2": 0.0, "imbalance": 0.0})["type7"], 0.8)
