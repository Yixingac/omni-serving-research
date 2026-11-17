"""实验 8 phase-1 report: the supplement rule and the pre-registered verdicts (synthetic summaries only)."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("exp08_report", HERE / "exp08_report.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

LEVELS = (8, 32, 64)


def phase(idle=None, bs=None, flush=0, iters=100, ms_tok=None):
    p = {}
    if idle is not None:
        p["graph_steps"] = {"idle_ms_p50": idle}
    if bs is not None or ms_tok is not None:
        p["steps"] = {"batch_size_decode": {"p50": bs}, "flush_reasons": {"mixed_chunk": flush} if flush else {},
                      "iterations": iters, "flush_only_iterations": 0}
        p["thinker_ms_per_token"] = {"p50": ms_tok}
    return p


def summary(cells):
    """cells: {(cond, c, mode): [value per block]} with value a dict for phase()."""
    runs = [{"run": f"b{i}", "phases": {}} for i in range(3)]
    for (cond, c, mode), per_block in cells.items():
        for i, kw in enumerate(per_block):
            runs[i]["phases"][f"{cond}_c{c}_{mode}"] = phase(**kw)
    return {"per_run": runs}


def full(idle_of, bs_of=lambda cond, c: c / 2, flush_of=lambda cond, c: 0):
    cells = {}
    for cond in ("w3t", "w3t_ns", "w3b", "w3b_ns"):
        for c in LEVELS:
            cells[(cond, c, "C")] = [{"idle": idle_of(cond, c, b)} for b in range(3)]
            cells[(cond, c, "B")] = [{"bs": bs_of(cond, c), "flush": flush_of(cond, c), "ms_tok": 30.0} for b in range(3)]
    return cells


def world(idle_of, **kw):
    """(main, supplement) from one synthetic world: the supplement holds the text C cells at 32 and 64."""
    cells = full(idle_of, **kw)
    supp = {k: v for k, v in cells.items() if k[0] in ("w3t", "w3t_ns") and k[1] in (32, 64) and k[2] == "C"}
    return summary(cells), summary(supp)


def test_text_cells_at_32_and_64_come_from_the_supplement():
    main = summary(full(lambda cond, c, b: 1.0))
    supp = summary({(cond, c, "C"): [{"idle": 7.0}] * 3 for cond in ("w3t", "w3t_ns") for c in (32, 64)})
    assert rp.cell_runs(main, supp, "w3t", 32, "C", "graph_steps.idle_ms_p50") == [7.0, 7.0, 7.0]
    assert rp.cell_runs(main, supp, "w3t", 8, "C", "graph_steps.idle_ms_p50") == [1.0, 1.0, 1.0]
    assert rp.cell_runs(main, supp, "w3b", 32, "C", "graph_steps.idle_ms_p50") == [1.0, 1.0, 1.0]
    # B-phase metrics never come from the supplement (it has only C phases).
    assert rp.cell_runs(main, supp, "w3t", 32, "B", "steps.batch_size_decode.p50") == [16.0, 16.0, 16.0]


def idle_world(cond, c, b):
    return {"w3t": 4.0, "w3t_ns": 0.05, "w3b": 4.0 + 0.1 * b, "w3b_ns": 4.2 - 0.1 * b}[cond]


def test_verdicts_when_lookahead_removes_the_idle_time_of_text():
    v = rp.verdicts(*world(idle_world, flush_of=lambda cond, c: c))
    assert v["H8c"]["verdict"] == "支持"
    assert v["H8d"]["verdict"] == "支持"  # w3b and w3b_ns differ by < 1 ms
    assert v["H8a"]["verdict"] == "推翻"  # both seeded: no lookahead on either side
    assert v["H8b"]["verdict"] == "支持"  # flush share rises with concurrency
    entry = v["phase3_entry"]
    assert entry["met"] is True and all(x["saving_ms"] >= 2 for x in entry["by_level"].values())


def test_inconsistent_direction_refutes_h8c_and_closes_the_entry():
    def idle(cond, c, b):
        if cond == "w3t_ns":
            return 0.05 if b < 2 else 6.0  # block 3 goes the other way
        return idle_world(cond, c, b)
    v = rp.verdicts(*world(idle))
    assert v["H8c"]["verdict"] == "推翻"
    assert v["phase3_entry"]["met"] is False


def test_a_missing_supplement_cell_is_insufficient_and_never_falls_back_to_the_main_blocks():
    main, _ = world(idle_world)
    v = rp.verdicts(main, summary({}))
    assert v["H8c"]["by_level"]["32"]["verdict"] == "数据不足" and v["H8c"]["verdict"] == "数据不足"


def test_missing_cells_are_insufficient_not_refuted():
    def idle(cond, c, b):
        return None if (cond == "w3t_ns" and c == 64) else idle_world(cond, c, b)
    v = rp.verdicts(*world(idle))
    assert v["H8c"]["verdict"] == "数据不足"
    assert v["H8c"]["by_level"]["64"]["verdict"] == "数据不足"


def test_saving_compares_speech_with_the_text_level_of_the_nearest_batch_size():
    bs = {("w3t_ns", 8): 4, ("w3t_ns", 32): 15, ("w3t_ns", 64): 30, ("w3b_ns", 8): 2, ("w3b_ns", 32): 9, ("w3b_ns", 64): 12}
    entry = rp.verdicts(*world(idle_world, bs_of=lambda cond, c: bs.get((cond, c), c / 2)))["phase3_entry"]
    assert {c: x["text_level"] for c, x in entry["by_level"].items()} == {"8": 8, "32": 8, "64": 32}
