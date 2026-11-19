"""P0 report: log parsing, talker window statistics and the registered verdict rules (no GPU, no data)."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("ppv_p0_report", HERE / "ppv_p0_report.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

LINE = ("2026-11-19 01:02:03,250 [INFO] sglang.srt.managers.scheduler_components.metrics_reporter: Decode batch, "
        "#running-req: {r}, #token: {t}, token usage: {u}, cuda graph: {g}, gen throughput (token/s): {gen}, #queue-req: {q}, ")


def test_parse_decode_uses_the_run_utc_offset_and_reads_usage():
    rows = rp.parse_decode([LINE.format(r=40, t=20000, u=0.15, g="True", gen=9000.0, q=120), "unrelated"], utc_offset_s=-5 * 3600)
    assert len(rows) == 1
    r = rows[0]
    # 01:02:03.250 local at UTC-5 is 06:02:03.250 UTC
    assert abs(r["ts"] - (1795068123.25)) < 1e-6
    assert (r["running"], r["tokens"], r["usage"], r["graph"], r["queue"]) == (40, 20000, 0.15, True, 120)
    assert rp.is_talker(r) and not rp.is_talker({**r, "tokens": 40 * 150})


def test_capture_max_bs_reads_every_capture():
    lines = ["... Capture target decode CUDA graph begin. backend=full, num_tokens_per_req=1, bs=[1, 2, 4, 8, 12, 16, 24, 32], avail mem=38.32 GB",
             "... Capture target decode CUDA graph begin. backend=full, num_tokens_per_req=1, bs=[1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64], avail mem=13.77 GB",
             "... Capture target decode CUDA graph end. elapsed=4.00 s"]
    assert rp.capture_max_bs(lines) == [32, 64]


def _row(running, queue=0, usage=0.2, graph=True, gen=8000.0, tokens=None):
    return {"ts": 0.0, "running": running, "tokens": tokens or running * 500, "usage": usage, "graph": graph,
            "gen_tok_s": gen, "queue": queue}


def test_talker_window_cap_queue_kv_trigger_and_graph_by_batch_size():
    rows = [_row(64, queue=10), _row(64, queue=0), _row(50, queue=5, usage=0.95, graph=False), _row(20, graph=True)]
    w = rp.talker_window(rows, cap=64, graph_max=32)
    assert w["lines"] == 4 and w["running_max"] == 64 and w["above_32_share"] == 0.75
    assert w["at_cap_share"] == 0.5 and w["at_cap_with_queue_share"] == 0.25
    assert w["kv_bound_share"] == 0.25  # only the line below the cap, with a queue and usage >= 0.9
    assert w["graph_share_le_graph_max"] == 1.0 and w["graph_share_above_graph_max"] == 2 / 3
    assert rp.talker_window([], 32, 32) == {"lines": 0}


def test_step_curve_drops_idle_lines_and_needs_enough_lines():
    rows = [_row(32, gen=8000.0)] * 25 + [_row(32, gen=10.0)] * 50 + [_row(64, gen=9600.0)] * 5
    curve = rp.step_curve(rows)
    assert curve == [{"batch": 32, "lines": 25, "gen_tok_s_median": 8000.0, "step_ms": 4.0}]


def test_goodput_needs_ttft_and_realtime_factor():
    ns = 1_000_000_000
    ok = {"status": "ok", "sent_ns": 0, "first_text_ns": int(0.5 * ns), "done_ns": 3 * ns, "output_audio_s": 8.0}
    slow_ttft = {**ok, "first_text_ns": int(1.5 * ns)}
    slow_audio = {**ok, "done_ns": 5 * ns}  # 5 s for 8 s of audio > 0.5
    failed = {"status": "error", "sent_ns": 0, "done_ns": 2 * ns}
    warm = {**ok, "warmup": True}
    g = rp.goodput([ok, slow_ttft, slow_audio, failed, warm])
    assert g["met"] == 1 and g["measured"] == 4 and g["share"] == 0.25
    assert abs(g["goodput_rps"] - 1 / 5.0) < 1e-9


def test_registered_verdicts():
    assert rp.verdict_h0a([0.06, 0.10, 0.05], [40, 48, 36]) == "支持"
    assert rp.verdict_h0a([0.06, 0.10, 0.05], [40, 32, 36]) == "不确定"  # running median not above 32 in every block
    assert rp.verdict_h0a([0.01, -0.03, 0.019], [40, 40, 40]) == "反驳"
    assert rp.verdict_h0a([0.06, None, 0.07], [40, 40, 40]) == "不确定"  # a missing block rules out 支持
    assert rp.verdict_h0c([0.0, 0.01, 0.02]) == "支持" and rp.verdict_h0c([-0.1, -0.2, -0.01]) == "反驳"
    assert rp.verdict_h0c([-0.1, 0.1, -0.01]) == "不确定"
    assert rp.effect_label([0.05, 0.08, 0.06]) == "有贡献"
    assert rp.effect_label([0.01, -0.02, 0.0]) == "没有可测的贡献"
    assert rp.effect_label([0.01, 0.03, 0.0]) == "不确定"


def test_h0b_classes():
    gpu = lambda busy: {"talker": {"busy": busy}}  # noqa: E731
    cpu_bound = {"bottleneck": {"stage": "talker"}, "gpu": gpu(0.70), "talker_cpu_percent": 99.0}
    assert rp.classify_h0b(cpu_bound, False) == ["CPU/调度受限"]
    assert rp.classify_h0b({"bottleneck": {"stage": "talker"}, "gpu": gpu(0.93), "talker_cpu_percent": 99.0}, True) == ["GPU 受限", "KV 预留受限"]
    assert rp.classify_h0b({"bottleneck": {"stage": "code2wav"}, "gpu": gpu(0.5), "talker_cpu_percent": 60.0}, False) == ["瓶颈转移到 code2wav"]
    assert rp.classify_h0b({"bottleneck": {"stage": "thinker"}, "gpu": gpu(0.5), "talker_cpu_percent": 60.0}, False) == ["未归类"]


def test_rel_by_block_and_interval():
    caps = {("A", 1): 18.0, ("B", 1): 19.8, ("A", 2): 17.0, ("B", 2): 18.7, ("A", 3): 18.0}
    rel = rp.rel_by_block(caps, "B", "A")
    assert [round(x, 6) if x is not None else None for x in rel] == [0.1, 0.1, None]
    ti = rp.t_interval(rel)
    assert ti["n"] == 2 and abs(ti["mean"] - 0.1) < 1e-12
    assert all(abs(x - 0.1) < 1e-9 for x in ti["ci95"])


def test_summary_marks_the_graph_decomposition_blocked_without_arm_g():
    run = lambda arm, block, cap: {"arm": arm, "block": block, "capacity": {"w3b": cap, "w3l": None}, "levels": []}  # noqa: E731
    runs = [run(a, b, c) for b in (1, 2, 3) for a, c in (("A", 17.0), ("B", 22.0), ("R", 23.0))]
    s = rp.summarize(runs)
    assert s["G_decomposition"]["status"] == "BLOCKED"
    assert s["H0c"]["verdict"] == "反驳" and s["H0a"]["relative"]["n"] == 3
