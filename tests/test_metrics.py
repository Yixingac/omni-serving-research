import pytest

from omni_research.analysis import metrics as m


def test_min_samples_rule():
    assert m.min_samples_for(0.5) == 10
    assert m.min_samples_for(0.9) == 50
    assert m.min_samples_for(0.95) == 100
    assert m.min_samples_for(0.99) == 500


def test_percentile_matches_numpy_linear():
    import numpy as np

    xs = [3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.3]
    for q in (0.1, 0.5, 0.9, 0.95):
        assert m.percentile(xs, q) == pytest.approx(float(np.percentile(xs, q * 100)))


def test_summarize_suppresses_unsupported_tails():
    s = m.summarize([float(i) for i in range(40)])
    assert s["n"] == 40 and s["p50"] is not None
    assert s["p95"] is None and s["p99"] is None
    assert "p95" in s["suppressed"] and s["median"] == pytest.approx(19.5)
    big = m.summarize([float(i % 100) for i in range(600)])
    assert big["p99"] is not None
    assert m.summarize([None, None]) == {"n": 0, "n_missing": 2}


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    a = m.bootstrap_ci(xs, seed=1)
    assert a == m.bootstrap_ci(xs, seed=1)
    assert a[0] <= 3.5 <= a[1]
    assert m.bootstrap_ci([1.0, 2.0]) is None


def _rec(sent, ft, fa, la, done, audio=0.0, tok=10, status="ok"):
    s = 1_000_000_000
    return {"status": status, "sent_ns": sent * s, "first_text_ns": ft and ft * s, "first_audio_ns": fa and fa * s,
            "last_audio_ns": la and la * s, "done_ns": done * s, "output_audio_s": audio, "completion_tokens": tok}


def test_request_metrics_and_rtf():
    r = m.request_metrics(_rec(10, 10.5, 11, 13, 13.2, audio=4.0))
    assert r["ttft_s"] == pytest.approx(0.5) and r["ttfa_s"] == pytest.approx(1.0)
    assert r["e2e_s"] == pytest.approx(3.2) and r["audio_ready_s"] == pytest.approx(3.0)
    assert r["audio_rtf"] == pytest.approx(0.5)
    assert m.request_metrics({}) == {}


def test_throughput_and_goodput():
    recs = [_rec(0, 0.2, None, None, 1, tok=10), _rec(1, 1.9, None, None, 3, tok=30), _rec(0, None, None, None, 2, status="error")]
    tp = m.throughput(recs)
    assert tp["requests"] == 2 and tp["makespan_s"] == pytest.approx(3.0)
    assert tp["output_tokens_per_s"] == pytest.approx(40 / 3)
    gp = m.goodput(recs, {"ttft_s": 0.5})
    assert gp["met"] == 1 and gp["total"] == 3


def test_playback_underruns():
    # 1 s chunks arriving every 1.2 s: each later chunk arrives 0.2 s after the buffer ran dry.
    chunks = [(0.0, 1.0), (1.2, 1.0), (2.4, 1.0)]
    out = m.playback_underruns(chunks)
    assert out["stalls"] == 2 and out["stall_s"] == pytest.approx(0.4)
    # A 0.5 s prebuffer absorbs the jitter entirely.
    assert m.playback_underruns(chunks, prebuffer_s=0.5)["stalls"] == 0
    assert m.playback_underruns([])["stalls"] == 0


def test_normalized_latency_per_audio_second_and_token():
    r = m.request_metrics(_rec(0, 0.1, 2.0, 2.0, 3.0, audio=12.0, tok=60))
    assert r["e2e_per_audio_s"] == pytest.approx(0.25)
    assert r["e2e_per_token_s"] == pytest.approx(0.05)
