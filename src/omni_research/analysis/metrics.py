"""Latency/throughput statistics with explicit sample-size rules.

A percentile ``q`` is reported only when at least ``TAIL_SAMPLES`` observations lie at or above
it, i.e. ``n >= ceil(TAIL_SAMPLES / (1 - q))``: p50 needs 10, p90 50, p95 100, p99 500.
Below that the value is ``None`` and the reason is recorded, so a table never shows a "p99"
computed from a dozen requests.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Callable, Iterable, Sequence

TAIL_SAMPLES = 5
QUANTILES = (0.5, 0.9, 0.95, 0.99)


def min_samples_for(q: float, tail_samples: int = TAIL_SAMPLES) -> int:
    if not 0 < q < 1:
        raise ValueError("q must be in (0, 1)")
    return math.ceil(tail_samples / (1 - q) - 1e-9)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (same convention as numpy's default)."""
    if not values:
        raise ValueError("percentile of empty sequence")
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = q * (len(xs) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float(xs[lo] + (xs[hi] - xs[lo]) * frac)


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[Sequence[float]], float] = statistics.fmean,
    iters: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Percentile-bootstrap confidence interval; None for fewer than 3 values."""
    if len(values) < 3:
        return None
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(iters))
    lo = stats[int((alpha / 2) * iters)]
    hi = stats[min(iters - 1, int((1 - alpha / 2) * iters))]
    return (float(lo), float(hi))


def summarize(values: Iterable[float | None], tail_samples: int = TAIL_SAMPLES, ci: bool = True) -> dict[str, Any]:
    """n, mean, std, min/max, guarded percentiles and bootstrap CIs of mean and median."""
    raw = list(values)
    xs = [float(v) for v in raw if v is not None and not (isinstance(v, float) and math.isnan(v))]
    out: dict[str, Any] = {"n": len(xs), "n_missing": len(raw) - len(xs)}
    if not xs:
        return out
    out.update(
        mean=statistics.fmean(xs),
        std=statistics.stdev(xs) if len(xs) > 1 else 0.0,
        min=min(xs),
        max=max(xs),
    )
    out["cv"] = out["std"] / out["mean"] if out["mean"] else None
    suppressed = {}
    for q in QUANTILES:
        key = f"p{int(round(q * 100))}"
        need = min_samples_for(q, tail_samples)
        if len(xs) >= need:
            out[key] = percentile(xs, q)
        else:
            out[key] = None
            suppressed[key] = f"n={len(xs)} < {need}"
    # The median is always meaningful as a location estimate, even when p50 is suppressed by
    # the tail rule; it is reported separately so tables can show it for small n.
    out["median"] = statistics.median(xs)
    if suppressed:
        out["suppressed"] = suppressed
    if ci and len(xs) >= 3:
        out["mean_ci95"] = bootstrap_ci(xs, statistics.fmean)
        out["median_ci95"] = bootstrap_ci(xs, statistics.median)
    return out


# --- per-request metrics from a replay/benchmark result record --------------------------------


def request_metrics(rec: dict[str, Any]) -> dict[str, float | None]:
    """Latency metrics (seconds) from one result record (see replay.engine.RequestResult)."""
    t0 = rec.get("sent_ns")
    if t0 is None:
        return {}

    def rel(key: str) -> float | None:
        v = rec.get(key)
        return None if v is None else (v - t0) / 1e9

    audio_s = rec.get("output_audio_s")
    gen_s = rel("done_ns")
    first_audio = rel("first_audio_ns")
    last_audio = rel("last_audio_ns")
    m = {
        "e2e_s": gen_s,
        "ttft_s": rel("first_text_ns"),
        "ttfa_s": first_audio,
        "audio_ready_s": last_audio,  # time until the last audio chunk arrived
        "output_text_tokens": rec.get("completion_tokens"),
        "output_audio_s": audio_s,
        "queue_wait_s": rel("dispatched_ns"),  # client-side admission wait if the client queued it
    }
    # Real-time factor of speech output: wall time spent producing audio / audio duration.
    if audio_s and first_audio is not None and last_audio is not None and audio_s > 0:
        m["audio_rtf"] = (last_audio - first_audio) / audio_s if last_audio > first_audio else None
    return m


def playback_underruns(chunks: Sequence[tuple[float, float]], prebuffer_s: float = 0.0) -> dict[str, float]:
    """Simulate a player fed by audio chunks.

    ``chunks`` are (arrival_time_s, chunk_duration_s) in arrival order. Playback starts when the
    first chunk arrives plus ``prebuffer_s``. Returns stall count and total stall time: the
    client-side streaming jitter that a listener would hear.
    """
    if not chunks:
        return {"stalls": 0, "stall_s": 0.0, "chunks": 0}
    start = chunks[0][0] + prebuffer_s
    play_head = start  # wall time at which the buffered audio runs out
    stalls = 0
    stall_s = 0.0
    for i, (arrival, dur) in enumerate(chunks):
        if i == 0:
            play_head = max(start, arrival) + dur
            continue
        if arrival > play_head:
            stalls += 1
            stall_s += arrival - play_head
            play_head = arrival + dur
        else:
            play_head += dur
    return {"stalls": stalls, "stall_s": stall_s, "chunks": len(chunks)}


def throughput(records: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    """Aggregate throughput over the makespan of successful requests (first send -> last done)."""
    ok = [r for r in records if r.get("status") == "ok" and r.get("sent_ns") is not None and r.get("done_ns") is not None]
    if not ok:
        return {"requests": 0, "makespan_s": None}
    t0 = min(r["sent_ns"] for r in ok)
    t1 = max(r["done_ns"] for r in ok)
    span = (t1 - t0) / 1e9
    tok = sum(r.get("completion_tokens") or 0 for r in ok)
    aud = sum(r.get("output_audio_s") or 0.0 for r in ok)
    return {
        "requests": len(ok),
        "makespan_s": span,
        "requests_per_s": len(ok) / span if span > 0 else None,
        "output_tokens_per_s": tok / span if span > 0 else None,
        "output_audio_s_per_s": aud / span if span > 0 else None,
    }


def goodput(records: Sequence[dict[str, Any]], slo: dict[str, float]) -> dict[str, Any]:
    """Requests meeting every SLO bound (e.g. {"ttft_s": 1.0, "ttfa_s": 2.0}), as count/rate."""
    met = 0
    total = 0
    for r in records:
        total += 1
        if r.get("status") != "ok":
            continue
        m = request_metrics(r)
        if all(m.get(k) is not None and m[k] <= bound for k, bound in slo.items()):
            met += 1
    tp = throughput(records)
    span = tp.get("makespan_s")
    return {
        "slo": slo,
        "met": met,
        "total": total,
        "fraction": met / total if total else None,
        "goodput_req_per_s": met / span if span else None,
    }
