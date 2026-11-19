"""The runner's ``plugin`` phase for full-duplex sessions:

    {"kind": "plugin", "module": "duplex.phase", "name": "duplex_s2", "sessions": 2, "duration_s": 300,
     "answer_window_s": 12, "barge_in_every": 0, "stagger_s": 1.0, "save_audio_sessions": 2,
     "skip_if_failed": ["duplex_s1"], "server_events": false}

Each session streams its own seeded script (script.build_script, seed = phase seed, session index) to
``ws://<server>/v1/realtime``. Raw traces, scripts, received audio and metrics go to
``raw/duplex_<phase>/``; the summary returned to the runner (phases.json) is the pooled view.
``server_events`` is handled by the runner itself (request-event recorder around the phase).
"""

from __future__ import annotations

import asyncio
import json
import random
import urllib.request
from pathlib import Path
from typing import Any

from .client import run_session
from .metrics import pool, session_metrics
from .script import build_script, load_clips

DEFAULT_INSTRUCTIONS = None  # the server's default ("Streaming Omni Conversation.")


def failed_before(runner: Any, names: list[str], max_unit_miss: float | None = None) -> str | None:
    """Name of a listed earlier phase that failed (error, skipped, a failed session or, with ``max_unit_miss``,
    a unit miss rate above it: the server did not keep up), else None."""
    for s in getattr(runner, "phase_summaries", []) or []:
        if s.get("phase") not in names:
            continue
        d = s.get("duplex") or {}
        if s.get("error") or s.get("skipped") or s.get("servers_exited") or d.get("sessions_failed"):
            return s.get("phase")
        miss = d.get("unit_miss_rate")
        if max_unit_miss is not None and (miss is None or miss > max_unit_miss):
            return s.get("phase")
    return None


def mean_open_sessions(traces: list[Path]) -> float | None:
    """Sessions open at once, averaged from the first session.updated to the last session.closed (upstream's
    "mean open"); a session counts from its session.updated to its session.closed (or its last record)."""
    spans = []
    for t in traces:
        recs = [json.loads(x) for x in t.read_text().splitlines() if x.strip()] if t.exists() else []
        opened = next((r["t_mono_ns"] for r in recs if r.get("type") == "session.updated" and r.get("d") == "recv"), None)
        if opened is None:
            continue
        closed = next((r["t_mono_ns"] for r in recs if r.get("type") == "session.closed" and r.get("d") == "recv"),
                      recs[-1]["t_mono_ns"])
        spans.append((opened, closed))
    if not spans:
        return None
    lo, hi = min(a for a, _ in spans), max(b for _, b in spans)
    return sum(b - a for a, b in spans) / (hi - lo) if hi > lo else float(len(spans))


def probe_capabilities(base_url: str, timeout_s: float = 30.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(f"{base_url}/v1/realtime/capabilities", timeout=timeout_s) as r:
            return {"status": r.status, "body": json.loads(r.read().decode())}
    except Exception as exc:  # noqa: BLE001 - recorded
        return {"status": None, "error": f"{type(exc).__name__}: {exc}"[:300]}


def session_update(ph: dict[str, Any]) -> dict[str, Any]:
    upd: dict[str, Any] = {"output_modalities": list(ph.get("output_modalities", ["audio"]))}
    if ph.get("instructions"):
        upd["instructions"] = ph["instructions"]
    ext: dict[str, Any] = {}
    if ph.get("sampling"):
        ext["sampling"] = dict(ph["sampling"])
    if ph.get("max_slice_nums"):
        ext["max_slice_nums"] = int(ph["max_slice_nums"])
    if ext:
        upd["sglang"] = ext
    return upd


def synthetic_frames(duration_s: float, fps: float, seed: int, size: tuple[int, int] = (640, 360)) -> list[tuple[float, bytes]]:
    """Deterministic camera frames (coloured circles on grey), JPEG, at t_ms = k * 1000 / fps."""
    import io

    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    colors = [(230, 57, 70), (29, 53, 87), (69, 123, 157), (42, 157, 143), (233, 196, 106)]
    circles = [{"x": rng.uniform(40, size[0] - 40), "y": rng.uniform(40, size[1] - 40), "r": rng.randint(18, 40),
                "vx": rng.uniform(-60, 60), "vy": rng.uniform(-60, 60), "c": colors[i % len(colors)]} for i in range(rng.randint(2, 5))]
    out = []
    k = 0
    while k * 1000.0 / fps < duration_s * 1000.0:
        t = k / fps
        img = Image.new("RGB", size, (200, 200, 200))
        dr = ImageDraw.Draw(img)
        for c in circles:
            x = (c["x"] + c["vx"] * t) % size[0]
            y = (c["y"] + c["vy"] * t) % size[1]
            dr.ellipse([x - c["r"], y - c["r"], x + c["r"], y + c["r"]], fill=c["c"])
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        out.append((k * 1000.0 / fps, buf.getvalue()))
        k += 1
    return out


async def run_phase(ctx: dict[str, Any], ph: dict[str, Any]) -> dict[str, Any]:
    runner = ctx["runner"]
    name = ctx["phase"]
    skipped = failed_before(runner, list(ph.get("skip_if_failed", [])), ph.get("stop_if_unit_miss_above"))
    if skipped:
        return {"skipped": True, "reason": f"earlier phase {skipped} failed (escalation stops)"}
    out = Path(ctx["raw_dir"]) / f"duplex_{name}"
    out.mkdir(parents=True, exist_ok=True)
    base = ctx["servers"][ph.get("server", "s0")]
    caps = await asyncio.to_thread(probe_capabilities, base)
    (out / "capabilities.json").write_text(json.dumps(caps, indent=1))
    data_root = Path(getattr(runner, "data_root", ctx["exp_dir"]))
    pool_dir = Path(ph["asset_dir"]) if ph.get("asset_dir") else data_root / "assets" / ph.get("asset_pool", "duplex-v1")
    clips = load_clips(pool_dir, ph.get("tag", "agent_question"))
    n = int(ph["sessions"])
    seed = int(ph.get("session_seed", ctx["seed"]))
    scripts = [build_script(clips, duration_s=float(ph["duration_s"]), seed=seed, session_index=i,
                            lead_in_s=float(ph.get("lead_in_s", 2.0)), answer_window_s=float(ph.get("answer_window_s", 12.0)),
                            barge_in_every=int(ph.get("barge_in_every", 0)),
                            barge_in_offset_s=float(ph.get("barge_in_offset_s", 3.0))) for i in range(n)]
    rng = random.Random(seed * 7919 + n)
    stagger = float(ph.get("stagger_s", 1.0))
    offsets = [0.0] + [round(rng.uniform(0.0, stagger), 3) for _ in range(n - 1)]
    fps = float(ph.get("image_fps", 0.0))
    frames = [synthetic_frames(float(ph["duration_s"]), fps, seed * 31 + i) if fps > 0 else None for i in range(n)]
    (out / "scripts.json").write_text(json.dumps({"start_offsets_s": offsets, "image_fps": fps,
                                                  "sessions": [s.to_json() for s in scripts]}, indent=1))
    url = base.replace("http://", "ws://", 1).replace("https://", "wss://", 1) + "/v1/realtime"
    upd = session_update(ph)
    save_audio = int(ph.get("save_audio_sessions", 2))
    results = await asyncio.gather(*[
        run_session(url, s, trace_path=out / f"session_{i:02d}.jsonl", session_update=upd, frames=frames[i],
                    audio_path=out / f"session_{i:02d}_output.wav" if i < save_audio else None,
                    pace_scale=float(ph.get("pace_scale", 1.0)), start_delay_s=offsets[i],
                    drain_timeout_s=float(ph.get("drain_timeout_s", 180.0)),
                    connect_retries=int(ph.get("connect_retries", 3)),
                    connect_backoff_s=float(ph.get("connect_backoff_s", 0.25)))
        for i, s in enumerate(scripts)])
    per = []
    for i, s in enumerate(scripts):
        recs = [json.loads(line) for line in (out / f"session_{i:02d}.jsonl").read_text().splitlines() if line.strip()]
        per.append(session_metrics(recs, s.to_json()))
    pooled = pool(per)
    (out / "metrics.json").write_text(json.dumps({"pooled": pooled, "sessions": per}, indent=1, default=str))
    compact = {k: pooled[k] for k in ("sessions", "sessions_failed", "errors", "fatal_errors", "units_planned",
                                      "units_missed", "unit_miss_rate", "turns", "answered", "no_response",
                                      "deadline_miss_all_turns", "audio_s_total", "decisions", "pacing_violations",
                                      "replies_with_audio", "replies_with_stall")}
    compact["unit_lag_ms"] = {k: pooled["unit_lag_ms"][k] for k in ("n", "p50", "p95", "p99", "max")}
    compact["ttfa_ms"] = {k: pooled["ttfa_ms"][k] for k in ("n", "p50", "p95", "p99", "max")}
    compact["session_status"] = [r.status for r in results]
    compact["session_reasons"] = [r.reason for r in results if r.reason]
    # Upstream's serving targets: no admission denial (a retried connection may have waited for a slot, so the
    # sessions did not all overlap) and the measured overlap of the sessions' open intervals.
    compact["sessions_with_admission_denials"] = sum(1 for r in results if r.admission_attempts > 1)
    compact["mean_open_sessions"] = mean_open_sessions([out / f"session_{i:02d}.jsonl" for i in range(n)])
    return {"duplex": compact, "duplex_dir": str(out), "capabilities_status": caps.get("status")}
