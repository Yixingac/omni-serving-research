"""R3: replay synthetic camera frames from several sessions over the request-response path.

Each session is a camera that delivers a frame every 1/fps seconds (``fps`` is the rate at which the robot
asks about what it sees). At every tick the session sends one perception request about the newest view:

- shape ``frame``: the newest frame as one image (PNG data URL);
- shape ``clip``: the last ``clip_s`` seconds of the camera at ``capture_fps`` as one short MP4 (exercises
  video decode and frame sampling, the path 实验 14 found serial).

Policy ``drop_if_busy`` (the default) models a perception loop that keeps one request in flight: a tick that
finds the session's previous request still running is dropped (never sent). ``send_all`` sends every tick
(open loop per session). A sent tick is late when its answer completes after the session's next tick.

This is a replay of synthetic frames through /v1/chat/completions, not a continuous real-camera stream and
not the full-duplex /v1/realtime path.

Records: ``raw/requests_<phase>.jsonl`` (one per sent request, the runner's usual fields) and
``raw/camera_<phase>.jsonl`` (one per tick, dropped ones included). All times are time.monotonic_ns.
"""

from __future__ import annotations

import asyncio
import math
import os
import random
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..replay.backends import ModelRequest
from ..replay.drivers import JsonlSink
from .frames import Camera, content_seed, data_url, encode_clip

DEFAULT_QUESTION = "How many circles are in this {what}? Answer with a number."


def tick_plan(sessions: int, fps: float, warmup_s: float, duration_s: float, seed: int) -> list[dict[str, Any]]:
    """Every tick of every session: (session, tick, offset from the phase start in seconds, warmup flag).
    Sessions start at a random phase within the first frame interval so their ticks do not line up."""
    rng = random.Random(seed)
    interval = 1.0 / fps
    out = []
    for s in range(sessions):
        offset = rng.uniform(0.0, interval)
        n = int(math.floor((warmup_s + duration_s - offset) / interval)) + 1
        for k in range(max(0, n)):
            t = offset + k * interval
            if t < warmup_s + duration_s:
                out.append({"session": s, "tick": k, "t_s": t, "warmup": t < warmup_s})
    out.sort(key=lambda x: (x["t_s"], x["session"]))
    return out


def frame_index(t_s: float, capture_fps: float) -> int:
    return int(round(t_s * capture_fps))


def first_number(text: str | None) -> int | None:
    m = re.search(r"-?\d+", text or "")
    return int(m.group()) if m else None


def render_media(ticks: list[dict[str, Any]], cameras: list[Camera], shape: str, clip_s: float, ffmpeg: str | None,
                 workers: int = 8) -> dict[tuple[int, int], str]:
    """Data URL per (session, tick), rendered before the phase so the client does no media work while measuring.
    One worker per session walks its ticks in order and keeps only the frames of the current clip window."""
    by_session: dict[int, list[dict[str, Any]]] = {}
    for t in ticks:
        by_session.setdefault(t["session"], []).append(t)

    def one_session(s: int) -> dict[tuple[int, int], str]:
        cam, out, window = cameras[s], {}, {}
        n = max(1, int(round(clip_s * cam.capture_fps)))
        for t in sorted(by_session[s], key=lambda x: x["tick"]):
            last = frame_index(t["t_s"], cam.capture_fps)
            if shape == "frame":
                out[(s, t["tick"])] = data_url(cam.png(last), "image/png")
                continue
            for k in [k for k in window if k < last - n + 1]:
                del window[k]
            out[(s, t["tick"])] = data_url(encode_clip(cam, last - n + 1, n, ffmpeg or "ffmpeg", window), "video/mp4")
        return out

    media: dict[tuple[int, int], str] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for part in pool.map(one_session, sorted(by_session)):
            media.update(part)
    return media


def build_request(phase: str, t: dict[str, Any], url: str, shape: str, question: str, max_tokens: int,
                  timeout_s: float) -> ModelRequest:
    what = "image" if shape == "frame" else "video"
    part = {"type": "image_url", "image_url": {"url": url}} if shape == "frame" else {"type": "video_url", "video_url": {"url": url}}
    # The session and tick in the text make every prompt unique from its first user tokens on, so no request
    # can reuse another's cached prefix beyond the shared system prompt.
    text = f"Camera {t['session']}, frame {t['tick']}. " + question.format(what=what)
    return ModelRequest(
        request_id=f"{phase}-s{t['session']:03d}-k{t['tick']:05d}",
        messages=[{"role": "user", "content": [{"type": "text", "text": text}, part]}],
        input_modalities=[what, "text"], output_modalities=["text"], max_tokens=max_tokens, stream=True,
        temperature=0.0, seed=None, timeout_s=timeout_s,
        metadata={"payload": {"workload_id": f"R3_camera_{shape}", "shape": shape, "media_bytes_b64": len(url)}})


async def run_phase(ctx: dict[str, Any], ph: dict[str, Any]) -> dict[str, Any]:
    phase = ctx["phase"]
    seed = content_seed(int(ph.get("seed", ctx.get("seed", 0))), str(ph.get("content_key", phase)))
    sessions, fps = int(ph["sessions"]), float(ph["fps"])
    shape = ph.get("shape", "frame")
    if shape not in ("frame", "clip"):
        raise ValueError(f"unknown camera shape {shape!r}")
    policy = ph.get("policy", "drop_if_busy")
    if policy not in ("drop_if_busy", "send_all"):
        raise ValueError(f"unknown frame policy {policy!r}")
    warmup_s, duration_s = float(ph.get("warmup_s", 10.0)), float(ph.get("duration_s", 75.0))
    capture_fps, clip_s = float(ph.get("capture_fps", 8.0)), float(ph.get("clip_s", 2.0))
    runner = ctx.get("runner")
    ffmpeg = ph.get("ffmpeg") or ((runner.cfg.get("runtime", {}) or {}).get("ffmpeg") if runner is not None else None) \
        or shutil.which("ffmpeg")
    ffmpeg = os.path.expanduser(ffmpeg) if ffmpeg else None
    cameras = [Camera(seed=seed, session=s, capture_fps=capture_fps, width=int(ph.get("width", 640)),
                      height=int(ph.get("height", 360))) for s in range(sessions)]
    ticks = tick_plan(sessions, fps, warmup_s, duration_s, seed)
    t_render = time.monotonic()
    media = render_media(ticks, cameras, shape, clip_s, ffmpeg, int(ph.get("render_workers", 8)))
    render_s = time.monotonic() - t_render

    factory = ctx.get("backend_factory") or (lambda: runner.backend(keep_text=True))
    backend = factory()
    raw = Path(ctx["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)
    req_sink = JsonlSink(raw / f"requests_{phase}.jsonl")
    tick_sink = JsonlSink(raw / f"camera_{phase}.jsonl")
    busy: dict[int, int] = {s: 0 for s in range(sessions)}
    tasks: list[asyncio.Task] = []
    question = ph.get("question", DEFAULT_QUESTION)
    max_tokens, timeout_s = int(ph.get("max_tokens", 16)), float(ph.get("request_timeout_s", 60.0))
    interval_ns = int(1e9 / fps)
    start_ns = time.monotonic_ns() + int(float(ph.get("start_delay_s", 0.5)) * 1e9)

    async def send(t: dict[str, Any], scheduled_ns: int) -> None:
        s = t["session"]
        busy[s] += 1
        req = build_request(phase, t, media[(s, t["tick"])], shape, question, max_tokens, timeout_s)
        try:
            res = await backend.run(req)
        finally:
            busy[s] -= 1
        rec = res.to_dict()
        rec.pop("text", None)
        answer = first_number(res.text)
        late = res.done_ns is not None and res.done_ns > scheduled_ns + interval_ns
        rec.update(phase=phase, kind="camera", session=s, tick=t["tick"], warmup=t["warmup"], scheduled_ns=scheduled_ns,
                   rate_rps=sessions * fps, sessions=sessions, fps=fps, shape=shape, policy=policy,
                   expected_count=cameras[s].n_circles, answer=answer,
                   answer_correct=answer == cameras[s].n_circles if answer is not None else False, late=late)
        req_sink.write(rec)
        tick_sink.write({"session": s, "tick": t["tick"], "scheduled_ns": scheduled_ns, "warmup": t["warmup"], "sent": True,
                         "request_id": req.request_id, "status": res.status, "late": late})

    try:
        for t in ticks:
            scheduled_ns = start_ns + int(t["t_s"] * 1e9)
            delay = (scheduled_ns - time.monotonic_ns()) / 1e9
            if delay > 0:
                await asyncio.sleep(delay)
            if policy == "drop_if_busy" and busy[t["session"]] > 0:
                tick_sink.write({"session": t["session"], "tick": t["tick"], "scheduled_ns": scheduled_ns,
                                 "warmup": t["warmup"], "sent": False, "dropped": True})
                continue
            tasks.append(asyncio.create_task(send(t, scheduled_ns)))
        if tasks:
            await asyncio.wait(tasks, timeout=timeout_s + 30.0)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await backend.close()
        req_sink.close()
        tick_sink.close()
    return summarize(tick_sink.records, req_sink.records) | {"render_s": round(render_s, 2), "shape": shape,
                                                             "policy": policy, "sessions": sessions, "fps": fps}


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def summarize(ticks: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    """Measured (non-warmup) ticks only. Percentiles of first response are over sent ticks that answered."""
    meas = [t for t in ticks if not t.get("warmup")]
    sent = [r for r in records if not r.get("warmup")]
    ok = [r for r in sent if r.get("status") == "ok" and r.get("first_text_ns") and r.get("sent_ns")]
    first = [(r["first_text_ns"] - r["sent_ns"]) / 1e9 for r in ok]
    done = [(r["done_ns"] - r["sent_ns"]) / 1e9 for r in ok if r.get("done_ns")]
    n = len(meas)
    return {
        "ticks": n,
        "sent": sum(1 for t in meas if t.get("sent")),
        "dropped": sum(1 for t in meas if t.get("dropped")),
        "drop_ratio": (sum(1 for t in meas if t.get("dropped")) / n) if n else None,
        "late_ratio": (sum(1 for r in sent if r.get("late")) / len(sent)) if sent else None,
        "errors": sum(1 for r in sent if r.get("status") != "ok"),
        "first_response_s": {"p50": _pct(first, 0.5), "p95": _pct(first, 0.95), "p99": _pct(first, 0.99), "n": len(first)},
        "completion_s": {"p50": _pct(done, 0.5), "p95": _pct(done, 0.95), "n": len(done)},
        "answer_correct_share": (sum(1 for r in ok if r.get("answer_correct")) / len(ok)) if ok else None,
    }

