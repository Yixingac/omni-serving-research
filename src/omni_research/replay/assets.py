"""Synthetic input assets with checkable answers (no external datasets).

* Spoken questions: the agent tasks (and a few longer read-aloud passages) are synthesized with
  the serving system's own TTS endpoint (``POST /v1/audio/speech``). The manifest records the
  model and revision that produced each clip (``source: synthetic:tts:...``).
* Counting videos: N coloured circles moving on a plain background, rendered frame by frame and
  encoded with ffmpeg. The question is "How many circles...?" and the answer is N.

Everything here is synthetic; manifests say so, and the files can be regenerated from the seed.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .backends import wav_duration
from .payloads import synthetic_text

PASSAGE_TEMPLATES = [
    "Please listen to this note and answer briefly at the end. {body} Question: what is the main topic of this note?",
    "Here is a short message from a colleague. {body} Please summarize the message in one sentence.",
]


async def tts_clip(session, base_url: str, text: str, model: str | None, voice: str = "default",  # noqa: ANN001
                   extra_body: dict[str, Any] | None = None) -> bytes:
    body: dict[str, Any] = {"input": text, "voice": voice, "response_format": "wav"}
    if model:
        body["model"] = model
    body.update(extra_body or {})  # e.g. Qwen3-TTS "references": [{"audio_path": ..., "text": ...}]
    async with session.post(f"{base_url.rstrip('/')}/v1/audio/speech", json=body) as resp:
        data = await resp.read()
        if resp.status != 200:
            raise RuntimeError(f"TTS HTTP {resp.status}: {data[:300]!r}")
        return data


async def generate_spoken_questions(
    base_url: str,
    out_dir: str | Path,
    tasks: list[Any],
    *,
    source_label: str,
    model: str | None = None,
    passages: int = 6,
    seed: int = 0,
    concurrency: int = 4,
    extra_body: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Synthesize audio for agent tasks plus ``passages`` longer clips of increasing length."""
    import aiohttp

    out = Path(out_dir)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    jobs: list[tuple[str, str, dict[str, Any]]] = []
    for t in tasks:
        jobs.append((f"q-{t.task_id}", t.question, {"tags": ["agent_question", t.kind], "question": t.question,
                                                  "answer": t.answer, "task_id": t.task_id}))
    for i in range(passages):
        n_tokens = [40, 80, 160, 240, 320, 400][i % 6]
        body = synthetic_text(n_tokens, rng)
        text = PASSAGE_TEMPLATES[i % len(PASSAGE_TEMPLATES)].format(body=body)
        jobs.append((f"passage-{i:02d}", text, {"tags": ["passage"], "transcript_chars": len(text)}))
    sem = asyncio.Semaphore(concurrency)
    rows: list[dict[str, Any]] = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        async def one(asset_id: str, text: str, meta: dict[str, Any]) -> None:
            async with sem:
                wav = await tts_clip(session, base_url, text, model, extra_body=extra_body)
            path = out / "audio" / f"{asset_id}.wav"
            path.write_bytes(wav)
            dur, rate = wav_duration(wav)
            if dur <= 0:
                raise RuntimeError(f"TTS returned no audio for {asset_id}")
            row = {"asset_id": asset_id, "kind": "audio", "path": f"audio/{asset_id}.wav", "duration_s": round(dur, 3),
                   "sample_rate": rate, "source": source_label, "transcript": text, **meta}
            # Runaway guard: TTS sometimes keeps generating until its frame cap (Qwen3-TTS: 163.84 s for a
            # 75-character question, seen 2026-11-11). Speech runs at roughly 12-16 characters per second, so
            # a clip far longer than its text is excluded from the pools instead of skewing the workload.
            if dur > max(20.0, len(text) / 8.0):
                row["tags"] = ["invalid_runaway"]
                row["note"] = f"{dur:.1f} s for {len(text)} characters: probably did not stop; excluded from pools"
            rows.append(row)
        await asyncio.gather(*(one(*j) for j in jobs))
    rows.sort(key=lambda r: r["asset_id"])
    return rows


def _render_counting_video(path: Path, n_circles: int, seconds: float, fps: int, width: int, height: int,
                           seed: int, ffmpeg: str) -> None:
    rng = random.Random(seed)
    colors = [(230, 57, 70), (29, 53, 87), (69, 123, 157), (42, 157, 143), (233, 196, 106), (244, 162, 97)]
    circles = []
    for i in range(n_circles):
        r = rng.randint(min(width, height) // 14, min(width, height) // 9)
        circles.append({"x": rng.uniform(r, width - r), "y": rng.uniform(r, height - r), "r": r,
                        "vx": rng.uniform(-90, 90), "vy": rng.uniform(-90, 90), "c": colors[i % len(colors)]})
    n_frames = max(1, int(round(seconds * fps)))
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", str(path)],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None
    bg = (245, 245, 240)
    row_bytes = bytes(bg) * width
    for f in range(n_frames):
        frame = bytearray(row_bytes * height)
        for c in circles:
            c["x"] += c["vx"] / fps
            c["y"] += c["vy"] / fps
            if not c["r"] <= c["x"] <= width - c["r"]:
                c["vx"] = -c["vx"]
                c["x"] = min(max(c["x"], c["r"]), width - c["r"])
            if not c["r"] <= c["y"] <= height - c["r"]:
                c["vy"] = -c["vy"]
                c["y"] = min(max(c["y"], c["r"]), height - c["r"])
            cx, cy, r = int(c["x"]), int(c["y"]), c["r"]
            col = bytes(c["c"])
            for y in range(max(0, cy - r), min(height, cy + r + 1)):
                half = int(math.sqrt(max(0, r * r - (y - cy) ** 2)))
                x0, x1 = max(0, cx - half), min(width, cx + half + 1)
                if x1 > x0:
                    start = (y * width + x0) * 3
                    frame[start : start + (x1 - x0) * 3] = col * (x1 - x0)
        proc.stdin.write(frame)
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")


def generate_counting_videos(out_dir: str | Path, *, durations_s=(2, 5, 10, 20), counts=(1, 3, 5), fps: int = 8,
                             width: int = 640, height: int = 360, seed: int = 0,
                             ffmpeg: str | None = None) -> list[dict[str, Any]]:
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")
    out = Path(out_dir)
    (out / "video").mkdir(parents=True, exist_ok=True)
    rows = []
    for d in durations_s:
        for n in counts:
            aid = f"circles-n{n}-{d}s"
            path = out / "video" / f"{aid}.mp4"
            _render_counting_video(path, n, d, fps, width, height, seed * 1000 + n * 37 + int(d), ffmpeg)
            rows.append({"asset_id": aid, "kind": "video", "path": f"video/{aid}.mp4", "duration_s": float(d), "fps": fps,
                         "resolution": f"{width}x{height}", "source": "synthetic:procedural:omni_research.replay.assets",
                         "question": "How many circles are moving in this video? Answer with a number.",
                         "answer": n, "tags": ["count_circles"]})
    return rows


def write_manifest(out_dir: str | Path, rows: list[dict[str, Any]], name: str = "manifest.jsonl") -> Path:
    """Merge rows into the pool manifest (existing asset ids are replaced, others kept)."""
    path = Path(out_dir) / name
    existing: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                existing[d["asset_id"]] = d
    for r in rows:
        existing[r["asset_id"]] = r
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(v) + "\n" for v in sorted(existing.values(), key=lambda d: d["asset_id"])))
    tmp.replace(path)
    return path
