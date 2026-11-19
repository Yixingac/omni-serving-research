"""R3 camera replay driver with a fake backend (no GPU, no network, no server)."""

import asyncio
import json
import shutil
import time

import pytest

from omni_research.replay.backends import RequestResult
from omni_research.sessions import camera


class SlowBackend:
    """Answers every request after ``latency_s`` with the number 3."""

    def __init__(self, latency_s):
        self.latency_s = latency_s
        self.requests = []
        self.results = []
        self.closed = False

    async def run(self, req):
        self.requests.append(req)
        res = RequestResult(request_id=req.request_id, backend="fake", metadata=dict(req.metadata))
        res.sent_ns = time.monotonic_ns()
        await asyncio.sleep(self.latency_s / 2)
        res.first_text_ns = time.monotonic_ns()
        await asyncio.sleep(self.latency_s / 2)
        res.done_ns = time.monotonic_ns()
        res.text = "3"
        self.results.append(res)
        return res

    async def close(self):
        self.closed = True


def _run(tmp_path, backend, **ph):
    ctx = {"phase": "cam", "seed": 5, "raw_dir": tmp_path / "raw", "backend_factory": lambda: backend}
    base = {"sessions": 2, "fps": 5.0, "warmup_s": 0.2, "duration_s": 1.0, "width": 64, "height": 36, "start_delay_s": 0.05}
    return asyncio.run(camera.run_phase(ctx, base | ph))


def test_tick_plan_covers_the_window_and_staggers_sessions():
    ticks = camera.tick_plan(sessions=3, fps=2.0, warmup_s=1.0, duration_s=4.0, seed=1)
    per = {s: [t for t in ticks if t["session"] == s] for s in range(3)}
    for s, ts in per.items():
        assert len(ts) == 10  # 5 s at 2 per second
        assert all(abs((b["t_s"] - a["t_s"]) - 0.5) < 1e-9 for a, b in zip(ts, ts[1:]))
        assert sum(t["warmup"] for t in ts) == 2
    assert len({round(ts[0]["t_s"], 6) for ts in per.values()}) == 3  # random phase per session
    assert [t["t_s"] for t in ticks] == sorted(t["t_s"] for t in ticks)


def test_fast_server_sends_every_tick_and_records_answers(tmp_path):
    be = SlowBackend(0.02)
    out = _run(tmp_path, be)
    assert out["dropped"] == 0 and out["sent"] == out["ticks"] and out["late_ratio"] == 0.0
    assert be.closed
    reqs = [json.loads(x) for x in (tmp_path / "raw" / "requests_cam.jsonl").read_text().splitlines()]
    assert len(reqs) == len(be.requests) and len({r["request_id"] for r in reqs}) == len(reqs)
    first = be.requests[0]
    parts = first.messages[0]["content"]
    assert parts[0]["type"] == "text" and parts[0]["text"].startswith("Camera ")
    assert parts[1]["type"] == "image_url" and parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert first.seed is None and first.output_modalities == ["text"]
    assert len({r.messages[0]["content"][1]["image_url"]["url"] for r in be.requests}) == len(be.requests)
    assert all(r["expected_count"] in range(1, 6) for r in reqs)


def test_slow_server_drops_ticks_while_busy_and_marks_late_answers(tmp_path):
    be = SlowBackend(0.45)  # longer than the 0.2 s frame interval
    out = _run(tmp_path, be)
    ticks = [json.loads(x) for x in (tmp_path / "raw" / "camera_cam.jsonl").read_text().splitlines()]
    assert out["dropped"] > 0 and 0 < out["drop_ratio"] < 1
    assert out["late_ratio"] == 1.0
    dropped = [t for t in ticks if t.get("dropped")]
    assert dropped and all(not t["sent"] for t in dropped)
    # at most one request in flight per session under drop_if_busy
    for s in (0, 1):
        spans = sorted((r.sent_ns, r.done_ns) for r in be.results if f"-s{s:03d}-" in r.request_id)
        assert all(b[0] >= a[1] for a, b in zip(spans, spans[1:]))


def test_send_all_never_drops(tmp_path):
    out = _run(tmp_path, SlowBackend(0.45), policy="send_all")
    assert out["dropped"] == 0 and out["late_ratio"] == 1.0


def test_unknown_shape_or_policy_fails(tmp_path):
    with pytest.raises(ValueError):
        _run(tmp_path, SlowBackend(0.01), shape="stream")
    with pytest.raises(ValueError):
        _run(tmp_path, SlowBackend(0.01), policy="queue")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_clip_shape_sends_a_video(tmp_path):
    be = SlowBackend(0.02)
    _run(tmp_path, be, shape="clip", duration_s=0.4, warmup_s=0.0, clip_s=0.5)
    part = be.requests[0].messages[0]["content"][1]
    assert part["type"] == "video_url" and part["video_url"]["url"].startswith("data:video/mp4;base64,")


def test_cells_differ_in_content_unless_they_share_a_content_key(tmp_path):
    a, b, c = SlowBackend(0.01), SlowBackend(0.01), SlowBackend(0.01)
    base = {"sessions": 1, "fps": 5.0, "warmup_s": 0.0, "duration_s": 0.4, "width": 64, "height": 36, "start_delay_s": 0.01}
    for be, phase, extra in ((a, "cellA", {}), (b, "cellB", {}), (c, "cellC", {"content_key": "cellA"})):
        ctx = {"phase": phase, "seed": 5, "raw_dir": tmp_path / phase, "backend_factory": lambda be=be: be}
        asyncio.run(camera.run_phase(ctx, base | extra))
    img = lambda be: [r.messages[0]["content"][1]["image_url"]["url"] for r in be.requests]  # noqa: E731
    assert not set(img(a)) & set(img(b))
    assert img(a) == img(c)
