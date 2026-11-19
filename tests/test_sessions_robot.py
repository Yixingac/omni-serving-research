"""R4 robot session driver with a fake backend (no GPU, no network, no server)."""

import asyncio
import json
import struct
import time

from omni_research.replay.backends import RequestResult
from omni_research.sessions import robot_sessions as rs


class EchoBackend:
    def __init__(self):
        self.requests = []

    async def run(self, req):
        self.requests.append(req)
        res = RequestResult(request_id=req.request_id, backend="fake", metadata=dict(req.metadata))
        res.sent_ns = time.monotonic_ns()
        await asyncio.sleep(0.001)
        res.first_text_ns = time.monotonic_ns()
        res.done_ns = time.monotonic_ns()
        res.text = f"answer to {req.request_id}"
        res.prompt_tokens = 100 + 10 * len(req.messages)
        return res

    async def close(self):
        pass


def _wav(path, n=320):
    pcm = b"\x02\x00" * n
    fmt = struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
    path.write_bytes(b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt +
                     b"data" + struct.pack("<I", len(pcm)) + pcm)
    return str(path)


def _run(tmp_path, **ph):
    clips = [{"path": _wav(tmp_path / f"q{i}.wav"), "transcript": f"question {i}"} for i in range(3)]
    be = EchoBackend()
    base = {"sessions": 3, "turns": [2, 3], "ramp_s": 0.01, "think_s": [0.001, 0.1],
            "tool": {"p": 0.5, "latency_s": [0.001, 0.1], "result_tokens": [5, 10]}, "audio_clips": clips,
            "width": 64, "height": 36}
    ctx = {"phase": "r4", "seed": 11, "raw_dir": tmp_path / "raw", "backend_factory": lambda: be}
    out = asyncio.run(rs.run_phase(ctx, base | ph))
    return out, be


def test_session_plans_are_seeded_and_differ_between_sessions():
    ph = {"turns": [4, 10], "preamble_tokens": [50, 300]}
    a, b = rs.session_plan(1, 7, ph), rs.session_plan(1, 7, ph)
    assert a == b
    others = [rs.session_plan(s, 7, ph)["preamble"] for s in range(20)]
    assert len(set(others)) == 20 and all(p.startswith(f"Session {s:03d}.") for s, p in enumerate(others))
    assert all(4 <= rs.session_plan(s, 7, ph)["n_turns"] <= 10 for s in range(20))
    assert rs.session_plan(0, 7, ph)["turns"][0]["think_s"] == 0.0


def test_requests_carry_text_history_and_fresh_media_per_turn(tmp_path):
    out, be = _run(tmp_path)
    recs = [json.loads(x) for x in (tmp_path / "raw" / "requests_r4.jsonl").read_text().splitlines()]
    sessions = [json.loads(x) for x in (tmp_path / "raw" / "sessions_r4.jsonl").read_text().splitlines()]
    assert out["sessions"] == 3 and len(sessions) == 3 and out["errors"] == 0
    assert len(recs) == len(be.requests) == sum(s["requests"] for s in sessions)
    for req in be.requests:
        assert req.messages[0]["role"] == "system" and req.messages[0]["content"] == rs.SYSTEM_PROMPT
        assert req.messages[1]["content"].startswith("Session ")
        cur = next(m for m in reversed(req.messages) if m["role"] == "user" and isinstance(m["content"], list))
        kinds = [p["type"] for p in cur["content"]]
        assert kinds == ["text", "image_url", "input_audio"]
        # history_media 0: only the current turn has media
        assert sum(isinstance(m["content"], list) for m in req.messages) == 1
    # audio of every (session, turn) is unique even though only 3 clips exist
    audios = {(r.request_id.rsplit("-", 1)[0]): next(p for m in r.messages if isinstance(m["content"], list)
                                                    for p in m["content"] if p["type"] == "input_audio")["input_audio"]["data"]
              for r in be.requests}
    assert len(set(audios.values())) == len(audios)
    # a later turn sees the earlier transcript and answer as text
    late = [r for r in be.requests if "-t01-" in r.request_id]
    assert late and any("The person said: question" in m["content"] for m in late[0].messages if isinstance(m["content"], str))
    tool_turns = [r for r in recs if r["step"] == "plan"]
    for p in tool_turns:
        ans = next(r for r in recs if r["session"] == p["session"] and r["turn"] == p["turn"] and r["step"] == "answer")
        assert ans["messages"] == p["messages"] + 2  # plan + tool result
    assert all(r["speech"] == (r["step"] == "answer") for r in recs)


def test_history_media_window_resends_earlier_media(tmp_path):
    _, be = _run(tmp_path, history_media=1, sessions=2, turns=[3, 3], tool={"p": 0.0})
    by_turn = {r.request_id: r for r in be.requests}
    t2 = by_turn["r4-s000-t02-answer"]
    media_msgs = [m for m in t2.messages if isinstance(m["content"], list)]
    assert len(media_msgs) == 2  # turn 2 (index 1) re-sent, plus the current turn
    recs = [json.loads(x) for x in (tmp_path / "raw" / "requests_r4.jsonl").read_text().splitlines()]
    rec = next(r for r in recs if r["request_id"] == "r4-s000-t02-answer")
    assert rec["images_in_request"] == 2 and rec["audios_in_request"] == 2


def test_phases_get_their_own_sessions_unless_they_share_a_content_key(tmp_path):
    from omni_research.sessions.frames import content_seed

    assert content_seed(1, "r4_n8") != content_seed(1, "r4_n16") and content_seed(1, "a") == content_seed(1, "a")
    (tmp_path / "a").mkdir()
    _, be1 = _run(tmp_path / "a", sessions=2)
    ctx = {"phase": "other", "seed": 11, "raw_dir": tmp_path / "b" / "raw", "backend_factory": lambda: EchoBackend()}
    pre1 = {r.messages[1]["content"] for r in be1.requests}
    be2 = EchoBackend()
    ctx["backend_factory"] = lambda: be2
    clips = [{"path": _wav(tmp_path / f"z{i}.wav"), "transcript": f"q {i}"} for i in range(3)]
    base = {"sessions": 2, "turns": [2, 2], "ramp_s": 0.01, "think_s": [0.001, 0.1], "tool": {"p": 0.0}, "audio_clips": clips,
            "width": 64, "height": 36}
    import asyncio as _a
    _a.run(rs.run_phase(ctx, base))
    assert not pre1 & {r.messages[1]["content"] for r in be2.requests}  # another phase: other preambles
    be3 = EchoBackend()
    ctx.update(phase="other2", raw_dir=tmp_path / "c" / "raw", backend_factory=lambda: be3)
    _a.run(rs.run_phase(ctx, base | {"content_key": "other"}))
    assert {r.messages[1]["content"] for r in be3.requests} == {r.messages[1]["content"] for r in be2.requests}
