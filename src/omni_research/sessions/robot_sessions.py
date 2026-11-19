"""R4: many independent multi-turn robot-assistant sessions over the request-response path.

A session is one robot talking with one person. It opens with a unique preamble (robot id, room, notes of a
seeded random length), so sessions share only the system prompt. Every turn the robot sees a new camera frame
(R3's synthetic camera) and hears a spoken request (a TTS clip of the asset pool, padded with a session- and
turn-specific number of silent samples so no two sessions or turns send byte-identical audio). With
probability ``tool.p`` the turn first asks for a short plan (text), waits a simulated tool latency and adds
the tool result; the final answer is spoken (``speak_final``) or text. Between turns the person thinks
(lognormal gap).

History: earlier turns go back as text (the header and the transcript of what the person said, the robot's
answers, tool results). ``history_media`` = W > 0 also re-sends the images and audio of the last W turns
inline. With W = 0 the prompt of turn k+1 starts with the prompt of turn k up to turn k's media, so a prefix
cache can reuse it; with W > 0 the re-sent media get new cache keys (sglang-omni keys one pad value per
modality on the whole media set of a request), which is the structural-recomputation contrast.

Records: ``raw/requests_<phase>.jsonl`` (one per request, with session/turn/step and the context sizes the
client knows) and ``raw/sessions_<phase>.jsonl`` (one per session). Times are time.monotonic_ns.
"""

from __future__ import annotations

import asyncio
import base64
import math
import random
import time
from pathlib import Path
from typing import Any

from ..replay.backends import ModelRequest
from ..replay.drivers import JsonlSink
from ..replay.payloads import synthetic_text
from .frames import Camera, content_seed, data_url, wav_with_padding

SYSTEM_PROMPT = (
    "You are the voice of a helpful home robot. You see the room through the robot's camera and hear the person "
    "you are helping. Answer briefly and concretely, refer to what is visible when it matters, and say when you "
    "need a tool. Keep every answer under three sentences."
)
ROOMS = ("kitchen", "living room", "workshop", "hallway", "storage room", "lab", "office", "garage")


def lognormal(rng: random.Random, median_s: float, sigma: float) -> float:
    return median_s * math.exp(rng.gauss(0.0, sigma)) if median_s > 0 else 0.0


def session_plan(session: int, seed: int, ph: dict[str, Any]) -> dict[str, Any]:
    """Everything random about one session, drawn up front from (seed, session) so a rerun sends the same load."""
    rng = random.Random(seed * 1_000_003 + session)
    lo, hi = ph.get("turns", [4, 10])
    n_turns = rng.randint(int(lo), int(hi))
    p_lo, p_hi = ph.get("preamble_tokens", [50, 300])
    tool = ph.get("tool", {"p": 0.5, "latency_s": [0.3, 0.6], "result_tokens": [40, 200]})
    think = ph.get("think_s", [2.0, 0.5])
    turns = []
    for k in range(n_turns):
        uses_tool = rng.random() < float(tool.get("p", 0.5))
        r_lo, r_hi = tool.get("result_tokens", [40, 200])
        turns.append({"turn": k, "tool": uses_tool,
                      "tool_wait_s": lognormal(rng, *tool.get("latency_s", [0.3, 0.6])) if uses_tool else 0.0,
                      "tool_result": synthetic_text(rng.randint(int(r_lo), int(r_hi)), rng) if uses_tool else "",
                      "think_s": lognormal(rng, float(think[0]), float(think[1])) if k else 0.0,
                      "clip_index": rng.randrange(1 << 30)})
    preamble = (f"Session {session:03d}. Robot R-{session:03d} is in the {ROOMS[session % len(ROOMS)]}. "
                f"Notes from earlier today: {synthetic_text(rng.randint(int(p_lo), int(p_hi)), rng)}")
    return {"session": session, "n_turns": n_turns, "preamble": preamble, "turns": turns,
            "start_offset_s": rng.uniform(0.0, float(ph.get("ramp_s", 20.0)))}


def user_parts(header: str, image_url: str | None, audio_b64: str | None) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [{"type": "text", "text": header}]
    if image_url:
        parts.append({"type": "image_url", "image_url": {"url": image_url}})
    if audio_b64:
        parts.append({"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}})
    return parts


def history_messages(turns: list[dict[str, Any]], window: int) -> list[dict[str, Any]]:
    """Messages for completed turns: the last ``window`` keep their media inline, older ones are text only."""
    out: list[dict[str, Any]] = []
    for i, t in enumerate(turns):
        keep_media = window > 0 and i >= len(turns) - window
        if keep_media:
            out.append({"role": "user", "content": user_parts(t["header"], t["image_url"], t["audio_b64"])})
        else:
            out.append({"role": "user", "content": f"{t['header']} (The person said: {t['transcript']})"})
        if t.get("plan_text") is not None:
            out.append({"role": "assistant", "content": t["plan_text"]})
            out.append({"role": "user", "content": f"[tool result] {t['tool_result']}"})
        out.append({"role": "assistant", "content": t.get("answer_text") or ""})
    return out


def load_clips(ph: dict[str, Any], runner: Any) -> list[dict[str, Any]]:
    """Spoken requests: explicit ``audio_clips`` [{path, transcript}] or the pool's agent questions."""
    if ph.get("audio_clips"):
        clips = ph["audio_clips"]
    else:
        pool = runner._pool(ph) if runner is not None else None
        assets = pool.of_kind("audio", ph.get("audio_tag", "agent_question")) if pool else []
        clips = [{"path": a.path, "transcript": a.meta.get("transcript") or a.meta.get("question") or ""} for a in assets
                 if "invalid_runaway" not in a.meta.get("tags", [])]
    if not clips:
        raise ValueError("robot sessions need spoken requests (audio_clips or an asset pool with agent questions)")
    return [{**c, "wav": Path(c["path"]).read_bytes()} for c in clips]


async def run_phase(ctx: dict[str, Any], ph: dict[str, Any]) -> dict[str, Any]:
    phase = ctx["phase"]
    seed = content_seed(int(ph.get("seed", ctx.get("seed", 0))), str(ph.get("content_key", phase)))
    runner = ctx.get("runner")
    n_sessions = int(ph["sessions"])
    window = int(ph.get("history_media", 0))
    speak_final = bool(ph.get("speak_final", True))
    use_image, use_audio = bool(ph.get("image", True)), bool(ph.get("audio", True))
    clips = load_clips(ph, runner) if use_audio else []
    plans = [session_plan(s, seed, ph) for s in range(n_sessions)]
    cameras = [Camera(seed=seed, session=s, width=int(ph.get("width", 640)), height=int(ph.get("height", 360)))
               for s in range(n_sessions)]
    max_plan, max_answer = int(ph.get("max_tokens_plan", 32)), int(ph.get("max_tokens_answer", 48))
    timeout_s = float(ph.get("request_timeout_s", 300.0))
    factory = ctx.get("backend_factory") or (lambda: runner.backend(keep_text=True))
    backend = factory()
    raw = Path(ctx["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)
    req_sink = JsonlSink(raw / f"requests_{phase}.jsonl")
    sess_sink = JsonlSink(raw / f"sessions_{phase}.jsonl")
    phase_start_ns = time.monotonic_ns()

    async def one_request(plan: dict[str, Any], turn: dict[str, Any], step: str, messages: list[dict[str, Any]],
                          speech: bool, n_images: int, n_audios: int) -> Any:
        s, k = plan["session"], turn["turn"]
        req = ModelRequest(
            request_id=f"{phase}-s{s:03d}-t{k:02d}-{step}", messages=messages,
            input_modalities=["text"] + (["image"] if n_images else []) + (["audio"] if n_audios else []),
            output_modalities=["text", "audio"] if speech else ["text"],
            max_tokens=max_plan if step == "plan" else max_answer, stream=True, temperature=0.0, seed=None,
            timeout_s=timeout_s, metadata={"payload": {"workload_id": "R4_robot_session"}})
        res = await backend.run(req)
        rec = res.to_dict()
        text = rec.pop("text", None)
        rec.update(phase=phase, kind="robot_session", session=s, turn=k, step=step, n_turns=plan["n_turns"],
                   tool_turn=turn["tool"], speech=speech, messages=len(messages), images_in_request=n_images,
                   audios_in_request=n_audios, history_media=window, sessions=n_sessions, warmup=False,
                   prompt_chars=sum(len(m["content"]) if isinstance(m["content"], str) else
                                    sum(len(p.get("text", "")) for p in m["content"]) for m in messages))
        req_sink.write(rec)
        return res, text

    async def session(plan: dict[str, Any]) -> None:
        s = plan["session"]
        await asyncio.sleep(plan["start_offset_s"])
        started = time.monotonic_ns()
        done_turns: list[dict[str, Any]] = []
        errors = requests = 0
        tool_wait = think = 0.0
        for turn in plan["turns"]:
            if turn["think_s"]:
                think += turn["think_s"]
                await asyncio.sleep(turn["think_s"])
            k = turn["turn"]
            t_s = (time.monotonic_ns() - phase_start_ns) / 1e9
            image_url = data_url(cameras[s].png(int(round(t_s * cameras[s].capture_fps))), "image/png") if use_image else None
            clip = clips[turn["clip_index"] % len(clips)] if use_audio else None
            audio_b64 = base64.b64encode(wav_with_padding(clip["wav"], 1 + 17 * s + 5 * k)).decode("ascii") if clip else None
            # the same header with or without media, so a text-only run measures the media tokens by difference
            header = f"Turn {k + 1}. The camera image of this moment and the person's spoken request:"
            cur = {"header": header, "image_url": image_url, "audio_b64": audio_b64,
                   "transcript": clip["transcript"] if clip else ""}
            base = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": plan["preamble"]},
                    {"role": "assistant", "content": "Understood."}]
            hist = history_messages(done_turns, window)
            media_turns = [t for t in done_turns[len(done_turns) - window:]] if window > 0 else []
            n_img = (1 if image_url else 0) + sum(1 for t in media_turns if t["image_url"])
            n_aud = (1 if audio_b64 else 0) + sum(1 for t in media_turns if t["audio_b64"])
            messages = base + hist + [{"role": "user", "content": user_parts(header, image_url, audio_b64)}]
            if turn["tool"]:
                res, text = await one_request(plan, turn, "plan", messages, False, n_img, n_aud)
                requests += 1
                errors += res.status != "ok"
                cur["plan_text"] = text or ""
                cur["tool_result"] = turn["tool_result"]
                tool_wait += turn["tool_wait_s"]
                await asyncio.sleep(turn["tool_wait_s"])
                messages = messages + [{"role": "assistant", "content": cur["plan_text"]},
                                       {"role": "user", "content": f"[tool result] {turn['tool_result']}"}]
            res, text = await one_request(plan, turn, "answer", messages, speak_final, n_img, n_aud)
            requests += 1
            errors += res.status != "ok"
            cur["answer_text"] = text or ""
            done_turns.append(cur)
        sess_sink.write({"phase": phase, "session": s, "n_turns": plan["n_turns"], "requests": requests, "errors": errors,
                         "started_ns": started, "ended_ns": time.monotonic_ns(), "tool_wait_s": round(tool_wait, 3),
                         "think_s": round(think, 3), "preamble_chars": len(plan["preamble"]), "history_media": window})

    try:
        await asyncio.gather(*(session(p) for p in plans))
    finally:
        await backend.close()
        req_sink.close()
        sess_sink.close()
    return summarize(sess_sink.records, req_sink.records) | {"history_media": window, "speak_final": speak_final}


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def summarize(sessions: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if r.get("status") == "ok" and r.get("first_text_ns") and r.get("sent_ns")]
    ttft = [(r["first_text_ns"] - r["sent_ns"]) / 1e9 for r in ok]
    dur = [(s["ended_ns"] - s["started_ns"]) / 1e9 for s in sessions]
    prompt = [r["prompt_tokens"] for r in records if r.get("prompt_tokens")]
    return {"sessions": len(sessions), "requests": len(records), "errors": sum(1 for r in records if r.get("status") != "ok"),
            "ttft_s": {"p50": _pct(ttft, 0.5), "p95": _pct(ttft, 0.95), "p99": _pct(ttft, 0.99), "n": len(ttft)},
            "session_s": {"p50": _pct(dur, 0.5), "p95": _pct(dur, 0.95), "n": len(dur)},
            "prompt_tokens": {"p50": _pct(prompt, 0.5), "max": max(prompt) if prompt else None, "n": len(prompt)}}
