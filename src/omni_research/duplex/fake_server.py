"""CPU-only stand-in for a MiniCPM-o full-duplex server (``--variant session --enable-realtime``), for tests.

It speaks the same wire protocol as sglang-omni 520b49ee's /v1/realtime (see client.py): 1 s units cut
from the appended 16 kHz PCM16, each unit's output events released together after the unit is processed
(like the CoordinatorAdapter), ``sglang.unit.done`` for every unit, HTTP 503 when ``max_sessions``
connections are open, ``sglang.input_audio.drained`` after the padded last unit, and an error
``not_supported`` for event types the schema does not know (such as ``response.cancel``).

Its "model" is a toy: after a unit with speech followed by a silent unit it speaks for ``reply_units``
units (1 s of a 220 Hz tone per unit). Its timings are made up; nothing it produces may be reported as a
measurement.

    python -m omni_research.duplex.fake_server --port 30000 [--max-sessions 2] [--unit-delay-s 0.1]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

UNIT_SAMPLES = 16000
OUT_RATE = 24000
KNOWN_TYPES = {"session.update", "input_audio_buffer.append", "sglang.input_image.append", "input_audio_buffer.clear",
               "sglang.input_audio.end", "session.close", "input_audio_buffer.commit", "response.create"}


def tone(seconds: float, rate: int = OUT_RATE, hz: float = 220.0, amp: float = 3000.0) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * math.pi * hz * t)).astype("<i2").tobytes()


def capabilities(max_sessions: int) -> dict[str, Any]:
    return {"model": "fake-duplex", "interaction": "native", "native_full_duplex": True, "native_unit_ms": 1000,
            "first_unit_ms": 1000, "input_audio_format": {"type": "audio/pcm", "rate": 16000},
            "output_audio_format": {"type": "audio/pcm", "rate": OUT_RATE}, "output_modalities": ["audio", "text"],
            "input_modalities": ["audio", "image"], "supports_server_interrupt": False, "tail_policy": "pad",
            "limits": {"max_input_bytes": 1920000}, "fake": True, "max_sessions": max_sessions}


def build_app(max_sessions: int = 2, unit_delay_s: float = 0.1, reply_units: int = 3,
              fatal_after_units: int | None = None, reject_after_packets: int | None = None):
    from aiohttp import WSMsgType, web

    state: dict[str, Any] = {"open": 0, "recorder": None}

    def record(event: str, **meta: Any) -> None:
        rec = state["recorder"]
        if rec is None:
            return
        with open(rec, "a") as fh:
            fh.write(json.dumps({"request_id": meta.pop("request_id", "x"), "stage": "fake", "event_name": event,
                                 "timestamp_ns": time.time_ns(), "metadata": meta}) + "\n")

    async def health(_req):
        return web.json_response({"status": "healthy"})

    async def caps(_req):
        return web.json_response(capabilities(max_sessions))

    async def start_profile(req):
        body = await req.json()
        d = Path(body["event_dir"])
        d.mkdir(parents=True, exist_ok=True)
        state["recorder"] = str(d / "events_fake.jsonl")
        return web.json_response({"run_id": body.get("run_id"), "event_dir": str(d)})

    async def stop_profile(_req):
        state["recorder"] = None
        return web.json_response({"run_id": None})

    async def realtime(req):
        if state["open"] >= max_sessions:
            return web.json_response({"error": {"code": "unavailable", "message": "connection capacity exhausted"}},
                                     status=503)
        state["open"] += 1
        ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
        await ws.prepare(req)
        sid = "sess_" + uuid.uuid4().hex[:12]
        units: asyncio.Queue = asyncio.Queue()
        st = {"pending": bytearray(), "accepted": 0, "index": 0, "ended": False, "open": False, "speech_seen": False,
              "speaking_left": 0, "reply": None, "serial": 0, "end_event": None, "closed": False}

        async def send(ev: dict[str, Any]) -> None:
            ev["event_id"] = "evt_" + uuid.uuid4().hex
            await ws.send_str(json.dumps(ev))

        def media(index: int, real: int) -> dict[str, Any]:
            return {"unit_id": f"unit_{index}", "chunk_seq": 0,
                    "media_time": {"t_start_ms": index * 1000.0, "duration_ms": real / 16.0}}

        async def process_units() -> None:
            while True:
                item = await units.get()
                if item is None:
                    return
                index, pcm, real, eos = item
                await asyncio.sleep(unit_delay_s)
                record("session_unit_finished", session_id=sid, input_seq=index)
                samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
                level = float(np.sqrt((samples ** 2).mean())) if samples.size else 0.0
                speech = level > 100.0
                events: list[dict[str, Any]] = []
                if fatal_after_units is not None and index >= fatal_after_units:
                    await send({"type": "error", "sglang": {"fatal": True},
                                "error": {"type": "server_error", "code": "internal", "message": "fake failure"}})
                    await send({"type": "session.closed", "reason": "internal"})
                    st["closed"] = True
                    await ws.close()
                    return
                if speech:
                    st["speech_seen"] = True
                elif st["speech_seen"] and st["speaking_left"] == 0 and st["reply"] is None:
                    st["speaking_left"] = reply_units
                    st["speech_seen"] = False
                if st["speaking_left"] > 0:
                    meta = media(index, real)
                    if st["reply"] is None:
                        st["reply"] = f"{sid}-response-{st['serial']}"
                        st["serial"] += 1
                        events.append({"type": "response.created", "response": {"id": st["reply"], "status": "in_progress"},
                                       "sglang": meta})
                    rid = st["reply"]
                    events.append({"type": "response.output_audio_transcript.delta", "response_id": rid, "delta": "word ",
                                   "sglang": meta})
                    events.append({"type": "response.output_audio.delta", "response_id": rid,
                                   "delta": base64.b64encode(tone(1.0)).decode("ascii"), "sglang": meta})
                    st["speaking_left"] -= 1
                    if st["speaking_left"] == 0:
                        events += [{"type": "response.output_audio.done", "response_id": rid, "sglang": meta},
                                   {"type": "response.done", "response": {"id": rid, "status": "completed"}, "sglang": meta}]
                        st["reply"] = None
                for ev in events:
                    await send(ev)
                await send({"type": "sglang.unit.done", "unit_id": f"unit_{index}", "sglang": media(index, real)})
                if eos:
                    await send({"type": "sglang.input_audio.drained", "accepted_end_ms": st["accepted"] / 16.0,
                                "consumed_ms": st["accepted"] / 16.0, "discarded_ms": 0.0,
                                "padding_ms": (UNIT_SAMPLES - real) / 16.0 if 0 < real < UNIT_SAMPLES else 0.0,
                                "client_event_id": st["end_event"]})
                    return

        worker = asyncio.create_task(process_units())
        await send({"type": "session.created", "session": {"id": sid, "object": "realtime.session"}})
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                ev = json.loads(msg.data)
                typ, eid = ev.get("type"), ev.get("event_id")
                if typ not in KNOWN_TYPES:
                    await send({"type": "error", "sglang": {"fatal": False},
                                "error": {"type": "invalid_request_error", "code": "not_supported",
                                          "message": f"unknown event {typ}", "event_id": eid}})
                elif typ == "session.update":
                    st["open"] = True
                    await send({"type": "session.updated", "client_event_id": eid,
                                "session": {**ev.get("session", {}), "id": sid, "sglang": {"granted": capabilities(max_sessions)}}})
                elif typ == "input_audio_buffer.append" and reject_after_packets is not None \
                        and ev["sglang"]["seq"] >= reject_after_packets:
                    await send({"type": "error", "sglang": {"fatal": False},
                                "error": {"type": "invalid_request_error", "code": "buffer_overflow",
                                          "message": "input budget exhausted; retry this seq", "event_id": eid}})
                elif typ == "input_audio_buffer.append":
                    pcm = base64.b64decode(ev["audio"])
                    st["pending"].extend(pcm)
                    st["accepted"] += len(pcm) // 2
                    await send({"type": "sglang.input_audio.accepted", "seq": ev["sglang"]["seq"],
                                "accepted_end_ms": st["accepted"] / 16.0, "client_event_id": eid})
                    while len(st["pending"]) >= 2 * UNIT_SAMPLES:
                        unit = bytes(st["pending"][:2 * UNIT_SAMPLES])
                        del st["pending"][:2 * UNIT_SAMPLES]
                        record("session_unit_ready", session_id=sid, input_seq=st["index"])
                        await units.put((st["index"], unit, UNIT_SAMPLES, False))
                        st["index"] += 1
                elif typ == "sglang.input_image.append":
                    await send({"type": "sglang.input_image.accepted",
                                "unit_id": f"unit_{int(ev['sglang']['t_ms'] // 1000)}", "client_event_id": eid})
                elif typ == "sglang.input_audio.end":
                    st["end_event"] = eid
                    await send({"type": "sglang.input_audio.ended", "accepted_end_ms": st["accepted"] / 16.0,
                                "tail_policy": "pad", "client_event_id": eid})
                    # Like SessionRuntime.cut_next_unit: a partial unit is padded with silence; when the input
                    # ended on a unit boundary the closing unit is empty (duration 0).
                    real = len(st["pending"]) // 2
                    pcm = bytes(st["pending"]) + b"\0" * (2 * UNIT_SAMPLES - len(st["pending"])) if real else b""
                    st["pending"].clear()
                    await units.put((st["index"], pcm, real, True))
                    st["index"] += 1
                elif typ == "session.close":
                    await units.put(None)
                    await asyncio.wait_for(worker, 30)
                    if not st["closed"]:
                        await send({"type": "session.closed", "reason": "client_closed", "client_event_id": eid})
                    await ws.close()
                    break
                else:
                    await send({"type": "error", "sglang": {"fatal": False},
                                "error": {"type": "invalid_request_error", "code": "not_applicable",
                                          "message": "manual turn commands are not granted", "event_id": eid}})
        finally:
            worker.cancel()
            state["open"] -= 1
        return ws

    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_get("/v1/realtime/capabilities", caps)
    app.router.add_get("/v1/realtime", realtime)
    app.router.add_post("/start_request_profile", start_profile)
    app.router.add_post("/stop_request_profile", stop_profile)
    return app


def main() -> int:
    from aiohttp import web

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-sessions", type=int, default=2)
    ap.add_argument("--unit-delay-s", type=float, default=0.1)
    ap.add_argument("--reply-units", type=int, default=3)
    a = ap.parse_args()
    web.run_app(build_app(a.max_sessions, a.unit_delay_s, a.reply_units), host=a.host, port=a.port, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
