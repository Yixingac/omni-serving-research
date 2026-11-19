"""One full-duplex session over ``/v1/realtime`` (aiohttp WebSocket), paced in real time, fully recorded.

Protocol (sglang-omni 520b49ee, sglang_omni/serve/realtime/protocol.py and schema.py):
  server ``session.created`` -> client ``session.update`` -> server ``session.updated``;
  client ``input_audio_buffer.append`` {audio: base64 PCM16 16 kHz, sglang: {seq, t_start_ms}} at the
  capture pace (80 ms packets here), each acknowledged by ``sglang.input_audio.accepted``;
  optional ``sglang.input_image.append`` {image: base64 JPEG/PNG, sglang: {t_ms}};
  the server answers per 1 s unit with ``response.created`` / ``response.output_audio.delta`` (base64
  PCM16 24 kHz) / ``response.output_audio_transcript.delta`` / ``response.done`` and always
  ``sglang.unit.done`` (each carrying sglang.unit_id and media_time);
  client ``sglang.input_audio.end`` -> ``sglang.input_audio.ended`` ... ``sglang.input_audio.drained``;
  client ``session.close`` -> ``session.closed``. A refused connection (all ``max_sessions`` busy) is an
  HTTP 503 before the upgrade. There is no ``response.cancel`` at this commit.

Every record has ``t_mono_ns`` (this process's CLOCK_MONOTONIC) and ``t_wall_ns`` (time.time_ns()). The
client and the server run on the same node, so wall times line up with the server's request-event
timestamps (also time.time_ns()); monotonic times are only compared with monotonic times of this
process. Audio payloads are not stored in the trace (only their size and level); ``audio_path`` keeps the
received PCM of chosen sessions.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audio import rms_dbfs
from .script import PACKET_MS, SessionScript

OUTPUT_RATE = 24000
MAX_MESSAGE_BYTES = 16 * 1024 * 1024
DENIED_STATUS = 503


@dataclass
class SessionResult:
    session_index: int
    status: str = "not_started"  # ok | failed | timeout | denied | error
    reason: str = ""
    packets_sent: int = 0
    packets_planned: int = 0
    events_received: int = 0
    audio_bytes: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    closed_reason: str | None = None
    admission_attempts: int = 0

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _now() -> tuple[int, int]:
    return time.monotonic_ns(), time.time_ns()


def _strip(event: dict[str, Any]) -> dict[str, Any]:
    """The event without bulky payloads: audio deltas become their size and level."""
    out = dict(event)
    if out.get("type") == "response.output_audio.delta" and isinstance(out.get("delta"), str):
        pcm = base64.b64decode(out.pop("delta"))
        out["delta_bytes"] = len(pcm)
        out["delta_dbfs"] = round(rms_dbfs(pcm), 2)
    if out.get("type") == "session.updated" and isinstance(out.get("session"), dict):
        s = dict(out["session"])
        s.pop("instructions", None)
        out["session"] = s
    return out


class Trace:
    def __init__(self, path: Path) -> None:
        self.fh = path.open("w", buffering=1)

    def write(self, rec: dict[str, Any]) -> None:
        self.fh.write(json.dumps(rec, allow_nan=False) + "\n")

    def close(self) -> None:
        self.fh.close()


async def run_session(url: str, script: SessionScript, *, trace_path: Path, session_update: dict[str, Any],
                      frames: list[tuple[float, bytes]] | None = None, audio_path: Path | None = None,
                      pace_scale: float = 1.0, start_delay_s: float = 0.0, connect_retries: int = 20,
                      connect_backoff_s: float = 0.5, update_timeout_s: float = 90.0, drain_timeout_s: float = 180.0,
                      close_timeout_s: float = 60.0) -> SessionResult:
    """Run one session; never raises for server or network trouble (it is recorded instead)."""
    import aiohttp

    res = SessionResult(script.session_index)
    packets = script.packets()
    res.packets_planned = len(packets)
    frames = sorted(frames or [])
    trace = Trace(trace_path)
    pcm_out = bytearray()
    m0, w0 = _now()
    trace.write({"d": "meta", "what": "anchor", "t_mono_ns": m0, "t_wall_ns": w0, "session_index": script.session_index,
                 "url": url, "pace_scale": pace_scale, "packets": len(packets), "packet_ms": PACKET_MS,
                 "pcm_sha256": script.sha256()})
    seen: dict[str, asyncio.Event] = {}
    fatal = asyncio.Event()
    # A refused append (buffer_overflow: the server is more than 60 s behind; invalid_state: a gap in seq) makes
    # every later append fail too, so the session stops sending and ends instead of piling up errors.
    rejected: dict[str, str] = {}

    def flag(name: str) -> asyncio.Event:
        return seen.setdefault(name, asyncio.Event())

    async def wait_for(name: str, timeout: float) -> bool:
        waits = [asyncio.create_task(flag(name).wait()), asyncio.create_task(fatal.wait())]
        try:
            await asyncio.wait(waits, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in waits:
                t.cancel()
        return flag(name).is_set()

    async def send(ws: Any, event: dict[str, Any], **extra: Any) -> None:
        event.setdefault("event_id", uuid.uuid4().hex)
        m1, w1 = _now()
        await ws.send_str(json.dumps(event))
        m2 = time.monotonic_ns()
        rec = {"d": "send", "t_mono_ns": m1, "t_done_mono_ns": m2, "t_wall_ns": w1, "type": event["type"],
               "event_id": event["event_id"], **extra}
        if event["type"] == "input_audio_buffer.append":
            rec.update(seq=event["sglang"]["seq"], t_start_ms=event["sglang"]["t_start_ms"])
        trace.write(rec)

    async def receive(ws: Any) -> None:
        async for msg in ws:
            m, w = _now()
            if msg.type != aiohttp.WSMsgType.TEXT:
                trace.write({"d": "recv_other", "t_mono_ns": m, "t_wall_ns": w, "msg_type": str(msg.type)})
                if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
                continue
            try:
                event = json.loads(msg.data)
            except ValueError:
                trace.write({"d": "error", "t_mono_ns": m, "t_wall_ns": w, "message": "malformed frame",
                             "excerpt": msg.data[:200]})
                continue
            typ = event.get("type", "?")
            res.events_received += 1
            if typ == "response.output_audio.delta" and isinstance(event.get("delta"), str):
                pcm = base64.b64decode(event["delta"])
                res.audio_bytes += len(pcm)
                if audio_path is not None:
                    pcm_out.extend(pcm)
            trace.write({"d": "recv", "t_mono_ns": m, "t_wall_ns": w, "type": typ, "ev": _strip(event)})
            if typ == "error":
                err = event.get("error") or {}
                is_fatal = bool((event.get("sglang") or {}).get("fatal"))
                res.errors.append({"code": err.get("code"), "message": str(err.get("message"))[:300],
                                   "fatal": is_fatal, "event_id": err.get("event_id")})
                if is_fatal:
                    fatal.set()
                elif err.get("code") in ("buffer_overflow", "invalid_state", "invalid_request") and not rejected:
                    rejected["code"] = str(err.get("code"))
            if typ == "session.closed":
                res.closed_reason = event.get("reason")
            flag(typ).set()

    async def connect(session: Any) -> Any:
        while True:
            res.admission_attempts += 1
            try:
                return await session.ws_connect(url, max_msg_size=MAX_MESSAGE_BYTES, autoping=True, heartbeat=None,
                                                compress=0)
            except aiohttp.WSServerHandshakeError as exc:
                m, w = _now()
                trace.write({"d": "admission", "t_mono_ns": m, "t_wall_ns": w, "status": exc.status,
                             "attempt": res.admission_attempts})
                if exc.status != DENIED_STATUS or res.admission_attempts > connect_retries:
                    raise
                await asyncio.sleep(connect_backoff_s)

    async def drive(ws: Any) -> None:
        if not await wait_for("session.created", update_timeout_s):
            res.status, res.reason = "failed", "no session.created"
            return
        await send(ws, {"type": "session.update", "session": session_update})
        if not await wait_for("session.updated", update_timeout_s):
            res.status, res.reason = "failed", "no session.updated" if not fatal.is_set() else "fatal error before open"
            return
        t0 = time.monotonic()
        fi = 0
        for seq, t_start_ms, chunk in packets:
            due = t0 + seq * PACKET_MS / 1000.0 * pace_scale
            await asyncio.sleep(max(0.0, due - time.monotonic()))
            if fatal.is_set():
                res.status, res.reason = "failed", "fatal error while streaming"
                return
            if rejected:
                res.status, res.reason = "failed", f"input rejected ({rejected['code']}) at packet {seq}"
                break
            # Frames go out just before the packet whose time span holds them (never ahead of the audio).
            while fi < len(frames) and frames[fi][0] < t_start_ms + PACKET_MS:
                t_ms, image = frames[fi]
                await send(ws, {"type": "sglang.input_image.append", "image": base64.b64encode(image).decode("ascii"),
                                "sglang": {"t_ms": float(t_ms)}}, t_ms=float(t_ms), image_bytes=len(image))
                fi += 1
            lateness_ms = (time.monotonic() - due) * 1000.0
            await send(ws, {"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode("ascii"),
                            "sglang": {"seq": seq, "t_start_ms": t_start_ms}}, bytes=len(chunk),
                       lateness_ms=round(lateness_ms, 3))
            res.packets_sent += 1
        await send(ws, {"type": "sglang.input_audio.end"})
        if not await wait_for("sglang.input_audio.drained", drain_timeout_s) and res.status == "not_started":
            res.status, res.reason = "failed", "not drained" if not fatal.is_set() else "fatal error while draining"
        await send(ws, {"type": "session.close"})
        await wait_for("session.closed", close_timeout_s)
        if res.status == "not_started":
            res.status = "ok" if flag("session.closed").is_set() and not fatal.is_set() else "failed"
            if res.status == "failed" and not res.reason:
                res.reason = "no session.closed" if not fatal.is_set() else "fatal error"

    total_timeout = start_delay_s + len(packets) * PACKET_MS / 1000.0 * pace_scale + drain_timeout_s + close_timeout_s \
        + 2 * update_timeout_s
    try:
        if start_delay_s > 0:
            await asyncio.sleep(start_delay_s)
        async with aiohttp.ClientSession() as session:
            try:
                ws = await connect(session)
            except aiohttp.WSServerHandshakeError as exc:
                res.status, res.reason = "denied", f"HTTP {exc.status} after {res.admission_attempts} attempts"
                return res
            async with ws:
                receiver = asyncio.create_task(receive(ws))
                driver = asyncio.create_task(drive(ws))
                done, _ = await asyncio.wait({receiver, driver}, timeout=total_timeout,
                                             return_when=asyncio.FIRST_COMPLETED)
                if driver in done:
                    driver.result()
                    try:
                        await asyncio.wait_for(asyncio.shield(receiver), 2.0)
                    except TimeoutError:
                        pass
                elif receiver in done:
                    receiver.result()
                    try:
                        await asyncio.wait_for(asyncio.shield(driver), 2.0)
                    except TimeoutError:
                        pass
                    if res.status in ("not_started", "ok") and not flag("session.closed").is_set():
                        res.status, res.reason = "failed", "connection closed without session.closed"
                else:
                    res.status, res.reason = "timeout", f"no end after {total_timeout:.0f} s"
                for t in (receiver, driver):
                    t.cancel()
                await asyncio.gather(receiver, driver, return_exceptions=True)
    except Exception as exc:  # noqa: BLE001 - recorded, the phase decides
        m, w = _now()
        trace.write({"d": "error", "t_mono_ns": m, "t_wall_ns": w, "message": f"{type(exc).__name__}: {exc}"[:500]})
        res.status, res.reason = "error", f"{type(exc).__name__}: {exc}"[:300]
    finally:
        if res.status == "not_started":
            res.status, res.reason = "failed", res.reason or "ended early"
        m, w = _now()
        trace.write({"d": "meta", "what": "result", "t_mono_ns": m, "t_wall_ns": w, **res.to_json()})
        trace.close()
        if audio_path is not None and pcm_out:
            import numpy as np

            from .audio import write_wav

            write_wav(audio_path, np.frombuffer(bytes(pcm_out), dtype="<i2"), OUTPUT_RATE)
    return res
