"""Serving backends used by the replay engine and benchmark drivers.

``OpenAIChatBackend`` talks to an SGLang-Omni (or any OpenAI-compatible) server over
``/v1/chat/completions`` with SSE streaming and records client-observed milestones.
``MockBackend`` simulates a server in-process so the replay/benchmark machinery can be tested
without a GPU; its outputs are labelled ``backend=mock`` everywhere and are never results.
``BackendPool`` routes requests across several servers (independent replicas).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import random
import struct
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

# --------------------------------------------------------------------------------------------
# Request / result records


@dataclass
class ModelRequest:
    request_id: str
    messages: list[dict[str, Any]]
    input_modalities: list[str]
    output_modalities: list[str]
    max_tokens: int | None = None
    audios: list[str] | None = None
    images: list[str] | None = None
    videos: list[str] | None = None
    stream: bool = True
    temperature: float | None = 0.0
    seed: int | None = 0
    extra_body: dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 300.0
    metadata: dict[str, Any] = field(default_factory=dict)  # session/step ids, replay_of, sizes

    def body(self, model: str | None) -> dict[str, Any]:
        b: dict[str, Any] = {
            "messages": self.messages,
            "stream": self.stream,
            "modalities": self.output_modalities,
            "request_id": self.request_id,
        }
        if model:
            b["model"] = model
        if self.max_tokens is not None:
            b["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            b["temperature"] = self.temperature
        if self.seed is not None:
            b["seed"] = self.seed
        for key in ("audios", "images", "videos"):
            v = getattr(self, key)
            if v:
                b[key] = v
        if "audio" in self.output_modalities:
            b.setdefault("audio", {"format": "wav"})
        b.update(self.extra_body)
        return b


@dataclass
class RequestResult:
    """Client-observed outcome of one model request. All ``*_ns`` are ``time.monotonic_ns``."""

    request_id: str
    backend: str
    server: str | None = None
    status: str = "ok"  # ok | error | timeout | cancelled
    error: str | None = None
    http_status: int | None = None
    sent_ns: int | None = None
    headers_ns: int | None = None
    first_text_ns: int | None = None
    first_audio_ns: int | None = None
    last_audio_ns: int | None = None
    done_ns: int | None = None
    text_chars: int = 0
    text: str | None = None  # kept only when the caller asks for it (public/synthetic inputs)
    text_chunks: int = 0
    audio_chunks: list[tuple[int, float, int]] = field(default_factory=list)  # (arrival_ns, dur_s, bytes)
    output_audio_s: float = 0.0
    audio_sample_rate: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Backend(Protocol):
    name: str

    async def run(self, req: ModelRequest) -> RequestResult: ...

    async def close(self) -> None: ...


# --------------------------------------------------------------------------------------------
# WAV parsing (chunks are complete WAV files; float32 WAVs are not readable by `wave`)


def wav_duration(data: bytes) -> tuple[float, int | None]:
    """Duration in seconds and sample rate of a RIFF/WAVE byte string (PCM or IEEE float)."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return 0.0, None
    pos = 12
    channels = rate = bits = None
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        size = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        body = pos + 8
        if cid == b"fmt " and size >= 16:
            _fmt, channels, rate, _brate, _align, bits = struct.unpack("<HHIIHH", data[body : body + 16])
        elif cid == b"data":
            if not (channels and rate and bits):
                return 0.0, rate
            avail = len(data) - body
            if size == 0xFFFFFFFF or size > avail:  # streaming header with unknown/over-long size
                size = avail
            frame = channels * (bits // 8)
            return (size // frame) / rate if frame else 0.0, rate
        pos = body + size + (size & 1)
    return 0.0, rate


# --------------------------------------------------------------------------------------------
# Real server


class OpenAIChatBackend:
    """SSE client for ``POST {base_url}/v1/chat/completions`` (SGLang-Omni extensions allowed)."""

    def __init__(self, base_url: str, model: str | None = None, keep_text: bool = False, name: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.keep_text = keep_text
        self.name = name or f"openai:{self.base_url}"
        self._session = None

    async def _get_session(self):  # noqa: ANN202
        if self._session is None:
            import aiohttp  # imported lazily so the mock path has no aiohttp dependency

            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=0, force_close=False),
                timeout=aiohttp.ClientTimeout(total=None),
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def run(self, req: ModelRequest) -> RequestResult:
        import aiohttp

        res = RequestResult(request_id=req.request_id, backend="openai", server=self.base_url, metadata=dict(req.metadata))
        session = await self._get_session()
        url = f"{self.base_url}/v1/chat/completions"
        text_parts: list[str] = []
        try:
            res.sent_ns = time.monotonic_ns()
            async with asyncio.timeout(req.timeout_s):
                async with session.post(url, json=req.body(self.model)) as resp:
                    res.headers_ns = time.monotonic_ns()
                    res.http_status = resp.status
                    if resp.status != 200:
                        res.status = "error"
                        res.error = (await resp.text())[:2000]
                        res.done_ns = time.monotonic_ns()
                        return res
                    if req.stream:
                        await self._consume_sse(resp, res, text_parts)
                    else:
                        payload = await resp.json()
                        self._consume_json(payload, res, text_parts)
            res.done_ns = time.monotonic_ns()
        except TimeoutError:
            res.status, res.error, res.done_ns = "timeout", f"no completion within {req.timeout_s}s", time.monotonic_ns()
        except asyncio.CancelledError:
            res.status, res.error, res.done_ns = "cancelled", "cancelled by client", time.monotonic_ns()
            raise
        except (aiohttp.ClientError, ConnectionError, OSError) as exc:
            res.status, res.error, res.done_ns = "error", f"{type(exc).__name__}: {exc}"[:2000], time.monotonic_ns()
        finally:
            res.text_chars = sum(len(t) for t in text_parts)
            if self.keep_text:
                res.text = "".join(text_parts)
        return res

    async def _consume_sse(self, resp, res: RequestResult, text_parts: list[str]) -> None:  # noqa: ANN001
        buf = b""
        async for raw in resp.content.iter_any():
            buf += raw
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    return
                now = time.monotonic_ns()
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                self._apply_chunk(obj, now, res, text_parts)

    def _apply_chunk(self, obj: dict[str, Any], now: int, res: RequestResult, text_parts: list[str]) -> None:
        if "error" in obj:
            err = obj["error"]
            res.status = "error"
            res.error = (err.get("message") if isinstance(err, dict) else str(err)) or "stream error"
            return
        usage = obj.get("usage")
        if usage:
            res.prompt_tokens = usage.get("prompt_tokens")
            res.completion_tokens = usage.get("completion_tokens")
        for ch in obj.get("choices", []):
            if ch.get("finish_reason"):
                res.finish_reason = ch["finish_reason"]
            delta = ch.get("delta") or {}
            content = delta.get("content")
            if content:
                if res.first_text_ns is None:
                    res.first_text_ns = now
                res.text_chunks += 1
                text_parts.append(content)
            audio = delta.get("audio")
            if audio and audio.get("data"):
                self._add_audio(audio["data"], now, res)

    def _consume_json(self, payload: dict[str, Any], res: RequestResult, text_parts: list[str]) -> None:
        now = time.monotonic_ns()
        usage = payload.get("usage") or {}
        res.prompt_tokens = usage.get("prompt_tokens")
        res.completion_tokens = usage.get("completion_tokens")
        for ch in payload.get("choices", []):
            res.finish_reason = ch.get("finish_reason")
            msg = ch.get("message") or {}
            if msg.get("content"):
                res.first_text_ns = now
                text_parts.append(msg["content"])
            audio = msg.get("audio")
            if audio and audio.get("data"):
                self._add_audio(audio["data"], now, res)

    @staticmethod
    def _add_audio(b64: str, now: int, res: RequestResult) -> None:
        raw = base64.b64decode(b64)
        dur, rate = wav_duration(raw)
        if res.first_audio_ns is None:
            res.first_audio_ns = now
        res.last_audio_ns = now
        res.audio_chunks.append((now, dur, len(raw)))
        res.output_audio_s += dur
        res.audio_sample_rate = rate or res.audio_sample_rate


# --------------------------------------------------------------------------------------------
# Mock server (tests only)


@dataclass
class MockServiceModel:
    """Toy processor-sharing service model. Parameters are arbitrary; this is not a model of
    any real system and its numbers must never be reported as measurements."""

    prefill_s: float = 0.01
    per_token_s: float = 0.002
    contention: float = 0.15  # decode step slows by this fraction per extra concurrent request
    output_tokens: int = 16
    audio_s_per_token: float = 0.05
    audio_chunk_tokens: int = 4
    max_concurrency: int | None = None
    error_rate: float = 0.0
    seed: int = 0


class MockBackend:
    name = "mock"

    def __init__(self, model: MockServiceModel | None = None) -> None:
        self.model = model or MockServiceModel()
        self._active = 0
        self._sem = asyncio.Semaphore(self.model.max_concurrency) if self.model.max_concurrency else None
        self._rng = random.Random(self.model.seed)
        self.max_active_seen = 0

    async def close(self) -> None:
        return None

    async def run(self, req: ModelRequest) -> RequestResult:
        res = RequestResult(request_id=req.request_id, backend="mock", server="mock", metadata=dict(req.metadata))
        res.sent_ns = time.monotonic_ns()
        try:
            async with asyncio.timeout(req.timeout_s):
                if self._sem:
                    async with self._sem:
                        await self._serve(req, res)
                else:
                    await self._serve(req, res)
        except TimeoutError:
            res.status, res.error = "timeout", f"no completion within {req.timeout_s}s"
        res.done_ns = time.monotonic_ns()
        return res

    async def _serve(self, req: ModelRequest, res: RequestResult) -> None:
        m = self.model
        self._active += 1
        self.max_active_seen = max(self.max_active_seen, self._active)
        try:
            res.headers_ns = time.monotonic_ns()
            await asyncio.sleep(m.prefill_s)
            if self._rng.random() < m.error_rate:
                res.status, res.error = "error", "mock injected failure"
                return
            n = req.max_tokens or m.output_tokens
            n = min(n, m.output_tokens)
            want_audio = "audio" in req.output_modalities
            for i in range(n):
                await asyncio.sleep(m.per_token_s * (1 + m.contention * (self._active - 1)))
                now = time.monotonic_ns()
                if res.first_text_ns is None:
                    res.first_text_ns = now
                res.text_chunks += 1
                res.text_chars += 4
                if want_audio and (i + 1) % m.audio_chunk_tokens == 0:
                    dur = m.audio_chunk_tokens * m.audio_s_per_token
                    if res.first_audio_ns is None:
                        res.first_audio_ns = now
                    res.last_audio_ns = now
                    res.audio_chunks.append((now, dur, int(dur * 48000)))
                    res.output_audio_s += dur
            res.prompt_tokens = 8
            res.completion_tokens = n
            res.finish_reason = "length" if req.max_tokens and n >= req.max_tokens else "stop"
        finally:
            self._active -= 1


# --------------------------------------------------------------------------------------------
# Routing across replicas


class BackendPool:
    """Route requests across backends (independent server replicas).

    policies: ``round_robin``; ``least_outstanding`` (fewest in-flight requests, ties by index);
    ``session_affinity`` (hash of metadata.session_id, so one agent stays on one replica).
    """

    def __init__(self, backends: list[Backend], policy: str = "least_outstanding") -> None:
        if not backends:
            raise ValueError("BackendPool needs at least one backend")
        if policy not in ("round_robin", "least_outstanding", "session_affinity"):
            raise ValueError(f"unknown routing policy {policy!r}")
        self.backends = backends
        self.policy = policy
        self.name = f"pool[{policy}]({','.join(b.name for b in backends)})"
        self._inflight = [0] * len(backends)
        self._rr = 0
        self.routed = [0] * len(backends)

    def _pick(self, req: ModelRequest) -> int:
        if self.policy == "round_robin":
            i = self._rr % len(self.backends)
            self._rr += 1
            return i
        if self.policy == "session_affinity" and req.metadata.get("session_id"):
            import zlib

            return zlib.crc32(str(req.metadata["session_id"]).encode()) % len(self.backends)
        return min(range(len(self.backends)), key=lambda i: (self._inflight[i], i))

    async def run(self, req: ModelRequest) -> RequestResult:
        i = self._pick(req)
        self._inflight[i] += 1
        self.routed[i] += 1
        try:
            res = await self.backends[i].run(req)
        finally:
            self._inflight[i] -= 1
        res.metadata["replica"] = i
        return res

    async def close(self) -> None:
        for b in self.backends:
            await b.close()


def silence_wav(seconds: float, rate: int = 16000) -> bytes:
    """A PCM16 mono WAV of silence (used by tests and as a last-resort payload)."""
    n = int(seconds * rate)
    data = b"\x00\x00" * n
    out = io.BytesIO()
    out.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE")
    out.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16))
    out.write(b"data" + struct.pack("<I", len(data)) + data)
    return out.getvalue()
