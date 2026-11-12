"""CPU-only stand-in for an SGLang-Omni server, used to test the experiment runner end to end.

It implements the endpoints the runner uses, with the same wire formats:
``GET /health`` (503 "starting" for ``--startup-s``, then 200 "healthy"),
``POST /v1/chat/completions`` (SSE text deltas and base64 WAV audio chunks, final usage),
``POST /v1/audio/speech`` (a WAV of silence whose length grows with the text),
``POST /start_request_profile`` / ``/stop_request_profile`` (writes a tiny event JSONL),
``POST /start_profile`` / ``/stop_profile`` (also writes a one-kernel trace, asynchronously after stop).

Its timings are made up. Nothing it produces may be reported as a measurement.

    python -m omni_research.replay.fake_server --port 30000 [--startup-s 2] [--token-ms 5]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import time
from pathlib import Path

from .backends import silence_wav


def build_app(startup_s: float, token_ms: float, agent_json: bool, die_after: int | None = None,  # noqa: ANN201
              die_marker: str | None = None):
    from aiohttp import web

    t_ready = time.monotonic() + startup_s
    state = {"recorder": None, "chats": 0}

    def emit(rid: str, event: str) -> None:
        rec = state["recorder"]
        if rec is None:
            return
        with open(rec["path"], "a") as fh:
            fh.write(json.dumps({"request_id": rid, "stage": "fake", "event_name": event, "timestamp_ns": time.time_ns(),
                                 "run_id": rec["run_id"], "pid": os.getpid(), "metadata": {}}) + "\n")

    async def health(_req):  # noqa: ANN001
        ok = time.monotonic() >= t_ready
        return web.json_response({"status": "healthy" if ok else "starting", "running": ok}, status=200 if ok else 503)

    async def chat(request):  # noqa: ANN001
        state["chats"] += 1
        if die_after is not None and state["chats"] > die_after and not (die_marker and os.path.exists(die_marker)):
            # Like a stage process dying under load (f2, 2026-11-11): the whole server goes away mid-run.
            # With die_marker the server dies only once: a restarted server finds the marker and keeps going.
            print(f"fake server: exiting after {die_after} chat requests", flush=True)
            if die_marker:
                Path(die_marker).write_text("died\n")
            os._exit(134)
        body = await request.json()
        rid = body.get("request_id") or "anon"
        emit(rid, "request_admission")
        n = min(int(body.get("max_tokens") or 16), 24)
        want_audio = "audio" in (body.get("modalities") or ["text"])
        last_user = next((m.get("content") for m in reversed(body.get("messages", [])) if m.get("role") == "user"), "")
        if agent_json and isinstance(last_user, str) and last_user.startswith("Tool result"):
            pieces = ['{"final": "', "The answer is ", "42", '"}']
        elif agent_json and body.get("messages") and body["messages"][0].get("role") == "system":
            pieces = ['{"tool": "calculator", ', '"args": {"expression": "6*7"}}']
        else:
            pieces = [f"tok{i} " for i in range(n)]
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for i, p in enumerate(pieces):
            await asyncio.sleep(token_ms / 1000)
            if i == 0:
                emit(rid, "stage_first_stream_chunk_sent")
            await resp.write(f"data: {json.dumps({'id': rid, 'choices': [{'delta': {'content': p}}]})}\n\n".encode())
            if want_audio and i % 4 == 3:
                b64 = base64.b64encode(silence_wav(0.2, 24000)).decode()
                await resp.write(f"data: {json.dumps({'id': rid, 'choices': [{'delta': {'audio': {'id': rid, 'data': b64}}}]})}\n\n".encode())
        usage = {"prompt_tokens": 10, "completion_tokens": len(pieces), "total_tokens": 10 + len(pieces)}
        await resp.write(f"data: {json.dumps({'id': rid, 'choices': [{'delta': {}, 'finish_reason': 'stop'}], 'usage': usage})}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        emit(rid, "terminal_response")
        return resp

    async def speech(request):  # noqa: ANN001
        body = await request.json()
        secs = 0.5 + len(body.get("input", "")) / 60
        return web.Response(body=silence_wav(secs, 24000), content_type="audio/wav")

    async def start_profile(request):  # noqa: ANN001
        return await start_profile_impl(await request.json())

    async def start_profile_impl(body):  # noqa: ANN001
        d = Path(body.get("event_dir", "/tmp/fake-events"))
        d.mkdir(parents=True, exist_ok=True)
        state["recorder"] = {"run_id": body.get("run_id", "run"), "path": d / f"events_fake_{os.getpid()}.jsonl"}
        return web.json_response({"ok": True})

    async def stop_profile(_request):  # noqa: ANN001
        state["recorder"] = None
        tpl = state.pop("torch_template", None)
        if tpl:
            # Mimic SGLang-Omni: each stage process exports its trace asynchronously after stop.
            async def export() -> None:
                import gzip

                await asyncio.sleep(1.0)
                path = Path(tpl.format(stage="fake", run_id="run") + f"_pid{os.getpid()}_rank0.trace.json.gz")
                path.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(path, "wt") as fh:
                    json.dump({"traceEvents": [{"ph": "X", "cat": "kernel", "name": "fake_kernel", "ts": 0, "dur": 10,
                                                "args": {"device": 0}}]}, fh)
            asyncio.get_running_loop().create_task(export())
        return web.json_response({"ok": True})

    async def start_torch_profile(request):  # noqa: ANN001
        body = await request.json()
        state["torch_template"] = body.get("trace_path_template")
        return await start_profile_impl(body)

    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/v1/audio/speech", speech)
    app.router.add_post("/start_request_profile", start_profile)
    app.router.add_post("/stop_request_profile", stop_profile)
    app.router.add_post("/start_profile", start_torch_profile)
    app.router.add_post("/stop_profile", stop_profile)
    return app


def main() -> None:
    from aiohttp import web

    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--startup-s", type=float, default=1.0)
    p.add_argument("--token-ms", type=float, default=5.0)
    p.add_argument("--agent-json", action="store_true", help="answer the agent protocol (tool call, then final)")
    p.add_argument("--die-after", type=int, default=None, help="exit abruptly when this many chat requests have been served")
    p.add_argument("--die-marker", default=None, help="die only if this file is absent, and create it when dying (die once across restarts)")
    a = p.parse_args()
    web.run_app(build_app(a.startup_s, a.token_ms, a.agent_json, a.die_after, a.die_marker), host="127.0.0.1", port=a.port, print=None)


if __name__ == "__main__":
    main()
