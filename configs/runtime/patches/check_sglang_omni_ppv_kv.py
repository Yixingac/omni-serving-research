"""CPU-only check of sglang_omni_ppv_kv.patch (R3 preprocessing concurrency, preprocessing timing, R4 KV counters).

    cd <sglang-omni checkout with the patch> && PYTHONPATH=$PWD <runtime python> <this file> [--model-path DIR]

Checks, without a GPU:
1. create_preprocessing_executor takes max_concurrency (default 1 = the serial loop of 520b49ee) and hands it to
   SimpleScheduler; 4 gives four workers.
2. The preprocessor's __call__ emits one exp_preprocess event per call with the phase times and the number of calls in
   flight; phases add up across nested calls of add_preprocess_time; a failing call still decrements the counter.
3. OmniScheduler.exp_kv_log: one exp_kv_prefill line per new request (first chunk only), with cached/extend tokens and
   the eviction counter; one exp_kv_pool line per second at most; nothing for a non-thinker stage.
4. exp_kv_count_evictions counts the tokens and calls of tree_cache.evict and returns its result unchanged.
5. With --model-path: the real MiniCPM-o preprocessor (CPU device) gives identical input_ids and pixel values for
   eight distinct inline images whether the calls run one at a time or four at a time in threads (re-entrancy).
On a checkout without the patch, check 1 fails first.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import inspect
import io
import logging
import threading
import time
from types import SimpleNamespace
from unittest import mock

from sglang_omni.models.minicpm_o import stages
from sglang_omni.models.minicpm_o.components import preprocessor as pre
from sglang_omni.scheduling import omni_scheduler
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


def check_factory() -> None:
    assert inspect.signature(stages.create_preprocessing_executor).parameters["max_concurrency"].default == 1
    with mock.patch.object(stages, "MiniCPMOPreprocessor", lambda *a, **k: (lambda payload: payload)):
        assert stages.create_preprocessing_executor("model").max_concurrency == 1
        assert stages.create_preprocessing_executor("model", max_concurrency=4).max_concurrency == 4


def check_timing_event() -> None:
    events = []
    obj = object.__new__(pre.MiniCPMOPreprocessor)

    async def fake_preprocess(payload):
        start = time.perf_counter_ns()
        await asyncio.sleep(0.02)
        pre.add_preprocess_time("media_load_ns", start)
        pre.add_preprocess_time("media_load_ns", time.perf_counter_ns())
        if payload.request_id == "bad":
            raise RuntimeError("boom")
        return payload

    obj.preprocess = fake_preprocess
    with mock.patch.object(pre, "emit_request_event", lambda **kw: events.append(kw)):
        async def main() -> None:
            await asyncio.gather(*(obj(SimpleNamespace(request_id=f"r{i}")) for i in range(3)))
            try:
                await obj(SimpleNamespace(request_id="bad"))
            except RuntimeError:
                pass
            else:
                raise AssertionError("the error must propagate")

        asyncio.run(main())
    assert [e["event_name"] for e in events] == ["exp_preprocess"] * 4
    assert {e["request_id"] for e in events} == {"r0", "r1", "r2", "bad"}
    md = [e["metadata"] for e in events]
    assert all(m["media_load_ns"] >= 20_000_000 and m["total_ns"] >= m["media_load_ns"] for m in md)
    assert max(m["inflight_at_start"] for m in md[:3]) == 3  # the three gathered calls overlap
    assert pre.PREPROCESS_INFLIGHT[0] == 0 and pre.PREPROCESS_TIMINGS.get() is None
    pre.add_preprocess_time("media_load_ns", 0)  # outside a call: a no-op


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def check_kv_log() -> None:
    sched = object.__new__(OmniScheduler)
    sched.exp_kv_enabled = True
    sched.exp_kv_evicted_tokens = 120
    sched.exp_kv_evict_calls = 2
    sched.exp_kv_seen_rids = {}
    sched.exp_kv_last_pool_log_s = 0.0
    sched.gpu_id = 1
    sched.max_total_num_tokens = 127039
    sched.token_to_kv_pool_allocator = SimpleNamespace(available_size=lambda: 100000)
    sched.tree_cache = SimpleNamespace(evictable_size=lambda: 2000, protected_size=lambda: 500)
    sched.running_batch = SimpleNamespace(reqs=[1, 2])
    sched.waiting_queue = [3]
    req = lambda rid, n, c: SimpleNamespace(rid=rid, origin_input_ids=list(range(n)), prefix_indices=list(range(c)),  # noqa: E731
                                             extend_input_len=n - c)
    batch = SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: True), reqs=[req("a", 400, 120), req("b", 300, 0)])
    cap = _Capture()
    omni_scheduler.logger.addHandler(cap)
    omni_scheduler.logger.setLevel(logging.INFO)
    try:
        with mock.patch.object(omni_scheduler, "_get_active_stage", lambda: "thinker"):
            sched.exp_kv_log(batch)
            sched.exp_kv_log(batch)  # same requests again (a later chunk): no new prefill lines, pool rate-limited
        with mock.patch.object(omni_scheduler, "_get_active_stage", lambda: "talker"):
            sched.exp_kv_last_pool_log_s = 0.0
            sched.exp_kv_log(SimpleNamespace(forward_mode=SimpleNamespace(is_extend=lambda: True), reqs=[req("c", 10, 0)]))
    finally:
        omni_scheduler.logger.removeHandler(cap)
    pre_lines = [x for x in cap.lines if x.startswith("exp_kv_prefill")]
    pool_lines = [x for x in cap.lines if x.startswith("exp_kv_pool")]
    assert len(pre_lines) == 2 and len(pool_lines) == 1, cap.lines
    assert "rid=a input_tokens=400 cached_tokens=120 extend_tokens=280 evicted_tokens_total=120" in pre_lines[0]
    assert "gpu=1 stage=thinker total_tokens=127039 available_tokens=100000 evictable_tokens=2000 protected_tokens=500 " \
           "running_reqs=2 waiting_reqs=1 evicted_tokens_total=120 evict_calls_total=2" in pool_lines[0]


def check_eviction_counter() -> None:
    sched = object.__new__(OmniScheduler)
    sched.exp_kv_evicted_tokens = 0
    sched.exp_kv_evict_calls = 0
    result = SimpleNamespace(num_tokens_evicted=17)
    cache = SimpleNamespace(evict=lambda params: result)
    sched.exp_kv_count_evictions(cache)
    assert cache.evict("p") is result and cache.evict(params="p") is result
    assert sched.exp_kv_evicted_tokens == 34 and sched.exp_kv_evict_calls == 2


def check_reentrant_preprocessor(model_path: str) -> None:
    import torch
    from PIL import Image

    from sglang_omni.proto import OmniRequest, StagePayload

    proc = pre.MiniCPMOPreprocessor(model_path, device=torch.device("cpu"))
    urls = []
    for i in range(8):
        img = Image.new("RGB", (640, 360), (20 * i, 255 - 20 * i, 128))
        img.paste((255, 0, 0), (40 * i, 40, 40 * i + 60, 100))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        urls.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())

    def payload(i: int) -> StagePayload:
        msg = [{"role": "user", "content": [{"type": "text", "text": f"Camera {i}: how many shapes?"},
                                            {"type": "image_url", "image_url": {"url": urls[i]}}]}]
        return StagePayload(request_id=f"r{i}", request=OmniRequest(inputs={"messages": msg}, params={}, metadata={}), data=None)

    def digest(out: StagePayload) -> str:
        d = out.data
        h = hashlib.sha256(d["prompt"]["input_ids"].numpy().tobytes())
        for t in d["encoder_inputs"]["image_encoder"]["pixel_values"]:
            h.update(t.float().numpy().tobytes())
        return h.hexdigest()

    serial = [digest(asyncio.run(proc(payload(i)))) for i in range(8)]
    results: dict[int, str] = {}

    def worker(idx: list[int]) -> None:
        for i in idx:
            results[i] = digest(asyncio.run(proc(payload(i))))

    threads = [threading.Thread(target=worker, args=([i, i + 4],)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [results[i] for i in range(8)] == serial, "preprocessing differs when four calls run at once"
    assert len(set(serial)) == 8


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-path", default=None, help="also run the re-entrancy check on the real preprocessor")
    a = ap.parse_args()
    check_factory()
    check_timing_event()
    check_kv_log()
    check_eviction_counter()
    if a.model_path:
        check_reentrant_preprocessor(a.model_path)
    print("ok: sglang_omni_ppv_kv patch checks passed" + (" (with the real preprocessor)" if a.model_path else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
