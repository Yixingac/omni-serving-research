"""CPU-only check of sglang_omni_ppv_kv.patch (R3 preprocessing concurrency, preprocessing timing, R4 KV counters).

    cd <sglang-omni checkout with the patch> && PYTHONPATH=$PWD <runtime python> <this file> [--model-path DIR]

Checks, without a GPU:
1. create_preprocessing_executor takes max_concurrency (default 1 = the serial loop of 520b49ee) and hands it to
   SimpleScheduler; 4 gives four workers.
2. The preprocessor's __call__ emits one exp_preprocess event per call with the phase times and the number of calls in
   flight; phases add up across nested calls of add_preprocess_time; a failing call still decrements the counter.
3. OmniScheduler.exp_kv_log: one exp_kv_prefill line per new request (first chunk only), with cached/extend tokens and
   the eviction counter; one exp_kv_pool line per second at most; nothing for a non-thinker stage.
4. The same through the real call site (get_next_batch_to_run) with a real SGLang Req in the state the prefill adder
   leaves it (prefix_indices matched, extend_range set) and the real ForwardMode; the batch fields read exist on the
   real ScheduleBatch. (The first GPU run crashed on Req.extend_input_len, which SGLang 0.5.21 does not have.)
5. Instrumentation never raises into the scheduler or the request: a raising logger or an unexpected object is
   reported once (logger.exception) and switches exp_kv off; a raising event emitter does the same for
   exp_preprocess while the request still returns; the eviction wrapper always returns evict's result.
6. exp_kv_count_evictions counts the tokens and calls of tree_cache.evict and returns its result unchanged.
7. With --model-path: the real MiniCPM-o preprocessor (CPU device) gives identical input_ids and pixel values for
   eight distinct inline images whether the calls run one at a time or four at a time in threads (re-entrancy).
On a checkout without the patch, check 1 fails first; on the first version of the patch, check 4 fails.
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
    req = real_req
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


def kv_scheduler() -> OmniScheduler:
    sched = object.__new__(OmniScheduler)
    sched.exp_kv_enabled = True
    sched.exp_kv_evicted_tokens = 0
    sched.exp_kv_evict_calls = 0
    sched.exp_kv_seen_rids = {}
    sched.exp_kv_last_pool_log_s = 0.0
    sched.gpu_id = 0
    sched.max_total_num_tokens = 127039
    sched.token_to_kv_pool_allocator = SimpleNamespace(available_size=lambda: 90000)
    sched.tree_cache = SimpleNamespace(evictable_size=lambda: 3000, protected_size=lambda: 400)
    sched.waiting_queue = []
    sched.chunked_req = None
    sched.last_batch = None
    sched.running_batch = SimpleNamespace(reqs=[], batch_is_full=False, is_empty=lambda: True)
    return sched


def real_req(rid: str, n_tokens: int, cached: int):  # noqa: ANN201
    """A real SGLang Req in the state the prefill adder leaves it: matched prefix and this chunk's extend range."""
    from array import array

    import torch
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    req = Req(rid=rid, origin_input_text="", origin_input_ids=array("q", range(n_tokens)),
              sampling_params=SamplingParams(max_new_tokens=8), vocab_size=151000)
    req.prefix_indices = torch.arange(cached, dtype=torch.int64)
    req.set_extend_range(cached, n_tokens)
    return req


def check_kv_log_real_req_through_call_site() -> None:
    import dataclasses

    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    fields = {f.name for f in dataclasses.fields(ScheduleBatch)} if dataclasses.is_dataclass(ScheduleBatch) else set(vars(ScheduleBatch))
    assert {"reqs", "forward_mode"} <= fields | set(dir(ScheduleBatch)), "ScheduleBatch fields read by exp_kv_log"
    sched = kv_scheduler()
    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND, reqs=[real_req("r-0", 400, 120), real_req("r-1", 64, 0)])
    plan = SimpleNamespace(batch_to_run=batch, running_batch=sched.running_batch)
    cap = _Capture()
    omni_scheduler.logger.addHandler(cap)
    omni_scheduler.logger.setLevel(logging.INFO)
    try:
        with mock.patch.object(omni_scheduler, "_get_active_stage", lambda: "thinker"), \
                mock.patch.object(omni_scheduler._Upstream, "get_next_batch_to_run", lambda self, rb, lb: plan):
            assert OmniScheduler.get_next_batch_to_run(sched) is batch
    finally:
        omni_scheduler.logger.removeHandler(cap)
    assert sched.exp_kv_enabled, f"exp_kv switched itself off: {cap.lines}"
    lines = [x for x in cap.lines if x.startswith("exp_kv_prefill")]
    assert "rid=r-0 input_tokens=400 cached_tokens=120 extend_tokens=280 evicted_tokens_total=0" in lines[0], lines
    assert "rid=r-1 input_tokens=64 cached_tokens=0 extend_tokens=64" in lines[1], lines


def check_instrumentation_never_raises() -> None:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    sched = kv_scheduler()
    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND, reqs=[real_req("r-2", 10, 0)])
    cap = _Capture()
    omni_scheduler.logger.addHandler(cap)
    try:
        with mock.patch.object(omni_scheduler, "_get_active_stage", lambda: "thinker"), \
                mock.patch.object(omni_scheduler.logger, "info", side_effect=RuntimeError("logger down")):
            for _ in range(3):  # the call-site pattern: only while enabled
                if sched.exp_kv_enabled:
                    sched.exp_kv_log(batch)
        sched2 = kv_scheduler()
        with mock.patch.object(omni_scheduler, "_get_active_stage", lambda: "thinker"):
            sched2.exp_kv_log(SimpleNamespace(forward_mode=ForwardMode.EXTEND, reqs=[object()]))  # no rid at all
    finally:
        omni_scheduler.logger.removeHandler(cap)
    assert not sched.exp_kv_enabled and not sched2.exp_kv_enabled
    assert sum("disabled" in x for x in cap.lines) == 2, cap.lines  # one report per scheduler
    sched3 = kv_scheduler()
    odd = object()
    cache = SimpleNamespace(evict=lambda params: odd)  # a result without num_tokens_evicted
    sched3.exp_kv_count_evictions(cache)
    assert cache.evict("p") is odd and not sched3.exp_kv_enabled

    obj = object.__new__(pre.MiniCPMOPreprocessor)
    calls = []

    async def fake_preprocess(payload):
        return payload

    obj.preprocess = fake_preprocess

    def broken_emit(**kw):
        calls.append(kw)
        raise RuntimeError("emitter down")

    pre.PREPROCESS_TIMING_ON[0] = True
    try:
        with mock.patch.object(pre, "emit_request_event", broken_emit):
            out = [asyncio.run(obj(SimpleNamespace(request_id=f"q{i}"))) for i in range(3)]
        assert [o.request_id for o in out] == ["q0", "q1", "q2"]  # requests still return
        assert len(calls) == 1 and pre.PREPROCESS_TIMING_ON[0] is False  # switched off after the first failure
        assert pre.PREPROCESS_INFLIGHT[0] == 0
    finally:
        pre.PREPROCESS_TIMING_ON[0] = True


def check_eviction_counter() -> None:
    sched = object.__new__(OmniScheduler)
    sched.exp_kv_enabled = True
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
    check_kv_log_real_req_through_call_site()
    check_instrumentation_never_raises()
    check_eviction_counter()
    if a.model_path:
        check_reentrant_preprocessor(a.model_path)
    print("ok: sglang_omni_ppv_kv patch checks passed" + (" (with the real preprocessor)" if a.model_path else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
