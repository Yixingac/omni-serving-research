"""CPU-only check of the research instrumentation patch (sglang_omni_instr_exp06_08.patch).

Run with the SGLang-Omni runtime's python from the checkout under test:
    cd <sglang-omni checkout> && SGLANG_OMNI_EXP_TRACE=c2w <python> <this file> traced
    cd <sglang-omni checkout> && <python> <this file> untraced

traced: a SimpleScheduler with trace_part="c2w" and an active request event recorder must write one
exp_batch_formed and one exp_batch_done event per request, with the batch id shared by the requests
of a batch, dequeue <= formed <= compute start <= done, and the results unchanged.
untraced: without SGLANG_OMNI_EXP_TRACE the same run writes no exp_* event and returns the same results.
On an unpatched checkout the traced check fails (SimpleScheduler has no trace_part).
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

from sglang_omni.profiler.event_recorder import get_recorder, set_active_stage
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


def double_all(values: list[int]) -> list[int]:
    time.sleep(0.02)  # a batch takes some time, so later requests queue behind it
    return [2 * value for value in values]


def run(traced: bool) -> tuple[list[dict], dict[str, int]]:
    events_dir = Path(tempfile.mkdtemp())
    get_recorder().start("check", str(events_dir), "code2wav@r0")
    scheduler = SimpleScheduler(
        lambda value: 2 * value,
        batch_compute_fn=double_all,
        max_batch_size=4,
        max_batch_wait_ms=5,
        trace_part="c2w" if traced else None,
    )

    def serve() -> None:
        set_active_stage("code2wav@r0")
        scheduler.start()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    for index in range(10):
        scheduler.enqueue(IncomingMessage(request_id=f"r{index}", type="new_request", data=index))
    results: dict[str, int] = {}
    deadline = time.time() + 10
    while len(results) < 10 and time.time() < deadline:
        message = scheduler.outbox.get(timeout=5)
        results[message.request_id] = message.data
    scheduler.stop()
    thread.join(timeout=5)
    get_recorder().stop()
    events = [json.loads(line) for path in events_dir.glob("*.jsonl") for line in path.read_text().splitlines()]
    return [e for e in events if e["event_name"].startswith("exp_")], results


def main() -> int:
    mode = sys.argv[1]
    events, results = run(traced=mode == "traced")
    assert results == {f"r{i}": 2 * i for i in range(10)}, results
    if mode == "untraced":
        assert events == [], events[:3]
        print("untraced: no exp_* events, results unchanged")
        return 0
    formed = {e["request_id"]: e for e in events if e["event_name"] == "exp_batch_formed"}
    done = {e["request_id"]: e for e in events if e["event_name"] == "exp_batch_done"}
    assert set(formed) == set(done) == set(results), (sorted(formed), sorted(done))
    by_batch: dict[int, list[str]] = {}
    for request_id, event in formed.items():
        meta = event["metadata"]
        by_batch.setdefault(meta["batch_id"], []).append(request_id)
        assert meta["dequeue_ns"] <= event["timestamp_ns"] <= done[request_id]["metadata"]["compute_start_ns"] <= done[request_id]["timestamp_ns"]
        assert done[request_id]["metadata"]["batch_id"] == meta["batch_id"]
    assert all(len(ids) == formed[ids[0]]["metadata"]["batch_size"] for ids in by_batch.values())
    assert max(len(ids) for ids in by_batch.values()) > 1, by_batch  # requests did queue and batch
    print(f"traced: {len(by_batch)} batches, sizes {sorted(len(v) for v in by_batch.values())}, events consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
