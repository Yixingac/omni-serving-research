#!/usr/bin/env python3
"""CPU check of the 实验 10 batching patch (sglang_omni_batch_exp10.patch): the control file and length-aware batches.

    PYTHONPATH=<patched sglang-omni checkout> <python with torch> configs/runtime/patches/check_sglang_omni_batch.py

No GPU and no model: SimpleScheduler gets dummy compute functions and a cost function that reads a number from the
payload. Checks:
1. control file: applied when it changes; a missing, unreadable or invalid file changes nothing;
2. length-aware batches: the oldest request first, then the closest costs; nothing lost or duplicated;
   requests not chosen keep their arrival order at the front of the queue, so the oldest anchors the next batch;
3. max_batch_cost still applies; the window bounds how many waiting requests are looked at;
4. FIFO is unchanged when the policy is fifo (and without a control file).
Exit code 0 when everything holds.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


class Payload:
    def __init__(self, cost: int):
        self.cost = cost


def scheduler(control_path: str | None, max_batch_size: int = 16, max_batch_cost: int | None = None) -> SimpleScheduler:
    return SimpleScheduler(lambda p: p, batch_compute_fn=lambda ps: ps, max_batch_size=max_batch_size,
                           request_cost_fn=lambda p: p.cost, max_batch_cost=max_batch_cost, control_path=control_path)


def fill(s: SimpleScheduler, costs: list[int]) -> list[IncomingMessage]:
    msgs = [IncomingMessage(request_id=f"r{i}", type="new_request", data=Payload(c)) for i, c in enumerate(costs)]
    for m in msgs:
        s.inbox.put(m)
    return msgs


def take(s: SimpleScheduler) -> list[int]:
    first = s.next_message()
    return [m.data.cost for m in s.collect_batch(first)] if first is not None else []


def write(path: str, content) -> None:  # noqa: ANN001
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(content if isinstance(content, str) else json.dumps(content))
    os.replace(tmp, path)
    time.sleep(0.01)  # a new modification time


def check(cond: bool, what: str, failures: list[str]) -> None:
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        failures.append(what)


def main() -> int:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c2w_control.json")
        s = scheduler(path)
        fill(s, [100, 500, 110])
        check(take(s) == [100, 500, 110], "no control file: FIFO with the configured max_batch_size", failures)

        write(path, {"max_batch_size": 4, "policy": "length", "window": 8})
        msgs = fill(s, [100, 500, 110, 480, 105, 900, 120])
        first = take(s)
        check(s.max_batch_size == 4 and s.batch_policy == "length" and s.length_window == 8, "control file applied", failures)
        check(first == [100, 110, 105, 120], f"length-aware: oldest first, then the closest costs in arrival order (got {first})", failures)
        second = take(s)
        check(second == [500, 480, 900], f"the rest keep their order and the oldest anchors the next batch (got {second})", failures)
        check(sorted(first + second) == sorted(m.data.cost for m in msgs), "nothing lost or duplicated", failures)

        write(path, "{not json")
        fill(s, [1])
        take(s)  # collect_batch re-reads the file here
        check(s.batch_policy == "length" and s.max_batch_size == 4, "an unreadable control file changes nothing", failures)
        write(path, {"policy": "bogus"})
        fill(s, [1])
        take(s)
        check(s.batch_policy == "length", "an unknown policy is ignored", failures)

        write(path, {"max_batch_size": 3, "policy": "length", "window": 3})
        fill(s, [100, 900, 800, 700, 101])
        got = take(s)
        # 101 is the closest cost but lies outside the window of 3 waiting requests
        check(got == [100, 800, 700] and s.last_window_size == 3,
              f"the window bounds how many waiting requests are compared (got {got})", failures)
        take(s)
        take(s)  # drain the rest
        write(path, {"max_batch_size": 16, "window": 2})
        fill(s, [1])
        take(s)
        check(s.length_window == 16, "the window is never smaller than the batch size", failures)

        write(path, {"max_batch_size": 16, "policy": "fifo"})
        fill(s, [100, 900, 105])
        check(take(s) == [100, 900, 105], "policy fifo restores the original batches", failures)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c2w_control.json")
        s = scheduler(path, max_batch_cost=300)
        write(path, {"max_batch_size": 8, "policy": "length"})
        fill(s, [100, 105, 110, 120, 2000])
        got = take(s)
        check(got == [100, 105], f"max_batch_cost still applies (got {got})", failures)
        rest = take(s) + take(s) + take(s)
        check(sorted(got + rest) == [100, 105, 110, 120, 2000], "requests over the cost come later, none lost", failures)

        s2 = scheduler(None)
        fill(s2, [100, 500, 110])
        check(take(s2) == [100, 500, 110] and s2.batch_policy == "fifo", "without a control path nothing changes", failures)

        s3 = scheduler(path)
        s3.inbox.put(IncomingMessage(request_id="r0", type="new_request", data=Payload(100)))
        s3.inbox.put(IncomingMessage(request_id="x", type="abort"))
        s3.inbox.put(IncomingMessage(request_id="r1", type="new_request", data=Payload(101)))
        got = take(s3)
        kept = [m.type for m in s3.pending_messages]
        check(got == [100, 101] and kept == ["abort"], f"other message types are kept for the main loop (got {got}, {kept})", failures)

    print("PASS" if not failures else f"FAIL: {len(failures)} check(s)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
