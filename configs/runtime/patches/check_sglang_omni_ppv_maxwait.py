#!/usr/bin/env python3
"""CPU check of the P1 max-wait patch (sglang_omni_batch_ppv_p1.patch = 实验 10's batching patch + max-wait aging + gc log).

    cd <patched sglang-omni checkout> && PYTHONPATH=$PWD <python with torch> <this file>

No GPU and no model: SimpleScheduler gets dummy compute functions, a cost function that reads a number from the payload
and a fake monotonic clock. Requests enter through enqueue(), as in the server, so their arrival times are recorded.
Checks (registered in research/next_steps/p0_p3_experiment_plan.md, P1 最长等待的设计):
1. control: max_wait_ms is applied when the file changes (number -> seconds, null -> no aging); a missing key keeps the
   current value; invalid values (text, negative, infinite) change nothing; settings switch back and forth at the next
   batch (FIFO16 -> Length8 + 1 s -> Length8 -> FIFO16);
2. aging: requests that have waited max_wait or longer come first, in arrival order, then the closest costs to the
   oldest; a request that has not waited long enough is never forced;
3. batch-size limit: forced picks never exceed the cap; the batch never exceeds the cap;
4. endpoints: max_wait 0 gives the FIFO batches of the same cap; no max_wait gives exactly 实验 10's length batches;
5. starvation: a seeded random arrival stream with a long service time per batch; every request leaves exactly once;
   no request that has waited max_wait is overtaken by one that has not; aged requests leave in arrival order;
6. an aged request that does not fit max_batch_cost closes the batch (nobody younger takes its place);
7. batch events carry enqueue time, waited ms, the forced flag and count, whether the anchor was aged, the setting;
   FIFO batches report no forced picks;
8. an aborted request skipped by the main loop leaves no arrival time behind;
9. the gc hook times generation-2 collections only, pairs start and stop, and is registered only when
   SGLANG_OMNI_EXP_TRACE names gc;
10. with stack in SGLANG_OMNI_EXP_TRACE and SGLANG_OMNI_EXP_STACK_DIR set, a process writes a registration line with
    its pid and start time and dumps every thread's stack on SIGUSR2 (Linux only: skipped where /proc is missing).
On the 实验 10 patch alone, check 1 fails first (max_wait_ms is ignored). Exit code 0 when everything holds.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
import time
from unittest import mock

from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


class Payload:
    def __init__(self, cost: int):
        self.cost = cost


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def scheduler(control_path: str | None, clock: Clock, max_batch_size: int = 16, max_batch_cost: int | None = None) -> SimpleScheduler:
    s = SimpleScheduler(lambda p: p, batch_compute_fn=lambda ps: ps, max_batch_size=max_batch_size,
                        request_cost_fn=lambda p: p.cost, max_batch_cost=max_batch_cost, control_path=control_path)
    s.clock = clock
    return s


def arrive(s: SimpleScheduler, clock: Clock, items: list[tuple[str, int, float]]) -> None:
    """items: (request id, cost, arrival time); the clock moves to each arrival time."""
    for rid, cost, at in items:
        clock.t = at
        s.enqueue(IncomingMessage(request_id=rid, type="new_request", data=Payload(cost)))


def take(s: SimpleScheduler) -> list[str]:
    first = s.next_message()
    return [m.request_id for m in s.collect_batch(first)] if first is not None else []


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


def drain(s: SimpleScheduler) -> list[list[str]]:
    out = []
    while True:
        b = take(s)
        if not b:
            return out
        out.append(b)


def main() -> int:
    failures: list[str] = []
    clock = Clock()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c2w_control.json")

        # 1. control file
        s = scheduler(path, clock)
        write(path, {"max_batch_size": 8, "policy": "length", "window": 64, "max_wait_ms": 1000})
        arrive(s, clock, [("a", 100, 0.0)])
        take(s)
        check(s.max_wait_s == 1.0 and s.batch_policy == "length" and s.max_batch_size == 8, "max_wait_ms 1000 -> 1 s", failures)
        for bad in ("abc", -5, "inf", float("nan")):
            write(path, {"max_wait_ms": bad} if not isinstance(bad, float) else '{"max_wait_ms": NaN}')
            arrive(s, clock, [("x", 1, clock.t)])
            take(s)
            check(s.max_wait_s == 1.0, f"an invalid max_wait_ms ({bad!r}) changes nothing", failures)
        write(path, {"max_batch_size": 8, "policy": "length", "window": 64})
        arrive(s, clock, [("y", 1, clock.t)])
        take(s)
        check(s.max_wait_s == 1.0, "a missing max_wait_ms key keeps the current value", failures)
        write(path, {"max_batch_size": 8, "policy": "length", "window": 64, "max_wait_ms": None})
        arrive(s, clock, [("z", 1, clock.t)])
        take(s)
        check(s.max_wait_s is None, "max_wait_ms null switches aging off", failures)
        seq = [({"max_batch_size": 16, "policy": "fifo", "window": 64, "max_wait_ms": None}, ("fifo", 16, None)),
               ({"max_batch_size": 8, "policy": "length", "window": 64, "max_wait_ms": 500}, ("length", 8, 0.5)),
               ({"max_batch_size": 8, "policy": "length", "window": 64, "max_wait_ms": None}, ("length", 8, None)),
               ({"max_batch_size": 16, "policy": "fifo", "window": 64, "max_wait_ms": None}, ("fifo", 16, None))]
        ok = True
        for content, want in seq:
            write(path, content)
            arrive(s, clock, [(f"h{len(content)}", 1, clock.t)])
            take(s)
            ok &= (s.batch_policy, s.max_batch_size, s.max_wait_s) == want
        check(ok, "hot switch FIFO16 -> Length8 + 0.5 s -> Length8 -> FIFO16 takes effect at the next batch", failures)

        # 2. aging order: oldest anchors; aged requests first in arrival order; then the closest costs
        s = scheduler(path, clock)
        write(path, {"max_batch_size": 4, "policy": "length", "window": 64, "max_wait_ms": 1000})
        arrive(s, clock, [("old", 500, 0.0), ("a1", 100, 0.2), ("a2", 900, 0.4), ("young1", 510, 1.5), ("young2", 490, 1.6),
                          ("young3", 505, 1.7)])
        clock.t = 1.8  # old, a1, a2 have waited >= 1 s; the young ones have not
        got = take(s)
        check(got == ["old", "a1", "a2", "young3"], f"aged requests first in arrival order, then the closest cost (got {got})", failures)
        check(s.last_forced == 2 and s.last_anchor_aged, f"two forced picks besides the aged anchor (got {s.last_forced})", failures)
        rest = take(s)
        check(rest == ["young1", "young2"], f"the rest keep their order (got {rest})", failures)

        # a request that has waited exactly max_wait counts as aged (registered: waited >= tau)
        s = scheduler(path, clock)
        write(path, {"max_batch_size": 2, "policy": "length", "window": 64, "max_wait_ms": 1000})
        arrive(s, clock, [("anchor", 500, 0.0), ("edge", 100, 0.5), ("close", 500, 1.25)])
        clock.t = 1.5
        got = take(s)
        check(got == ["anchor", "edge"], f"waited exactly max_wait counts as aged (got {got})", failures)
        take(s)

        # without aging the same queue gives 实验 10's batch (closest costs to the anchor)
        s = scheduler(path, clock)
        write(path, {"max_batch_size": 4, "policy": "length", "window": 64, "max_wait_ms": None})
        arrive(s, clock, [("old", 500, 0.0), ("a1", 100, 0.2), ("a2", 900, 0.4), ("young1", 510, 1.5), ("young2", 490, 1.6),
                          ("young3", 505, 1.7)])
        clock.t = 1.8
        got = take(s)
        check(got == ["old", "young1", "young2", "young3"], f"no max_wait: closest costs to the oldest (got {got})", failures)

        # 3. the cap bounds forced picks
        s = scheduler(path, clock)
        write(path, {"max_batch_size": 3, "policy": "length", "window": 64, "max_wait_ms": 100})
        arrive(s, clock, [(f"r{i}", 100 * (i + 1), 0.01 * i) for i in range(10)])
        clock.t = 5.0
        batches = drain(s)
        check(all(len(b) <= 3 for b in batches) and batches[0] == ["r0", "r1", "r2"],
              f"forced picks stay within the cap of 3 and keep arrival order (got {batches[:2]})", failures)

        # 4. endpoints: max_wait 0 = FIFO of the same cap; none = 实验 10 length batches
        costs = [100, 900, 110, 880, 120, 860, 130, 840, 140, 820, 150]
        items = [(f"q{i}", c, 0.001 * i) for i, c in enumerate(costs)]
        s_zero, s_fifo = scheduler(path, clock), scheduler(None, clock, max_batch_size=4)
        write(path, {"max_batch_size": 4, "policy": "length", "window": 64, "max_wait_ms": 0})
        arrive(s_zero, clock, items)
        arrive(s_fifo, clock, items)
        clock.t = 1.0
        zero, fifo = drain(s_zero), drain(s_fifo)
        check(zero == fifo, f"max_wait 0 gives the FIFO batches of cap 4 (got {zero[:2]} vs {fifo[:2]})", failures)

        # 5. starvation: seeded arrivals, service 0.4 s per batch, max_wait 1 s
        rng = random.Random(20261119)
        s = scheduler(path, clock)
        write(path, {"max_batch_size": 8, "policy": "length", "window": 64, "max_wait_ms": 1000})
        t, pending, arrivals = 0.0, [], {}
        for i in range(400):
            t += rng.expovariate(25.0)
            cost = rng.choice([rng.randint(120, 200), rng.randint(450, 700)])
            pending.append((f"s{i}", cost, t))
            arrivals[f"s{i}"] = t
        clock.t, left, out, overtaken, order_bad = 0.0, list(pending), [], 0, 0
        while left or s.inbox.qsize() or s.pending_messages:
            while left and left[0][2] <= clock.t:
                rid, cost, at = left.pop(0)
                saved = clock.t
                arrive(s, clock, [(rid, cost, at)])
                clock.t = saved
            if not (s.inbox.qsize() or s.pending_messages):
                clock.t = left[0][2]
                continue
            waiting = [m.request_id for m in list(s.pending_messages)] + [m.request_id for m in list(s.inbox.queue)]
            now = clock.t
            batch = take(s)
            aged_left = [r for r in waiting if r not in batch and now - arrivals[r] >= 1.0]
            young_in = [r for r in batch if now - arrivals[r] < 1.0]
            if aged_left and young_in:
                overtaken += 1
            aged_in = [r for r in batch if now - arrivals[r] >= 1.0]
            if aged_in != sorted(aged_in, key=lambda r: arrivals[r]) or (aged_left and aged_in and
                                                                         max(arrivals[r] for r in aged_in) > min(arrivals[r] for r in aged_left)):
                order_bad += 1
            out.extend(batch)
            clock.t += 0.4
        check(sorted(out) == sorted(arrivals) and len(out) == len(set(out)), "starvation run: every request leaves exactly once", failures)
        check(overtaken == 0, f"no aged request is overtaken by a younger one ({overtaken} batches did)", failures)
        check(order_bad == 0, f"aged requests leave in arrival order ({order_bad} batches did not)", failures)

        # 6. an aged request over max_batch_cost closes the batch
        s = scheduler(path, clock, max_batch_cost=1000)
        write(path, {"max_batch_size": 8, "policy": "length", "window": 64, "max_wait_ms": 1000})
        arrive(s, clock, [("anchor", 300, 0.0), ("big", 900, 0.1), ("small", 290, 2.0)])
        clock.t = 2.5
        got = take(s)
        check(got == ["anchor"], f"an aged request that does not fit closes the batch (got {got})", failures)
        check(take(s) == ["big"] and take(s) == ["small"], "it anchors the next batch", failures)

        # 7. batch events
        events = []
        s = scheduler(path, clock)
        s.trace_part = "c2w"
        write(path, {"max_batch_size": 4, "policy": "length", "window": 64, "max_wait_ms": 1000})
        arrive(s, clock, [("e0", 500, 0.0), ("e1", 100, 0.5), ("e2", 510, 2.0)])
        clock.t = 2.2
        import asyncio

        from sglang_omni.profiler import exp_trace

        with mock.patch.object(exp_trace, "active", lambda part: True), \
                mock.patch.object(exp_trace, "emit_for_requests", lambda ids, name, md, ts: events.append((ids, name, dict(md)))):
            first = s.next_message()
            batch = s.collect_batch(first)
            loop = asyncio.new_event_loop()
            s.run_traced_batch(batch, loop)
            loop.close()
        formed = {ids[0]: md for ids, name, md in events if name == "exp_batch_formed"}
        check(set(formed) == {"e0", "e1", "e2"}, f"one exp_batch_formed per request (got {sorted(formed)})", failures)
        e1 = formed.get("e1", {})
        check(e1.get("is_forced") is True and formed.get("e2", {}).get("is_forced") is False and formed.get("e0", {}).get("is_forced") is False,
              "is_forced marks the aged non-anchor request only", failures)
        check(e1.get("forced_count") == 1 and e1.get("anchor_aged") is True and e1.get("max_wait_ms") == 1000.0,
              f"batch fields forced_count / anchor_aged / max_wait_ms (got {[e1.get(k) for k in ('forced_count', 'anchor_aged', 'max_wait_ms')]})", failures)
        check(abs(e1.get("waited_ms", -1) - 1700.0) < 1e-6 and e1.get("enqueue_ns", -1) > 0,
              f"waited_ms from the scheduler clock and a wall-clock enqueue time (got {e1.get('waited_ms')})", failures)
        check(not s.enqueue_times and not s.trace_wait, "no arrival times left behind after the batch", failures)

        events.clear()
        s = scheduler(None, clock)
        s.trace_part = "c2w"
        s.last_forced, s.last_anchor_aged = 3, True  # stale values of an earlier length batch
        arrive(s, clock, [("f0", 5, 3.0), ("f1", 6, 3.1)])
        with mock.patch.object(exp_trace, "active", lambda part: True), \
                mock.patch.object(exp_trace, "emit_for_requests", lambda ids, name, md, ts: events.append((ids, name, dict(md)))):
            batch = s.collect_batch(s.next_message())
            loop = asyncio.new_event_loop()
            s.run_traced_batch(batch, loop)
            loop.close()
        md = [m for _, n, m in events if n == "exp_batch_formed"]
        check(md and all(m["forced_count"] == 0 and m["anchor_aged"] is False and m["max_wait_ms"] == -1.0 for m in md),
              "FIFO batches report no forced picks and no max_wait", failures)

    # 8. the abort path drops the arrival time
    import inspect

    import sglang_omni.scheduling.simple_scheduler as simple

    src = inspect.getsource(simple.SimpleScheduler.start_serial)
    check("self.enqueue_times.pop(msg.request_id, None)" in src, "an aborted request skipped by the main loop drops its arrival time", failures)

    # 9. gc hook
    from sglang_omni.profiler import exp_trace

    exp_trace.gc_records.clear()
    exp_trace.on_gc("start", {"generation": 1})
    exp_trace.on_gc("stop", {"generation": 1, "collected": 3})
    exp_trace.on_gc("start", {"generation": 2})
    exp_trace.on_gc("stop", {"generation": 2, "collected": 7})
    rec = list(exp_trace.gc_records)
    check(len(rec) == 1 and rec[0][2] == 7 and rec[0][1] >= 0, f"gc hook records generation-2 collections only (got {rec})", failures)
    exp_trace.on_gc("stop", {"generation": 2, "collected": 1})
    check(len(exp_trace.gc_records) == 1, "a stop without a start records nothing", failures)
    code = "import gc; from sglang_omni.profiler import exp_trace as e; print(e.on_gc in gc.callbacks)"
    on = subprocess.run([sys.executable, "-c", code], env={**os.environ, "SGLANG_OMNI_EXP_TRACE": "c2w,gc"},
                        capture_output=True, text=True).stdout.strip().splitlines()[-1:]
    off = subprocess.run([sys.executable, "-c", code], env={**os.environ, "SGLANG_OMNI_EXP_TRACE": "c2w"},
                         capture_output=True, text=True).stdout.strip().splitlines()[-1:]
    check(on == ["True"] and off == ["False"], f"the gc hook is registered only with gc in SGLANG_OMNI_EXP_TRACE (got {on}, {off})", failures)

    # 10. stack dump on SIGUSR2
    if os.path.exists("/proc/self/stat"):
        with tempfile.TemporaryDirectory() as d:
            code = ("import os, signal, time; from sglang_omni.profiler import exp_trace; "
                    "os.kill(os.getpid(), signal.SIGUSR2); time.sleep(0.2); print(os.getpid())")
            env = {**os.environ, "SGLANG_OMNI_EXP_TRACE": "c2w,stack", "SGLANG_OMNI_EXP_STACK_DIR": d}
            res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
            pid = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else "?"
            path = os.path.join(d, f"stack_{pid}.txt")
            text = open(path).read() if os.path.exists(path) else ""
            check(res.returncode == 0 and text.startswith(f"registered pid={pid} starttime=") and "File " in text,
                  f"SIGUSR2 dumps the stacks of a registered process (rc {res.returncode}, {len(text)} bytes)", failures)
            res = subprocess.run([sys.executable, "-c", "from sglang_omni.profiler import exp_trace; import os; print(len(os.listdir(%r)))" % d],
                                 env={**os.environ, "SGLANG_OMNI_EXP_TRACE": "c2w", "SGLANG_OMNI_EXP_STACK_DIR": d}, capture_output=True, text=True)
            check(res.stdout.strip().splitlines()[-1:] == ["1"], "without stack in SGLANG_OMNI_EXP_TRACE nothing registers", failures)
    else:
        print("skip SIGUSR2 stack dump: no /proc on this system (checked on Linux)")

    print("PASS" if not failures else f"FAIL: {len(failures)} check(s)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
