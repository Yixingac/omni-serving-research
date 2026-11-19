#!/usr/bin/env python3
"""CPU check of the P2 verify mode (sglang_omni_async_ppv_p2.patch = 实验 11's prototype + sidecopy_verify).

    cd <patched sglang-omni checkout> && PYTHONPATH=$PWD <python with torch, sglang 0.5.21 on the path> <this file>

There is no GPU, so a small device model stands in for CUDA ordering (registered in
research/next_steps/p0_p3_experiment_plan.md, P2: "用替身流和事件，构造读早了、被覆盖的情形"):
- work on the launch stream is ordered; its results become visible to a side stream only after the side stream
  waited for an event recorded after them (or after a device synchronize); a side-stream read of a launch-stream
  result it may not see yet returns stale memory (a sentinel value);
- the caching allocator may hand a snapshot's memory to later launch-stream work as soon as Python drops it, unless
  record_stream(side) protected it; a side-stream copy of an unprotected snapshot then reads reused memory.
Checks:
1. modes: sidecopy_verify = lookahead + snapshot + side copy + verify; sidecopy does not collect references;
2. correct code: every finished request's side copy equals the launch-stream reference and the post-synchronize
   re-read bit for bit, rows = expected rows, rows one per step; downstream rows are the side copy;
3. "read early" per step (the side stream skips the wait for the snapshot event): reported, first bad row = the
   lookahead row; "overwritten" (no record_stream, the memory is reused): reported; "read early" at the end (the
   last sync row's event lost): reported, and the re-read after synchronize differs from the side copy;
4. known_tts requests: the reference is sliced like the side rows; expected rows = len(known_tts_output_ids);
5. a missed step (positions jump) and a row count that does not match len(output_ids) are reported;
6. mismatches are logged as warnings even without request events; the event carries the step context;
7. reset_request (abort) drops the request's references.
On 实验 11's patch alone, check 1 fails first (sidecopy_verify is an unknown mode). Exit code 0 when everything holds.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from types import SimpleNamespace
from unittest import mock

import torch

from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.models.minicpm_o import thinker_model_runner as mr
from sglang_omni.models.minicpm_o.thinker_model_runner import MiniCPMOThinkerModelRunner
from sglang_omni.profiler import exp_trace

STALE = -12345.0
ORIG_CLONE = torch.Tensor.clone
ORIG_STACK = torch.stack


class Device:
    def __init__(self) -> None:
        self.event_count = 0
        self.unpublished: list[int] = []        # launch-stream results since the last launch-stream event
        self.published: dict[int, int] = {}      # storage -> number of the launch-stream event that covers it
        self.synced_upto = 0                     # everything up to this event is visible to all streams
        self.contexts: list[FakeStream] = []     # current side-stream context (empty = launch stream)
        self.protected: set[int] = set()
        self.side_reads: list[tuple[torch.Tensor, int]] = []

    def on_launch(self) -> bool:
        return not self.contexts

    def produced(self, tensor: torch.Tensor) -> None:
        if self.on_launch():
            self.unpublished.append(tensor.untyped_storage().data_ptr())

    def visible(self, tensor: torch.Tensor) -> bool:
        if self.on_launch():
            return True
        ptr = tensor.untyped_storage().data_ptr()
        if ptr in self.unpublished:
            return False
        if ptr not in self.published:
            return True  # not a launch-stream result (test input or a side-stream copy)
        return self.published[ptr] <= max(self.contexts[-1].horizon, self.synced_upto)


DEV = Device()


class FakeEvent:
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        self.number: int | None = None
        self.recorded_on = None
        self.ignore = False

    def record(self, stream=None) -> None:  # noqa: ANN001
        self.recorded_on = "launch" if DEV.on_launch() else "side"
        if DEV.on_launch():
            DEV.event_count += 1
            self.number = DEV.event_count
            for ptr in DEV.unpublished:
                DEV.published[ptr] = self.number
            DEV.unpublished.clear()


class FakeStream:
    def __init__(self, device=None) -> None:  # noqa: ANN001
        self.horizon = 0
        self.waits: list[FakeEvent] = []

    def wait_event(self, event: FakeEvent) -> None:
        self.waits.append(event)
        if event.number is not None and not event.ignore:
            self.horizon = max(self.horizon, event.number)


@contextlib.contextmanager
def use_stream(stream):  # noqa: ANN001, ANN201
    DEV.contexts.append(stream)
    try:
        yield
    finally:
        DEV.contexts.pop()


def fake_clone(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN201
    out = ORIG_CLONE(self, *args, **kwargs)
    if not DEV.visible(self):
        out.fill_(STALE)
    if not DEV.on_launch() and self.untyped_storage().data_ptr() in DEV.published:
        DEV.side_reads.append((out, self.untyped_storage().data_ptr()))
    DEV.produced(out)
    return out


def fake_stack(tensors, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN201
    rows = [t if DEV.visible(t) else torch.full_like(t, STALE) for t in tensors]
    out = ORIG_STACK(rows, *args, **kwargs)
    DEV.produced(out)
    return out


def fake_record_stream(self, stream) -> None:  # noqa: ANN001
    if isinstance(stream, FakeStream):
        DEV.protected.add(self.untyped_storage().data_ptr())


def fake_synchronize(device=None) -> None:  # noqa: ANN001
    DEV.event_count += 1
    for ptr in DEV.unpublished:
        DEV.published[ptr] = DEV.event_count
    DEV.unpublished.clear()
    DEV.synced_upto = DEV.event_count


def allocator_reuse() -> None:
    """After a step: side copies of snapshots nobody protected read memory the launch stream has reused."""
    for out, ptr in DEV.side_reads:
        if ptr not in DEV.protected:
            out.fill_(STALE - 1)
    DEV.side_reads.clear()


def runner(mode: str) -> MiniCPMOThinkerModelRunner:
    r = object.__new__(MiniCPMOThinkerModelRunner)
    r.speech_lookahead, r.snapshot_hidden = mr.speech_lookahead_mode(mode)
    r.side_copy = mode in ("sidecopy", "sidecopy_verify")
    r.verify_copy = mode == "sidecopy_verify"
    r.hidden_copy_stream, r.hidden_ready, r.main_row_ready, r.fingerprint_matrix = None, {}, {}, None
    r.pending_hidden, r.reference_hidden, r.verify_positions = {}, {}, {}
    r.verify_step_index = r.verify_step_batch_size = r.verify_step_finished = 0
    return r


class Request:
    def __init__(self, request_id: str, known_tts: list[int] | None = None) -> None:
        self.request_id = request_id
        prompt = {"known_tts_output_ids": known_tts} if known_tts is not None else {}
        self.req = SimpleNamespace(inflight_middle_chunks=0, output_ids=[])
        self.data = SimpleNamespace(stage_payload=SimpleNamespace(data={"prompt": prompt}), req=self.req)


def step(r: MiniCPMOThinkerModelRunner, requests: list[Request], rows: torch.Tensor, lookahead: bool, graph: torch.Tensor,
         tokens_added: int = 1) -> None:
    """One decode step: rows[i] are request i's hidden rows. A lookahead step snapshots the graph buffer at launch, the
    next launch overwrites the buffer, then the step resolves. A sync step resolves the buffer directly."""
    graph.copy_(rows)
    if lookahead:
        result = SimpleNamespace(logits_output=SimpleNamespace(hidden_states=graph))
        r.post_decode_launch(result, None, [])
        hidden = result.logits_output.hidden_states  # the snapshot
        graph.add_(1000.0)  # the next launch writes the graph buffer
    else:
        hidden = graph
        result = SimpleNamespace(logits_output=SimpleNamespace(hidden_states=hidden))
    for req in requests:
        req.req.output_ids.extend([7] * tokens_added)
    outputs = {req.request_id: SimpleNamespace(extra={"hidden_states": hidden[i]}) for i, req in enumerate(requests)}
    r.post_process_outputs(result, SimpleNamespace(requests=requests), outputs)
    allocator_reuse()


def finish(r: MiniCPMOThinkerModelRunner, req: Request, events: list) -> dict:
    data = SimpleNamespace(stage_payload=req.data.stage_payload, extra_model_outputs={}, req=req.req)
    r.on_request_finished(req.request_id, data)
    return {"rows": data.extra_model_outputs.get("hidden_states_seq"),
            "verify": next((md for ids, name, md in reversed(events) if name == "exp_hidden_verify" and ids == [req.request_id]), None)}


def scenario(mode: str, bug: str | None, events: list, known_tts: bool = False, skip_step: bool = False,
             extra_token: bool = False) -> dict:
    """prefill (sync), lookahead, sync, sync for one request 'a' (or a known_tts request with 2 rows per step)."""
    global DEV
    DEV = Device()
    r = runner(mode)
    width = 4
    per_step = 2 if known_tts else 1
    req = Request("a", known_tts=[11, 12, 13, 14, 15] if known_tts else None)
    graph = torch.zeros(per_step, width)
    plan = [False, True, False, False]
    for k, is_lookahead in enumerate(plan):
        rows = torch.arange(per_step * width, dtype=torch.float32).reshape(per_step, width) + 10.0 * (k + 1)
        if bug == "read_early_step" and is_lookahead:
            graph.copy_(rows)
            result = SimpleNamespace(logits_output=SimpleNamespace(hidden_states=graph.unsqueeze(0)))
            r.post_decode_launch(result, None, [])
            for _, event in r.hidden_ready.values():
                event.ignore = True  # the side stream skips the wait for the snapshot
            hidden = result.logits_output.hidden_states
            graph.add_(1000.0)
            req.req.output_ids.append(7)
            out = {"a": SimpleNamespace(extra={"hidden_states": hidden[0]})}
            r.post_process_outputs(result, SimpleNamespace(requests=[req]), out)
            allocator_reuse()
            continue
        step(r, [req], rows.unsqueeze(0), is_lookahead, graph.unsqueeze(0),
             tokens_added=2 if (skip_step and k == 2) else 1)
    if extra_token:
        req.req.output_ids.append(7)
    if bug == "read_early_end":
        r.main_row_ready.pop("a", None)  # the last sync row's event is lost
    out = finish(r, req, events)
    out["runner"] = r
    return out


def check(cond: bool, what: str, failures: list[str]) -> None:
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        failures.append(what)


def main() -> int:
    failures: list[str] = []
    # 1. modes
    check(mr.speech_lookahead_mode("sidecopy_verify") == (True, True), "sidecopy_verify = lookahead with the snapshot", failures)
    src = open(mr.__file__).read()
    check('in ("sidecopy", "sidecopy_verify")' in src and 'self.verify_copy = os.environ.get("SGLANG_OMNI_SPEECH_LOOKAHEAD") == "sidecopy_verify"' in src,
          "the side copy is on in both modes, verify only in sidecopy_verify", failures)

    events: list = []
    logs: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: logs.append(record.getMessage())  # noqa: E731
    mr.logger.addHandler(handler)
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(torch.cuda, "Event", FakeEvent))
        stack.enter_context(mock.patch.object(torch.cuda, "Stream", FakeStream))
        stack.enter_context(mock.patch.object(torch.cuda, "stream", use_stream))
        stack.enter_context(mock.patch.object(torch.cuda, "synchronize", fake_synchronize))
        stack.enter_context(mock.patch.object(torch.Tensor, "record_stream", fake_record_stream, create=True))
        stack.enter_context(mock.patch.object(torch.Tensor, "clone", fake_clone))
        stack.enter_context(mock.patch.object(torch, "stack", fake_stack))
        stack.enter_context(mock.patch.object(ThinkerModelRunner, "post_decode_launch", lambda self, result, fb, reqs: "launched"))
        stack.enter_context(mock.patch.object(mr.MiniCPMOPipelineState, "from_dict",
                                              lambda data: SimpleNamespace(prompt=data.get("prompt") or {})))
        stack.enter_context(mock.patch.object(exp_trace, "emit_for_requests", lambda ids, name, md, ts: events.append((ids, name, dict(md)))))

        # sidecopy (no verify) collects no references
        plain = scenario("sidecopy", None, events)
        check(not plain["runner"].reference_hidden and plain["verify"] is None, "sidecopy collects no references and emits no verify event", failures)

        # 2. correct code
        good = scenario("sidecopy_verify", None, events)
        v = good["verify"] or {}
        check(v.get("is_equal_reference") is True and v.get("is_equal_resync") is True and v.get("first_bad_row") == -1,
              f"correct code: side copy == reference == re-read (got {v})", failures)
        check(v.get("rows") == 4 and v.get("expected_rows") == 4 and v.get("is_rows_expected") is True and v.get("is_aligned") is True,
              "correct code: 4 rows = len(output_ids), one per step", failures)
        check(good["rows"] is not None and torch.equal(ORIG_STACK(good["rows"]), ORIG_STACK([torch.arange(4.0) + 10 * (k + 1) for k in range(4)])),
              "downstream gets the side copy with the true rows", failures)
        check(not good["runner"].reference_hidden and not good["runner"].verify_positions, "references dropped after the request", failures)
        check(not any("exp_hidden_verify mismatch" in m for m in logs), "no warning for a correct request", failures)

        # 3. the three bug classes
        early = scenario("sidecopy_verify", "read_early_step", events)
        v = early["verify"] or {}
        check(v.get("is_equal_reference") is False and v.get("first_bad_row") == 1,
              f"read early (step): reported at the lookahead row 1 (got {v.get('is_equal_reference')}, {v.get('first_bad_row')})", failures)
        check(v.get("side_sha256") != v.get("reference_sha256"), "read early (step): the two hashes differ", failures)

        with mock.patch.object(torch.Tensor, "record_stream", lambda self, stream: None, create=True):
            reused = scenario("sidecopy_verify", None, events)
        v = reused["verify"] or {}
        check(v.get("is_equal_reference") is False and v.get("first_bad_row") == 1,
              f"overwritten (no record_stream): reported at row 1 (got {v.get('is_equal_reference')}, {v.get('first_bad_row')})", failures)

        late = scenario("sidecopy_verify", "read_early_end", events)
        v = late["verify"] or {}
        check(v.get("is_equal_reference") is False and v.get("is_equal_resync") is False and v.get("first_bad_row") == 2,
              f"read early (end): reported at the first sync row after the last side wait (got {v})", failures)
        check(sum("exp_hidden_verify mismatch request_id=a" in m for m in logs) == 3, "each mismatch is logged as a warning", failures)
        check(all(k in v for k in ("step_index", "step_batch_size", "finished_in_step", "dtype", "reference_dtype")),
              "the event carries the step context and dtypes", failures)

        # 4. known_tts slicing
        kt = scenario("sidecopy_verify", None, events, known_tts=True)
        v = kt["verify"] or {}
        check(v.get("is_equal_reference") is True and v.get("rows") == 5 and v.get("expected_rows") == 5 and v.get("known_tts") is True,
              f"known_tts: reference sliced like the side rows, 5 text rows (got {v.get('rows')}, {v.get('expected_rows')})", failures)

        # 5. a missed step and a wrong row count
        skip = scenario("sidecopy_verify", None, events, skip_step=True)
        v = skip["verify"] or {}
        check(v.get("is_aligned") is False and v.get("is_equal_reference") is True, "a jump in output positions is reported", failures)
        extra = scenario("sidecopy_verify", None, events, extra_token=True)
        v = extra["verify"] or {}
        check(v.get("is_rows_expected") is False and v.get("expected_rows") == 5 and v.get("rows") == 4,
              "rows that do not match len(output_ids) are reported", failures)

        # 7. abort
        r = runner("sidecopy_verify")
        req = Request("x")
        g = torch.zeros(1, 1, 4)
        step(r, [req], torch.ones(1, 1, 4), False, g)
        check("x" in r.reference_hidden, "references are collected while the request runs", failures)
        r.reset_request("x")
        check("x" not in r.reference_hidden and "x" not in r.verify_positions and "x" not in r.pending_hidden,
              "reset_request drops the request's references", failures)
    mr.logger.removeHandler(handler)

    print("PASS" if not failures else f"FAIL: {len(failures)} check(s)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
