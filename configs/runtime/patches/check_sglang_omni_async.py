"""CPU-only check of the 实验 11 prototype (sglang_omni_async_exp11.patch): SGLANG_OMNI_SPEECH_LOOKAHEAD=sidecopy copies
the thinker's hidden rows on a side stream that waits only for the step's snapshot.

    cd <sglang-omni checkout> && PYTHONPATH=$PWD <python with torch> <this file>

There is no GPU here, so CUDA events, streams and record_stream are replaced by recording stand-ins: the check is
about control flow and data, not real concurrency (the GPU correctness runs cover that). Checks:
1. the mode table: sidecopy = speech lookahead with the snapshot, plus the side copy; 0/1/nosnapshot unchanged;
2. a lookahead step: post_decode_launch records an event right after the snapshot; at resolve the side stream waits
   for exactly that event, the rows are the snapshot's (a later write into the graph buffer does not leak in),
   record_stream protects the snapshot, and the entry is removed;
3. a synchronous step (no snapshot registered): rows are copied on the launch stream and an event marks them;
4. an entry whose tensor is not the result's snapshot (an id reused after a free) never matches;
5. at the end of a request the side stream waits for the last launch-stream row, stacks and copies there, and the
   values equal the default path's; mode 1 does not touch the side stream;
6. the fingerprint (row norms and 4 random projections) is deterministic and changes with the data;
7. reset_request drops the request's pending event;
8. the output-token digest is emitted by the scheduler after it sets the output ids (on_request_finished runs before
   that, so a digest there would always hash an empty list), and the hidden-state event no longer carries one.
On a checkout without the patch, check 1 fails first.
"""

from __future__ import annotations

import contextlib
import sys
from types import SimpleNamespace
from unittest import mock

import torch

from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.models.minicpm_o import thinker_model_runner as mr
from sglang_omni.models.minicpm_o.thinker_model_runner import MiniCPMOThinkerModelRunner

STREAMS: list = []
RECORDED: list = []


class FakeEvent:
    def __init__(self) -> None:
        self.recorded_on = None

    def record(self, stream=None) -> None:  # noqa: ANN001
        self.recorded_on = STREAMS[-1] if STREAMS else "launch"


class FakeStream:
    def __init__(self, device=None) -> None:  # noqa: ANN001
        self.waits: list = []

    def wait_event(self, event) -> None:  # noqa: ANN001
        self.waits.append(event)


@contextlib.contextmanager
def use_stream(stream):  # noqa: ANN001, ANN201
    STREAMS.append(stream)
    try:
        yield
    finally:
        STREAMS.pop()


def record_stream(self, stream) -> None:  # noqa: ANN001
    RECORDED.append((self.untyped_storage().data_ptr(), stream))


def runner(mode: str) -> MiniCPMOThinkerModelRunner:
    r = object.__new__(MiniCPMOThinkerModelRunner)
    r.speech_lookahead, r.snapshot_hidden = mr.speech_lookahead_mode(mode)
    r.side_copy = mode == "sidecopy"
    r.hidden_copy_stream, r.hidden_ready, r.main_row_ready, r.fingerprint_matrix = None, {}, {}, None
    r.pending_hidden = {}
    return r


def scheduler_output(rids: list[str]) -> SimpleNamespace:
    reqs = [SimpleNamespace(request_id=rid, data=SimpleNamespace(stage_payload=SimpleNamespace(data={"prompt": {}}),
                                                                 req=SimpleNamespace(inflight_middle_chunks=0))) for rid in rids]
    return SimpleNamespace(requests=reqs)


def outputs_for(hidden: torch.Tensor, rids: list[str]) -> dict:
    return {rid: SimpleNamespace(extra={"hidden_states": hidden[i]}) for i, rid in enumerate(rids)}


def patches():  # noqa: ANN201
    return contextlib.ExitStack()


def main() -> int:
    # 1. modes
    assert mr.speech_lookahead_mode("sidecopy") == (True, True)
    assert mr.speech_lookahead_mode("1") == (True, True) and mr.speech_lookahead_mode("nosnapshot") == (True, False)
    assert mr.speech_lookahead_mode("0") == (False, False) and mr.speech_lookahead_mode(None) == (False, False)
    try:
        mr.speech_lookahead_mode("async")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown mode must be refused")

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(torch.cuda, "Event", FakeEvent))
        stack.enter_context(mock.patch.object(torch.cuda, "Stream", FakeStream))
        stack.enter_context(mock.patch.object(torch.cuda, "stream", use_stream))
        stack.enter_context(mock.patch.object(torch.Tensor, "record_stream", record_stream, create=True))
        stack.enter_context(mock.patch.object(ThinkerModelRunner, "post_decode_launch", lambda self, result, fb, reqs: "launched"))
        stack.enter_context(mock.patch.object(mr.MiniCPMOPipelineState, "from_dict", lambda data: SimpleNamespace(prompt={})))

        # 2. a lookahead step
        r = runner("sidecopy")
        graph_buffer = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        result = SimpleNamespace(logits_output=SimpleNamespace(hidden_states=graph_buffer[:2]))
        assert r.post_decode_launch(result, None, []) == "launched"
        snapshot = result.logits_output.hidden_states
        (entry_tensor, event), = r.hidden_ready.values()
        assert entry_tensor is snapshot and event.recorded_on == "launch", "the event is recorded right after the snapshot"
        graph_buffer.add_(100)  # the next launch writes into the graph buffer
        r.post_process_outputs(result, scheduler_output(["a", "b"]), outputs_for(snapshot, ["a", "b"]))
        side = r.hidden_copy_stream
        assert side is not None and side.waits == [event, event], side.waits
        assert torch.equal(torch.stack(r.pending_hidden["a"] + r.pending_hidden["b"]), torch.arange(8, dtype=torch.float32).reshape(2, 4))
        assert any(ptr == snapshot.untyped_storage().data_ptr() and s is side for ptr, s in RECORDED), "record_stream on the snapshot"
        assert not r.hidden_ready and not r.main_row_ready, "the entry is removed at resolve"

        # 3. a synchronous step: no snapshot registered
        sync_hidden = torch.ones(2, 4)
        r.post_process_outputs(SimpleNamespace(logits_output=SimpleNamespace(hidden_states=sync_hidden)),
                               scheduler_output(["a", "b"]), outputs_for(sync_hidden, ["a", "b"]))
        assert len(side.waits) == 2, "no side-stream wait for a synchronous step"
        assert set(r.main_row_ready) == {"a", "b"} and r.main_row_ready["a"].recorded_on == "launch"

        # 4. a stale entry under the result's id but holding another tensor never matches
        other = torch.zeros(2, 4)
        stale = FakeEvent()
        fresh = torch.full((2, 4), 7.0)
        r.hidden_ready[id(fresh)] = (other, stale)
        r.post_process_outputs(SimpleNamespace(logits_output=SimpleNamespace(hidden_states=fresh)),
                               scheduler_output(["c"]), outputs_for(fresh, ["c"]))
        assert stale not in side.waits and "c" in r.main_row_ready

        # 5. end of a request
        req_data = SimpleNamespace(stage_payload=SimpleNamespace(data={}), extra_model_outputs={})
        expected = torch.stack(r.pending_hidden["a"])
        main_ready = r.main_row_ready["a"]
        r.on_request_finished("a", req_data)
        assert side.waits[-1] is main_ready, "the side stream waits for the last row copied on the launch stream"
        assert torch.equal(torch.stack(req_data.extra_model_outputs["hidden_states_seq"]), expected)
        r1 = runner("1")
        r1.pending_hidden["z"] = [torch.ones(4), torch.zeros(4)]
        data1 = SimpleNamespace(stage_payload=SimpleNamespace(data={}), extra_model_outputs={})
        r1.on_request_finished("z", data1)
        assert r1.hidden_copy_stream is None and torch.equal(torch.stack(data1.extra_model_outputs["hidden_states_seq"]),
                                                             torch.stack([torch.ones(4), torch.zeros(4)]))

        # 6. fingerprint
        x = torch.randn(5, 16)
        fp = r.fingerprint(x)
        assert fp == r.fingerprint(x.clone()) and fp != r.fingerprint(x + 1)
        assert len(fp["norm"]) == 5 and all(len(row) == 4 for row in fp["proj"])

        # 7. reset
        r.reset_request("b")
        assert "b" not in r.main_row_ready and "b" not in r.pending_hidden

    # 8. the output-token digest comes after the output ids are set
    import inspect

    import sglang_omni.models.minicpm_o.thinker_model_runner as minicpm_runner
    import sglang_omni.scheduling.omni_scheduler as omni_scheduler

    scheduler_source = inspect.getsource(omni_scheduler)
    assigned = scheduler_source.find("data.output_ids = list(req.output_ids)")
    emitted = scheduler_source.find('"exp_thinker_output"')
    assert 0 <= assigned < emitted, "the output digest must be emitted after data.output_ids is set"
    assert "output_ids_sha256" not in inspect.getsource(minicpm_runner), "on_request_finished must not hash output ids"

    print("async prototype: sidecopy mode; lookahead rows copied on the side stream after their snapshot only; sync rows "
          "marked on the launch stream; stale entries never match; end-of-request copy on the side stream equals the "
          "default; fingerprint deterministic; reset clears pending events; output digest after the output ids are set")
    return 0


if __name__ == "__main__":
    sys.exit(main())
