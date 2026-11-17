"""CPU-only check of the 实验 8 prototype (sglang_omni_proto_exp08.patch): speech requests may use the thinker's
one-step lookahead when SGLANG_OMNI_SPEECH_LOOKAHEAD is set, and each launched step keeps its own hidden states.

    cd <sglang-omni checkout> && PYTHONPATH=$PWD <runtime python> <this file>

Checks, without a GPU or a model:
1. SGLANG_OMNI_SPEECH_LOOKAHEAD: unset or 0 = off (upstream behaviour), 1 = on with the snapshot,
   nosnapshot = on without it (a diagnostic arm that is expected to give wrong hidden states); other values fail.
2. lookahead_eligible: with the switch on, a speech request is eligible; logprob, a sampling seed or a
   repetition penalty still make a batch ineligible; with the switch off, speech is rejected as upstream does.
3. post_decode_launch: with the snapshot, the result handed to resolve holds a copy of the hidden states, so a
   later write into the graph's output buffer (the next launch) does not change what resolve reads; off or
   nosnapshot, it is the buffer itself.
4. SGLANG_OMNI_EXP_TORCH_SKIP (measurement only): stages whose names start with a listed prefix do not start the
   torch profiler, while every stage still starts its request event recorder.
5. exp_trace.digest: the short hash used to compare hidden states and codec tokens across runs.
On a checkout without the prototype, check 1 fails first.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock

import torch

from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.models.minicpm_o import thinker_model_runner as minicpm_runner
from sglang_omni.models.minicpm_o.thinker_model_runner import MiniCPMOThinkerModelRunner
from sglang_omni.pipeline.stage import runtime as stage_runtime
from sglang_omni.profiler import exp_trace


def request(speech: bool, logprob: bool = False, seed: int | None = None, penalty: float = 1.0) -> SimpleNamespace:
    # should_generate_audio_output reads metadata["output_modalities"] of the request it is given.
    payload = SimpleNamespace(metadata={"output_modalities": ["text", "audio"] if speech else ["text"]})
    sampling = SimpleNamespace(repetition_penalty=penalty, presence_penalty=0.0, frequency_penalty=0.0, min_new_tokens=0,
                               sampling_seed=seed, logit_bias=None, custom_params=None)
    return SimpleNamespace(omni_data=SimpleNamespace(stage_payload=payload, return_logprob=logprob), sampling_params=sampling)


def runner(mode: str) -> MiniCPMOThinkerModelRunner:
    r = object.__new__(MiniCPMOThinkerModelRunner)
    r.speech_lookahead, r.snapshot_hidden = minicpm_runner.speech_lookahead_mode(mode)
    return r


def check_modes() -> None:
    mode = minicpm_runner.speech_lookahead_mode
    assert mode(None) == (False, False) and mode("0") == (False, False)
    assert mode("1") == (True, True)
    assert mode("nosnapshot") == (True, False)
    try:
        mode("yes")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown value must fail instead of silently meaning off")


def check_eligibility() -> None:
    from sglang_omni.models.qwen3_omni.request_builders import should_generate_audio_output

    speech = request(speech=True)
    assert should_generate_audio_output(speech.omni_data.stage_payload), "the fixture must look like a speech request"
    batch = lambda *reqs: SimpleNamespace(reqs=list(reqs))  # noqa: E731
    for on in (runner("1"), runner("nosnapshot")):
        assert on.lookahead_eligible(batch(speech, request(speech=False))), "speech should be eligible with the switch on"
        assert not on.lookahead_eligible(batch(speech, request(speech=False, logprob=True))), "logprob must stay synchronous"
        assert not on.lookahead_eligible(batch(request(speech=True, seed=0))), "a sampling seed must stay synchronous"
        assert not on.lookahead_eligible(batch(request(speech=False, penalty=1.05))), "a repetition penalty must stay synchronous"
    assert not runner("0").lookahead_eligible(batch(speech)), "with the switch off speech is rejected as upstream does"


def check_snapshot() -> None:
    graph_buffer = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    with mock.patch.object(ThinkerModelRunner, "post_decode_launch", lambda self, result, forward_batch, requests: "launch-buffer"):
        for mode in ("1", "nosnapshot", "0"):
            result = SimpleNamespace(logits_output=SimpleNamespace(hidden_states=graph_buffer[:2]))
            assert runner(mode).post_decode_launch(result, None, []) == "launch-buffer"
            seen = result.logits_output.hidden_states
            graph_buffer.add_(100)  # the next launch replays the graph into the same buffer
            if mode == "1":
                assert torch.equal(seen, torch.arange(8, dtype=torch.float32).reshape(2, 4)), seen
            else:
                assert seen.data_ptr() == graph_buffer.data_ptr(), f"{mode}: resolve should read the live buffer"
            graph_buffer.sub_(100)


def check_torch_skip() -> None:
    started: list[str] = []
    recorded: list[str] = []
    recorder = SimpleNamespace(start=lambda run_id, event_dir, stage: recorded.append(stage))
    message = SimpleNamespace(run_id="run", enable_torch=True, trace_path_template="/traces/{stage}_trace", event_dir="/events")
    names = ["code2wav@r0", "talker@r1", "thinker@r0", "preprocessing@r0"]
    with mock.patch.dict(os.environ, {"SGLANG_OMNI_EXP_TORCH_SKIP": "code2wav,talker"}), \
            mock.patch.object(stage_runtime.TorchProfiler, "is_active", lambda: False), \
            mock.patch.object(stage_runtime.TorchProfiler, "start", lambda template, run_id=None: started.append(template)), \
            mock.patch.object(stage_runtime, "_get_recorder", lambda: recorder):
        for name in names:
            stage = object.__new__(stage_runtime.Stage)
            stage.name = name
            stage.on_profiler_start(message)
    assert [t.split("/")[-1].split("_trace")[0] for t in started] == ["thinker@r0", "preprocessing@r0"], started
    assert recorded == names, recorded


def check_digest() -> None:
    assert exp_trace.digest([1, 2, 3]) == exp_trace.digest([1, 2, 3]) != exp_trace.digest([1, 2, 4])
    assert len(exp_trace.digest([])) == 16
    hidden = torch.randn(3, 8).to(torch.bfloat16)
    assert exp_trace.digest(hidden) == exp_trace.digest(hidden.clone()) != exp_trace.digest(hidden + 1)
    assert exp_trace.digest(hidden.float()) != exp_trace.digest(hidden), "the dtype is part of what is compared"


def main() -> int:
    check_modes()
    check_eligibility()
    check_snapshot()
    check_torch_skip()
    check_digest()
    print("prototype: modes 0/1/nosnapshot; speech eligible only when on; other rejections kept; only the snapshot mode "
          "keeps each step's hidden states; torch profiler skipped for listed stages; digest stable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
