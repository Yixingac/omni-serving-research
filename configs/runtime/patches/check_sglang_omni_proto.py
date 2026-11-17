"""CPU-only check of the 实验 8 prototype (sglang_omni_proto_exp08.patch): speech requests may use the thinker's
one-step lookahead when SGLANG_OMNI_SPEECH_LOOKAHEAD=1, and each launched step keeps its own hidden states.

    cd <sglang-omni checkout> && PYTHONPATH=$PWD <runtime python> <this file>

Checks, without a GPU or a model:
1. lookahead_eligible: with the switch on, a speech request is eligible; logprob, a sampling seed or a
   repetition penalty still make a batch ineligible; with the switch off, speech is rejected as upstream does.
2. post_decode_launch: with the switch on, the result handed to resolve holds a copy of the hidden states, so a
   later write into the graph's output buffer (the next launch) does not change what resolve reads; with the
   switch off it is the buffer itself.
On a checkout without the prototype the first check fails (speech is rejected with the switch on).
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.models.minicpm_o.thinker_model_runner import MiniCPMOThinkerModelRunner


def request(speech: bool, logprob: bool = False, seed: int | None = None, penalty: float = 1.0) -> SimpleNamespace:
    # should_generate_audio_output reads metadata["output_modalities"] of the request it is given.
    payload = SimpleNamespace(metadata={"output_modalities": ["text", "audio"] if speech else ["text"]})
    sampling = SimpleNamespace(repetition_penalty=penalty, presence_penalty=0.0, frequency_penalty=0.0, min_new_tokens=0,
                               sampling_seed=seed, logit_bias=None, custom_params=None)
    return SimpleNamespace(omni_data=SimpleNamespace(stage_payload=payload, return_logprob=logprob), sampling_params=sampling)


def runner(switch: bool) -> MiniCPMOThinkerModelRunner:
    r = object.__new__(MiniCPMOThinkerModelRunner)
    r.speech_lookahead = switch
    return r


def main() -> int:
    from sglang_omni.models.qwen3_omni.request_builders import should_generate_audio_output

    speech = request(speech=True)
    assert should_generate_audio_output(speech.omni_data.stage_payload), "the fixture must look like a speech request"
    on, off = runner(True), runner(False)
    batch = lambda *reqs: SimpleNamespace(reqs=list(reqs))  # noqa: E731
    assert on.lookahead_eligible(batch(speech, request(speech=False))), "speech should be eligible with the switch on"
    assert not on.lookahead_eligible(batch(speech, request(speech=False, logprob=True))), "logprob must stay synchronous"
    assert not on.lookahead_eligible(batch(request(speech=True, seed=0))), "a sampling seed must stay synchronous"
    assert not on.lookahead_eligible(batch(request(speech=False, penalty=1.05))), "a repetition penalty must stay synchronous"
    assert not off.lookahead_eligible(batch(speech)), "with the switch off speech is rejected as upstream does"

    graph_buffer = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    original = ThinkerModelRunner.post_decode_launch
    ThinkerModelRunner.post_decode_launch = lambda self, result, forward_batch, requests: "launch-buffer"
    try:
        for switch in (True, False):
            result = SimpleNamespace(logits_output=SimpleNamespace(hidden_states=graph_buffer[:2]))
            assert runner(switch).post_decode_launch(result, None, []) == "launch-buffer"
            seen = result.logits_output.hidden_states
            graph_buffer.add_(100)  # the next launch replays the graph into the same buffer
            if switch:
                assert torch.equal(seen, torch.arange(8, dtype=torch.float32).reshape(2, 4)), seen
            else:
                assert seen.data_ptr() == graph_buffer.data_ptr(), "without the switch resolve reads the live buffer"
            graph_buffer.sub_(100)
    finally:
        ThinkerModelRunner.post_decode_launch = original
    print("prototype: speech eligible only with the switch; other rejections kept; launched steps keep their hidden states")
    return 0


if __name__ == "__main__":
    sys.exit(main())
