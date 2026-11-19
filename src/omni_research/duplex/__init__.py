"""Full-duplex (``/v1/realtime``) measurement for the pre-presentation validation (R0, R1, R2).

- ``audio``: PCM16 WAV reading/writing, a deterministic polyphase resampler, silence trimming;
- ``script``: the input timeline one simulated user sends (questions, answer windows, barge-in turns);
- ``client``: one WebSocket session at real-time pace, recording every sent and received event;
- ``metrics``: unit lag, TTFA per turn, first-audio latency per reply, delta gaps, playback stalls,
  deadline misses, listen/speak units, all with sample counts;
- ``phase``: the runner's ``plugin`` phase (``{"kind": "plugin", "module": "duplex.phase", ...}``);
- ``fake_server``: a CPU stand-in that speaks the same protocol, for tests only (made-up timings);
- ``report``: offline summary of a run's traces (and of the server's request events when present).

The protocol facts come from sglang-omni 520b49ee (see
research/next_steps/presentation_validation/r0_duplex_source_audit.md).
"""
