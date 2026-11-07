# Omni Serving: Workloads, Bottlenecks, and Disaggregation

## Goal

Omni models combine audio/video encoding, language generation, and speech generation. These stages do not necessarily use the GPU in the same way. A stage may be overloaded while another GPU sits idle, and the bottleneck may change with the workload.

I want to understand **when Encoder, Thinker, Talker, or Vocoder becomes the bottleneck**, where time is lost between stages, and whether better scheduling or stage placement can help. The longer-term motivation is real-time multimodal serving, including robotics, where latency, jitter, and the age of an observation matter as much as throughput.

I'll start by measuring the current system rather than assuming disaggregation is the right answer.

## What to investigate

- **Workloads:** audio/video understanding, speech-to-speech, long-context conversations, and continuous streaming. Vary input/output length, chunk size, arrival rate, and concurrency.
- **Stage bottlenecks:** break down Encoder, Thinker prefill/decode, Talker, and waveform generation. Map these names to the actual stages of each model.
- **GPU imbalance and bubbles:** look for idle GPUs, queueing, CPU dispatch gaps, memory pressure, and stage-to-stage transfer or synchronization costs.
- **Disaggregation:** compare colocated and split stages on 1, 2, and up to 4 GPUs, including placement and replication where supported.
- **Optimization opportunities:** reuse ideas from LLM serving—batching, chunked prefill, caching, and scheduling—where the profiles suggest they might help.

## Plan: four two-week milestones

| Weeks | Focus | Output |
| --- | --- | --- |
| **1–2** | **Understand the workloads.** Pick representative scenarios, check the model/runtime setup, and run unprofiled baselines. | Workload definitions, reproducible runs, baseline numbers. |
| **3–4** | **Find the bottlenecks.** Trace each stage and compare workloads and GPU configurations. Identify critical-path delays, GPU bubbles, and load imbalance. | Nsight timelines and a few reproducible bottleneck cases. |
| **5–6** | **Study disaggregation.** Compare colocation, stage placement, and replication at matched GPU budgets. Measure communication and queueing overhead. | Cases where splitting helps or hurts, with an explanation. |
| **7–8** | **Try one targeted improvement.** Pick a measured bottleneck and test a scheduling, batching, caching, or overlap idea. | A small prototype or concrete proposal, with results if implemented. |

Each milestone should lead to a short, reviewable follow-up PR. We can adjust the later stages based on what the first profiles show.

## Starting point

- **Models:** MiniCPM-o 4.5 first; Qwen Omni as a possible comparison.
- **Serving:** vLLM-Omni or SGLang-Omni. Get one model/runtime pair working before comparing frameworks.
- **Hardware:** RTX A6000 48 GB; 1, 2, or 4 GPUs depending on availability and supported placement.
- **Tools:** Nsight Systems for timelines, PyTorch Profiler for operators, and Nsight Compute for specific kernels.

**Measurements:** end-to-end latency, TTFT/first audio latency, per-stage compute and queue time, GPU idle time, memory, inter-stage transfers, throughput, P95/P99 latency, and streaming jitter. For streaming and robotics-like cases, also track missed deadlines and stale inputs where applicable.

Compare identical workloads and total GPU budgets. Measure normal latency without profiler overhead, record software versions and GPU topology, and verify that generated outputs remain valid. If a configuration cannot run, record that rather than treating it as a benchmark result.

## Feedback I'd like

1. Are these the right workloads to start with? Which ones best reflect real usage?
2. Is MiniCPM-o a good starting point, or should we prioritize another model/runtime?
3. Which stage boundaries or GPU placement choices look most interesting?
4. What real-time or robotics-like workload would make a useful follow-up?

This PR is the research plan only. Experiments and code changes will come in separate PRs. The public repo will contain only public or synthetic inputs and material cleared for release.
