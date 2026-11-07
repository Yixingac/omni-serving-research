# Characterizing and Optimizing Omni Serving Across Heterogeneous Workloads

**Status: Proposed research; all experiments below are planned.** No measurements, benchmarks, bottleneck findings, or optimization results have been produced for this project. The eight-week schedule starts when experimental work begins and depends on hardware availability and runtime compatibility.

## Motivation and scope

Omni serving connects stages with different computational and timing requirements. Audio/video understanding, language generation, and speech generation may stress different resources as input length, output length, streaming cadence, and concurrency change. We will first characterize these workloads, then use profiling evidence to decide whether scheduling, disaggregation, or resource allocation changes are justified.

The goal is to explain hardware utilization, stage bottlenecks, GPU idle bubbles, and load imbalance under reproducible conditions. Robotics and embodied AI motivate additional questions about response deadlines, observation freshness, and continuous interaction. Initial experiments will use public or synthetic multimodal workloads; they will not establish robot-control safety or closed-loop task performance.

Encoder, Thinker, Talker, and Vocoder are profiling categories, not an assumption that every model exposes the same four modules. Before profiling, map these categories to each model's actual graph, including modality encoders, language-model prefill/decode, speech-token generation, and waveform synthesis. Record absent, fused, or differently named stages and their dependencies.

## Research questions

1. Under what workload conditions do the Encoder, Thinker, Talker, or Vocoder become bottlenecks?
2. How do input modality, context length, output length, streaming behavior, and concurrency affect performance?
3. Where do GPU idle bubbles, resource contention, and load imbalance occur, and what causes them?
4. How do 1, 2, and 4 GPUs affect latency, throughput, utilization, and scaling efficiency?
5. When should execution stages be colocated or disaggregated?
6. Can existing LLM serving techniques improve Omni serving while preserving output quality and streaming behavior?
7. What additional constraints arise for real-time multimodal and robotics workloads?

## Initial experimental environment

| Component | Planned starting point |
| --- | --- |
| Models | MiniCPM-o 4.5 first; a Qwen Omni variant as a possible comparison |
| Runtimes | vLLM-Omni and/or SGLang-Omni; prioritize one working model/runtime pair before cross-runtime comparisons |
| GPUs | NVIDIA RTX A6000 48GB; 1, 2, and up to 4 GPUs when available |
| Profiling | Nsight Systems for timelines; PyTorch Profiler for operator attribution; Nsight Compute for selected kernels after identifying a relevant bottleneck |
| Inputs | Publicly available data with suitable usage terms, or explicitly labeled synthetic inputs |

First verify the selected checkpoint, runtime revision, supported modalities, streaming path, GPU architecture support, memory fit, and profiler access. Runtime support for a model does not establish support for every stage placement or streaming mode. Record compatibility failures and unavailable configurations rather than treating them as performance results. Choose the Qwen variant after checking these constraints.

Record GPU topology and actual interconnects, CPU and host memory, driver/CUDA/PyTorch versions, runtime commits, model revisions, precision, parallelism, and stage placement. Do not assume that four GPUs have uniform communication bandwidth. Keep model weights, datasets, and raw traces outside the repository unless an artifact has been reviewed for public release.

## Workload characterization and measurement

Start with a small set of representative cases, then vary one factor at a time before testing selected mixed workloads. Avoid a full Cartesian product of all settings.

| Workload family | Planned variations | Main questions |
| --- | --- | --- |
| Audio understanding with text output | Audio duration, input chunk size, prompt length, output token count | Encoder cost versus language prefill/decode and queueing |
| Video or audio/video understanding | Clip duration, frame sampling rate, resolution, synchronized audio, context length | Input processing, memory pressure, and inter-stage transfer |
| Speech generation and speech-to-speech | Text/audio input length, generated speech duration, output chunk size | Speech-token generation, waveform synthesis, and time to first audio |
| Long-context multimodal interaction | Conversation history, modality mix, context and output lengths | Prefill/decode balance, cache pressure, and interference with short requests |
| Real-time streaming interaction | Input cadence, concurrent streams, interruptions where supported, bursty arrivals | Deadline misses, audio underruns, jitter, and backpressure |

For each case, record modality combinations, input/output lengths in their native units, chunk size and cadence, request concurrency, and arrival process. An initial concurrency sweep may use 1, 2, 4, and 8 in-flight requests, stopping at memory or latency limits; these are proposed settings, not measured capacities. Select numerical length and chunk ranges after checking model limits and representative public workloads. Synthetic stress cases must be distinguished from representative application traffic.

Use fixed concurrency to establish basic capacity, then controlled arrival-rate and burst tests to expose queueing. Report offered load, completed requests, failures, timeouts, and cancellations. Keep workload traces and generation settings fixed across placements. If output lengths differ, report the actual lengths and compare matched subsets or controlled-length runs.

### Metrics and definitions

| Metric | Measurement plan |
| --- | --- |
| End-to-end latency | Client request submission to final output receipt for finite requests; report connection setup and cold start separately |
| TTFT | Client submission to first text token received; mark not applicable for paths without text output |
| Time to first audio | Client submission to first playable audio chunk received; record chunk duration and any buffering |
| Streaming response latency | For continuous input, timestamp the relevant input event or chunk and corresponding output; distinguish session-start timing from response after end-of-turn |
| Stage execution and queue waiting | Per-request/per-chunk stage-ready, enqueue, start, and finish events; separate host dispatch from GPU execution |
| GPU utilization and idle bubbles | Per-device utilization samples plus timeline intervals without GPU work; correlate gaps with queues, CPU work, transfers, and synchronization |
| Peak memory | Per-device peak allocated/reserved framework memory and device memory use, including weights, caches, and buffers |
| Communication overhead | Bytes transferred, transfer time, synchronization waits, and exposed critical-path cost between stages |
| Throughput and scaling efficiency | Completed requests/s for a fixed workload, with text tokens/s or generated audio seconds/s where useful; efficiency = throughput on N GPUs / (N × single-GPU throughput) under matched conditions |
| Tail latency and streaming jitter | P50/P95/P99 latency, inter-chunk gap distribution, deviation from intended playout cadence, and underrun/deadline-miss rate |

Do not infer compute efficiency from a utilization percentage alone. Account for overlapping stages when reconstructing the critical path; summing stage durations may exceed wall time. If the model or workload cannot run on one GPU, report that limitation and use a clearly named smallest-feasible baseline instead of claiming single-GPU scaling efficiency.

Separate warm-up from steady-state measurement, repeat runs, and report sample counts, run duration, and run-to-run variation. Tail percentiles require enough observations; label unstable estimates rather than presenting them as reliable. Measure the unprofiled baseline separately because instrumentation can perturb latency. Use synchronized or calibrated clocks for cross-process timing. Preserve output validity and check for truncation or dropped chunks when comparing performance.

## Eight-week roadmap

### Milestone 1 — Workload Characterization (Weeks 1–2)

- Review representative Omni architectures and serving runtimes using public sources. Map each selected model's execution stages and streaming dependencies.
- Start with MiniCPM-o 4.5; consider Qwen Omni for comparison after validating runtime support and memory fit.
- Define workloads covering audio/video understanding, speech generation, long context, and real-time streaming.
- Characterize input/output length, modality combinations, chunk size/cadence, request concurrency, and arrival patterns.
- Establish reproducible workload configurations and unprofiled baseline benchmarks on the first feasible GPU configuration. Record versions, seeds, warm-up, repetitions, and limitations.

**Deliverables:** Workload taxonomy, benchmark configurations, and baseline measurements for configurations that can be run. If setup is blocked, document the blocker and report no measurement for that configuration.

**Decision point:** Select a small representative workload set and one reliable model/runtime baseline for deeper profiling.

### Milestone 2 — Profiling and Bottleneck Analysis (Weeks 3–4)

- Profile Encoder, Thinker prefill/decode, Talker, and Vocoder according to the model-specific stage mapping.
- Measure stage execution, queue waiting, CPU preprocessing/dispatch overhead, memory use, and inter-stage communication.
- Identify GPU idle bubbles, contention, and load imbalance by correlating timelines with request and chunk events.
- Compare available 1/2/4 GPU configurations, recording placement, replication, parallelism, and topology. Hold workload and generation settings constant.
- Analyze when bottlenecks shift as modality, length, streaming cadence, and concurrency change. Distinguish unavoidable dependency gaps from potentially avoidable scheduling gaps.

**Deliverables:** Profiling timelines, performance breakdowns, and documented bottleneck scenarios with reproduction configurations and alternative explanations.

**Decision point:** Identify which stage or interaction limits the critical path or sustainable throughput; do not choose an optimization from utilization alone.

### Milestone 3 — Disaggregation and Resource Allocation (Weeks 5–6)

- Compare stage colocation and disaggregated execution at equal total GPU budgets where the runtime supports both.
- Investigate GPU placement, selective stage replication, and supported parallelism choices. Begin with a few configurations justified by Milestone 2.
- Measure communication, synchronization, scheduling, and buffering overhead, including transfer costs exposed on the critical path.
- Compare static allocation with a simple workload-aware allocation policy or offline allocation study; document any runtime changes needed for dynamic reassignment.
- Explore whether stage isolation or dedicated capacity could help deadline-sensitive multimodal and robotics-inspired streams. Include observation freshness, interruption handling, and backpressure in the analysis where supported.

**Deliverables:** Disaggregation trade-off analysis and candidate resource allocation strategies, including cases where colocation remains preferable.

**Decision point:** Determine whether measured overlap, isolation, or balancing benefits justify communication and coordination costs. An unsupported split is a feasibility limitation, not evidence that disaggregation is slower.

### Milestone 4 — Optimization Exploration (Weeks 7–8)

- Select the most promising bottleneck from the profiling and placement evidence.
- Assess existing LLM inference techniques: continuous batching, chunked prefill, stage-aware scheduling, caching, and computation/communication overlap. Document which mechanisms the runtime already provides before proposing changes.
- Prototype one focused optimization if feasible, with an explicit hypothesis and a matched baseline. Submit implementation work in a separate PR.
- Evaluate throughput, latency, GPU efficiency, memory cost, and streaming jitter. Include output-validity checks, an ablation, and workloads where the optimization may regress.
- Summarize limitations and follow-up research opportunities, including real-time and embodied workloads that need a richer evaluation setup.

**Deliverables:** An optimization proposal or focused prototype, evaluation results where available, and next-step recommendations. A proposal without a completed prototype or evaluation must be labeled as such.

**Decision point:** Continue only if evidence supports the claimed benefit; report negative results and unresolved trade-offs.

## Review questions

- Which workload families and input/output ranges best represent useful applications, and which should be deferred?
- Is the stage-level profiling scope reasonable for eight weeks? Which measurements are essential first?
- Which model/runtime pair should be prioritized, and is a Qwen comparison worth the setup cost?
- Which stage boundaries offer plausible disaggregation opportunities given A6000 memory and interconnect limits?
- Which robotics or real-time multimodal scenarios provide meaningful deadlines, freshness requirements, and streaming behavior for this study?

## Public references and project workflow

Starting points for architecture and compatibility review:

- [MiniCPM-o 4.5 model documentation](https://github.com/OpenBMB/MiniCPM-V/blob/main/docs/minicpm_o4dot5_en.md)
- [Qwen3-Omni repository](https://github.com/QwenLM/Qwen3-Omni)
- [vLLM-Omni documentation](https://docs.vllm.ai/projects/vllm-omni/en/stable/) and [supported models](https://docs.vllm.ai/projects/vllm-omni/en/latest/models/supported_models/)
- [SGLang-Omni repository](https://github.com/sgl-project/sglang-omni)

These sources guide the initial review; their feature lists are not evidence that this project's configurations have been validated. Pin exact revisions when experiments begin.

This PR introduces only the research plan. Add benchmark configurations, experiment results, and implementations through separate, focused PRs as work begins. Clearly separate planned, attempted, and completed work. Publish only original project material and public technical references; exclude private lab code, credentials, internal discussions, unpublished research artifacts, and non-public datasets. No project license is selected by this proposal.
