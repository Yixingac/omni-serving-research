# Final experiment results (pre-presentation validation)

> Branch `research/pre-presentation-validation` (parent `4e0f105`). Every section follows Research Question → Hypothesis → Setup → Evidence → Result → Interpretation → Limitations, in English, with a short Chinese summary at the end. Only experiments that actually ran are reported here; status, GPU-hours and run IDs of every attempt (failed ones included) are in `experiment_manifest.md`.
> Common conditions unless stated: MiniCPM-o 4.5 (revision `fb09628e`), SGLang-Omni `520b49ee` (SGLang 0.5.21, torch 2.13.0), one node with 8× RTX A6000 48 GB, Slurm job 454290, synthetic inputs. Evidence levels: Measured, Derived, Estimated, Hypothesis, Unknown, Blocked.

---

## R0 — Full-duplex functional validation

**Research question.** Can MiniCPM-o 4.5 deliver continuous, real-time audio responses through SGLang-Omni's native full-duplex endpoint (`/v1/realtime`) on one RTX A6000?

**Hypothesis (registered, `preregistration.md` R0–R2, 22:53 before the run).** Functional pass requires all five conditions: (1) the server starts and reports `native_full_duplex`; (2) one- and two-session phases complete 300 s with every planned packet sent and a normal close after the input drains; (3) no fatal error, server exit or restart; (4) every session receives at least one reply with decodable, non-silent audio; (5) at least 95 % of planned 1-s units can be matched to their `sglang.unit.done`. A unit miss rate (lag > 1 s or no `unit.done`) above 1 % turns a pass into "runs but not real-time".

**Setup.** Run `ppv_r0_duplex_functional` (`20261118_ppv_r0_duplex_functional_run01`), 23:14–23:27, 1 GPU (Slurm step of job 454290). Pinned stock runtime (no research patch), launched with `--variant session --enable-realtime` and defaults: four stages (perception 0.12, thinker 0.52, talker 0.15, speech 0.15 of the GPU) on one A6000, `max_sessions` 2, demo sampling (temperature 0.7). Input: the `duplex-v1` pool — 24 synthetic spoken questions (the Qwen3-TTS clips of the existing synthetic-v1 assets) resampled to 16 kHz and trimmed, SHA-256 per clip; each session's fixed timeline is 2 s silence → question → 12 s answer window of silence → next question; seed 20261119; 80 ms packets sent on an absolute schedule from a client on the same node. Phases: 30 s smoke, 1 session × 300 s, 2 sessions × 300 s; no server request events (lowest overhead). Clocks: client intervals use the client's monotonic clock only.

**Evidence.**

| Metric (definition in `preregistration.md`) | 1 session | 2 sessions |
|---|---|---|
| Sessions completed / failed | 1 / 0 | 2 / 0 |
| Errors / fatal errors / server restarts | 0 / 0 / 0 | 0 / 0 / 0 |
| 1-s units planned / missed (miss rate) | 300 / 0 (0 %) | 600 / 0 (0 %) |
| Unit lag (last input sample sent → `unit.done`) P50 / P90 / P99 / max | 75 / 397 / 479 / 879 ms | 75 / 413 / 517 / 695 ms |
| Units in which the model spoke / listened / text only | 111 / 187 / 2 | 235 / 360 / 5 |
| User turns answered with audio / no response | 12 / 2 (of 14) | 19 / 9 (of 28) |
| TTFA (end of the user's utterance → first assistant audio delta) P50 / P90 / max | 1.46 / 2.10 / 2.51 s (n = 12; P95, P99 N/A) | 1.66 / 2.44 / 2.57 s (n = 19; P95, P99 N/A) |
| Reply first-audio service time (unit ready → its first audio delta) P50 | 391 ms (n = 14) | 388 ms (n = 23) |
| Turns missing a 0.5 / 1 / 2 s TTFA deadline (no response counts as a miss) | 100 / 71 / 29 % | 100 / 86 / 43 % |
| Audio delta spacing within a reply P50 | 994 ms | 992 ms |
| Replies with a playback stall at 0 / 80 / 250 ms start-up buffer | 13 / 4 / 0 of 14 | 20 / 10 / 1 of 23 |
| Assistant audio received | 113.8 s | 238.6 s |
| GPU utilization (nvidia-smi, 0.2 s samples) mean; memory max | 4.4 %; 32.7 GB | 11.3 % (71 % of samples at 0); 33.9 GB |
| Server process-tree CPU mean (100 = one core) | 23 % | 44 % |
| Client packets later than 80 ms behind schedule | 28 of 3,750 (max 1.19 s) | 43 of 7,500 (max 0.75 s) |

Verdict (`scripts/analysis/ppv_r0_verdict.py`, registered rules): all five conditions hold and the unit miss rate is 0 % in both phases → **pass, real-time at 1 and 2 sessions**. Sources: run directory above (cluster), summary JSON `results/r0_summary.json`.

**Result.** The native full-duplex path runs on a single A6000 with the pinned version: every 1-s unit of every session was processed within its own second (P99 unit lag ≤ 0.52 s), audio replies were valid, and nothing failed over 10 minutes of streamed speech. The GPU was mostly idle (4–11 % sampled utilization).

**Interpretation.** On this path the first spoken response is not limited by compute: a speaking unit's audio is ready ~0.39 s after the unit's input is complete, yet TTFA from the end of a question is ~1.5 s at the median and only 29 % (1 session) of turns meet a 1-s deadline. The gap comes from the unit structure — speech is cut into 1-s units, so the system first waits for the unit that contains the end of the utterance, and the model often listens for one or more further units before it decides to speak (187 of 300 units were listening units with one session). Output arrives once per unit (deltas every ~1 s), so a player needs a start-up buffer of roughly 250 ms to avoid stalls. Going from 1 to 2 sessions left unit lag unchanged at the median but raised TTFA (P50 1.46 → 1.66 s) and the share of unanswered turns (2/14 → 9/28); whether this is load or model turn-taking is the question of R1/R2. This path differs from the request-response path measured in 实验 1–19 (different pipeline configuration, FP32 flow without compile in the speech stage, streaming sessions), so these numbers are not comparable to the chat-path TTFA without the semantic caveats in R1.

**Limitations.** Synthetic TTS questions with fixed pauses, not real users or a robot; one run (functional validation, not a latency study); TTFA samples (12 and 19) are too few for P95/P99; demo sampling (temperature 0.7) makes turn-taking stochastic; the client shares the node with the server; 0.75 % of client packets were sent > 80 ms late (max 1.19 s), which the unit-lag metric accounts for (it uses actual send times) but which made the server's input bursty at those moments; `response.cancel` does not exist at the pin, so interruption can only be studied as the model stopping by itself.

**中文小结。** 固定版本的原生全双工在一张 A6000 上能稳定运行：1 个和 2 个会话各 5 分钟，900 个 1 秒单元全部在当秒内处理完（错过率 0%），没有错误，音频有效，GPU 大部分时间空闲。但“话说完到开口”的中位数约 1.5 秒，1 秒内开口的回合只有约 3 成——时间主要花在等单元切出和模型多听的那几个单元上，而不是计算（一个开口单元的音频约 0.39 秒就准备好）。这条路径和请求-响应路径是不同的流水线，数字不能直接比较。

---

## R1 — What determines the latency of the first spoken response (single session)

**Research question.** What determines the latency of the first spoken response of a full-duplex session on one A6000, and how does it relate to the request-response path?

**Hypothesis / registered expectation (Estimated, not a decision rule; `preregistration.md` R1).** Unit lag P50 0.3–0.9 s; duplex TTFA ≈ wait for the unit boundary (0–1 s) + unit processing + k × 1 s for k extra listening units; chat-path TTFA ≈ end-to-end, 2–3 s on one GPU. Analysis rules (TTFA decomposition, stage shares on speaking units, event-overhead flag) were registered at 23:51, before any R1 data was analysed.

**Setup.** Three blocks (runs `ppv_r1_duplex_block{1,2,3}`, 23:41–00:24, 1 GPU each, chained so that no two latency runs overlapped; block 1 overlapped the C1/C2 microbenchmarks on other GPUs — deviation D1). Each run: 30 s smoke, then one session for 300 s without request events (phase A, primary) and one with server request events (phase B, stage timing), order reversed in block 2; every third question is followed 3 s later by an interrupting question. Same pinned stock runtime and inputs as R0. Request-response reference `ppv_r1_chat_ref_block{1,2,3}` on the same single A6000 (default speech pipeline with code2wav in its own process): the same 24 questions as speech input, speech output, `max_tokens` 64 and "answer in one short sentence", temperature 0, concurrency 1, 48 requests per phase.

**Evidence** (pooled over 3 blocks, phase A unless stated; `results/r1_summary.json`).

| Metric | Full duplex (1 session) | Request-response reference (1 request at a time) |
|---|---|---|
| Turns / answered with audio / no response | 48 / 37 / 11 | 144 requests, 144 ok |
| TTFA P50 (block P50s, t 95 % CI) | 1,361 ms (1,373 / 1,169 / 1,373; 1,013–1,597) | 724 ms (726 / 709 / 735; 691–756) |
| TTFA P95 (N) | 1,708 ms (37); P99 N/A | 1,112 ms (144); P99 1,799 ms |
| Reply length P50 | 9.6 s of audio, 107 characters | 4.1 s of audio, 52 characters |
| Unit lag P50 / P99 (N = 900) | 72 / 429 ms; 0 of 900 units missed | — |
| Speaking-unit service time (unit ready → first audio delta) P50 | 353 ms (N = 41) | — |
| Audio delta spacing P50 | 991 ms | one chunk per request |
| Replies stalling at 0 / 80 / 250 ms start-up buffer | 36 / 16 / 0 of 41 (buffer needed P50 65 ms, P90 165 ms) | — |
| Interruptions (scripted / valid) and stop latency P50 / max | 12 / 12; the interrupted reply kept playing 5.06 s (P90 11.3 s, max 11.6 s) | not applicable |

TTFA decomposition per answered turn (registered definition; the three parts add up to TTFA exactly; N = 37, no early starts):

| Part | P50 |
|---|---|
| (a) wait until the unit holding the end of the utterance is complete | 319 ms |
| (b) extra listening units k before the first speaking unit | k = 0 in 17 turns, k = 1 in 20 turns (1 unit ≈ 0.96 s) |
| (c) speaking unit's service time | 350 ms |
| TTFA when k = 0 / k = 1 | 862 ms (n = 17) / 1,472 ms (n = 20) |

Server-side stage shares of a speaking unit's processing time (phase B, 381 speaking units, unit time P50 351 ms): thinker 41 % (queue + service P50 142 ms), talker 24 % (88 ms), speech/vocoder 24 % (82 ms), perception 5 % (17 ms), hand-offs 3 %, other 2 %. Request events did not visibly change the client-side unit lag (B vs A P50 +4.9 %, below the registered 10 % flag). Deviation D1 check: block 1 vs blocks 2–3 unit lag P50 +1.7 %, P90 +0.5 %, TTFA P50 +8 % (within the block-to-block spread; the B phase shows the opposite order).

**Result.** With one session, the full-duplex path always keeps up (no missed units) and starts speaking 1.36 s (median) after the user stops talking. Compute is not the dominant term: a speaking unit is processed in ~0.35 s, of which the thinker is the largest stage (41 %). The rest is the unit structure and turn-taking: on average ~0.32 s waiting for the current 1-s unit to fill, plus a full extra unit in 54 % of answered turns, in which the model keeps listening. When the model does not wait an extra unit, TTFA is 0.86 s. Interruptions are not handled promptly at the pin: an interrupted reply continued for a median 5.1 s, because `response.cancel` does not exist and the model stops only by its own decision.

**Interpretation.** For the first spoken response, the lever is the unit size and the turn-taking decision, not kernel speed: the k = 1 turns cost ~0.6 s more than k = 0 turns at the median, which is larger than the entire speaking unit's compute. Within compute, the thinker matters most, so compiler work on the vocoder alone (C1/C2) can shorten at most ~24 % of 0.35 s per unit on this path. The request-response reference is faster at concurrency 1 (0.72 s) but answers are half as long, it has no end-of-speech detection in the measurement (a real application would add a voice-activity timeout), and it returns all audio at once; the comparison shows the two paths have different latency structures, not that one implementation is faster.

**Limitations.** One session; synthetic questions with fixed pauses; temperature 0.7 makes k stochastic; interruptions are scripted at fixed offsets; stage timing only in phase B (with request events); TTFA P99 is N/A (N = 37); the reference differs in pipeline (FP16 flow + compile vs FP32 flow without compile in the duplex speech stage), answer length, sampling and the starting point of TTFA.

**中文小结。** 一个会话时，全双工从“话说完”到“开口”的中位数是 1.36 秒，单元处理始终跟得上（900 个单元 0 个错过）。拆开看：等当前 1 秒单元凑满约 0.32 秒，一半以上的回合模型还要再多听一个单元（约 1 秒），真正的计算只要约 0.35 秒（其中 thinker 占 41%，talker、语音合成各约 24%）。模型不多听的回合 TTFA 只有 0.86 秒。所以首段语音延迟的主要杠杆是单元大小和轮次决策，不是 kernel。打断处理是明显的短板：固定版本没有取消接口，被打断的回答中位数还要继续说 5 秒。请求-响应路径一次一个请求时 TTFA 0.72 秒，但回答只有一半长、也没算端点检测，不能说它“更快”。

---

## C1 — Is packed variable-length flow attention faster on our real batch shapes?

**Research question.** Does the flow DiT's packed variable-length path (with and without block compilation) reduce code2wav GPU time per useful audio second (E) on the batch shapes our serving runs actually produced, without degrading audio?

**Hypothesis and gates (registered in `compiler_runtime_opportunities.md` §4 and `c1_c2_design.md` §5 before the run; Erratum E-C1 registered before the run).** Performance: P1 (packed + `enable_flow_block_compile`) lowers E by ≥ 5 % on W3b or MIX representative shapes in every one of 3 timed passes. Quality first: waveform lengths correct and finite; mean MCD against an FP32 reference no more than 0.1 dB above the dense path's; ASR exact matches at most 1 fewer and pooled WER at most +0.5 points. Prediction (Estimated): P0 5–25 % slower than D; P1 −5..−12 % on W3b and −10..−20 % on MIX; P1 likely to fail the MCD rule because the packed path is BF16 by design.

**Setup.** Run `ppv_c1_c2_microbench`, 23:40–00:53, 2 GPUs (C1 arms one after another on one GPU, C2 on the other, then the ASR quality pass); it overlapped R1 block 1 on a third GPU (deviation D1). Only the code2wav flow (+ HiFT for E) is loaded, as the component does, from the pinned stock checkout. Inputs: 47,484 real batches exported from the code2wav batch events of 实验 9 and 10 (`research/next_steps/results/c1_real_shapes.json`), replayed group by group with the same token inputs in every arm. Arms: REF (FP32 dense, quality reference only), D (default: FP16 dense masked SDPA, compiled DiT, dense CUDA graphs), D_ng (D without graphs), P0 (packed varlen, no compile), P1 (packed + block compile), P2 (P0 + packed graphs), P1h (P1 with the packed path in FP16 instead of BF16; exploratory, not the registered candidate). Timing: CUDA events, 3 timed passes after a warm-up (compile and capture time reported separately), recompiles counted; a separate short profiler pass for kernel families. Capability probe on sm_86: `varlen_attn` (FlashAttention-2) runs in BF16 and FP16; SDPA with a boolean mask cannot use the flash backend (falls back to mem-efficient).

**Evidence** (`results/c1_microbench.json`, `results/c1_quality.json`). E = code2wav GPU ms per useful audio second (flow + HiFT); mean of 3 passes (pass-to-pass spread ≤ 0.3 %):

| Real batch-shape group (source) | D | P0 | P1 | P1h | P1 vs D |
|---|---|---|---|---|---|
| W3b, fifo16, code2wav saturated (实验 10) | 8.52 | 11.73 | 6.47 | 6.47 | **−24.1 %** (−24.1, −24.1, −24.0 per pass) |
| MIX, fifo16, code2wav saturated (实验 10) | 11.21 | 11.13 | 6.55 | 6.58 | **−41.5 %** |
| W3b, length8 (实验 10) | 7.50 | 11.86 | 6.52 | — | −13.0 % |
| MIX, length8 (实验 10) | 8.40 | 10.98 | 6.30 | — | −25.0 % |
| W3L, 2+2+4 closed loop (实验 9) | 8.76 | 10.58 | 6.36 | — | −27.4 % |
| W3b, 2+2+4 open loop, ~1.2 requests per batch (实验 10 validation) | 8.62 | 8.81 | 8.56 | — | −0.8 % |

D_ng (no CUDA graphs) is only +3.3–3.6 % above D. Kernel families (ms per audio second): attention 1.43 → 0.59 (−59 %), mask preparation 0.60 → 0, GEMM 1.94 → 1.69, Inductor-fused +0.22. Requests on a CUDA graph: ≤ 3 % in batched groups for every arm. No recompilation in timed passes; peak memory unchanged (5.5 vs 5.1 GiB on W3b fifo16).

Quality against the FP32 reference (26 segments; MCD on MFCC-style cepstra, `src/omni_research/analysis/audio_quality.py`; ASR Qwen3-ASR-1.7B):

| Arm | MCD vs REF (dB) | Δ vs D | ASR rule | Registered quality gate |
|---|---|---|---|---|
| D | 4.06 | — | — | — |
| D_ng (same math as D, no graphs) | 4.06 | +0.00 | **fails** | fails (ASR) |
| P1 (registered candidate, BF16) | 5.07 | **+1.01** | passes | **fails (MCD)** |
| P0 / P2 (BF16) | 5.00 / 5.01 | +0.94 / +0.95 | passes | fails (MCD) |
| P1h (FP16, exploratory) | 4.12 | +0.06 | passes | passes |

The noise floor of these comparisons is high: two numerically equivalent dense runs (D vs D_ng) already differ by 4.09 dB MCD and relative RMS ≈ 1.2, because the sampled excitation differs between runs; only differences relative to D against the same reference are interpreted. At the mel level the packed BF16 path deviates 0.36 % (relative RMS) from REF versus 0.06 % for D. The original literal rule (P1 within D's own repeat difference) is computed as registered: D repeats bit-identically, so the literal bound is 0 and P1 fails it too.

**Result.** Performance gate: met (P1 −24 % on W3b and −42 % on MIX real shapes, identical in all three passes). Quality gate: **not met** for the registered candidate P1 (MCD +1.01 dB from the BF16 packed path). Per the registered "correctness first" rule, no performance conclusion is drawn for P1 and no end-to-end run was made (the conditional dense/packed × FIFO/length P1 variant was not placed). The exploratory FP16 packed arm P1h kept quality within every registered margin (MCD +0.06 dB) at the same speed (−24 % / −41 %).

**Interpretation.** Packing removes the masked attention and its mask-preparation kernels — the two largest flow-specific costs found by the kernel census — and cuts E by 13–42 % on batched real shapes, more when FIFO batching mixes lengths (MIX fifo16, the most padded case). Length-aware batching already recovers part of this (D: 11.21 → 8.40 on MIX), so the two optimizations overlap: the packed gain is −25 % on length8 batches versus −42 % on fifo16 batches. Without compilation the packed path is slower (P0 +21..+58 %), and packed CUDA graphs add nothing (P2 ≈ P0); CUDA graphs matter little anywhere here (D_ng +3 %), because batched shapes rarely match a captured graph. On the open-loop 2+2+4 service below saturation (≈1.2 requests per batch) there is nothing to pack and the gain vanishes (−1 %). The quality failure is a precision choice of the pinned implementation (BF16 in the packed path), not a property of packing: the same packed path in FP16 matches the dense path's quality. A registered follow-up should test P1h end-to-end in the saturated regime.

**Limitations.** Microbenchmark only (no serving-level effect claimed); quality set of 26 TTS-generated segments re-tokenized and re-synthesized (not talker output); MCD here is an MFCC-style variant comparable only between arms; the ASR rule's margin is smaller than run-to-run sampling differences (D_ng fails it), so it is a weak discriminator; P1h is exploratory and was not the registered candidate; C1 shared the node with R1 block 1 (deviation D1), timed with CUDA events.

**中文小结。** 在真实批形状上，打包 + 编译（P1）让 code2wav 每秒语音的 GPU 时间下降 13%–42%（W3b −24%、MIX −42%，3 遍完全一致）：带 mask 的注意力少了 59%，构造 mask 的 kernel 完全消失；不编译的打包反而更慢，CUDA graph 在批处理下几乎没用（关掉只慢 3%）；开环未饱和、每批约 1.2 个请求时没有东西可打包，收益接近 0。但登记的候选 P1 没过音质门槛：打包路径设计上用 BF16，MCD 比稠密路径高 1.0 dB。按“正确性先行”的规则，P1 不下性能结论、不做端到端。探索性的 FP16 打包（P1h）速度相同、音质在门槛内，说明问题出在精度选择而不是打包本身——这是下一轮应该登记的端到端实验。

---

## C2 — Which part of HiFT benefits from weight-norm folding, channels-last layout or half precision?

**Research question.** Do the three compiler-level changes found by the kernel census — folding HiFT's 82 weight-norm reparametrizations, removing NCHW→NHWC layout conversion with channels-last, and running the HiFT body in half precision — reduce HiFT GPU time per useful audio second without changing the waveform?

**Hypothesis and gates (registered before the run; `c1_c2_design.md` §5).** B (fold): weights within 1 ulp and eager output bit-identical. C (layout): eager relative RMS ≤ 1e-4 and MCD ≤ 0.01 dB vs A. D (FP16 body, STFT/iSTFT/F0/SineGen kept FP32): MCD < 0.1 dB vs A and the ASR rule. Performance: B + C saving < 3 % of code2wav E → no end-to-end; D must pass quality and cut HiFT time by ≥ 20 % in every pass. Prediction (Estimated): B −3..−10 %, C −5..−15 %, D −20..−35 % of HiFT time.

**Setup.** Same run as C1, second GPU, 23:40–23:54. Each arm is a fresh `HiFTGenerator` loaded with the same weights (a deep copy would share the parametrization class — a bug caught by the CPU check before the run); fixed input mels from three real shape groups and fixed excitation noise; `decode_body` compiled as in the stock component; 3 timed passes with CUDA events; a separate profiler pass.

**Evidence** (`results/c2_microbench.json`, `results/c2_quality.json`). HiFT GPU ms per audio second, mean of 3 passes:

| Shape group | A default | B fold | C channels-last | D FP16 body |
|---|---|---|---|---|
| W3b fifo16 | 2.02 | 2.11 | 2.08 | 2.83 |
| MIX fifo16 | 1.79 | 1.81 | 1.82 | 2.37 |
| W3L 2+2+4 | 1.54 | 1.50 | 1.50 | 1.75 |

Per-pass changes vs A: B −5..+7 %, C −5..+7 % (within the ±5 % pass-to-pass spread of A), D +31..+50 % on W3b, +31..+33 % on MIX, −2..+23 % on W3L. Correctness: B folded 82 modules with 0 ulp difference and bit-identical eager output (pass); C eager relative RMS 9.1e-4 and MCD 0.18 dB (fail); D MCD 0.83 dB (fail; ASR rule passes). Kernel families (ms per audio s): C removes layout conversion (0.231 → 0.005) but its TF32 convolutions get slower (0.723 → 0.940); D halves the convolutions (0.723 TF32 → 0.295 FP16) and Snake (0.343 → 0.176). Kernel count per call: A 486, B 404, C 486, D 569. Summed kernel time is 77 % of the event-timed HiFT time for A–C but only 35 % for D (Derived from a separate profiler pass).

**Result.** None of the three changes made HiFT measurably faster; layout and half precision also failed their numerical gates. Folding is exact but saves no measurable time (−17 % kernels, no time change). By the registered stop rule (B + C < 3 % of code2wav E; D neither passes quality nor reaches −20 %), the HiFT direction stops here and no end-to-end run was made.

**Interpretation.** HiFT's cost is not dominated by any one kernel family it can shed: removing layout conversions moved time into slower convolutions, and halving the kernel time with FP16 raised the event-timed time, because HiFT runs once per request at small sizes and the GPU sits idle between its kernels for much of the call (kernel time 35 % of the call in FP16). This points at launch/host overhead and per-request invocation rather than arithmetic — a Hypothesis that batching HiFT across requests or capturing it in a graph would be the lever (not tested).

**Limitations.** Microbenchmark only; three shape groups; HiFT timing has ±5 % pass-to-pass spread; kernel sums and event times come from different passes; the FP16 arm converts the whole body including weight-norm parametrizations and uses dynamic-shape compilation, so part of its overhead may be specific to this implementation; overlapped R1 block 1 (D1).

**中文小结。** HiFT 的三项编译期优化都没有带来可测的加速：weight-norm 折叠逐位正确但时间没变；channels-last 去掉了布局转换，卷积却变慢，而且数值差异超过门槛；FP16 把 kernel 时间减了 42%，音质没过门槛（MCD 0.83 dB），总时间反而多了 30%——因为 HiFT 每个请求单独调用、kernel 很小，GPU 有大量时间在等主机端。按登记的放弃条件，这个方向停止。如果还要优化 HiFT，杠杆更可能是跨请求批量调用或给 HiFT 录 graph（假设，未测）。
