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
| GPU power mean (idle ≈ 35 W); memory max | 143 W; 31.9 GiB (32,683 MiB) | 166 W; 33.1 GiB (33,921 MiB) |
| Server process-tree CPU mean (100 = one core) | 23 % | 44 % |
| Client packets later than 80 ms behind schedule | 28 of 3,750 (max 1.19 s) | 43 of 7,500 (max 0.75 s) |

Verdict (`scripts/analysis/ppv_r0_verdict.py`, registered rules): all five conditions hold and the unit miss rate is 0 % in both phases → **pass, real-time at 1 and 2 sessions**. Sources: run directory above (cluster), summary JSON `results/r0_summary.json`.

**Result.** The native full-duplex path runs on a single A6000 with the pinned version: every 1-s unit of every session was processed within its own second (P99 unit lag ≤ 0.52 s), audio replies were valid, and nothing failed over 10 minutes of streamed speech. nvidia-smi's utilization counter is not usable for this load: it read 0 % in 87 % (1 session) and 71 % (2 sessions) of samples while the card drew 143–166 W against ≈ 35 W idle, most likely aliasing with the strictly periodic 1-s units (a hypothesis from the R2 analysis, not verified); utilization is therefore not reported (power and stage timing are).

**Interpretation.** On this path the first spoken response is not limited by compute: a speaking unit's audio is ready ~0.39 s after the unit's input is complete, yet TTFA from the end of a question is ~1.5 s at the median and only 29 % (1 session) of turns meet a 1-s deadline. The gap comes from the unit structure — speech is cut into 1-s units, so the system first waits for the unit that contains the end of the utterance, and the model often listens for one or more further units before it decides to speak (187 of 300 units were listening units with one session). Output arrives once per unit (deltas every ~1 s), so a player needs a start-up buffer of roughly 250 ms to avoid stalls. Going from 1 to 2 sessions left unit lag unchanged at the median but raised TTFA (P50 1.46 → 1.66 s) and the share of unanswered turns (2/14 → 9/28); whether this is load or model turn-taking is the question of R1/R2. This path differs from the request-response path measured in 实验 1–19 (different pipeline configuration, FP32 flow without compile in the speech stage, streaming sessions), so these numbers are not comparable to the chat-path TTFA without the semantic caveats in R1.

**Limitations.** Synthetic TTS questions with fixed pauses, not real users or a robot; one run (functional validation, not a latency study); TTFA samples (12 and 19) are too few for P95/P99; demo sampling (temperature 0.7) makes turn-taking stochastic; the client shares the node with the server; 0.63 % of client packets (0.75 % in the 1-session phase) were sent > 80 ms late (max 1.19 s), which the unit-lag metric accounts for (it uses actual send times) but which made the server's input bursty at those moments; `response.cancel` does not exist at the pin, so interruption can only be studied as the model stopping by itself.

**中文小结。** 固定版本的原生全双工在一张 A6000 上能稳定运行：1 个和 2 个会话各 5 分钟，900 个 1 秒单元全部在当秒内处理完（错过率 0%），没有错误，音频有效（GPU 利用率读数在这种每秒一次的周期负载下不可信，改看功耗：143–166 W，空闲约 35 W）。但“话说完到开口”的中位数约 1.5 秒，1 秒内开口的回合只有约 3 成——时间主要花在等单元切出和模型多听的那几个单元上，而不是计算（一个开口单元的音频约 0.39 秒就准备好）。这条路径和请求-响应路径是不同的流水线，数字不能直接比较。

---

## R1 — What determines the latency of the first spoken response (single session)

**Research question.** What determines the latency of the first spoken response of a full-duplex session on one A6000, and how does it relate to the request-response path?

**Hypothesis / registered expectation (Estimated, not a decision rule; `preregistration.md` R1).** Unit lag P50 0.3–0.9 s; duplex TTFA ≈ wait for the unit boundary (0–1 s) + unit processing + k × 1 s for k extra listening units; chat-path TTFA ≈ end-to-end, 2–3 s on one GPU. Analysis rules (TTFA decomposition, stage shares on speaking units, event-overhead flag) were committed at 00:10 (`fbe3e73`) — after R1 duplex blocks 1–2 had run, before any R1/R2 data was analysed (the analysis code came in the same commit; the “23:51” in the registration heading cannot be verified from git).

**Setup.** Three blocks (runs `ppv_r1_duplex_block{1,2,3}`, 23:41–00:24, 1 GPU each, chained so that no two latency runs overlapped; all three duplex blocks and the three reference runs overlapped the C1/C2 microbenchmark on other GPUs — C1 arms 23:40–00:45, C2 during block 1 only — and block 1's first ~2.5 min also overlapped the crashed R3 screening attempt — deviation D1). Each run: 30 s smoke, then one session for 300 s without request events (phase A, primary) and one with server request events (phase B, stage timing), order reversed in block 2; every third question is followed 3 s later by an interrupting question. Same pinned stock runtime and inputs as R0. Request-response reference `ppv_r1_chat_ref_block{1,2,3}` on the same single A6000 (default speech pipeline with code2wav in its own process): the same 24 questions as speech input, speech output, `max_tokens` 64 and "answer in one short sentence", temperature 0, concurrency 1, 48 requests per phase.

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
| (b) extra listening units k before the first speaking unit | k = 0 in 17 turns, k = 1 in 20 turns (the extra unit adds a median 1.04 s in the k = 1 turns) |
| (c) speaking unit's service time | 350 ms |
| TTFA when k = 0 / k = 1 | 862 ms (n = 17) / 1,472 ms (n = 20) |

Server-side stage shares of a speaking unit's processing time (phase B, 381 speaking units, unit time P50 351 ms): thinker 41 % (queue + service P50 142 ms), talker 24 % (88 ms), speech/vocoder 24 % (82 ms), perception 5 % (17 ms), hand-offs 3 %, other 2 %. Request events did not visibly change the client-side unit lag (B vs A P50 +4.9 %, below the registered 10 % flag). Deviation D1 check: block 1 vs blocks 2–3 unit lag P50 +1.7 %, P90 +0.5 %, TTFA P50 +8 % (within the block-to-block spread; the B phase shows the opposite order). Blocks 2–3 also overlapped C1, so this check covers only block 1's additional C2 and R3 overlap.

**Result.** With one session, the full-duplex path always keeps up (no missed units) and starts speaking 1.36 s (median) after the user stops talking. Compute is not the dominant term: a speaking unit is processed in ~0.35 s, of which the thinker is the largest stage (41 %). The rest is the unit structure and turn-taking: a median 0.32 s (mean 0.31 s) waiting for the current 1-s unit to fill, plus a full extra unit in 54 % of answered turns, in which the model keeps listening. When the model does not wait an extra unit, TTFA is 0.86 s. Interruptions are not handled promptly at the pin: an interrupted reply continued for a median 5.1 s, because `response.cancel` does not exist and the model stops only by its own decision.

**Interpretation.** For the first spoken response, the lever is the unit size and the turn-taking decision, not kernel speed: the k = 1 turns cost ~0.6 s more than k = 0 turns at the median, which is larger than the entire speaking unit's compute. Within compute, the thinker matters most, so compiler work on the vocoder alone (C1/C2) can shorten at most ~24 % of 0.35 s per unit on this path. The request-response reference is faster at concurrency 1 (0.72 s) but answers are half as long, it has no end-of-speech detection in the measurement (a real application would add a voice-activity timeout), and it returns all audio at once; the comparison shows the two paths have different latency structures, not that one implementation is faster.

**Limitations.** One session; synthetic questions with fixed pauses; temperature 0.7 makes k stochastic; interruptions are scripted at fixed offsets; stage timing only in phase B (with request events); TTFA P99 is N/A (N = 37); the reference differs in pipeline (FP16 flow + compile vs FP32 flow without compile in the duplex speech stage), answer length, sampling and the starting point of TTFA; every R1 run overlapped the C1 microbenchmark on another GPU (D1).

**中文小结。** 一个会话时，全双工从“话说完”到“开口”的中位数是 1.36 秒，单元处理始终跟得上（900 个单元 0 个错过）。拆开看：等当前 1 秒单元凑满约 0.32 秒，一半以上的回合模型还要再多听一个单元（约 1 秒），真正的计算只要约 0.35 秒（其中 thinker 占 41%，talker、语音合成各约 24%）。模型不多听的回合 TTFA 只有 0.86 秒。所以首段语音延迟的主要杠杆是单元大小和轮次决策，不是 kernel。打断处理是明显的短板：固定版本没有取消接口，被打断的回答中位数还要继续说 5 秒。请求-响应路径一次一个请求时 TTFA 0.72 秒，但回答只有一半长、也没算端点检测，不能说它“更快”。

---

## C1 — Is packed variable-length flow attention faster on our real batch shapes?

**Research question.** Does the flow DiT's packed variable-length path (with and without block compilation) reduce code2wav GPU time per useful audio second (E) on the batch shapes our serving runs actually produced, without degrading audio?

**Hypothesis and gates (registered in `compiler_runtime_opportunities.md` §4 and `c1_c2_design.md` §5 before the run; Erratum E-C1 registered before the run).** Performance: P1 (packed + `enable_flow_block_compile`) lowers E by ≥ 5 % on W3b or MIX representative shapes in every one of 3 timed passes. Quality first: waveform lengths correct and finite; mean MCD against an FP32 reference no more than 0.1 dB above the dense path's; ASR exact matches at most 1 fewer and pooled WER at most +0.5 points. Prediction (Estimated): P0 5–25 % slower than D; P1 −5..−12 % on W3b and −10..−20 % on MIX; P1 likely to fail the MCD rule because the packed path is BF16 by design.

**Setup.** Run `ppv_c1_c2_microbench`, 23:40–00:53, 2 GPUs (C1 arms one after another on one GPU, C2 on the other, then the ASR quality pass); it overlapped every R1 run, the R2 smokes and R2 blocks 1–2 on other GPUs (the ASR pass ran during R2 blocks 1–2; deviation D1). Only the code2wav flow (+ HiFT for E) is loaded, as the component does, from the pinned stock checkout. Inputs: a seeded sample of the real batches exported from the code2wav batch events of 实验 9 and 10 (`research/next_steps/results/c1_real_shapes.json`, 47,484 batches in 38 groups): 32 batches from each primary group (W3b fifo16, of 95; MIX fifo16, of 98) and 12 from each of four secondary groups, i.e. 112 batches per arm (D_ng, P2 and P1h: the 64 primary batches only), replayed with the same token inputs in every arm. Arms: REF (FP32 dense, quality reference only), D (default: FP16 dense masked SDPA, compiled DiT, dense CUDA graphs), D_ng (D without graphs), P0 (packed varlen, no compile), P1 (packed + block compile), P2 (P0 + packed graphs), P1h (P1 with the packed path in FP16 instead of BF16; exploratory, not the registered candidate). Timing: CUDA events, 3 timed passes after a warm-up (compile and capture time reported separately), recompiles counted; a separate short profiler pass for kernel families. Capability probe on sm_86: `varlen_attn` (FlashAttention-2) runs in BF16 and FP16; SDPA with a boolean mask cannot use the flash backend (falls back to mem-efficient).

**Evidence** (`results/c1_microbench.json`, `results/c1_quality.json`). E = code2wav GPU ms per useful audio second (flow + HiFT); mean of 3 passes (pass-to-pass spread ≤ 0.3 %):

| Real batch-shape group (source) | D | P0 | P1 | P1h | P1 vs D |
|---|---|---|---|---|---|
| W3b, fifo16, code2wav saturated (实验 10) | 8.52 | 11.73 | 6.47 | 6.47 | **−24.1 %** (−24.1, −24.1, −24.0 per pass) |
| MIX, fifo16, code2wav saturated (实验 10) | 11.21 | 11.13 | 6.55 | 6.58 | **−41.5 %** |
| W3b, length8 (实验 10) | 7.50 | 11.86 | 6.52 | — | −13.0 % |
| MIX, length8 (实验 10) | 8.40 | 10.98 | 6.30 | — | −25.0 % |
| W3L, 2+2+4 closed loop (实验 9) | 8.76 | 10.58 | 6.36 | — | −27.4 % |
| W3b, 2+2+4 open loop, ~1.2 requests per batch (实验 10 validation) | 8.62 | 8.81 | 8.56 | — | −0.8 % |

D_ng (no CUDA graphs) is only +3.3–3.6 % above D. Kernel families (ms per audio second): attention 1.43 → 0.59 (−59 %), mask preparation 0.60 → 0, GEMM 1.94 → 1.69, Inductor-fused +0.22. Requests on a CUDA graph in the sampled batched groups: 0–3.2 %, except D on MIX length8 (9.3 %, 12 batches). No recompilation in timed passes; peak memory unchanged (5.5 vs 5.1 GiB on W3b fifo16).

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

**Interpretation.** Packing removes the masked attention and its mask-preparation kernels — the two largest flow-specific costs found by the kernel census — and cuts E by 13–42 % on batched real shapes, more when FIFO batching mixes lengths (MIX fifo16, the most padded case). Length-aware batching already recovers part of this (D: 11.21 → 8.40 on MIX), so the two optimizations overlap: the packed gain is −25 % on length8 batches versus −42 % on fifo16 batches. Without compilation the packed path is slower in four of six groups (P0 +21..+58 %; −0.7 % on MIX fifo16, +2.2 % on the open-loop group), and packed CUDA graphs add nothing (P2 ≈ P0); CUDA graphs matter little anywhere here (D_ng +3 %), because batched shapes rarely match a captured graph. On the open-loop 2+2+4 service below saturation (≈1.2 requests per batch) there is nothing to pack and the gain vanishes (−1 %). The quality failure is a precision choice of the pinned implementation (BF16 in the packed path), not a property of packing: the same packed path in FP16 matches the dense path's quality. A registered follow-up should test P1h end-to-end in the saturated regime.

**Limitations.** Microbenchmark only (no serving-level effect claimed); 112 sampled batches per arm (the two headline groups 32 batches each); ran alongside the R1 and R2 latency runs on other GPUs (D1); quality set of 26 TTS-generated segments re-tokenized and re-synthesized (not talker output); MCD here is an MFCC-style variant comparable only between arms; the ASR rule's margin is smaller than run-to-run sampling differences (D_ng fails it), so it is a weak discriminator; P1h is exploratory and was not the registered candidate; C1 shared the node with R1 block 1 (deviation D1), timed with CUDA events.

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

**中文小结。** HiFT 的三项编译期优化都没有带来可测的加速：weight-norm 折叠逐位正确但时间没变；channels-last 去掉了布局转换，卷积却变慢，而且数值差异超过门槛；FP16 把 kernel 时间减了 42%，音质没过门槛（MCD 0.83 dB），两个批处理组的总时间反而多了 31–50%（W3L 组 −2…+23%）——因为 HiFT 每个请求单独调用、kernel 很小，GPU 有大量时间在等主机端。按登记的放弃条件，这个方向停止。如果还要优化 HiFT，杠杆更可能是跨请求批量调用或给 HiFT 录 graph（假设，未测）。

---

## R2 — How does interaction latency change with the number of concurrent sessions?

**Research question.** What is the first stage or resource that limits real-time interaction as the number of full-duplex sessions on one A6000 grows?

**Hypothesis and rules (registered, `preregistration.md` R2 and "R1/R2 分析口径补充", before any R2 data).** Placement chosen by the smoke rule: if `max_sessions 8` starts on one GPU and 8 sessions run 30 s without failure, use one GPU with levels 1/2/4/8. Each block runs the levels from low to high for 180 s; escalation stops after a level with a failed session or unit miss > 1 %. Serving capacity = highest level that ran in all three blocks with no failed session, no 503, no pacing violation and pooled unit miss ≤ 1 %. Deadline capacity = highest level with deadline miss ≤ 5 % (1 s; 0.5 and 2 s sensitivity). First limiting stage = largest increase in per-unit queue + service median from the lowest level to the first level with miss > 1 %.

**Setup.** Smokes `ppv_r2_smoke_ms8_1gpu` and `_ms8_2gpu` (00:28–00:32): the one-GPU smoke passed (unit miss 3.3 %, unit lag P50 262 ms), so the registered rule selected `ms8_1gpu`. The two-GPU smoke also met the registered start-and-no-failure rule, but all 240 of its units were late (unit lag P50 ≈ 22 s, P99 ≈ 40 s). Main runs `ppv_r2_ms8_1gpu_block{1,2,3}` (00:33–01:10, 1 GPU each, chained, no overlap with other latency runs; the smokes and blocks 1–2 overlapped the last part of the C1/C2 microbenchmark on another GPU — C1 arms, then its ASR pass until 00:53; deviation D1 item 4): stock runtime, `--variant session --enable-realtime --max_sessions 8`, same inputs as R0/R1, server request events on.

**Evidence** (`results/r2_summary.json`; P99 only with N ≥ 100).

| Sessions | Blocks run | Unit miss rate (pooled) | Speaking-unit lag P50 / P99 | Listening-unit lag P50 | TTFA P50 (N) | Turns missing the 1 s deadline | Extra listening units k = 0 / 1 / 2 | Failures |
|---|---|---|---|---|---|---|---|---|
| 1 | 3 / 3 | 0.56 % | 353 / 608 ms | 74 ms | 935 ms (18) | 50 % | 12 / 3 / 3 | none |
| 2 | 2 / 3 | 0 % | 395 / 744 ms | 74 ms | 1,737 ms (22) | 88 % | 6 / 16 / 0 | none |
| 4 | 2 / 3 | 0.14 % | 464 / 902 ms | 77 ms | 1,557 ms (44) | 92 % | 17 / 25 / 2 | none |
| 8 | 2 / 3 | **45.6 %** | 707 / 1,480 ms | 98 ms | 1,755 ms (74) | 94 % | 34 / 39 / 1 | block 1: miss 3.5 %, no failed session, 25 server-log lines matching CUDA OOM (allocator warnings only); block 3: miss 87.7 %, 7 of 8 sessions failed with CUDA OOM (49 OOM lines) |

Registered verdicts: serving capacity **below the lowest level**; deadline capacity (1 s, also 0.5 and 2 s) **below the lowest level**; first limiting stage **talker, resource "neither saturated (uncertain)"**; at 1 session, 11 of 12 deadline misses are model-driven (no response, or k ≥ 1 with a + c ≤ 1 s).

Why the registered verdicts came out this way (analysis after the fact, rules unchanged):
- Block 2 stopped escalating after level 1 because its unit miss rate was 1.67 %: all three missed units belong to the **first reply after server start** (unit lags 2.5, 2.2, 1.5 s). The first answer after start is slow in every block (TTFA 2.8, 4.5, 2.8 s vs ≈ 0.9 s later). Hypothesis: a one-time lazy initialization on the first talker generation, which start-up warm-up never reaches because the first three units are forced to listen; unlike R0/R1, the R2 runs had no smoke phase before level 1.
- Block 1's level 1 had 16 late packets: one 1.34 s **client stall** in which both client processes were in uninterruptible I/O sleep, Linux I/O pressure (avg10) rose from 0.18 to 7.01 (full) / 7.15 (some), the CPU sampler skipped a second and the shared server log has no line for that second. Hypothesis: the client's trace writes to NFS blocked its event loop (see P3).
- Without these two artefacts, level 1 would meet the serving target in all blocks; levels 2–8 still cannot qualify because block 2 never ran them.
- The first-limiting-stage verdict is not robust: controlling for unit kind, all three GPU stages slow by about the same amount from 1 to 8 sessions (speaking units only: thinker +89.8, talker +88.7, speech +88.2 ms); the talker leads by 7 ms only because of the speaking/listening mix.
- The apparent 3–4× slower single session (all-unit lag P50 279 ms vs 72 ms in R1) is a statistic, not a slower server: unit lag is bimodal (speaking ≈ 350 ms, listening ≈ 74 ms) and level 1 had 50 % speaking units against 42–44 % in R1; per kind, level 1 matches R1.
- nvidia-smi utilization is unreliable under this periodic load (0 % readings at 120–220 W); power and stage timing are used instead.

**Result.** On one A6000 with `max_sessions 8`, the duplex path kept its 1-s units real-time with 1, 2 and 4 concurrent sessions in the blocks that ran (unit miss ≤ 0.6 %), and broke at 8 sessions (46 % of units late or missing) because the card ran out of memory: the speech process grew to 13.6 GiB against its nominal 0.15 share (≈ 7.1 GiB) and sessions failed with CUDA OOM. Compute slowed gradually rather than hitting a wall — the speaking unit's processing rose from 353 to 707 ms (P50) and to 1.48 s at P99, so at 8 sessions some units no longer fit in their second even before the OOM failures. The registered capacity verdicts are "below the lowest level", driven by a first-reply warm-up and a client I/O stall rather than by sustained load.

**Interpretation.** The first resource to give out as sessions grow is **GPU memory on the shared card** (the speech stage's state grows beyond its nominal fraction; at 8 sessions both blocks logged CUDA OOM allocator failures, and in one of them 7 of 8 sessions failed), not a single compute stage; compute degrades evenly across thinker, talker and vocoder (≈ +89 ms each per speaking unit from 1 to 8 sessions). Turn-taking dominates TTFA at every level (deadline misses are mostly model-driven), so more sessions mostly widen the tail of the speaking unit. A placement that splits the stages over two GPUs (the registered `ms8_2gpu` arm; its smoke met the start-and-no-failure rule, but all 240 of its units were late) or a bounded speech-state budget is the next experiment; the split is now registered and queued as R2b.

**Limitations.** Synthetic sessions; levels 2–8 have two blocks, not three; the one-GPU smoke (30 s) did not reach the memory growth that broke level 8, so the placement rule chose an arm that fails at its top level; first-reply warm-up and the client stall contaminate level 1; at 8 sessions the two blocks differ widely (unit miss 3.5 % vs 87.7 %), so the pooled 45.6 % hides that one block failed and one did not; the first-limiting-stage rule cannot see out-of-memory failures and its GPU criterion uses nvidia-smi utilisation, which is unreliable for this load; stage timing comes from server request events; the smokes and blocks 1–2 overlapped the C1 microbenchmark on another GPU (D1).

**中文小结。** 一张 A6000、`max_sessions 8` 时，1、2、4 个会话的 1 秒单元都能按时处理（跑了的区组里错过率 ≤ 0.6%）；8 个会话时两个区组都出现了显存分配失败：区组 3 有 7/8 的会话因 CUDA OOM 失败、88% 的单元超时或丢失；区组 1 没有会话失败，错过率 3.5%（两个区组合并 46%）。语音进程的状态涨到 13.6 GiB，远超它 0.15 的份额。计算是逐渐变慢的（说话单元的处理从 353 ms 涨到 707 ms），三个 GPU 阶段各慢约 89 ms，不是某一个阶段先满。按登记规则，两个“容量”都是“低于最低档”，但原因是两个和负载无关的问题：服务启动后第一次回答特别慢（区组 2），以及客户端写 NFS 时卡住 1.34 秒（区组 1）——后者（假设：客户端写 NFS 的记录阻塞了事件循环）是 P3 要找的那种停顿的一个候选实例。另外，nvidia-smi 的利用率在这种每秒一次的周期负载下读数不可信，改看功耗。

---

## P0 — Is the Talker's admission cap what limits the 2+1+5 placement?

**Research question.** In the 2+1+5 placement (thinker 2 GPUs, one talker GPU, code2wav 5 GPUs), the single talker was the bottleneck while its GPU was only ~65 % busy and its scheduler sat at 32 running requests with ~180 queued (实验 9). Does raising the talker's `max_running_requests` from 32 to 64 raise the service's capacity, and does 2+1+5 then catch up with 2+2+4?

**Hypothesis (registered in `p0_p3_experiment_plan.md` P0 before any run; addendum and Erratum E-P0-G in `preregistration.md`).** H0a: B's W3b capacity ≥ +5 % over A in all 3 blocks and B's talker runs > 32 requests (median) at W3b 256 → supported; < +2 % in all blocks → refuted. H0c: B ≥ reference 2+2+4 in all blocks → supported; < in all → refuted (registered prediction: refuted). H0b classifies what limits B. Prediction (Estimated from the batch-size/step-time curve): talker throughput +10–20 %, 2+1+5 capacity 19.5–21.3 req/s, still below 2+2+4.

**Setup.** Runs `ppv_p0_{cap32,cap64,ref_t2_k2_c4}_block{1,2,3}_a2` (22:35–02:49, 8 GPUs each, 9 runs; block order seeded with 20261119), instrumented runtime of 实验 9 (`unicorn_instr`), server restarted for every run. Arm A = 实验 9's 2+1+5 unchanged; arm B = only `stages.talker.engine.max_running_requests: 64`, which at the pinned version also captures CUDA graphs up to batch 64; reference = 2+2+4 unchanged (W3b only). Workloads W3b (closed loop 64/128/256 as separate phases + 256 with request events; open loop 15.5–31 req/s, 60 s each) and W3L (A and B only; closed 128, open 8–12 req/s). Arm G (cap 64, graphs ≤ 32) could not start: the pinned version refuses a graph limit below the cap (Erratum E-P0-G), so B vs A is the **joint** effect of the admission cap and graph coverage, not a single-variable admission effect. Four attempt-1 runs failed at start for a harness reason (F1) and are kept in the manifest.

**Evidence** (`results/p0_summary.json`; `figures/fig02_talker_admission.png`, `figures/fig03_talker_cap_capacity.png`).

| Arm | W3b capacity by block (req/s) | mean | W3L capacity by block | stable open-loop W3b rate by block |
|---|---|---|---|---|
| A cap 32 | 17.35 / 17.37 / 17.39 | 17.37 | 7.24 / 7.19 / 7.34 | 21.0 / 23.5 / 21.0 |
| B cap 64 (+ graphs to 64) | 22.63 / 22.55 / 22.94 | 22.71 | 9.50 / 9.48 / 9.35 | ≥ 31.0 (top rate) in all blocks |
| Reference 2+2+4 | 23.05 / 23.57 / 24.11 | 23.58 | not run | ≥ 31.0 (top rate) in all blocks |

- **H0a: supported.** B vs A W3b capacity +30.4 / +29.8 / +31.9 % (mean +30.7 %, t 95 % CI +28.0..+33.4 %); B's talker ran 64 requests (median) at W3b 256 in every block. W3L: +31.1 / +31.7 / +27.3 % (mean +30.1 %).
- **H0c: refuted** (as predicted). B vs reference −1.8 / −4.3 / −4.9 % (mean −3.7 %, CI −7.7..+0.3 %).
- **H0b** (registered classes): "CPU/scheduling-limited" in all blocks — the talker remains the stage holding most requests (73–74 % of residence), its GPU is 53–58 % busy and its process uses ~100 % of a core. The CPU criterion cannot separate real work from a busy-polling scheduler loop (limitation M1); KV is not limiting (KV-trigger share 0 % over 3,017 high-load lines; token usage ≤ 0.16), so arm C was not run.
- Talker scheduler at W3b 256: A runs 32 with 180–183 queued (talker queue P50 8.3–8.7 s); B runs 64 with 127–129 queued (queue P50 4.2–4.4 s); reference talkers run 15–16 with nothing queued, and code2wav becomes the stage holding most requests (60–62 % of residence, code2wav GPUs 86–89 % busy).
- Talker decode step time (SGLang log; mean of the three runs' per-run medians): 3.79 ms at batch 32 and 6.13 ms at 64 for B, i.e. 8.5k → 10.4k tokens/s (+23 %); A's step at 32 is 4.09 ms.
- Open loop W3b (throughput / P95 / goodput, mean of blocks): at 26 req/s A is unstable (20.5 req/s, P95 17 s, goodput 5.4) while B (25.0, 5.2 s, 25.0) and the reference (25.2, 4.0 s, 25.2) are stable; at 31 req/s B is still steady but at P95 8.1–8.3 s and goodput 16–19 req/s, whereas the reference holds P95 4.9 s and goodput 29.5.

**Result.** Raising the talker's admission cap (with the graph coverage that comes with it at the pin) is worth +31 % capacity for 2+1+5 on both workloads — more than the +10–20 % we predicted — and brings it within 4 % of 2+2+4 in closed-loop capacity, but not to parity: near saturation its tail latency and goodput stay well behind (P95 8 s vs 5 s at 31 req/s). With the higher cap the talker immediately runs at the new limit with ~128 requests waiting while its GPU is half idle.

**Interpretation.** The 2+1+5 deficit in 实验 9 was mostly a software admission limit, not a lack of GPU: one configuration value recovered most of the gap between placements. Larger talker batches amortize per-step overhead (step time grows from 3.8 to 6.1 ms when the batch doubles), and the talker is still admission-bound at 64, so the remaining gap to 2+2+4 is consistent with the cap still binding and with the single talker GPU's step time, not with KV memory. Because the pinned implementation ties the CUDA-graph limit to the cap, we cannot say how much of the +31 % comes from admitting more requests versus capturing graphs for larger batches. The capacity model therefore has to include per-stage admission limits, not only per-stage GPU throughput: a placement search that only balances GPU work would have rejected 2+1+5 for the wrong reason.

**Limitations.** One node, synthetic W3b/W3L; joint change of cap and graph coverage (Erratum E-P0-G); the H0b CPU criterion is weak (M1); the reference ran W3b only; open-loop rates top out at 31 req/s, so the stable capacities of B and the reference are lower bounds; the registered capacity includes the final drain (as in 实验 9).

**中文小结。** 只把 talker 的同时处理名额从 32 改成 64（固定版本会同时把 CUDA graph 录到 64），2+1+5 的容量在 W3b 和 W3L 上都提高约 31%（3 个区组 +30.4/+29.8/+31.9%），比事先预测的 +10–20% 还多，闭环容量离最好的 2+2+4 只差约 4%；但高负载下尾延迟和 goodput 仍明显落后（31 请求/秒时 P95 8 秒对 5 秒），按登记规则“追上 2+2+4”被反驳。名额放开后 talker 立刻又顶在 64、排队约 128 个，卡只忙一半——瓶颈仍是软件准入，而不是 GPU 或 KV（KV 触发条件 0%）。由于固定版本不允许 graph 上限低于名额，名额和 graph 覆盖的贡献分不开。

---

## R3 — Where does continuous camera input stop keeping up on the request-response path?

**Research question.** When synthetic camera sessions send one frame (or one 2-s clip) per tick to a text-only replica on one A6000, at which number of sessions × frame rate does the replica stop keeping up, which resource limits it first, and does raising the CPU preprocessing concurrency from 1 to 4 help?

**Hypothesis and gates (registered 2026-11-18 22:55, before any R3 run; `r3_r4_design.md` §2.4).** A cell is *saturated* when ≥ 10 % of ticks are dropped or ≥ 20 % of sent requests finish after the next tick. A saturated cell is *preprocessing-bound* when preprocessing (queue + service) is ≥ 50 % of server time and the preprocessing slot is ≥ 80 % busy, *GPU-bound* when GPU busy ≥ 80 % and the slot < 80 % busy, otherwise *unclear*. Concurrency 4 *improves* a cell when its drop ratio falls by ≥ 5 points and ≥ 20 % relative, or its first-response P95 falls by ≥ 20 %. Prediction (Hypothesis): `clip` at 16 sessions and ≥ 1 fps is preprocessing-bound, and concurrency 4 helps `clip` but hardly `frame`. Confirmation: at most 6 cells picked by a fixed rule, 3 blocks × both concurrencies; “supported” only when all 3 blocks improve.

**Setup.**
- Replica: `--text-only`, 1 GPU and 7 CPU cores (the per-GPU share of the node), runtime `unicorn_ppv_kv` = 520b49ee plus a default-off research patch. Stock MiniCPM-o hard-codes preprocessing concurrency 1 and rejects the parameter, so both arms use the patched code and differ only in that parameter.
- Camera: synthetic 640×360 scene with 1–5 moving circles; each frame carries a code so no two frames are byte-identical (no media or prefix cache hits). `frame` sends the newest PNG and asks how many circles there are; `clip` sends the last 2 s at 8 fps as H.264.
- Load: one request in flight per session (`drop_if_busy`): a tick that arrives while the previous request is outstanding is dropped. 1/4/16 sessions × 0.5/1/2 fps × 2 shapes = 18 cells, each 10 s warm-up + 60 s measured, in random order (seed 20261119, same order for both arms), request events on.
- Screening: one run per concurrency, side by side on two GPUs, 02:53–03:20; no other run overlapped them. A first attempt crashed at its first request (the patch used a field SGLang 0.5.21 does not have) and was re-run under new IDs; both failed runs are kept.
- Confirmation: the five cells picked by the registered rule, 3 blocks run one after another (07:54–08:24; the first, unordered plan was withdrawn before it started, deviation D2), each block with both concurrencies side by side on two GPUs and the cells in a seeded random order per block; nothing else ran on the node.
- This is frame replay through `/v1/chat/completions`, not a live camera and not the duplex image path.

**Evidence** (`results/r3_screen.json`; screening = one run per arm). Concurrency 1, the five saturated cells and the three cells next to them; at 1 session every cell answers in 0.14–0.15 s (P95) with image-encoder service 47–67 ms and thinker prefill 40–51 ms:

| Cell | Offered → sent (req/s) | Dropped | Late | First response P50 / P95 (s) | Image encoder P50 (ms) | Thinker prefill P50 (ms) | Preprocessing share / slot busy | GPU util / power | Class |
|---|---|---|---|---|---|---|---|---|---|
| frame, 4 sessions, 2 fps | 8 → 5.9 | 26 % | 36 % | 0.40 / 0.61 | 219 | 104 | 8 % / 0.25 | 89 % / 289 W | GPU-bound |
| frame, 16, 0.5 | 8 → 7.7 | 4 % | 5 % | 1.50 / 1.86 | 1,232 | 111 | 2 % / 0.33 | 86 % / 290 W | not saturated |
| frame, 16, 1 | 16 → 7.5 | 53 % | 99 % | 1.51 / 2.23 | 1,219 | 111 | 2 % / 0.33 | 82 % / 291 W | GPU-bound |
| frame, 16, 2 | 32 → 7.6 | 76 % | 100 % | 1.71 / 2.20 | 1,401 | 111 | 3 % / 0.39 | 88 % / 291 W | GPU-bound |
| clip, 4, 2 | 8 → 7.4 | 8 % | 9 % | 0.34 / 0.46 | 158 | 77 | 12 % / 0.41 | 82 % / 284 W | not saturated |
| clip, 16, 0.5 | 8 → 8.0 | 0 % | 0 % | 0.37 / 0.59 | 167 | 80 | 10 % / 0.39 | 80 % / 283 W | not saturated |
| clip, 16, 1 | 16 → 9.4 | 41 % | 71 % | 1.05 / 1.50 | 778 | 84 | 6 % / 0.65 | 82 % / 285 W | GPU-bound |
| clip, 16, 2 | 32 → 9.5 | 70 % | 100 % | 1.28 / 1.63 | 1,023 | 85 | 5 % / 0.76 | 81 % / 285 W | GPU-bound |

- **Ceiling (Measured):** accepted requests flatten at 7.5–7.7 req/s for `frame` (235 prompt tokens) and 9.4–9.5 req/s for `clip` (167 tokens) per GPU, whatever the offered rate.
- **Where the time goes (Measured):** against the 1-session cell of the same shape and rate, the image-encoder stage's service time grows 16–22× in the four 16-session saturated cells (to 0.78–1.40 s) and 3.4× in the 4-session one; thinker prefill grows 2.0–2.2× and thinker queueing stays ≤ 7 ms; in the saturated cells CPU preprocessing is 2–8 % of server time (at most 21 %, in 1-session `clip` cells) and its process uses 1.5–1.9 of the 7 cores.
- **GPU (Measured):** in every 16-session cell the GPU drew 283–291 W at concurrency 1 (279–290 W at concurrency 4), near the A6000's 300 W board limit (spec value; the limit was not queried), and the median SM clock fell from 1.9 GHz at light load to 1.64–1.70 GHz. In 15 of the 1- and 4-session cells nvidia-smi read 0 % for (nearly) the whole minute at 140–224 W (probably aliasing with the request cadence, as in R0; not verified); none of those cells is saturated, so no class depends on those readings (descriptive columns added after seeing the data; the rule was not changed).
- **Consistency check (Derived):** at 1 session one request costs ≈ 117 ms (`frame`) or ≈ 89 ms (`clip`) of encoder + prefill. If all GPU work of the replica runs one request at a time, the ceiling is ≈ 8.5 and ≈ 11 req/s; the measured ceilings are 89 % and 85 % of that.
- **Concurrency 4 (Measured):** drop ratio within −3.5 … +1.2 points of concurrency 1 in all 18 cells, first-response P95 −14 … +9 %; no cell meets the improvement rule. The circle-count accuracy differs by −0.02 … +0.01, so the 4 parallel preprocessing workers did not corrupt inputs.
- **Picked for confirmation (registered rule):** the five saturated cells, most-dropping first: `frame_s16_f2`, `clip_s16_f2`, `frame_s16_f1`, `clip_s16_f1`, `frame_s4_f2`.

**Result (screening).** At concurrency 1 all five saturated cells are GPU-bound by the registered rule (at concurrency 4 two of them read 74–75 % GPU busy and are “saturated, cause unclear”); none is preprocessing-bound. The registered prediction — `clip` preprocessing-bound at 16 sessions, helped by concurrency 4 — is not supported at screening: in the saturated cells preprocessing is at most 8 % of server time, and raising its concurrency changed no cell. The 3-block confirmation below tests this on the five saturated cells.

**Confirmation** (`results/r3_confirm.json`, figure `fig10_r3_camera`; Measured; values per block 1 / 2 / 3, concurrency 1 → 4):

| Cell | Offered (req/s) | Accepted (req/s) | Dropped | First response P95 (s) | Class at concurrency 1 | Blocks improved by concurrency 4 | Verdict |
|---|---|---|---|---|---|---|---|
| frame, 16 sessions, 2 fps | 32 | 7.9/7.8/7.7 → 7.9/7.5/7.7 | 75/76/76 → 75/76/76 % | 2.22/1.95/2.07 → 1.94/2.16/2.08 | GPU-bound ×3 | 0 of 3 | **not supported** |
| clip, 16, 2 | 32 | 11.4/9.8/9.3 → 9.3/9.5/9.2 | 64/69/71 → 71/70/71 % | 1.30/1.63/1.73 → 1.82/1.66/1.79 | unclear ×3 | 0 of 3 | **not supported** |
| frame, 16, 1 | 16 | 8.1/7.8/7.5 → 7.3/7.6/7.2 | 49/51/53 → 54/52/55 % | 1.78/1.91/2.12 → 2.30/2.08/2.35 | GPU-bound ×3 | 0 of 3 | **not supported** |
| clip, 16, 1 | 16 | 11.1/9.2/9.2 → 9.3/9.2/9.4 | 31/43/42 → 42/42/41 % | 1.30/1.66/1.58 → 1.69/1.55/1.50 | unclear ×3 | 0 of 3 | **not supported** |
| frame, 4, 2 | 8 | 6.0/5.9/5.9 → 5.9/5.9/5.9 | 25/26/26 → 26/26/26 % | 0.60/0.62/0.61 → 0.62/0.63/0.61 | GPU-bound ×3 | 0 of 3 | **not supported** |

- **Registered verdict:** concurrency 4 improved no cell in any block, so all five cells are *not supported* (rule: supported only when all 3 blocks improve, not supported when none does). No request failed; circle-count accuracy is the same in both arms (`frame` 0.96–0.98, `clip` 0.76–0.79 at 1 fps and 0.91 at 2 fps).
- **Classes:** the three `frame` cells are GPU-bound in every block (GPU busy 80.7–90.4 %, preprocessing slot ≤ 0.39). The two `clip` cells are “saturated, cause unclear” at concurrency 1 in every block, for two different reasons: in blocks 2–3 GPU busy read 72.7–79.4 %, just under the 80 % line; in block 1 GPU busy was 87.5–88.6 % but the preprocessing-slot metric read 1.37–1.46. A single serial slot cannot be more than fully busy: the metric uses the stage's dispatch-to-done interval (P50 108–118 ms there), about twice the preprocessing call's own wall time (P50 49–51 ms), so it over-counts. The class is reported as the registered rule gives it. Preprocessing never queued: its wait P95 was ≤ 0.8 ms in every cell of both arms, and its share of server time ≤ 11.5 %. No cell is preprocessing-bound.
- **Board effect (found after the runs):** the concurrency-4 replica ran on the same physical GPU in all three blocks and in the screening (identified by GPU UUID in `logs/env`); concurrency 1 ran on a second board in blocks 2–3 and the screening and on a third in block 1. At concurrency 1, block 1's board accepted 1–22 % more requests than the other board did in blocks 2–3 (`clip`, 16 sessions, 1 fps: 11.1 vs 9.2 / 9.2 req/s; `frame`, 16, 1 fps: 8.1 vs 7.8 / 7.5), while the concurrency-4 board varied by at most 5.5 % across the three blocks, so the difference is most likely the board (or its place in the node), not the block. It is as large as or larger than any difference between the arms. Arm and board are therefore confounded; no verdict changes (no block improved), but the per-block differences between concurrency 1 and 4 must not be read as effects of concurrency.
- **Ceiling:** at 16 sessions the replica accepted 7.2–8.1 frames/s and 9.2–11.4 clips/s across the three boards, whatever the offered rate.

**Result (confirmation).** The registered prediction is not supported: in all 5 saturated cells and all 3 blocks, raising CPU preprocessing concurrency from 1 to 4 neither lowered the drop ratio nor the first-response P95 by the registered margins. The three single-PNG cells are GPU-bound in every block; the two clip cells are “cause unclear” by the rule, but with no preprocessing queue and ≤ 11.5 % of server time in preprocessing. Between the screening and the confirmation, the per-GPU ceiling held within the range set by board-to-board differences.

**Interpretation.** On this path the camera stops keeping up because one GPU has to run the image encoder and the thinker for every frame: the replica tops out at ≈ 7.6 frames/s (single PNG) or ≈ 9.5 clips/s, about what the GPU time per request predicts. More CPU parallelism in preprocessing cannot help. “Keeping up” is also not the same as “fresh”: with single PNGs at 16 sessions × 0.5 fps only 4.4 % of ticks are dropped (below the 10 % threshold), yet the median answer arrives 1.50 s after its frame (P95 1.86 s; ≈ 10× the 1-session P50 of 0.14 s). Levers that follow from this (Hypothesis, not tested): fewer image tokens per frame (the 2-s clip is cheaper than one PNG), skipping unchanged frames, batching frames across sessions in the encoder, or a second GPU for the encoder.

**Limitations.** Synthetic frames of ≈ 2.5 KB (CPU decode cost is a lower bound); request-response path only; text-only replica; one run per arm at screening; arm and physical GPU board confounded in screening and confirmation (one board accepted 1–22 % more requests than another at the same settings); the encoder's “service” time is dispatch-to-completion on a GPU shared by all stages of the replica, so the events cannot separate its own compute from waiting for the GPU or for requests batched with it; nvidia-smi utilisation is unusable at light load; `processor` timing includes host-to-device copies.

**中文小结。** 一张 A6000 上的纯文字副本，合成摄像头每个时刻发一帧图（或 2 秒短片）。16 个会话时，每秒能接受的请求封顶在约 7.6 帧（单张 PNG）或约 9.5 段短片，再提高帧率只会多丢帧。时间都花在图像编码器上（服务时间从约 50–70 ms 涨到 0.8–1.4 s），GPU 功耗顶在 300 W 上限附近、频率被压低；饱和格子里 CPU 预处理只占服务端时间的 2–8%。按登记规则，5 个饱和格子都是“GPU 受限”，没有一个是“预处理受限”。把预处理并发从 1 提到 4，18 个格子都没有改善，事先的预测（短片在 16 个会话时卡在预处理上，提高并发有帮助）在筛选阶段不成立。另外，“跟得上”不等于“新鲜”：单张 PNG、16 个会话 × 0.5 帧/秒只丢 4.4%（低于 10% 的饱和线），但回答中位数晚 1.50 秒（P95 1.86 秒）。3 个区组的确认（5 个饱和格子，两种并发并排）结果相同：并发 4 在任何区组、任何格子都没有达到登记的改善标准，5 个格子全部判为“不支持”；预处理的排队等待 P95 在所有格子都不到 1 毫秒，从来没有请求排队等预处理。确认时还发现，并发 4 每次都落在同一块 GPU 上，并发 1 在区组 1 用的另一块卡比区组 2、3 的卡多接受了 1–22% 的请求，所以两组之间的小差别不能当成并发的作用。

---

## P1 — Does length-aware code2wav batching with a maximum wait help the full service near saturation?

**Research question.** 实验 10 found that grouping code2wav requests by length raises code2wav throughput when code2wav alone is saturated, but on the full service below saturation batches held only ~1.2 requests, so nothing could be shown there. On the full 2+2+4 service driven to and past its capacity, does Length8 batching with a maximum wait τ (old requests are forced into the next batch) raise goodput, keep short requests from starving, and leave short-answer traffic no worse?

**Hypotheses and gates (registered in `../p0_p3_experiment_plan.md` P1; details written before any P1 data in `p1_p2_p3_design.md` §1.3; X fixed by the pre-run, Erratum E-P1-X).**
- H1a: on MIX at 1.0X and 1.1X, Length8 + τ 1 s goodput ≥ 1.05 × FIFO16 in all 3 blocks → supported; ≤ FIFO16 everywhere → refuted; otherwise inconclusive.
- H1b: on MIX at 1.0X and 1.1X, short requests' code2wav queue P99 with τ 1 s ≤ 0.8 × Length8's, with audio seconds per second ≥ 0.95 × Length8's, in all 3 blocks → supported; otherwise inconclusive.
- H1c: on W3b at every rate, Length8 + τ 1 s end-to-end P95 ≤ 1.05 × FIFO16's in all 3 blocks → supported; otherwise inconclusive.
- Registered failure modes: the talker or thinker may saturate before code2wav; in open loop any dequeue order only redistributes queueing; τ too small degenerates to FIFO, too large does not protect short requests.

**Setup.**
- Runtime `unicorn_ppv_maxwait` = 520b49ee + 实验 10's batching patch + the max-wait rule (code2wav only) + gc logging; 2+2+4 (`t2_k2_c4`), flow CUDA graphs on, request events on in every cell.
- X = busy-window throughput of the FIFO16 closed-loop pre-run: W3b 28.96, W3L 14.52, MIX 14.67 req/s (Erratum E-P1-X). Rates 0.9X/1.0X/1.1X rounded to 0.5: W3b 26/29/32, W3L and MIX 13/14.5/16 req/s.
- 5 settings: FIFO16, Length8, Length8 + τ 0.5/1/2 s (look-ahead window 64). Length8 differs from FIFO16 in two ways: it groups by length *and* caps a batch at 8 requests instead of 16. Length8 anchors every batch on the oldest waiting request and fills it with the requests closest in length; with τ, requests that have waited ≥ τ are taken first, in arrival order.
- One open-loop cell per workload × setting × rate (45 per block), each after a control phase that writes all four keys; 15 s (W3L 20 s) warm-up dropped by scheduled send time, then 75 s (W3L 90 s) measured. Poisson arrivals with seed 20261119 + block (+ rate), so the five settings of a block and rate see the same requests at the same times.
- 3 blocks, one server start each, workload order rotated per block, cells in a seeded random order. Block 2 died when P3's SIGUSR2 stack dump segfaulted a code2wav process (F6, deviation D3) and was re-run as `ppv_p1_block2_a2` without the stack hook (06:31–07:54); the verdicts use blocks 1, 2_a2 and 3. Block 1 ran 03:23–04:44, block 3 05:07–06:31, block 2_a2 06:31–07:54; no other run overlapped any block (orchestrator start and end times).

**Evidence** (`results/p1_summary.json`, figure `fig11_p1_batching`; Measured; per-block values in block order 1 / 2_a2 / 3).
- **Validity.** 135 of 135 cells usable: the control file was applied to every batch, no request failed or timed out, no server restarted. Every measured request completed in every cell, so the completion rate equals the realised arrival rate (MIX 1.1X: 17.1 / 15.3 / 16.8 req/s) and the settings differ only in latency. First-token P95 ≤ 0.34 s in every cell, so goodput is lost to the end-to-end criterion (code2wav queueing), not to TTFT.
- **Registered contrasts:**

| Hypothesis | Cells | Ratio per block (1 / 2_a2 / 3) | Gate | Verdict |
|---|---|---|---|---|
| H1a goodput, τ 1 s ÷ FIFO16 − 1 | MIX 1.0X | +0.05 % / +1.9 % / +0.09 % | ≥ +5 % in all 6 | **inconclusive** |
| | MIX 1.1X | +87.6 % / +1.6 % / +18.5 % | | |
| H1b short-request queue P99, τ 1 s ÷ Length8 | MIX 1.0X | 0.88 / 1.26 / 0.88 | ≤ 0.8 in all 6 (audio ratio 0.995–1.003, gate met) | **inconclusive** |
| | MIX 1.1X | 0.97 / 1.84 / 2.30 | | |
| H1c e2e P95, τ 1 s ÷ FIFO16 | W3b 0.9X | 1.046 / 1.100 / 0.991 | ≤ 1.05 in all 9 | **inconclusive** |
| | W3b 1.0X | 0.991 / 1.006 / 0.979 | | |
| | W3b 1.1X | 1.085 / 0.795 / 0.969 | | |

- **Which cells were overloaded.** MIX and W3L at 1.1X: the in-flight count kept growing in blocks 1 and 3 under FIFO16. Block 2_a2's 1.1X draw was lighter (MIX: 15.3 req/s realised and 338 audio-s/s offered, against 17.1 / 16.8 req/s and 380 / 366 audio-s/s in blocks 1 and 3), and its FIFO16 MIX cell stayed steady. W3b was steady in all 45 cells: 32 req/s (1.1X) is still below W3b's open-loop capacity on 2+2+4, so H1c was tested below saturation only.
- **Length8 against FIFO16 at overload (descriptive, not a registered test; order and batch cap change together):**

| Cell | Goodput FIFO16 → Length8 (req/s) | e2e P95 FIFO16 → Length8 (s) | Padding FIFO16 → Length8 | Requests per batch FIFO16 → Length8 |
|---|---|---|---|---|
| MIX 1.1X | 6.6 / 14.4 / 11.4 → 15.4 / 15.1 / 16.1 | 20.3 / 9.7 / 17.2 → 11.5 / 8.1 / 9.7 | 33 / 21 / 29 % → 25 / 17 / 21 % | 8.9 / 2.3 / 4.0 → 4.4 / 1.9 / 2.6 |
| W3L 1.1X | 2.2 / 8.2 / 4.3 → 13.9 / 15.0 / 14.3 | 32.2 / 24.5 / 38.2 → 18.1 / 13.2 / 18.3 | 19 / 16 / 19 % → 7 / 11 / 7 % | 14.9 / 8.8 / 12.5 → 7.8 / 7.2 / 7.3 |

  Below saturation the policy hardly matters for MIX and W3b (all five settings within 3 % on goodput at 0.9X and 1.0X; MIX 1.0X means 14.6–14.8 req/s). Long answers (W3L) already separate at 1.0X: FIFO16 is unsteady in 2 of 3 blocks and 2–15 % below Length8 (12.5 / 14.0 / 14.4 vs 14.7 / 15.0 / 14.5 req/s), i.e. 1.0X is already past FIFO16's W3L capacity; at 0.9X block 1 FIFO16's e2e P95 is 15.7 s against Length8's 10.0 s.
- **What the maximum wait did at overload.** At 1.1X a large share of batches contained forced (aged) requests: MIX τ 0.5 / 1 / 2 s: 65 / 10 / 44 %, 69 / 6 / 26 %, 40 / 1 / 10 %; W3L: 99 / 71 / 100 %, 92 / 84 / 92 %, 96 / 48 / 97 %. Goodput fell below Length8's wherever the cell was overloaded (MIX τ 1 s 12.5 / 14.7 / 13.5 against 15.4 / 15.1 / 16.1; W3L τ 1 s 4.3 / 15.0 / 7.4 against 13.9 / 15.0 / 14.3; τ 0.5 s and 2 s alike), padding rose back (W3L τ 1 s 16 / 14 / 15 % against 7 / 11 / 7 %), and short requests did not wait less (short-request queue P99 longer in 4 of the 6 block × workload cells; MIX τ 1 s 5.1 / 3.4 / 7.0 s against 5.2 / 1.9 / 3.0 s; W3L 19.4 / 5.6 / 19.9 s against 10.8 / 5.8 / 11.4 s).
- **Starvation.** The share of requests whose code2wav queue exceeded 3 × FIFO16's P95 of the same length group, block and rate was ≤ 0.6 % in 134 of 135 cells (2.5 % in one W3b 1.1X τ 1 s cell); Length8 without a maximum wait never exceeded 0.4 %.
- **Stable capacity** (highest steady rate per block; “≥” = the top rate was steady, a lower bound): MIX FIFO16 14.5 / ≥ 16 / 14.5, Length8 and every τ ≥ 16 / ≥ 16 / 14.5; W3L FIFO16 14.5 / 13 / 13, Length8 ≥ 16 / ≥ 16 / 13, τ 1 s 14.5 / ≥ 16 / 14.5; W3b ≥ 32 for every setting. Several W3L cells sit near the steadiness threshold, so these steps are noisy.

**Result.** All three registered hypotheses are **inconclusive**: H1a because at 1.0X the maximum wait changes goodput by +0.05 to +1.9 % and at 1.1X the gain over FIFO16 (+1.6 to +87.6 %) depends on whether the block was overloaded; H1b because the maximum wait never lowered the short-request queue P99 to 0.8 × Length8's, and at 1.1X it raised it in two of three blocks (× 1.84, × 2.30); H1c because two of nine W3b cells exceed the 5 % margin (× 1.10 and × 1.085) while the others are within it. Descriptively, the maximum wait is a negative result: wherever the service was overloaded it gave back part of Length8's goodput (W3L block 1: 13.9 → 4.3 req/s), and it protected no one, because Length8 starved no one to begin with. Length8 itself is the only setting that kept goodput near the offered rate past saturation (MIX and W3L 1.1X, all three blocks), but this round cannot say whether length grouping or the smaller batch cap does it.

**Interpretation.** Length8 already bounds waiting: every batch is anchored on the oldest waiting request, so a request is skipped at most until it becomes the oldest. The maximum wait therefore had nothing to fix. Under overload every waiting request is older than τ (queue P99 of several seconds against τ ≤ 2 s), so the rule fills batches in arrival order: Length8 degenerates into first-come batching with a cap of 8, padding returns and goodput falls toward FIFO16's. This is the registered failure mode “τ too small degenerates to FIFO”, here for every τ tested, because the overload queue is longer than any of them. Below saturation the dequeue order only redistributes queueing (the other registered failure mode): for MIX and W3b, 0.9X and 1.0X look the same for all five settings; for W3L, FIFO16 is already at its limit at 1.0X. The batching policy matters only once code2wav is overloaded, and there what helps is fewer, length-matched requests per batch (Hypothesis on the mechanism: at overload FIFO16's batches averaged up to 14.9 requests with 16–33 % padding, so the flow model spends GPU time on padding. On code2wav alone, 实验 10 measured MIX audio throughput over FIFO16 at +7 % for FIFO8, +11 % for Length16 and +30 % for Length8 (3 runs each), so the cap alone explains part of the difference, not all of it; P1 has no FIFO8 arm to separate them on the full service). For the system: a stage policy cannot replace admission control; past capacity, the queue grows under every setting and only the slope differs.

**Limitations.** One node; synthetic W3b/W3L/MIX; open loop at three rates only, and “1.1X” was not the same load in every block (block 2_a2's draw was lighter); unsteady cells are window-dependent, so their goodput describes these 75–90 s, not a capacity; Length8 and FIFO16 differ in order and cap; W3b never reached saturation, so H1c covers the unsaturated regime only; X came from one pre-run (E-P1-X); block 1 had stack dumps in its drains, 2_a2 and 3 none (D3); the interaction with packed flow was not run because C1 failed its quality gate (NOT APPLICABLE).

**中文小结。** P1 在完整服务（2+2+4）上，用开环泊松到达把负载推到 0.9X、1.0X、1.1X，比较 code2wav 的 5 种组批：FIFO16、Length8，以及 Length8 加 0.5/1/2 秒的最长等待。3 个区组、135 个格子全部有效，没有一个请求失败。登记的三个假设都是**不确定**：最长等待 1 秒相对 FIFO16 的 goodput 在 1.0X 只差 +0.05% 到 +1.9%；它没有把短请求的排队 P99 降到 Length8 的 0.8 倍，过载时反而在两个区组里升到 1.84 倍和 2.30 倍；W3b 有两格超过 5% 的容差。描述性的结果是：只有过载时组批策略才有影响。MIX 和 W3L 在 1.1X 时，Length8 的 goodput 在三个区组都保持在提供负载附近（W3L 13.9/15.0/14.3 请求/秒，FIFO16 只有 2.2/8.2/4.3）；加上最长等待以后，过载时大部分批次变成强制选入（W3L 48–100%），实际退化成按到达顺序组批，goodput 又掉回去一大截（W3L 区组 1 从 13.9 掉到 4.3）。原因是 Length8 每一批都以最老的请求为锚，本来就不会饿死请求（超过 FIFO16 同组 P95 三倍的比例最多 0.4%），最长等待没有东西可保护。注意：Length8 同时改了顺序和批上限（8 对 16），这一轮分不开两者各自的作用；1.1X 在区组 2 的实际负载偏轻；W3b 在 32 请求/秒仍未饱和。

---

## R4 — Do multimodal robot sessions put the thinker's KV cache under pressure?

**Research question.** Do multimodal robot sessions create a working set large enough to cause meaningful KV-cache eviction and recomputation in the thinker, and if reuse is lost, is it lost to capacity (eviction) or to request-level replica routing?

**Hypotheses and gates (registered 2026-11-18 before any R4 run; `r3_r4_design.md` §3.5; analysis corrected by Erratum E-R4-REF before blocks 2–3 were analysed).** Per level, *eviction pressure* = reuse lost to eviction ≥ 10 % of the level's prefill tokens **and** those requests' TTFT median ≥ 1.2 × that of clean pairs; *routing cost* likewise for reuse lost to a replica change. Conclusion in order: eviction pressure anywhere → A/B of session affinity and cost-aware eviction; else routing cost → session-affinity A/B only; neither → **“当前实验条件下，尚无证据支持引入更复杂的 Agent-aware Cache Management。”** Stable only if all 3 blocks agree. Ratios (working set ÷ capacity) are reported but do not decide. ThunderAgent and FlashStates are not ported.

**Setup.**
- Runtime `unicorn_ppv_kv` (520b49ee + default-off KV logging patch, `SGLANG_OMNI_EXP_KV=1`); 2+2+4 request-response path with two thinker replicas behind request-level routing; request events on.
- Synthetic robot sessions (`src/omni_research/sessions/robot_sessions.py`): a unique preamble per session (only the system prompt is shared); every turn a new synthetic camera frame, a unique synthetic spoken request and a caption; half the turns first ask for a short text plan, wait for a simulated tool (median 0.3 s) and add a 40–200-token tool result; spoken answers; 4–10 turns, think time median 2 s. Earlier turns go back as text; control: the last two turns' image and audio re-sent (`history_media=2`).
- One run per block: 8 → 16 → 32 → 64 → 128 sessions, then 32 sessions with media history; same content in every block; 3 blocks 08:24–08:56, nothing else on the node. Smoke passed before (every request joined to its prefill line; ≈ 308 media tokens per turn).

**Evidence** (`results/r4_working_set.json`, figure `fig12_r4_working_set`; Measured; blocks 1 / 2 / 3; reference per E-R4-REF):

| Sessions | Working set ÷ capacity (peak) | Reuse lost to eviction | Reuse lost to routing | Routed TTFT ÷ clean (registered) | … within a turn (plan → answer) | … next turn | Clean pairs | Eviction pressure | Routing cost |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.05 / 0.04 / 0.05 | 0 / 0 / 0 % | 23 / 29 / 27 % | 1.28 / 1.21 / 1.27 | 2.02 / 2.25 / 2.55 | 1.27 / 1.15 / 1.16 | 47 / 36 / 41 | no ×3 | **yes ×3** |
| 16 | 0.06 / 0.07 / 0.07 | 0 / 0 / 0 % | 24 / 27 / 27 % | 1.53 / 2.22 / 1.62 | 3.36 / 2.80 / 4.12 | 1.14 / 1.08 / 1.01 | 85 / 66 / 77 | no ×3 | **yes ×3** |
| 32 | 0.15 / 0.14 / 0.16 | 0.14 / 0.05 / 0.05 % | 27 / 25 / 26 % | 1.07 / 1.14 / 1.08 | 1.72 / 1.70 / 1.74 | 0.95 / 0.94 / 0.86 | 125 / 118 / 94 | no ×3 | no ×3 |
| 64 | 0.28 / 0.25 / 0.25 | 3.0 / 2.2 / 2.9 % | 27 / 26 / 28 % | not comparable | — | — | 0 / 0 / 2 | no ×3 | undetermined ×3 |
| 128 | 0.53 / 0.49 / 0.53 | 2.4 / 2.2 / 2.6 % | 33 / 33 / 34 % | not comparable | — | — | 1 / 0 / 0 | no ×3 | undetermined ×3 |

- **Validity.** No request failed in any level of any block; every request was joined to its prefill line; both thinker replicas logged in every block.
- **Capacity.** Each thinker replica holds 127,039 KV tokens; running requests never held more than 43,069 of them (128 sessions) and no request ever waited for KV. The pool reads 100 % resident from 32 sessions on only because the radix cache keeps finished prefixes until it needs the space; the logical working set peaks at about half of capacity at 128 sessions.
- **Eviction.** Evictions start at 32 sessions and reach 714–724 k tokens per run (both replicas) at 128 sessions, but they almost never hit a prefix a session needed next: ≤ 3.0 % of prefill tokens at every registered level (8–128 sessions). The media-history control, which does not enter the verdict, is the exception: 12.2–13.9 % lost to eviction (run-level reference; TTFT not comparable there), because re-sent media make prompts much longer.
- **Routing.** 40–57 % of a session's consecutive requests land on the other replica, and each recomputes the session's history: 23–34 % of all prefill tokens at every level. At 8 and 16 sessions, in all 3 blocks, those requests' median first token is ≥ 1.21× that of clean pairs (registered routing cost); at 32 sessions 1.07–1.14×; at 64–128 sessions too few clean pairs remain to compare.
- **Where it hurts (descriptive, not registered).** Split by transition, a routed answer that follows its own plan is 1.70–4.12× slower to its first token than a clean one at 8–32 sessions; a routed next-turn request 0.86–1.27×, because it brings new media anyway and reuses only ~60 % of its prompt even on the same replica.
- **Media-history control (32 sessions, descriptive).** Re-sending the last two turns' media cuts next-turn reuse from 0.53–0.55 to 0.23–0.24 (all pairs) and raises median TTFT from 0.78–0.85 s to 1.65–1.87 s.
- The registered code's verdict (`verdict_v1_registered_code`) gives the same conclusion, but its 64–128-session entries are unreliable: zero loss where a level had no clean pair, and a TTFT comparison against 1–2 clean pairs where it had one or two (e.g. routing cost “true” at 128 sessions in block 1).

**Result.** Stable across all 3 blocks: **no eviction pressure, but cross-replica routing loses reuse that costs first-token time → the registered next step is a session-affinity routing A/B, not an eviction policy.** The registered sentence for the “neither” case does not apply, because routing does cost time. For eviction-side cache management specifically, nothing in the registered levels calls for it: eviction cost ≤ 3 % of prefill even when the working set reached half of the KV pool; only the media-history control (re-sent images and speech) lost 12–14 % to eviction, which is a reason to keep history as text rather than evidence for a new eviction policy.

**Interpretation.** With two replicas and request-level routing, KV reuse in agent sessions is lost to placement, not to capacity: a session's history lives on one replica and about half of its requests are sent to the other. The cost is largest exactly where agent turns are tight — the spoken answer right after a tool call — and at low load, where prefill time is a large share of first-token time. A session-affinity rule (or cache-aware routing) addresses this directly and more cheaply than a new eviction policy.

**Limitations.** One node; synthetic sessions (no real robot, camera or speech); text history by design; two replicas only; the TTFT criterion could not be evaluated at 64–128 sessions; the within-turn / next-turn split is descriptive; the previous level's evictable cache shares the pool with the next level (only own-session losses are counted).

**中文小结。** R4 用合成的机器人会话（每轮新画面、新语音，一半的轮次先计划、调用工具、再回答）在 2+2+4（两个 thinker 副本）上从 8 个会话加到 128 个，3 个区组结论一致：**没有驱逐压力**（驱逐造成的复用损失最多占 prefill 的 3%，128 个会话时工作集也只占 KV 容量的一半左右），但**路由有代价**：约一半请求被分到另一个副本，每一档都有 23–34% 的 prefill 在重算会话历史；8、16 个会话时这些请求的首个 token 慢 1.21–2.22 倍，最疼的是“工具调用之后的回答”（慢 1.7–4.1 倍，描述性）。按登记规则，下一步是会话粘性路由的 A/B，而不是新的驱逐策略；登记的“尚无证据支持引入更复杂的 Agent-aware Cache Management”这句话针对的是“两种都没有”的情形，这里不适用。分析途中发现登记规则在缓存池满时没有定义（没有“干净的一对”），按勘误 E-R4-REF 在分析区组 2、3 之前修正，原代码的判定一并保留，结论相同。

---

## P2 — Is the side-stream hidden-state copy bit-exact?

**Research question.** 实验 11's asynchronous hidden-state copy (`sidecopy`) failed its correctness check, so its performance was never benchmarked. With a verify mode that keeps a launch-stream reference copy of every row, is the side-stream copy bit-exact in every finished request?

**Gate (registered; `p1_p2_p3_design.md` §2.3).** Pass only if every `exp_hidden_verify` event has all four checks true, every speech request that finished in a check phase has exactly one verify event, the server log has no mismatch warning, and there is no client error except the deliberate timeouts and no restart. Any failure → stop, report, no performance runs. 实验 11's original failure is not rewritten.

**Setup.** Runtime `unicorn_ppv_verify` (实验 11's patch + `sidecopy_verify`; only one file differs); 2+6; 实验 11's check phases (greedy load at concurrency 1/8/32/64, text/speech, short/long, with/without seed, a client-timeout abort phase and the phase after it, W3XL), events and digests on; run `ppv_p2_verify`, 08:56–09:01.

**Evidence** (`results/p2_verify.json`; Measured). 885 verify events, all four checks true in every one; each of the 662 speech requests that finished in a check phase has exactly one (0 unverified, 0 duplicates); the deliberate client-timeout phase has none, as designed; 0 mismatch warnings in the log, no erratum on expected rows, no restart.

**Result.** **Gate passed**: under verify timing the side-stream copy is bit-exact in all 885 verified copies (662 finished speech requests). This does not turn 实验 11's failed check into a pass: that run used a different code path (no reference copy, different timing) and its failure stands. Following the registration, the 9 performance runs (实验 11's design, runtime `unicorn_async`, whose worktree was checked to equal 520b49ee + 实验 11's patch file by file) were queued at 10:19 and finish after the presentation; they are not reported here.

**Limitations.** Verify checks the copy path, not the snapshot itself (if the graph buffer were overwritten before the snapshot, both copies would agree on wrong data); verify adds a clone, a blocking copy and a device sync per request, so “bit-exact under verify timing” is not a proof for plain `sidecopy` timing.

**中文小结。** P2 用加了“参照拷贝”的检查模式重跑实验 11 的检查段落：885 次校验全部逐位相等（覆盖检查段落里 662 个完成的语音请求），没有一处不一致，登记的门槛**通过**。这只说明在检查模式的时序下旁路拷贝是对的；实验 11 原来“未通过”的结论不改。按登记，9 个性能运行已在 10:19 放入队列，结果在报告之后才有。

---

## R2b — With the thinker on a second GPU, do 8 duplex sessions stop failing?

**Research question and registration (2026-11-19 04:00, after R2, before any R2b run; erratum E-R2b 05:48).** R2 on one GPU ran out of memory at 8 sessions. Does the registered two-GPU placement `ms8_2gpu` (thinker alone on a second GPU, memory fraction 0.85) remove the failures and keep units on time? Supported if 8 sessions meet R2's serving target in all 3 blocks without OOM; refuted if no block's 8-session level meets it (including when the stop rule skips it); the smoke had already shown every unit late (E-R2b).

**Setup.** Stock runtime, `--variant session --enable-realtime --max_sessions 8 --thinker.gpu 1 --thinker.gpu_memory_fraction 0.85`; 4 then 8 sessions × 180 s (stop rule: 8 skipped if 4 fails); 3 blocks 09:01–09:15. All three blocks got the same GPU pair (boards d4ff46d0 + 51b87964), linked through the PCIe host bridge (`NODE`, no NVLink); the smoke's pair was `PIX`.

**Evidence** (`results/r2b_summary.json`; Measured). At 4 sessions, in every block: all 4 sessions failed and 100 % of units missed (lag P50 24.3 / 26.2 / 26.2 s); no user turn got a timed answer, so TTFA is undefined (some audio was produced, far behind real time). The thinker's GPU read 91–94 % busy at 186–188 W with 41.6 GiB (42,556 MiB) used; the other GPU 10–13 % (utilisation suspect, aliasing) at 157–158 W; the thinker process used ~1.7 CPU cores. No OOM line. The stop rule skipped 8 sessions in all blocks. On one GPU (R2), 4 sessions missed 0 % and 0.28 % of units in the two blocks that reached that level (pooled 0.14 %; R2's own serving target was still not met there because of failures elsewhere in the run).

**Result.** **Refuted**: moving the thinker to its own GPU did not rescue 8 sessions; it broke 4 sessions that kept their units on time on one GPU. The cause is unknown (Hypothesis: the thinker path behaves differently when it is on another card — e.g. the cross-GPU hand-off of the duplex session state over a non-NVLink link — the thinker GPU's high utilisation says it is busy, not idle; not verified).

**Limitations.** One GPU pair, not controlled (orchestrator choice); no profiling; the duplex path's multi-GPU support at the pinned commit is untested upstream on A6000s.

**中文小结。** R2b 把 thinker 单独放到第二张卡上，想解决 R2 在 8 个会话时的显存失败。结果 3 个区组在 4 个会话时就全部失败：单元 100% 超时、延迟中位数 24–26 秒，按停止规则没有跑 8 个会话，登记判定为**反驳**。而同样 4 个会话放在一张卡上（R2）时，跑到这一档的两个区组单元错过率是 0% 和 0.28%。原因不明；thinker 那张卡 91–94% 忙，两张卡之间不是 NVLink（假设，未验证）。

---

## P3 — What were 实验 9's short full-service stalls?

**Research question.** 实验 9 saw two short full-service pauses (one of 3.09 s with no line in the shared server log). Do such pauses recur, and when they do, is the shared network file system (NFS), garbage collection or something else involved?

**Design (registered; no decision rule beyond “report every gap with its evidence”).** Low-overhead observation in every run of this round: Linux pressure-stall information (PSI) and process states in the runner's 1-s sampler; gen-2 GC timing (P1 runtime); a live watcher that stats the server log every 0.1 s and records silences > 1 s during load phases and stats slower than 0.2 s. Stack dumps through SIGUSR2 were switched off after one crashed a code2wav process (F6, D3). A gap counts as *under load* only between a phase's first and last send. The registered NFS-vs-local-disk control was queued at 10:19 after this report (go decision and expected power written before it ran, `preregistration.md`).

**Evidence** (scan of 36 runs, 19 with dense logs, ≈ 6.5 h of dense-log measurement; `scripts/analysis/ppv_p3_stalls.py`; Measured).
- 11 under-load log silences > 1 s (1.03–1.26 s). Eight fall in the first 0.4–1.8 s of a closed-loop level, before any of its requests could complete, with no I/O pressure; two are quiet logging while completions continued (W3L open loop). One is a real stall (next bullet).
- **One full-service stall under load:** P1 block 2_a2, 07:37:48, cell W3b / Length8 + τ 1 s / 0.9X (inside its measured window). The client received no completion and no first audio for 1.30 s while it kept sending (28 requests; ≈ 24 completions/s around it); the server log was silent for 1.075 s; in that second the node's PSI recorded 0.66 s of full I/O stall, one server process was in uninterruptible I/O sleep and none was running, and the watcher's own stat of the server log on NFS took 1.115 s (so the watcher itself was blocked and logged no pause); no gen-2 GC overlapped. This cell is one of the two H1c cells over the 5 % margin; P1's verdicts are unchanged (no registered exclusion for stalls; H1c inconclusive either way).
- Client-side stalls with the same I/O signature in duplex runs: R0 (2.25-s sampler gap, I/O full 474 ms/s, two client processes in uninterruptible sleep) and R2 block 1 (1.34 s, both client processes in uninterruptible sleep).
- Watcher: 267 log silences in 13 runs, 255 of them in drains after the last send (a cell's long requests finishing); slow stats up to 1.5 s; 154 gen-2 collections in measured phases, longest 747 ms, none overlapping a gap. Stack dumps at drain silences (P1 block 1, 120 process dumps): main threads idle in the event loop, the only active work code2wav finishing the last requests; no thread in file I/O.

**Result.** A stall was observed, so the registered no-stall sentence does not apply. **Root cause remains unknown**, but the one full-service stall coincided with a node-wide I/O stall, a server process blocked in I/O and a 1.1-s metadata call on the shared NFS, and the duplex client stalls show the same signature: blocking I/O on the shared file system is now the leading candidate (Hypothesis), GC is not (no overlap). One event shows coincidence, not cause.

**Interpretation.** Every stage process logs synchronously to the shared NFS home; if one write or metadata call blocks for a second, that process blocks, and because the stages wait on each other one blocked process can stop the whole pipeline. The control (NFS vs node-local disk, ABBA) tests this, but at one stall in ~6.5 h its 1 h per arm has about a 14 % chance of catching a stall in the NFS arm.

**Limitations.** Rare events; 1-s process-state sampling (the blocked process is not identified by name); PSI is node-wide (other users' I/O shows the same way); no stack for the stall (dumps were off after F6).

**中文小结。** 本轮在约 6.5 小时的日志密集测量里，负载期间只出现了 **1 次** 全服务停顿（P1 区组 2_a2，07:37:48，客户端 1.3 秒收不到任何完成），同一秒里整台机器的 I/O 压力有 0.66 秒是“全部卡住”，一个服务进程处于不可中断的 I/O 睡眠，监视器自己在 NFS 上 stat 日志也卡了 1.1 秒；没有 GC 重叠。其他 10 次日志静默都不是停顿（闭环刚开始的正常静默，或者只是日志安静）。全双工的客户端也有两次同样特征的卡顿。所以“没有观察到停顿”那句固定措辞不适用；根本原因仍未确定，共享文件系统上的阻塞 I/O 是目前最可能的候选（假设）。本地盘对 NFS 的对照已放进队列，但按这个频率它很可能什么也看不到（每组约 14% 的机会）。

---

## Annotation of every performance result

Common to all rows unless stated: MiniCPM-o 4.5 (revision `fb09628e`), SGLang-Omni `520b49ee` (SGLang 0.5.21, torch 2.13.0), RTX A6000 48 GB, one node, Slurm job 454290.

| Exp. | 1 Model / version (runtime) | 2 GPU | 3 Workload | 4 Concurrency / arrival rate | 5 Batch setting | 6 GPU placement | 7 Repetitions | 8 Measurement window | 9 Metric definition | 10 Baseline | 11 Uncertainty / limitations |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | stock | 1 | synthetic spoken questions (duplex-v1), real-time paced 16 kHz | 1 and 2 sessions | server default (`--variant session --enable-realtime`) | all duplex stages on one GPU | 1 functional run | 300 s per level | unit on time = 1-s unit finished within its second; 5 registered conditions | none (pass/fail) | single run; synthetic speech |
| R1 | stock | 1 | duplex-v1 sessions with scripted barge-in; chat reference: 48 one-sentence requests | 1 session; chat: one request at a time | default; chat max_tokens 64 | one GPU | 3 blocks | phase A (no events) and B (events) per block | TTFA = last packet of the utterance → first audio delta; a + b + c decomposition | request-response path (different semantics) | t interval (df 2) of block P50s; N = 37 turns; block 1 overlapped C1/C2 (D1) |
| R2 | stock | 1 | duplex-v1 sessions | 1/2/4/8 sessions | `max_sessions 8` | one GPU (ms8_1gpu) | 3 blocks (levels 2–8: 2 blocks, stop rule) | 180 s per level | unit miss, unit lag, deadline miss (1 s; 0.5/2 s sensitivity); capacity = takewhile | the 1-session level | warm-up and client stall in level 1; OOM at 8 |
| C1 | flow DiT of 520b49ee | 1 | real batch shapes exported from 实验 10 (W3b/MIX fifo16, length8, W3L, exp10val) | n/a (microbenchmark) | dense (D), packed (P0), packed + compile (P1, BF16), FP16 packed (P1h, exploratory) | one GPU | 3 timed passes | CUDA events per shape group | E = code2wav GPU ms per audio second; MCD vs FP32 reference; ASR | D | microbenchmark only; quality gate failed; MCD noise floor |
| C2 | HiFT of 520b49ee | 1 | 3 real shape groups | n/a | A default, B fold, C channels-last, D FP16 body | one GPU | 3 timed passes | CUDA events | HiFT GPU ms per audio second | A | ±5 % pass-to-pass; overlapped R1 block 1 |
| P0 | `unicorn_instr` (= 实验 9 instrumentation) | 8 | W3b, W3L | closed loop 64/128/256 in flight; open loop W3b 15.5–31 req/s, W3L 8–12 req/s | talker `max_running_requests` 32 (A) vs 64 (B; graph max bs follows, Erratum E-P0-G) | 2+1+5 (A, B), 2+2+4 (reference) | 3 blocks | closed-loop levels; open loop 60 s (W3b) / 90 s (W3L) | capacity = completed / (last completion − first send); goodput (TTFT ≤ 1 s and e2e/audio ≤ 0.5) | arm A | t interval df 2; joint cap + graph change; H0b CPU criterion weak (M1) |
| R3 | `unicorn_ppv_kv` (default-off preprocessing-concurrency patch) | 1 per arm (+ 7 CPU cores) | synthetic camera frames (PNG) / 2-s clips (H.264), circle-count question | screening 1/4/16 sessions × 0.5/1/2 fps; confirmation the 5 saturated cells; one request in flight per session | preprocessing concurrency 1 vs 4; server batching default | one `--text-only` replica per arm, arms side by side on two GPUs | screening 1 run per arm; confirmation 3 blocks | 60 s per cell after 10 s warm-up | saturated = ≥ 10 % dropped or ≥ 20 % late; class by preprocessing share/slot and GPU busy; improvement = drop −5 points and −20 % relative, or first-response P95 −20 % | concurrency 1 | arm and GPU board confounded (M9); nvidia-smi aliasing at light load; small synthetic frames; slot metric over-counts |
| P1 | `unicorn_ppv_maxwait` (实验 10 batching patch + max-wait rule + gc log) | 8 | W3b, W3L, MIX (实验 9/10 synthetic workloads) | open-loop Poisson at 0.9X/1.0X/1.1X: W3b 26/29/32, W3L and MIX 13/14.5/16 req/s (X from the FIFO16 pre-run, E-P1-X) | code2wav FIFO16, Length8, Length8 + τ 0.5/1/2 s (window 64) | 2+2+4 (`t2_k2_c4`) | 3 blocks (block 2 re-run as 2_a2, D3) | 75 s (W3L 90 s) per cell after 15 s (20 s) warm-up dropped by scheduled send time | goodput = measured requests with TTFT ≤ 1 s and e2e/audio ≤ 0.5 ÷ send window; code2wav queue = dispatch → dequeue | FIFO16 (H1a, H1c), Length8 (H1b) | t interval df 2 over block means; open loop at 3 rates; block 1 had stack dumps in its drains; stable capacity is a lower bound when the top rate is steady |
| R2b | stock | 2 | duplex-v1 sessions (same content as R2) | 4 and 8 sessions | `max_sessions 8`, thinker `gpu_memory_fraction 0.85` | thinker on its own GPU, other duplex stages on the first (`ms8_2gpu`); GPU pair from `logs/env/nvidia_topo.txt` | 3 blocks | 180 s per level | as R2 (unit miss, lag, deadline miss, OOM lines) | R2 `ms8_1gpu` (descriptive only) | the smoke had every unit late (E-R2b); GPU pair not controlled |
| R4 | `unicorn_ppv_kv` (KV pool / prefill events, default-off) | 8 | synthetic robot sessions: unique preamble, a new camera frame + synthetic speech per turn, half the turns with a tool call; text history (control: last 2 turns' media) | 8 → 16 → 32 → 64 → 128 sessions, then 32 with media history; 4–10 turns, think time median 2 s | server default | 2+2+4, two thinker replicas, request-level routing | 3 blocks | one phase per level, each until its sessions finish | lost reuse (recomputed tokens) by cause: eviction / routing / structural; TTFT of those requests vs clean pairs; working set ÷ KV capacity | clean pairs (same replica, no eviction in between) | one node; synthetic sessions; leftover prefix cache from the previous level shares the pool (only the level's own misses are judged) |
| P2 | `unicorn_ppv_verify` (实验 11 patch + `sidecopy_verify`) | 8 | 实验 11 check phases: greedy load at concurrency 1/8/32/64, text/speech, short/long, with/without seed, client-timeout abort phase, W3XL | per phase as in 实验 11 | server default | 2+6 | 1 run (correctness gate) | every finished speech request of every check phase | gate: all four checks true in every `exp_hidden_verify` event, one event per finished speech request, no mismatch line, no client error except the intended timeouts, no restart | the launch-stream reference copy | checks the copy path, not the snapshot itself; verify timing differs from plain sidecopy |
| P3 | observation in every run (PSI, process states, gc log and live watcher in P1-runtime runs) | all GPUs of each run | whatever each run ran | — | — | — | every run once | each run's load phases (send window = first to last send) | under-load log gap > 1 s; watcher pause > 1 s while armed; sampler gap; slow stat > 0.2 s; gen-2 GC | none | dense-log time only for the watcher; stack dumps stopped after F6 (D3) |

---

## Synthesis — what the evidence says about the three research questions

Numbering follows the task brief (system bottleneck / compiler and runtime / real-time robotics); in `../robotics_application_intro.md` §5 these are RQ1 / RQ3 / RQ2. Labels: Measured, Derived, Estimated, Hypothesis, Unknown, Blocked.

### RQ1 — Where is the system-level bottleneck of omni serving on one 8-GPU node?

- **Evidence.** 实验 9 (placement, 3 blocks); P0 (talker admission cap, 3 blocks); P0 stage residence; P1 (code2wav batching past saturation, 3 blocks, 135 cells); P3 (stall observation in every run).
- **Main result (Measured).** Capacity is set by how the GPUs are split *and* by per-stage software limits. In 2+1+5 the talker's admission cap of 32 bound the whole service: raising it to 64 added 30–32 % capacity in every block while the talker GPU stayed 53–58 % busy. In 2+2+4 code2wav holds most of a request's residence (61 %), and its batching policy matters only near and past code2wav's saturation: for MIX and W3b at 0.9X–1.0X all five P1 settings are within 3 %, while W3L under FIFO16 is already unsteady at 1.0X; past saturation, length-aware batching keeps goodput near the offered load where first-come batching collapses (W3L 1.1X: 13.9 / 15.0 / 14.3 vs 2.2 / 8.2 / 4.3 req/s; descriptive).
- **Failures / negative results.** H0c refuted (2+1+5 with cap 64 is still 3.7 % behind 2+2+4); arm G blocked by the pinned version, so admission and CUDA-graph coverage cannot be separated; P1's registered H1a–H1c inconclusive and the maximum wait a negative result.
- **Remaining unknown.** Whether a cap of 128 helps further; how much of Length8's overload gain is length grouping and how much the smaller cap; the root cause of the rare full-service stalls (one in ~6.5 h, coinciding with an NFS I/O stall).
- **Next hypothesis.** A downstream-aware admission rule (bounding what the talker hands to an overloaded code2wav) keeps goodput near the offered load past saturation under every batching policy; Length8's overload gain comes mostly from length grouping (FIFO8 on the same cells recovers less than half of it).

### RQ2 — Which compiler and runtime techniques reduce the cost of the speech stages?

- **Evidence.** Kernel census (existing); C1 (flow dense vs packed, microbenchmark on 112 real batches per arm); C2 (HiFT); P1 (runtime batching); P2 (side-stream copy correctness).
- **Main result.** Packed variable-length attention plus compilation cuts code2wav GPU time per audio second by 13–42 % on batched real shapes (Measured, microbenchmark; masked attention −59 %, mask preparation removed) and does nothing at ~1.2 requests per batch; HiFT gains nothing from folding, layout or FP16 (per-call overhead dominates). The hidden-state side copy that failed 实验 11 is bit-exact under the verify mode in all 885 verified copies (Measured; performance runs queued, not reported).
- **Failures / negative results.** The registered packed path runs in BF16 and fails the quality gate (MCD +1.01 dB), so no end-to-end test ran; FP16 HiFT is slower; the maximum wait hurts at overload.
- **Remaining unknown.** End-to-end value of FP16 packed flow at saturation; whether sidecopy pays off (P2 performance runs pending).
- **Next hypothesis.** FP16 packed flow lowers code2wav GPU time per audio second in the saturated full service by roughly its microbenchmark gain on Length8 batches (−25 % on MIX) and keeps MCD within +0.1 dB of the dense path.

### RQ3 — What does real-time multimodal robot interaction need from the serving system?

- **Evidence.** R0 (duplex functional), R1 (TTFA decomposition), R2 + R2b (sessions and a two-GPU split), R3 screening + confirmation (camera), R4 (agent working set).
- **Main result (Measured).** The duplex path works on one A6000 and keeps units real-time for up to 4 sessions; the first spoken response (1.36 s median) is dominated by the unit structure and the model's turn-taking (compute ≈ 0.35 s); at 8 sessions GPU memory fails first. Camera input with single frames is GPU-bound by the registered rule (≈ 7–8 frames per second per A6000); for two-second clips the rule says “cause unclear” and GPU-bound is our reading (Derived: no preprocessing queue, ≤ 11.5 % of server time; 9–11 clips per second). CPU preprocessing never limited a cell. In agent sessions KV reuse is lost to request-level routing across replicas (23–34 % of prefill; the answer after a plan 1.7–4.1× slower to first token), not to eviction (≤ 3 %).
- **Failures / negative results.** R2's registered capacity verdict is “below the lowest level”; the two-GPU split (R2b) is refuted and breaks even 4 sessions (all units late); preprocessing concurrency 4 improved no camera cell in any block; interruptions are not honoured (5.1 s).
- **Remaining unknown.** Why the thinker-on-its-own-GPU placement fails; real camera and microphone behaviour; the duplex path with a cancel mechanism.
- **Next hypothesis.** Session-affinity routing removes most of the routed recompute (from ~25 % to < 5 % of prefill) and cuts the answer-after-plan first-token time by ≥ 40 % at 8–16 sessions; a turn-taking change (end-of-turn trigger or shorter units) lowers duplex TTFA by ≈ 0.5 s at the median without raising unit misses.
