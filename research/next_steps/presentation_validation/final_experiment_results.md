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
