# R0 全双工路径的源码审计（sglang-omni 520b49ee）

> 2026-11-18。只读源码和一次 CPU 上的 `config resolve`（登录节点，不占 GPU），**没有在 GPU 上启动过全双工服务**。
> 行号都指 sglang-omni `520b49ee`（上游 1845546e + 两个 pre-Hopper 修复，集群上 `repos/sglang-omni`，工作区干净）。
> 分三类：**源码事实**（读代码或 resolve 输出得到）、**推断**（由事实推出，未测）、**未知**（R0 要回答）。

## 一句话

全双工是另一条流水线：4 个阶段（感知、thinker、talker、语音）默认全在 1 张卡上，客户端按实时节奏送 16 kHz 音频，服务端每凑满 1 秒切一个单元，**一个会话的单元严格一个接一个处理**，每个单元处理完才把这个单元的全部输出（音频、文字）一次发出，再发 `sglang.unit.done`。模型每个单元自己决定“继续听”还是“开口说”；固定版本**没有** `response.cancel`，打断只能靠模型自己停。

## 1. 启动方式和配置（源码事实）

| 项 | 内容 | 位置 |
|---|---|---|
| 启动命令 | `python -m sglang_omni.cli serve --model-path <MiniCPM-o-4_5> --variant session --enable-realtime`；`--variant` 不能和 `--config` 一起用，所以只能用命令行覆盖（如 `--max_sessions 4`、`--thinker.gpu 1`） | `docs/cookbook/minicpm_o.md:69-81`；`sglang_omni/cli/serve.py:64-76`（`--variant cannot be combined with --config`）、`:467-470`（只支持 realtime 的配置必须加 `--enable-realtime`） |
| 只接受 `/v1/realtime` | 这个服务不接 chat completions | `docs/cookbook/minicpm_o.md:114`；`native_config.py:145`（`is_realtime_only = True`） |
| 阶段和放置 | perception（0.12）→ thinker（0.52）→ talker（0.15）→ speech（0.15），都在 `gpu=0`，合计 0.94 张卡；speech 进程设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | `sglang_omni/models/minicpm_o/native_config.py:24-67` |
| 采样默认值 | `greedy false, temperature 0.7, top_k 20, top_p 0.8, repetition_penalty 1.05, force_listen_count 3, max_new_tokens_per_unit 20`；会话可在 `session.update` 的 `sglang.sampling` 里改 | `native_config.py:70-85`；`serve/realtime/schema.py:67-78` |
| 语音阶段 | flow 默认 **FP32**、不编译，10 步；只录流式 chunk 的 CUDA graph（最多 min(8, max_sessions) 路） | `native_config.py:110-118`；`native_stages.py:475-519`；cookbook `:98` |
| 对比 chat 路径的 code2wav | chat 路径 flow 是 **FP16**、DiT 和 HiFT 都编译 | `sglang_omni/models/minicpm_o/config.py:141-160` |
| 历史上限 | 每个会话最多 8192 token，满了发 `context_exhausted` 并关闭会话 | cookbook `:100`；`serve/realtime/runtime.py:500-503` |

`config resolve`（CPU，登录节点）已确认：默认、`--max_sessions 4`、`--max_sessions 8`、`--max_sessions 8 --thinker.gpu 1 --thinker.gpu_memory_fraction 0.85` 都能通过配置校验（每卡显存比例合计 ≤ 1.0）。**能通过校验不等于能启动**（实验 9–11 的教训）。

## 2. max_sessions：在哪里限制、受什么约束

- **限制点**（源码事实）：WebSocket 升级之前，`len(manager.sessions) >= max_connections` 就返回 HTTP 503 “connection capacity exhausted”（`serve/openai_api.py:1443-1450`）；`max_connections = config.max_sessions`（`models/minicpm_o/session_adapters.py:265`）。默认 2（`native_config.py:17`）。
- **显存预留**（源码事实）：`request_slots = max_sessions + 1`（`sglang_omni/admission.py:6`）。thinker 的 `max_running_requests = slots`，KV 按满长会话预留 `slots × 8192` token；talker `slots × 4096` token（`native_config.py:147-205`）。只要改了 thinker 的显存比例或 KV 设置，就不再按会话数预留（`:174-188`）。thinker KV 超过“比例 × 整卡”时启动直接报错（`native_stages.py:441-462`）。
- resolve 的实际值（源码事实）：max_sessions 4 → thinker 5 个名额、40960 token；8 → 9 个名额、73728 token；talker 20480 / 36864 token。
- **显存推算**（推断，Derived）：thinker 每 token KV = 36 层 × 8 个 KV 头 × 128 × K/V × BF16 = 147456 B。max_sessions 2/4/8 → KV 3.6 / 6.0 / 10.9 GB。thinker 的份额是 0.52 × 48 GiB ≈ 26.8 GB，扣掉约 16.4 GB 的 LLM 权重（8.2B × BF16）后约 10 GB 给 KV、激活和 CUDA graph。所以 **4 个会话大概率放得下，8 个会话在默认放置下大概率启动失败**；要 8 个会话，要么把 thinker 单独放一张卡并调大比例（resolve 通过，能不能跑未知），要么减小 KV 预留（会话变长后可能被拒，不采用）。
- 语音阶段每会话状态上限 2 GiB（`speech_state_bytes_per_session`，超了只关这个会话），是上限不是预留（cookbook `:90`）。

## 3. 协议（源码事实）

**客户端 → 服务端**（`serve/realtime/schema.py:169-222`，分发在 `protocol.py:151-210`；每个事件都要 `event_id`）：

| 事件 | 字段 | 备注 |
|---|---|---|
| `session.update` | `session`：`output_modalities`、`instructions`、`sglang.{sampling, reference_audio, max_slice_nums}` | 第一个音频包之前发；`session.created` 后 60 s 内不完成就关闭（`runtime.py:131-144`） |
| `input_audio_buffer.append` | `audio`（base64 PCM16 16 kHz 单声道）、`sglang.seq`（必须连续）、`sglang.t_start_ms`（必须和已收样本数对齐） | 缓冲超过 60 s 音频返回 `buffer_overflow`，要求重发同一个 seq（`runtime.py:241-301`、`types.py:24,41-43`） |
| `sglang.input_image.append` | `image`（base64 JPEG/PNG ≤ 512 KiB）、`sglang.t_ms`（在音频时间轴上的位置） | 每单元最多 4 帧、最多提前 2 个单元（`runtime.py:303-339`） |
| `sglang.input_audio.end` | — | 之后等 `sglang.input_audio.drained` |
| `session.close` | — | 回 `session.closed` |
| `input_audio_buffer.commit`、`response.create` | — | 返回 `not_applicable`（“manual turn commands are not granted”） |
| `response.cancel` | — | **不在 schema 里**，返回 `not_supported` |

**服务端 → 客户端**（`serve/realtime/projection.py:129-294`）：`session.created`、`session.updated`、`sglang.input_audio.accepted`（每个包一次，带 `seq`、`accepted_end_ms`）、`sglang.input_image.accepted`、`response.created`、`response.output_audio.delta`（base64 PCM16 **24 kHz**）、`response.output_audio_transcript.delta/.done`（只要音频时文字走这里）、`response.output_audio.done`、`response.done`、**`sglang.unit.done`（每个单元一次）**、`sglang.input_audio.ended`、`sglang.input_audio.drained`、`error`（`sglang.fatal` 区分是否致命）、`session.closed`。每个由单元产生的事件都带 `sglang.unit_id`、`chunk_seq` 和 `media_time{t_start_ms, duration_ms}`（`protocol.py:80-95`）。

## 4. 1 秒单元怎么切、怎么处理（源码事实）

- 每凑满 16000 个样本切一个单元；输入结束时不满 1 秒的尾巴补零（`tail_policy=pad`）；正好在单元边界结束时，最后一个单元是空的（`runtime.py:390-432`）。第一个单元按 1030 ms 的窗口算 mel（`components/streaming_perception.py:25,32-34`）。
- **一个会话的单元严格串行**：`pump` 切出一个单元后 `await adapter.process(unit)`，等它走完 4 个阶段才切下一个，然后才发 `sglang.unit.done`（`runtime.py:434-496`）。
- **输出按单元整批发**：`CoordinatorAdapter` 把一个单元的输出事件攒起来，等这个单元的 `input_done` 才一起交给发送队列（`serve/realtime/adapters.py:41-43,106-156,158-199`）。所以同一个单元的音频增量和 `sglang.unit.done` 几乎同时到达客户端。
- 每个音频单元在 thinker 里是：1 个单元 token + 10 个音频嵌入（`audio_pool_step = 5`：100 帧 mel → 卷积减半 → 每 5 帧池化一次，`components/audio_encoder.py:81-91`）+ 本单元生成的 token（最多 20 个）；每张图 64 个 token × 切片数（`streaming_perception.py:26,401-466`）。

## 5. “继续听”和“开口说”在客户端怎么区分（源码事实）

- thinker 每个单元的第一个输出 token 是 listen token 就算“听”（`session_adapters.py:128-145`）；前 `force_listen_count = 3` 个单元强制听（`:125`、`native_config.py:81`）。
- 没有文字也没有音频的单元**不产生任何输出事件**，客户端只看到这个单元的 `sglang.unit.done`（`session_adapters.py:182-211`）。所以：某个 `unit_id` 有音频增量 = 开口说；只有文字 = 只出字；什么都没有 = 继续听。回答的开始是 `response.created`，结束是 `response.done`（模型的 turn_eos 或输入结束）。

## 6. 打断（源码事实 + 推断）

- 固定版本没有客户端取消：`response.cancel` 不在 schema 里；`capabilities` 里 `supports_server_interrupt: false`、`supports_truncate: false`（cookbook `:111`）；上游 benchmarks 写明 PR #2070 去掉了 `response.cancel`（`benchmarks/README.md:439-441`）。
- 推断：打断只能是“用户在模型说话时开口，模型在后面的单元里自己改成听”。可以定义并测量“从打断语句的第一个包到被打断回答的最后一个音频增量”的时间，但这是模型行为加单元粒度的结果，不是系统的取消延迟。

## 7. 服务端的计时事件（源码事实）

- 请求事件记录器（`POST /start_request_profile`、`/stop_request_profile`，`serve/launcher.py:415-475`）在全双工下额外记 `session_unit_ready / admitted / stage_started / stage_bypassed / output_emitted / unit_finished`（`profiler/duplex_events.py`；发出点 `pipeline/sessions.py:286-299,368-375`）。时间戳是 `time.time_ns()`（`profiler/event_recorder.py:112,279`）。
- `profiler/duplex_metrics.py:340` 的 `build_duplex_session_metrics` 把它们还原成每个单元、每个阶段的排队（AR 阶段：`scheduler_queue_enter → prefill_start`）、执行和交接时间，以及 `ready_to_finished_ms`；文档说它和客户端测的单元延迟相差几毫秒（`docs/developer_reference/profiler.md:96-165`、`full_duplex_bench.md:266`）。研究仓库的 runner 对任何 phase 都能开关这个记录器，不需要改 sglang-omni。
- 上游客户端侧的服务指标（单元延迟、超过 1 个单元算“错过”、回复首音频延迟、播放缓冲）定义在 `benchmarks/duplex/serving.py:82-180`，目标是“无失败会话、无 503、无节奏违规、单元错过率 ≤ 1%”（`full_duplex_bench.md:243-276`）。本轮的 `omni_research.duplex.metrics` 沿用同样的单元延迟定义。

## 8. 和 chat 路径的区别（源码事实，汇总）

| | chat 路径（实验 1–19） | 全双工（R0–R2） |
|---|---|---|
| 接口 | `/v1/chat/completions`（HTTP，SSE） | `/v1/realtime`（WebSocket），服务**不接** chat |
| 请求单位 | 一轮问答 | 一个长连接会话，每 1 秒一个单元 |
| 阶段 | 预处理/编码 → thinker → talker → code2wav，放置可配（2+2+4 等） | perception → thinker → talker → speech，默认全在 1 张卡 |
| 语音输出 | 实测：三个阶段全做完后一次返回（首音频 ≈ 端到端） | 每个单元做完就发这个单元的音频（设计如此，**未测**） |
| 会话状态 | 只在客户端；服务端只有 thinker 前缀缓存 | thinker、talker 的 KV 跨单元保留（流式会话，`stages.py:196-212`） |
| 声码器 | flow FP16 + 编译 | flow FP32、不编译 |
| 并发上限 | thinker 64、talker 32 个名额 | `max_sessions`（默认 2），按满长会话预留 KV |

所以实验 1–19 的结论**不能**直接搬到全双工上。

## 9. 推断（未测，R0/R1 验证）

1. **首音频有一个和算力无关的下限**：用户最后一个包落在某个 1 秒单元里，这个单元要等后面的静音把它填满才会被切出来（平均约 0.5 s），再加这个单元的处理时间；模型如果还要“听”几个单元再开口，每多一个单元加 1 s。所以全双工的 TTFA 大约是 0.5 s + 单元处理时间 + k × 1 s（k = 决定开口前多听的单元数）。
2. **单元处理时间必须小于 1 秒**：单元串行，一个单元超过 1 s，后面的单元就排队，延迟只增不减，直到 60 s 的输入缓冲满后报 `buffer_overflow`。所以“单元错过率”是能不能实时的直接判据。
3. 同卡多会话：4 个阶段共享一张卡，会话越多单元处理越慢；第一个撑不住的可能是 thinker（每单元一次 prefill + 最多 20 步 decode）或语音阶段（FP32 flow）。

## 10. 未知（R0 要回答）

- 4 个阶段能不能在一张 48 GB 的 A6000 上一起启动、要多久；
- 单会话的单元处理时间、单元延迟；
- 模型对 TTS 合成的、带停顿的提问会不会开口回答（多快、多长）；
- 2 个会话时单元延迟是否仍 < 1 s；
- max_sessions 8 能否单卡启动（推断：不能）；thinker 放第二张卡时跨卡交接（CUDA IPC）能否工作。
