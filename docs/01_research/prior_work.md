# Prior Work：Omni 模型 Serving 与 Agent Workload

## 范围与方法

本文整理与本项目重叠或相邻的公开工作，覆盖六类来源：
- SGLang-Omni 与 vLLM-Omni 两个 omni serving runtime；
- 会带来多阶段 serving 问题的 omni / speech 模型结构；
- LLM 与多模态 serving 中的 disaggregation（把 encoder、prefill、decode 等计算阶段拆到不同进程或 GPU，各自调度、各自扩缩）与调度；
- agent workload 的 serving 系统与刻画；
- 公开的 serving trace。

目的有两个：一是弄清上游和文献已经测过什么，避免重复；二是找出在已审阅来源中找不到的实验维度。本文不为我们主张任何首创，缺口一律写成“在所审阅的来源中没有找到”。

调研截止日期为 2026-11-11。SGLang-Omni 以本项目 pin 的 commit `1845546e`（2026-10-15，介于 v0.1.8 和 v0.1.9 之间）为准；pin 之后的变化以 upstream main `ddb7942f`（2026-11-11）为准，并在正文注明。核实方式如下：
- arXiv 论文：通过 abs 页面或 arXiv API 核对标题、作者和首版日期；被引用的句子和数字，在摘要或 `pdftotext` 提取的正文中逐条查到。
- 会议信息：取自 arXiv comment/journal-ref、USENIX/NeurIPS/PMLR 页面或 Crossref DOI。
- GitHub 仓库、issue、PR：用 `gh`/`git` 读原文。
- 博客：用 `curl` 抓取正文。

所有数字都保留原文给出的模型、硬件、负载和指标；原文没写硬件的，标为“未说明”。无法核实的内容放在文末“未核实线索”。

## 1. SGLang-Omni

### 1.1 架构（pin 版本）

请求依次经过 HTTP API、Client、Coordinator、Stage、Scheduler 和 ModelRunner[1]。Stage 只负责 I/O，计算交给与 stage 负载形态匹配的 scheduler：
- `OmniScheduler` 用于自回归（AR）stage。它复用 SGLang 的 batch 选择、KV cache（已处理 token 的 attention key/value 缓存）管理、prefill/decode 调度和前缀树缓存。prefill 指一次处理整段输入并写入 KV cache，decode 指此后逐 token 生成。文档写明它不支持 overlap scheduling。
- `SimpleScheduler` 用于预处理、encoder、聚合等非 AR stage，可选本地 batch。
- `SessionScheduler` 用于按 session 保存状态的 stage，full-duplex 模式使用它。
- `Code2WavScheduler` 是流式 vocoder 的循环。

控制消息走 ZMQ。数据按位置选通道：同进程直接传对象，同节点 GPU 之间走 CUDA IPC，CPU stage 走 SHM，跨节点走 Mooncake[1]。

Qwen3-Omni 的 speech 模式有 8 个 stage：`preprocessing`、`image_encoder`、`audio_encoder`、`mm_aggregate`、`thinker`、`decode`、`talker_ar`、`code2wav`[1]。各角色如下[2]：
- **Thinker**：生成文本的主 LLM（30B-A3B MoE）。
- **Talker**：另一个自回归模型，把 Thinker 的输出转成语音 codec token。
- **code predictor**：每一步补齐其余 codebook，并把结果反馈给下一步 Talker；因为这种逐步反馈，它与 Talker 合并成一个 `talker_ar` stage。
- **Code2Wav**（vocoder）：把 codec token 流式合成为波形。

部署形态写在配置里，与模型代码分开；上游称之为 Stage–Process–GPU（SPG）topology[2]。可调维度如下[1]：
- **process grouping**（`StageConfig.process`）：决定哪些 stage 共享一个 OS 进程。
- **process replica**（`processes.<name>.num_replicas`）：复制整个逻辑进程。请求在准入时绑定一个副本，生命周期内不变；默认按轮询分配。
- **`replica_devices`**：决定每个副本或 TP rank 放在哪张 GPU。
- **CUDA MPS 与 CUDA IPC 权重共享**：用于同卡多副本。MPS 即 Multi-Process Service，让多个进程的 kernel 在同一 GPU 上并发执行。
- **`gpu_memory_fraction`**：多个进程共用一张卡时，每个 GPU stage 都要声明，启动前统一校验。

我们在文档中没有找到按负载自动调整副本数的机制。找到的唯一显式 backpressure 控制是 coordinator 对未投递流数据设的上限 `stream_delivery.max_total_backlog_bytes`[1]。backpressure 指下游消费不过来时，积压被限制或向上游传导。该上限默认 1 GiB，超出后最久未读的流直接失败。

与本项目测量直接相关的默认行为[1]：
- thinker 最多同时运行 64 个请求；
- 多模态预处理（tokenize，以及图像/视频/音频特征提取）默认串行执行，`--preprocessing.factory.max_concurrency` 可改为线程池；
- 默认放置是 thinker 独占一卡、talker 独占一卡、`code2wav` 与 thinker 同卡；
- talker 的前 10 个 codec 帧逐帧发出，之后每 10 帧合并发送；
- `code2wav` 默认用固定 shape 的 CUDA graph，最多把 8 个就绪窗口合成一次 replay；
- Qwen3-Omni audio encoder 的 attention 在 Hopper 上走 FA3，其他 GPU 走 Triton。

MiniCPM-o 4.5 有两种模式[1]：
- **chat**：走 `/v1/chat/completions`。
- **原生 full-duplex**：模型边听边说，自己决定何时开口；走 `/v1/realtime`，启动参数为 `--variant session`。默认 `max_sessions` 为 2，每 1 s 音频决定一次说或听，单个 session 最多保留 8,192 token 历史。

**可观测性。** request-level profiler 以 JSONL 记录以下事件，并汇总 stage 与 hop 耗时[1]：admission、preprocess、encoder、prefill、首 token / 首 chunk、stage 间 hop、`code2wav` 窗口。另有可选的 torch profiler 和 full-duplex unit 指标。

**负载生成。** 基准脚本有以下几种模式[1, 13]：
- 默认是 closed-loop 并发扫描：并发数固定，一个请求结束后才发下一个。
- `--request-rate` 设为有限值时，改为带随机种子的 Poisson open-loop：按到达时间发送，不等前一个请求结束。
- `bench_sweep.py` 用 open-loop Poisson 扫描同卡多副本，注释写道 "past saturation a closed-loop client absorbs the backlog into TTFC and hides the capacity knee"。
- 同一个 prompt 重复 N 次的 rollout 压测。
- full-duplex 基准：按真实时间回放音频，按并发 session 数扫描。

我们没有找到带 session 或请求依赖结构的 trace 回放模式。

**Disaggregation。** SPG 博客把 encoder–prefill–decode（EPD）拆分和 stage offload 都列为 SPG topology 的特例[2]。PD 拆分机制已于 2026-09-13 合入（#1773），但 Qwen3-Omni thinker 的 PD 集成（PR #2756）到 2026-11-11 仍未合入[14]。因此在 pin 版本上，Qwen3-Omni 可调的放置维度只有 TP、进程分组、副本数和副本位置，不含 thinker 的 PD 拆分。

### 1.2 上游已经测过什么

下表只收录与本项目研究问题重叠的测量。除非另注，负载都是固定 prompt 集合上的 closed-loop 并发扫描。指标含义如下：
- TTFT：首个文本 token 的延迟。
- TTFA：首个音频包的延迟，vLLM-Omni 称为 TTFP。
- RTF：生成耗时除以音频时长，小于 1 表示快于实时。
- xRT：每墙钟秒生成的音频秒数。

| 来源 | 模型 / 负载 | 硬件 | 结果（保留原文条件） |
|---|---|---|---|
| PR #1235[5] | Qwen3-Omni speech，thinker TP2，c=8，每组 16 个请求，两对交错 | H200 | **talker 独占 GPU vs 与 thinker rank 0 共卡**：wall-clock 低 9–16%，TTFA p90 低 46–47%；talker 所在卡中位利用率 86%，`code2wav` 9–13%。<br>**`code2wav` 再单独占一卡**：wall-clock −0.7%，E2E −0.9%（单对）。<br>**c=1**：talker 独占使 E2E 变差 14–17%，TTFA 改善约 30%。<br>作者注明同机单对 10–30% 的差异在噪声带内 |
| PR #1175[6] | Qwen3-Omni，SeedTTS EN 固定集合，c∈{1,8,16,32,64}，共 11,370 个请求 | H200 | **同为 3 张 GPU**（1 个 thinker + 2 组 talker+`code2wav` 副本 vs 3 卡隔离对照）：QPS 在 c=16/32/64 为 +14.0%/+19.4%/+41.7%，c=1/8 为 −3.9%/−3.8%。<br>**相对 2 卡基线**：c=64 时文本 TTFT +32.5%，p95 inter-chunk gap +30.4% |
| Issue #1018[4] | Qwen3-Omni 性能路线图，thinker TP2 | H100 | **text→text+audio，c=1→16**：xRT 8.14→45.86，TTFA 中位数 0.302→1.247 s。<br>**c=8 时各 stage 从入队到结束的时长**：talker 1.54 s，`code2wav` 1.27 s，thinker 0.48 s。<br>**c=1 文本请求**：TTFT 88.1 ms，其中 prefill 窗口 76.3 ms；trace 中设备计算只有 3.2–3.8 ms，其余是约 2,400 次 CUDA API 调用的 host dispatch。<br>**c=8 talker**：每步 13.0–13.2 ms，GPU kernel 下限 6.29 ms。<br>**`code2wav` batching**：组件层面 batch 2/4/8 的效率为 1.55×/2.00×/2.64×，端到端平均只高 2.95%（三对中一对为负） |
| Issue #1236[7] | Qwen3-Omni `code2wav` 的 batching 与到达方式 | H100 | **请求错开到达**：固定 batching 的 xRT 为 24–34，逐请求 + CUDA graph 为 104–114。<br>**对齐到达、c=16、floor 8**：batching 高约 17%（136 vs 116） |
| Issue #907[8] | 每个 stage 一个单卡 server，c=16 | H100 80GB SXM5 | **Qwen3-ASR**：GPU 94.3% 的时间空闲；SM 时钟降到 ×0.455 时吞吐只降到 ×0.90，host CPU 降到约 1/4 时吞吐降到 ×0.31。<br>**各 stage**（含 Qwen3-Omni thinker decode 与 talker）：Tensor Core 活跃度 0.6–4.9% |
| ASR 并发剖析[9] | Qwen3-ASR-1.7B，SeedTTS，c=1–64 | H100 80GB | GPU 平均利用率 33–54%；吞吐在 c=32 饱和；c=64 时 queued→scheduled 为 412.8 ms |
| Issue #1914[10] | Qwen3-Omni text-only，MMSU 音频输入 | 1×H20 96GB | c=32 时 GPU busy 66.9%（trace）/66.5%（nvidia-smi）；c=8→64 吞吐 33.9→69.3 req/s。<br>约 6k token 的音频 prefill 在 scheduler 线程上执行，会阻塞正在 decode 的 batch。<br>预处理排队 56→114 ms，实际工作约 9 ms |
| Issue #2399[11] | MiniCPM-o 4.5 text→speech，三个生成 stage 同卡 | 1×H20 96GB | FP32 eager 的 `code2wav` 把吞吐限制在约 34 audio-s/s，改 FP16 + compile 后为 80.7。<br>三个同卡进程分时使用 GPU，把 `code2wav` 移到另一张卡后吞吐 +35% |
| Issue #2580[12] | MiniCPM-o 4.5 Video-AMME（视频 + 语音提问输入，文本 + 语音输出），c=1/8 | 1×H200 | GPU busy 15.6%（c=1 冷启动）、17.1–27.6%（c=8）。<br>单个串行预处理 slot 把 c=8 吞吐限制在约 0.33 qps：c=1 时端到端 62.4% 的时间花在该 slot 内，c=8 时 78.6–80.2% 在排队等它 |
| PR #2843[13] | MiniCPM-o 4.5 full-duplex，FDB v1.5，每级 32 个 session | 1×H200，`max_sessions`=8 | 并发 session N=1→8：unit lag p50 180→282 ms，p99 271→604 ms；无 unit miss，容量至少为 8 |
| 交互 runtime 博客[3] | FDB 行为、serving、听感；每级 128 个 session | 每个系统 1×H200 | 满足实时目标的并发 session 数：MiniCPM-o（native full-duplex）8，Qwen3-Omni（turn-based）2，ASR→LLM→TTS cascade 16（此时已到 admission 上限） |
| SPG 博客[2] | 与三个匿名开源引擎对比，c=16 | 1×H200 | 吞吐为最快对手的 1.32×（Qwen3-Omni）和 6.41×（MiniCPM-o 4.5）；Qwen3-Omni streaming TTFA p95 为 0.41 s，对手 0.50 s |

### 1.3 上游明确延后或尚未覆盖的部分

- **#1018 中被 gate 的项目**[4]：
  - "Multi turn tensor prefix cache: Requires a multi turn session benchmark"；
  - "TP/DP placement and prefill and decode disaggregation: Reopen after TP2 attribution under the same GPU budget"；
  - "Bounded active vocoder streams: Reopen after an overload signal exists, such as playback stalls or RTF p99 ≥ 1 at concurrency 8 and 16"。
- **#1760**[14]：指出 omni 的 prefill 占比随输入变化（"the split ratio is not a constant"），计划在相同 GPU 数下比较不同放置；音频输入一侧尚未测量。
- **#2208**[15]：`/v1/realtime` 每一轮都把完整对话历史重新放进 messages，历史没有上限。issue 写道 "latency creeps up as prefill grows with each turn"，但没有给出测量。
- **#2994**[16]：2026-10-20 创建（晚于 pin），计划在 cascade 的 Session Graph 中加入异步工具节点和多工具示例（D3/D4），相关 PR 仍为 draft。
- **交互 runtime 博客**[3]：把 "Scheduling around Interaction Timing"、"Coordinating Interaction with Deeper Work"、"Preserving Coherence over Longer Sessions" 列为后续工作。
- **#760**[17]：作者在 2×RTX A6000 上运行 Qwen3-Omni。单卡放不下，必须 thinker TP=2。在 16–32 个并发多轮流式 session（以 prefill 为主）下，作者观察到 prefill batch 碎片化，吞吐明显低于另一框架。issue 只给出了问题描述和计划中的验收测试（2×A6000、TP=2 上的并发扫描），没有给出测量数值；后续讨论转到 H100 上进行。

### 1.4 与 RTX A6000（sm_86）相关的限制

- **fused code predictor kernel。** pin 版本的 Qwen3-Omni talker 使用这组 kernel（`sglang_omni/utils/predictor_layers.py`，3 级 pipeline）[1]。PR #3284（2026-10-28 合入，晚于 pin）说明，在每个 block 共享内存上限为 101,376 B 的 GPU（"SM 8.6, 8.9 and 12.x"）上，这些 kernel 会启动失败；修复方式是失败后降到 2 级 pipeline[18]。在 A6000 上跑 pin 版本需要 backport 这一修复。pin 代码里能绕开 fused 路径的只有 deterministic inference 开关。文档记载该开关在 Qwen3-TTS 上会串行化预处理与 vocoder 解码、关闭 Talker 编译和初始 vocoder CUDA graph，并明显降低吞吐，因此不适合用于性能测量[1]。
- **MoE 后端选择。** `platforms/cuda.py` 在 Qwen3-Omni talker 未量化且 `moe_runner_backend=auto` 时，对非 H20 设备固定选 `flashinfer_cutlass`；pin 与 2026-11-11 的 main 都是这样[1]。我们没有找到公开来源说明这个后端是否支持 sm_86（见“未核实线索”）。`engine.*` 配置会原样传给 SGLang `ServerArgs`，可以显式指定后端[1]。
- **显存。** Qwen3-Omni README 给出的 BF16 理论最低显存（transformers + FlashAttention-2）为：Instruct 处理 15 s 视频需 78.85 GB，120 s 视频需 144.81 GB[25]。#760 的作者在 48 GB 的 A6000 上只能用 thinker TP=2[17]。上游的单卡 colocated 配置只有 H20、H200、H100、gfx950 几种；FP8 与 AutoRound INT4 路径在文档中只写了 H100/H20[1]。
- **CI 覆盖。** pin 时上游 CI 的 NVIDIA GPU 任务只在自托管 H100 上运行[1]，只在 sm_86/89 上出现的问题 CI 发现不了。上面的 fused kernel 问题由 #2489 引入，到 #3284 合入，前后存在了约 26 天[18]。

## 2. vLLM-Omni

vLLM-Omni 是 vLLM 项目下的 omni serving 子项目[19]：
- 仓库于 2025-09-11 创建；
- 首个预发布 v0.11.0rc1（2025-12-01）基于 vLLM 0.11.0；
- 最新稳定版 v0.32.0 发布于 2026-10-28。

设计要点：
- **拓扑与部署。** 拓扑由模型实现中的 Python `PipelineConfig` 定义，stage 类型为 `LLM_AR`、`LLM_GENERATION`、`DIFFUSION`。部署 YAML 只负责每个 stage 的 `devices`、`num_replicas`、`max_num_seqs`、`gpu_memory_utilization`、TP、connector 与 `async_chunk`[19]。
- **副本。** 副本组成 pool，按 round-robin 或最短队列分发；chunked 请求固定在同一副本上[21]。
- **传输。** 单节点默认使用 `SharedMemoryConnector`。文档写明 "Current connectors operate in D2H2D (device to host to device) mode"，GPU 之间直传仍在路线图中；跨节点可用 Mooncake、NIXL 等[19]。
- **流式。** `async_chunk` 默认开启：Thinker→Talker 每个 decode step 传一次；Talker→Code2Wav 按 `codec_chunk_frames`（默认 25）累积；下游 stage 预先启动；Code2Wav 使用 static batching[19]。
- **默认部署。** Qwen3-Omni 默认部署注明 "verified on 2x H100"：stage 0 在 GPU0，stage 1、2 在 GPU1，每个 stage `max_num_seqs: 64`[19]。
- **工具调用。** 支持 Qwen3-Omni Thinking 模型的 function tools，可经 chat completions 或 turn-based `/v1/realtime` 调用；`tool_speech.py` 在 Thinker 发出工具调用时停止 Talker[19]。
- **基准工具。** `vllm bench serve --omni` 提供 `audio_ttfp`、`audio_rtf`、goodput（满足 SLO 的请求吞吐，SLO 即延迟目标）和逐 stage 指标，但没有 agent 或多轮 trace 模式。其 Qwen3 realtime 负载的说明写明 "It does not measure interruptions, multi-turn context, or concurrency scaling"[19]。

已发表的数字：
- **arXiv 2602.02204**[20]。条件：2 个 80 GB 加速器（型号未说明），离线推理，LibriSpeech、Food101、UCF101 各取前 100 条，对比 HF Transformers。结果：
  - Qwen3-Omni 的 RTF 降低 90.7%，JCT 降低 91.4%；
  - 端到端延迟大部分花在 Talker；
  - Qwen2.5-Omni 的 Thinker→Talker 传输在 SHM 下为 5.49 ms，在 Mooncake 下为 8.28 ms。

  实验不含到达过程、多轮或工具调用。
- **技术报告 arXiv 2610.09307**[21]。条件：H100/H200；closed-loop C∈{1,4,8,16,32}；Random-MM 的 open-loop request rate ∈{0.1,0.5,1.0}；duplex 只有一个 4 轮 session。结果：
  - 2×H100、Qwen3-Omni Fixed-Len-2500/900、C=32，async-chunk 开/关：TTFT 507.8 vs 3420.1 ms，E2EL 72.63 vs 136.62 s，RTF 0.630 vs 0.927。
  - 3×H100、talker 与 code2wav 各 2 个副本：C=32 的 RTF 反而高于 2 卡配置，作者归因于 "thinker admission and cross-stage handoff dominate"。
  - 1×H200、C=32：text-only 路径的 TTFT 为 860.1 ms，只跑 thinker 的 vLLM 为 562.4 ms。

  报告没有跨框架对比。
- **vLLM 博客**[22]。条件：Seed-TTS en，Qwen3-Omni-30B-A3B-Instruct，3 张 GPU（型号未说明），C=64。结果：
  - 逐项叠加优化后，吞吐从 2.2 req/s 升到 11.3 req/s，平均 audio TTFP 从 5884 ms 降到 631 ms；
  - 再加 2 组 talker/code2wav 副本后为 11.7 req/s；
  - 原文："Talker and Code2Wav saturate first: at concurrency 64 a single replica of either becomes the tail bottleneck while Thinker still has headroom."
- **Ampere 与工作站卡。** 仓库设计文档有一组 Qwen3-Omni 在 A100 上的单请求离线对比：相对 HF Transformers，E2E 23.78 s vs 336.10 s，TTFP 0.934 s vs 336.10 s[19]。在已审阅来源中，没有多 stage omni 模型在 A6000 或其他 PCIe 工作站卡上带负载的结果；仓库里有 `minicpmo_4_5_8x4090.yaml` 部署文件，但没有配套数字[19]。

## 3. 与 Serving 相关的 Omni 模型结构

| 模型 | stage 链 | 规模（原文） | 流式粒度 | 延迟数字及条件 |
|---|---|---|---|---|
| Qwen2.5-Omni[23] | 音频/视觉 encoder → Thinker → 双轨 AR Talker → sliding-window DiT（flow matching）+ BigVGAN | 视觉 encoder 约 675M，其余未给出 | 音频 encoder 按 2 s block 做 attention；DiT 感受野为 4 个 block（回看 2 个、前看 1 个） | 报告只列出延迟的构成，没有给数值 |
| Qwen3-Omni-30B-A3B[24] | AuT 音频 encoder + SigLIP2 视觉 encoder → Thinker（MoE）→ Talker（MoE，预测第 0 个 codebook）+ MTP（预测其余 codebook）→ 因果 ConvNet Code2Wav | 依次为 650M / 540M / 30B-A3B / 3B-A0.3B / 80M / 200M | 12.5 Hz，1 个 token 对应 80 ms 音频；Talker 每生成一个 token 就能输出波形 | 冷启动理论首包延迟 234 ms（音频）/ 547 ms（视频）；并发 4 时 728/1517 ms，并发 6 时 1172/2284 ms；RTF 依次为 0.47/0.56/0.66。条件为 vLLM、"under typical computational resources"，GPU 未说明 |
| MiniCPM-o 4.5[26] | SigLIP + resampler；Whisper-medium + projector → Qwen3-8B → speech token decoder（约 0.3B）→ 流式 flow-matching decoder | 总计 9.34B，其中 LLM 8,189.2M；flow decoder 规模未给出 | Omni-Flow 1.0 s 时间片，每秒决定一次说或听；speech token 25 个/s | 单张 RTX 4090、vLLM、BF16：首 token 0.59 s（64 帧视觉输入）；文本吞吐 154.3 tok/s，显存 19 GB（吞吐和显存在 text-only 任务上测得）。同条件下 Qwen3-Omni-30B-A3B BF16 OOM |
| Moshi[27] | Mimi encoder → Helium 7B Temporal Transformer + Depth Transformer → Mimi decoder | 7B；Depth Transformer 6 层 | 80 ms 一帧；用户与模型两路音频并行建模 | 理论 160 ms，实际 200 ms。论文未写硬件，README 写明是 L4 GPU |
| PersonaPlex[28] | 沿用 Moshi 架构与权重 | — | 80 ms 一帧；2048 token 上下文约合 163.84 s | — |
| NemotronLabs VoiceChat[29] | 600M 流式 encoder → Nemotron-Nano-9B-v2（含工具调用输出流）→ 778M TTS decoder → 199M codec | 见左栏 | 输入输出均按 80 ms 的整数倍切块 | 单张 H100 PCIe 80GB、4 路并发：每个 160 ms chunk 的 p95 推理延迟为 118 ms，相当于 1.36 倍实时 |

几条与 serving 直接相关的结构事实：
- **Qwen3-Omni 的 MoE 与 Talker 条件。** Thinker 和 Talker 都用 MoE，报告的理由是 "support high concurrency and fast inference"。Talker 不再读取 Thinker 的高层文本表示，只以多模态特征和当前轮的流式文本为条件。报告说这种设计 "allows external modules (e.g., RAG, function calling, safety filters) to intervene on the Thinker's textual output"[24]。对 agent 场景，这意味着工具调用可以插在 Thinker 与 Talker 之间。
- **234 ms 的口径。** 报告中的并发首包延迟表是把各组件延迟相加得到的理论值，不是在线 serving 测量[24]。
- **Qwen3-Omni README**[25]：推荐使用 vLLM-Omni；"vLLM serve for Qwen3-Omni currently only supports the thinker model"；`disable_talker()` 约节省 10 GB 显存。
- **SGLang-Omni 的覆盖。** pin 版本为 Qwen3-Omni、MiniCPM-o 4.5、PersonaPlex、Nemotron VoiceChat 提供了 cookbook[1]。后两者是原生 full-duplex 模型，其中 VoiceChat 原生支持工具调用[29]。

## 4. Disaggregation 与多模态 Serving 系统

### 4.1 Prefill/decode 拆分（文本 LLM）

| 系统 | 机制 | 结果（原文条件） |
|---|---|---|
| DistServe[30]（OSDI'24） | 把 prefill 与 decode 放到不同 GPU，分别选并行度，以 goodput 为目标 | OPT-13B/66B/175B，A100 集群，对比 vLLM 与 DeepSpeed-MII："7.4x more requests or 12.6x tighter SLO"（>90% 的请求满足延迟约束） |
| Splitwise[31]（ISCA'24） | 按 phase 把请求拆到不同机器；基于 Azure trace，用模拟器设计集群 | "1.4x higher throughput at 20% lower cost"，或在相同成本与功耗下 "2.35x more throughput" |
| Mooncake[32]（FAST'25） | PD 拆分之上加入 CPU DRAM/SSD 上的 KVCache 池，以 KVCache 为中心调度 | 真实 trace 下 "effective request capacity by 59%~498%"（FAST 版本的数字；arXiv 最新版数字不同） |
| TetriInfer[33] | PD 拆分 | OPT-13B，4×V100（互连为模拟），对比 vLLM：资源少用 38%，平均 TTFT 降 97%，平均 JCT 降 47% |

这四项只处理文本，实验平台都是带 NVLink 或 InfiniBand 的数据中心 GPU；TetriInfer 的互连是模拟的。

### 4.2 Encoder 与 EPD 拆分（多模态输入、文本输出）

- **EPD Disaggregation**[34]（ICML'25）：把 encode、prefill、decode 分到专用资源上。在 8×A100 上对 MiniCPM-V 2.6、InternVL2 测试，相对不拆分的系统，SLO attainment "up to 90–100% improvement"，TTFT "up to 71% reduction"。音频只出现在附录的 Ultravox 实验中。
- **ModServe**[35]（SoCC'25）：按模态和阶段拆分资源，在 128 GPU 集群上用生产 trace 测得 "3.3-5.5x higher throughput"。其刻画在 DGX-A100（96 个 CPU 核）上完成，结论有两条：
  - CPU 上的图像预处理 "contributes minimally to the overall TTFT"；
  - 在 TP-8 的 Llama3.2-11B/90B 上，图像 encoding 占 TTFT 的 79% / 65%。
- **其他系统**：HydraInfer[36] 做混合 EPD 布局；ElasticMM[37]（NeurIPS'25 Oral）按模态分组做弹性并行，其动机数据是图像预处理与 encoding "often taking more than five times longer than the prefill stage"；RServe[38] 在单个请求内让 encoding 与 prefill 重叠。
- **CPU 侧开销的结论互相矛盾**：
  - ModServe 认为图像预处理影响很小；
  - ServeGen 认为 "a significant portion of TTFT stems from preprocessing (i.e., downloading, normalization, and encoding)"[81]；
  - FlashCodec/UnifiedServe 认为视频解码 "often dominates" TTFT，并在 4×A100 上把解码移到 NVDEC[41]。
- **EAServe**[39]（PACT'26）：主实验平台为 8×RTX 6000 Ada 48GB（PCIe）。输入包括图像、视频和音频（Ultravox），输出为文本；goodput 相对 NVIDIA Dynamo 和 vLLM 分别 "up to 4.3x and 1.7x"。这是已审阅来源中与 8×A6000 PCIe 最接近的多 GPU 平台，但不涉及语音输出。
- **反例 RPS-Serve**[40]：在 4×H100 的等预算比较中，ModServe 式 2+2 拆分 "approximately 60–70% of aggregate GPU capacity remaining idle"。作者因此主张在同一张卡上按模态优先级调度。
- **CPU 导致的 GPU 空闲**：
  - [42]（IISWC'26）指出多 GPU 推理中 "CPUs fail to keep them busy"：长序列、大 batch 时 tokenization 可占 TTFT 约 80%。该文还讨论了 agent 场景反复 tokenize 累积上下文的问题，实验平台包括 H100/H200 和无 NVLink 的 8×RTX Pro 6000。
  - Meta 对多模态生成（含 Seamless 语音输出）的刻画[43]指出，自回归生成 "typically dominated by GPU idle time"。该结论在 A100、batch 1 下得出，不是 serving 场景。

### 4.3 面向 omni 与语音输出的 serving 系统

| 系统 | 机制 | 负载 | 硬件 | 结果（原文条件） |
|---|---|---|---|---|
| Cornfigurator[44] / Cornserve[45] | 在 GPU 预算内搜索部署计划，范围从全同卡到全拆分；按组件独立扩缩 | Qwen3-Omni、Qwen2.5-Omni；按 ServeGen 到达，音频输出比例设为 1/3 或 2/3（Cornserve 为 20%） | 2×8 A100-80GB，NVSwitch | goodput 比 vLLM-Omni 方案和全拆分高 1.24–2.73×。原文："the thinker LLM achieves nearly 30× higher request throughput than the talker LLM on A100"。SGLang-Omni 因 "early development stage" 未纳入对比 |
| M*[46]（NeurIPS'26） | dataflow 图抽象；Talker 与 Code2Wav 放在同一 worker | Qwen3-Omni，Seed-TTS，固定 batch size B（不是到达过程） | 2×H200 | B=16 时吞吐为 vLLM-Omni 的 2.7×、SGLang-Omni（commit 4a3960）的 4.0× |
| HorizonServe[47] | 在单 GPU 上协调准入与 SM 分配，基于 vLLM-Omni v0.16.0 | Qwen2.5-Omni、MiMo-Audio、BAGEL，Poisson 到达 | 单张 RTX 6000 Ada 48GB 或 RTX PRO 6000 | SLO attainment 在到达率扫描中最高提升 4.9×，下游负载重时最高 7.0× |
| OmniCast[48] | 按播放进度和首音频优先级调度；按预计下次使用时间驱逐 KV；处理 barge-in（用户插话打断） | Qwen3-Omni、Ming-Flash-Omni 2.0。多轮交互 trace 含 session ID、时间戳、token 长度、turn index；到达用 Poisson 与 BurstGPT；barge-in 按 Bernoulli 注入 | 8×H200（thinker DP=4，talker DP=4） | 相对基于 vLLM-Omni 的基线：P90 audio TTFP 平均降低 1.55×（最高 2.21×），完成请求吞吐平均提高 1.15× |
| Omni-Flow[51] | 用 Python DSL 编排 workflow；session 状态保存在服务端，KV 跨实例共享 | Qwen3-Omni-30B-A3B；50 轮多模态 session（每轮新增一张图、一段音频和一个问题；输出为 10 个 token、16 帧 codec） | 2×H20-3e（GPU0：Thinker 与 encoder；GPU1：Talker 与 decoder） | 单轮负载的总 JCT 比 vLLM-Omni / SGLang-Omni 分别高 1.2% / 4.9%。50 轮 session 中，服务端保存 session 状态、不再重复提交累积输入，平均 JCT 低 7.5% / 19.3% |
| VoxServe[49] | LM 与 codec 分占 SM 分区并发执行；紧急 chunk 使用整卡 | Qwen3-TTS 等，LibriTTS，open-loop Poisson；SLO 为 TTFA ≤200 ms 且不卡顿 | 1×H100 SXM | 48 req/s，为最强基线 SGLang-Omni 0.1.8 的 1.5×。原文："Disaggregation avoids interference but strands capacity" |
| Metronome[50] | 每个 session 的 KV 窗口加在线准入 | full-duplex，Qwen3-Omni-30B 与 MiniCPM-o 4.5 | 1×RTX PRO 6000 Blackwell | N=96/128、每次 300 s 的运行中，无界 KV 在 14/20 次运行里一步跳到约 1.6 s 的墙，窗口化 KV 为 0/20。原文："The wall is reached by duration, not concurrency — a 90s test cannot see it" |

## 5. Agent Workload 的 Serving 与刻画

### 5.1 以程序或 workflow 为调度单位的系统（文本 LLM）

- **Parrot**[52]（OSDI'24）：用 Semantic Variable 把请求之间的数据依赖告诉服务端，据此联合调度并共享前缀。报告 "up to 11.7× speedup or 12× higher throughput"，其中 12× 来自 4 张 A6000（48GB）、每卡一个 LLaMA 7B 引擎的多应用场景。请求按 Poisson 到达，不模拟工具耗时。
- **SGLang**[53]（NeurIPS'24）：RadixAttention 用 radix tree 复用相同前缀的 KV（即 prefix cache）。ReAct 和 generative agents 的 trace 从原论文抽出后回放，不含工具耗时；吞吐在大 batch 下测量。
- **Agentix**[54]（NSDI'26，arXiv 上名为 Autellix）：拦截 LLM 调用，按程序已获得的服务量做抢占调度。同等延迟下，吞吐比 vLLM 等系统高 "4-15×"。论文明确把工具与人工等待时间排除在外（"Since component (3) is unrelated to LLM serving"），程序按 Poisson 到达。
- **Teola**[55]（ASPLOS'25）：在 LLM 与非 LLM 引擎（embedding、向量库、搜索）之间做 primitive 级的数据流调度。
- **Murakkab**[56]（OSDI'26）：按 SLO 为 workflow 选择模型、工具和硬件。其视频问答 workflow 含 Whisper 语音转写 stage。论文写道 "Since no publicly available traces exist for production agentic workflow serving"，因此改用 Azure LLM trace 近似到达。
- **AMPD**[57]（ICML'26）：把 PD 拆分用于多轮 agent workflow（ToolBench、GAIA 等），在 4×8 H20 上报告了相对同卡与拆分两类基线的 SLO attainment 提升。

### 5.2 工具调用期间的 KV 保留与工具并行

- **InferCept**[58]（ICML'24）：现有系统在工具返回后把后续部分当作新请求处理，重算占 "37-40% of total model forwarding time"。InferCept 在拦截期间选择保留、丢弃后重算或换出 KV。拦截类型包括 Bark TTS 和图像生成，其中 TTS 的执行时间为实测，人的响应时间为估计值。
- **Continuum**[59]：按重新加载 KV 的代价，为工具调用期间的 KV 设定 TTL（保留时限）。工具耗时呈长尾分布："Slowest 10% of fetch_url account for 52.5% of the total delay"。
- **KVFlow**[60]（NeurIPS'25）按 agent 步骤图决定 radix 节点的驱逐优先级；**TokenCake**[61]（EuroSys'27）在函数调用期间卸载空闲 KV、之后预测性上传，其工具服务器按采集到的耗时分布重放工具调用。
- **Ask the Tool**[62]：在空闲 sandbox 上重放 leaderboard 记录的命令，"The same command finished an order of magnitude faster"。录制下来的工具耗时不能直接搬到另一个环境使用。
- **Conveyor**[63]：让工具在 LLM 解码过程中部分执行，并写道 "there are unfortunately no publicly available realistic datasets of tool-assisted LLM serving scenarios"。
- **PASTE**[64]：并行执行工具与生成，按生产 Azure Functions 调用 trace 的时间戳发出 agent 请求。

### 5.3 回放方法

- **AgentServeSim**[65] 的论点："Replaying its turns at recorded timestamps fixes the release schedule and cache state that a different policy is supposed to change; the counterfactual serving trajectory must instead be generated causally." 这一点只在模拟器中论证过。
- **TraceLab 的 runner** 在 session 内采用 closed-loop：发出第 i 轮，等完整响应，睡眠 `tool_wait_after_ms`，再发第 i+1 轮[73]。
- **CacheWise** 观察到 "Tool-initiated requests dominate, indicating a predominantly closed-loop request generation process"[66]。
- **常见做法**是程序级按 Poisson 或 Azure trace 到达、程序内部 closed-loop（Agentix、TokenCake、PASTE 等）[54, 61, 64]。我们没有找到在真实硬件上量化“按时间戳 open-loop 回放”与“依赖感知回放”之间差异的研究。

### 5.4 Agent workload 刻画

- **The Cost of Dynamic Reasoning**[71]（HPCA'26）：在 Llama-3.1-8B/70B、vLLM、A100 上，LLM 推理与工具执行分别占总延迟的 69.4% 与 30.2%。
- **GitHub Copilot 生产 trace 分析**[72]：覆盖 3.2M 用户、13M 个 session、761M 次 LLM 调用。
  - KV cache 命中率在 turn 内平均 90%，跨 turn 边界降到 55%；
  - 87% 的 LLM 调用由 agent 发起；
  - 28% 的 prompt token 来自工具结果。
- **TraceLab**[73]：前缀命中率 95.7%；大部分 miss 发生在人的阅读和输入间隔超过缓存驱逐时间时。命中率由 API 提供方报告，不是在自建引擎上测得。
- **AgentSysBench**[74]：分析三个生产应用的 24 小时 trace（35,037 个编码 session、141,376 个搜索问答 session、2,386 个 Openclaw 类 session），结论是 "production sessions hold state idle for minutes to hours"。这些 trace 没有公开。
- **IISWC'26 的 benchmark 实测**[75]：跨 turn 的 cache 命中率为 84.6–99.5%，decode 占 LLM 时间的 91.0–98.6%。

### 5.5 语音/omni 与 agent 的交叉

- **模型侧**：
  - NemotronLabs VoiceChat 原生支持工具调用[29]；
  - arXiv 2609.19334 为 full-duplex 语音模型提出前后端分离的工具调用架构[69]；
  - Qwen3-Omni 报告称其 Talker 设计允许 function calling 等外部模块介入 Thinker 的文本输出[24]。

  这些都是模型论文，没有负载下的 serving 测量。
- **端侧**：arXiv 2610.07641 在 Android 上用投机执行隐藏级联语音 agent 的工具延迟[70]。
- **Serving 侧**：
  - Nova[67] 在单张 RTX A6000 上服务 agentic VLM（GUI agent），每一步视为独立请求，按 Poisson 到达模拟，理由是 "Due to the lack of public agentic VLM serving traces"；
  - OmniGAIA[68] 是带工具调用的 omni agent 准确率基准，没有 serving 时间数据；
  - Omni-Flow 与 OmniCast 处理多轮 omni session（见 4.3），都没有工具调用[51, 48]；
  - SGLang-Omni 的异步工具节点仍是 draft[16]；vLLM-Omni 只提供工具调用功能，没有相关测量[19]。

## 6. 公开 Trace 与 Workload 刻画

| Trace | 范围 | 字段（原文） | session / 多轮 | 模态 | agent / 工具 / 依赖 |
|---|---|---|---|---|---|
| Azure LLM 2023 / 2024[77, 31, 78] | 2023-11 两个服务（code、conversation）；2024-05 一周 | `TIMESTAMP, ContextTokens, GeneratedTokens` | 无 | 无 | 无 |
| Azure LMM 2025[77, 35] | 2024-10-15 至 10-22 | `TIMESTAMP, NumImages, ContextTokens, GeneratedTokens` | 无 | 只有每个请求的图像数 | 无 |
| GitHub Copilot Coding Agent 2026[77, 72] | 2026-06-01 至 06-07；301K session，1.2M turn，9.3M 次 LLM 调用，8.7M 次工具调用 | session → turn → `llm_calls` / `tool_batches`，含时间戳、耗时、token（含 cached）；工具批次的 `request_id` 对应 LLM 调用的 `message_id`；仅元数据 | 有 | 纯文本 | 有 |
| BurstGPT[79] | 121 天 + 110 天（v2.0） | `Timestamp, Model, Request tokens, Response tokens, Total tokens, Log Type`；v2.0 新增 `Session ID`（仅 conversation 模式）和 `Elapsed time` | v2.0 有 | 无 | 无 |
| Mooncake[32] | conversation 与 toolagent 取自 1 小时线上数据；synthetic 由公开数据集构造，到达过程为 Poisson | `timestamp, input_length, output_length, hash_ids` | 无 ID，只能从 `hash_ids` 看出前缀复用 | 无 | “tool and agent” 只是 trace 类别，没有结构 |
| Qwen-Bailian[80] | 两小时抽样 | `chat_id, parent_chat_id, timestamp, input_length, output_length, type, turn, hash_ids` | 有（`parent_chat_id`、`turn`） | `type` 取值含 text / search / image / file | 无工具边界 |
| ServeGen[81] | 按 client 参数化的分布 | 到达率与模式参数、token 分布；`multimodal/` 下只有 `mm-image`；另有哈希化的对话样本 | 部分（哈希化对话） | 论文刻画了 mm-audio（Qwen2-Audio-7B）和 mm-omni（Qwen2.5-Omni-7B），但仓库只发布了 mm-image | 无；论文把 plugin 调用列为待研究问题 |
| TraceLab[73] | 43 名开发者使用 Claude Code / Codex 的记录，357K 个 LLM 轮次 | 逐轮的 token 数（前缀与新增部分）、时间事件、工具发出与返回时间 | 有 | 纯文本 | 有 |
| Chutes[82] | 2025-04-11 至 2026-04-12 | 按调用记录；不含 prompt 和 session 标识，论文用启发式重建多轮 session | 无（需重建） | 论文只分析了文本 API | 无 |

**刻画类结论：**
- **ServeGen**[81]：mm-omni 负载中 "audio load rises during the day, while image load becomes prominent past midnight"。
- **Mooncake**[32]：要求按时间戳回放真实 trace；其 synthetic trace 使用 Poisson 到达。
- **SMetric**[76]：在 Bailian 的 agent 负载上测得前缀复用超过 80%，对话负载为 54–62%。作者计划在论文发表时公开 trace（"We will open-source SMETRIC and the traces upon publication"）。

**问题一：有没有公开 trace 包含多模态或语音请求？**
- **图像**：Azure LMM 2025 有每个请求的 `NumImages`[77]；Bailian trace 的 `type` 字段有 image[80]。
- **语音**：ServeGen 刻画过生产中的音频与 omni 负载，但没有发布对应数据[81]。
- **结论**：在已审阅来源中，我们没有找到带逐请求语音/音频信息的公开 trace，也没有任何 trace 记录语音输出。

**问题二：有没有公开 trace 包含 agent 步骤与依赖结构？**
- **完整结构**：GitHub Copilot 2026 和 TraceLab 记录了 session、turn、LLM 调用与工具调用之间的关联和时间[77, 73]。
- **部分结构**：Bailian 有 `parent_chat_id`，BurstGPT v2.0 有 Session ID，Mooncake 的 toolagent trace 只有类别标签[80, 79, 32]。
- **结论**：以上都是纯文本。在已审阅来源中，没有一个公开 trace 同时包含多模态/语音请求和 agent 步骤依赖。

## 7. 现有测量的局限

- **硬件。** 现有测量几乎全部在数据中心 GPU 上：SGLang-Omni 上游用 H100、H200、H20，vLLM-Omni 技术报告用 H100/H200，其余 omni serving 论文以 H100、H200、A100（NVLink/NVSwitch）为主。工作站卡只出现在单卡 omni（HorizonServe）、文本输出的 EPD（EAServe）和纯文本 CPU 研究中[47, 39, 42]。A6000 上的 omni 只有 #760 的定性描述[17]。PCIe-only 多卡上 Thinker→Talker→Code2Wav 跨卡传输与 TP all-reduce 的开销，我们没有找到测量。
- **负载形态。** 上游以固定集合上的 closed-loop 并发扫描为主（SeedTTS EN、MMSU、Video-AMME CI 子集、FDB），open-loop 用合成 Poisson[1]。#1236 已在 H100 上表明，请求是否对齐到达会决定 `code2wav` batching 的收益是正是负[7]；其余结论在其他到达方式下的复查，我们没有找到。
- **请求结构。** omni 测量中的请求彼此独立，或者只是对话型多轮（OmniCast、Omni-Flow、vLLM-Omni 的 4 轮 duplex）[48, 51, 21]。以下特征都没有出现：工具调用造成的空档；上一步输出决定下一步输入的依赖；同一 session 内输出模态的变化（中间步骤只出文本，最终步骤才出语音）。
- **样本量与噪声。** 上游 PR 常见 2–5 次配对重复，作者自己报告同机噪声带为 10–30%[5]。Omni-Flow 的多轮实验是单 session、每轮 10 个文本 token[51]。
- **归因。** GPU 空闲的归因已有两项：host dispatch（#907，H100）和预处理 slot（#2580，H200）[8, 12]。两者都在单卡、固定负载下完成。profiler 能给出 stage 与 hop 时间[1]，但我们没有找到突发负载下各级队列深度、积压字节随时间变化的测量，也就是 backpressure 如何逐级传导的定量描述。
- **Disaggregation 文献。** PD/EPD 系统都只输出文本。omni 系统中，Cornfigurator 用了 ServeGen 到达，但音频输出比例是人为设定的；M* 用固定 batch[44, 46]。

## 8. 已有来源未覆盖的 Workload

在所审阅的来源中，我们没有找到以下 workload 在 omni serving runtime 上的测量：

1. **语音 agent 的完整步骤。** 语音输入，Thinker 产生工具调用，工具执行（外部延迟），结果回填，最后语音输出。相关的模型和架构工作有 VoiceChat、2609.19334，端侧有 2610.07641，但都不是多 GPU serving 测量[29, 69, 70]。
2. **同一 session 内模态比例的变化。** 中间步骤只出文本（例如工具调用），只有面向用户的步骤需要 Talker 和 Code2Wav。上游的副本实验默认每个请求都输出语音[6, 5]；Cornfigurator 的音频输出比例是人为设定，不来自 trace[44]。
3. **并发 agent session 中的媒体累积。** 多轮对话里反复携带图像、截图、音频，带来预处理和 encoder 的重复开销。Omni-Flow 只测了单个 50 轮 session，输出很短[51]；上游把 multi-turn prefix cache 列为需要多轮 session 基准才能推进的项目[4]。
4. **两种回放方式的差异。** 依赖感知回放与按时间戳 open-loop 回放在多阶段 omni pipeline 上会得出多大不同的结论。
5. **工具等待期间的 KV 保留。** 工具等待期间 thinker 的 KV 能否保留、恢复时前缀能否命中，在 omni 模型上没有测量；文本 LLM 上有 InferCept、Continuum、KVFlow、TokenCake[58, 59, 60, 61]。
6. **PCIe 工作站卡上的等预算放置。** 在 PCIe-only 工作站 GPU 上，以相同 GPU 预算比较放置、副本与拆分。Qwen3-Omni 在 48 GB 卡上必须用 thinker TP=2[17, 25]。
7. **突发负载下的长时 session。** 分钟级以上、突发负载下的 omni session 中，backpressure、播放 underrun 与 KV 耗尽的相互作用。Metronome 指出短时测试看不到 KV 墙[50]。

## 9. 我们可以开展、且不与已有工作重复的方向

“对应”一列使用同目录 `research_questions.md` 中的问题编号 Q1–Q10。每一行都写明已有工作做到了哪里，以及我们改变的是哪个维度。

| 方向 | 对应 | 已有工作（不重复的部分） | 在 8×A6000 上的具体测量 |
|---|---|---|---|
| D1. 回放方式对结论的影响 | Q2、Q9、Q5 | 上游：固定集合 closed-loop 加合成 Poisson[1]。OmniCast：多轮 trace 加 BurstGPT 到达，无工具，H200[48]。AgentServeSim：只在模拟器中论证[65] | 同一组带工具延迟的 agent session 用四种方式回放：(a) 固定并发；(b) 平均速率相同的 Poisson；(c) 原始时间戳 open-loop；(d) 依赖感知 closed-loop。比较 TTFT/TTFA/E2E 的 p50/p99、profiler 中各 stage 的排队时间，以及按 SLO 计算的 session 容量 |
| D2. 文本步骤与语音步骤的比例 | Q3、Q7 | #1175、#1235 默认全部请求输出语音（H200）[6, 5]。Cornfigurator 的音频输出比例为人为设定（A100）[44] | Qwen3-Omni（TP2 thinker + talker）上扫描需要语音输出的步骤比例 {0, 0.25, 0.5, 1}，记录 thinker/talker/`code2wav` 的利用率与排队，找出 talker 副本开始带来收益的比例 |
| D3. 并发 session 中的媒体累积 | Q1、Q4、Q6 | #2580：单轮视频，H200[12]。#1914：单轮音频，H20[10]。Omni-Flow：单个 50 轮 session，H20[51]。#2208 未测量[15] | 4–16 个并发 session，每轮追加图像或音频。记录每轮 `preprocess_*`、`encoder_*`、prefill 时间与前缀命中，并用 taskset 把 CPU 限制在 8/16/32 核，看 GPU 空闲比例的变化；对比 `max_concurrency` 为 1 和 4 |
| D4. 工具等待期间的 KV 保留 | Q9、Q6 | 文本 LLM：InferCept、Continuum、KVFlow、TokenCake[58, 59, 60, 61]。录制的工具耗时随环境变化[62] | 依赖感知回放中，按 Copilot/TraceLab 中的工具耗时分布注入延迟[77, 73]，并对耗时分布做敏感性扫描。记录恢复时的前缀命中率、重算 token 数和 TTFT 与 48 GB 卡上 KV 池大小（`mem_fraction_static`）的关系 |
| D5. PCIe 上的等预算放置 | Q7、Q8 | #1235、#1175（H200）[5, 6]。vLLM-Omni 技术报告的 2 卡 vs 3 卡比较不是等预算（H100）[21]。Cornfigurator（A100 NVSwitch）[44]。RPS-Serve（图像输入，H100）[40] | 以 6 卡为等预算比较两种 Qwen3-Omni 配置：(i) 2 个完整 pipeline 副本（各为 TP2 thinker + talker，共 6 卡），经 router 分发[1]；(ii) 1 个 TP2 thinker + 4 组 talker/`code2wav` 副本。MiniCPM-o 另做一组 2 卡预算的比较：两个完整副本、thinker/talker 拆分、只复制 talker（即 Q7 的计划）。记录 thinker→talker hop、TP all-reduce 时间、吞吐和 TTFA，并记录实际拓扑（NVLink 或 PCIe）。pin 版本不能做 thinker PD 拆分[14] |
| D6. 突发 agent 负载下的 backpressure | Q5、Q9 | #2843：MiniCPM-o，N≤8，H200[13]。Metronome：单卡 duplex 的 KV 墙[50]。上游等待 "overload signal" 再推进 vocoder 限流[4] | 至少 30 分钟的长时回放。记录 stream backlog 字节（对比 1 GiB 上限）、talker/`code2wav` inbox 深度、inter-chunk gap 和 underrun 率，确定积压从哪一级开始向上游传导 |
| D7. 上游 host 瓶颈结论在 A6000 上复核 | Q3、Q4、Q6 | #907（H100）、#1914（H20）、#2580（H200）[8, 10, 12] | 在 sm_86 上用相同负载重复两组测量：GPU busy，以及 SM 时钟和 CPU 配额两侧的敏感度。目的是确认上游结论在较慢 GPU 和不同 CPU/GPU 配比下是否成立。这是复核，不是新方向 |

D1、D4、D6 依赖带工具耗时和步骤依赖的 session 数据。现有公开 trace 中只有纯文本的 Copilot 2026 和 TraceLab 提供这种结构；语音和多模态部分需要我们自己合成，并在报告中标明是合成数据。

## 参考文献

[1] sgl-project. SGLang-Omni 仓库，commit `1845546e`（2026-10-15）：`README.md`；`docs/developer_reference/{main,pipeline,communication,config,profiler}.md`；`docs/basic_usage/{process_topology,qwen3_omni,mps_dp}.md`；`docs/cookbook/{qwen3_omni,minicpm_o,personaplex,nemotron_voicechat}.md`；`benchmarks/README.md`、`benchmarks/benchmarker/runner.py`、`benchmarks/eval/{bench_sweep,benchmark_omni_rollout_stress}.py`；`sglang_omni/platforms/cuda.py`、`sglang_omni/utils/predictor_layers.py`、`sglang_omni/models/qwen3_omni/`；`examples/configs/`；`.github/workflows/`。GitHub, 2026. https://github.com/sgl-project/sglang-omni/tree/1845546e40c435bb98e6bfc417331f96c553caeb 已核实: 是（2026-11-11）

[2] SGLang-Omni Team. "SGLang Omni: Serving Multi-Stage Omni Models with Heterogeneous Stage Scheduling."（pin 版 README 以 "Fish Audio Drama 3 Meets SGLang-Omni: Day-0 Support and the Stage–Process–GPU Topology" 为题引用同一 URL）SGLang Blog, 2026-10. https://www.sglang.io/blog/spg-topology 已核实: 是（2026-11-11）

[3] SGLang-Omni Team. "Unified Runtime for Full-Duplex Interactive Intelligence with SGLang Omni." SGLang Blog, 2026-10. https://www.sglang.io/blog/omni-interactive-runtime 已核实: 是（2026-11-11）

[4] edwingao28 et al. "[RFC][Qwen3-Omni Perf] Serving Optimization Roadmap." sgl-project/sglang-omni issue #1018, 2026-07-10（状态头更新于 2026-08-15）. https://github.com/sgl-project/sglang-omni/issues/1018 已核实: 是（2026-11-11）

[5] edwingao28. "[Qwen3-Omni Perf] Default code2wav onto the thinker GPU so the talker gets its own device." sgl-project/sglang-omni PR #1235, 2026-08-13 合入. https://github.com/sgl-project/sglang-omni/pull/1235 已核实: 是（2026-11-11）

[6] Peter-Nie2003. "[Feat] Process replica." sgl-project/sglang-omni PR #1175, 2026-08-22 合入. https://github.com/sgl-project/sglang-omni/pull/1175 已核实: 是（2026-11-11）

[7] dyyfk. "[Feature] Traffic-aware adaptive dispatch policy for Code2Wav (Qwen3-Omni first)." sgl-project/sglang-omni issue #1236, 2026-07-31. https://github.com/sgl-project/sglang-omni/issues/1236 已核实: 是（2026-11-11）

[8] JiaxinD. "Serving workloads are host/CPU-dispatch-bound on H100: implications for optimization direction." sgl-project/sglang-omni issue #907, 2026-06-30. https://github.com/sgl-project/sglang-omni/issues/907 已核实: 是（2026-11-11）

[9] luojiaxuan. "Qwen3-ASR High-Concurrency Benchmark and Bottleneck Profile." `docs/developer_reference/qwen3_asr_concurrency_profile.md`，sgl-project/sglang-omni PR #1340, 2026-08-06 合入. https://github.com/sgl-project/sglang-omni/pull/1340 已核实: 是（2026-11-11）

[10] AlexxxXiong. "[Runtime Profiling] Qwen3-Omni." sgl-project/sglang-omni issue #1914, 2026-09-02. https://github.com/sgl-project/sglang-omni/issues/1914 已核实: 是（2026-11-11）

[11] Dayuxiaoshui. "[Runtime Profiling] MiniCPM-o 4.5 speech pipeline." sgl-project/sglang-omni issue #2399, 2026-09-27. https://github.com/sgl-project/sglang-omni/issues/2399 已核实: 是（2026-11-11）

[12] db-ol. "[Runtime Profiling] MiniCPM-o 4.5 Video-AMME: video plus spoken question in, text and speech out." sgl-project/sglang-omni issue #2580, 2026-10-06. https://github.com/sgl-project/sglang-omni/issues/2580 已核实: 是（2026-11-11）

[13] 0xjeffro. "[Benchmark] add client-side serving metrics and concurrency sweeps for duplex servers." sgl-project/sglang-omni PR #2843（其合入 commit 即本项目 pin `1845546e`）；另见 `docs/developer_reference/full_duplex_bench.md`. https://github.com/sgl-project/sglang-omni/pull/2843 已核实: 是（2026-11-11）

[14] jzheng17. "[Roadmap] Prefill/Decode Disaggregation." sgl-project/sglang-omni issue #1760, 2026-08-27（更新至 2026-11-11）；以及 PR #2756 "feat(qwen3-omni): add explicit thinker PD stages"（2026-11-11 仍为 open）. https://github.com/sgl-project/sglang-omni/issues/1760 已核实: 是（2026-11-11）

[15] izhl. "[Feature] Bound conversation history growth in `/v1/realtime` sessions." sgl-project/sglang-omni issue #2208, 2026-09-16. https://github.com/sgl-project/sglang-omni/issues/2208 已核实: 是（2026-11-11）

[16] AkazaAkane. "[Roadmap] Interactive Intelligence Runtime Optimization." sgl-project/sglang-omni issue #2994, 2026-10-20. https://github.com/sgl-project/sglang-omni/issues/2994 已核实: 是（2026-11-11）

[17] luojiaxuan. "Qwen3-Omni thinker (TP=2): prefill batches fragment under concurrent streaming audio — large throughput gap vs other framework." sgl-project/sglang-omni issue #760, 2026-06-12. https://github.com/sgl-project/sglang-omni/issues/760 已核实: 是（2026-11-11）

[18] migumnakkittikul. "[Fix][Qwen3-TTS] Fit the fused code predictor's pipeline to the GPU's shared memory." sgl-project/sglang-omni PR #3284, 2026-10-28 合入. https://github.com/sgl-project/sglang-omni/pull/3284 已核实: 是（2026-11-11）

[19] vLLM project. vllm-project/vllm-omni 仓库与文档：`docs/configuration/stage_configs.md`、`docs/design/feature/{disaggregated_inference,async_chunk}.md`、`docs/design/qwen3_omni_tts_performance_optimization.md`、`docs/cli/bench/serve.md`、`vllm_omni/deploy/`、`vllm_omni/model_executor/models/qwen3_omni/tool_speech.py`、release 列表。GitHub, 2025–2026. https://github.com/vllm-project/vllm-omni 已核实: 是（2026-11-11）

[20] Peiqi Yin, Jiangyun Zhu, Han Gao, et al. "vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models." arXiv:2602.02204, 2026. https://arxiv.org/abs/2602.02204 已核实: 是（2026-11-11）

[21] vLLM-Omni Team（arXiv 署名 "Omni Team"）. "vLLM-Omni Technical Report: A Unified Serving Runtime for Omni-Modality Generation." arXiv:2610.09307, 2026. https://arxiv.org/abs/2610.09307 已核实: 是（2026-11-11）

[22] vLLM-Omni Team, Ant Group SCT Team. "Experience and Lessons Learned from Serving Multi-Stage Qwen3-Omni in vLLM-Omni." vLLM Blog, 2026-07-01. https://vllm.ai/blog/2026-07-01-qwen3-omni-optimization 已核实: 是（2026-11-11）

[23] Jin Xu, Zhifang Guo, Jinzheng He, et al.（Qwen Team）. "Qwen2.5-Omni Technical Report." arXiv:2503.20215, 2025. https://arxiv.org/abs/2503.20215 已核实: 是（2026-11-11）

[24] Jin Xu, Zhifang Guo, Hangrui Hu, et al.（Qwen Team）. "Qwen3-Omni Technical Report." arXiv:2509.17765, 2025. https://arxiv.org/abs/2509.17765 已核实: 是（2026-11-11）

[25] Qwen Team. QwenLM/Qwen3-Omni README（"Minimum GPU memory requirements" 表与 vLLM 部署说明）. GitHub, 2025–2026. https://github.com/QwenLM/Qwen3-Omni 已核实: 是（2026-11-11）

[26] Junbo Cui, Bokai Xu, Chongyi Wang, et al.（MiniCPM-o Team, OpenBMB）. "MiniCPM-o 4.5: Towards Real-Time Full-Duplex Omni-Modal Interaction." arXiv:2604.27393, 2026. https://arxiv.org/abs/2604.27393 已核实: 是（2026-11-11）

[27] Alexandre Défossez, Laurent Mazaré, Manu Orsini, et al.（Kyutai）. "Moshi: a speech-text foundation model for real-time dialogue." arXiv:2410.00037, 2024；以及 kyutai-labs/moshi README. https://arxiv.org/abs/2410.00037 已核实: 是（2026-11-11）

[28] Rajarshi Roy, Jonathan Raiman, Sang-gil Lee, et al.（NVIDIA）. "PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models." arXiv:2602.06053, 2026. https://arxiv.org/abs/2602.06053 已核实: 是（2026-11-11）

[29] Jagadeesh Balam, Travis Bartley, Edresson Casanova, et al.（NVIDIA）. "NemotronLabs VoiceChat: An Open Full-duplex Speech-to-Speech Model with Tool Calling Capabilities." arXiv:2609.21967, 2026. https://arxiv.org/abs/2609.21967 已核实: 是（2026-11-11）

[30] Yinmin Zhong, Shengyu Liu, Junda Chen, et al. "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving." OSDI 2024；arXiv:2401.09670. https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin 已核实: 是（2026-11-11）

[31] Pratyush Patel, Esha Choukse, Chaojie Zhang, et al. "Splitwise: Efficient Generative LLM Inference Using Phase Splitting." ISCA 2024（DOI 10.1109/ISCA59077.2024.00019）；arXiv:2311.18677. https://arxiv.org/abs/2311.18677 已核实: 是（2026-11-11）

[32] Ruoyu Qin, Zheming Li, Weiran He, et al. "Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot." FAST 2025（Best Paper）；arXiv:2407.00079（arXiv 当前标题与数字不同）；trace 见 kvcache-ai/Mooncake `FAST25-release/`. https://www.usenix.org/conference/fast25/presentation/qin 已核实: 是（2026-11-11）

[33] Cunchen Hu, Heyang Huang, Liangliang Xu, et al. "Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads." arXiv:2401.11181, 2024. https://arxiv.org/abs/2401.11181 已核实: 是（2026-11-11）

[34] Gursimran Singh, Xinglu Wang, Yifan Hu, et al. "Efficiently Serving Large Multimodal Models Using EPD Disaggregation." ICML 2025（PMLR 267）；arXiv:2501.05460. https://arxiv.org/abs/2501.05460 已核实: 是（2026-11-11）

[35] Haoran Qiu, Anish Biswas, Zihan Zhao, et al. "ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving." ACM SoCC 2025（DOI 10.1145/3772052.3772254）；arXiv:2502.00937. https://arxiv.org/abs/2502.00937 已核实: 是（2026-11-11）

[36] Xianzhe Dong, Tongxuan Liu, Yuting Zeng, et al. "HydraInfer: Hybrid Disaggregated Scheduling for Multimodal Large Language Model Serving." arXiv:2505.12658, 2025. https://arxiv.org/abs/2505.12658 已核实: 是（2026-11-11）

[37] Zedong Liu, Shenggan Cheng, Guangming Tan, et al. "ElasticMM: Efficient Multimodal LLMs Serving with Elastic Multimodal Parallelism." NeurIPS 2025（Oral）；arXiv:2507.10069. https://arxiv.org/abs/2507.10069 已核实: 是（2026-11-11）

[38] Tianyu Guo, Tianming Xu, Xianjie Chen, et al. "RServe: Overlapping Encoding and Prefill for Efficient LMM Inference." arXiv:2509.24381, 2025. https://arxiv.org/abs/2509.24381 已核实: 是（2026-11-11）

[39] Kunxiong Zhu, Zhihao Shu, Hangyu Zheng, et al. "EAServe: Encode-Aware Disaggregated Serving for Multimodal Large Language Models." PACT 2026；arXiv:2609.31551. https://arxiv.org/abs/2609.31551 已核实: 是（2026-11-11）

[40] Konstantinos Papaioannou, Thaleia Dimitra Doudali. "Rocks, Pebbles and Sand: Modality-aware Scheduling for Multimodal Large Language Model Inference." arXiv:2603.26498, 2026. https://arxiv.org/abs/2603.26498 已核实: 是（2026-11-11）

[41] Lingxiao Zhao, Haoran Zhou, Yuezhi Che, et al. "Enabling Disaggregated Multi-Stage MLLM Inference via GPU-Internal Scheduling and Resource Sharing." arXiv:2512.17574, 2025. https://arxiv.org/abs/2512.17574 已核实: 是（2026-11-11）

[42] Euijun Chung, Yuxiao Jia, Aaron Jezghani, et al. "Characterizing CPU-Induced Slowdowns in Multi-GPU LLM Inference." IISWC 2026；arXiv:2603.22774. https://arxiv.org/abs/2603.22774 已核实: 是（2026-11-11）

[43] Yejin Lee, Anna Sun, Basil Hosmer, et al. "Characterizing and Efficiently Accelerating Multimodal Generation Model Inference." IEEE Micro, 2025（DOI 10.1109/MM.2025.3596539）；arXiv:2410.00215. https://arxiv.org/abs/2410.00215 已核实: 是（2026-11-11）

[44] Jeff J. Ma, Jae-Won Chung, Jisang Ahn, et al. "Cornfigurator: Automated Planning for Any-to-Any Multimodal Model Serving."（v1 题为 "Cornserve: Efficiently Serving Any-to-Any Multimodal Models"）arXiv:2512.14098, 2025–2026. https://arxiv.org/abs/2512.14098 已核实: 是（2026-11-11）

[45] Jae-Won Chung, Jeff J. Ma, Jisang Ahn, et al. "Cornserve: A Distributed Serving System for Any-to-Any Multimodal Models." CAIS 2026 Demo Track；arXiv:2603.12118. https://arxiv.org/abs/2603.12118 已核实: 是（2026-11-11）

[46] Atindra Jha, Naomi Sagan, Keisuke Kamahori, et al. "M*: A Modular, Extensible, Serving System for Multimodal Models." NeurIPS 2026（poster）；arXiv:2606.12688. https://arxiv.org/abs/2606.12688 已核实: 是（2026-11-11）

[47] Yuning Zhang, Dong Yuan. "HorizonServe: Coordinating Request Scheduling with GPU Sharing for Omni-Model Serving." arXiv:2608.01785, 2026. https://arxiv.org/abs/2608.01785 已核实: 是（2026-11-11）

[48] Xiangyu Zhi, Peiqi Yin, Sheng Guan, et al. "OmniCast: Interaction-Aware Serving for Real-Time Omni-Modal LLMs."（v1 题为 LiveServe）arXiv:2606.22983, 2026. https://arxiv.org/abs/2606.22983 已核实: 是（2026-11-11）

[49] Keisuke Kamahori, Rohan Kadekodi, Garv Ghai, et al. "VoxServe: Streaming-Centric Serving System for Speech Language Models." arXiv:2602.00269（引用 v2，2026-10-31）, 2026. https://arxiv.org/abs/2602.00269 已核实: 是（2026-11-11）

[50] Jiaying Meng, Bojie Li. "Metronome: Bound the Cache, Keep the Beat for Real-Time Interaction Model Serving." arXiv:2607.02640, 2026. https://arxiv.org/abs/2607.02640 已核实: 是（2026-11-11）

[51] Bin Xiao, Jingfu Dong, Changran Wang, et al. "Omni-Flow: A Unified Workflow Orchestration and Distributed KV Cache Sharing Framework for Multimodal Inference." arXiv:2606.31093, 2026. https://arxiv.org/abs/2606.31093 已核实: 是（2026-11-11）

[52] Chaofan Lin, Zhenhua Han, Chengruidong Zhang, et al. "Parrot: Efficient Serving of LLM-based Applications with Semantic Variable." OSDI 2024；arXiv:2405.19888. https://www.usenix.org/conference/osdi24/presentation/lin-chaofan 已核实: 是（2026-11-11）

[53] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, et al. "SGLang: Efficient Execution of Structured Language Model Programs." NeurIPS 2024；arXiv:2312.07104. https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html 已核实: 是（2026-11-11）

[54] Michael Luo, Xiaoxiang Shi, Colin Cai, et al. "Agentix: An Efficient Serving Engine for LLM Agents as General Programs."（arXiv 题为 "Autellix: …"）NSDI 2026；arXiv:2502.13965. https://www.usenix.org/conference/nsdi26/presentation/luo 已核实: 是（2026-11-11）

[55] Xin Tan, Yimin Jiang, Yitao Yang, Hong Xu. "Teola: Towards End-to-End Optimization of LLM-based Applications." ASPLOS 2025（arXiv journal-ref）；arXiv:2407.00326. https://arxiv.org/abs/2407.00326 已核实: 是（2026-11-11）

[56] Gohar Irfan Chaudhry, Esha Choukse, Haoran Qiu, et al. "Murakkab: Resource-Efficient Agentic Workflow Orchestration in Cloud Platforms." OSDI 2026；arXiv:2508.18298. https://www.usenix.org/conference/osdi26/presentation/chaudhry 已核实: 是（2026-11-11）

[57] Wenhao He, Youhe Jiang, Penghao Zhao, et al. "Efficient Multi-round LLM Inference over Disaggregated Serving." ICML 2026（arXiv comment）；arXiv:2602.14516. https://arxiv.org/abs/2602.14516 已核实: 是（2026-11-11）

[58] Reyna Abhyankar, Zijian He, Vikranth Srivatsa, et al. "InferCept: Efficient Intercept Support for Augmented Large Language Model Inference." ICML 2024（PMLR 235）；arXiv:2402.01869. https://proceedings.mlr.press/v235/abhyankar24a.html 已核实: 是（2026-11-11）

[59] Hanchen Li, Runyuan He, Qiuyang Mang, et al. "Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live." arXiv:2511.02230（v7）, 2025–2026. https://arxiv.org/abs/2511.02230 已核实: 是（2026-11-11）

[60] Zaifeng Pan, Ajjkumar Patel, Zhengding Hu, et al. "KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows." NeurIPS 2025；arXiv:2507.07400. https://arxiv.org/abs/2507.07400 已核实: 是（2026-11-11）

[61] Zhuohang Bian, Feiyang Wu, Zhuoran Li, et al. "TokenCake: A KV-Cache-centric Serving Framework for LLM-based Multi-Agent Applications." EuroSys 2027（arXiv comment）；arXiv:2510.18586. https://arxiv.org/abs/2510.18586 已核实: 是（2026-11-11）

[62] Yipeng Liu, Yingqiang Zhang, Feifei Li, Huanchen Zhang. "Ask the Tool, Don't Guess: Agent Tool Calls Hold Their Progress, and the Serving System Should Read It." arXiv:2609.18849, 2026. https://arxiv.org/abs/2609.18849 已核实: 是（2026-11-11）

[63] Yechen Xu, Xinhao Kong, Tingjun Chen, Danyang Zhuo. "Conveyor: Efficient Tool-aware LLM Serving with Tool Partial Execution." arXiv:2406.00059, 2024. https://arxiv.org/abs/2406.00059 已核实: 是（2026-11-11）

[64] Yifan Sui, Han Zhao, Rui Ma, et al. "Parallelizing Tool Execution and LLM Generation for Low-Latency Agent Serving." arXiv:2603.18897, 2026. https://arxiv.org/abs/2603.18897 已核实: 是（2026-11-11）

[65] Rakibul Hasan Rajib, Mengxin Zheng, Qian Lou. "AgentServeSim: Serving-System Simulation and Policy Search for LLM Agent Programs." arXiv:2606.09613, 2026. https://arxiv.org/abs/2606.09613 已核实: 是（2026-11-11）

[66] Shubham Tiwari, Tapan Chugh, Nash Rickert, et al. "CacheWise: Understanding Workloads and Optimizing KVCache Management for Efficiently Serving LLM Coding Agents." arXiv:2606.16824, 2026. https://arxiv.org/abs/2606.16824 已核实: 是（2026-11-11）

[67] Yuhang Xu, Shengzhong Liu, Dong Zhang, et al. "Nova: Real-Time Agentic Vision-Language Model Serving with Adaptive Cross-Stage Parallelization." arXiv:2509.21301, 2025. https://arxiv.org/abs/2509.21301 已核实: 是（2026-11-11）

[68] Xiaoxi Li, Wenxiang Jiao, Jiarui Jin, et al. "OmniGAIA: Towards Native Omni-Modal AI Agents." arXiv:2602.22897, 2026. https://arxiv.org/abs/2602.22897 已核实: 是（2026-11-11）

[69] Ke Hu, Slyne Deng, Chen Chen, et al. "A frontend-backend architecture for tool calls in full-duplex speech models." arXiv:2609.19334, 2026. https://arxiv.org/abs/2609.19334 已核实: 是（2026-11-11）

[70] Kyudan Jung, Hyunsin Park, Yoonhyung Lee, et al. "Hiding Tool Latency in On-Device Cascaded Voice Agent through Speculative Execution." arXiv:2610.07641, 2026. https://arxiv.org/abs/2610.07641 已核实: 是（2026-11-11）

[71] Jiin Kim, Byeongjun Shin, Jinha Chung, Minsoo Rhu. "The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective." HPCA 2026；arXiv:2506.04301. https://arxiv.org/abs/2506.04301 已核实: 是（2026-11-11）

[72] Banruo Liu, Haoran Qiu, Íñigo Goiri, et al. "Agentic Coding in the Wild: Characterizing GitHub Copilot Traces at Production Scale." arXiv:2608.00101, 2026. https://arxiv.org/abs/2608.00101 已核实: 是（2026-11-11）

[73] Kan Zhu, Mathew Jacob, Chenxi Ma, et al. "TraceLab: Characterizing Coding Agent Workloads for LLM Serving." arXiv:2606.30560, 2026；数据与 runner 见 uw-syfi/TraceLab（`README.md`、`replay/README.md`）. https://github.com/uw-syfi/TraceLab 已核实: 是（2026-11-11）

[74] Chaokun Chang, Yukun Zhou, Kaihua Fu, et al. "From LLM Inference to Agentic Workloads: Characterization and Implications for Serving Systems." arXiv:2608.15127, 2026. https://arxiv.org/abs/2608.15127 已核实: 是（2026-11-11）

[75] Yichao Yuan, Ankita Nayak, Souvik Kundu, Nishil Talati. "Agentic AI Workload Characteristics." IISWC 2026（arXiv comment）；arXiv:2605.26297. https://arxiv.org/abs/2605.26297 已核实: 是（2026-11-11）

[76] Jiahao Wang, Kaizhan Lin, Kaixi Zhang, et al. "SMetric: Rethink LLM Scheduling for Serving Agents with Balanced Session-centric Scheduling." arXiv:2607.08565, 2026. https://arxiv.org/abs/2607.08565 已核实: 是（2026-11-11）

[77] Microsoft Azure. Azure/AzurePublicDataset：`AzureLLMInferenceDataset2023.md`、`AzureLLMInferenceDataset2024.md`、`AzureLMMInferenceDataset2025.md`、`GitHubCopilotCodingAgentDataset2026.md`. GitHub, 2023–2026. https://github.com/Azure/AzurePublicDataset 已核实: 是（2026-11-11）

[78] Jovan Stojkovic, Chaojie Zhang, Íñigo Goiri, et al. "DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency." HPCA 2025（DOI 10.1109/HPCA61900.2025.00102）；arXiv:2408.00741. https://arxiv.org/abs/2408.00741 已核实: 是（2026-11-11）

[79] Yuxin Wang, Yuhan Chen, Zeyu Li, et al. "BurstGPT: A Real-World Workload Dataset to Optimize LLM Serving Systems." KDD 2025（DOI 10.1145/3711896.3737413）；arXiv:2401.17644；数据见 HPMLL/BurstGPT. https://github.com/HPMLL/BurstGPT 已核实: 是（2026-11-11）

[80] Jiahao Wang, Jinbo Han, Xingda Wei, et al. "KVCache Cache in the Wild: Characterizing and Optimizing KVCache Cache at a Large Cloud Provider." USENIX ATC 2025；arXiv:2506.02634；trace 见 alibaba-edu/qwen-bailian-usagetraces-anon. https://arxiv.org/abs/2506.02634 已核实: 是（2026-11-11）

[81] Yuxing Xiang, Xue Li, Kun Qian, et al. "ServeGen: Workload Characterization and Generation of Large Language Model Serving in Production." NSDI 2026；arXiv:2505.09999；数据见 alibaba/ServeGen. https://www.usenix.org/conference/nsdi26/presentation/xiang-servegen 已核实: 是（2026-11-11）

[82] William Nixon, Jon Durbin, Florian Standhartinger, et al. "A Year in LLM Serving: Workload Evolution, Caching and Load-Balancing." arXiv:2608.13573, 2026. https://arxiv.org/abs/2608.13573 已核实: 是（2026-11-11）

## 未核实线索

- **`flashinfer_cutlass` 在 sm_86 上的可用性。** 本项目此前在 A6000 上的本地记录显示，Qwen3-Omni talker 在该后端下报 "Invalid backend: 86"，改用 triton 后可以运行。我们没有找到对应的上游 issue 或文档，需要在 pin 版本上重新确认。
- **AutoRound INT4 thinker 在 A6000 上的失败。** 本地记录为 SGLang Marlin MoE 断言 bf16 activation 与 fp16 scale 一致。未在公开来源中核实。
- **PCIe-only A6000 节点上 NCCL P2P 挂起。** 本地记录显示，某些节点在 NCCL P2P/CUMEM 下挂起，需要 `NCCL_P2P_DISABLE=1` 并关闭 custom all-reduce。这与节点的 ACS/IOMMU 配置有关，未在公开来源中核实。
- **Green-Lab LiveKit 语音 agent trace**（Hugging Face 数据集 `Green-Lab/2026-10-16-111868-agentic-traces`）。检索显示其中只有语音 agent 的 LLM 文本部分（合成文本），并带 session 与父请求字段。来源是单一客户的非存档上传，provenance 无法核实。
- **vLLM RFC #37003**。检索到的摘录为 "40–60% of session wall time is spent paused on too…"（原文在此截断），没有给出数据来源和方法。
- **Qwen3.5-Omni Technical Report**（arXiv 2604.15804）。元数据已核实，其中的并发首包延迟数字未核实。
- **VoiceChat-TTS**（arXiv 2608.13831）。检索摘要提到 RTX A6000 上的延迟，未读原文。
- **OpenRSI × SGLang-Omni 博客。** upstream main README（2026-11）的新闻条目称，14 个合入 PR 在 H100 上带来 +7.6% 到 +326.5% 的吞吐提升。未读正文，各项条件未核实。
- **数字随版本变化。** Mooncake、HydraInfer、VoxServe 的 arXiv 摘要数字随版本改变。正文对 Mooncake 引 FAST'25 版本，对 VoxServe 引 v2；其他版本的数字未逐一核对。
