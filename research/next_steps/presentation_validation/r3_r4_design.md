# R3、R4 的设计和事先登记

> 2026-11-18 晚。本文是 R3（持续摄像头输入的瓶颈）和 R4（机器人会话的工作集和 KV 容量）的**事先登记**，写于任何 R3/R4 运行之前；提交时间就是登记时间。
> 原登记见 `research/next_steps/robotics_experiment_plan.md`（R3、R4 两节）；本文补上原登记没写到的细节，判定规则只增加不放宽。
> 代码事实都指 sglang-omni `520b49ee`（`~/Projects/sglang-omni-learning/repos/sglang-omni`）。

## 0. 先说结论性的源码事实（影响设计）

| 事实 | 位置 | 对设计的影响 |
|---|---|---|
| MiniCPM-o 的预处理阶段是 `SimpleScheduler(preprocessor)`，用默认的 `max_concurrency=1`，**工厂函数不接受 `max_concurrency` 参数**（别的模型如 dots_tts、auk 的预处理工厂接受） | `sglang_omni/models/minicpm_o/stages.py:50-63`；`sglang_omni/scheduling/simple_scheduler.py:62, 298-302, 343-401` | 实验 14 的“一次只做一个”是代码默认值，不是 CPU 不够。原计划里的 `--preprocessing.factory.max_concurrency 4` **在 520b49ee 上会被拒绝**，必须加一行补丁才能测“提高到 4”。`max_concurrency > 1` 时每个请求在线程里跑（`asyncio.to_thread`），要求预处理可重入，所以补丁附带重入检查 |
| 图像的缩放和切片在**预处理阶段的 GPU** 上做（`CUDAImageProcessor`，预处理阶段和 thinker 同卡），视频解码和抽帧在 CPU | `components/preprocessor.py`（`processor` 属性）、`components/image_processing.py:97-160`、`prompt_frontend.py`（`load_timed_video`） | R3 的“CPU 解码 / 抽帧 / 缩放切片 / H2D”要分开量：补丁在预处理里加分段计时（媒体加载、缓存键、视频帧、processor 调用） |
| 一个请求里**同一模态的所有媒体共用一个缓存键和一个 pad 值**：图像键是请求里全部图像的哈希，音频键是全部音频 + 所在轮次的哈希 | `components/preprocessor.py`（`compute_image_cache_key(images)`、`audio_cache_key`）、`request_builders.py:72-115`（`apply_mm_pad_values`） | 多轮会话如果每轮重发以前的图像/语音，第 k+1 轮的媒体集合变了，**所有媒体位置的 pad 值都变**，前缀缓存在第一个媒体 token 处就断开；编码器缓存（`StageOutputCache`，64 项）也按整组键查，同样不命中。这是“结构性重算”，和容量无关。R4 用“文字历史”做主设计，另设“重发 2 轮媒体”的对照来量它 |
| 同一会话的连续请求按**请求**轮转绑定副本，不按会话 | `sglang_omni/pipeline/replicas.py:269-315`（`RoundRobinBindingPolicy`） | 2 个 thinker 副本时，同一会话下一轮大约一半落到另一个副本，那里没有这个会话的前缀。R4 必须把“路由造成的未命中”和“驱逐造成的未命中”分开 |
| SGLang 的 `Prefill batch` 日志行只有每批的 `#cached-token`，**不写进程或副本**；thinker 和 talker 的行混在同一个服务日志里；驱逐只进 Prometheus 指标（`update_eviction_metrics`），sglang-omni 的阶段进程不暴露 | `sglang/srt/managers/scheduler_components/metrics_reporter.py:676-760`、`sglang/srt/mem_cache/radix_cache.py:553-580`、`base_prefix_cache.py:394-399`；`sglang_omni/pipeline/stage_workers.py:548-552`（日志格式不含 pid） | 现有日志**做不到**按副本统计 KV 占用、命中和驱逐。补丁在 thinker 调度器里加两种日志行（见第 1 节），带 pid 和 GPU |
| thinker 每个副本的 KV 池 127,039 token（实测，实验 19 等的启动日志）；thinker 最长上下文 8,192 | 启动日志；`config.py:73`（`max_seq_len=8192`） | R4 的会话长度要控制在 8k 以内（见 3.2） |

## 1. 补丁：`configs/runtime/patches/sglang_omni_ppv_kv.patch`

新 worktree：Mac `~/Projects/sglang-omni-ppv-kv`，集群 `~/workspace/sglang-omni-learning/repos/sglang-omni-ppv-kv`，都是 `520b49ee` + 这个补丁；运行配置 `configs/models/runtime_unicorn_ppv_kv.json`（`SGLANG_OMNI_EXP_KV=1`）。三处改动，默认都与原版行为相同：

1. `create_preprocessing_executor(..., max_concurrency=1)`：把参数传给 `SimpleScheduler`。命令行 `--preprocessing.factory.max_concurrency 4`。
2. 预处理分段计时：每次调用写一个请求事件 `exp_preprocess`（只在请求事件开着时写），字段 `media_load_ns`（解码/读入媒体，含视频抽帧前的解码）、`inline_media_load_ns`、`cache_key_ns`、`video_frames_ns`、`processor_ns`（切片、缩放、tokenize；图像在 GPU 上做，时间是 CPU 墙钟，**没有加 CUDA 同步**）、`total_ns`、`inflight_at_start`（本进程同时在跑几个预处理调用）、`images`、`video_frames`、`audios`、`input_tokens`。
3. thinker 的 KV 日志（`SGLANG_OMNI_EXP_KV=1`，只在 thinker 阶段写）：
   - `exp_kv_prefill pid gpu stage rid input_tokens cached_tokens extend_tokens evicted_tokens_total`：每个请求第一次进入 prefill 时一行（分块 prefill 只记第一块）。字段取自 SGLang 0.5.21 的 `Req`（在 `get_next_batch_to_run` 返回时已设置）：`input_tokens = len(req.origin_input_ids)`；`cached_tokens = len(req.prefix_indices)`，即匹配到的设备端前缀，prefill adder 记进自己 `#cached-token` 的正是同一个量（`schedule_policy.py:921`，`prefix_len` 来自 `:1161`/`:1298`）；`extend_tokens = req.extend_range.length`，即第一块的新 token（`Range` 由 prefill adder 的 `set_extend_range` 设置，`scheduler.py:4108` 的 `prepare_for_extend` 之前），没设置时写 −1。分析里每个请求要计算的 token 用 `input_tokens − cached_tokens`，不依赖分块；
   - `exp_kv_pool pid gpu stage total_tokens available_tokens evictable_tokens protected_tokens running_reqs waiting_reqs evicted_tokens_total evict_calls_total`：每秒最多一行；
   - 驱逐计数：包住 `tree_cache.evict`，累计 `num_tokens_evicted`（`EvictResult` 的字段）。
   - **测量永远不能拖垮服务**：`exp_kv_log`、池子那一行、驱逐计数和 `exp_preprocess` 都包在 try/except 里；第一次失败用 `logger.exception` 记一次（带异常），然后在这个调度器（或这个进程的预处理）里关掉对应的测量，请求照常进行。调用处用 `getattr(self, "exp_kv_enabled", False)`，不调用父类初始化的子类也不会出错。

CPU 检查：`configs/runtime/patches/check_sglang_omni_ppv_kv.py`（工厂参数、计时事件和在跑计数、日志行格式和限速、驱逐计数；`--model-path` 时再用真实预处理器检查 8 张不同图像串行和 4 线程并行的 `input_ids`、像素值逐位相同）。结果见第 6 节。

## 2. R3：持续摄像头输入在请求-响应路径上卡在哪

### 2.1 一句话的研究问题

When a robot sends camera frames continuously through the request-response API, which resource stops it first — the serial preprocessing slot, the CPU, or the GPU — and does a preprocessing concurrency of 4 move that limit?

### 2.2 设计

| 项 | 内容 | 依据 |
|---|---|---|
| 放置 | 1 张卡一个完整副本（`--text-only`），每个运行 1 张卡、7 个 CPU 核（8 卡节点平均每卡的份额） | 与实验 14 相同；预处理位是每个副本一个，结论按副本换算 |
| 运行时 | `unicorn_ppv_kv`（两种并发用同一份代码，只差一个参数） | 默认并发的方案不能用原版运行时，否则两组代码不同 |
| 摄像头 | 合成场景：渐变背景上 1–5 个运动的彩色圆（种子固定），640×360；每帧另有一条黑白方块编码（会话号、帧号），保证**没有两帧字节相同**，媒体缓存和前缀缓存都不会命中重复帧。`src/omni_research/sessions/frames.py` | 实验 14 的视频生成方式的单帧版 |
| 请求形状 | `frame`：每个时刻一个请求，带最新的一帧（PNG data URL）和问题“图里有几个圆”；`clip`：每个时刻带最近 2 秒、8 帧/秒的 H.264 短片（data URL），走视频解码和抽帧 | `frame` 对应“机器人每次问一张图”；`clip` 对应实验 14 发现串行的视频路径。用 data URL 是为了避免服务端从 NFS 读文件（真实机器人也是上传字节） |
| 每个会话的节奏 | 每 1/fps 秒一个时刻，fps ∈ {0.5, 1, 2}；各会话的起始相位在第一个间隔内随机 | — |
| 丢帧规则 | `drop_if_busy`：一个会话最多一个请求在飞，时刻到了而上一个还没回来，这一帧**丢弃**（不发）；发出的请求如果在下一个时刻之后才完成，记作**迟到** | 感知循环通常只处理最新的一帧；丢帧比例直接回答“能不能跟上摄像头” |
| 会话数 | 1、4、16 | 登记 |
| 预处理并发 | 1（默认）、4（补丁参数） | 登记 |
| 每格 | 预热 10 秒（不计）+ 计量 60 秒；开请求事件；问题里带会话号和帧号，每个请求从第一个用户 token 起就不同 | — |
| 筛选 | 2 形状 × 3 fps × 3 会话数 = 18 格，每种并发一个运行，格子顺序随机（种子 20261119，两种并发同一顺序）。两个 1 卡运行可以同时在两张卡上跑 | — |
| 确认 | 按 2.4 的规则选出的格子（最多 6 个），3 个区组 × 两种并发，配对 | 登记：先筛选再对有明显瓶颈或改善的配置做 3 区组 |

**这不是持续真实摄像头流的验证**：帧是合成的（压缩后只有约 2.5 KB，比真实摄像头的 JPEG 小得多，CPU 解码成本是下限），通过 `/v1/chat/completions` 一帧一个请求送进去；全双工 `/v1/realtime` 的图像追加（`sglang.input_image.append`）没有在这里测。

### 2.3 测量

| 指标 | 来源 |
|---|---|
| 丢帧比例、迟到比例、首个回答时间和完成时间 P50/P95（P99 只在样本 ≥ 100 时报）、数圆正确率、每请求输入 token | 客户端（`raw/camera_<phase>.jsonl`、`raw/requests_<phase>.jsonl`） |
| 预处理排队（`stage_input_received → stage_dispatch`）、预处理服务、图像编码器、thinker 排队和 prefill | 请求事件（`stage_breakdown`） |
| 预处理内部：媒体加载（CPU 解码 + 抽帧前的解码）、缓存键、视频帧、processor（GPU 上切片缩放 + tokenize） | 补丁的 `exp_preprocess` 事件 |
| 预处理位利用率 = 预处理服务时间之和 ÷（计量时长 × 并发数） | 派生 |
| 预处理所在进程的 CPU 核数 | 每进程 CPU 采样（pid 从事件文件名对应） |
| GPU 忙碌 | nvidia-smi 0.2 秒采样 |

H2D 传输没有单独的计时（`processor_ns` 包含它）；这一点写进限制。

### 2.4 判定（事先写定，`scripts/analysis/ppv_r3_report.py` 实现）

- 一格**饱和**：丢帧比例 ≥ 10% 或迟到比例 ≥ 20%。
- 饱和格的瓶颈：
  - **预处理受限**：预处理（排队 + 服务）≥ 服务端时间的 50%，且预处理位利用率 ≥ 0.8；
  - **GPU 受限**：GPU 忙碌 ≥ 80%，且预处理位利用率 < 0.8；
  - 其余记作“饱和、原因不明”，不硬归类。CPU 受限另看预处理进程的 CPU：并发 4 时如果进程用满 7 个核附近（≥ 6.3 核），记作“CPU 份额受限”。
- **并发 4 有改善**（同一格配对）：丢帧比例降低 ≥ 5 个百分点且相对 ≥ 20%，或首个回答 P95 降低 ≥ 20%。
- 选确认格：并发 1 饱和、或并发 4 有改善的格，按并发 1 的丢帧比例从大到小（相同时会话多的优先），最多 6 格。确认时 3 个区组都满足“有改善”才写“支持”；3 个区组都不满足写“不支持”；其余“不确定”。
- 预先写下的预测（Hypothesis）：`clip` 在 16 个会话、fps ≥ 1 时预处理受限（按实验 14 每 10 秒视频约 570 ms 推算，2 秒短片约 0.1 秒，单位置每秒约 9 个请求）；并发 4 在 `clip` 上有改善，在 `frame` 上改善小，因为单帧的解码很便宜、切片在 GPU 上。

### 2.5 预算

筛选：2 个 1 卡运行，每个约 30 分钟（18 格 × 约 85 秒 + 启动），同时跑约 0.5 小时墙钟，约 1 GPU-时。确认：最多 6 格 × 2 × 3 区组，约 1.2 GPU-时。合计约 2–2.5 GPU-时（原登记估计 2 小时 × 8 卡，按 1 卡副本设计用不了那么多）。

## 3. R4：多模态机器人会话会不会让 thinker 的 KV 缓存不够用

### 3.1 一句话的研究问题

Do multimodal robot sessions create a working set large enough to cause meaningful KV cache eviction and recomputation in the thinker, and if reuse is lost, is it lost to capacity or to request-level replica routing?

### 3.2 负载（`src/omni_research/sessions/robot_sessions.py`）

| 项 | 内容 |
|---|---|
| 会话 | 每个会话 = 一个机器人和一个人。开头是会话独有的前言（机器人编号、房间、长度随机 50–300 token 的笔记），所以会话之间只共享系统提示词 |
| 每轮 | 一张新的摄像头画面（R3 的合成摄像头）+ 一句合成语音请求（资产池里 Qwen3-TTS 生成的 24 个问题之一，按会话和轮次在末尾补不同数量的静音采样，保证**没有两段字节相同的语音**）+ 一行文字说明 |
| 工具 | 一半的轮先要一个简短的计划（文字，最多 32 token），等待模拟工具（对数正态，中位数 0.3 秒），把合成的工具结果（40–200 token）放进对话，再要最终回答 |
| 回答 | 最终回答用语音（最多 48 个文字 token），计划用文字 |
| 轮数、间隔 | 每个会话 4–10 轮（随机）；轮间是人的思考时间（对数正态，中位数 2 秒）；会话在前 20 秒内随机开始 |
| 历史 | 主设计：以前的轮以**文字**回传（说明 + 人说的话的文字稿 + 机器人的回答 + 工具结果），媒体只在当前轮。对照：重发最近 2 轮的图像和语音（`history_media=2`） |
| 不重复 | 种子固定；每个会话的前言、画面、语音都唯一，没有为了提高命中而重复同一个提示词 |
| 长度 | 估计 10 轮会话的最后一个请求约 2–4k token（文字历史），带 2 轮媒体时再多约 1–1.5k，都在 thinker 的 8,192 以内（冒烟核对） |

### 3.3 设计

- 放置：2+2+4（`configs/models/minicpm_o_4_5_exp_t2_k2_c4.yaml`），请求-响应路径，运行时 `unicorn_ppv_kv`，开请求事件。
- **先冒烟**（`ppv_r4_smoke`，4 个会话 × 2–3 轮，文字历史一段、2 轮媒体历史一段）：核对每个请求的输入 token 随轮次增长、带图像和语音的请求比只有文字的多出合理的 token 数、每个请求都能在 `exp_kv_prefill` 里找到（`rid` 对得上）、两个 thinker 副本都有 `exp_kv_pool` 行。不通过就修好再放正式运行。
- 正式：3 个区组，每个区组一个运行：会话数 8 → 16 → 32 → 64 → 128（每档一段，从小到大），最后 32 个会话带 2 轮媒体历史。每个区组用同一批会话内容（同一种子），区组之间只差服务重启和时间。
- 已知的混杂：同一运行里上一档留下的前缀缓存（可驱逐）会占着池子，下一档的驱逐里有一部分是驱逐这些“旧垃圾”。分析里每档报“开始时可驱逐的 token 数”，并且**判定只用本档会话自己的未命中**（下面的“损失的复用”），不用原始驱逐总数。

### 3.4 测量和定义（`scripts/analysis/ppv_r4_report.py`）

| 量 | 定义 |
|---|---|
| 副本 | thinker 调度器的（pid, GPU） |
| KV 容量 | `exp_kv_pool` 的 `total_tokens` |
| **驻留 KV** | total − available（在跑请求占的 + 可驱逐的缓存前缀） |
| 在跑请求占的 KV | total − available − evictable |
| **逻辑工作集** | 某时刻，每个活着的会话最近一个请求的（输入 + 输出）token，按那个请求所在的副本相加 |
| 共享前缀 | 每个会话第一个请求的 `cached_tokens` 中位数（= 所有会话共享的系统提示词） |
| 多模态 token | 同一轮里带媒体的请求和下一轮文字历史请求的输入 token 差（冒烟核对）；编码器缓存在这个设计里不会命中（媒体全部唯一），对照组里也不会命中（整组键变了） |
| 复用比例 | `cached_tokens` ÷ 同一会话上一个请求的 `input_tokens` |
| 干净的一对 | 上一个请求在同一副本，两次 prefill 之间这个副本没有驱逐（`evicted_tokens_total` 不变） |
| 参考复用 | 干净的一对的复用比例中位数，按“轮内（计划 → 回答）”和“下一轮”分开 |
| **损失的复用（= 重算的 token）** | max(0, 参考复用 × 上一个请求的输入 − 实际 cached)，按原因归类：换了副本 → **路由**；同一副本且中间有驱逐 → **驱逐**；对照组里媒体键变化造成的 → 记作**结构性** |
| 对首 token 的影响 | 各原因的请求的 TTFT 中位数，对比干净的一对 |
| 其他 | 会话完成时间、工具等待、TTFT P50/P95/P99（样本不够就写 N/A）、GPU 显存 |

### 3.5 判定（事先写定）

原登记（`robotics_experiment_plan.md` R4）的“比值始终 < 0.8 就停止”**不单独使用**。每档判定：

- **驱逐压力**：驱逐造成的损失 ≥ 本档 prefill token 的 10%，并且这些请求的 TTFT 中位数 ≥ 干净的一对的 1.2 倍。
- **路由代价**：路由造成的损失 ≥ 本档 prefill token 的 10%，并且这些请求的 TTFT 中位数 ≥ 干净的一对的 1.2 倍。
- 比值（逻辑工作集 ÷ 容量、驻留 ÷ 容量）照常报告，≥ 0.8 标出来，但不决定结论。
- 结论（按优先顺序）：
  1. 任何一档有驱逐压力 → 下一步设计“会话粘性绑定 / 按代价驱逐”的 A/B（`thunderagent_flashstates_related_work.md` 假设 B 的第二步）；
  2. 没有驱逐压力、但有路由代价 → 下一步只做会话粘性绑定的 A/B，不做驱逐策略；
  3. 都没有 → 写：**“当前实验条件下，尚无证据支持引入更复杂的 Agent-aware Cache Management。”**
- 3 个区组都给出同一结论才算稳定；不一致写“不确定”，并列出各区组的结论。
- 对照组（2 轮媒体历史）只报告结构性重算的大小和它对 TTFT 的影响，不进入上面的结论。
- 本轮**不移植** ThunderAgent 或 FlashStates。

### 3.6 预算

冒烟约 15 分钟 × 8 卡；正式 3 个运行，每个约 15–25 分钟（启动约 5 分钟，128 个会话那档在 2+2+4 上接近饱和，可能要几分钟），合计约 1.3 小时 × 8 卡 ≈ 10 GPU-时（原登记估计 2.5 小时 × 8 卡）。

## 4. 计划文件（`scripts/slurm/make_plan_ppv_r34.py` 生成）

| 文件 | 内容 | 什么时候放 |
|---|---|---|
| `configs/experiments/plan_ppv_r3_screen.json` | `ppv_r3_screen_c1_a2`、`ppv_r3_screen_c4_a2`（各 1 卡；第一次的 `ppv_r3_screen_c1`、`_c4` 因补丁崩溃，见 6.0） | 补丁的 CPU 检查通过、集群 worktree 就绪后 |
| `configs/experiments/plan_ppv_r3_confirm.json` | `--r3-confirm <格子>` 时生成，3 个区组 × 两种并发 | 筛选分析按规则选出格子以后 |
| `configs/experiments/plan_ppv_r4_smoke.json` | `ppv_r4_smoke`（8 卡） | 补丁检查通过后 |
| `configs/experiments/plan_ppv_r4.json` | `ppv_r4_block1..3`（8 卡） | 冒烟核对通过后 |

命令都是 `cd ~/workspace/omni-ppv && OMNI_REPO="$PWD" bash scripts/slurm/run_experiment.sh configs/experiments/ppv/<id>.json`。

## 5. 风险

- 视频 data URL：`video_url` 的 data URL 是否被 `load_timed_video` 接受，只在服务上能验证（冒烟会暴露）；不接受时改成写到节点本地 `/tmp` 的文件（`/tmp` 只剩约 2.8 GB，够用），设计不变。
- 并发 4 的预处理在线程里跑，`CUDAImageProcessor` 和 HF processor 的线程安全只做了 CPU 上的检查；服务上如果出现异常或输出不同（数圆正确率明显下降），这一组作废并报告。
- `processor_ns` 不含 GPU 同步，GPU 上的切片时间可能被低估。
- R4 的 `cached_tokens` 是请求第一次进入 prefill 时的匹配长度；被 retract 后重新 prefill 的请求不再记录（SGLang 自己的日志会记 retract）。驱逐计数在同一批里包括本请求自己分配时触发的驱逐，是上界。
- R4 的 128 个会话在 2+2+4 上会接近语音容量，TTFT 会被排队拉长；分析把“排队”与“缓存未命中”分开看（TTFT 对比只在同一档内做）。

## 6. 已完成的 CPU 准备（2026-11-18 晚，没有用 GPU）

### 6.0 补丁第一版的事故和修复（2026-11-18 23:30 前后）

- **现象**：R3 筛选的两个 1 卡运行（`ppv_r3_screen_c1`、`ppv_r3_screen_c4`）都在第一个请求时崩溃，没有任何测量。服务日志：`omni_scheduler.py` 的 `exp_kv_log` 里 `req.extend_input_len` → `AttributeError: 'Req' object has no attribute 'extend_input_len'` → “scheduler crashed”，pipeline 进程退出。两次运行的记录保留，不删。
- **根因**：补丁第一版按旧版 SGLang 的字段名写了 `req.extend_input_len`；SGLang 0.5.21（e00930c5）的 `Req` 没有这个属性，改用 `extend_range: Range(start, end)`。第一版的 CPU 检查用的是自己构造的替身对象，替身上正好有这个属性，所以没发现。
- **修复**：字段按上面第 1 节逐个对照 0.5.21 的源码；所有测量包进 try/except（失败一次、记录、关掉）；CPU 检查改用**真实的** `Req`（`sglang.srt.managers.schedule_batch.Req`，按 prefill adder 留下的状态设置 `prefix_indices` 和 `extend_range`）、真实的 `ForwardMode`，并经过真实的调用处 `get_next_batch_to_run`；另加“日志器抛错、对象缺字段、事件发送抛错时都不会抛进调度器或请求”的检查。
- **变异检查**：新检查在第一版补丁上失败，报的正是 GPU 上的同一个错误（`AttributeError: 'Req' object has no attribute 'extend_input_len'`）；在修复后的补丁上通过。
- 重跑的筛选用新编号 `ppv_r3_screen_c1_a2`、`ppv_r3_screen_c4_a2`；阶段名不变，所以内容种子不变，发出的帧和第一次完全相同。

| 检查 | 结果 |
|---|---|
| 补丁 CPU 检查 `check_sglang_omni_ppv_kv.py --model-path openbmb/MiniCPM-o-4_5`（登录节点，集群 worktree；修复后重跑，`PYTHONDONTWRITEBYTECODE=1`，先删掉了 GPU 运行留下的 3 个旧 .pyc） | 通过：真实 `Req` 经过真实调用处，`cached_tokens=120 extend_tokens=280`；抛错的日志器、缺字段的对象、缺字段的驱逐结果、抛错的事件发送都只记一次并关掉测量，请求照常返回；以及工厂参数默认 1、传 4 得到 4 个 worker；计时事件每次调用一个、在跑计数正确、出错也会减回；`exp_kv_prefill` 每个请求一行、`exp_kv_pool` 每秒最多一行、非 thinker 阶段不写；驱逐计数正确；**真实预处理器**（CPU 设备）上 8 张不同图像串行和 4 线程并行的 `input_ids` 与像素值逐位相同 |
| text 变体（= `--text-only`）是否接受 `--preprocessing.factory.max_concurrency 4` | 打补丁的 worktree：接受，预处理工厂参数为 `{"max_concurrency": 4, "gpu_id": 0}`；**原版 520b49ee**：拒绝，`Stage 'preprocessing' sets factory.max_concurrency but factory ...create_preprocessing_executor does not accept a 'max_concurrency' parameter` |
| 补丁一致性 | Mac worktree、集群 worktree 三个文件的 blob 哈希（preprocessor.py `04517b05a`、stages.py `21d6ac3fe`、omni_scheduler.py `6ef605ff4`）与补丁文件里的新 blob 相同；补丁文件 sha256 `23f75d8f518585ad0757636cb62b50b79ccdc347d5d90f6bd2d7916024940d1d` |
| 本地测试 | `tests/test_sessions_frames.py`、`test_sessions_camera.py`、`test_sessions_robot.py`、`test_ppv_r3_report.py`、`test_ppv_r4_report.py`、`test_make_plan_ppv_r34.py` 共 34 个，全部通过 |

没有验证的（只能在 GPU 上看）：视频 data URL 能否被服务端解码；GPU 上的 `CUDAImageProcessor` 在 4 个线程下是否同样逐位一致（CPU 检查走的是 CPU 路径）；`exp_kv_*` 的 `rid` 是否就是客户端的 `request_id`（源码上是 `payload.request_id`，冒烟核对）。
