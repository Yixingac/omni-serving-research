# ThunderAgent 和 FlashStates：机制、源码，以及和 Omni Serving 的结合点

> 2026-11-18。两篇都读了论文全文和公开源码（只读，没有安装或运行）。每条事实标明出处：论文的节/图/公式，或 `路径:行号@commit`。
> 标注：**[论文]** 论文的设计或主张；**[实测]** 论文报告的测量（附条件）；**[代码]** 公开源码里确实实现的；**[本项目]** 我们仓库里的实测或源码事实；**[推断]** 我们的分析。

## 0. 结论先行

- **ThunderAgent 值得借鉴的是“以程序（会话）为单位的准入和放置”这个思路**，不是它的具体实现：公开代码是一个只代理 `/v1/chat/completions` 的 HTTP 反向代理，不改引擎、不发驱逐命令，目标是吞吐而不是尾延迟；它的容量模型只有一个引擎的 KV 池。我们的多阶段服务（thinker、talker、code2wav 各有各的队列和容量）里，它的假设只有一部分成立。
- **FlashStates 的核心机制（任意位置的循环状态 checkpoint、依赖感知驱逐）对 MiniCPM-o 的纯 Transformer thinker 没有意义**——任何 token 的 KV 本来就可以作为续算点，而且公开代码遇到没有 Mamba 层的模型会直接拒绝启动。能借鉴的是**按“重算代价 ÷ 占用空间”做准入和驱逐**（GreedyDual-Size）以及它的评估方法（工作集 vs 容量曲线、miss 归因）。它和 Code2Wav 的组批/补齐**没有关系**，不放进 P1。
- **我们的现有证据还不支持“联合调度一定比单独优化组批好”。** 当前合成 Agent 会话太短（中位 2 次请求、约 425 个输入 token、工具耗时微秒级），按会话固定路由只提高命中率 0.7–1.8 个百分点（实验 19）。两篇论文的收益都只出现在“工作集超过缓存容量”的时候。所以第 5 节的 3 个假设都从“先测工作集和容量”开始，任何一个都可能在第一步就被证伪。

---

## 1. ThunderAgent

**出处**：Hao Kang, Ziyang Li, Xinyu Yang, Weili Xu, Yinfang Chen, Junxiong Wang, Beidi Chen, Tushar Krishna, Chenfeng Xu, Simran Arora. “ThunderAgent: A Fast, Simple, and Program-Aware Agentic Inference System.” ICML 2026, PMLR 306:55630–55657. https://proceedings.mlr.press/v306/kang26j.html ；arXiv:2602.13692。代码 https://github.com/ThunderAgent-org/ThunderAgent （读的是 `7ddc861`，2026-06-05）。

### 1.1 七个问题

| 问题 | 回答 |
|---|---|
| 1. 什么是 LLM Program | [论文] “跨多次模型调用和工具执行持续存在的一等调度单位”（p.2）。形式化为 P = ⟨ID, c, T, L, τ, s⟩：标识、上下文 token 数、工具环境集合、后端放置、阶段（Reasoning / Acting）、状态（Active / Paused / Terminated）（式 1，表 1）。[代码] 程序标识从请求体的 `program_id`、`extra_body.program_id` 或 `X-Session-ID` 头里取，都没有就归到 `"default"`（`app.py:15-36`）；阶段由请求边界推断：请求到达记为 Reasoning，响应带回 usage 记为 Acting（`router.py:476-477`） |
| 2. 和请求级调度的区别 | [论文] 按程序记账：周期性检查 Σc_p > C_total（式 6，“thrashing”条件）；超了就暂停程序——先停 Acting 的、再停短的（S_pause = 1/c + 1[τ=A]）；恢复时先 Reasoning、再短的（S_restore）；所有数据并行副本共用一个全局等待队列，恢复的程序可以去任何副本（“动态迁移”）。[代码] 每 5 秒一次（`router.py:746-771`），暂停逻辑 `:773-805`，恢复逻辑 `:807-932`（按空闲容量最大的副本放），容量 = C_total − (Σ_R c + w·Σ_A c + 每程序 100)（`backend/state.py:185-194`） |
| 3. 为什么工具时间影响 KV 利用率 | [论文] 每一轮的上下文是上一轮的延伸，理论上几乎全部可复用；请求级引擎在程序等工具时把它的 KV 驱逐掉，工具返回后整段重算（thrashing）；一直保留又浪费显存，而工具耗时长尾、不可预测，TTL 类方法（如 Continuum）不好用。[实测] 图 1b（vLLM，GLM-4.6，SWE-Agent，8×H100）：并行工作流从 24 增加到 72–96 时，命中率从约 98% 掉到约 31%，延迟最多 7.14 倍。**论文没有给出“工具等待占比”或“等待期间占着的 KV”的直接测量** |
| 4. 如何管理多个 Agent 的依赖和资源 | [论文] 上面的暂停/恢复 + 全局队列 + 工具环境的生命周期管理（Docker、端口回收、异步准备环境）。[代码] 只有前两项；`/programs/release` 只删调度器里的状态，**工具环境的回收和异步准备没有在公开代码里找到** |
| 5. 准入、路由、工具资源管理各解决什么 | 准入：避免 KV 超额导致的反复驱逐；路由：把恢复的程序放到空闲容量最大的副本，平衡负载；工具资源：回收沙箱、提前准备环境、省磁盘（图 6a：3.3 TB → 0.8 TB）。前两项有代码，第三项没有 |
| 6. 哪些有公开实现 | [代码] 有：程序状态、周期容量检查、“先停 Acting、先停短的”、分组恢复和负载均衡放置、全局队列、可选的衰减（`--use-acting-token-decay`，默认关，t 以秒计，只用在恢复路径，`state.py:196-214`）。没有：工具 GC、异步环境准备、工具 API。代码里还有论文没写的东西（RL 权重同步时的保持）和与论文不一致的地方（复现脚本没开衰减开关；`shared_tokens` 从不更新）。它是代理，**不修改引擎**：真正驱逐什么由引擎自己的 LRU 前缀缓存决定 |
| 7. 在 SGLang-Omni 的多阶段模型上是否成立 | 见 1.3 |

### 1.2 实测结果和条件

| 结果 | 条件 |
|---|---|
| 对 vLLM 每分钟步数 1.48–3.58 倍（图 4 的 6 个子图里有一个是 1.24 倍，低于正文写的范围）；对 Continuum 1.17–3.31 倍 | vLLM 引擎；GLM-4.6 或 Qwen3-235B FP8，TP8，8×H100；ToolOrchestra 在一张 RTX 5090 |
| RL rollout 对 vLLM + SGLang Gateway 1.79 倍、3.92 倍（表 2） | 2×8×H100，N=144 |
| 扩展到 8 节点 64 卡，对 SGLang Gateway 的加速从 1.79 到 2.39 倍；调度一次 13.5–22.7 ms（A.3） | — |
| 消融：全局队列 vs 本地队列 1.28 倍（4 节点）；“先驱逐短的”对“先驱逐长的”最多约 2 倍吞吐 | — |
| **低并发时没有收益**（A.5）；论文明确目标是吞吐“而不是尾延迟” | — |

我们的调研代理在论文里还发现几处数字和标签不一致（表 2、表 7、图 14/16 同一组数对应不同模型和基线标签；A.4 的 P50 正文和图不一致）。引用时用正文的范围，并注明“图 4(f) 是 1.24 倍”。

### 1.3 在 Omni 服务上：哪些成立，哪些不成立

| ThunderAgent 的前提 | 在我们的服务上 | 依据 |
|---|---|---|
| 上下文是前缀延伸，可复用 | **成立**（thinker） | [本项目] thinker 用 RadixCache（`EvictHeapRadixCache`，LRU），多模态占位符按内容哈希，相同图像/音频能命中（调度审计 §5） |
| 程序等工具时 KV 被驱逐导致重算 | **未测**；当前合成 Agent 不太可能触发 | [本项目] 会话中位 2 次请求、约 425 个输入 token，工具 24–60 µs；命中率约 85% 主要来自共享系统提示（实验 19、12） |
| 只有一个 KV 池要管 | **不成立** | talker 有自己的 KV（131072 token），code2wav 有批和队列；语音回复要依次占用三个阶段 |
| 目标是吞吐 | **对实时语音不合适** | 暂停一个正在说话的回合会直接加到首段音频时间上；它的暂停最长可到下一个 5 秒周期，强制恢复要 1800 秒 |
| 按 JSON 字符估 token、按 SSE 事件计数 | **不成立** | base64 音频、文字和音频混合的流 |
| 外部 HTTP 代理能看到所有请求 | **不成立** | `/v1/realtime` 是 WebSocket；阶段之间的交接在服务内部，代理看不到 |
| 衰减 2^−t（t 以秒计） | **时间尺度不对** | 约 5 秒就放掉一个会话的保留，短于一次用户说话的回合 |

---

## 2. FlashStates

**出处**：Chenxi Li\*, Weili Xu\*†, Tianyi Zhang, Jue Wang, Zhongzhu Zhou, Chenfeng Xu. “FlashStates: Unlocking Flexible Hybrid State Management for Agentic LLM Inference.” alphaXiv `2611.flashstate`（v1 2026-11-09，v2 2026-11-10；**没有 arXiv 编号，没有会议**）。网站 https://flashstates.ai/ ；代码 https://github.com/FutureMLS-Lab/FlashStates （`6d3503e`，只有脚本、配置和文档；引擎代码在两个子模块：vLLM 分支 `3cfd432`，基于 vLLM v0.33.0；Mooncake 分支 `59ca03f`）。注意论文实验用的是 vLLM `ec8a372`（0.32.0rc1），公开代码是 v0.33.0。

### 2.1 八个问题

| 问题 | 回答 |
|---|---|
| 1. Attention KV 和循环状态的区别 | KV 随前缀长度线性增长，可以截断到任何更早的位置续算；循环状态（Mamba/GDN/KDA 等）大小固定、与前缀长度无关，**只能在保存它的那个位置续算，不能回退**。论文的模型里一个 checkpoint 占 5 个 KV 块，相当于 16K–42K token 的 KV 字节数 [论文] |
| 2. Hybrid State | 混合模型（注意力层 + 循环层）的完整状态 = 整段前缀的注意力 KV + 恰好在前缀末尾的循环 checkpoint（+ 卷积/索引器的辅助状态）[论文] |
| 3. 任意位置 checkpoint 怎么做 | 在每个请求的确切结束位置无拷贝地保存；可选地在一次 prefill 内从 kernel 里导出一个中间位置（KDA 每 16 token；GDN kernel 里 16、对调度器声明 64）[论文]。[代码] 导出 kernel 只在计算能力 10.0（B200 级）上运行 |
| 4. 为什么 checkpoint 要和 KV 依赖匹配 | 前缀树上只有“KV 和循环状态同时在”的节点才能续算。只剩 KV、没有对应 checkpoint 时，命中了也要重算（“依赖 miss”）。论文把冗余 prefill 拆成放置、容量、依赖三类 miss（式 5）[论文] |
| 5. 依赖感知的准入/驱逐 | 把一个前缀末端（frontier）的 KV 和 checkpoint 作为一个整体缓存和驱逐；会马上被驱逐的新条目不准入；等待中的请求命中的条目先钉住 [论文/代码] |
| 6. 按代价驱逐 | 准入和驱逐共用一个 GreedyDual-Size 优先级 H(u) = stamp + g/s：g 是从最近可用状态重算到 u 的 prefill 秒数，s 是只有 u 才释放的块数 [论文] |
| 7. Prefill 重算代价和保留代价 | 重算代价按模型拟合一次：C(a,b) = d(b−a) + e(b²−a²)/2·10⁵（GLM 的 d、e = 1.862e-5、3.33e-6）。理论部分把工具等待建模成 TTL：最优保留时间只取决于代价密度 g/s（命题 3）；按代价密度设保留时间不差于 LRU 的共享时间，只有代价密度不同时才严格更好（命题 4）。**主机层的回载时间没有进模型** [论文] |
| 8. CPU/GPU/SSD 层次 | [代码] 实现了 GPU + Mooncake 主机内存（请求结束时写穿，按字节的 GDS 驱逐）；**SSD 没有实现**，Mooncake 的 offload 模式被拒绝；两种策略都不计传输代价 |

### 2.2 实测结果和条件

| 结果 | 条件 |
|---|---|
| 48 个会话时，对 vLLM：GLM-5.3-Flash 解码 3.02×、prefill 3.14×、p90 TTFT −88%；Nemotron 3 Super 2.32×、2.27×、−77%；Qwen3.8-Flash-Next 1.54×、1.54×、−79% | 2×B200；SemiAnalysis 的 Claude Code trace 按 token id 回放 |
| 动机测量：vLLM 默认 4,352 token 的块、decode 中不存 checkpoint，每请求约重算 2.5K token；SGLang v0.5.22 把循环状态放在单独的 LRU 池，24–32 个会话时依赖 miss 占未缓存 token 的 33–43%，受影响请求重算中位数 66–68K token | 同上 |
| **收益只在工作集超过缓存容量（32–49 个会话）后出现**；低负载约 1.0 倍，GLM 在 24/32 个会话时 p90 TTFT 反而 +51%/+47%（论文没有讨论） | 同上 |
| 512 GiB 主机层：输出吞吐比 vLLM+Mooncake 高 2.1–41.4%，比 SGLang+HiCache 高 3.2–36.2%；但 32–48 个会话时 SGLang+HiCache 的 p90 TTFT 更好 | 同上 |
| 没有和 LMCache、SSD 层比较；只有 Blackwell GPU | — |

网站上的“输出 2.8 倍、p50 TTFT 低 16.3 倍”来自 48 个 Agent 的演示回放，不在论文里，代码不能复现，不引用。

### 2.3 对我们的适用性

| 部分 | 对 MiniCPM-o thinker（纯 Transformer，SGLang RadixCache） | 对混合模型 thinker（Qwen3-Next、GLM、Nemotron-H 类） | 对 Code2Wav |
|---|---|---|---|
| 任意位置 checkpoint、kernel 导出、依赖 miss | **无意义**：任何 token 的 KV 都是续算点；radix 树只驱逐叶子，缓存天然前缀闭合 | 有意义；请求末端的 checkpoint 不需要 kernel（论文里 Qwen、Nemotron 的收益没用到 kernel 导出），A6000 上也能做 | 无关 |
| GreedyDual-Size 按代价驱逐 | **只能借设计思想**：每 token 的代价密度 ≈ d + e·depth；按命题 4，只有代价密度差别大时才比 LRU 好——短语音上下文时接近 LRU；长上下文、或把多模态编码器的重算代价算进去时才可能有差别 | 直接有意义 | 无关 |
| “会马上被驱逐的不准入”、钉住等待请求的命中 | 可借鉴 | 可借鉴 | 无关 |
| 工具等待的 TTL 分析 | 可借鉴，用来定保留时间；它的 60 s / 300 s 常数是编码 Agent 的值，不适用于语音 | 同左 | 无关 |
| 主机层写穿 + 整会话驱逐 | SGLang 0.5.21 有 HiCache，但 **sglang-omni 写死关闭**（`omni_scheduler.py:596-597`）；要用先得接上 | 同左 | 无关 |
| 评估方法（按 token id 回放、miss 归因、工作集 vs 容量曲线、拟合 d/e） | **直接可用**；注意 MiniCPM-o 的模板是否会删掉之前的推理 token，删掉就不能精确续接 | 同左 | 无关 |

接入点（如果要做）：SGLang 的 `EvictionStrategy.get_priority` 可插拔（`python/sglang/srt/mem_cache/evict_policy.py`），但 GDS 还需要在驱逐循环里更新水位（`radix_cache.py:553-581`）；sglang-omni 自己构造了 `EvictHeapRadixCache`（`sglang_omni/scheduling/sglang_backend/cache.py:62-63`），改动要落在这里。

---

## 3. 我们的服务实际暴露了哪些 Program / Session 信息和调度入口

都是 sglang-omni `520b49ee` 的源码事实（上游 main 在这些点上没有变化），详见调度审计笔记。

| 项 | 现状 | 位置 |
|---|---|---|
| 请求 API 里的会话/优先级字段 | `/v1/chat/completions` 只接受 `request_id` 和 `user`；**`user` 收下但从不使用**；没有 `session_id`、`metadata`、`priority`；多余字段被 Pydantic 静默丢弃 | `sglang_omni/serve/protocol.py:50-110`；`serve/openai_api.py:1063-1188` |
| 阶段之间传的元数据 | `request_id`、采样参数、输出模态、媒体列表、副本绑定；**没有会话、用户、优先级或截止时间** | `sglang_omni/proto/request.py:33-94`；`proto/messages.py:52-62` |
| 副本选择 | 准入时按轮转一次性绑定所有阶段的副本，忽略 `request_id`，不看负载；同一会话的连续请求会轮流落到不同的 thinker 副本，**会抵消 radix 缓存** | `sglang_omni/pipeline/replicas.py:269-315`；`pipeline/coordinator.py:555-559` |
| 阶段内的排序 | thinker、talker 走 SGLang 的 `SchedulePolicy`，0.5.21 支持 `priority` 策略和抢占，sglang-omni 也把这些开关传进去了；但 **MiniCPM-o 的请求构造从不设 `Req.priority`**，所以开了也没用 | `scheduling/omni_scheduler.py:563-582, 1716-1720`；`models/minicpm_o/request_builders.py:186-192` |
| code2wav 的排序 | `SimpleScheduler.collect_batch`，FIFO；实验 10 的 `collect_length_aware` 是现成的模板 | `scheduling/simple_scheduler.py:149-199` |
| 缓存 | thinker RadixCache（LRU）；talker 没有 radix 缓存；HiCache 关闭；没有“钉住某个会话前缀”的 API | `scheduling/sglang_backend/cache.py:57-65` |
| 全双工会话 | `--variant session` 时每个 `session_id` 只绑定一次副本，整个会话走同一条路线；默认最多 2 个会话 | `pipeline/sessions.py:149-153`；`models/minicpm_o/native_config.py:17` |

**可以插入的地方**（按改动从小到大）：
1. 准入时的绑定：实现 `BindingPolicy`（`replicas.py:263-266`）传给 `Coordinator(binding_policy=...)`；现在 `mp_runner` 构造 `Coordinator` 时没传这个参数，要改一行（`pipeline/mp_runner.py:719-729`）。
2. API：把 `user`（或新加的字段）拷进 `metadata`，供绑定策略和阶段使用。
3. 阶段内排序：在 MiniCPM-o 的请求构造里设 `Req.priority`；code2wav 的出队顺序沿用实验 10 的补丁。
4. 交接时的二次选择：`Stage.resolve_target_instance`（`pipeline/stage/runtime.py:219-233`），但没有下游负载信号，要先加。

---

## 4. 系统级对照表

| 层次 | ThunderAgent | FlashStates | 当前 Omni Serving（MiniCPM-o，520b49ee） | 可能的结合点 |
|---|---|---|---|---|
| Agent / Program | 程序是一等单位，`program_id` + 阶段（Reasoning/Acting）+ 状态 | 没有程序概念，按会话的前缀树和 frontier 管理 | 服务端没有程序或会话概念（只有全双工模式有 `session_id`）；我们的客户端有会话结构（`src/omni_research/tracing/agent_harness.py`） | 把会话 id 和“这一步要不要说话”带进服务（第 3 节入口 1、2） |
| Request Admission | 全局容量检查 Σc ≤ C_total，暂停/恢复程序 | “会马上被驱逐的不准入”，按 GDS 优先级 | 每阶段各自准入：thinker 64、talker 32 个名额 + KV 预留；code2wav 无上限的 FIFO 队列；没有跨阶段的限流 | 按**下游语音阶段的队列**做准入（假设 A） |
| Stage Scheduler | 单阶段（一个 LLM 引擎） | 单阶段 | 三个阶段独立调度；实验 3：系统里超过约 50–80 个请求后只是在排队 | 阶段感知的优先级：要说话的最终步、截止时间近的回复优先 |
| Batching | 引擎原样 | 引擎原样 | thinker/talker 连续批；code2wav FIFO、上限 16、不等待（实验 10 的长度组批是原型） | 截止时间 + 长度分桶（假设 C，P1 的延伸） |
| KV / Hybrid State | 用程序记账代替 KV 记账；不直接管 KV | 混合状态的 checkpoint、依赖感知驱逐、GDS | thinker 纯注意力 KV（144 KiB/token），LRU；talker 的 KV 不复用 | 会话级保留 + 按代价驱逐（假设 B）；只有工作集超过容量才有意义 |
| GPU Placement | 恢复时放到空闲最大的副本，可迁移 | 无 | 静态配置（2+2+4 等），准入时轮转绑定副本 | 会话粘性绑定（在空闲边界才换副本）；负载感知的绑定 |
| Memory / SSD | 不涉及（论文假设暂停的 KV 已丢失） | GPU + 主机内存（Mooncake）；SSD 未实现 | 只用 GPU；SGLang 的 HiCache 在 sglang-omni 里关闭 | 对长会话启用主机层（要先接上 HiCache） |
| Deadline / Tail Latency | 明确不优化尾延迟 | 报 p90 TTFT，但低负载时可能变差 | 没有截止时间；SLO 只在我们的分析里（TTFT ≤ 1 s 且 端到端/语音秒 ≤ 0.5） | 实时交互的截止时间（首段音频、打断响应）作为调度目标（R 主线） |

---

## 5. 核心问题：一起考虑 Agent 依赖、缓存状态和阶段信息，是否有比单独优化组批更好的策略？

**目前的回答：有理由相信在两种情况下会更好，但现有数据都还不能证明。**
1. 负载里既有只出文字的工具步、又有要说话的最终步时，单独的组批优化只作用在 code2wav 上，而“哪一步会进入语音阶段”这个程序级信息能在更上游（thinker 准入、副本绑定）就控制语音阶段的排队。实验 19 的数据说明负载组成会改变瓶颈（文字 Agent 时拆分放置的 talker 卡 0% 忙，混合负载时 2+6 的 thinker 名额排队），但那是**静态放置**的比较，没有测过动态策略。
2. 会话上下文（多模态、多轮）大到超过 thinker 的 KV 容量时，按会话保留和按代价驱逐才会有用；当前合成负载远没到这个规模。

下面最多 3 个假设，按研究价值排序。

### 假设 A：按下游语音队列和“这一步是否要说话”做 thinker 准入

| 项 | 内容 |
|---|---|
| 具体问题 | 服务没有跨阶段限流：thinker 不断把写完的回答交给 talker/code2wav，过载时语音阶段的队列失控（实验 9：2+1+5 时 talker 前排队 8 秒；实验 6：并发 256 时 code2wav 阶段 42% 在排队）。对实时交互，这些排队直接变成首段音频的等待 |
| 现有证据 | 实验 3：系统里约 50–80 个请求以后只是在排队，吞吐几乎不涨；实验 9 的开环：超过容量后在途请求一路上涨；实验 19：负载组成决定瓶颈在哪个阶段 |
| 相关工作 | ThunderAgent 的程序级准入（按 KV 容量）；Metronome 对全双工会话的在线准入（按 KV 窗口）；HorizonServe 在单卡上协调准入和 SM 分配；OmniCast 按首音频优先级调度（`docs/01_research/prior_work.md` 4.3） |
| 和已有工作的区别 | 准入信号来自**下游的语音阶段**（talker/code2wav 的队列深度）和程序阶段（最终的说话步 vs 工具步），而不是单个引擎的 KV |
| 需要的源码修改 | 最小版不改服务：在客户端（或一个反向代理）对“要说话的请求”限制在途数量 N，工具步不受限。正式版：API 带“是否要说话”和会话 id（第 3 节入口 2），coordinator 读下游队列深度 |
| 最小实验 | 2+2+4，W3b 和一个“工具步 + 最终说话步”的合成语音 Agent；开环速率超过容量（实验 9 的 23.5 请求/秒以上）；在途说话请求上限 N ∈ {48, 64, 80}（取自实验 3 的 50–80 区间）对不限 |
| 对照组 | 不限流（当前行为），同一放置、同一负载、同一速率 |
| 指标 | goodput（SLO）、首段音频时间 P95/P99、被推迟的比例和推迟时间、会话完成时间 |
| 可能失败的原因 | 限流只是把排队从服务里搬到客户端，goodput 不变；或者工具步太短，区分步骤没有作用 |
| 研究价值 | 中。做法本身（准入控制）不新；新的是在多阶段 omni 流水线里用程序阶段决定准入，并给出“准入点放在哪个阶段”的定量答案。成本低，可以很快证伪 |

### 假设 B：多模态机器人会话的工作集会超过 thinker 的 KV 容量，此时会话级保留和按代价驱逐有用

| 项 | 内容 |
|---|---|
| 具体问题 | 机器人会话每一轮都带新的图像/音频，历史不断变长；多个会话同时进行、并在工具执行或用户说话时空闲。LRU 会在空闲时驱逐会话的前缀，下一轮要重算（还可能重跑视觉编码器） |
| 现有证据 | **只有间接证据**：thinker 每个副本的 KV 池 127,039 token（启动日志，Measured）；图像每个切片 64 token、最多 9 片（源码）；轮转绑定会让同一会话的连续请求落到不同 thinker 副本（源码）。直接证据（工作集是否超过容量）**没有**：实验 19 的会话太短，按会话路由只多 0.7–1.8 个百分点命中率 |
| 相关工作 | ThunderAgent（程序级暂停/恢复，避免 thrashing）；FlashStates（GDS 按代价驱逐、工作集 vs 容量曲线）；InferCept、Continuum、KVFlow、TokenCake（工具期间的 KV 保留/卸载，`prior_work.md` 5.2）；Omni-Flow（服务端保存会话状态，50 轮多模态会话 JCT −7.5%/−19.3%）；Metronome（全双工会话的 KV 窗口） |
| 和已有工作的区别 | 已有工作都是纯文本 Agent 或无工具的 omni 多轮对话；我们是“多模态输入 + 工具等待 + 最后语音输出”的会话，重算代价里包括编码器（GPU）和 CPU 预处理（实验 14：视频 87% 在 CPU），代价密度差别可能远大于纯文本——这正是按代价驱逐比 LRU 好的前提（FlashStates 命题 4） |
| 需要的源码修改 | 第一步不需要：只做测量（每个 thinker 副本的活跃会话上下文之和、命中率、重算 token 数、编码器缓存命中）。第二步：会话粘性绑定（`BindingPolicy` + 把 `user` 当会话 id，约几十行）；第三步才考虑驱逐策略（`EvictHeapRadixCache` 的优先级） |
| 最小实验 | 合成机器人会话（每轮 1 张图或 N 帧 + 一句语音提问 → 一次文字工具步 → 语音回答），会话数 8–128 扫描，2+2+4；先画“活跃上下文之和 / KV 容量”随会话数的曲线 |
| 对照组 | 轮转绑定 + LRU（当前行为）对 会话粘性绑定 + LRU；只有曲线越过 1 时才做驱逐策略的比较 |
| 指标 | 前缀命中率、每轮重算的 token 数、编码器调用次数、TTFT、首段音频时间、会话完成时间 |
| 可能失败的原因 | 到 128 个会话工作集仍低于容量（那就证明对这个模型和 48 GB 卡不需要会话级缓存管理）；或者 sglang-omni 的阶段缓存（编码器输出缓存）已经吸收了大部分重算 |
| 研究价值 | **高（如果工作集确实超过容量）**：把 Agent KV 管理的问题带到多模态、多阶段的场景，并给出“什么时候需要”的边界。第一步的测量本身就有价值，成本低 |

### 假设 C：在语音阶段按截止时间（slack）组批，同时保留长度分桶的效率

| 项 | 内容 |
|---|---|
| 具体问题 | 实验 10 的按长度组批能多出 30% 的处理能力，但让一部分短请求多等 56%/21%；实时交互里短回复（“好的”“我看到了”）往往最急 |
| 现有证据 | 实验 10（公平性代价，描述性）；实验 6（code2wav 排队）；SLO 的定义（端到端 ÷ 语音秒 ≤ 0.5）本身给了每个请求一个可计算的截止时间 |
| 相关工作 | VoxServe（按 TTFA 和卡顿 SLO 的流式调度）、OmniCast（按播放进度和首音频优先级）；ThunderAgent 不考虑截止时间；FlashStates 无关 |
| 和已有工作的区别 | 在不能流式的批处理声码器里（MiniCPM-o 的 chat 路径整段生成），把“长度分桶”和“截止时间”放进同一个出队规则，量化两者的取舍 |
| 需要的源码修改 | 在实验 10 的 `collect_length_aware` 上加截止时间（P1 的“最长等待”是它的特例），截止时间 = 到达时间 + 0.5 × 语音秒 − 上游已用时间 |
| 最小实验 | 合并进 P1：多一个 EDF+长度 的设置 |
| 对照组 | FIFO16、Length8、Length8 + 最长等待 |
| 指标 | 截止时间未满足的比例（按长度分组）、语音秒/秒、P99 |
| 可能失败的原因 | 和“最长等待”没有可区分的差别；或者完整服务没饱和时没有可调的空间（实验 10 验证已经看到） |
| 研究价值 | 中低：主要是工程上的取舍，适合作为 P1 的一个设置，不单独立项 |

## 6. 不建议做的结合

- **把 FlashStates 的 checkpoint 机制搬到 MiniCPM-o**：thinker 没有循环状态，机制不适用；公开代码会拒绝启动。
- **把 FlashStates 放进 P1（Code2Wav 组批）**：没有任何机制上的联系。
- **直接部署 ThunderAgent 代理**：它代理不了 WebSocket 和服务内部的阶段交接，容量模型只覆盖一个 KV 池，目标是吞吐；在实时语音上可能让首段音频变差。
- **在当前合成 Agent 负载上比较驱逐策略**：工作集远低于容量（会话中位 2 次请求、约 425 token），任何策略都等价于 LRU。
- **在 A6000 上复现 FlashStates 的导出 kernel**：只支持计算能力 10.0。

## 7. 本文引用的我们自己的证据

| 证据 | 位置 |
|---|---|
| 会话结构、工具耗时、按会话路由的效果、名额排队 | `research/01_性能瓶颈/实验19_负载类型和最佳分工/README.md` 3.3–3.6、图 4–7 |
| 回放与实时运行的差别、命中率 85% → 2%（完全替代内容时） | `research/01_性能瓶颈/实验12_回放能不能代替真实负载/README.md` |
| 系统里请求数的有用区间 50–80 | `research/01_性能瓶颈/实验03_同时处理多少请求最好/README.md` |
| 长度组批的收益和公平性代价 | `research/01_性能瓶颈/实验10_Code2Wav_Batching/README.md` 图 9 |
| thinker KV 池大小、talker 名额 | `results/p0_*_talker_slots.json`；`p0_p3_experiment_plan.md` P0 |
