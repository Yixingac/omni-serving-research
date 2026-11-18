# 实验 6–8 的数据字典

本文件说明实验 6、7、8 在仓库里的结果文件有哪些字段、单位是什么、从哪里算出来。原始数据（请求记录、服务端事件、kernel trace、日志）在集群的 `results/experiments/<运行目录>/` 下，不进公开仓库。

## 通用约定

- **时间单位**：字段名带 `_ms` 的是毫秒，`_s` 是秒，`_ns` 是纳秒，`_us` 是微秒。
- **时钟**：
  - 服务端事件用各进程的墙钟（`time.time_ns()`）。
  - 客户端请求记录用客户端进程的 `monotonic_ns`，带一个墙钟锚点。
  - torch trace 的时间是 `baseTimeNanoseconds + ts × 1000`（墙钟纳秒）。
  - 同一台机器上，三者可以对齐。实验 6 实测了打点事件和 trace 的对齐误差：中位数 39 微秒，见 `实验06_Code2Wav时间拆解/results/clock_alignment.json`。
- **重复单位**：一次独立启动的服务（区组）。
- **汇总格式**：JSON 里的 `interval` 结构是 `{"runs": [每个区组的值], "n": 有效个数, "mean": 均值, "ci95": [下限, 上限]}`，区间用 t 分布（自由度 = n − 1）。
- **百分位**：`p50`、`p95`、`p99` 是一个区组内全部请求或批的百分位，不是跨区组的。
- **段（phase）的后缀**：
  - A = 不打点、不开 profiler；
  - B = 开服务端事件；
  - C = 开 3 秒 torch profiler 窗口，窗口内也有事件。
- **错误**：`errors` 是失败的请求数，不含预热；`timeouts` 单列。

## 新增的服务端事件（测量补丁 `configs/runtime/patches/sglang_omni_instr_exp06_08.patch`）

只在环境变量 `SGLANG_OMNI_EXP_TRACE` 列出该部分、并且这一段开着事件记录时才写。和 sglang-omni 自带的事件写在同一个 JSONL 文件里，每行有 `request_id`、`stage`、`event_name`、`timestamp_ns`、`pid`、`run_id`、`metadata`。

| 事件 | 部分 | 什么时候写 | metadata 里的字段 |
|---|---|---|---|
| `exp_batch_formed` | c2w | code2wav 凑齐一批、开始计算时，每个请求一条 | `batch_id`（进程内递增）、`batch_size`、`batch_cost`（这一批的编码 token 数）、`dequeue_ns`（这个请求被调度线程取出的时刻）、`queue_depth_at_dequeue`、`queue_depth_at_formed` |
| `exp_batch_done` | c2w | 这一批算完、交出之前，每个请求一条 | `batch_id`、`batch_size`、`compute_start_ns` |
| `exp_vocode` | c2w | 每一批 vocode 结束时，每个请求一条（内容是这一批的） | 时间点 `prepare_start_ns`、`references_start/end_ns`、`upload_start_ns`、`flow_issue_start/end_ns`、`hift_issue_end_ns`、`download_end_ns`、`split_end_ns`、`postprocess_start_ns`；GPU 跨度 `gpu_flow_start_to_flow_end_ms`、`gpu_flow_end_to_hift_end_ms`（CUDA event，在这一批原本就有的同步之后读取）；`rows`、`token_lengths`、`hift_groups`；flow 走的路径 `flow_solve_path`（dense_graph / packed_graph / packed_eager / eager）、`flow_solve_rows`、`flow_solve_frames`；`request_codec_tokens`、`request_ids`；`SGLANG_OMNI_EXP_TRACE` 含 `digest` 时还有 `request_codec_sha256` |
| `exp_thinker_step` | thinker | thinker 调度循环的每一轮（`request_id` 固定为 `__thinker_step__`） | `batch_size`、`forward_mode`、`use_lookahead`、`blocked_by`（none / below_min_batch / not_decode / runner_ineligible）、`blocker_speech`、`blocker_sampling`、`blocker_logprob`（被挡的请求数）、`waiting_requests`、`has_chunked_request`、`flush_reason`（mixed_chunk / sync_batch）、各步 CPU 时间 `launch_ns`、`resolve_ns`、`drain_ns`、`run_ns`、`process_ns` |
| `exp_thinker_hidden` | digest（只在原型副本里） | thinker 把一个请求的隐藏状态交给 talker 时 | `rows`、`hidden_sha256`（隐藏状态 bf16 字节的 SHA-256 前 16 位，含 dtype）、`speech_lookahead`、`snapshot_hidden` |

## 实验 6：`research/01_性能瓶颈/实验06_Code2Wav时间拆解/results/`

### `requests.csv`：每个请求在 code2wav 里的时间（B 段，成功、非预热的请求）

| 列 | 含义 |
|---|---|
| `run` | 运行目录名 |
| `concurrency` | 闭环并发 |
| `request_id` | 请求 ID（客户端生成，服务端沿用） |
| `total_ms` | code2wav 阶段总时间：放进收件队列（stage_dispatch）到结果交出（stage_complete） |
| `queue_ms` | 收件队列里的排队：stage_dispatch → 被调度线程取出（dequeue_ns） |
| `batching_ms` | 凑批：取出 → 这一批凑齐（exp_batch_formed） |
| `compute_ms` | 这一批的计算：凑齐 → 算完（exp_batch_done），整批共享 |
| `handoff_ms` | 交出：算完 → stage_complete |
| `queue_depth_at_dequeue` | 被取出时队列里还有几个请求 |
| `batch_pid`、`batch_id` | 所在的批：进程号 + 批号 |
| `audio_s` | 这个请求的语音秒数（客户端收到的） |

`total_ms = queue_ms + batching_ms + compute_ms + handoff_ms`，四段首尾相接。

### `batches.csv`：每一批（B 段）

| 列 | 含义 |
|---|---|
| `size`、`batch_cost`、`rows` | 批里的请求数、编码 token 数、vocode 的行数 |
| `compute_ms` | 整批计算的墙钟时间 |
| `prepare_ms`、`references_ms`、`upload_ms` | CPU：准备、参考音频、上传 |
| `flow_issue_ms`、`hift_issue_ms` | CPU：发 flow 的 kernel、发 HiFT 的 kernel（含等 flow 算完的同步） |
| `download_wait_ms`、`split_ms`、`postprocess_ms` | CPU：等波形拷回、拆分、后处理 |
| `gpu_flow_ms`、`gpu_hift_ms` | GPU 跨度（CUDA event）：flow 开始 → flow 结束、flow 结束 → HiFT 结束。包含同卡 talker 插进来的时间，不是独占计算时间；和 CPU 各步时间重叠，不能相加 |
| `flow_solve_path`、`flow_solve_rows`、`flow_solve_frames` | flow 走的路径（是否用 CUDA graph）、行数、帧数 |
| `hift_groups` | HiFT 按长度分了几组 |
| `queue_depth_at_formed` | 凑齐时队列里还有几个请求 |
| `audio_s`、`max_tokens` | 这一批的语音秒数（编码 token 数 ÷ 25）、最长请求的编码 token 数 |
| `pid`、`batch_id`、`formed_ns`、`run`、`concurrency` | 标识和时间 |

### `summary.json`

- `per_run[].phases["c<并发>_<A|B|C>"]`：每段的数据。
  - `client`：吞吐、语音秒/秒、端到端/TTFT/首段音频的百分位、错误数。
  - `busy_thinker_gpus`、`busy_talker_gpus`：nvidia-smi 忙碌比例。
  - B 段有 `breakdown`（请求和批的统计）；C 段有 `kernel`（kernel trace 统计）。
- `by_level["c<并发>_<A|B|C>"]`：上面各指标跨区组的 `interval`。
  - `c<并发>_overhead_B`、`c<并发>_overhead_C`：B、C 段相对同区组 A 段的变化。
  - `c<并发>_flow_path`：flow 各路径所占比例。
- kernel 统计（C 段）：
  - `graph_kernel_time_share`：CUDA graph 里的 kernel 时间占全部 kernel 时间的比例；
  - `launch_api_ms`、`sync_api_ms`：CPU 在发 kernel、同步调用上花的时间；
  - `queue_delay_ms`：每次发 kernel 到它的第一个 kernel 开始；
  - `by_range`：按打点事件的 CPU 区间归属到 c2w.prepare / flow / hift / download。

### 其他

- `figures_data.json`：每张图背后的数。
- `clock_alignment.json`：打点事件和 trace 两个时钟的对齐检查。

## 实验 7：`research/01_性能瓶颈/实验07_Code2Wav独立GPU/results/`

### `summary.json`

- `per_run[]`：
  - `arm`（放置）、`block`、`restarts`（服务中途重启）、`after_restart`（重启发生在哪一段）、`issues`（事件缺失等）。
  - `closed["<并发>"]`、`open["<速率>"]`：
    - `client`；
    - `breakdown`：code2wav 的拆分，同实验 6；
    - `c2w_flow_us_per_frame_b1`：批大小为 1 的批，flow GPU 跨度 ÷ 帧数的中位数，单位微秒/帧，带样本数 `n`；
    - `talker_gen_ms_per_audio_s`：talker 从开始 prefill 到模型路径结束的时间 ÷ 语音秒数；
    - `talker_queue_ms`：talker 排队，scheduler_queue_enter → scheduler_prefill_start；
    - `gpu[角色]`：这一档时间窗内各角色卡的 nvidia-smi 忙碌比例 `busy`、最大显存 `mem_max_mib`、平均 SM 时钟 `sm_clock_mhz` 和平均功耗 `power_w`。角色是 thinker、talker、code2wav，同卡时是 talker+code2wav，还有 unused；
    - 描述性指标（不参与判定）：
      - `hop_ms`：talker → code2wav 的传递时间，talker 的 stage_hop_sent → code2wav 的 stage_input_received；
      - `c2w_gpu_ms_per_audio_s`：这一档各批 flow + HiFT 的 GPU 时间之和 ÷ 这些批的语音秒数之和（`batches` 是计入的批数）；
      - `thinker_ms_per_token`：同实验 8，thinker 的 prefill 结束到最后一个 token，每个 token 平均多少毫秒。
  - `kernel_c128`：并发 128 的 3 秒 trace 里，每张 talker 卡上 talker 和 code2wav 的 kernel 忙碌时间、两者同时在跑的时间（`overlap_ms`），以及 `overlap_share_mean`。
- `by_arm[放置]`：各指标跨区组的 `interval`。错误率 > 5% 的档位不计入，`usable_blocks` 列出计入的区组。
- `verdicts`：H7a、H7b、H7c，判定规则见实验 7 的 hypotheses.md。

### `levels.csv`

每个运行的每一档（`section` = closed 或 open，`level` = 并发或速率），列出吞吐、端到端 P50/P95/P99、TTFT P50、错误数、code2wav 排队和总时间、talker 每秒语音的时间和排队、flow 每帧时间，以及首个音频的 P50（`first_audio_p50_s`）、传递时间的 P50（`hop_p50_ms`）、code2wav 每秒语音的 GPU 时间（`c2w_gpu_ms_per_audio_s`）、thinker 每个 token 的 P50（`thinker_ms_per_token_p50`）。`arm` 是配置名，对应 README 里的方案 A–E。

## 实验 8：`research/01_性能瓶颈/实验08_Thinker_Lookahead/results/`

| 文件 | 内容 |
|---|---|
| `summary_main.json`、`summary_supp.json` | 第一阶段主区组、补测区组的汇总（`exp08_lookahead.py`）。`per_run[].phases["<条件>_c<并发>_<B|C>"]`：B 段有 `steps`（每轮事件的统计：提前发射比例、被挡原因、flush 原因、批大小、CPU 时间）、`thinker_ms_per_token`（prefill 结束到最后一个 token，每个 token 平均多少毫秒）、`client`；C 段有 `graph_steps`（`exp05_step_sync.py` 按相邻两次 cudaGraphLaunch 切步：`idle_ms_p50` 每步 GPU 空档、`busy_ms_p50` 每步 GPU 忙、`steps` 步数、`waited_share` CPU 等这一步算完才发下一步的比例） |
| `phase1.json` | 合并后的表 `cells[条件][并发][指标]`（`interval`），以及判定 `verdicts`：H8a–H8d、第三阶段进入条件。`flush_share` 是 flush 的轮次占全部轮次的比例 |
| `zoom.json` | 图 1 用的几步时间线：`gpu` 是（开始毫秒、时长、是否在 graph 里），`cpu` 是（开始、时长、类型：sync / launch / copy_wait / api） |
| `proto_check.json` | 第三阶段正确性检查。`check_c1`：按问题分组的比较（`per_arm` 每组自己重复之间一致的问题数；`vs_off` 和“关”相同、不同、不可比的问题数）。`check_c1_by_index`：第 k 个请求对第 k 个请求，`same` 是三样东西（文字、隐藏状态哈希、语音编码哈希）各有多少个相同。`check_c8`：混合批的错误数、缺哈希数、隐藏状态行数减文字 token 数的分布。`c1_requests`：每个请求的明细 |
| `phase3.json` | 第三阶段性能对比（`exp08_proto_perf.py`）：`contrasts[并发][段][指标]` 的“关”“开”、配对差、相对差，以及 `verdicts`。`sync_calls_per_step` 是每步里 CPU 等 GPU 的调用次数（同步调用，加上阻塞超过 1 毫秒的拷贝） |
