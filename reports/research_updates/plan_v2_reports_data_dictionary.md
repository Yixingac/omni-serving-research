# 用 plan v2 已有数据补写的报告：数据字典

2026-11-18 补写了几个“跑过但没写报告”的实验（实验 19 等），数据都来自 plan v2 两批 8 卡运行（2026-11-15、11-16，以及 11-17 的重跑）。这份文档说明它们的 `results/` 里每个文件、每个字段的含义。原始数据的格式和通用约定（单位后缀、时钟、预热请求）见[实验 9–11 数据字典](exp09_11_data_dictionary.md)第一、二节，这里不重复。

## 一、和实验 9–11 不同的约定

| 约定 | 含义 |
|---|---|
| 运行次数 | 每种配置只跑过 1 次，没有区组，所以没有置信区间；结论都是初步的 |
| 百分位 | 线性插值（`omni_research.analysis.metrics.percentile`，和 numpy 默认相同）。`summarize_plan_v2.py` 用的是“取最近的名次”，小样本时两者会差一点，所以本文档的数和旧汇总表可能有末位差别 |
| Agent 的百分位 | 单次请求和会话时长的 P50、P90 直接取自 `analyze_experiment.py` 的 `analysis/analysis.json`（也是线性插值） |
| SLO（plan v2） | 请求成功，TTFT ≤ 1 秒；语音回答另要“端到端 ÷ 输出语音秒”≤ 0.5 |
| 节点 | `第一批节点`（Job 625796，驱动 570 + cuda-compat）或 `compute-03`（Job 454290，驱动 580），由运行目录的日期决定 |
| 分法的名字 | `r8` = 8 个完整副本，`p4` = 4 组 1+1，`s<t>x<k>` = t+k（GPU 0 到 t−1 放 thinker，其余放 talker 和 code2wav） |

## 二、实验 19（`exp19_report.py`）

### `summary.json`

| 字段 | 含义 |
|---|---|
| `rule` | 判定用的常数：`near_best_tolerance`（0.03）、`near_best_tolerance_wide`（0.06）、`hit_gain_pts`（3）、`ttft_rel`（0.05）和 SLO 的两个阈值 |
| `w3b[]` | 纯语音参照（实验 2 的运行）：`run`、`layout`、`label`、`thinker_gpus`、`closed_max_rps`（W3b 闭环各档吞吐的最大值）、`closed_max_at`（出现在哪个并发） |
| `mix[]` | 每种分法一个混合负载运行，见下 |
| `agent[]`、`agent_short[]` | 第二批加长版、第一批每档太短的 Agent 运行，见下 |
| `verdicts`、`verdicts_tol_wide` | 容差 3%、6% 时的判定：`w3b_reference`、`B-H1`、`B-H2`、`B-H3` |

`mix[]` 每一项：

| 字段 | 含义 |
|---|---|
| `run`、`experiment_id`、`layout`、`label`、`thinker_gpus`、`node` | 运行目录、配置名、分法、中文名、放 thinker 的 GPU 数、节点 |
| `closed[]` | 闭环每档：`concurrency`、`rps`（`analyze_experiment.py` 的 `tp_requests_per_s`）、`n`（计量请求数）、`ok`、`slo_all`（全部请求满足 SLO 的比例）、`by_type`、`role_busy` |
| `closed[].by_type.<W1/W2/W3b>` | `n`、`ok`、`errors`、`slo`、`ttft_p50_s`、`ttft_p90_s`、`e2e_p50_s`、`e2e_p90_s`、`e2e_per_audio_p50`、`e2e_per_audio_p90`（只有 W3b 有值） |
| `closed[].role_busy` | 这一档（第一个请求发出到最后一个完成）各角色 GPU 的 nvidia-smi 忙碌比例均值（%）：`thinker`、`talker`；8 个完整副本是 `full` |
| `closed_max_rps`、`closed_max_at` | 闭环各档吞吐的最大值和它出现的并发 |
| `open[]` | 开环每档：`rate_rps`、`of_closed_max`（速率 ÷ 本运行闭环最高吞吐）、`n`、`ok`、`slo_all`、`by_type` |
| `errors` | 闭环里失败请求按原因计数（`CUDA out of memory`、`client fd limit` 或错误信息的前 80 个字符） |

`agent[].phases.<档名>`（档名如 `text_c256`、`lo_voice_c512`、`toollat2_text_c256`）：

| 字段 | 含义 |
|---|---|
| `concurrency`、`agent`、`routing`、`tool_latency_s` | 同时进行的会话数；`text` 或 `voice`；`least_outstanding`（默认）或 `session_affinity`；模拟工具延迟的中位数（0、0.5、2） |
| `sessions`、`task_success`、`model_requests`、`requests_per_session` | 会话数、任务成功的会话数、模型请求数、每会话平均请求数 |
| `request_p50_s`、`request_p90_s`、`session_p50_s`、`session_p90_s` | 单次模型请求、整个会话的耗时百分位 |
| `ttft_p50_s`、`ttft_p90_s` | 由请求记录算的 TTFT（只算成功的请求） |
| `prefix_hit` | 前缀缓存命中：这一档所有 prefill 里，从缓存拿到的输入 token ÷ 全部输入 token |
| `span_s`、`requests_per_s`、`mean_in_flight` | 这一档的时长（第一个请求发出到最后一个完成）、每秒完成的模型请求、同时在模型里的平均请求数（耗时之和 ÷ 时长） |
| `tool_s_total` | 这一档所有工具调用的总耗时（秒） |
| `role_busy` | 这一档各角色 GPU 的 nvidia-smi 忙碌比例均值（取自 `analysis.json` 的 `gpu_by_phase`） |
| `errors` | 失败请求按原因计数 |

`verdicts.B-H1`、`B-H2.levels.<档>`：`verdict`（支持 / 不支持 / 不确定）、`near_best`（最好的分法及容差内的分法）、`thinker_gpus`、`reference_thinker_gpus`（固定为 2）、`metric`、`values`。`B-H3`：`part_a`（无法检验）、`part_b`（`hit`、`ttft` 和范围）、`comparison[]`（每档的 `hit_gain_pts`、`ttft_p50_rel`、`ttft_p90_rel`、`request_p50_rel`、`session_p50_rel`，都是“按会话固定 ÷ 默认 − 1”或百分点差）。

### `mix_closed.csv`、`agent_phases.csv`

和 `summary.json` 里 `mix[].closed[]`、`agent[].phases` 的字段相同，摊平成表，方便用表格软件看。`agent_phases.csv` 的 `batch` 列区分第二批（加长）和第一批（每档太短）。

## 三、实验 12（`exp12_report.py`）

### `summary.json`

| 字段 | 含义 |
|---|---|
| `rule.tolerance` | 判定用的 10%（取自预测 D-H1） |
| `live` | 被回放的实时运行（实验 19 的 8 个完整副本、文本 Agent、256 个会话那一档），字段同下面的 `replays[]`，另有 `phase` |
| `replays[]` | 每个回放运行一项，见下 |
| `verdicts.part1_copies_overestimate` 等 | D-H1 三句话的判定：`verdict`、用到的相对差（`e2e_p50_rel` 或 `rel`）、`compared`（比的是哪两个运行） |

`replays[]` 每一项（只算模型请求，不算工具调用）：

| 字段 | 含义 |
|---|---|
| `run`、`experiment_id`、`label` | 运行目录、配置名、短名（如 `dep_prefix_1x`） |
| `mode`、`payload`、`speed`、`copies`、`max_sessions` | 回放方式（`dependency` 依赖感知 / `request` 请求级）、内容（`auto` 原文 / `prefix_surrogate` 保留前缀的替代 / `surrogate` 完全替代）、倍速、每个会话复制几份、只取前多少个会话 |
| `source` | 被回放的 trace：“实验名 段名”（不保存路径） |
| `engine_status` | 回放引擎自己记的各状态数量，包括工具调用和等待 |
| `model_requests`、`ok`、`skipped`、`errors` | 模型请求数、成功数、因同一会话前面的步骤失败而跳过的数、失败按原因计数 |
| `valid` | 没有客户端句柄错误时为 true；false 的运行不参加比较 |
| `span_s`、`requests_per_s` | 第一个请求发出到最后一个完成的时间；成功请求数 ÷ 这段时间 |
| `mean_in_flight`、`peak_in_flight` | 同时在服务里的请求数：平均（耗时之和 ÷ 时长）、最多 |
| `e2e_p50_s`、`e2e_p90_s`、`ttft_p50_s`、`ttft_p90_s` | 单次请求耗时、首个文字时间的百分位 |
| `sessions`、`session_p50_s`、`session_p90_s` | 会话数（按成功的请求算）、会话时长百分位 |
| `output_tokens_mean`、`prompt_tokens_p50` | 平均输出 token、输入 token 中位数 |
| `finish` | 成功请求按结束原因计数：`stop`（遇到结束符）、`length`（到了长度上限） |
| `prefix_hit` | 服务端统计的前缀缓存命中（输入 token 中从缓存拿到的比例） |
| `gpu_util_mean` | 8 张卡 nvidia-smi 采样利用率的平均（%） |
| `series` | 从第一个请求发出起，每秒同时在服务里的成功请求数的平均（图 2） |
| `vs_live`、`vs_dep_orig_same_speed` | 相对实时运行、相对同样倍速的“依赖感知、原文、不复制”的变化（回放 ÷ 参照 − 1），指标有 `requests_per_s`、`e2e_p50_s`、`e2e_p90_s`、`ttft_p50_s`、`ttft_p90_s`、`session_p50_s`、`session_p90_s`、`output_tokens_mean`、`prompt_tokens_p50`、`prefix_hit`、`mean_in_flight`；无效的运行为 null |

### `runs.csv`

每个运行（含实时运行）一行，列是上表的主要数字。

## 四、实验 14（`exp14_report.py`）

### `summary.json`

| 字段 | 含义 |
|---|---|
| `rule` | `linear_efficiency`（0.8）、`gpu_flat_pts`（5） |
| `runs[]` | 1、2、4、8 个副本的运行：`run`、`experiment_id`、`replicas`、`gpus`、`phases`、`phase_gpu_busy` |
| `runs[].phases.<段>.<并发>` | 每档：`n`、`ok`、`span_s`（第一个请求发出到最后一个完成）、`rps`（成功数 ÷ `span_s`）、`e2e_mean_s`、`e2e_p50_s`、`e2e_p90_s`、`little_rps`（并发 ÷ 平均耗时）、`cpu_cores_per_replica`（服务进程平均用的核数 ÷ 副本数）、`gpu_busy_mean`（这一档里用到的卡的 nvidia-smi 采样平均，%）、`per_replica_concurrency`、`rps_per_replica` |
| `runs[].phase_gpu_busy.<段>` | 整段（4 档合起来）用到的卡的忙碌比例平均（%，取自 `analysis.json` 的 `gpu_by_phase`） |
| `events` | 事件补测：`phases.<段>.<并发>` 里是各阶段时间的中位数（毫秒）：`preprocessing.service`、`image_encoder.service`、`thinker.queue`、`thinker.prefill`、`thinker.service`、`decode.service`、阶段之间的传递 `hop.*`、服务端从接收到完成 `request.admission_to_last_terminal`，以及各阶段占总时间的比例 `share.*`（阶段中位数 ÷ 总时间中位数）、`requests` |
| `events_run_levels` | 事件补测运行本身的每档数字，字段同 `runs[]` |
| `scaling.<段>` | `best_rps`、`best_little_rps`（各副本数下各档的最大值）、`efficiency`、`efficiency_little`（扩展效率）、`gpu_busy_at_best`（吞吐最高那一档的 GPU 忙碌比例）、`gpu_busy_phase`（整段平均） |
| `verdicts` | `linear`（首尾时间算的效率）、`linear_little`（对照）、`gpu_flat`（整段平均，主要）、`gpu_flat_best_level`（最初的指标）、`linear_overall`、`gpu_flat_overall` |

## 五、实验 13（`exp13_report.py`）

### `summary.json`

| 字段 | 含义 |
|---|---|
| `rule.tolerance` | 判定用的 5%（取自预测 G-H1） |
| `topology.links` | 每对 GPU 的连接类型（nvidia-smi topo：`NV4`、`PIX`、`NODE`、`SYS`），键是“GPU-GPU” |
| `topology.pairs[]` | 每对 GPU：`pair`、`link`、`GBps`（两个方向的点对点拷贝带宽）、`peer_access` |
| `topology.bandwidth_by_link` | 按连接类型：`pairs`（几对）、`median_GBps`、`min_GBps`、`max_GBps`（两个方向都算） |
| `topology.allreduce[]` | NCCL all-reduce：`link`、`devices`、`MiB`、`algbw_GBps`、`busbw_GBps`、`p2p_disabled` |
| `runs[].phases.<段>.<服务>` | 每个服务：`gpus`（thinker 卡、talker 卡）、`link`、`requests`、`hop_events`、`hop_thinker_talker_p50_ms`、`hop_thinker_talker_p90_ms`（thinker 交给 talker）、`hop_talker_code2wav_p50_ms`（同一张卡上 talker 交给 code2wav）、`levels.<档>`（`n`、`ok`、`e2e_p50_s`、`e2e_p90_s`、`slo`） |
| `verdicts.<段>` | `verdict`、`hop_by_link_p50_ms`（每种连接的传递中位数）、`hop_spread_ms`、`hop_spread_share_of_e2e`、`e2e_top_level_p50_s`（每个服务最高一档的端到端中位数，键是“服务#组”）、`e2e_rel_non_nvlink`、`same_link_spread`、`e2e_median_s` |
| `verdict_overall` | 两段一致时取该结论，否则“不确定” |

服务属于哪个请求，按服务端事件里的请求编号认领（请求记录里只有服务的地址）。

## 六、实验 3（`exp03_report.py`）

### `summary.json`

| 字段 | 含义 |
|---|---|
| `rule.slo_fraction` | 0.95：一档里至少这么多请求满足 SLO 才算合格 |
| `curves.<分法>` | 闭环曲线：`label`、`runs[]`（`run`、`phase`、`node`）、`levels[]`、`doubling_gain`、`highest_concurrency_within_slo` |
| `curves.<分法>.levels[]` | 每个并发：`concurrency`、`n`、`errors`、`rps`、`slo`、`ttft_p50_s`、`ttft_p90_s`、`e2e_p50_s`、`e2e_p90_s`、`e2e_mean_s`、`e2e_per_audio_p90`、`in_flight_little`（吞吐 × 平均端到端时间）、`stages_ms`、`split`、`eff_batch`、`gpu_util`、`run`、`node` |
| `levels[].stages_ms` | 各阶段时间的中位数（毫秒，服务端事件）：`thinker.queue`、`thinker.model_path`（prefill 加全部解码步）、`talker.queue`、`talker.model_path`、`code2wav.service`（含它内部的排队） |
| `levels[].split` | 端到端中位数里各阶段的占比：`thinker_share`、`talker_share`、`code2wav_share`，以及剩下的 `other_ms`（HTTP、预处理、阶段之间的传递；各阶段中位数相加不等于端到端中位数，所以可能是负数） |
| `levels[].eff_batch`、`gpu_util` | 见 `scripts/analysis/stage_by_level.py`：每个副本平均同时在算的请求数；各角色 GPU 的 nvidia-smi 采样平均 |
| `doubling_gain.<c>` | 并发从 c/2 加到 c 时吞吐涨了多少（比例） |
| `open.<分法>` | 开环：`run`、`levels[]`（`rate_rps`、`n`、`ok`、`errors`、`rps`、`slo`、`ttft_p90_s`、`e2e_p50_s`、`e2e_p90_s`、`e2e_mean_s`、`e2e_per_audio_p90`）、`capacity`（SLO 容量）、`censored`（最高一档也合格时为 true，容量只是下限）、`in_flight_at_capacity`（容量那一档的吞吐 × 平均端到端时间） |
| `verdicts.A-H1_second_half` | `verdict`、`r8_near_best`（每个并发下 8 个完整副本是否在最好的 3% 以内）、`tolerance` |
