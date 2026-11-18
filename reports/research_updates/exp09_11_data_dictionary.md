# 实验 9–11 数据字典

这份文档说明实验 9–11 的每个数据文件里有哪些字段，以及每个字段的单位和含义。

- 字段名和代码里一模一样，可以直接搜。
- 分析脚本在 `scripts/analysis/`，每个脚本开头的说明写了怎样运行。

## 一、通用约定

| 约定 | 含义 |
|---|---|
| 单位后缀 | `_s` 秒，`_ms` 毫秒，`_ns` 纳秒，`_mib` MiB；没有后缀的比例是 0–1 的小数 |
| `rps` | 每秒完成的请求数 |
| `audio_s_per_s` | 每秒完成的语音秒数（完成请求的语音时长之和 ÷ 时间窗） |
| 客户端时钟 | 请求记录里的 `*_ns` 是发起请求那台机器的单调时钟（CLOCK_MONOTONIC），只能互相相减 |
| 服务端时钟 | 服务端事件的 `timestamp_ns` 是墙上时钟；thinker 的 `model_path_*` 事件另在 `metadata.monotonic_ns` 记单调时钟，用来对齐两种时钟（实验 11） |
| GPU 采样时钟 | `gpu_samples.csv` 的 `timestamp` 是墙上时钟，每 0.2 秒一行 |
| 区组（block） | 一组按随机顺序跑完的运行；比较都在同一区组内配对 |
| 组（arm） | 被比较的设置：实验 9 是方案 1–4；实验 10 是 graph 开或关 × 组批设置；实验 11 是 A、B、C |
| 段（phase） | 一次运行里的一段负载，例如 `w3b_closed_A` |
| 档（level） | 段里的一个并发数（闭环）或到达速率（开环） |
| 预热请求 | `warmup: true` 的请求不进任何统计 |
| 区间结构 | `{"runs": [每个区组的值], "n": 有效个数, "mean": 均值, "ci95": [下界, 上界]}`，t 分布 95% 区间，自由度 n−1 |
| 失败按无穷大 | “截尾百分位”（censored）把失败和超时的请求当作无穷长；JSON 里写成 `Infinity`，CSV 里写成 `inf`。汇总成区间时，无穷大的值不进均值，另记在 `infinite_runs` |

## 二、原始数据（不公开）

位置：`PROJECT_DATA_ROOT/experiments/<日期>_<experiment_id>_run01/`。

原始数据含机器名、路径和 GPU 序列号，只放在计算集群上，不进公开仓库。

| 文件 | 内容 |
|---|---|
| `result.json` | `experiment_id`、`status`、`wall_s`、`gpu`、`server_ready_s`、`server_exit_codes`、`server_restarts`（每项有 `server`、`exit_code`、`where`，`where` 的最后一个词是出事的段）、`phases`（每段有 `phase`、`kind`、`levels`、`started`、`ended`、`stopped_reason`） |
| `raw/requests_<段>.jsonl` | 每行一个请求，字段见下表 |
| `raw/server_events/<段>/s0/events_<阶段>@r<副本>_<进程号>.jsonl` | 服务端事件，只有打点的段才有。每行有 `request_id`、`stage`、`event_name`、`timestamp_ns`、`run_id`、`pid`、`metadata` |
| `raw/gpu_samples.csv` | nvidia-smi 采样：`timestamp`、`index`、`uuid`、`utilization.gpu`（%）、`utilization.memory`（%）、`memory.used`（MiB）、`memory.total`（MiB）、`power.draw`（W）、`clocks.sm`（MHz）、`pstate` |
| `raw/cpu_samples.jsonl` | 每秒一行：`t_wall_ns`、`system_cpu_percent`，以及按进程组的 CPU 占用 |
| `profiling/torch/<段>/*.trace.json(.gz)` | torch profiler 的 Chrome trace（实验 11 的 C 段），时间单位微秒 |
| `c2w_control.json` | 实验 10：code2wav 组批控制文件最后一次写入的内容 |
| `step.log`、`logs/` | 运行过程的文字记录和服务日志 |

### 请求记录（`requests_<段>.jsonl`）的字段

| 字段 | 含义 |
|---|---|
| `request_id` | `<段>-c<并发>-<序号>-<随机串>`（闭环）或 `<段>-r<速率>-<序号>-<随机串>`（开环）；服务端事件用同一个 ID |
| `status` | `ok`、`error`、`timeout`、`cancelled` |
| `error`、`http_status` | 失败原因和 HTTP 状态码 |
| `sent_ns` | 发出请求 |
| `headers_ns` | 收到响应头 |
| `first_text_ns` | 收到第一段文字（TTFT 的终点） |
| `first_audio_ns`、`last_audio_ns` | 收到第一段、最后一段语音 |
| `done_ns` | 请求结束（端到端延迟的终点） |
| `completion_tokens`、`prompt_tokens` | 输出、输入的 token 数 |
| `finish_reason` | `stop`（自然结束）或 `length`（到了 `max_tokens`） |
| `output_audio_s`、`audio_sample_rate` | 返回语音的总时长（秒）和采样率 |
| `text_chars`、`text_chunks`、`audio_chunks` | 文字字数、文字分段数、每段语音的（到达时间，时长，字节数） |
| `text` | 回答文字，只在需要核对正确性的段里保存 |
| `phase`、`index`、`worker`、`workload_id` | 段、序号、闭环里的发送者编号、负载里的提示编号 |
| `concurrency` | 闭环的并发数 |
| `rate_rps`、`scheduled_ns`、`lateness_s` | 开环的到达速率、计划发送时间、实际比计划晚了多少秒 |
| `warmup` | 是否预热请求 |
| `max_tokens`、`input_modalities`、`output_modalities`、`payload` | 请求参数 |

## 三、实验 9 的结果文件

由 `exp09_report.py` 生成，放在 `research/01_性能瓶颈/实验09_最佳GPU分工/results/`。

### `summary.json`

- `runs[]`：每个运行一项。
  - `run`、`placement`（如 `t2_k1_c5`）、`arm`（方案 1–4）、`block`、`restarts`
  - `workload.w3b` / `workload.w3l`：
    - `capacity_rps`：登记的容量，即最高闭环并发 A 段的完成速率 = 完成数 ÷（最后完成 − 最先发出）；
    - `audio_s_per_s`；
    - `capacity_from`：取自 A 段还是 B 段；
    - `bottleneck`：B 段驻留时间最长的阶段；
    - `excluded`：按登记的规则没有用上的值和原因（这一档错误 + 超时 > 5%，或服务在这一段重启），这时对应的容量或瓶颈记为空。
  - `levels[]`：每段每档一行，字段见下表。
- `by_arm["<方案>"]["<负载>|<类型>|<A或B>|<档>"]`：
  - 三个区组汇总成区间的指标：`throughput_rps`、`busy_window_rps`、`audio_s_per_s`、`e2e_p50_s`、`e2e_p95_s`、`e2e_p99_s`、`ttft_p50_s`、`first_audio_p50_s`、`growth_slope_rps`、`tokens_mean`、`audio_s_mean`；
  - `steady_runs`：每个区组是否稳态；
  - `unusable_runs`、`after_restart_runs`：每个区组这一档是否不可用、是否在重启之后。
  - 不可用的闭环档位不进区间（区间的 `runs` 里记为空），原始值仍在 `runs[]` 和 `levels.csv` 里。开环档位不排除（登记：超时按无穷大计入）。
- `verdicts`：
  - `H9a`、`H9b`、`H9b_w3l`、`H9c`：每项有 `verdict`（支持 / 反驳 / 不确定 / 数据不足），以及判定用到的每个区组的数字；
  - `H9d`、`H9d_w3l`：留一法的容量模型，`predictions` 是每个方案的预测值和实测值。
  - 判定要求 3 个区组的值都在；任何一个区组缺失（不可用、重启、运行失败），这条判定就是“数据不足”。
- `sensitivity`：补充登记一的敏感度分析（描述性，看过第 1 个运行后才加），把容量换成另一种算法，再算一遍 H9a、H9c、H9d、H9d_w3l：
  - `busy_window`：最高并发档的“满额时段吞吐”；
  - `stable_open_loop`：这次运行开环里仍是稳态的最高共同速率。

### `levels[]` 每一行

| 字段 | 含义 |
|---|---|
| `workload`、`phase`、`kind`、`mode` | 负载（`w3b`、`w3l`）、段、`closed_loop` 或 `open_loop`、`A`（不打点）或 `B`（打点） |
| `concurrency` / `rate_rps` | 档 |
| `client` | 客户端统计：`measured`、`ok`、`errors`、`timeouts`、`throughput_rps`、`audio_s_per_s`、`e2e_s`、`ttft_s`、`first_audio_s`（各含 n、均值、百分位）、`completion_tokens_mean`、`audio_s_mean` |
| `censored` | 截尾百分位：`n`、`failed`、`p50`、`p95`、`p99`（秒，失败按无穷大） |
| `busy_window` | 描述性指标（补充登记一）：`throughput_rps` = 在（第一个测量请求发出，最后一个测量请求发出]之间完成的请求数 ÷ 这段时长；`window_s` |
| `output` | `tokens_mean`、`tokens_p95`、`audio_s_mean`、`audio_s_p95` |
| `lateness_p99_s` | 开环：发送比计划晚的 P99，用来确认客户端没有拖后腿 |
| `gpu["<角色>"]` | 这一档时间窗内按角色（thinker、talker、code2wav，或同卡的 talker+code2wav）平均：`busy`（nvidia-smi 的忙碌比例）、`mem_max_mib`、`sm_clock_mhz`、`power_w` |
| `growth` | 开环：`slope_rps` = 发送窗口后半段“已发出未完成”请求数的最小二乘斜率；`steady` = 斜率 ≤ 0.05 × 到达速率；`inflight_end`；`window_s` |
| `inflight_series` | 开环：每秒一个点 [距第一次发送的秒数, 已发出未完成的请求数]，到最后一个请求完成为止 |
| `residence_ms` | B 段：每个阶段所有请求驻留时间之和（从 `stage_input_received` 到 `stage_complete`） |
| `in_stage_mean` | B 段：每个阶段平均同时有几个请求（驻留时间之和 ÷ 时间窗，即利特尔法则） |
| `bottleneck` | B 段：驻留时间之和最大的阶段 |
| `code2wav` | B 段：`batch_size`（每批大小的统计）、`queue_ms`（在 code2wav 收件队列里等待的毫秒数） |
| `talker_gen_ms_per_audio_s`、`talker_queue_ms` | B 段：talker 每生成 1 秒语音用的毫秒数；在 talker 前排队的毫秒数 |
| `usable` | 错误 + 超时 ≤ 测量请求的 5%。只对闭环档位起作用 |
| `restart_here`、`after_restart` | 服务是否在这一段重启；这一段是否在某次重启之后 |

### `levels.csv`

每行是一个运行的一段一档，列为：

- `arm`、`placement`、`block`、`workload`、`kind`、`mode`、`level`；
- `throughput_rps`、`busy_window_rps`、`audio_s_per_s`；
- `e2e_p50_s`、`e2e_p95_s`、`e2e_p99_s`、`failed`、`ttft_p50_s`；
- `growth_slope_rps`、`steady`；
- `tokens_mean`、`audio_s_mean`、`bottleneck`；
- `usable`、`restart_here`、`after_restart`。

## 四、实验 10 的结果文件

由 `exp10_report.py` 生成，放在 `research/01_性能瓶颈/实验10_Code2Wav_Batching/results/summary.json`。

- `runs[]`：
  - `run`、`graph`（flow CUDA graph 是否打开）、`block`、`restarts`；
  - `cells["<负载>|<设置>"]`：负载是 `w3b` 或 `mix`，设置是 `fifo1`…`fifo16`、`length8`、`length16`。每个格子的字段见下表。
- `by_setting["graph|<负载>|<设置>"]` 或 `by_setting["nograph|<负载>|<设置>"]`：
  - 三个区组的区间：`audio_s_per_s`、`gpu_ms_per_audio_s`、`padding`、`graph_share`、`wait_long_p95`、`wait_long_max`、`batch_size_mean`、`throughput_rps`、`e2e_p50_s`、`e2e_p95_s`；
  - 以及 `control_ok`、`usable`、`audio_mismatch`、`blocks`（每个区组一个值）。
  - 无效的格子（不可用，或控制文件没有生效）不进区间，原始值仍在 `runs[]` 里。
- `verdicts`：H10a–H10e，每项有 `verdict` 和每个区组的数字。
  - 正确性规则没通过时，H10c、H10d 写“停止（正确性规则未通过）”；
  - 没有正确性检查运行时，写“数据不足（缺正确性检查）”。
- `correctness`：
  - `audio_codec_mismatch`：所有性能运行里，语音秒数和编码数对不上（差超过 1 个编码）的请求数；
  - `waveform_length_mismatch`：正确性检查里，同一编码却得到不同样本数的请求数；
  - `ok`：两者都为 0；没有检查运行时为空。
- `validation`：
  - `candidates`：G1 上 W3b 或 MIX 每个区组都比 fifo16 多 ≥ 5% 语音秒/秒的设置，以及每个区组的增幅；
  - `decision`：是否在实验 9 最好的放置上做完整服务验证；
  - `verdict`（只有做了验证才有）：按补充登记一的判定，`rows` 是每个区组在 21 和 23.5 请求/秒两档的 P50 降幅（`p50_reduction`，失败按无穷大处理）和完成请求/秒之比（`throughput_ratio`）。
- `validation_runs[]`（只有做了验证才有）：每次验证运行的 `placement`、`setting`、`block`、`restarts`；
  - `phases["fifo16" | "<候选设置>"]["<速率>"]`：`e2e_p50`、`e2e_p95`、`e2e_p99`（截尾，秒）、`failed`、`throughput_rps`、`audio_s_per_s`、`steady`、`growth_slope_rps`；
  - `control`：每段逐批核对控制文件是否生效。
- `flags`：`unusable`、`control_invalid`、`after_restart`。
- `check`：正确性检查运行。
  - `requests_with_digest`：每段带摘要的请求数；
  - `vs_fifo1["<段>"]`：和批上限 1 的同一请求比较，字段 `pairs`、`sample_mismatch`（样本数不同的请求数）、`rel_diff_max`、`rel_diff_p95`、`rms_rel_change_max`。

### 每个格子（一种负载 × 一种组批设置）

| 字段 | 含义 |
|---|---|
| `client` | 同实验 9 的 `client` |
| `batches` | 这一段（大多数请求属于这一段的）批数 |
| `batch_size` | `mean`、`p50`、`at_cap`（等于批上限的批占多少） |
| `gpu_ms_per_audio_s` | flow + HiFT 的 CUDA event 跨度之和 ÷ 有效语音秒数。越低越省；同卡有别的进程时，跨度里包括等待 |
| `padding` | 补齐比例 = 1 − 真实帧数之和 ÷（行数 × 最长一行的帧数），按帧加权 |
| `graph_share` | flow 走 CUDA graph（`dense_graph` 或 `packed_graph`）的批占多少 |
| `control` | 逐批核对实际生效的批上限和策略：`batches`、`mismatched`、`ok` |
| `correspondence` | 每批的请求 ID 和语音长度是否和客户端收到的对得上：`checked`、`mismatched` |
| `wait` | 按语音长度的中位数分成短、长两类（`split_s`），每类在收件队列里等待的 `n`、`p50`、`p95`、`max`（毫秒） |
| `queue_ms` | 所有请求在收件队列里的等待（毫秒） |
| `audio_s_per_s`、`wait_long_p95`、`wait_long_max` | 从上面提出来的常用值 |
| `usable`、`after_restart` | 错误 + 超时 ≤ 5%；是否在某次服务重启之后 |

## 五、实验 11 的结果文件

放在 `research/01_性能瓶颈/实验11_Thinker异步优化/results/`。

> **注意，两套 A、B、C 是两回事。**
> - **组** A、B、C 是服务的设置：同步执行、提前发射 + 快照、提前发射 + 快照 + 旁路复制。它们由 `SGLANG_OMNI_SPEECH_LOOKAHEAD` 设为 `0`、`1`、`sidecopy`，每组是单独启动的服务。
> - **段** A、B、C 是同一个服务里的三种测量方式：不打点、打事件、录 thinker 的 trace。
> - 在 `performance.json` 里，`runs[].arm` 是组，`levels[...]["A" | "B" | "C"]` 是段。
> - 图和 README 里把段叫作“不打点段、事件段、trace 段”。

### `check.json`（`exp11_check.py`，正确性检查）

- `runs`：A、C、A2 各是哪个运行。
- `gate`：
  - `pass`：是否通过；`reasons`：没通过的原因；
  - `c1_exact`：并发 1 时 C 和 A 逐个请求比较。`pairs` 是比较的请求数，`text`、`hidden`、`codec` 分别是文字、隐藏状态哈希、语音编码哈希完全相同的个数。
  - `phases["<段>"]`：
    - `errors`；
    - `rows`：隐藏状态行数是否等于文字 token 数，有 `checked`、`mismatched`；
    - `C_vs_A`、`A2_vs_A`：输出 token 完全相同的请求里，比较隐藏状态指纹（每行的范数和 4 个随机投影）。字段有 `pairs`、`over_tolerance`（相对偏差超过 0.05 的个数）、`share_over`、`max_deviation`。
- `lengths`：长回答段的 token 数和语音时长，以及 `codec_at_talker_cap`（语音到了 talker 上限 2048 个编码的请求数）。
- `errors`：每组每段的失败数。

### `performance.json`（`exp11_report.py`，性能对比）

- `runs[]`：
  - `run`、`block`、`arm`（A、B、C）、`restarts`；
  - `levels["<负载>"]["<并发>"]["A" | "B" | "C"]`：每段一个格子，字段见下表。
- `verdicts`：
  - H11a–H11e，每项有 `verdict` 和每个区组的数字；
  - H11a 的 `parts` 是四个比较各自的结果；
  - H11d 分 `C_vs_B`、`C_vs_A`。
- `recommendation`：按登记的规则：
  - `default_on`、`advice`；
  - `h11d_C_vs_A`；
  - `ttft_increase_ms`：每档 C 减 A 的 TTFT P50 区间；
  - `ttft_increase_below_10ms`。
- `table["<负载>|<并发>|<段>|<指标>|<组>-<基准>"]`：两组各自的区间和配对差 `diff`。
- `flags`：`unusable`（错误 + 超时 > 5% 的段）、`after_restart`（服务重启所在及之后的段）。

### 格子的字段

| 段 | 字段 | 含义 |
|---|---|---|
| 全部 | `client`、`usable`、`after_restart` | 客户端统计；这段是否可用；是否在重启之后 |
| A（不打点） | `e2e_p50`、`e2e_p95`、`e2e_p99`、`failed` | 截尾的端到端延迟（秒） |
| A | `throughput_rps`、`ttft_p50_ms`、`ttft_p95_ms` | 吞吐；客户端 TTFT |
| A | `thinker_mem_max_mib` | thinker 两张卡在这段时间里的最大显存 |
| B（打事件） | `steps` | thinker 每轮的统计：`iterations`、`lookahead_share`、`decode_lookahead_share`、`mode_share`、`flush_reasons`、`blocked_by`、`batch_size_decode`、`cpu_ms` |
| B | `lookahead_share`、`mixed_share` | 提前发射的轮次比例；MIXED（解码和 prefill 混在一起）的轮次比例 |
| B | `ms_per_token` | 每个请求在 thinker 里平均每个 token 用的毫秒数（prefill 结束到 thinker 结束 ÷（token 数 − 1））的中位数 |
| B | `tokens_mean` | 平均输出 token 数 |
| B | `ttft_mean_ms`、`queue_wait_mean_ms`、`ttft_segments_mean_ms` | TTFT 和它的 9 段拆分（均值，各段之和等于 TTFT），见下表 |
| B | `clock` | 两种时钟的对齐：`offset_ns`、`n`、`spread_ns`、`p05_p95_ns` |
| C（thinker 的 trace） | `procs[]` | 每个 thinker 进程一项（trace 时间单位微秒）：`window`、`index_jumps`（步号不连续的次数）、`gap_p50`、`gap_p95`、`gap_p99`、`between_step_idle_share`、`total_idle_share`、`by_type`（按“模式.发射方式”分的空档）、`graph_steps`、`non_graph_steps`、`n_steps`、`blocking` |
| C | `blocking` | 超过 1 毫秒的阻塞调用：`sync`（同步调用）、`small`（≤ 1 KB 的拷贝）、`large`（> 1 KB 的拷贝）、`unknown`。每类有 `n`、`inside`（落在 `thinker.request_finished_copy` 标注里的个数）、`share_inside` |
| C | `steps`、`large_copies`、`large_inside`、`large_copies_per_1k_steps` | 所有进程合计 |
| C | `between_idle_share` | 按时间加权的空档比例 = 每步之前的空档之和 ÷ 时间窗（两个进程的平均）。H11c 用这个 |
| C | `total_idle_share` | GPU 完全没有工作的时间 ÷ 时间窗 |
| C | `gap_p50_ms`、`gap_p95_ms`、`gap_p99_ms` | 每步之前的空档百分位（两个进程的中位数） |
| C | `decode_busy_ms_p50` | DECODE 步的 GPU 忙碌时间（这一步 kernel 区间的并集）的中位数 |

### TTFT 的 9 段（`exp11_ttft.py`）

| 段 | 从 | 到 |
|---|---|---|
| `send_to_admission` | 客户端发出 | coordinator 接纳请求 |
| `admission_to_thinker` | 接纳 | thinker 收到输入 |
| `thinker_in_to_queue` | thinker 收到 | 进入调度队列 |
| `queue_wait` | 进入调度队列 | prefill 开始 |
| `prefill` | prefill 开始 | prefill 结束 |
| `prefill_to_first_emit` | prefill 结束 | 调度器第一次产出 token |
| `first_emit_to_thinker_chunk` | 第一次产出 | thinker 发出第一段文字 |
| `thinker_chunk_to_decode_chunk` | thinker 发出 | decode 阶段发出第一段文字 |
| `decode_chunk_to_client` | decode 发出 | 客户端收到第一段文字 |

### `ttft.json`、`ttft_exp08data.json`（`exp11_ttft.py`）

- `runs[].levels["<并发>"]`：
  - `clock`、`requests`；
  - `complete`：9 段都有的请求数；
  - `median_ms`、`mean_ms`：每段和 `ttft` 的中位数、均值。
- `contrasts["<组>"]["<并发>"]["<段>"]`：基准组和这一组的区间，以及配对差 `diff`。用的是每个区组的中位数。
