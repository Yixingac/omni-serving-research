# 指标定义

代码：`src/omni_research/analysis/metrics.py`、`intervals.py`、`experiment_analysis.py`、`nsys.py`。所有时间都在客户端用 `time.monotonic_ns()` 记录，除非特别说明。

## 单个请求

| 指标 | 定义 |
|---|---|
| `e2e_s`（端到端延迟） | 客户端发出请求到收到最后一个数据块（SSE `[DONE]` 之前的最后一条）的时间 |
| `ttft_s`（首个文本 token 时间） | 发出请求到收到第一个非空文本增量 |
| `ttfa_s`（首个音频块时间） | 发出请求到收到第一个音频块。每个音频块是一个完整的 WAV，时长从 WAV 头计算 |
| `audio_ready_s`（音频就绪时间） | 发出请求到收到最后一个音频块 |
| `output_audio_s` | 所有音频块时长之和 |
| `audio_rtf`（语音生成的实时率） | (最后一个音频块到达时间 − 第一个音频块到达时间) / 音频总时长。小于 1 表示生成比播放快 |
| `completion_tokens` | 服务端在最后一个数据块里返回的 usage |
| 播放卡顿 | 模拟一个收到第一个音频块就开始播放的播放器：后面的块晚于缓冲耗尽时记一次卡顿，累计卡顿时长；可以设预缓冲时长 |

## 一组请求

| 指标 | 定义 |
|---|---|
| 样本量 | 每组都报告请求总数、成功数、失败数和 warmup 数。warmup 不计入统计 |
| 中位数、均值、标准差 | 只对成功的请求计算 |
| 分位数 | 只有当至少 5 个样本落在分位点之上时才报告：p50 需要 10 个样本，p90 需要 50 个，p95 需要 100 个，p99 需要 500 个。样本不够时该格写“—”，并注明 n |
| 置信区间 | 均值和中位数的 95% bootstrap 区间（2000 次重采样，固定种子），样本少于 3 个时不报告 |
| 吞吐 | 在成功请求的总跨度（第一个请求发出到最后一个请求完成）内：请求数/秒、生成 token 数/秒、生成音频秒数/秒 |
| Goodput | 同时满足所有 SLO（例如 `ttft_s ≤ 1`、`ttfa_s ≤ 2`）的请求数，按比例和按每秒请求数两种方式报告 |

## Agent 会话

| 指标 | 定义 |
|---|---|
| 任务成功率 | 最终答案中包含正确数字（相对误差 1% 以内）的会话比例 |
| 每个会话的请求数、工具调用数 | 直接计数 |
| 会话时间拆分 | 在每个会话的时间窗内取区间并集：等模型的时间、工具时间（扣除与模型请求重叠的部分）、用户思考时间、其他（Agent 自身的处理和间隙）。四部分加起来等于会话总时长 |

## 服务端阶段（SGLang-Omni request event recorder）

来自服务端事件，单位毫秒，按 request_id 匹配。对数据里出现的**每一个阶段**（preprocessing、thinker、decode、talker、code2wav 等，名字取自事件本身）计算下面这些区间；只报告数据中真实存在的事件对，不会给模型没有的阶段补数字。

| 区间 | 起点事件 → 终点事件 | 含义 |
|---|---|---|
| `<阶段>.queue` | `scheduler_queue_enter` → `scheduler_prefill_start` | 在该阶段调度器里排队 |
| `<阶段>.prefill` | `scheduler_prefill_start` → `scheduler_prefill_end` | 该阶段的 prefill |
| `<阶段>.first_emit` | `scheduler_prefill_start` → `scheduler_first_emit` | 到第一个输出 |
| `<阶段>.model_path` | `model_path_start` → `model_path_end` | 该阶段模型执行的全过程（含逐步解码） |
| `<阶段>.service` | `stage_dispatch` → `stage_complete` | 整个阶段，包括阶段内部的等待 |
| `hop.<来源>-><目标>` | 来源阶段最近一次 `stage_hop_sent` → 目标阶段 `stage_input_received` | 阶段之间的数据传递 |
| `request.admission_to_first_terminal` | `request_admission` → 第一个 `terminal_response` | 例如文本结束 |
| `request.admission_to_last_terminal` | `request_admission` → 最后一个 `terminal_response` | 例如语音结束 |

例：Spark 上一个 MiniCPM-o 语音请求的事件显示，thinker 生成完整段文本（`thinker.model_path` 约 3.7 秒）后，talker 才收到输入（`hop.thinker->talker` 约 4 毫秒），talker 执行约 1.15 秒，code2wav 约 0.19 秒。三个阶段对同一请求是串行的。

`service` 一类数字表示“开始执行到阶段完成”，包含等待其他请求的时间，不等于纯 GPU 计算时间。

### 关于“首个音频块时间”

在我们固定版本的 MiniCPM-o 4.5 上，`/v1/chat/completions` 的流式响应里文本是逐块到达的，但音频在请求结束时**整段一次性**到达（每个请求一个音频块；Spark 上观察到，待 A6000 复核）。因此对这个模型，`ttfa_s`（首个音频块时间）等于“整段语音生成完”的时间，`audio_rtf` 无法计算（只有一个块）。比较不同模型或版本时要先确认音频是否真的流式下发。

## GPU 使用：几个不能混用的数字

| 名称 | 含义 | 来源 |
|---|---|---|
| **采样利用率**（sampled utilization） | `nvidia-smi` 的 `utilization.gpu`：采样周期内至少有一个 kernel 在运行的时间比例。它**不是** SM 占用率。一个很小的 kernel 一直在跑，也会显示 100% | `raw/gpu_samples.csv`，每 200 ms 一次 |
| **GPU 活动忙时**（busy time） | 一段时间里 kernel 和拷贝区间的**并集**长度。不同 stream 上重叠的 kernel 不重复计算 | Nsight Systems 导出的 SQLite，`analysis/nsys.py` |
| kernel 时长之和 | 所有 kernel 时长直接相加。kernel 重叠时会大于 busy time，只用来展示重叠程度，不作为利用率 | 同上 |
| SM 利用率 | SM 上 warp 的活跃程度 | Nsight Compute 或 nsys 的 GPU metrics 采样，需要单独开启 |
| 请求墙钟时间、排队时间 | 来自客户端和服务端事件，与 GPU 是否忙无关 | 见上 |
| CPU 时间 | 服务进程树的 CPU 使用率（psutil，每秒一次） | `raw/cpu_samples.jsonl` |

GPU 空闲区间：busy time 的补集。按长度分桶报告（<10 µs、10–100 µs、0.1–1 ms、1–10 ms、10–100 ms、≥100 ms），再结合同一时刻的请求状态解释空闲来源，例如所有会话都在等工具、CPU 预处理还没完成、调度间隙。
