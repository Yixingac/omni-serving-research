# Agent trace 格式（`omni_research.agent_trace/v1`）

代码：`src/omni_research/tracing/`（`schema.py` 定义格式，`collector.py` 写，`io.py` 读，`validate.py` 校验）。

## 文件结构

一个 trace 是一个 JSONL 文件（每行一个 JSON 对象）：

1. 第一行是 `trace_header`；
2. 之后是任意顺序的 `span`（一段有起止时间的活动）和 `event`（某个时刻发生的事）；
3. 正常关闭时最后一行是 `trace_footer`，记录写入的 span 和 event 数量，用来发现截断。

span 在**结束时**写入一次，带最终状态。如果程序在 span 结束前退出，`close()` 会把仍然打开的 span 以 `cancelled` 状态、`attributes.incomplete = true` 写出，所以已经开始的活动不会无声消失。文件只追加写，并定期 `fsync`。

## 时钟

所有 span 和 event 的时间都是整数纳秒，属于 header 里声明的时钟域：

```json
"clock": {"domain": "monotonic_ns", "host": "...", "pid": 123,
          "mono_anchor_ns": 651458000000000, "wall_anchor_ns": 1762889000000000000,
          "anchor_uncertainty_ns": 40, "alignment": "single-host"}
```

- `monotonic_ns` 是写 trace 的进程里的 `time.monotonic_ns()`。它不受系统时间调整影响，适合计算间隔。
- 锚点是连续读取的一对（monotonic，墙钟）值，取几次读取中间隔最短的一次。换算公式：`wall = wall_anchor_ns + (t - mono_anchor_ns)`。SGLang-Omni 的服务端事件用 `time.time_ns()`（墙钟），客户端和服务端在同一台机器上时，可以用这个公式对齐。
- 不同机器的时钟不能假设同步。当前所有实验的客户端和服务端都在同一节点上（`alignment: single-host`）。

## 记录类型

### trace_header

| 字段 | 含义 |
|---|---|
| `trace_id` | 全局唯一 id |
| `schema` | `omni_research.agent_trace/v1` |
| `clock` | 见上 |
| `source.kind` | `observed`（真实 Agent 对真实服务运行时记录）/ `synthetic`（生成器产生，没有真实执行）/ `replay`（回放另一个 trace 时记录） |
| `source.generator`、`source.seed` | 产生这个 trace 的程序和随机种子。合成 trace 必须填写 |
| `content_policy` | `none`（只记大小和时长，默认）/ `hashed`（加内容摘要）/ `public`（可以保存内容，只用于合成或公开输入） |
| `workload`、`notes` | 负载参数和说明，例如哪些部分是合成的 |

### span

| 字段 | 含义 |
|---|---|
| `span_id`、`parent_span_id` | span 自己的 id，以及父 span（例如模型请求的父 span 是某个 agent_step） |
| `kind` | `session`、`interaction`、`agent_step`、`model_request`、`tool_call`、`wait` |
| `session_id`、`interaction_id`、`step_id`、`request_id` | 关联 id。子 span 默认继承父 span 的 session/interaction/step id |
| `start_ns`、`end_ns` | 起止时间（header 的时钟域） |
| `depends_on` | 必须先结束的 span 列表，构成依赖 DAG。例如第二次模型请求依赖工具调用 |
| `status` | `ok`、`error`、`cancelled`、`timeout` |
| `attributes` | 各类型的附加信息，见下表 |

各类型必须有的 id 和属性（校验器会检查）：

| kind | 必须的 id | 必须的属性 | 常用的其他属性 |
|---|---|---|---|
| `session` | session_id | — | `task_kind`、`success`、`steps`、`tool_calls` |
| `interaction` | session_id、interaction_id | — | 一次用户回合 |
| `agent_step` | session_id、step_id | — | `parse_error` |
| `model_request` | session_id、request_id | `input_modalities`、`output_modalities`、`endpoint` | `max_tokens`、`input_text_chars`、`audio_input_s`、`completion_tokens`、`output_audio_s`、`ttft_s`；`content_policy=public` 时还有 `request_body` |
| `tool_call` | session_id | `tool_name` | `tool_status`、`local_exec_s`、`simulated_latency_s`（模拟的远程延迟，`synthetic_latency=true`） |
| `wait` | session_id | `reason` | 例如 `user_think`（合成的用户思考时间，带分布参数） |

### event

`model_request` 里记录流式输出的里程碑：`request_sent`、`first_text_token`、`first_audio_chunk`、`response_complete`，以及 `error`、`cancel`。时间来自客户端收到 SSE 数据块的时刻。

## 校验规则（`validate_trace`）

错误（trace 不能用于分析或回放）：
- header 的 schema、时钟域、source.kind 不合法；合成 trace 没写 generator；
- span_id 重复，kind 或 status 未知，缺少必须的 id 或属性，没有结束时间或结束早于开始；
- 父 span 不存在，子 span 开始早于父 span，子 span 和父 span 的 session_id 不同；
- `depends_on` 指向不存在的 span，或者依赖关系有环；
- 对 `observed` 的 trace：一个 span 在它依赖的 span 结束之前就开始了（违反因果，允许 1 ms 误差）；
- 模态名称不在 `text/audio/image/video` 之内；footer 计数与实际读到的记录数不一致。

警告（需要知道，但不妨碍使用）：子 span 比父 span 结束得晚（通常是被取消得晚），event 落在 span 区间之外，没有 footer。

## 一个最小例子

```jsonl
{"record_type":"trace_header","trace_id":"tr-1","schema":"omni_research.agent_trace/v1","clock":{"domain":"monotonic_ns","host":"node","pid":1,"mono_anchor_ns":0,"wall_anchor_ns":1762889000000000000,"anchor_uncertainty_ns":50,"alignment":"single-host"},"source":{"kind":"observed","generator":"omni_research.tracing.agent_harness","seed":2},"content_policy":"public","workload":{},"notes":""}
{"record_type":"span","trace_id":"tr-1","span_id":"sp-r1","kind":"model_request","name":"plan","start_ns":1000000,"end_ns":2900000000,"parent_span_id":"sp-step0","session_id":"sess-a","step_id":"sess-a-s0","request_id":"agent-sess-a-s0-1","depends_on":[],"status":"ok","attributes":{"input_modalities":["text"],"output_modalities":["text"],"endpoint":"chat.completions","max_tokens":192}}
{"record_type":"span","trace_id":"tr-1","span_id":"sp-t1","kind":"tool_call","name":"calculator","start_ns":2901000000,"end_ns":2901200000,"parent_span_id":"sp-step0","session_id":"sess-a","step_id":"sess-a-s0","depends_on":["sp-r1"],"status":"ok","attributes":{"tool_name":"calculator","tool_status":"ok"}}
```

（为便于阅读，这里省略了 session、step、event 和 footer。真实的例子见 `examples/traces/`。）

## 与服务端事件的关联

客户端把 `request_id` 放进 `/v1/chat/completions` 的请求体，SGLang-Omni 的 request event recorder 在所有阶段事件里使用同一个 id（事件格式见上游 `docs/developer_reference/profiler.md`）。所以一个 `model_request` span 可以和它在 preprocessing、encoder、thinker、talker、code2wav 的事件对齐。服务端事件不写进 Agent trace，单独存放在实验目录的 `raw/server_events/`，避免混淆两层数据。
