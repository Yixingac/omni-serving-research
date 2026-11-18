# 示例 Agent trace

两个真实运行的 Agent trace，来自 2026-11-11 在个人 DGX Spark（GB10）上的功能验证 s4。格式见 `docs/02_workloads/agent_trace_schema.md`。

| 文件 | 内容 | 会话 | 模型请求 | 工具调用 |
|---|---|---|---|---|
| `spark_minicpm_o_text_agents.jsonl` | 文本输入的 Agent，并发 3；算术、单位换算、库存汇总三类合成任务，9/9 答对 | 9 | 18 | 12（库存类任务每步并行调用两次） |
| `spark_minicpm_o_voice_agents.jsonl` | 语音提问输入（MiniCPM-o 自带 TTS 合成），最后一步语音回答，并发 2，3/4 答对 | 4 | 15 | 7 |

- `source.kind = observed`：模型请求是对真实 MiniCPM-o 4.5 服务发出的；任务、库存数据都是合成的，没有模拟工具延迟。
- `content_policy = public`：保存了完整请求体（包括模型的回复），所以可以原样回放。
- 发布前做过的修改：文件头里的主机名和 pid 换成了通用值，音频文件的绝对路径改成了相对 `PROJECT_DATA_ROOT` 的路径。时间、id 和依赖关系没有改动。
- 这些数字来自 GB10，只用于说明格式和功能，不代表 A6000 上的性能。

检查和回放：

```python
from omni_research.tracing import load_trace, validate_trace
from omni_research.replay import build_plan

t = load_trace("examples/traces/spark_minicpm_o_text_agents.jsonl")
print(validate_trace(t).to_dict()["stats"])
plan = build_plan(t)            # 依赖来自 trace 中显式记录的 depends_on
print(plan.n_requests, plan.dependency_source)
```
