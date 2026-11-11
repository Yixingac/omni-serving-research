# 数据来源

本项目只使用公开数据、合成数据，或明确授权的数据。仓库是公开的，原始实验数据不提交。

| 数据 | 来源 | 性质 | 存放 |
|---|---|---|---|
| 文本提示词 | 代码里的固定词表随机生成（`replay/payloads.py`）和几条自写的短问题 | 合成 | 由配置和种子重建 |
| Agent 任务与库存数据 | `tracing/agent_harness.py` 按种子生成 | 合成 | 由种子重建 |
| 语音提问、朗读段落 | 用 Qwen3-TTS-12Hz-0.6B-Base 合成（参考音色取自 SGLang-Omni 仓库的测试音频 `tests/data/query_to_cars.wav`，Apache-2.0） | 合成 | `$PROJECT_DATA_ROOT/assets/synthetic-v1/` |
| 计数视频 | `replay/assets.py` 逐帧绘制，用 ffmpeg 编码 | 合成 | 同上 |
| 模型权重 | Hugging Face，固定 revision（见 `configs/models/`） | 公开 | Unicorn NFS、Spark 本地，不进 git |
| Agent trace | 我们自己的 Agent 对真实服务运行时记录 | 真实执行 + 合成任务 | 实验目录 `raw/` |

素材的 manifest（`manifest.jsonl`）为每个文件记录 `source`，例如 `synthetic:tts:Qwen/Qwen3-TTS-12Hz-0.6B-Base@5d83992` 或 `synthetic:procedural:omni_research.replay.assets`。

## 合成数据带来的局限

- 合成语音的音色单一、背景干净，音频编码器的计算量主要取决于时长，影响有限，但语音理解的正确率不能代表真实用户。
- 合成视频画面简单，视频解码和抽帧的 CPU 开销与真实视频相比可能偏低（编码复杂度低）。
- Agent 任务短而规整，工具延迟是模拟的。这类 trace 能反映“请求-工具-请求”的依赖结构，但请求数量、间隔分布不代表真实产品。
- 回放时如果 trace 没有保存请求内容（`content_policy=none`），只能按大小构造替代输入，模型输出长度会和原始请求不同；这种情况在结果里标为 `payload.kind = surrogate`。

## 没有使用的数据

- 实验室未公开的数据、任何用户数据：不使用。
- 公开的生产 trace（Azure、BurstGPT、Mooncake 等）：只有文本请求，没有语音或 Agent 依赖结构（见 `docs/01_research/prior_work.md` 第 6 节），目前没有用于回放。
