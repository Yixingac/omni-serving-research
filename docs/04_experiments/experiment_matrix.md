# 实验矩阵

## 第一批：Job 409712（3×A6000，10 小时）

计划文件：`configs/experiments/plan_a6000_3gpu.json`；每个实验的配置在 `configs/experiments/a6000/`。作业启动后由编排器按依赖和优先级自动运行。状态一列在实验结束后更新。

| 实验 | GPU | 依赖 | 内容 | 回答的问题 | 状态 |
|---|---|---|---|---|---|
| x0_envcheck | 1 | — | 导入检查、驱动、拓扑、包版本 | — | 待运行 |
| x1_assets | 1 | x0 | Qwen3-TTS 合成语音提问和朗读段落；生成计数视频 | — | 待运行 |
| x2_minicpm_text_1gpu | 1 | x0 | W1 文本，并发 1–64 | Q3、Q6 | 待运行 |
| x3_minicpm_speech_1gpu | 1 | x1 | W3b、W3 语音输出，W1 在语音服务上；并发扫描 | Q3、Q5 | 待运行 |
| x4_minicpm_av_text_1gpu | 1 | x1 | W2、W2L、W4（2/5/10/20 秒视频） | Q3、Q4 | 待运行 |
| x5_minicpm_agents | 1 | x1 | Agent 会话：文本 c=1/4/8（含模拟工具延迟），语音 c=4（含带事件版本） | Q2、Q1 | 待运行 |
| x7a/b/c | 2 | x3、x5 | 2 卡预算的三种放置：完整副本 / 拆分 / 只复制 talker；语音扫描 + Agent 回放 | Q7、Q8 | 待运行 |
| x3r | 1 | x5 | 与 x7 相同的 Agent 回放，作为 1 卡对照 | Q7、Q9 | 待运行 |
| x6_minicpm_replay | 1 | x5 | 请求级 vs 依赖感知回放；会话复制、加速 | Q9 | 待运行 |
| x8_minicpm_profiled_1gpu | 1 | x3 | 同一负载开/关服务端事件记录；阶段分解 | Q1、Q5 | 待运行 |
| x4c_minicpm_video_4cpu | 1（4 CPU） | x4 | 与 x4 的 10/20 秒视频阶段相同，只给 4 个 CPU | Q4 | 待运行 |
| x3m_minicpm_speech_1gpu_memsplit | 1 | x1 | 与 x3 相同的语音扫描，thinker `mem_fraction_static` 从 0.85 降到 0.6。Spark 上默认配比会让 code2wav 跳过 CUDA graph，但没有测出耗时差异；A6000 显存更紧，在这里复核 | Q5、Q6 | 待运行 |
| x9_qwen3_omni_3gpu | 3 | x0 | Qwen3-Omni thinker TP=2 + talker；W3b、W1 | Q3（第二个模型） | 待运行 |

依赖关系的设计：语音素材由单独的 TTS 模型生成（x1），所以即使 MiniCPM-o 的语音模式在单卡上出问题，音频理解类的实验（x4）也不受影响。2 卡放置实验依赖单卡语音基线（x3）成功，否则先查单卡的问题。

## 计划中但需要更多资源或更新版本的实验

| 实验 | 需要 | 说明 |
|---|---|---|
| 4 卡、8 卡的副本扩展与放置 | 8×A6000 的分配（节点 B） | 1/2/4/8 卡的扩展曲线；4 卡上比较 2×2 拆分与 4 个完整副本 |
| NVLink 对与非 NVLink 对的拆分对比 | 拓扑信息（Job 409712 的 preflight） | 检验阶段间传输的代价 |
| 预处理并发度 A/B | 更新的 sglang-omni 版本 | 当前固定版本的 MiniCPM-o 预处理没有并发参数 |
| Nsight Systems 时间线 | 先在 Spark 上验证采集和解析 | 区分 GPU 空闲的来源 |
| 全双工 `/v1/realtime` | WebSocket 客户端与新的测量方法 | 暂缓 |
