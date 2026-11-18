# 证据索引：每个关键结论对应的代码、报告、数据和外部来源

> 2026-11-18。用法：在做 PPT 或写报告时，每引用一条结论，先在这里找到它的等级和出处。等级：Measured / Derived / Estimated / Hypothesis / Proposed / Unknown。
> 路径约定：`research/…`、`docs/…`、`scripts/…` 是本仓库；`sglang_omni/…` 是 sglang-omni `520b49ee`；`python/sglang/…` 是 SGLang `e00930c5`（0.5.21）。原始数据（日志、trace、事件）在集群的 `PROJECT_DATA_ROOT/experiments/<运行目录>/`，不公开；各实验的 `metadata.json` 列出运行目录。

## 1. 系统行为

| ID | 结论 | 等级 | 代码 | 报告 | 数据 |
|---|---|---|---|---|---|
| S1 | MiniCPM-o chat 路径：thinker 写完 → talker 全部生成 → code2wav 整段合成，语音一次返回 | 源码 + Measured | `sglang_omni/models/minicpm_o/config.py:80-117`；`models/minicpm_o/routing.py:155-194`；`models/minicpm_o/talker_request.py:26-87` | `docs/04_experiments/metrics.md`“关于首个音频块时间”；实验 6 6.1 | 实验 6 `results/summary.json` |
| S2 | 准入时按轮转绑定所有阶段的副本，不看负载、不看会话 | 源码 | `sglang_omni/pipeline/replicas.py:269-315`；`pipeline/coordinator.py:555-559` | `thunderagent_flashstates_related_work.md` §3 | — |
| S3 | chat API 只接受 `request_id`、`user`；`user` 不使用；没有会话/优先级字段 | 源码 | `sglang_omni/serve/protocol.py:50-110`；`serve/openai_api.py:1063-1188` | 同上 | — |
| S4 | SGLang 优先级调度存在且被传入，但 MiniCPM-o 不设 `Req.priority`，所以无效 | 源码 | `python/sglang/srt/arg_groups/fields/schedule.py:99-125`；`sglang_omni/scheduling/omni_scheduler.py:563-582, 1716-1720` | 同上 | — |
| S5 | thinker RadixCache（LRU，`EvictHeapRadixCache`）开、talker 关；HiCache 被写死关闭 | 源码 | `sglang_omni/scheduling/sglang_backend/cache.py:57-65`；`scheduling/omni_scheduler.py:596-597`；`models/minicpm_o/bootstrap.py:50-55` | 同上 | — |
| S6 | talker 名额 32、KV 上限 32×4096；graph 最大批跟随名额 | 源码 + Measured | `sglang_omni/models/minicpm_o/stages.py:143-161`；`scheduling/generation_batch_policy.py:217-243` | `p0_p3_experiment_plan.md` P0 | 实验 9 启动日志（集群） |
| S7 | talker KV 每 token 60 KiB，池 131072 token ≈ 7.5 GiB；thinker 144 KiB/token，`mem_fraction_static` 0.697，池 127,039 token | 源码 + Measured | 同上 | `research_baseline.md` 第 4 节 | 实验 9 启动日志 |
| S8 | code2wav：FIFO、批上限 16、不等待；flow CUDA graph 155 个形状、批 11–16 没有 | 源码 | `sglang_omni/models/minicpm_o/config.py:136-160`；`models/minicpm_o/components/token2wav/flow_graph_shapes.py:7-91`；`token2wav/flow_cuda_graph.py:306-354` | 实验 10 5.1；`compiler_runtime_opportunities.md` 3 | — |
| S9 | 中止会广播到所有阶段；code2wav 正在算的批算完再丢弃 | 源码 | `sglang_omni/pipeline/coordinator.py:705-760`；`pipeline/stage/runtime.py:2283-2318`；`scheduling/simple_scheduler.py:270-282, 417-426` | `robotics_experiment_plan.md` 1 | — |
| S10 | 原生全双工：`--variant session --enable-realtime`，1 秒单元，默认 2 个会话、四阶段同卡；能追加图像；520b49ee 不支持 `response.cancel` | 源码 | `sglang_omni/models/minicpm_o/native_config.py:17, 24-67, 142-145`；`serve/realtime/schema.py`；`serve/realtime/types.py:110-135` | `robotics_application_intro.md` 2.3 | — |
| S11 | MiniCPM-o 语音流水线不支持轮次式 realtime 的音频输出 | 源码 | `sglang_omni/serve/launcher.py:656-658`；`serve/realtime/session.py:241-248` | 同上 | — |

## 2. 性能发现（A6000，520b49ee）

| ID | 结论 | 等级 | 报告 | 数据 / 脚本 |
|---|---|---|---|---|
| F1 | 2+2+4 容量 23.77 对 2+6 的 22.12（+7.5%，+2.8..+12.1）；开环 W3b 23.5 请求/秒 P95 3.74 对 4.34 s；W3L 10.5 请求/秒只有 2+2+4 稳定 | Measured / Derived | `research/01_性能瓶颈/实验09_最佳GPU分工/README.md` 第一、七部分；`reports/research_updates/exp09_11_findings.md` | 实验 9 `results/summary.json`、`levels.csv` |
| F2 | 2+1+5 容量 −25%，talker 卡忙 65% | Measured | 同上 | 同上 |
| F3 | 2+1+5 的 talker 在 80–83% 的日志行正好 32 个在跑、排队 178–184（W3L：81–84%、53–58） | Measured（启发式区分） | `research_baseline.md` 第 4 节 | `research/next_steps/data/p0_k1c5_talker_slots.json`、`p0_k1c5_w3l_talker_slots.json`、`p0_k2c4_talker_slots.json`；`scripts/analysis/next_talker_slots.py` |
| F4 | talker 每步时间：批 1/16/32 = 1.43/2.62/4.01 ms | Measured | 同上 | `p0_k1c5_talker_slots.json` 的 `decode_curve_all_lines` |
| F5 | 同卡时 talker 每秒语音的时间 2.2 倍，code2wav 每秒语音的 GPU 时间 +49–59% | Measured | `research/01_性能瓶颈/实验07_Code2Wav独立GPU/README.md` | 实验 7 `results/` |
| F6 | code2wav 每请求约 0.2 s GPU（flow ≈150、HiFT ≈45 ms）；并发 256 时阶段 5.25 s，42% 排队 | Measured | `research/01_性能瓶颈/实验06_Code2Wav时间拆解/README.md` 6.1–6.2 | 实验 6 `results/summary.json`、`requests.csv`、`batches.csv` |
| F7 | MIX fifo16 补齐 28%；length8 语音秒/秒 +30%（+17..+43），只在 code2wav 饱和时 | Measured / Derived | `research/01_性能瓶颈/实验10_Code2Wav_Batching/README.md` | 实验 10 `results/summary.json` |
| F8 | 完整服务（≤ 23.5 请求/秒）按长度组批没有差别，每批约 1.2 个请求 | Measured | 同上，完整服务验证 | 同上 |
| F9 | code2wav kernel：GEMM 25%、mem-efficient 注意力 19%（占用率估计 33%）+ 准备 6%、Inductor 融合 16%、HiFT TF32 卷积 10%、布局转换 3%；graph 0.3% | Measured | `compiler_runtime_opportunities.md` 1.5 | `research/next_steps/data/c2w_kernel_census_exclusive_c128.json`（另有 `_shared_c128`、`_shared_c8`）；`scripts/analysis/next_c2w_kernel_census.py` |
| F10 | talker 每步约 180 个 kernel、中位 1.9 µs、96–97% graph、卡忙 52–60%（profiler 窗口） | Measured | `compiler_runtime_opportunities.md` 1.2 | 实验 7 talker trace（集群）；本轮一次性统计，未入库为脚本 |
| F11 | thinker 每步空 3–15 ms；原型去掉空档、每 token −9..−10%，语音端到端 −2..−5%（区间含 0），TTFT +17..21 ms | Measured | `research/01_性能瓶颈/实验08_Thinker_Lookahead/README.md` 6.6 | 实验 8 `results/phase3.json` |
| F12 | 旁路复制正确性门槛未通过 | Measured | `research/01_性能瓶颈/实验11_Thinker异步优化/README.md` | 实验 11 `results/check.json`、`diagnostic.json` |
| F13 | 10 秒视频 87% 在 CPU 预处理，每副本串行，GPU 忙 8–15% | Measured（初步） | `research/01_性能瓶颈/实验14_视频请求加副本/README.md` | 实验 14 `results/summary.json` |
| F14 | 混合负载 3+5 最好；文字 Agent 8 个完整副本最好、拆分时 talker 卡 0% 忙；按会话路由命中率 +0.7–1.8 个百分点 | Measured（初步） | `research/01_性能瓶颈/实验19_负载类型和最佳分工/README.md` | 实验 19 `results/summary.json` |
| F15 | 系统里约 50–80 个请求以后只是在排队 | Measured | `research/01_性能瓶颈/实验03_同时处理多少请求最好/README.md` | 实验 3 `results/` |
| F16 | thinker→talker 交接 3.4–4.5 ms，与链路无关 | Measured | `research/01_性能瓶颈/实验13_GPU间连接快慢/README.md` | 实验 13 `results/` |
| F17 | 停顿 2：全进程日志 3.09 s 无输出，服务和客户端 CPU 同时掉到接近 0 | Measured | `p0_p3_experiment_plan.md` P3 | `research/next_steps/data/p3_stall2_t2_k3_c3_block3_w3b.json`；`scripts/analysis/next_stall_alignment.py` |
| F18 | 停顿 1：code2wav 24–27 s 无批，talker 照常；之后积压 | Measured | 同上 | `research/next_steps/data/p3_stall1_t2_k2_c4_block3_w3l.json` |

## 3. 推算、假设和提议

| ID | 内容 | 等级 | 文档 |
|---|---|---|---|
| H1 | talker 名额改到 64：吞吐 +10–20%，2+1+5 仍低于 2+2+4 | Estimated（事先登记的预测） | `p0_p3_experiment_plan.md` P0 |
| H2 | 停顿 2 来自共享文件系统或主机级暂停 | Hypothesis | 同上 P3 |
| H3 | 打包变长注意力 + 打包 graph 降低 code2wav GPU 时间 | Hypothesis | `compiler_runtime_opportunities.md` 推荐 1 |
| H4 | HiFT 折叠 + 布局 + 半精度：code2wav GPU 时间 −7..−12% | Estimated | 同上 推荐 2 |
| H5 | thinker W4A16：decode 每步约 8–10 ms | Estimated | 同上 候选 C |
| H6 | 多模态机器人会话的工作集超过 thinker KV 容量 | Hypothesis | `thunderagent_flashstates_related_work.md` 假设 B |
| H7 | 按下游语音队列做准入能提高 goodput | Proposed | 同上 假设 A |
| H8 | 全双工在 A6000 上的 TTFA、打断延迟、会话容量 | Unknown | `robotics_experiment_plan.md` |

## 4. 外部来源

| 来源 | 用在哪里 | 链接 |
|---|---|---|
| ThunderAgent（ICML 2026，PMLR 306:55630–55657；arXiv:2602.13692）；代码 `7ddc861` | 相关工作 §1 | https://proceedings.mlr.press/v306/kang26j.html ；https://github.com/ThunderAgent-org/ThunderAgent |
| FlashStates（alphaXiv `2611.flashstate`，v2 2026-11-10）；代码 `6d3503e`、vLLM 分支 `3cfd432`、Mooncake 分支 `59ca03f` | 相关工作 §2 | https://flashstates.ai/ ；https://github.com/FutureMLS-Lab/FlashStates |
| SGLang-Omni（本项目 pin `520b49ee`；上游 main 审计于 `d84bd1c5`） | 全部源码事实 | https://github.com/sgl-project/sglang-omni |
| SGLang-Omni PR #2843（全双工基准，H200） | 机器人介绍 2.3 | https://github.com/sgl-project/sglang-omni/pull/2843 |
| SGLang-Omni 交互运行时博客（H200 上 8 个会话） | 同上 | https://www.sglang.io/blog/omni-interactive-runtime |
| SGLang-Omni issue #2399（MiniCPM-o code2wav FP16 + compile，H20） | compiler 推荐 2 | 见 `docs/01_research/prior_work.md` 参考文献 [11] |
| SGLang-Omni issue #2580（MiniCPM-o 视频的串行预处理，H200） | 机器人介绍 Finding E | 见 `docs/01_research/prior_work.md` [12] |
| OmniCast、VoxServe、Metronome、Cornfigurator、HorizonServe、M\*、Omni-Flow | 研究空白 | `docs/01_research/prior_work.md` 4.3 及参考文献 [44]–[51] |
| InferCept、Continuum、KVFlow、TokenCake、Parrot、Agentix/Autellix | 研究空白 | `docs/01_research/prior_work.md` 5.1–5.2 及参考文献 [52]–[61] |

ThunderAgent 和 FlashStates 的逐条引文（论文页码、公式、代码行号）在本轮调研的笔记里；仓库里的两份文档只摘录了结论和关键位置。
