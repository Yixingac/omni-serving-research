# 研究基线：已经知道什么、不知道什么、哪些结论有条件（Part 0 审计）

> 2026-11-18 撰写。审计对象是分支 `research/agent-trace-end-to-end` 上 `research/`、`reports/research_updates/`、`docs/01–05`、`src/omni_research/`、`configs/runtime/patches/` 的内容，状态以各实验 README 顶部的状态行和 `research/实验总目录.md` 为准，不以历史报告里的“RUNNING / SUBMITTED”为准。
> 同一天另一个会话用已有数据补写了实验 3、4、12、13、14、19 的报告（实验 3 `9e8fd2c`、实验 4 `85086db`、实验 13 `2c86ead` 等），本文按这些提交核对。
> 证据等级沿用仓库约定：Measured（实测）、Derived（由实测推算）、Estimated（估算）、Hypothesis（假设）、Unknown（未知）。

所有实验的共同条件（下文不再重复）：MiniCPM-o 4.5，sglang-omni `520b49ee`（加我们自己的测量或原型补丁），SGLang 0.5.21，torch 2.13.0，一台 8×RTX A6000 节点（compute-03，NVLink 只连 0-3、1-2、4-5、6-7，其余 PCIe），合成负载（W3b：6 个短问题、文字进、文字+语音出；W3L：约 100 词的长回答；MIX：两者各半），`/v1/chat/completions` 请求-响应路径。**没有任何实验走过 `/v1/realtime` 全双工路径。**

## 1. 五类状态

| 类别 | 实验 | 说明和证据位置 |
|---|---|---|
| ① 已运行并完成（有按新格式写的报告） | 实验 1、2、3、4、5、6、7、8、9、10、12、13、14、18、19 | 报告在 `research/01_性能瓶颈/实验NN_*/README.md`；实验 18 是旧格式，见 `docs/05_findings/baseline_results.md` 第 17、20 节。实验 2、12、13、14、19 的多数配置只跑过 1 次，报告标“初步” |
| ② 有数据、分析未完成 | 审计结束时（11-18 晚）没有整编号的实验还停在这一类 | 实验 3、4 原来属于这一类，11-18 由另一会话补写了报告（`9e8fd2c`、`85086db`）。要注意：实验 4 的批大小是负载自然形成的，主动改批上限的只有实验 10（code2wav），thinker、talker 的批上限从没改过。Spark（GB10）上的 s1–s14 是功能验证，按约定不和 A6000 比较，不算待分析数据 |
| ③ 只做过局部验证的原型 | 实验 8 的“发射时快照”提前发射原型；实验 10 的按长度组批原型 | 实验 8：并发 1 逐位正确，性能 3 个区组，decode 空档消失但端到端 −2..−5%（区间含 0）；实验 10：只在 code2wav 是瓶颈的 t2_k4_c2 上测到 +30%，完整服务（t2_k2_c4，≤ 23.5 请求/秒，未饱和）验证为“不确定” |
| ④ 因正确性问题停止的优化 | 实验 11 的旁路复制（side copy） | 正确性门槛未通过（6 段指纹偏差比例超参照 5 个百分点以上），按登记没有做性能对比；诊断也排除不了旁路复制的问题。`research/01_性能瓶颈/实验11_Thinker异步优化/README.md` 第一、五、七部分 |
| ⑤ 完全没有开展 | 实验 15（流水线重叠 / 流式）、16（投机解码）、17（量化）；全双工 `/v1/realtime`；带长上下文的 Agent 会话、工具等待期间的 KV 保留；持续视频流 | `research/实验总目录.md` 实验 15–17 行；`docs/01_research/research_questions.md`“暂不研究”一节；`docs/01_research/prior_work.md` 第 8、9 节 D3、D4 |

## 2. 实验 6–11 逐个核对

| 实验 | 真实状态 | 能说的结论（等级） | 不能说的 | 依据 |
|---|---|---|---|---|
| 6 Code2Wav 时间拆解 | 完成，3 个区组 | 每个请求固定约 0.2 秒 GPU（flow ≈150 ms、HiFT ≈45 ms，CUDA event，Measured）；自然形成的批不让单请求变便宜（Measured，相关）；并发 256 时 code2wav 阶段 5.2 秒，42% 排队、58% 等同批（Measured）；高负载时 kernel 发出后排队 12–94 ms，GPU 是瓶颈（Measured） | 凑批无效的因果（批大小随负载自然变化）；HiFT 开头同步的原因（只是推断） | `实验06_Code2Wav时间拆解/README.md` 6.1–6.4；`results/summary.json` |
| 7 Code2Wav 独立 GPU | 完成，5 种放置 × 3 个区组 | 同卡时 talker 每秒语音 2.2 倍慢、code2wav 每秒语音 GPU 时间 +49–59%（Measured）；等预算只有 2+2+4 胜过 2+6（并发 256 +8.2%，区间 −0.0..+16.5，P95 −17.6%）（Measured） | H7b 按登记标准只是“不确定”（吞吐未达 +10%）；开环只测到 10.5 请求/秒 | `实验07_Code2Wav独立GPU/README.md`；`reports/research_updates/exp06_08_findings.md` |
| 8 Thinker 提前发射 | 完成，三阶段 | 同步执行每步空 3–15 ms（Measured）；语音被挡是因为下一步 graph 重放覆盖隐藏状态（源码 + “不做快照”诊断组全部出错，Measured）；原型逐位正确（并发 1，18 个请求）；decode 空档 ≤ 0.06 ms，每 token −9..−10%，端到端 P50 −2..−5%（区间含 0），TTFT +17..21 ms（Measured） | 原型能改善语音端到端（未证实）；剩余 25% 空档里混合轮次和阻塞拷贝各占多少（Unknown） | `实验08_Thinker_Lookahead/README.md` 6.5、6.6 |
| 9 最佳 GPU 分工 | 完成，12 个运行 | 2+2+4 在 W3b、W3L 都最好：W3b 容量 23.77 对 2+6 的 22.12（+7.5%，区间 +2.8..+12.1）；开环 W3b 23.5 请求/秒 P95 3.74 对 4.34 s（Measured/Derived）；2+1+5 容量 −25%，唯一 talker 是瓶颈但 GPU 只忙 65%（Measured） | “32 个名额导致 2+1+5 变差”只是 Hypothesis（本轮补了日志证据，见第 4 节，但没有改名额的运行）；两次短暂停顿的原因（Unknown） | `实验09_最佳GPU分工/README.md` 第七部分；`results/stalls.json` |
| 10 Code2Wav 组批 | 完成：正确性通过、6 个性能运行、3 个完整服务验证 | code2wav 是瓶颈且长短混合（MIX）时，fifo16 补齐 28%，length8 每秒语音 +30%（+17..+43）（Measured，t2_k4_c2）；默认配置下批 ≥2 时 flow graph 只命中 4–18%（Measured） | 完整服务上有收益（验证为“不确定”：≤23.5 请求/秒未饱和，平均每批 1.2 个请求）；公平性代价是否可控（短请求 P95 排队 +56%、+21%，描述性） | `实验10_Code2Wav_Batching/README.md` 第一、六部分 |
| 11 Thinker 异步拷贝 | 正确性未通过；性能 BLOCKED | 并发 1 时逐位相同、无报错、隐藏状态行数都对（Measured）；最大指纹偏差 0.143（读错一行约为 1） | 旁路复制正确（不能说）；任何性能收益（没测）；不能把它写成成功 | `实验11_Thinker异步优化/README.md` 第一、七部分 |

## 3. 研究基线：知道 / 不知道 / 有条件

### 3.1 已经知道（Measured 或 Derived，条件见上）

1. **高负载下决定吞吐的是语音那一半，不是 thinker。** 2+6 时 talker 卡 90–97% 在算，code2wav 占其中一半以上，请求在 code2wav 门口排队，占端到端 62–76%（实验 5、6）。
2. **code2wav 每个请求的 GPU 成本几乎固定**（约 0.2 s），自然凑批不省（实验 6）；但长短混合、主动按长度组批时能省 24% 的 GPU 时间（实验 10，code2wav 饱和时）。
3. **放置比“加副本”更重要，且最优点在两个阶段能力相等处**：2+2+4 最好；再挪一张卡给 code2wav（2+1+5）瓶颈就转到 talker，容量 −25%（实验 7、9）。
4. **thinker 的 GPU 空档是真实的、可去掉的，但对语音端到端影响小**（实验 5、8）。纯文字、不带 seed 时提前发射能让端到端 P50 快 9–14%（实验 8 第一阶段）。
5. **MiniCPM-o 在当前请求-响应路径上，语音在请求结束时一次性返回**，首段音频时间（TTFA）≈ 端到端时间（`docs/04_experiments/metrics.md`“关于首个音频块时间”；实验 6 6.1；实验 9 3.4）。
6. **视频理解受 CPU 预处理限制**：10 秒视频 87% 的时间在 CPU 预处理，每个副本一次只做一个，GPU 只忙 8–15%（实验 14，单次运行）。
7. **负载组成改变最佳分工**：混合负载下 3+5 最好；thinker 每个副本最多 64 个在跑请求，2+6 在混合负载下排队（Derived，实验 19，单次运行）。
8. **按会话固定路由对当前合成 Agent 几乎没用**：前缀命中率只高 0.7–1.8 个百分点，延迟没有一致变化（实验 19，单次运行）。
9. **卡间连接不是问题**：thinker→talker 交接 3.4–4.5 ms，与链路类型无关（实验 13）。
10. **talker 和 code2wav 同进程并发会崩溃**，拆成单独进程后 31 个实验约 6 小时 0 次崩溃（实验 18）。

### 3.2 还不知道（Unknown）

| 问题 | 为什么还不知道 | 本轮计划 |
|---|---|---|
| talker 名额从 32 改到 64，2+1+5 能否追上 2+2+4 | 没有改名额的运行 | P0（`p0_p3_experiment_plan.md`） |
| 按长度组批在饱和的完整服务上是否有收益、加“最长等待”后公平性是否可接受 | 验证只覆盖未饱和区间 | P1 |
| 旁路复制是否正确 | 指纹检查分辨不了小差别 | P2：同一次运行里保存两份、逐位比较 |
| 两次短暂停顿的原因 | 当时没有调用栈、GC、交接事件 | P3：本轮已用现有数据缩小范围（见第 4 节） |
| 全双工 `/v1/realtime`、流式音频、打断（barge-in）在 A6000 上的行为 | 从没跑过 | R 主线（`robotics_experiment_plan.md`） |
| 工具等待期间 thinker 的 KV 保留和重算代价 | 现有 Agent 会话只有约 2 次请求、上下文短、工具耗时 24–60 µs | ThunderAgent/FlashStates 的结合研究（`thunderagent_flashstates_related_work.md`） |
| code2wav 里具体哪个 kernel 值得优化、能快多少 | 以前只有按类别的拆分 | C 主线：本轮已做 kernel 普查（见第 4 节）；收益仍需微基准 |
| 量化对速度和音质的影响 | 实验 17 未开始 | C 主线候选 C |

### 3.3 有条件的结论（引用时必须带上条件）

| 结论 | 条件 |
|---|---|
| “2+2+4 最好” | W3b/W3L（文字进、语音出、6 个提问），8×A6000，520b49ee；混合负载和文字 Agent 下最优分工不同（实验 19） |
| “按长度组批 +30%” | 只在 code2wav 饱和、长短混合（MIX）时；未饱和的完整服务上无差别 |
| “提前发射每 token 快 9–15%” | 是局部指标；语音端到端只有 −2..−5%（区间含 0），且 TTFT 变长 |
| 容量的绝对数字 | 取决于算法：登记的容量比满额时段吞吐低 20–36%（实验 9）；排序在三种算法下一致 |
| “CUDA graph 帮助小” | 指高负载（GPU 忙）；低负载时 flow 逐个发 kernel 每帧慢约 16%（实验 6），对低并发的实时场景可能重要 |
| 所有 Agent 结论 | 合成任务、短会话（中位 2 次请求）、本地工具（微秒级），不代表真实 Agent |

## 4. 本轮新增的证据（只用已有数据，没有新的 GPU 运行）

| 发现 | 等级 | 数据 | 工具 |
|---|---|---|---|
| 2+1+5、W3b 并发 256：唯一 talker 的 decode 日志行有 80–83% 正好是 32 个在跑请求，且同时排队 178–184 个（3 个区组）；W3L 并发 128 时 81–84%，排队 53–58 个。2+2+4 时每个 talker 在跑中位数 16–18，排队中位数 0 | Measured（按 token/请求区分 talker 行，启发式） | `data/p0_k1c5_talker_slots.json`、`data/p0_k1c5_w3l_talker_slots.json`、`data/p0_k2c4_talker_slots.json` | `scripts/analysis/next_talker_slots.py` |
| 独占一张卡的 talker：批 1 → 16 → 32 时每步 1.43 → 2.62 → 4.01 ms，吞吐 701 → 6111 → 7982 token/s；16→32 只多 31% | Measured（SGLang 日志的 gen throughput，3 个区组合并） | `data/p0_k1c5_talker_slots.json` 的 `decode_curve_all_lines` | 同上 |
| talker 的 CUDA graph 只录了批大小 `[1, 2, 4, 8, 12, 16, 24, 32]`；KV 池固定 131072 token（K、V 各 3.75 GB，共约 7.5 GiB），可用于 KV 的显存是 18.13 GiB；thinker 的 `mem_fraction_static` 是 0.697，KV 池 127,039 token | Measured（服务启动日志） | 实验 9 方案 3 的 `logs/server_s0.log`（集群，不公开） | — |
| 两处旧文档的数字需要更正：`docs/04_experiments/plan_8gpu_v2.md`、`configs/` 里写的 talker KV“约 15 GB”是按显存比例算的预算，实际池约 7.5 GiB（上限 131072 token × 60 KiB）；`docs/05_findings/baseline_results.md` 第 8 节里 thinker 的 0.85 不是本节点 plan v2 以后的实际值（实测 0.697） | Measured（日志）+ 源码 | 同上；`sglang_omni/models/minicpm_o/stages.py:160-161` | 本轮只记录，不改那些文件（它们属于已完成的历史报告） |
| code2wav 独占卡、并发 128 的 kernel 时间：GEMM 25%、PyTorch mem-efficient 注意力 `fmha_cutlassF_f16` 19%（占用率估计 33%）+ 构造 mask/布局的融合 kernel 6%、Inductor 融合 kernel 16%、HiFT 的 FP32/TF32 卷积 10%、`nchwToNhwc<float>` 布局转换 3%；CUDA graph 只占 0.3% | Measured（CUPTI GPU 时间戳）；flow/HiFT 归属是按名字的启发式，和实验 6 的 CUDA event 拆分（77%/23%）一致 | `data/c2w_kernel_census_*.json` | `scripts/analysis/next_c2w_kernel_census.py` |
| 停顿 2（2+3+3 区组 3，W3b 21 请求/秒）：所有进程共用的服务日志 3.09 s 没有一行；同一时刻服务进程组 CPU 从约 800% 掉到 6%，客户端也一起掉，整机 CPU 13% → 0–2% | Measured | `data/p3_stall2_t2_k3_c3_block3_w3b.json` | `scripts/analysis/next_stall_alignment.py` |
| 停顿 1（2+2+4 区组 3，W3L 5.5 请求/秒）：第 24–27 秒 code2wav 一个批都没开始，talker 照常解码、排队 0；之后 code2wav 一次性收到积压（第 29、31 秒的批合计 15、29 个请求） | Measured | `data/p3_stall1_t2_k2_c4_block3_w3l.json` | 同上 |
