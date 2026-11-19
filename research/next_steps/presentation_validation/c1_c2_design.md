# C1 / C2 微基准：设计、源码事实和事先登记

> 2026-11-18 晚。本文在任何 C1/C2 的 GPU 运行之前写定（提交时间即登记时间）。原登记见 `research/next_steps/compiler_runtime_opportunities.md` 推荐 1、推荐 2 和 `research_priorities.md` 第 3、6 行；本文只补执行细节、阈值和一处 Erratum，不改原文。
> 证据等级：Measured（实测）/ Derived（由实测推出）/ Estimated（估计）/ Hypothesis。代码位置都指 sglang-omni `520b49ee`（`sglang_omni/models/minicpm_o/...`，下文简写 `mo/`，`t2w/` = `mo/components/token2wav/`）。

## 1. 一句话

C1 问：在**真实的批形状**上，把 code2wav flow 的稠密带 mask 注意力换成变长打包（varlen）能不能让“每秒有效语音的 GPU 时间 E”下降 ≥ 5%，且音质不变差。C2 问：HiFT 的 weight-norm 折叠、channels-last 布局、FP16 主体三项各自值多少，哪项过得了音质门槛。两者都是**微基准**，结论不直接等于端到端收益。

## 2. 重新核对过的源码事实（520b49ee）

| 事实 | 位置 |
|---|---|
| code2wav 的默认工厂参数：`max_batch_size=16`、`dtype="float16"`、`enable_dit_torch_compile=True`、`enable_hift_torch_compile=True`、`enable_flow_variable_length=False`、`enable_flow_block_compile=False`；`flow_cuda_graph_capture_shapes` 默认 155 个形状，`packed_dit_cuda_graph_capture_shapes` 默认 100 个（SeedTTS EN 调出来的） | `mo/config.py:120-161`，`t2w/flow_graph_shapes.py` |
| 稠密 graph 形状：B=1 有 44 个帧数桶（128–1024），B=2..10 各有一组（最长 784 帧），B ≥ 11 没有；`fit` 取同一批大小里能装下的最小桶，装不下就 eager | `t2w/flow_graph_shapes.py:7-91`，`t2w/flow_cuda_graph.py:306-325` |
| 稠密路径：`F.scaled_dot_product_attention(q, k, v, attn_mask=布尔 mask)`；布尔 mask 不能走 flash 后端，只能走 mem-efficient（与 kernel 普查的 `fmha_cutlass` 一致） | `t2w/dit.py:105-152`（mask 来自 `:610`） |
| 打包路径：`torch.nn.attention.varlen.varlen_attn`；在 torch 2.13 里它调用 `aten._flash_attention_forward`（cuDNN 分支被写死关闭），即 PyTorch 自带的 FlashAttention-2，需要 FP16/BF16 | `t2w/dit.py:15, 154-175`；torch `nn/attention/varlen.py:30-33, 110` |
| **打包路径固定用 BF16**：`with torch.autocast(..., dtype=torch.bfloat16)`，而稠密路径是 FP16（flow 权重整体转成 FP16）；上游注释说 FP16 的 mel 更接近 FP32 | `t2w/dit.py:616-618`，`t2w/vocoder.py:177-178`，`mo/config.py:148-150` |
| 打包只在 B ≥ 2 个请求时发生（CFG 后 ≥ 3 行）；B = 1 永远走稠密（有 graph 就用） | `t2w/flow.py:137`，`t2w/dit.py:38, 611-615` |
| 打包 DiT 的 CUDA graph 要求：同批大小有一个容量 C，满足 补齐后帧数 ≤ min(C, 1024)、C > 有效帧数×2、C − 有效帧数×2 ≤ 补齐后帧数；外面的 pre/post 边界 graph 还要求 (B, 帧数) 有稠密桶 | `t2w/packed_dit_cuda_graph.py:236-256`，`t2w/flow.py:138-158`，`t2w/flow_cuda_graph.py:141-151` |
| `enable_flow_block_compile` 只编译 `forward_packed`；`enable_dit_torch_compile` 只编译稠密 `block.forward` | `mo/components/code2wav.py:110-117, 151-158` |
| **`config resolve` 接受、但代码不生效的组合**：不开 `enable_flow_variable_length` 时，`enable_flow_block_compile` 和 `packed_dit_cuda_graph_capture_shapes` 都没有作用；`dtype` 只改 flow，HiFT 永远 FP32 | 同上；`t2w/vocoder.py:201-209`，`mo/components/code2wav.py:458-462`（`.float()`） |
| HiFT：82 个 weight-norm 参数化（`torch.nn.utils.parametrizations.weight_norm`），其中 77 个在 `decode_body` 里、5 个在 F0 预测器里；每次调用都重算权重 | `t2w/hift.py:62-80, 142-194`，`t2w/hift_layers.py:66-92` |
| HiFT 编译：只编译 `decode_body`（`dynamic=True, fullgraph=True`），STFT/iSTFT、F0 预测、SineGen 保持 eager；cuFFT 在默认 stream 上跑 | `mo/components/code2wav.py:187-217`，`t2w/hift.py:37-53, 212-245` |
| HiFT 按“同一 token 长度的行”分组调用（一批里长度各不相同时就是每个请求一次） | `mo/components/code2wav.py:451-467` |
| HiFT 的激励噪声来自专用 `torch.Generator`，第一次调用时用 `torch.initial_seed()` 播种，之后随调用前进；flow 的噪声是构造时 `torch.randn` 的固定缓冲区，所有行都从偏移 0 取 | `t2w/hift.py:283-296`，`t2w/hift_layers.py:141-177`，`t2w/flow.py:103-107, 290-307` |
| 默认参考音频 `HT_ref_audio.wav` = 6.016 s，S3 分词后 151 个 token（实验 10 的事件里每批都记的是 151） | 模型 assets；`results/c1_real_shapes.json` |
| 工具陷阱（2026-11-18 的 CPU 检查里实际触发）：`copy.deepcopy` 一个带 weight-norm 参数化的模块，副本和原模块共用 PyTorch 动态生成的 `Parametrized*` 类；对副本 `remove_parametrizations` 会从这个类上删掉 `weight` 属性，原模块随之报 `'ParametrizedConv1d' object has no attribute 'weight'`。所以 C2 的每个方案都是新建的 `HiFTGenerator()` 再加载同一份 state_dict，不用深拷贝 | torch `nn/utils/parametrize.py`（`_inject_new_class`、`remove_parametrizations`） |

## 3. 真实批形状（Measured，从已有运行的 code2wav 批事件导出）

脚本 `scripts/microbench/c1_shapes.py`（导出 + 汇总），数据 `research/next_steps/results/c1_real_shapes.json`、回放样本 `c1_replay_batches.json`（每组最多 200 批，种子 20261119）。来源：实验 10 的 t2_k4_c2 闭环（code2wav 是瓶颈，graph 开 / 关各 3 个区组）、实验 10 的 t2_k2_c4 开环验证、实验 9 的带事件段。一共 47,484 批。每行帧数 = 2 ×（151 + 语音 token 数）。按这些规则复算 520b49ee 的路径，和事件里记录的 `flow_solve_path` **100% 一致**（全部组）。

| 组（来源 \| 负载 \| 组批） | 批数 | 平均批大小 | 每请求 token 中位 | 补齐比例（帧加权） | 打包/稠密 注意力计算量 | 默认配置下走 CUDA graph 的请求 | 打包 + 默认打包 graph 下走 graph 的请求 |
|---|---|---|---|---|---|---|---|
| 实验 10 t2_k4_c2 \| W3b \| fifo16（**主组 1**） | 95 | 12.1 | 343 | 15.7% | 0.72 | 2.8% | 1.2% |
| 实验 10 t2_k4_c2 \| MIX \| fifo16（**主组 2**） | 98 | 8.8 | 474 | 28.3% | 0.54 | 2.5% | 2.4% |
| 实验 10 t2_k4_c2 \| W3b \| length8 | 159 | 7.3 | 349 | 5.0% | 0.91 | 5.7% | 1.0% |
| 实验 10 t2_k4_c2 \| MIX \| length8 | 139 | 6.2 | 485 | 9.6% | 0.83 | 4.7% | 1.9% |
| 实验 9 t2_k2_c4 \| W3L \| fifo16（闭环 128） | 267 | 4.3 | 731 | 10.4% | 0.80 | 0% | 0% |
| 实验 10 验证 t2_k2_c4 \| W3b \| fifo16（开环 ≤ 23.5 请求/秒） | 14,503 | 1.21 | 342 | 3.6% | 0.93 | 39.4% | 39.0% |

从这张表直接得到的结论（Derived）：

1. **CUDA graph 覆盖受的是长度和批大小，不是补齐**：code2wav 饱和时（主组）只有 2.5–2.8% 的请求能走 graph；原因是批大小常超过 10（没有表），或补齐后超过 1024 帧（≈ 14.5 s 语音 + 6 s 参考）。打包后再加上游的打包 graph 表，也只有 1–2% 的请求能用上——那张表是为 SeedTTS 的短句（≤ 784 帧）调的。所以“减少补齐就能提高 graph 命中”在这些形状上不成立。
2. **打包能省的注意力计算量**：饱和时 W3b 省 28%、MIX 省 46%；但按长度组批（length8）之后只剩 9% / 17%，2+2+4 不饱和时只有 7%。两项优化的收益大量重叠，正好对应 P1 要回答的“Flow Varlen 与 Length-aware Batching 是否重叠”。
3. 2+2+4 在实验 10 验证的负载下平均批大小只有 1.21：在 P1 的放置上，打包只有在 code2wav 真的排队时才有东西可省。

## 4. 方案（一个方案一个进程，同一张卡，同样输入）

### C1（`scripts/microbench/c1_flow_packed.py`）

| 方案 | 设置 | 作用 |
|---|---|---|
| REF | flow FP32、eager、不用 graph、每次一个请求 | 只用于质量集，数值参考 |
| **D** | 生产默认：FP16、编译稠密块、155 个稠密 graph | 基线 |
| D_ng | D 关掉 flow graph（只跑主组） | 把“graph”和“打包”分开 |
| **P0** | `enable_flow_variable_length`，不编译打包块，不录打包 graph（B=1 仍用稠密 graph） | 打包本身 |
| **P1** | P0 + `enable_flow_block_compile` | 登记的候选 |
| P2（可选） | P0 + 默认打包 graph 表（只跑主组），单独标注 | 打包 graph 的作用（第 3 节预计 ≤ 2% 的请求用得上） |
| P1h（探索，不进判定） | P1，但把打包分支从 BF16 改成 FP16（脚本里替换 `DiT.forward`，不改 sglang-omni） | 分开“打包”和“BF16 精度” |

输入：批组成来自 `c1_replay_batches.json`（主组各取 32 批，次要组各 12 批，种子固定）；每行的 token 内容来自 synthetic-v1 的 TTS 语音（Qwen3-TTS 0.6B 合成、有文字稿）经 S3 分词器得到的真实语音 token，按（组, 批, 行）确定地切片；所有方案用同一份 token（记录哈希）、同一个 flow 噪声缓冲（构造前 `torch.manual_seed(20261119)`，记录哈希）。

测量：每批 CUDA event 计 flow 和 HiFT 的 GPU 时间（同实验 10 的补丁），外加墙钟；每个方案 1 个预热遍（编译、graph 捕获、cuDNN 计划，**单独报告**）+ **3 个计时遍**；每遍记录 Dynamo 新编译数（计时遍里出现就标出）、每批实际走的路径（graph runner 计数器）、显存峰值；单独一遍 torch profiler（只录 CUDA kernel，主组各 3 批）给注意力、mask 准备、GEMM 等的 kernel 时间，**不进 E**；另对 4 个代表形状单独计一次 DiT 前向（16 块）的时间。E = 一组所有批的 GPU 时间之和 ÷ 有效语音秒之和（自然按真实分布加权）。

### C2（`scripts/microbench/c2_hift.py`）

| 方案 | 设置 |
|---|---|
| A | 生产 HiFT：FP32 存储（cuDNN 默认 TF32 卷积）、每次重算 weight-norm、编译 `decode_body` |
| B | 只做 weight-norm 折叠（`remove_parametrizations(..., leave_parametrized=True)`，82 个） |
| C | 只做布局：`decode_body` 的卷积改成 channels-last（(B, C, 1, T) 的 NHWC，conv2d / conv_transpose2d），其余不变 |
| D | 只做精度：`decode_body` 的模块转 FP16；STFT/iSTFT、F0 预测器、SineGen 保持 FP32 |
| Dbf | D 改 BF16，**只在 FP16 出现非有限值时**才跑（登记的后备） |
| BC、BD、CD、BCD | 组合，**只在各单项都过门槛后**才跑 |

选 FP16 不选 BF16 的理由：声码器输出要的是尾数精度（FP16 10 位、BF16 7 位）；HiFT 的激活范围不大，`exp` 之后的幅度在 iSTFT 前会被截到 100，溢出时截断仍有定义；所以 FP16 是首选，BF16 只做后备。

输入：用生产 flow（D）为同样的真实批组成生成 mel，按生产代码的方式分组调用 HiFT，在优先级 −1 的 stream 上跑（同 code2wav 的 decode stream）；所有方案在**同一个进程**里，3 个计时遍，每遍里方案顺序随机（交错，减少漂移）。另有 profiler 遍（卷积、布局转换、FFT、weight-norm/Snake 的 kernel 时间和每次调用的 kernel 数）。

### 质量集（C1、C2 共用，`scripts/microbench/quality_eval.py`）

synthetic-v1 里 26 段有文字稿的 TTS 语音（24 个问题 5.7–13.4 s，2 段 27 s、40 s 的短文；5 段 163.8 s 的“失控生成”按清单标记排除）。每段先 S3 分词得到 token，再由各方案重新合成。C1：每 8 段一批（混合长度，像 MIX 的批）；D 另做“同一批再跑一遍”（登记的字面下限）和“每段单独一批”（批组成造成的变化）。C2：每段单独调用；激励噪声按请求重播种（种子 s1），A 另用 s2（生产路径自己的随机性），A/B/C 另跑一份不编译的版本做精确性检查。

ASR：用集群上已缓存的 **Qwen3-ASR-1.7B**，经 sglang-omni 自己的 `/v1/audio/transcriptions` 服务转写（与 2026-10-16 在 A6000 上验证过的 0.6B 配置相同，只换模型）。没有 Whisper 权重（只有 openai-whisper 这个包），PESQ/STOI 的包也没装：**PESQ/STOI 不适用**，替代指标就是 MCD + ASR + 波形误差，事先说明，不事后挑选。

## 5. 事先登记的规则

指标定义见 `src/omni_research/analysis/audio_quality.py` 的模块说明（rel RMS、能量相对差、MCD：24 kHz、25 ms 窗、5 ms 步长、80 个 mel 带、24 阶倒谱、去掉 c0、只算参考能量在最大值 60 dB 以内的帧；ASR 文本归一化和 WER）。MCD 是 MFCC 式定义，只用于方案之间比较，不和文献里的 MCD 数值比较。

### C1

- **正确性先行**（不过就不下性能结论）：
  1. 所有方案每段波形样本数 = 960 × token 数，全部有限；
  2. MCD(候选, REF) 的均值 ≤ MCD(D, REF) 的均值 + 0.1 dB；
  3. ASR：候选的完全一致段数 ≥ min(D, D 单独) − 1，且合并 WER ≤ max(D, D 单独) + 0.5 个百分点（对照文字稿）。
- **性能（原登记）**：P1 的 E（flow + HiFT）在主组 W3b 或 MIX 上、**3 个计时遍每一遍**都比 D 低 ≥ 5% → 进入端到端（与 P1 实验合并）；否则停止，并逐项报告：打包是否真的启用（每批路径）、graph 回退（D 与 P 的 graph 份额）、打包/解包开销（profiler 里的 index/copy kernel）、计时遍里有没有重新编译、sm_86 能力探测（varlen FA2 能否运行）。
- **Erratum E-C1（登记于本文，任何运行之前）**：原登记的正确性规则是“P1 对 D 的波形差异不超过 D 自己重复两次的差异”。在离线微基准里，D 对同一批输入是确定性的，重复两次的差异就是 0，这条规则就退化成“必须逐位相同”；而 520b49ee 的打包路径**设计上就是 BF16**（`t2w/dit.py:616-618`），不可能和 FP16 稠密逐位相同，所以这条规则测不出“打包有没有引入错误”，只会因为精度设计而失败。修正版（就是上面的 1–3 条，记为 C1-Q）用 FP32 参考和 ASR 判断质量是否变差；原规则照写照算、照样报告结果（字面下限 = D 的重复；另报“D 单独一批”这个批组成下限，它对应实验 10 在服务里测到的“D 自己的变化”）。两个版本的结论都写进最终结果，不只报对候选有利的那个。
- **预测（Estimated，不作为判定）**：P0 比 D 慢 5–25%（eager 打包，小 kernel 多；上游注释也这么说）；P1 对 D：W3b 主组 −5% 到 −12%，MIX 主组 −10% 到 −20%；P2 ≈ P0（第 3 节：≤ 2% 的请求能用上打包 graph）；P1 很可能过不了 C1-Q 的 MCD 条件（BF16），P1h 若比 P1 好很多，则瓶颈是 BF16 的选择而不是打包本身。

### C2

- **B（折叠）**：折叠后的权重与参数化算出的权重相差 ≤ 1 ulp；不编译的版本输出逐位相同（若权重差 1 ulp，则 rel RMS ≤ 1e-6）。
- **C（布局）**：不编译的版本对 A 的 rel RMS ≤ 1e-4 且 MCD ≤ 0.01 dB（TF32 卷积换算法时的正常浮点差异量级）。
- **D（FP16）**：全部有限；MCD(D, A)（同一激励噪声）均值 < 0.1 dB；ASR：完全一致段数 ≥ min(A, A 换种子) − 1，合并 WER ≤ max(A, A 换种子) + 0.5 个百分点。
- 组合方案用其中最不精确的那一项的门槛（含 D 用 D 的，含 C 用 C 的，只有 B 用 B 的）。
- **性能（原登记）**：(a)+(b) 合计（BC，没有 BC 时用 B、C 之和）在微基准里省下的时间 < code2wav 时间（flow + HiFT 的 E）的 3% → 不做端到端；(c) D 过质量门槛且 HiFT 时间在 3 个计时遍**每一遍**都 −20% 以上 → 端到端候选。flow 的 E 取 C1 的 D（同一批形状）；C1 没跑成时用实验 9/10 记录的服务内 flow 时间（标为 Derived）。
- **预测（Estimated，不作为判定）**：B −3% 到 −10%（HiFT 时间），C −5% 到 −15%，D −20% 到 −35%，D 的 MCD 在 0.1 dB 以内。

### 共同规则

微基准的任何改善都只写成“局部 GPU 时间”，不写成端到端收益；失败、报错、过不了门槛的方案都保留记录和原始日志；每个数字注明组、遍数和范围。

## 6. GPU 运行命令和时间

在 the 8-GPU node 上、Job 454290 内（ssh 进入作业的 cgroup），只在父会话分配的空闲卡上运行；脚本不申请资源、不碰其他卡：

```bash
cd ~/workspace/omni-ppv   # 本分支在集群上的 worktree（这些文件提交并拉取之后）
bash scripts/microbench/run_c1_c2.sh --gpu-c1 <i> --gpu-c2 <j> \
    --out ~/workspace/omni-serving-research/results/experiments/<日期>_ppv_c1_c2_run01 [--optional-arms]
```

- C1（卡 i，按顺序 REF → D → P0 → P1 → D_ng）：约 55–70 分钟；`--optional-arms` 再加 P2、P1h 约 20 分钟。
- C2（卡 j，与 C1 同时）：约 15–25 分钟；之后 ASR 服务在卡 j 上启动并转写两套质量集，约 10–15 分钟；最后生成 `c1_microbench.json`、`c2_microbench.json`。
- 合计约 1.2–1.5 小时墙钟、约 1.7–2.1 GPU-时（Estimated）。原登记的“各约 0.5 小时 × 1 卡”偏低，原因是 3 个计时遍 + 主组每批约 1 秒的 flow 时间 + 每个方案的冷编译和 graph 捕获。
- C2 的组合方案（BC、BD、CD、BCD）只在单项过门槛后另跑一次：`c2_hift.py run --arms A,<组合>`，约 15 分钟。
- 结果：只把 `c1_microbench.json`、`c2_microbench.json`、两个 `*_quality.json`（去掉音频路径以外的东西都是数字）拷进 `research/next_steps/results/`；音频和 trace 留在数据目录，不提交。

## 7. 风险

| 风险 | 后果和处理 |
|---|---|
| varlen（FA2）在这个 torch 2.13 + sm_86 的构建里跑不起来 | 能力探测会先报出来；P0/P1 记为“不适用于本机”，不算性能失败 |
| P1 的 `fullgraph=True` 编译失败或计时遍里反复重编译 | 记录失败和重编译次数；不把预热时间混进 E |
| BF16 打包的音质过不了 C1-Q | 预期内的负面结果；P1h 用来判断是不是精度造成的 |
| 质量集是 TTS 语音重新分词后再合成，不是 talker 生成的 token | 质量结论只对“同一 flow/声码器在两种实现下的差别”成立；时间不受内容影响 |
| channels-last 原型每次调用都要把权重转成 NHWC | 可能抵消收益；编译版本可能融合掉；如实报告 |
| ASR 服务起不来 | 质量门槛记为“未评估 = 未通过”，性能数字照报但不下 go 结论 |
| 两个微基准同时跑在同一节点的两张卡上 | 共享 CPU/PCIe；两者都是 GPU 受限，影响小；只在没有 8 卡实验时运行 |
