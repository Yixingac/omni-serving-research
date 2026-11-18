# 主线 C：GPU Compiler / Runtime 优化机会

> 2026-11-18。方法是 **Profiling → Root Cause → Optimization Hypothesis → Prototype → Correctness → Performance Validation**；本轮完成了前两步（加上第三步的假设），没有写任何 kernel，也没有新的 GPU 运行。
> 代码位置都指 sglang-omni `520b49ee`（下文 `mo/` = `sglang_omni/models/minicpm_o/`，`t2w/` = `mo/components/token2wav/`），SGLang `e00930c5`（0.5.21）。模型维度来自 HF 仓库 `openbmb/MiniCPM-o-4_5` 固定版本的 `config.json`、`assets/token2wav/flow.yaml`。
> 证据等级：Measured / Derived / Estimated / Hypothesis / Unknown。**本文所有“收益”都是 Estimated 或 Hypothesis，没有一个是测出来的加速。**

## 0. 结论先行

| 排名 | 方向 | 类型 | 一句话 |
|---|---|---|---|
| 1 | **Flow DiT 的形状特化：变长打包（varlen）注意力 + 打包后的 CUDA graph** | Compiler/Kernel + Runtime | code2wav GPU 时间里最大的单个 kernel 是带布尔 mask 的 SDPA 退化成的 mem-efficient 注意力（19%，估计占用率 33%），加上为它构造 mask/布局的融合 kernel（6%）；长短混合时还有 15–28% 的补齐。上游已有一条默认关闭的打包路径，可以先零代码 A/B |
| 2 | **HiFT 的编译期优化：weight-norm 常量折叠 + channels-last 布局 + 半精度（带音质门槛）** | Compiler | HiFT 约占 code2wav GPU 时间的 17–23%，全程 FP32（TF32 卷积）、每次调用重算 82 个 weight-norm、每个卷积前做 NCHW→NHWC 转换（这类转换 kernel 占 code2wav 3%，并发 128 时 3 秒窗口内 4.9 万次） |

两者都在 code2wav 上，因为 code2wav 是语音负载在所有拆分放置下的吞吐瓶颈（实验 5、6、7、9）。thinker 的量化、talker 的每步 CPU 开销、code2wav 的 CUDA graph 覆盖排在后面，理由见第 4 节。

---

## 1. C1：热点清单（Hot Path Inventory）

### 1.1 Thinker（Qwen3-8B 结构：36 层，hidden 4096，32/8 头，head_dim 128）

| 项 | 实测/源码 | 等级 | 位置 |
|---|---|---|---|
| decode 每步 GPU 时间 | 约 24 ms（CUDA graph），每步约读一遍 15–16 GB BF16 权重 → 约 85% 的显存带宽（768 GB/s） | Measured（时间）/ Derived（带宽） | 实验 1、5；`research/01_性能瓶颈/实验05_GPU为什么会等待/README.md` |
| 每步空档 | 同步执行时 3–15 ms（11–23%），提前发射原型降到 ≤ 0.06 ms | Measured | 实验 8 6.6 |
| 同步的原因 | 只要批里有一个语音请求，整批走同步（`lookahead_eligible`），每步 `ids.tolist()` | 源码 | `sglang_omni/model_runner/thinker_model_runner.py:495-540`；`sglang_omni/scheduling/sglang_backend/output_processor.py:45` |
| 隐藏状态 | 每步对每个请求 `clone` 最后一行（每请求 8 KiB D2D，B 次 launch）；结束时 `torch.stack(...).to("cpu")` 阻塞、可分页内存 | 源码 + Measured（实验 8 的 22–24 ms 阻塞拷贝） | `mo/thinker_model_runner.py:139-162, 189` |
| KV | BF16，144 KiB/token；每个 thinker 副本 127,039 token 的池（K、V 各 8.72 GB） | 源码 + Measured（启动日志） | — |
| prefill | W3b 只有约 26 token，约 30 ms；视频/图像请求的 prefill 更长 | Measured | `research/00_当前系统和已有实验说明.md` 1.2 |
| 编码器 | 视觉 SigLIP 式 27 层 × 1152 + 64 query resampler；音频 Whisper-medium + MLP + AvgPool；都在 thinker 进程和卡上、一次一个请求 | 源码 | hot-path 审计 §1.5、§6、§7 |

**判断**：thinker decode 是**显存带宽受限**的 GEMV/GEMM（批小时），对语音负载不是瓶颈，但在 W3L（2+2+4 里 thinker 驻留占 38–42%）、混合负载（实验 19）和长推理的机器人场景里会变重要。编译器能做的是**降低权重字节数**（量化，候选 C），而不是改 kernel。

### 1.2 Talker（Llama 结构：20 层，hidden 768，12 头 MHA，6562 个语音码）

| 项 | 实测/源码 | 等级 | 位置 |
|---|---|---|---|
| 每步时间随批大小 | 批 1/8/16/24/32：1.43/2.00/2.62/3.30/4.01 ms（独占卡） | Measured（SGLang 日志） | `results/p0_k1c5_talker_slots.json` |
| 每步 kernel | 约 180 个，中位时长 1.9 µs；96–97% 来自 CUDA graph；卡只忙 52–60% | Measured（实验 7 profiler 窗口） | 本轮 trace 统计 |
| kernel 构成 | bf16 wmma GEMM 22–27%、FlashInfer decode 注意力 14–17%、Triton 融合的采样（argmax/log/clamp）9–10%、cuBLAS GEMM 8–13% | Measured | 同上 |
| 主机侧 | 异步提前发射写死为开；采样在 GPU 的“采样 graph”里；每步仍有 3 次 `copy_`、1 次 `fill_`、1 次 `clone` 围着 graph 重放 | 源码 | `mo/talker_model_runner.py:252-262` |
| 名额 | 32（graph 也只到 32） | 源码 + Measured | `mo/stages.py:144`；`scheduling/generation_batch_policy.py:217-219` |
| 准入时 | 隐藏状态经 SHM 到 CPU，再可分页 H2D、eager 的 2 层 MLP 投影 | 源码 | `mo/components/talker.py:24-53`；`mo/talker_request.py:35-86` |

**判断**：talker 的 kernel 很小、已经全部在 graph 里，GPU 空的 40–48% 来自每步之间的 CPU 工作（Python 调度循环）。这是**运行时**问题（launch/调度开销），不是 kernel 质量问题。

### 1.3 Code2Wav：flow（token → mel）

| 项 | 实测/源码 | 等级 | 位置 |
|---|---|---|---|
| GPU 时间 | 每请求约 150 ms（CUDA event），约占 code2wav 的 77% | Measured | 实验 6 |
| 结构 | 编码器（6+4 层相对位置注意力 + FFN，宽 512，×2 上采样，**无卷积模块**）；估计器是 **DiT**：16 块、宽 512、8×64 头、adaLN、因果卷积块、GELU MLP；输入 320 通道（x‖mu‖speaker‖cond），输出 80 mel | 源码 + HF 配置 | `t2w/dit.py`、`t2w/conformer*.py`；`flow.yaml` |
| ODE | 10 步 Euler；**CFG 让每步在 2B 行上算**（`cat([x,x])`，系数 0.7） | 源码 | `t2w/flow.py:126-133, 227-255` |
| 精度 | flow 参数 FP16、autocast FP16，但 LayerNorm 输出和 mu/cond 是 FP32 | 源码 | `t2w/vocoder.py:177-200`；`mo/components/code2wav.py:387-398` |
| 编译 | 每个 `DiTBlock.forward` 用 `torch.compile(dynamic=True)`；编码器 eager | 源码 | `mo/components/code2wav.py:151-158` |
| 注意力 | DiT 用 SDPA + `(2B,1,1,T)` 布尔 key mask；编码器是显式 matmul→softmax | 源码 | `t2w/dit.py:138-148, 610`；`t2w/conformer_layers.py:118-132` |
| 补齐 | 右补齐到批内最长行；参考音（默认约 6 s，约 300 帧）在每个请求里都算 | 源码 + Measured（实验 10：fifo16 补齐 W3b 15%、MIX 28%） | `t2w/flow.py:607-633` |
| 每次 solve 的主机同步 | 2 次 `.tolist()`、1 次 `.item()`、最后的阻塞 `.cpu()`；按行的 Python 循环 | 源码 | `t2w/flow.py:605-606, 620`；`t2w/conformer.py:339`；`mo/components/code2wav.py:468-470` |

### 1.4 Code2Wav：HiFT（mel → 波形）

| 项 | 实测/源码 | 等级 | 位置 |
|---|---|---|---|
| GPU 时间 | 每请求约 45 ms（CUDA event），约占 code2wav 的 23% | Measured | 实验 6 |
| 结构 | 3 个 weight-norm `ConvTranspose1d` 上采样（×8、×5、×3）+ 12 个 ResBlock（72 次 Snake）+ F0 预测 + SineGen（9 谐波）+ iSTFT（n_fft 16，hop 4） | 源码 | `t2w/hift.py:93-303`；`t2w/hift_layers.py` |
| 精度和布局 | **FP32，(B,C,T)**；cuDNN 选 TF32 卷积，每个卷积前做 `nchwToNhwc<float>` 转换 | 源码 + Measured（kernel 名） | `t2w/vocoder.py:201-209`；`mo/components/code2wav.py:458-462` |
| weight-norm | **每次调用重算 82 个**，从不移除 | 源码 | `t2w/hift.py:62-80, 142-155, 192`；`t2w/hift_layers.py:69-92` |
| 批 | 按**完全相同的 token 长度**分组，每种长度单独调用一次（批里 B 种长度就串行调 B 次） | 源码 | `mo/components/code2wav.py:451-467` |
| STFT/iSTFT | 强制在旧式默认流上做，前后各一次跨流等待 | 源码 | `t2w/hift.py:37-52, 212-245` |
| 编译 | 只有 `decode_body` 编译（dynamic），STFT、F0、SineGen eager | 源码 | `mo/components/code2wav.py:187-217` |

### 1.5 Code2Wav 的 kernel 普查（本轮新增，Measured）

数据：`results/c2w_kernel_census_exclusive_c128.json`（实验 7 的 t2_k2_c4、t2_k4_c2 各 3 个区组，code2wav 独占卡，W3b 并发 128，3 秒 profiler 窗口，共 39.5 s kernel 时间、37 万个 kernel）。脚本：`scripts/analysis/next_c2w_kernel_census.py`。

| 类别 | 占 code2wav kernel 时间 | 代表 kernel（启动配置） |
|---|---|---|
| GEMM（FP16 tensor core） | 25.0% | `ampere_fp16_s16816gemm_fp16_256x128...`（grid [2,18,1]，只有 36 个 CTA，A6000 有 84 个 SM）；`cutlass_80_tensorop_f16_s16816gemm_relu_f16_128x256...` |
| 注意力 | 18.9% | `fmha_cutlassF_f16_aligned_64x64_rf_sm80`（PyTorch mem-efficient），平均 963 µs/次，grid [18,8,14]（约 1152 帧、8 头、14 行 = 7 个请求 × CFG），128 寄存器/线程，估计占用率 33% |
| Inductor 融合的小 kernel | 15.6% | `triton_poi_fused_gelu_view_9`（单独的 GELU，没有和 GEMM 融合）、LayerNorm/Mish/split |
| FP32/TF32 卷积（HiFT） | 9.9% | `sm86_xmma_fprop_implicit_gemm_tf32f32...nchw`，占用率 17% |
| FP16 卷积（DiT 因果卷积） | 9.7% | `sm86_xmma_fprop_implicit_gemm_f16f16...nhwc` |
| 注意力前的 mask/布局准备 | 6.4% | `triton_poi_fused__scaled_dot_product_efficient_attention__..._expand_..._where_3` |
| Snake + weight-norm（HiFT） | 4.5% | `triton_poi_fused__weight_norm_interface_..._sin_...` |
| 布局转换 | 3.0% | `cudnn::...::nchwToNhwcKernel<float,...>`，4.9 万次，平均 22 µs |
| 其他（逐元素、拷贝/转型、softmax、归约） | 7% | 编码器 `cunn_SoftMaxForward` 1.3% |

- **CUDA graph 只占 0.3%**（并发 128）；同一类统计在并发 8（2+6 同卡，`results/c2w_kernel_census_shared_c8.json`）是 24.6%。
- 按名字推断的子模型：flow 75%、HiFT 17%、无法归类 7%；和实验 6 的 CUDA event 拆分（77%/23%）一致，说明归类大体可信（无法归类的多半属于 HiFT）。
- 局限：profiler 窗口只有 3 秒；没有记录张量形状，算术强度只能由 grid 和模型维度推算；kernel 时长在 profiler 下可信，但 CPU 侧的空档不可信。

### 1.6 视频 / 图像 / 音频输入

| 项 | 实测/源码 | 等级 | 位置 |
|---|---|---|---|
| 视频请求时间分布 | 10 秒视频：87% 在 CPU 预处理（约 570 ms），GPU thinker 约 70 ms；每个副本一次一个；GPU 只忙 8–15% | Measured（单次运行） | 实验 14 |
| 解码 | decord，CPU，1 fps、最多 64 帧，每帧带约 1 秒音频 | 源码 | hot-path 审计 §6 |
| 图像切片和缩放 | 在 GPU 上，用自定义 Triton 双三次插值；形状全是 `constexpr`，**每种新图像尺寸编译一次** | 源码 | `mo/components/image_processing.py:62-69` |
| 编码器输出 | 每次缓存未命中都阻塞地拷回 CPU（可分页） | 源码 | `mo/stages.py:73-77, 92`；`sglang_omni/scheduling/stage_cache.py:33-48` |
| 音频 | `np.interp` 线性重采样（float64）；CPU 上 `torch.stft` 算 mel，每段先补到 30 秒 | 源码 | `pp/audio.py:191-195`；HF `processing_minicpmo.py:1456-1463` |

**判断**：视频路径的瓶颈是 CPU 和串行化（每副本一个预处理位），先是**运行时并行**（`--preprocessing.factory.max_concurrency`、解码线程）的问题，再是编译器问题（Triton resize 的形状特化导致反复编译、可分页拷贝）。

---

## 2. C2：Compiler 和 Runtime 分开列

### 2.1 Compiler / Kernel 候选

| 候选 | 依据（实测或源码） | 是否值得写 kernel |
|---|---|---|
| K1 DiT 注意力：布尔 mask → 变长（varlen）FlashAttention，去掉补齐 | 注意力 19% + mask 准备 6%；占用率 33%；补齐 15–28% | **先不写**。上游已有 `varlen_attn` 打包路径（`t2w/dit.py:15, 154-175, 611-628`），先测它；只有它在 sm_86 上不可用或明显慢，才考虑 Triton 变长注意力 |
| K2 HiFT weight-norm 常量折叠 | 82 次/调用，融合在 Snake kernel 里（4.5% 的一部分） | 不需要 kernel：加载后做一次 `remove_parametrizations`，是图级的常量折叠 |
| K3 HiFT 布局传播（channels-last）和半精度 | TF32 卷积 10% + `nchwToNhwc` 3%；占用率 17% | 不需要 kernel：让 cuDNN 选 FP16 NHWC 的 tensor-core 卷积；需要音质门槛 |
| K4 DiT 输入投影的循环不变量外提 | 320 个输入通道里 240 个（mu、speaker、cond）在 10 步之间不变，每步都重新打包、重新过 `in_proj`；speaker 每步用 `einops.repeat` 铺满 T | 不需要 kernel：把 `in_proj` 拆成 `W_x·x + (W_rest·[mu‖spk‖cond])`，后一项每次 solve 算一次。占比小（估计 < 2% 的 flow 时间，Estimated），但是一个干净的编译器例子 |
| K5 adaLN 的冗余 GEMM | 每块的 adaLN Linear 作用在 (2B,1,512) 上，各行完全相同（t 对所有行一样）；每步 16 次 | 不需要 kernel：先算一行再广播 |
| K6 GEMM + GELU 尾融合 | 单独的 GELU kernel 4.0%（每次约 10.3M 元素、平均 202 µs，估算只有约 200 GB/s，远低于 768 GB/s） | 可以用 cuBLASLt 的 GELU epilogue 或 Triton GEMM；先确认 Inductor 为什么没融合（`dynamic=True`、外部 GEMM） |
| K7 分离的 q/k/v GEMM | DiT `to_q/to_k/to_v` 三次 GEMM、编码器同样 | 打包成一次 GEMM（权重拼接），不需要新 kernel |
| K8 编码器注意力 eager 实现 | `cunn_SoftMaxForward` 1.3%，48 次 × 1.8 ms | 改用 SDPA；收益小 |
| K9 Triton resize 的 constexpr 形状 | 每种新尺寸重新编译 | 把尺寸改成运行时参数或按桶特化；只影响图像/视频首包延迟 |
| K10 thinker 量化（W4A16/W8A16 Marlin） | decode 带宽受限（约 85% 带宽） | 用 SGLang 现成的 Marlin kernel，不写新 kernel（候选 C） |

**不建议**：重写 GEMM（25%，已是 cuBLAS/CUTLASS 的 tensor-core kernel）和 cuDNN 卷积本身；它们的低效来自形状（小 grid、补齐、FP32），应该改形状和精度，而不是重写库。

### 2.2 Runtime / Scheduling 候选

| 候选 | 依据 | 已证实 / 未证实 |
|---|---|---|
| R1 code2wav CUDA graph 覆盖（分桶） | 155 个形状，批 11–16 没有，批 2–10 的帧数上限很窄；带默认参考音时，批 1 只覆盖 ≤ 约 14.5 s 的回答、批 2 ≤ 约 9.4 s（Derived）；并发 128 时 graph 只占 0.3% | 已证实 graph 很少命中；已证实高负载时收益小（GPU 忙，实验 6、10 H10e）；低并发时每帧省约 16%（实验 6，Derived） |
| R2 talker 每步 CPU 开销 | GPU 忙 52–60%，每步约 180 个小 kernel 已在 graph 里 | 已证实空档存在；“多步连续解码”在 SGLang 0.5.21 有参数（`num_continuous_decode_steps`，`python/sglang/srt/arg_groups/fields/schedule.py:194`），但 sglang-omni 的 `OmniScheduler` 里没有引用，是否生效 **Unknown** |
| R3 thinker 提前发射（语音） | 实验 8：decode 空档消失，端到端 −2..−5% | 局部收益已证实，端到端未证实 |
| R4 thinker 结束时的阻塞拷贝 | 实验 8：22–24 ms、每 5 步一次 | 现象已证实；旁路复制的正确性未通过（实验 11）→ P2 |
| R5 HiFT 按长度逐个调用 | 批里 B 种长度就调 B 次 | 源码已证实；收益未测 |
| R6 STFT 在默认流上的跨流同步 | 每次 HiFT 两次跨流等待 | 源码已证实；它也和 talker/HiFT 重叠导致崩溃的问题（实验 18）有关，改动要谨慎 |
| R7 code2wav 组批方式 | 实验 10：code2wav 饱和时 length8 +30% | 局部已证实，完整服务未证实 → P1 |
| R8 预处理并发 | 实验 14：每副本一个预处理位 | 现象已证实；`max_concurrency` 的效果未测 |

**实验 8、10、11 的教训**：三个优化都在局部指标上测到了改善（每步空档、code2wav 每秒语音的 GPU 时间），但都没有（或还没有）转化成端到端收益，原因是**它们不在当时的关键路径上**（实验 8：thinker 只占语音请求一半时间；实验 10：完整服务没饱和）。所以 C 主线的每个候选都要先回答“它在不在关键路径上”，并且做端到端验证。

---

## 3. C3：三个重点候选

### 候选 A：Code2Wav 动态形状 + CUDA graph

- **graph 的捕获条件**：flow 每个 Euler 步一个 graph（DiT over 2B 行 + guidance + 更新 x），每次 solve 重放 10 次；编码器、HiFT、参考音准备不在 graph 里（`t2w/flow_cuda_graph.py`）。
- **为什么回退到 eager**：批大小必须完全匹配、帧数必须 ≤ 该批的最大桶；批 11–16 没有桶，而 `max_batch_size=16`（`mo/config.py:144`）；启动时显存余量低于 3 GiB 的形状会被跳过（`t2w/flow_cuda_graph.py:80, 155-162`）。
- **按长度分桶能否同时改善补齐和 graph 复用**：能改善一部分。实验 10 的按长度组批把补齐降到 5–10%，但 graph 命中仍只有 4–18%，因为限制是“批大小 × 帧数”的表太窄，不是长度差。
- **代价**：155 个 graph 共用一个显存池（按最大的 B=8、784 帧定），静态输入约 456 MiB（Derived）；预热 620 次 DiT 前向。
- **更好的形状特化**：把“批大小 × 最长帧数”换成“总帧数”（打包）为键——这正是候选 B 的打包路径；上游有 `packed_dit_cuda_graph_capture_shapes`（`mo/config.py:125-127`）。
- **结论**：graph 本身对高负载吞吐的帮助有限（GPU 已经忙），对低并发（机器人单会话）的延迟有帮助（每帧约 16%）。**不单独立项**，并入候选 B（打包后的 graph），并在 R 主线的低并发延迟实验里测。

### 候选 B：Flow / HiFT 的 kernel 优化

最耗时的具体 kernel 是 DiT 的注意力（`fmha_cutlassF_f16_aligned_64x64_rf_sm80`，19%）：
- **为什么是它**：DiT 给 SDPA 传了布尔 key-padding mask（`t2w/dit.py:610`），有显式 mask 时 PyTorch 的 flash 后端不可用，于是退到 mem-efficient（CUTLASS fmha）——kernel 名证实了这一点（Measured）。
- **计算还是访存受限**：按 grid 推算一次调用约 B=14、T≈1152、H=8、D=64，FLOP ≈ 4·B·H·T²·D ≈ 38 GFLOP，963 µs → 约 40 TFLOPS（Derived，含补齐的无效计算）。对 head_dim 64、长序列的注意力，这是计算受限的形状；估计占用率 33%（寄存器 128/线程）。是否离峰值很远需要用 ncu 或微基准确认（集群上没有 ncu；Spark 上有，但 GB10 的数字不能和 A6000 比）。
- **补齐**：注意力的计算量随 T_max² 增长，补齐的浪费在注意力上比在 GEMM 上更大；MIX 的 fifo16 补齐 28%（按帧），按平方算的浪费更高（Derived）。
- **已有替代**：打包路径 `enable_flow_variable_length=True` 用 `torch.nn.attention.varlen.varlen_attn`（bf16），配 `enable_flow_block_compile=True` 才编译。上游注释说“编译过的稠密 DiT 比 eager 的打包路径快，即使在长短混合的批上”（`mo/config.py:152-160`）——但这句话比较的是**eager** 打包路径，**编译后的打包路径没有被比较过**（源码注释，Hypothesis 待测）。
- **适不适合写 Triton**：先不写。顺序是：(1) 测现成的打包路径（编译开/关）；(2) 如果 `varlen_attn` 在 sm_86 + torch 2.13 上不可用或比 mem-efficient 慢，再写一个 Triton 变长注意力（in-kernel 按 cu_seqlens 截断，不物化 mask），这是一个对 GPU compiler 背景很合适的原型；(3) GEMM 不重写。

HiFT 一侧最有价值的不是单个 kernel，而是**整段的精度和布局**（第 4 节推荐 2）。

### 候选 C：量化

| 对象 | 现状 | A6000（sm_86）能用什么 | GB10 多了什么（只是硬件能力，未测） | 收益假设 | 质量验证 |
|---|---|---|---|---|---|
| Thinker 权重 | BF16；sglang-omni 只通过通用 `engine.*` 透传 `quantization`，MiniCPM-o 的配置一个都没设 | `gptq_marlin`/`awq_marlin`（W4A16、W8A16）；`w8a8_int8`（要 int8 检查点）；`fp8` 在 sm_86 上**静默变成 Marlin W8A16 仅权重**（`fp8_utils.py:2336-2342`）；`w8a8_fp8`、`modelopt_fp8` 需要 sm89，被拒 | 原生 W8A8 FP8、NVFP4 | decode 带宽受限：W4A16 把每步读的权重从约 16 GB 降到约 4–5 GB，每步可能从 24 ms 降到约 8–10 ms（Estimated，按带宽线性外推，忽略反量化开销） | 文字输出一致性（贪心下与 BF16 的 token 一致率）、talker 读到的隐藏状态分布变化（talker 依赖 thinker 的隐藏状态，这是 omni 特有的风险）、最终语音的 ASR 回读一致性 |
| Thinker KV | BF16 | `fp8_e4m3/e5m2` KV 没有能力检查，FlashInfer、Triton 都能用；默认缩放因子 1.0 | nvfp4/mxfp8 KV | 每 token 144 → 72 KiB，同样显存放 2 倍上下文（对机器人长会话、多模态上下文有意义）；对 decode 速度影响小 | 长上下文下的输出漂移；缩放因子为 1.0 时的溢出风险 |
| Talker | BF16，0.19B 层参数 + 117M 文本嵌入 | 同 thinker 的方法理论可用，但嵌入、投影、`head_code` 不能量化 | 同上 | 小：talker 的瓶颈是每步 CPU 开销，不是权重带宽（1.2 节） | — |
| Code2Wav flow/HiFT | flow FP16、HiFT FP32；只有 fp32/fp16/bf16 三种精度可选，没有量化路径 | 半精度（不是量化） | — | 见推荐 2 | 音质（见下） |

**结论**：A6000 上能做的是**仅权重量化**（W4A16/W8A16，Marlin）和 **FP8 KV 存储**；W8A8-FP8 在 A6000 上不存在，不能套用。thinker 量化对语音吞吐帮助有限（thinker 不是瓶颈），但对 W3L、混合负载、机器人长推理的 TTFT 和会话容量有意义，排在第 3。

---

## 4. C4：优先级

| 候选 | Evidence | Impact | Feasibility | Engineering Cost | Correctness Risk | Research Value | Reproducibility |
|---|---|---|---|---|---|---|---|
| **B1 DiT 打包变长注意力 + 打包 graph** | 强：kernel 第 1 名 + 补齐实测 | 高：code2wav 是瓶颈；MIX 上补齐 28% | 高：现成开关 | 低（先 A/B）→ 中（Triton） | 中：bf16 与 fp16 数值不同 | 高：形状特化 × 注意力 kernel × 调度（与 P1 交叉） | 高 |
| **B2 HiFT 折叠 + 布局 + 半精度** | 强：kernel 名 + 源码 | 中高：HiFT 占 code2wav 17–23% | 高：PyTorch 层面改动 | 低–中 | **高**：声码器对精度敏感 | 中高：典型的编译期优化（常量折叠、布局分配、精度降低） | 高 |
| C thinker 量化 | 中：带宽推算 | 中：只在 thinker 重的负载 | 中：需要量化检查点 | 中 | 高：隐藏状态传给 talker | 中 | 中 |
| R2 talker 多步解码 | 中：GPU 空 40–48% | 中：只在 talker 是瓶颈的放置（2+1+5、P0） | 低–中：OmniScheduler 未接 | 中 | 低 | 中 | 高 |
| A code2wav graph 分桶 | 中 | 低（高负载）/ 中（低并发延迟） | 高 | 低 | 低 | 低 | 高 |
| K4–K7 小的图级优化 | 源码 | 低（各 < 5%，Estimated） | 高 | 低 | 低 | 中（教学价值） | 高 |
| R8 预处理并发 | 强（实验 14） | 高（视频）/ 无（语音） | 高 | 低 | 低 | 低（工程） | 中 |

### 推荐 1：Flow DiT 的变长打包（varlen）+ 打包 graph

1. **现有证据**：kernel 普查（注意力 19% + mask 准备 6%，估计占用率 33%）；实验 10（fifo16 补齐 W3b 15%、MIX 28%，补齐和每秒语音的 GPU 时间 E 正相关）；实验 6、10（graph 命中 0.3%–18%）。
2. **代码入口**：`mo/config.py:143-160`（`enable_flow_variable_length`、`enable_flow_block_compile`、`packed_dit_cuda_graph_capture_shapes`）；`t2w/dit.py:138-175, 609-628`；`t2w/packed_dit_cuda_graph.py`；`mo/components/code2wav.py:151-158, 387-411`。
3. **Baseline**：默认稠密路径（FP16、DiT 编译、稠密 graph），fifo16 和 length8 两种组批。
4. **最小原型**：不改代码，三种配置：D（默认）、P0（打包、不编译）、P1（打包 + `enable_flow_block_compile`）。先在 CPU 上用 `config resolve` 检查配置能被接受，再在 GPU 上做微基准。
5. **微基准**：单独加载 flow（不起服务），按实验 10 的批事件里记录的真实（行数、各行长度）分布重放 DiT 一步和完整 solve；记录每个 kernel 的时间（torch profiler，`record_shapes=True`）、每秒有效语音的 GPU 时间 E、显存。一张卡、约 30 分钟。
6. **端到端验证**：沿用实验 10 的设计（t2_k4_c2，code2wav 是瓶颈；W3b 和 MIX；fifo16、length8），D 对 P1，3 个区组；再在 t2_k2_c4 上做开环（与 P1 实验合并）。
7. **正确性 / 质量门槛**：同一个服务里同样的语音编码 → 波形长度一致；P1 对 D 的波形差异不超过 D 自己重复两次的差异（实验 10 的规则，能量相对差和均方根差）；另加客观指标：Whisper 回读文本与 thinker 文本的一致率、mel 倒谱失真（MCD）对 FP32 参考。任何一项超过事先定的门槛就停止性能结论。
8. **预期收益和依据**（Estimated）：E 下降上界 ≈ 补齐比例（W3b 约 15%、MIX 约 28%）加上注意力 kernel 本身的效率差。如果 varlen FlashAttention 在 sm_86 上比 mem-efficient 快 1.3–2 倍（文献常见范围，未在本机测），19% 的注意力份额再省 4–10%。但 length8 组批已经把补齐降到 5–10%，两者的收益会重叠。端到端：只在 code2wav 饱和时可见。
9. **何时放弃**：微基准里 P1 的 E 不比 D 低 ≥ 5%（W3b 和 MIX 都不行），或 `varlen_attn` 在 sm_86 上不可用且 Triton 原型在微基准里追不上 mem-efficient。

### 推荐 2：HiFT 的编译期优化

1. **现有证据**：kernel 普查（TF32 卷积 10%、`nchwToNhwc<float>` 3%、Snake+weight-norm 4.5%，HiFT 合计约 17%）；实验 6（HiFT 每请求约 45 ms，约 23%）；源码（FP32、82 个 weight-norm、按长度逐个调用）。上游 issue #2399 在 H20 上把 MiniCPM-o 的 code2wav 从 FP32 eager 改成 FP16 + compile，吞吐从约 34 到 80.7 语音秒/秒（不同版本、不同 GPU，只说明方向；`docs/01_research/prior_work.md` 1.2）。
2. **代码入口**：`t2w/vocoder.py:201-209`（HiFT 不转精度）、`t2w/hift.py:62-80, 142-155, 192`（weight-norm）、`mo/components/code2wav.py:187-217`（compile）、`:451-467`（按长度调用）、`:458-462`（`.float()`）。
3. **Baseline**：默认 HiFT（FP32、weight-norm、编译 `decode_body`）。
4. **最小原型**（一步一个，每一步单独测）：(a) 加载后 `remove_parametrizations`（数值上应与原来逐位或在 1 ulp 内一致）；(b) `channels_last`（对 1D 卷积用 conv2d 视图，flow 侧已有 `sglang_omni/utils/channels_last_conv.py` 可参考）；(c) FP16 或 BF16 的 HiFT 主体，STFT/iSTFT、F0、SineGen 保持 FP32。
5. **微基准**：单独加载 HiFT，按实测的 mel 长度分布（W3b、W3L）跑，记录每秒语音的 GPU 时间和 kernel 构成。
6. **端到端验证**：同推荐 1 的 t2_k4_c2 设计，只换 HiFT；3 个区组。
7. **质量门槛**（声码器最敏感）：(a) 必须逐位或 1 ulp 内一致；(b) 只改布局，应在浮点误差内；(c) 半精度要过客观指标——对 FP32 参考的 MCD、PESQ/STOI 类指标、Whisper 回读一致率，门槛事先定（例如 MCD 增量 < 0.1 dB、回读文字完全一致的比例不下降）。听感抽查作为补充，不作为门槛。
8. **预期收益和依据**（Estimated）：(a) 约 1–3%（weight-norm 的计算融合在 Snake kernel 里，占 4.5% 的一部分）；(b) 去掉约 3% 的布局转换；(c) FP16 tensor core 的峰值是 TF32 的 2 倍，卷积部分（约 10%）最多减半。合计 code2wav GPU 时间约 −7% 到 −12%。
9. **何时放弃**：(c) 过不了质量门槛就只保留 (a)(b)；(a)(b) 合计在微基准里 < 3% 就不做端到端。

## 5. 和其他主线的关系

- 推荐 1 和 P1（组批方式）作用在同一个“补齐”上，必须在同一个设计里比较，不能把两者的收益相加。
- talker 的每步 CPU 开销（R2）和 P0 是同一个问题的两面：P0 测“批更大能不能摊薄”，R2 是“每步本身能不能更便宜”。
- 机器人场景是低并发、低延迟的：那里 CUDA graph 覆盖（候选 A）、HiFT 的单请求延迟、thinker 的 TTFT 比吞吐更重要，见 `robotics_application_intro.md` 和 `robotics_experiment_plan.md`。
