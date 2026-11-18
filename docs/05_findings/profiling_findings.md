# GPU 时间线的结论（kernel 级）

本文汇总 torch profiler 和 Nsight Systems 窗口给出的 GPU 忙碌、空闲和重叠情况，以及从中得到的测量方法上的注意事项。延迟和吞吐见 `baseline_results.md`，研究问题的回答见 `research_conclusions.md`。

标注：**[实测]** 直接测量；**[计算]** 由实测数据算出；**[假设]** 尚未验证；**[未知]** 还没有答案。开着 profiler 的阶段只用来看时间线，延迟不作基线：x8 中开 profiler 后端到端 ÷ 语音秒高约 5%。

## 怎么算的

- 忙碌时间是一个窗口内所有 kernel、显存拷贝和 memset 区间的并集，重叠部分只算一次（`analysis/intervals.py`）。kernel 时长直接相加的结果只用来显示重叠程度。
- 每个 torch profiler trace 文件对应一个进程。MiniCPM-o 的语音流水线有两个进程：pipeline 进程（预处理、编码器、thinker、decode）和 talker 进程（talker、code2wav）。按进程分别计算忙碌时间，再计算两个进程同时有 GPU 工作的时间（`analysis/torch_trace.py`）。
- 没有 MPS 时，不同进程的 kernel 在 GPU 上轮流执行，不会真正同时运行。所以“两个进程同时有工作”的时间是争用，不是并行。
- “忙碌”只表示 GPU 上有 kernel 在执行，不表示 SM 被占满。这次没有测 SM 占用率或显存带宽利用率。

## A6000 上的两个窗口

| 窗口 | 负载 | 长度 | 忙碌比例 | kernel 时长相加 ÷ 并集 | 空闲间隙 | 最长间隙 |
|---|---|---|---|---|---|---|
| x4，`--text-only` | W4 10 s 视频，并发 4 | 3.8 s | 15% | 1.00 | 10,337 个，9 个超过 100 ms | 324 ms |
| x8，默认语音流水线 | W3b，并发 4 | 9.8 s | 93% | 1.43 | 239,744 个，97% 短于 10 µs | 4 ms |

**视频理解（x4）**
- **[实测]** GPU 只有 15% 的时间在执行 kernel，空闲集中在少数几段长间隙（9 段超过 100 ms），与服务端事件中预处理占 95% 的时间一致（`baseline_results.md` 第 3 节）。
- **[实测]** 执行的 kernel 主要是 bf16 tensor core GEMM（编码器和 thinker）。GPU 在等 CPU 预处理，不是在等其他 GPU 工作。

**语音输出（x8）**
- **[实测]** GPU 93% 的时间有 kernel 在执行，最长空闲间隙只有 4 ms。
- **[实测]** 按进程分：pipeline 进程（thinker）忙 7.47 s，talker 进程忙 4.82 s，两者同时有工作 3.21 s，占窗口的 33%。
- **[实测]** thinker 进程耗时最多的是 bf16 tensor core GEMM（64×64 分块，前四种直接相加约 7.0 s），gemv 只有 0.21 s。
- **[实测]** talker 进程是大量很小的 kernel：gemvx 调用 51,900 次共 0.45 s，平均约 9 µs 一次；采样用的融合 argmax kernel 调用 1,865 次共 0.48 s。
- **[计算]** talker 进程的工作由大量短 kernel 组成，更容易受 kernel 启动开销和调度间隙影响。thinker 是较大的 GEMM。两者在同一张卡上轮流执行，互相拉长对方的执行时间。这与拆分部署在中等负载下的收益方向一致（`baseline_results.md` 第 7 节）。
- **[假设]** 拆分后两张卡各自的忙碌情况还没有用 profiler 看过；采样利用率是 thinker 卡 75%、talker 卡 64%。

## 多卡时的采样利用率（没有 profiler）

| 实验 | 布局 | 负载 | 各卡采样利用率 |
|---|---|---|---|
| x7a | 两个完整副本 | W3b | 92% / 92% |
| x7b | thinker / talker 拆开 | W3b | 75% / 64% |
| x7c | thinker + talker 副本 r0 / talker 副本 r1 | W3b | 92% / 27% |
| x9（Qwen3-Omni） | thinker TP=2 在两张卡上，talker 在第三张卡上 | W3b | 42% / 35% / 76% |
| x9（Qwen3-Omni） | 同上 | W1（只有文本） | 77% / 81% / 1% |

- **[实测]** Qwen3-Omni 在语音输出时，最忙的是 talker 所在的卡（76%），thinker 的两张卡都不到一半；只输出文本时正好相反。
- **[计算]** 对 Qwen3-Omni 来说，这个 3 卡布局在语音负载下给 thinker 分了两张卡，而负载主要落在 talker 那一张上。是否值得把更多 GPU 分给 talker，需要做放置对比（不在这次的范围内）。

## Spark 上的功能验证（GB10，不与 A6000 比较）

- **[实测]** s6：W3b 的 20.5 s 窗口里 GPU 忙 97%，kernel 时长相加 26.1 s，超过窗口本身；两个进程约 6.3 s 同时有工作。
- **[实测]** s8：Nsight Systems 默认只在 graph 层面记录 CUDA graph，graph 里的 kernel 不出现在 KERNEL 表里。只读 KERNEL 表时，GPU 看起来只忙了 6%；把 321 次 graph 执行（每次 66–111 ms，即一次解码步）算进去后是 99%。

## 测量方法上的注意事项

1. **用 Nsight 时必须把 CUDA graph 算进去。** 只读 KERNEL 表会把使用 CUDA graph 的 decode 当成空闲（6% vs 99%）。解析器已经改正（`analysis/nsys.py`），并加了测试。torch profiler 逐个记录 graph 里的 kernel，没有这个问题。
2. **kernel 时长不能直接相加。** 同一张卡上的两个进程会重叠记录，相加的结果可能超过窗口长度（x8：1.43 倍）。要用区间并集，并单独报告进程间的重叠。
3. **采样利用率只看是否有 kernel 在跑。** 在这些窗口里，它和 kernel 级的忙碌比例接近（x4：整个阶段的采样均值 19%，并发 4 那段 profiler 窗口 15%，两者不是同一时间段；x8：都约 93%），但它说明不了 SM 是否被占满，也看不出两个进程是否在争用。
4. **profiler 本身有开销。** 开着 torch profiler 时，语音输出的端到端 ÷ 语音秒高约 5%。导出 trace 需要等约 20 秒，每个进程写一个文件，文件名模板里要用 `{stage}`，否则会互相覆盖。

## 还没有做的

- 拆分部署下两张卡各自的 kernel 时间线，以及跨卡传递在时间线上的位置。
- SM 占用率和显存带宽（Nsight Compute）。这次没有用，因为它会大幅放慢执行，在多进程流水线上意义有限。
- 崩溃前后的时间线。z1（`CUDA_LAUNCH_BLOCKING=1`）只能给出出错的 kernel，给不出时间线。
