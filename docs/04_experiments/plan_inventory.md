# 全部实验计划清单（审核用）

由 `scripts/analysis/list_plans.py` 生成，2026-11-16 22:35。

- **实验内容**：每个实验的内容都直接取自仓库里的计划文件和实验配置，没有手写。第二、五节是例外，是对着代码和计划文档写的。
- **运行记录**：来自集群上每个作业的编排器状态（`orchestrator/state.json`）、手动步骤日志、Slurm 记账和每次运行的 `result.json`。Spark 的运行记录来自 Spark 上的 `result.json`。
- **result.json 的状态**：`ok` 只表示实验流程正常结束。服务是否崩溃要看“服务退出码”和“跑了几档”。
- **第二批**：Job 454290 还在运行，这里的状态是生成时的快照。
- **节点名**：用中性名称代替（组内节点 A、第一批节点、compute-03）。

## 一、总览

| 计划 | 怎么运行 | 计划文件 | 实验数 | 状态 |
|---|---|---|---|---|
| Job 409712 自动计划（3×A6000） | 作业启动后由编排器自动运行 | `configs/experiments/plan_a6000_3gpu.json` | 15 | 成功 13，失败 2 |
| Job 409712 追加（plan.d 001–012） | 作业运行中放进 plan.d，由同一个编排器运行；仓库里只保存了 001 | `configs/experiments/followups_a6000_3gpu_001.json` | 27 | 成功 23，失败 2，未运行 2 |
| Job 409712 手动步骤（编排器结束后的崩溃定位） | 编排器结束后，用与编排器相同参数的 srun --exact 在 Job 409712 里逐个运行，每个 1 张卡 | — | 16 | 跑完 14，手动取消 2 |
| 8 卡计划 v1 | 写好但从未提交，被 plan v2 取代 | `configs/experiments/plan_a6000_8gpu.json` | 13 | 未运行 13 |
| plan v2 第一批（8×A6000） | 在第一批节点上作为自动计划运行 | `configs/experiments/plan_a6000_8gpu_v2.json` | 33 | 成功 33 |
| plan v2 第二批（8×A6000，Job 454290） | 第 1 部分由登录节点上的编排器运行，第 2、3、4、6 部分放进 plan.d；第 5 部分（D）还没放入 | `configs/experiments/plan_a6000_8gpu_v2_batch2.json`<br>`configs/experiments/plan_a6000_8gpu_v2_batch2_part2.json`<br>`configs/experiments/plan_a6000_8gpu_v2_batch2_part3.json`<br>`configs/experiments/plan_a6000_8gpu_v2_batch2_part4.json`<br>`configs/experiments/plan_a6000_8gpu_v2_batch2_part5.json`<br>`configs/experiments/plan_a6000_8gpu_v2_batch2_part6.json` | 37 | 成功 17，运行中 1，排队（可运行） 6，排队（等依赖） 3，未运行 10 |
| Spark（GB10）功能验证 | 在个人 DGX Spark 上逐个手动运行；只用于功能验证，不是 A6000 基准 | `configs/experiments/spark/` | 14 | 最后一次运行 ok 14，第一次失败后重跑（s1、s10） 2 |
| 不在任何计划里的配置 | 写好但没有通过计划运行（备选方案或未用上） | — | 2 | 见第六节 |

另外，2026-11-11 在登录节点上用假服务测试过两次编排器的自启动（`test-autostart-*`，计划 `autostart-test`），只检查编排机制，不是实验，没有列入上表。

## 二、测量手段（按代码里的实际做法）

**每个实验都有，不需要在配置里打开：**

1. **客户端请求记录** `raw/requests_<阶段>.jsonl`：每个请求的发送、响应头、首个文本块、首个和最后一个音频块、完成时间（单调时钟，纳秒），输出 token 数、语音秒数、状态和错误；开环还记发送比计划晚了多少（`lateness_s`）。
2. **客户端时间线** `raw/bench_trace.jsonl`：带时钟锚点，用来和服务端事件对时。
3. **GPU 采样** `raw/gpu_samples.csv`：`nvidia-smi --query-gpu=… -lms=200`，每 200 ms 一次，字段是 utilization.gpu、utilization.memory、memory.used、memory.total、power.draw、clocks.sm、pstate。utilization.gpu 是采样周期里有 kernel 在跑的时间比例，不是 SM 占用率。
4. **CPU 采样** `raw/cpu_samples.jsonl`：每 1 s 用 psutil 统计每个服务进程树和客户端自己的 CPU 百分比。从第二批起，一个进程只算给第一个包含它的组，客户端排最后；第一批的客户端数字因此有偏差，已在报告里更正。
5. **服务日志** `logs/server_*.log`：分析脚本从 SGLang 的 `Prefill batch` 行算前缀缓存命中。第二批中期报告里 code2wav 每批的请求数，是从 `minicpm_code2wav_batch` 行临时统计的，还没有写进分析脚本。
6. **环境快照** `logs/env/`，以及服务退出码、重启记录（`result.json` 的 `server_exit_codes`、`server_restarts`）。

**按阶段打开：**

7. **服务端事件**（阶段配置 `"server_events": true`）：阶段开始时对每个服务发 `POST /start_request_profile`（带 run_id 和事件目录），结束时发 `/stop_request_profile`。sglang-omni 的每个阶段进程把每个请求的事件写到 `raw/server_events/<阶段>/<服务>/`，包括进入队列、prefill 开始和结束、模型路径开始和结束、阶段派发和完成、阶段之间的发送和接收、首个流式块。
8. **torch profiler**（阶段配置 `"torch_profile": true`）：发 `POST /start_profile`，带 `enable_torch=True`，同时也记服务端事件。
   - 每个阶段进程各写一个 Chrome trace，路径是 `profiling/torch/<阶段>/<服务>/<stage>_trace_rank0.trace.json.gz`。
   - 记录 CPU 和 CUDA 活动，从开始到结束连续记录，不用 schedule。
   - 没有打开 record_shapes、profile_memory、with_stack、with_flops。这几项只有设置 `SGLANG_TORCH_PROFILER_*` 环境变量才会打开，我们没有设。
   - profiler 会扰动时序，所以打开它的阶段从不当作基线。
9. **Nsight Systems**：只在 Spark 的 s8 上用过，通过服务的启动前缀接入（见 s8）。
10. **拓扑和通信**（Y0，`scripts/slurm/env_topology.sh`）：
    - `nvidia-smi topo -m`、NVLink 状态；
    - 用 `gpu_p2p_check.py` 测点对点拷贝带宽，以及 NVLink 对上的 NCCL all-reduce，放在带超时的子进程里跑；
    - 计时读一遍模型权重文件。


## 三、开了额外采集的阶段（按手段汇总）

| 手段 | 实验 | 阶段 | 负载和档位 | 实验是否运行过 |
|---|---|---|---|---|
| torch profiler | `s5_minicpm_replay_profile_smoke` | `w3b_torchprof` | W3b（文本提问→文本+语音）；档位 2 | 是 |
| torch profiler | `s6_minicpm_torchprof_smoke` | `w3b_torchprof` | W3b（文本提问→文本+语音）；档位 2 | 是 |
| torch profiler | `v2_a_r8_run1` | `w3b_torch_profile_c64` | W3b（文本提问→文本+语音）；档位 64 | 是 |
| torch profiler | `v2_a_s2x6_run1` | `w3b_torch_profile_c64` | W3b（文本提问→文本+语音）；档位 64 | 是 |
| torch profiler | `x4_minicpm_av_text_1gpu` | `w4_video_10s_torchprof` | W4（视频+文本→文本），视频 10 s（素材标签 count_circles）；档位 4 | 是 |
| torch profiler | `x8_minicpm_profiled_1gpu` | `w3b_torchprof` | W3b（文本提问→文本+语音）；档位 4 | 是 |
| nsys | `s8_minicpm_nsys_smoke` | `服务 s0（启动前缀：进程启动 200 s 后采 40 s）` | align；w1_nsys；flush | 是 |
| 服务端事件 | `s10_minicpm_talker_replica_functional` | `w3b_small` | W3b（文本提问→文本+语音）；档位 1, 4 | 是 |
| 服务端事件 | `s13_minicpm_pipeline_replica_functional` | `w3b_small` | W3b（文本提问→文本+语音）；档位 1, 4 | 是 |
| 服务端事件 | `s14_minicpm_code2wav_replica_functional` | `w3b_small` | W3b（文本提问→文本+语音）；档位 1, 4 | 是 |
| 服务端事件 | `s1_minicpm_text_smoke` | `w1_small` | W1（文本→文本）；档位 1, 4 | 是 |
| 服务端事件 | `s2_minicpm_speech_smoke` | `w3b_small` | W3b（文本提问→文本+语音）；档位 1, 2 | 是 |
| 服务端事件 | `s3_minicpm_speech_memtune_smoke` | `w3b_small` | W3b（文本提问→文本+语音）；档位 1, 2 | 是 |
| 服务端事件 | `v2_a_p1` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 4, 8, 16, 32, 64 | 是 |
| 服务端事件 | `v2_a_p2_run1` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 8, 16, 32, 64, 128 | 是 |
| 服务端事件 | `v2_a_p2_run2` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 8, 16, 32, 64, 128 | 是 |
| 服务端事件 | `v2_a_p4_run1` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 16, 32, 64, 128, 256 | 是 |
| 服务端事件 | `v2_a_p4_run2` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 16, 32, 64, 128, 256 | 是 |
| 服务端事件 | `v2_a_r1` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 1, 2, 4, 8, 16, 32 | 是 |
| 服务端事件 | `v2_a_r2` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 4, 8, 16, 32, 64 | 是 |
| 服务端事件 | `v2_a_r4_run1` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 8, 16, 32, 64, 128 | 是 |
| 服务端事件 | `v2_a_r4_run2` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 8, 16, 32, 64, 128 | 是 |
| 服务端事件 | `v2_a_r8_run1` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 16, 32, 64, 128, 256 | 是 |
| 服务端事件 | `v2_a_r8_run2` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 16, 32, 64, 128, 256 | 是 |
| 服务端事件 | `v2_a_s1x3_run1` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 8, 16, 32, 64, 128 | 是 |
| 服务端事件 | `v2_a_s1x3_run2` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 8, 16, 32, 64, 128 | 是 |
| 服务端事件 | `v2_a_s1x7_run1` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 16, 32, 64, 128, 256 | 是 |
| 服务端事件 | `v2_a_s1x7_run2` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 16, 32, 64, 128, 256 | 是 |
| 服务端事件 | `v2_a_s2x2_run1` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 8, 16, 32, 64, 128 | 是 |
| 服务端事件 | `v2_a_s2x2_run2` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 8, 16, 32, 64, 128 | 是 |
| 服务端事件 | `v2_a_s2x6_run1` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 16, 32, 64, 128, 256 | 是 |
| 服务端事件 | `v2_a_s2x6_run2` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 16, 32, 64, 128, 256 | 是 |
| 服务端事件 | `v2_a_s3x5_run1` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 16, 32, 64, 128, 256 | 是 |
| 服务端事件 | `v2_a_s3x5_run2` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 16, 32, 64, 128, 256 | 是 |
| 服务端事件 | `v2_a_s4x4_run1` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 16, 32, 64, 128, 256 | 是 |
| 服务端事件 | `v2_a_s4x4_run2` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 16, 32, 64, 128, 256 | 是 |
| 服务端事件 | `v2_b1_p4` | `mix_closed` | 混合负载（W1（文本→文本） 40%、W3b（文本提问→文本+语音） 40%、W2（语音提问→文本） 20%，每个请求按权重抽取）；档位 16, 32, 64, 128, 256 | 是 |
| 服务端事件 | `v2_b1_r8` | `mix_closed` | 混合负载（W1（文本→文本） 40%、W3b（文本提问→文本+语音） 40%、W2（语音提问→文本） 20%，每个请求按权重抽取）；档位 16, 32, 64, 128, 256 | 是 |
| 服务端事件 | `v2_b1_s2x6` | `mix_closed` | 混合负载（W1（文本→文本） 40%、W3b（文本提问→文本+语音） 40%、W2（语音提问→文本） 20%，每个请求按权重抽取）；档位 16, 32, 64, 128, 256 | 是 |
| 服务端事件 | `v2_b2_p4` | `text_c256` | agent；档位 256 | 是 |
| 服务端事件 | `v2_b2_r8` | `lo_text_c256` | agent；档位 256 | 是 |
| 服务端事件 | `v2_b2_r8` | `sa_text_c256` | agent；档位 256 | 是 |
| 服务端事件 | `v2_b2_s2x6` | `text_c256` | agent；档位 256 | 是 |
| 服务端事件 | `v2b_a_r8_c03` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 16, 32, 64, 128, 256 | 是 |
| 服务端事件 | `v2b_a_s2x6_c03` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 16, 32, 64, 128, 256 | 是 |
| 服务端事件 | `v2b_b1_s3x5` | `mix_closed` | 混合负载（W1（文本→文本） 40%、W3b（文本提问→文本+语音） 40%、W2（语音提问→文本） 20%，每个请求按权重抽取）；档位 16, 32, 64, 128, 256 | 是 |
| 服务端事件 | `v2b_b1_s4x4` | `mix_closed` | 混合负载（W1（文本→文本） 40%、W3b（文本提问→文本+语音） 40%、W2（语音提问→文本） 20%，每个请求按权重抽取）；档位 16, 32, 64, 128, 256 | 是 |
| 服务端事件 | `v2b_b1_s5x3` | `mix_closed` | 混合负载（W1（文本→文本） 40%、W3b（文本提问→文本+语音） 40%、W2（语音提问→文本） 20%，每个请求按权重抽取）；档位 16, 32, 64, 128, 256 | 是 |
| 服务端事件 | `v2b_b1_s6x2` | `mix_closed` | 混合负载（W1（文本→文本） 40%、W3b（文本提问→文本+语音） 40%、W2（语音提问→文本） 20%，每个请求按权重抽取）；档位 16, 32, 64, 128, 256 | 是 |
| 服务端事件 | `v2b_b2l_p4` | `text_c256` | agent；档位 256 | 否 |
| 服务端事件 | `v2b_b2l_r8` | `lo_text_c256` | agent；档位 256 | 否 |
| 服务端事件 | `v2b_b2l_r8` | `sa_text_c256` | agent；档位 256 | 否 |
| 服务端事件 | `v2b_b2l_s2x6` | `text_c256` | agent；档位 256 | 否 |
| 服务端事件 | `v2b_b2l_s4x4` | `text_c256` | agent；档位 256 | 否 |
| 服务端事件 | `v2b_g1a_links` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 16, 32 | 否 |
| 服务端事件 | `v2b_g1a_links` | `w3b_open_abs` | W3b（文本提问→文本+语音）；档位 6, 10, 13, 15 | 否 |
| 服务端事件 | `v2b_g1b_links` | `w3b_closed` | W3b（文本提问→文本+语音）；档位 16, 32 | 否 |
| 服务端事件 | `v2b_g1b_links` | `w3b_open_abs` | W3b（文本提问→文本+语音）；档位 6, 10, 13, 15 | 否 |
| 服务端事件 | `v2b_sb_p4` | `w3b_small` | W3b（文本提问→文本+语音）；档位 1, 2, 4, 8 | 是 |
| 服务端事件 | `v2b_sb_r8` | `w3b_small` | W3b（文本提问→文本+语音）；档位 1, 2, 4, 8 | 是 |
| 服务端事件 | `v2b_sb_s1x7` | `w3b_small` | W3b（文本提问→文本+语音）；档位 1, 2, 4, 8 | 是 |
| 服务端事件 | `v2b_sb_s2x6` | `w3b_small` | W3b（文本提问→文本+语音）；档位 1, 2, 4, 8 | 是 |
| 服务端事件 | `x5_minicpm_agents` | `agents_voice_c4_events` | agent；档位 4 | 是 |
| 服务端事件 | `x7e_minicpm_2gpu_disagg_events` | `w3b_events` | W3b（文本提问→文本+语音）；档位 4, 8, 16, 32 | 是 |
| 服务端事件 | `x7f_minicpm_3gpu_split_talker2` | `w3b_events` | W3b（文本提问→文本+语音）；档位 8, 16, 32, 64 | 是 |
| 服务端事件 | `x8_minicpm_profiled_1gpu` | `w3b_events` | W3b（文本提问→文本+语音）；档位 1, 4, 16 | 是 |

## 四、各计划的实验明细

## Job 409712 自动计划（3×A6000）

### 来源：`configs/experiments/plan_a6000_3gpu.json`

#### `x0_envcheck`
- 计划条目：优先级 100；依赖 无；申请 1 卡、8 核、90 GB；预计 15 分钟；超时 40 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/x0_envcheck.json`（硬件 a6000x1，运行环境 unicorn，素材 —，种子 0）
- 阶段：
  1. `idle` 空闲 1 s
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：运行环境检查：导入 torch/sglang/sglang_omni，记录驱动、GPU、拓扑和包版本。不启动服务。
- 运行记录：编排器：成功（尝试 1 次），11-11 16:16 开始，组内节点 A（3×A6000），GPU 时长 0.06 h；结果 `20261111_x0_envcheck_run01`：ok，1 分钟

#### `x1_assets`
- 计划条目：优先级 95；依赖 `x0_envcheck`；申请 1 卡、8 核、90 GB；预计 35 分钟；超时 90 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/x1_assets.json`（硬件 a6000x1，运行环境 unicorn，素材 —，种子 0）
- 服务 tts：qwen3_tts_0_6b，GPU 0，参数 `--config examples/configs/qwen3_tts_0_6b.yaml --model-name Qwen/Qwen3-TTS-12Hz-0.6B-Base`，启动时限 2400 s
- 阶段：
  1. `assets` 生成合成素材：pool=synthetic-v1，question_tasks=24，task_seed=11，kb_seed=7，passages=6，tts_model=Qwen/Qwen3-TTS-12Hz-0.6B-Base，source_label=synthetic:tts:Qwen/Qwen3-TTS-12Hz-0.6B-Base@5d83992
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：生成合成输入素材：Qwen3-TTS 合成语音提问（答案可核对）和朗读段落；程序生成数圆圈视频。Qwen3-TTS 只是素材工具，不是被测系统。
- 运行记录：编排器：成功（尝试 1 次），11-11 16:19 开始，组内节点 A（3×A6000），GPU 时长 0.13 h；结果 `20261111_x1_assets_run01`：ok，8 分钟

#### `x2_minicpm_text_1gpu`
- 计划条目：优先级 90；依赖 `x0_envcheck`；申请 1 卡、8 核、90 GB；预计 40 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/x2_minicpm_text_1gpu.json`（硬件 a6000x1，运行环境 unicorn，素材 —，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，参数 `--text-only`，启动时限 2400 s
- 阶段：
  1. `smoke_w1` 冒烟检查：W1（文本→文本），3 个请求，检查输出非空
  2. `w1_low` 闭环：W1（文本→文本）；并发 1, 2, 4, 8；每档 40 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 60 s
  3. `w1_high` 闭环：W1（文本→文本）；并发 16, 32, 64；每档 128 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 60 s
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：W1 文本→文本基线：MiniCPM-o 4.5 --text-only，单卡，闭环并发扫描，关闭 profiler。
- 运行记录：编排器：成功（尝试 1 次），11-11 16:19 开始，组内节点 A（3×A6000），GPU 时长 0.18 h；结果 `20261111_x2_minicpm_text_1gpu_run01`：ok，11 分钟

#### `x3_minicpm_speech_1gpu`
- 计划条目：优先级 85；依赖 `x1_assets`；申请 1 卡、8 核、90 GB；预计 60 分钟；超时 150 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/x3_minicpm_speech_1gpu.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b` 闭环：W3b（文本提问→文本+语音）；并发 1, 2, 4, 8, 16；每档 32 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
  3. `w3` 闭环：W3（语音提问→文本+语音）；并发 1, 2, 4, 8；每档 32 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
  4. `w1_on_speech_server` 闭环：W1（文本→文本）；并发 1, 8；每档 32 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：语音输出基线：MiniCPM-o 4.5 默认语音流水线（所有阶段在一张卡上），W3b 文本→语音、W3 语音→语音、W1 文本请求，关闭 profiler。
- 运行记录：编排器：成功（尝试 1 次），11-11 16:27 开始，组内节点 A（3×A6000），GPU 时长 0.17 h；结果 `20261111_x3_minicpm_speech_1gpu_run01`：ok，10 分钟

#### `x4_minicpm_av_text_1gpu`
- 计划条目：优先级 80；依赖 `x1_assets`；申请 1 卡、8 核、90 GB；预计 60 分钟；超时 150 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/x4_minicpm_av_text_1gpu.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，参数 `--text-only`，启动时限 2400 s
- 阶段：
  1. `smoke_w2` 冒烟检查：W2（语音提问→文本），3 个请求，检查输出非空
  2. `smoke_w4` 冒烟检查：W4（视频+文本→文本），2 个请求，检查输出非空
  3. `w2` 闭环：W2（语音提问→文本）；并发 1, 2, 4, 8, 16；每档 40 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2
  4. `w2l` 闭环：W2L（长语音→文本）；并发 1, 2, 4, 8；每档 24 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2
  5. `w4_video_2s` 闭环：W4（视频+文本→文本），视频 2 s（素材标签 count_circles）；并发 1, 2, 4, 8；每档 24 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
  6. `w4_video_5s` 闭环：W4（视频+文本→文本），视频 5 s（素材标签 count_circles）；并发 1, 2, 4, 8；每档 24 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
  7. `w4_video_10s` 闭环：W4（视频+文本→文本），视频 10 s（素材标签 count_circles）；并发 1, 2, 4, 8；每档 24 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
  8. `w4_video_20s` 闭环：W4（视频+文本→文本），视频 20 s（素材标签 count_circles）；并发 1, 2, 4, 8；每档 24 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
  9. `w4_video_10s_torchprof` 闭环：W4（视频+文本→文本），视频 10 s（素材标签 count_circles）；并发 4；每档 8 个请求（另加预热请求：2 个，不计入统计）；停止条件：错误率 > 0.2；**torch profiler**
- 额外采集：服务端事件 无；torch profiler w4_video_10s_torchprof；nsys 无
- 配置里的说明：音频/视频理解基线（文本输出，--text-only）：W2 短语音提问、W2L 长朗读、W4 合成视频（2/5/10/20 秒），用于观察 CPU 预处理和编码器开销。 最后加一个 10 秒视频、并发 4 的 torch profiler 窗口，用来看 CPU 预处理期间 GPU 是否空闲；这一段不作为基线。
- 运行记录：编排器：成功（尝试 1 次），11-11 16:27 开始，组内节点 A（3×A6000），GPU 时长 0.14 h；结果 `20261111_x4_minicpm_av_text_1gpu_run01`：ok，8 分钟

#### `x5_minicpm_agents`
- 计划条目：优先级 75；依赖 `x1_assets`；申请 1 卡、8 核、90 GB；预计 50 分钟；超时 150 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/x5_minicpm_agents.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s
- 阶段：
  1. `agents_text_c1` Agent（合成任务，真实模型请求，录 trace）：12 个会话，并发 1，种子 1，每会话最多 5 步
  2. `agents_text_c4` Agent（合成任务，真实模型请求，录 trace）：24 个会话，并发 4，种子 2，每会话最多 5 步
  3. `agents_text_c8_toollat` Agent（合成任务，真实模型请求，录 trace）：24 个会话，并发 8，种子 3，每会话最多 5 步，模拟工具延迟 对数正态（中位数 0.3 s，σ 0.5）
  4. `agents_voice_c4` Agent（合成任务，真实模型请求，录 trace）：16 个会话，并发 4，种子 11，每会话最多 5 步，语音提问，最后一步语音回答，用户思考时间 对数正态（中位数 2.0 s，σ 0.5）
  5. `agents_voice_c4_events` Agent（合成任务，真实模型请求，录 trace）：8 个会话，并发 4，种子 11，每会话最多 5 步，语音提问，最后一步语音回答；**服务端事件**
- 额外采集：服务端事件 agents_voice_c4_events；torch profiler 无；nsys 无
- 配置里的说明：收集真实 Agent trace：带工具调用的合成任务（答案可核对），文本输入与语音输入两类，不同并发；一组开启服务端请求事件记录。
- 运行记录：编排器：成功（尝试 1 次），11-11 16:30 开始，组内节点 A（3×A6000），GPU 时长 0.06 h；结果 `20261111_x5_minicpm_agents_run01`：ok，3 分钟

#### `x7a_minicpm_2gpu_two_replicas`
- 计划条目：优先级 70；依赖 `x3_minicpm_speech_1gpu`、`x5_minicpm_agents`；申请 2 卡、16 核、180 GB；预计 55 分钟；超时 140 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/x7a_minicpm_2gpu_two_replicas.json`（硬件 a6000x2，运行环境 unicorn，素材 synthetic-v1，种子 0，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b` 闭环：W3b（文本提问→文本+语音）；并发 2, 4, 8, 16, 32；每档 48 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
  3. `replay_voice_dep_x2` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_voice_c4.jsonl`，模式 dependency，速度 1×，会话复制 2 份，会话到达 trace，内容 auto
  4. `replay_text_dep_x4` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_text_c4.jsonl`，模式 dependency，速度 1×，会话复制 4 份，会话到达 poisson:0.5，内容 auto
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：2 卡预算之一：两个独立的完整副本（每卡一个默认语音流水线），客户端按在途请求最少路由。 另外用 x5 的 Agent trace 做依赖感知回放，在 Agent 负载下比较放置（上游的放置对比只用了固定 TTS 负载）。
- 运行记录：编排器：成功（尝试 1 次），11-11 16:42 开始，组内节点 A（3×A6000），GPU 时长 0.33 h；结果 `20261111_x7a_minicpm_2gpu_two_replicas_run01`：ok，10 分钟

#### `x7b_minicpm_2gpu_disagg`
- 计划条目：优先级 69；依赖 `x3_minicpm_speech_1gpu`、`x5_minicpm_agents`；申请 2 卡、16 核、180 GB；预计 55 分钟；超时 140 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/x7b_minicpm_2gpu_disagg.json`（硬件 a6000x2，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0, 1，参数 `--talker.gpu 1 --code2wav.gpu 1`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b` 闭环：W3b（文本提问→文本+语音）；并发 2, 4, 8, 16, 32；每档 48 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
  3. `replay_voice_dep_x2` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_voice_c4.jsonl`，模式 dependency，速度 1×，会话复制 2 份，会话到达 trace，内容 auto
  4. `replay_text_dep_x4` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_text_c4.jsonl`，模式 dependency，速度 1×，会话复制 4 份，会话到达 poisson:0.5，内容 auto
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：2 卡预算之二：拆分放置，thinker 与编码器在 GPU 0，talker 与 code2wav 在 GPU 1（一个服务）。 另外用 x5 的 Agent trace 做依赖感知回放，在 Agent 负载下比较放置（上游的放置对比只用了固定 TTS 负载）。
- 运行记录：编排器：成功（尝试 1 次），11-11 16:52 开始，组内节点 A（3×A6000），GPU 时长 0.33 h；结果 `20261111_x7b_minicpm_2gpu_disagg_run01`：ok，10 分钟

#### `x7c_minicpm_2gpu_talker_replica`
- 计划条目：优先级 68；依赖 `x3_minicpm_speech_1gpu`、`x5_minicpm_agents`；申请 2 卡、16 核、180 GB；预计 55 分钟；超时 140 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/x7c_minicpm_2gpu_talker_replica.json`（硬件 a6000x2，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0, 1，参数 `--config configs/models/minicpm_o_4_5_talker_replica2.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b` 闭环：W3b（文本提问→文本+语音）；并发 2, 4, 8, 16, 32；每档 48 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
  3. `replay_voice_dep_x2` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_voice_c4.jsonl`，模式 dependency，速度 1×，会话复制 2 份，会话到达 trace，内容 auto
  4. `replay_text_dep_x4` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_text_c4.jsonl`，模式 dependency，速度 1×，会话复制 4 份，会话到达 poisson:0.5，内容 auto
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：2 卡预算之三：thinker 一份在 GPU 0，talker+code2wav 进程复制两份（GPU 0、GPU 1），只复制语音生成阶段。 另外用 x5 的 Agent trace 做依赖感知回放，在 Agent 负载下比较放置（上游的放置对比只用了固定 TTS 负载）。
- 运行记录：编排器：失败（尝试 1 次），11-11 17:02 开始，组内节点 A（3×A6000），GPU 时长 0.02 h；失败原因：server_crashed；结果 `20261111_x7c_minicpm_2gpu_talker_replica_run01`：failed，server_crashed，服务退出码 s0=2，1 分钟

#### `x3r_minicpm_speech_1gpu_agent_replay`
- 计划条目：优先级 71；依赖 `x5_minicpm_agents`；申请 1 卡、8 核、90 GB；预计 25 分钟；超时 80 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/x3r_minicpm_speech_1gpu_agent_replay.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s
- 阶段：
  1. `replay_voice_dep_x2` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_voice_c4.jsonl`，模式 dependency，速度 1×，会话复制 2 份，会话到达 trace，内容 auto
  2. `replay_text_dep_x4` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_text_c4.jsonl`，模式 dependency，速度 1×，会话复制 4 份，会话到达 poisson:0.5，内容 auto
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：单卡对照：与 x7a/b/c 相同的 Agent trace 回放阶段，作为 1 卡基准。
- 运行记录：编排器：成功（尝试 1 次），11-11 16:34 开始，组内节点 A（3×A6000），GPU 时长 0.12 h；结果 `20261111_x3r_minicpm_speech_1gpu_agent_replay_run01`：ok，7 分钟

#### `x6_minicpm_replay`
- 计划条目：优先级 60；依赖 `x5_minicpm_agents`；申请 1 卡、8 核、90 GB；预计 45 分钟；超时 120 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/x6_minicpm_replay.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s
- 阶段：
  1. `replay_dep_x1` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_text_c4.jsonl`，模式 dependency，速度 1×，会话复制 1 份，会话到达 trace，内容 auto
  2. `replay_req_x1` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_text_c4.jsonl`，模式 request，速度 1×，会话复制 1 份，会话到达 trace，内容 auto
  3. `replay_dep_x4_poisson` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_text_c4.jsonl`，模式 dependency，速度 1×，会话复制 4 份，会话到达 poisson:0.5，内容 auto
  4. `replay_req_x4_speed2` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_text_c4.jsonl`，模式 request，速度 2×，会话复制 4 份，会话到达 trace，内容 auto
  5. `replay_voice_dep_x2` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_voice_c4.jsonl`，模式 dependency，速度 1×，会话复制 2 份，会话到达 trace，内容 auto
  6. `replay_req_x8_speed4` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_text_c4.jsonl`，模式 request，速度 4×，会话复制 8 份，会话到达 trace，内容 auto
  7. `replay_dep_x8_speed4` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_text_c4.jsonl`，模式 dependency，速度 4×，会话复制 8 份，会话到达 trace，内容 auto
  8. `replay_req_x16_speed8` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_text_c4.jsonl`，模式 request，速度 8×，会话复制 16 份，会话到达 trace，内容 auto
  9. `replay_dep_x16_speed8` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_text_c4.jsonl`，模式 dependency，速度 8×，会话复制 16 份，会话到达 trace，内容 auto
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：用 x5 收集的 Agent trace 做回放：请求级（开环）与依赖感知（闭环）对比，会话复制与加速以提高并发。 2026-11-11 在 Spark 上按原规模（最多 4 倍会话、2 倍加速）回放时，两种模式结果几乎相同，因为服务远未饱和；因此加了 8 倍会话×4 倍加速、16 倍会话×8 倍加速两档，把负载推向饱和，两种模式各跑一次。
- 运行记录：编排器：成功（尝试 1 次），11-11 16:35 开始，组内节点 A（3×A6000），GPU 时长 0.16 h；结果 `20261111_x6_minicpm_replay_run01`：ok，9 分钟

#### `x8_minicpm_profiled_1gpu`
- 计划条目：优先级 55；依赖 `x3_minicpm_speech_1gpu`；申请 1 卡、8 核、90 GB；预计 30 分钟；超时 90 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/x8_minicpm_profiled_1gpu.json`（硬件 a6000x1，运行环境 unicorn，素材 —，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s
- 阶段：
  1. `w3b_events` 闭环：W3b（文本提问→文本+语音）；并发 1, 4, 16；每档 32 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2；**服务端事件**
  2. `w3b_noevents` 闭环：W3b（文本提问→文本+语音）；并发 1, 4, 16；每档 32 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2
  3. `w3b_torchprof` 闭环：W3b（文本提问→文本+语音）；并发 4；每档 8 个请求（另加预热请求：2 个，不计入统计）；停止条件：错误率 > 0.2；**torch profiler**
- 额外采集：服务端事件 w3b_events；torch profiler w3b_torchprof；nsys 无
- 配置里的说明：开启服务端请求级事件记录（request event recorder），得到各阶段排队与执行时间；与 x3 相同负载对比，评估记录本身的开销。 最后加一个短的 torch profiler 窗口（并发 4），得到 kernel 级的 GPU 忙闲区间；这一段不作为延迟基线。
- 运行记录：编排器：成功（尝试 1 次），11-11 16:45 开始，组内节点 A（3×A6000），GPU 时长 0.14 h；结果 `20261111_x8_minicpm_profiled_1gpu_run01`：ok，8 分钟

#### `x3m_minicpm_speech_1gpu_memsplit`
- 计划条目：优先级 54；依赖 `x1_assets`；申请 1 卡、8 核、90 GB；预计 35 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/x3m_minicpm_speech_1gpu_memsplit.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，参数 `--thinker.engine.mem_fraction_static 0.6`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b` 闭环：W3b（文本提问→文本+语音）；并发 1, 2, 4, 8, 16；每档 32 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
  3. `w3` 闭环：W3（语音提问→文本+语音）；并发 1, 2, 4, 8；每档 32 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：显存配比对照：与 x3 相同的 W3b、W3 扫描，只把 thinker 的 engine.mem_fraction_static 从默认 0.85 降到 0.6。在 Spark 上观察到，默认配比下 thinker 占走大部分显存，code2wav 因余量不足跳过 Flow CUDA graph；这里检验 A6000 上是否同样如此、对首个音频块时间和吞吐的影响。
- 运行记录：编排器：成功（尝试 1 次），11-11 16:53 开始，组内节点 A（3×A6000），GPU 时长 0.11 h；结果 `20261111_x3m_minicpm_speech_1gpu_memsplit_run01`：ok，6 分钟

#### `x9_qwen3_omni_3gpu`
- 计划条目：优先级 50；依赖 `x0_envcheck`；申请 3 卡、24 核、260 GB；预计 60 分钟；超时 150 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/x9_qwen3_omni_3gpu.json`（硬件 a6000x3，运行环境 unicorn，素材 —，种子 0）
- 服务 s0：qwen3_omni_30b_a3b，GPU 0, 1, 2，参数 `--model-name qwen3-omni --thinker.tp_size 2 --thinker.gpu [0, 1] --talker_ar.gpu 2 --thinker.engine.disable_custom_all_reduce true --thinker.engine.mem_fraction_static 0.80`，启动时限 3600 s，环境变量 `NCCL_P2P_DISABLE=1`、`SGLANG_OMNI_STARTUP_TIMEOUT=2400`
- 阶段：
  1. `smoke_w1` 冒烟检查：W1（文本→文本），2 个请求，检查输出非空
  2. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  3. `w3b` 闭环：W3b（文本提问→文本+语音）；并发 1, 2, 4, 8；每档 24 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 120 s
  4. `w1` 闭环：W1（文本→文本）；并发 1, 4, 16；每档 32 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：Qwen3-Omni-30B-A3B BF16 语音流水线：thinker TP=2（GPU 0,1）+ talker（GPU 2）。为稳妥起见关闭 NCCL P2P 和自定义 all-reduce（PCIe 节点上曾挂起）。
- 运行记录：编排器：失败（尝试 1 次），11-11 17:10 开始，组内节点 A（3×A6000），GPU 时长 0.54 h；失败原因：server_crashed；结果 `20261111_x9_qwen3_omni_3gpu_run01`：failed，server_crashed，服务退出码 s0=1，11 分钟

#### `x4c_minicpm_video_4cpu`
- 计划条目：优先级 52；依赖 `x4_minicpm_av_text_1gpu`；申请 1 卡、4 核、90 GB；预计 30 分钟；超时 90 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/x4c_minicpm_video_4cpu.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，参数 `--text-only`，启动时限 2400 s
- 阶段：
  1. `w4_video_10s` 闭环：W4（视频+文本→文本），视频 10 s（素材标签 count_circles）；并发 1, 2, 4, 8；每档 24 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
  2. `w4_video_20s` 闭环：W4（视频+文本→文本），视频 20 s（素材标签 count_circles）；并发 1, 2, 4, 8；每档 24 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：CPU 配额对照：与 x4 相同的 10 秒、20 秒视频阶段，但 step 只分配 4 个 CPU（x4 是 8 个），检验视频预处理对 CPU 的敏感度。当前固定版本的 MiniCPM-o 预处理没有并发参数，无法直接做线程池 A/B。
- 运行记录：编排器：成功（尝试 1 次），11-11 17:00 开始，组内节点 A（3×A6000），GPU 时长 0.07 h；结果 `20261111_x4c_minicpm_video_4cpu_run01`：ok，4 分钟

## Job 409712 追加（plan.d 001–012）

### 来源：集群运行目录里的 `plan.d/001-followups.json`

#### `f1_minicpm_text_saturation`
- 计划条目：优先级 45；依赖 无；申请 1 卡、8 核、90 GB；预计 25 分钟；超时 75 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/f1_minicpm_text_saturation.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，参数 `--text-only`，启动时限 2400 s
- 阶段：
  1. `w1_c128_256` 闭环：W1（文本→文本）；并发 128, 256；每档 512 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 60 s
  2. `w1_open` 开环（泊松到达）：W1（文本→文本）；速率 8, 12, 16, 20, 24, 32 请求/秒；每档 300 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：容量刻画（文本）：x2 在并发 64 时吞吐仍随并发线性增长（12 请求/秒，延迟中位数 4.2 秒），没有到饱和点。这里把闭环并发加到 128/256，并用开环 Poisson 到达测饱和拐点和满足 SLO 的有效吞吐（goodput）。
- 运行记录：编排器：成功（尝试 1 次），11-11 17:03 开始，组内节点 A（3×A6000），GPU 时长 0.11 h；结果 `20261111_f1_minicpm_text_saturation_run01`：ok，6 分钟

#### `f2_minicpm_speech_saturation`
- 计划条目：优先级 44；依赖 无；申请 1 卡、8 核、90 GB；预计 30 分钟；超时 90 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/f2_minicpm_speech_saturation.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s
- 阶段：
  1. `w3b_c32_64` 闭环：W3b（文本提问→文本+语音）；并发 32, 64；每档 128 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 120 s
  2. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1, 2, 3, 4, 6 请求/秒；每档 120 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：容量刻画（语音输出）：x3 的 W3b 在并发 16 时约 2.7 请求/秒。这里加到并发 32/64，并用开环 Poisson 到达（1–6 请求/秒）找饱和拐点。
- 运行记录：编排器：成功（尝试 1 次），11-11 17:20 开始，组内节点 A（3×A6000），GPU 时长 0.12 h；结果 `20261111_f2_minicpm_speech_saturation_run01`：ok，服务退出码 s0=1，w3b_open 跑了 1/5 档，7 分钟

#### `f3_minicpm_agents_large`
- 计划条目：优先级 43；依赖 无；申请 1 卡、8 核、90 GB；预计 25 分钟；超时 75 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/f3_minicpm_agents_large.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s
- 阶段：
  1. `agents_text_c1` Agent（合成任务，真实模型请求，录 trace）：32 个会话，并发 1，种子 101，每会话最多 5 步
  2. `agents_text_c4` Agent（合成任务，真实模型请求，录 trace）：64 个会话，并发 4，种子 104，每会话最多 5 步
  3. `agents_text_c16` Agent（合成任务，真实模型请求，录 trace）：128 个会话，并发 16，种子 116，每会话最多 5 步
  4. `agents_text_c32` Agent（合成任务，真实模型请求，录 trace）：128 个会话，并发 32，种子 132，每会话最多 5 步
  5. `agents_text_c16_toollat` Agent（合成任务，真实模型请求，录 trace）：128 个会话，并发 16，种子 116，每会话最多 5 步，模拟工具延迟 对数正态（中位数 0.3 s，σ 0.5）
  6. `agents_voice_c8` Agent（合成任务，真实模型请求，录 trace）：24 个会话，并发 8，种子 11，每会话最多 5 步，语音提问，最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：更大规模的 Agent trace 与 Agent 并发扫描：x5 每组只有 12–24 个会话。这里文本 Agent 在并发 1/4/16/32 下各跑 32–128 个会话，另有带模拟工具延迟的 128 个会话和 24 个语音会话，用于负载刻画和后续回放。
- 运行记录：编排器：成功（尝试 1 次），11-11 17:03 开始，组内节点 A（3×A6000），GPU 时长 0.08 h；结果 `20261111_f3_minicpm_agents_large_run01`：ok，5 分钟

#### `f4_minicpm_replay_large`
- 计划条目：优先级 42；依赖 `f3_minicpm_agents_large`；申请 1 卡、8 核、90 GB；预计 30 分钟；超时 90 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/f4_minicpm_replay_large.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s
- 阶段：
  1. `replay_req_speed1` 回放：trace `latest/f3_minicpm_agents_large/raw/agent_trace_agents_text_c16_toollat.jsonl`，模式 request，速度 1×，会话复制 1 份，会话到达 trace，内容 auto
  2. `replay_dep_speed1` 回放：trace `latest/f3_minicpm_agents_large/raw/agent_trace_agents_text_c16_toollat.jsonl`，模式 dependency，速度 1×，会话复制 1 份，会话到达 trace，内容 auto
  3. `replay_req_speed2` 回放：trace `latest/f3_minicpm_agents_large/raw/agent_trace_agents_text_c16_toollat.jsonl`，模式 request，速度 2×，会话复制 1 份，会话到达 trace，内容 auto
  4. `replay_dep_speed2` 回放：trace `latest/f3_minicpm_agents_large/raw/agent_trace_agents_text_c16_toollat.jsonl`，模式 dependency，速度 2×，会话复制 1 份，会话到达 trace，内容 auto
  5. `replay_req_speed4` 回放：trace `latest/f3_minicpm_agents_large/raw/agent_trace_agents_text_c16_toollat.jsonl`，模式 request，速度 4×，会话复制 1 份，会话到达 trace，内容 auto
  6. `replay_dep_speed4` 回放：trace `latest/f3_minicpm_agents_large/raw/agent_trace_agents_text_c16_toollat.jsonl`，模式 dependency，速度 4×，会话复制 1 份，会话到达 trace，内容 auto
  7. `replay_req_speed8` 回放：trace `latest/f3_minicpm_agents_large/raw/agent_trace_agents_text_c16_toollat.jsonl`，模式 request，速度 8×，会话复制 1 份，会话到达 trace，内容 auto
  8. `replay_dep_speed8` 回放：trace `latest/f3_minicpm_agents_large/raw/agent_trace_agents_text_c16_toollat.jsonl`，模式 dependency，速度 8×，会话复制 1 份，会话到达 trace，内容 auto
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：用 f3 的大 trace（128 个会话、带模拟工具延迟）比较两种回放模式，在 1/2/4/8 倍加速下逐步接近饱和。
- 运行记录：编排器：成功（尝试 1 次），11-11 17:20 开始，组内节点 A（3×A6000），GPU 时长 0.09 h；结果 `20261111_f4_minicpm_replay_large_run01`：ok，5 分钟

#### `x7c_minicpm_2gpu_talker_replica_rep2`
- 计划条目：优先级 41；依赖 无；申请 2 卡、16 核、180 GB；预计 40 分钟；超时 120 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/x7c_minicpm_2gpu_talker_replica_rep2.json`（硬件 a6000x2，运行环境 unicorn，素材 synthetic-v1，种子 1）
- 服务 s0：minicpm_o_4_5，GPU 0, 1，参数 `--config configs/models/minicpm_o_4_5_talker_replica2.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b` 闭环：W3b（文本提问→文本+语音）；并发 2, 4, 8, 16, 32；每档 48 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
  3. `replay_voice_dep_x2` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_voice_c4.jsonl`，模式 dependency，速度 1×，会话复制 2 份，会话到达 trace，内容 auto
  4. `replay_text_dep_x4` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_text_c4.jsonl`，模式 dependency，速度 1×，会话复制 4 份，会话到达 poisson:0.5，内容 auto
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：重复运行（第二次，与第一次的运行顺序相反）：2 卡预算之三：thinker 一份在 GPU 0，talker+code2wav 进程复制两份（GPU 0、GPU 1），只复制语音生成阶段。 另外用 x5 的 Agent trace 做依赖感知回放，在 Agent 负载下比较放置（上游的放置对比只用了固定 TTS 负载）。
- 运行记录：编排器：成功（尝试 1 次），11-11 17:48 开始，组内节点 A（3×A6000），GPU 时长 0.39 h；结果 `20261111_x7c_minicpm_2gpu_talker_replica_rep2_run01`：ok，11 分钟

#### `x7b_minicpm_2gpu_disagg_rep2`
- 计划条目：优先级 40；依赖 无；申请 2 卡、16 核、180 GB；预计 40 分钟；超时 120 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/x7b_minicpm_2gpu_disagg_rep2.json`（硬件 a6000x2，运行环境 unicorn，素材 synthetic-v1，种子 1）
- 服务 s0：minicpm_o_4_5，GPU 0, 1，参数 `--talker.gpu 1 --code2wav.gpu 1`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b` 闭环：W3b（文本提问→文本+语音）；并发 2, 4, 8, 16, 32；每档 48 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
  3. `replay_voice_dep_x2` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_voice_c4.jsonl`，模式 dependency，速度 1×，会话复制 2 份，会话到达 trace，内容 auto
  4. `replay_text_dep_x4` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_text_c4.jsonl`，模式 dependency，速度 1×，会话复制 4 份，会话到达 poisson:0.5，内容 auto
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：重复运行（第二次，与第一次的运行顺序相反）：2 卡预算之二：拆分放置，thinker 与编码器在 GPU 0，talker 与 code2wav 在 GPU 1（一个服务）。 另外用 x5 的 Agent trace 做依赖感知回放，在 Agent 负载下比较放置（上游的放置对比只用了固定 TTS 负载）。
- 运行记录：编排器：成功（尝试 1 次），11-11 18:00 开始，组内节点 A（3×A6000），GPU 时长 0.31 h；结果 `20261111_x7b_minicpm_2gpu_disagg_rep2_run01`：ok，9 分钟

#### `x7a_minicpm_2gpu_two_replicas_rep2`
- 计划条目：优先级 39；依赖 无；申请 2 卡、16 核、180 GB；预计 40 分钟；超时 120 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/x7a_minicpm_2gpu_two_replicas_rep2.json`（硬件 a6000x2，运行环境 unicorn，素材 synthetic-v1，种子 1，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b` 闭环：W3b（文本提问→文本+语音）；并发 2, 4, 8, 16, 32；每档 48 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
  3. `replay_voice_dep_x2` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_voice_c4.jsonl`，模式 dependency，速度 1×，会话复制 2 份，会话到达 trace，内容 auto
  4. `replay_text_dep_x4` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_text_c4.jsonl`，模式 dependency，速度 1×，会话复制 4 份，会话到达 poisson:0.5，内容 auto
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：重复运行（第二次，与第一次的运行顺序相反）：2 卡预算之一：两个独立的完整副本（每卡一个默认语音流水线），客户端按在途请求最少路由。 另外用 x5 的 Agent trace 做依赖感知回放，在 Agent 负载下比较放置（上游的放置对比只用了固定 TTS 负载）。
- 运行记录：编排器：成功（尝试 1 次），11-11 18:30 开始，组内节点 A（3×A6000），GPU 时长 0.30 h；结果 `20261111_x7a_minicpm_2gpu_two_replicas_rep2_run01`：ok，9 分钟

#### `x3_minicpm_speech_1gpu_rep2`
- 计划条目：优先级 38；依赖 无；申请 1 卡、8 核、90 GB；预计 20 分钟；超时 60 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/x3_minicpm_speech_1gpu_rep2.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 1）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b` 闭环：W3b（文本提问→文本+语音）；并发 1, 2, 4, 8, 16；每档 32 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：重复运行（第二次，与第一次的运行顺序相反）：语音输出基线：MiniCPM-o 4.5 默认语音流水线（所有阶段在一张卡上），W3b 文本→语音、W3 语音→语音、W1 文本请求，关闭 profiler。
- 运行记录：编排器：成功（尝试 1 次），11-11 16:37 开始，组内节点 A（3×A6000），GPU 时长 0.08 h；结果 `20261111_x3_minicpm_speech_1gpu_rep2_run01`：ok，5 分钟

### 来源：集群运行目录里的 `plan.d/002-x7c-retry.json`

#### `x7c_minicpm_2gpu_talker_replica_retry`
- 计划条目：优先级 42；依赖 无；申请 2 卡、16 核、180 GB；预计 40 分钟；超时 120 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/x7c_minicpm_2gpu_talker_replica.json`（硬件 a6000x2，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0, 1，参数 `--config configs/models/minicpm_o_4_5_talker_replica2.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b` 闭环：W3b（文本提问→文本+语音）；并发 2, 4, 8, 16, 32；每档 48 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
  3. `replay_voice_dep_x2` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_voice_c4.jsonl`，模式 dependency，速度 1×，会话复制 2 份，会话到达 trace，内容 auto
  4. `replay_text_dep_x4` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_text_c4.jsonl`，模式 dependency，速度 1×，会话复制 4 份，会话到达 poisson:0.5，内容 auto
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：2 卡预算之三：thinker 一份在 GPU 0，talker+code2wav 进程复制两份（GPU 0、GPU 1），只复制语音生成阶段。 另外用 x5 的 Agent trace 做依赖感知回放，在 Agent 负载下比较放置（上游的放置对比只用了固定 TTS 负载）。
- 运行记录：编排器：成功（尝试 1 次），11-11 17:32 开始，组内节点 A（3×A6000），GPU 时长 0.32 h；结果 `20261111_x7c_minicpm_2gpu_talker_replica_retry_run01`：ok，服务退出码 s0=1，w3b 跑了 3/5 档，w3b 停止原因：error rate 37/48 at concurrency 8，10 分钟

### 来源：集群运行目录里的 `plan.d/003-x9-retry.json`

#### `x9_qwen3_omni_3gpu_retry`
- 计划条目：优先级 50；依赖 无；申请 3 卡、24 核、260 GB；预计 60 分钟；超时 150 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/x9_qwen3_omni_3gpu.json`（硬件 a6000x3，运行环境 unicorn，素材 —，种子 0）
- 服务 s0：qwen3_omni_30b_a3b，GPU 0, 1, 2，参数 `--model-name qwen3-omni --thinker.tp_size 2 --thinker.gpu [0, 1] --talker_ar.gpu 2 --thinker.engine.disable_custom_all_reduce true --thinker.engine.mem_fraction_static 0.80`，启动时限 3600 s，环境变量 `NCCL_P2P_DISABLE=1`、`SGLANG_OMNI_STARTUP_TIMEOUT=2400`
- 阶段：
  1. `smoke_w1` 冒烟检查：W1（文本→文本），2 个请求，检查输出非空
  2. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  3. `w3b` 闭环：W3b（文本提问→文本+语音）；并发 1, 2, 4, 8；每档 24 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 120 s
  4. `w1` 闭环：W1（文本→文本）；并发 1, 4, 16；每档 32 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：Qwen3-Omni-30B-A3B BF16 语音流水线：thinker TP=2（GPU 0,1）+ talker（GPU 2）。为稳妥起见关闭 NCCL P2P 和自定义 all-reduce（PCIe 节点上曾挂起）。
- 运行记录：编排器：失败（尝试 1 次），11-11 17:28 开始，组内节点 A（3×A6000），GPU 时长 0.23 h；失败原因：server_crashed；结果 `20261111_x9_qwen3_omni_3gpu_retry_run01`：failed，server_crashed，服务退出码 s0=1，5 分钟

### 来源：集群运行目录里的 `plan.d/004-x9-retry2.json`

#### `x9_qwen3_omni_3gpu_retry2`
- 计划条目：优先级 50；依赖 无；申请 3 卡、24 核、260 GB；预计 60 分钟；超时 150 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/x9_qwen3_omni_3gpu.json`（硬件 a6000x3，运行环境 unicorn，素材 —，种子 0）
- 服务 s0：qwen3_omni_30b_a3b，GPU 0, 1, 2，参数 `--model-name qwen3-omni --thinker.tp_size 2 --thinker.gpu [0, 1] --talker_ar.gpu 2 --thinker.engine.disable_custom_all_reduce true --thinker.engine.mem_fraction_static 0.80`，启动时限 3600 s，环境变量 `NCCL_P2P_DISABLE=1`、`SGLANG_OMNI_STARTUP_TIMEOUT=2400`
- 阶段：
  1. `smoke_w1` 冒烟检查：W1（文本→文本），2 个请求，检查输出非空
  2. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  3. `w3b` 闭环：W3b（文本提问→文本+语音）；并发 1, 2, 4, 8；每档 24 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 120 s
  4. `w1` 闭环：W1（文本→文本）；并发 1, 4, 16；每档 32 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：Qwen3-Omni-30B-A3B BF16 语音流水线：thinker TP=2（GPU 0,1）+ talker（GPU 2）。为稳妥起见关闭 NCCL P2P 和自定义 all-reduce（PCIe 节点上曾挂起）。
- 运行记录：编排器：成功（尝试 1 次），11-11 17:42 开始，组内节点 A（3×A6000），GPU 时长 0.32 h；结果 `20261111_x9_qwen3_omni_3gpu_retry2_run01`：ok，6 分钟

### 来源：集群运行目录里的 `plan.d/005-followups-2.json`

#### `x7e_minicpm_2gpu_disagg_events`
- 计划条目：优先级 38；依赖 无；申请 2 卡、16 核、180 GB；预计 25 分钟；超时 90 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/x7e_minicpm_2gpu_disagg_events.json`（硬件 a6000x2，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0, 1，参数 `--talker.gpu 1 --code2wav.gpu 1`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_events` 闭环：W3b（文本提问→文本+语音）；并发 4, 8, 16, 32；每档 48 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
- 额外采集：服务端事件 w3b_events；torch profiler 无；nsys 无
- 配置里的说明：x7b（thinker 在 GPU 0，talker+code2wav 在 GPU 1）开启服务端请求事件记录，W3b 并发 4–32。检验第 7 节的推测：拆分是否消除了 thinker 与 talker 的同卡争用，以及并发 32 时瓶颈在哪个阶段。
- 运行记录：编排器：成功（尝试 1 次），11-11 18:39 开始，组内节点 A（3×A6000），GPU 时长 0.13 h；结果 `20261111_x7e_minicpm_2gpu_disagg_events_run01`：ok，4 分钟

#### `f5_minicpm_speech_openloop`
- 计划条目：优先级 37；依赖 无；申请 1 卡、8 核、90 GB；预计 20 分钟；超时 75 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/f5_minicpm_speech_openloop.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 0.5, 1, 1.5, 2, 2.5, 3 请求/秒；每档 90 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：单卡语音输出（W3b）的开环饱和曲线：泊松到达 0.5–3 请求/秒，每档 90 个请求，用全新的服务，前面不跑闭环高负载。补上 f2 因崩溃没有得到的部分；闭环测得的上限约 2.9 请求/秒。
- 运行记录：编排器：成功（尝试 1 次），11-11 17:48 开始，组内节点 A（3×A6000），GPU 时长 0.17 h；结果 `20261111_f5_minicpm_speech_openloop_run01`：ok，服务退出码 s0=1，w3b_open 跑了 3/6 档，10 分钟

#### `f2_minicpm_speech_saturation_rep2`
- 计划条目：优先级 36；依赖 无；申请 1 卡、8 核、90 GB；预计 30 分钟；超时 90 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/f2_minicpm_speech_saturation_rep2.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s
- 阶段：
  1. `w3b_c32_64` 闭环：W3b（文本提问→文本+语音）；并发 32, 64；每档 128 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 120 s
  2. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1, 2, 3, 4, 6 请求/秒；每档 120 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：f2 的重复（同一配置，新的服务）。f2 在开环 1 请求/秒阶段第 53.8 秒因 CUDA illegal memory access 崩溃，这次检验崩溃能否在同一位置复现，并重复并发 32/64 的饱和测量。
- 运行记录：编排器：成功（尝试 1 次），11-11 18:00 开始，组内节点 A（3×A6000），GPU 时长 0.15 h；结果 `20261111_f2_minicpm_speech_saturation_rep2_run01`：ok，服务退出码 s0=1，9 分钟

### 来源：集群运行目录里的 `plan.d/006-crash-hunt.json`

#### `z1_minicpm_crash_blocking`
- 计划条目：优先级 41；依赖 无；申请 1 卡、8 核、90 GB；预计 22 分钟；超时 75 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/z1_minicpm_crash_blocking.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s，环境变量 `CUDA_LAUNCH_BLOCKING=1`
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1, 1, 1, 1, 1, 1, 1, 1 请求/秒；每档 120 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：定位语音负载下的 CUDA illegal memory access（f2、x7c、f5 各崩溃一次，见 docs/05_findings 第 17 节）。负载与 f5 崩溃时相同：W3b 开环泊松到达，每 2 分钟一档、速率不变；某档错误率超过 20% 即停止，崩溃后很快释放 GPU。CUDA_LAUNCH_BLOCKING=1，让报错停在真正出错的 kernel 上（同步执行会降低吞吐，所以用 1.0 请求/秒）。
- 运行记录：编排器：成功（尝试 1 次），11-11 18:10 开始，组内节点 A（3×A6000），GPU 时长 0.35 h；结果 `20261111_z1_minicpm_crash_blocking_run01`：ok，21 分钟

#### `z4_minicpm_crash_default`
- 计划条目：优先级 40；依赖 无；申请 1 卡、8 核、90 GB；预计 22 分钟；超时 75 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/z4_minicpm_crash_default.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：定位语音负载下的 CUDA illegal memory access（f2、x7c、f5 各崩溃一次，见 docs/05_findings 第 17 节）。负载与 f5 崩溃时相同：W3b 开环泊松到达，每 2 分钟一档、速率不变；某档错误率超过 20% 即停止，崩溃后很快释放 GPU。默认配置，作对照：同样负载下的崩溃频率。
- 运行记录：编排器：成功（尝试 1 次），11-11 18:10 开始，组内节点 A（3×A6000），GPU 时长 0.19 h；结果 `20261111_z4_minicpm_crash_default_run01`：ok，服务退出码 s0=1，w3b_soak 跑了 4/8 档，12 分钟

#### `z2_minicpm_crash_nocompile`
- 计划条目：优先级 40；依赖 无；申请 1 卡、8 核、90 GB；预计 22 分钟；超时 75 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/z2_minicpm_crash_nocompile.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，参数 `--code2wav.factory.enable_dit_torch_compile false --code2wav.factory.enable_hift_torch_compile false`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：定位语音负载下的 CUDA illegal memory access（f2、x7c、f5 各崩溃一次，见 docs/05_findings 第 17 节）。负载与 f5 崩溃时相同：W3b 开环泊松到达，每 2 分钟一档、速率不变；某档错误率超过 20% 即停止，崩溃后很快释放 GPU。关闭 code2wav 中 DiT 与 HiFT 的 torch.compile。
- 运行记录：编排器：成功（尝试 1 次），11-11 18:10 开始，组内节点 A（3×A6000），GPU 时长 0.16 h；结果 `20261111_z2_minicpm_crash_nocompile_run01`：ok，服务退出码 s0=1，w3b_soak 跑了 3/8 档，10 分钟

#### `z3_minicpm_crash_nograph`
- 计划条目：优先级 40；依赖 无；申请 1 卡、8 核、90 GB；预计 22 分钟；超时 75 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/z3_minicpm_crash_nograph.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，参数 `--code2wav.factory.enable_flow_cuda_graph false --talker.engine.disable_cuda_graph true`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：定位语音负载下的 CUDA illegal memory access（f2、x7c、f5 各崩溃一次，见 docs/05_findings 第 17 节）。负载与 f5 崩溃时相同：W3b 开环泊松到达，每 2 分钟一档、速率不变；某档错误率超过 20% 即停止，崩溃后很快释放 GPU。关闭 code2wav 的 flow CUDA graph 和 talker 的 CUDA graph。
- 运行记录：编排器：成功（尝试 1 次），11-11 18:20 开始，组内节点 A（3×A6000），GPU 时长 0.30 h；结果 `20261111_z3_minicpm_crash_nograph_run01`：ok，18 分钟

### 来源：集群运行目录里的 `plan.d/007-crash-hunt-2.json`

#### `z5_minicpm_crash_no_async_decode`
- 计划条目：优先级 40；依赖 无；申请 1 卡、8 核、90 GB；预计 22 分钟；超时 75 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/z5_minicpm_crash_no_async_decode.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，参数 `--talker.factory.enable_async_decode false`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：定位语音负载下的 CUDA illegal memory access（f2、x7c、f5 各崩溃一次，见 docs/05_findings 第 17 节）。负载与 f5 崩溃时相同：W3b 开环泊松到达，每 2 分钟一档、速率不变；某档错误率超过 20% 即停止，崩溃后很快释放 GPU。关闭 talker 的异步 decode（`--talker.factory.enable_async_decode false`）。五次崩溃中退出的都是 talker 进程，其中三次第一个报错在 talker 的异步 decode 同步点；z2 关闭 code2wav 的 torch.compile 后仍然崩溃。
- 运行记录：编排器：失败（尝试 1 次），11-11 18:21 开始，组内节点 A（3×A6000），GPU 时长 0.02 h；失败原因：server_crashed；结果 `20261111_z5_minicpm_crash_no_async_decode_run01`：failed，server_crashed，服务退出码 s0=1，1 分钟

### 来源：集群运行目录里的 `plan.d/008-crash-hunt-3.json`

#### `z6_minicpm_crash_no_talker_graph`
- 计划条目：优先级 37；依赖 无；申请 1 卡、8 核、90 GB；预计 22 分钟；超时 75 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/z6_minicpm_crash_no_talker_graph.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，参数 `--talker.engine.disable_cuda_graph true`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：定位语音负载下的 CUDA illegal memory access（f2、x7c、f5 各崩溃一次，见 docs/05_findings 第 17 节）。负载与 f5 崩溃时相同：W3b 开环泊松到达，每 2 分钟一档、速率不变；某档错误率超过 20% 即停止，崩溃后很快释放 GPU。只关闭 talker 的 CUDA graph（`--talker.engine.disable_cuda_graph true`），与 z3 拆开检验。
- 运行记录：编排器：成功（尝试 1 次），11-11 18:38 开始，组内节点 A（3×A6000），GPU 时长 0.27 h；结果 `20261111_z6_minicpm_crash_no_talker_graph_run01`：ok，服务退出码 s0=1，w3b_soak 跑了 6/8 档，16 分钟

#### `z7_minicpm_crash_no_flow_graph`
- 计划条目：优先级 37；依赖 无；申请 1 卡、8 核、90 GB；预计 22 分钟；超时 75 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/z7_minicpm_crash_no_flow_graph.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，参数 `--code2wav.factory.enable_flow_cuda_graph false`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：定位语音负载下的 CUDA illegal memory access（f2、x7c、f5 各崩溃一次，见 docs/05_findings 第 17 节）。负载与 f5 崩溃时相同：W3b 开环泊松到达，每 2 分钟一档、速率不变；某档错误率超过 20% 即停止，崩溃后很快释放 GPU。只关闭 code2wav flow 的 CUDA graph（`--code2wav.factory.enable_flow_cuda_graph false`），与 z3 拆开检验。
- 运行记录：编排器：成功（尝试 1 次），11-11 18:43 开始，组内节点 A（3×A6000），GPU 时长 0.06 h；结果 `20261111_z7_minicpm_crash_no_flow_graph_run01`：ok，服务退出码 s0=1，w3b_soak 跑了 1/8 档，3 分钟

### 来源：集群运行目录里的 `plan.d/009-x7f.json`

#### `x7f_minicpm_3gpu_split_talker2`
- 计划条目：优先级 45；依赖 无；申请 3 卡、24 核、260 GB；预计 20 分钟；超时 75 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/x7f_minicpm_3gpu_split_talker2.json`（硬件 a6000x3，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2，参数 `--config configs/models/minicpm_o_4_5_talker_replica2_split3.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_events` 闭环：W3b（文本提问→文本+语音）；并发 8, 16, 32, 64；每档 48 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
- 额外采集：服务端事件 w3b_events；torch profiler 无；nsys 无
- 配置里的说明：3 卡：thinker 单独一张卡，talker+code2wav 复制两份、各占一张卡。x7e（2 卡拆分）在并发 32 时瓶颈转到 talker 那张卡上，这里检验再加一张 talker 卡能否抬高上限。开启服务端事件记录，并发 8–64。
- 运行记录：编排器：成功（尝试 1 次），11-11 18:54 开始，组内节点 A（3×A6000），GPU 时长 0.20 h；结果 `20261111_x7f_minicpm_3gpu_split_talker2_run01`：ok，4 分钟

### 来源：集群运行目录里的 `plan.d/010-crash-hunt-4.json`

#### `z3_minicpm_crash_nograph_rep2`
- 计划条目：优先级 44；依赖 无；申请 1 卡、8 核、90 GB；预计 22 分钟；超时 75 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/z3_minicpm_crash_nograph_rep2.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，参数 `--code2wav.factory.enable_flow_cuda_graph false --talker.engine.disable_cuda_graph true`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：定位语音负载下的 CUDA illegal memory access（f2、x7c、f5 各崩溃一次，见 docs/05_findings 第 17 节）。负载与 f5 崩溃时相同：W3b 开环泊松到达，每 2 分钟一档、速率不变；某档错误率超过 20% 即停止，崩溃后很快释放 GPU。关闭 code2wav 的 flow CUDA graph 和 talker 的 CUDA graph。 重复运行：z6、z7 只关一种 CUDA graph 时仍然崩溃，检验 z3 不崩溃是否可重复（与默认配置并排运行）。
- 运行记录：编排器：成功（尝试 1 次），11-11 18:58 开始，组内节点 A（3×A6000），GPU 时长 0.32 h；结果 `20261111_z3_minicpm_crash_nograph_rep2_run01`：ok，19 分钟

#### `z4_minicpm_crash_default_rep2`
- 计划条目：优先级 44；依赖 无；申请 1 卡、8 核、90 GB；预计 22 分钟；超时 75 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/z4_minicpm_crash_default_rep2.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：定位语音负载下的 CUDA illegal memory access（f2、x7c、f5 各崩溃一次，见 docs/05_findings 第 17 节）。负载与 f5 崩溃时相同：W3b 开环泊松到达，每 2 分钟一档、速率不变；某档错误率超过 20% 即停止，崩溃后很快释放 GPU。默认配置，作对照：同样负载下的崩溃频率。 重复运行：z6、z7 只关一种 CUDA graph 时仍然崩溃，检验 z3 不崩溃是否可重复（与默认配置并排运行）。
- 运行记录：编排器：成功（尝试 1 次），11-11 18:58 开始，组内节点 A（3×A6000），GPU 时长 0.07 h；结果 `20261111_z4_minicpm_crash_default_rep2_run01`：ok，服务退出码 s0=1，w3b_soak 跑了 1/8 档，4 分钟

### 来源：集群运行目录里的 `plan.d/011-crash-hunt-5.json`

#### `z3_minicpm_crash_nograph_rep3`
- 计划条目：优先级 44；依赖 无；申请 1 卡、8 核、90 GB；预计 22 分钟；超时 75 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/z3_minicpm_crash_nograph_rep3.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，参数 `--code2wav.factory.enable_flow_cuda_graph false --talker.engine.disable_cuda_graph true`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：定位语音负载下的 CUDA illegal memory access（f2、x7c、f5 各崩溃一次，见 docs/05_findings 第 17 节）。负载与 f5 崩溃时相同：W3b 开环泊松到达，每 2 分钟一档、速率不变；某档错误率超过 20% 即停止，崩溃后很快释放 GPU。关闭 code2wav 的 flow CUDA graph 和 talker 的 CUDA graph。 第三次运行（与第二次并排，累计更多运行时长）：z6、z7 只关一种 CUDA graph 时仍然崩溃，检验 z3 不崩溃是否可重复（与默认配置并排运行）。
- 运行记录：编排器：成功（尝试 1 次），11-11 19:04 开始，组内节点 A（3×A6000），GPU 时长 0.31 h；结果 `20261111_z3_minicpm_crash_nograph_rep3_run01`：ok，19 分钟

### 来源：集群运行目录里的 `plan.d/012-crash-hunt-6.json`

#### `z8_minicpm_crash_split_1p5`
- **注意**：这个文件放进 plan.d 时（19:25）编排器已经在 19:23 结束，没有被接收，实验没有运行
- 计划条目：优先级 44；依赖 无；申请 2 卡、16 核、180 GB；预计 22 分钟；超时 75 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/z8_minicpm_crash_split_1p5.json`（硬件 a6000x2，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0, 1，参数 `--talker.gpu 1 --code2wav.gpu 1`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：九次崩溃中，talker 进程都和 thinker 进程在同一张卡上；talker 单独占卡的拆分布局（x7b、x7e、x7f）累计约 13 分钟没有崩溃，样本不够。这里用 x7b 的拆分布局（thinker 在 GPU 0，talker+code2wav 在 GPU 1）做开环耐久测试：W3b 泊松到达 1.5 请求/秒，每 2 分钟一档，最多 8 档。
- 运行记录：没有运行记录

#### `z9_minicpm_crash_split_3p0`
- **注意**：这个文件放进 plan.d 时（19:25）编排器已经在 19:23 结束，没有被接收，实验没有运行
- 计划条目：优先级 43；依赖 无；申请 2 卡、16 核、180 GB；预计 22 分钟；超时 75 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/z9_minicpm_crash_split_3p0.json`（硬件 a6000x2，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0, 1，参数 `--talker.gpu 1 --code2wav.gpu 1`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 3, 3, 3, 3, 3, 3, 3, 3 请求/秒；每档 360 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：九次崩溃中，talker 进程都和 thinker 进程在同一张卡上；talker 单独占卡的拆分布局（x7b、x7e、x7f）累计约 13 分钟没有崩溃，样本不够。这里用 x7b 的拆分布局（thinker 在 GPU 0，talker+code2wav 在 GPU 1）做开环耐久测试：W3b 泊松到达 3.0 请求/秒，每 2 分钟一档，最多 8 档。
- 运行记录：没有运行记录

## Job 409712 手动步骤（编排器结束后的崩溃定位）

### 来源：`manual/steps.log`

#### `z10_minicpm_crash_coredump`
- 手动步骤：srun 返回码 0；资源 1 卡、8 核、90 GB（Slurm 记账）
- 配置：`configs/experiments/a6000/z10_minicpm_crash_coredump.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s，启动前缀 `env CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_ENABLE_CPU_COREDUMP_ON_EXCEPTION=0 CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1 CUDA_COREDUMP_GENERATION_FLAGS=skip_global_memory,skip_shared_memory,skip_local_memory CUDA_COREDUMP_SHOW_PROGRESS=1 CUDA_COREDUMP_FILE={exp_dir}/cuda_core_%h_%p_%t`
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：定位语音负载下的 CUDA illegal memory access（第 17 节：九次崩溃，退出的都是 talker 进程）。负载与 z4 相同：W3b 开环泊松到达 1.5 请求/秒，每 2 分钟一档，最多 8 档。服务启动时开启 GPU core dump：出现 GPU 异常时驱动把出错的 kernel 和位置写进实验目录，用 cuda-gdb 读取；只保留 kernel 信息，不保存显存内容，也不生成 CPU core dump，以免文件过大。与 CUDA_LAUNCH_BLOCKING 不同，它不改变执行时序。默认配置。
- 运行记录：结果 `20261111_z10_minicpm_crash_coredump_run01`：ok，19 分钟

#### `z11_minicpm_crash_split_process`
- 手动步骤：srun 返回码 0；资源 1 卡、8 核、90 GB（Slurm 记账）
- 配置：`configs/experiments/a6000/z11_minicpm_crash_split_process.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，参数 `--code2wav.process code2wav`，启动时限 2400 s，启动前缀 `env CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_ENABLE_CPU_COREDUMP_ON_EXCEPTION=0 CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1 CUDA_COREDUMP_GENERATION_FLAGS=skip_global_memory,skip_shared_memory,skip_local_memory CUDA_COREDUMP_SHOW_PROGRESS=1 CUDA_COREDUMP_FILE={exp_dir}/cuda_core_%h_%p_%t`
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：定位语音负载下的 CUDA illegal memory access（第 17 节：九次崩溃，退出的都是 talker 进程）。负载与 z4 相同：W3b 开环泊松到达 1.5 请求/秒，每 2 分钟一档，最多 8 档。服务启动时开启 GPU core dump：出现 GPU 异常时驱动把出错的 kernel 和位置写进实验目录，用 cuda-gdb 读取；只保留 kernel 信息，不保存显存内容，也不生成 CPU core dump，以免文件过大。与 CUDA_LAUNCH_BLOCKING 不同，它不改变执行时序。把 code2wav 放到单独的进程（`--code2wav.process code2wav`），检验崩溃是否来自 talker 进程内两个线程（talker 与 code2wav）之间的竞争；CUDA graph 和 torch.compile 保持默认开启。
- 运行记录：结果 `20261111_z11_minicpm_crash_split_process_run01`：ok，19 分钟

#### `z12_minicpm_crash_eager_loading`
- 手动步骤：srun 返回码 0；资源 1 卡、8 核、90 GB（Slurm 记账）
- 配置：`configs/experiments/a6000/z12_minicpm_crash_eager_loading.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s，启动前缀 `env CUDA_MODULE_LOADING=EAGER`
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：定位语音负载下的 CUDA illegal memory access（第 17 节）。z10 开启 GPU core dump 后 16 分钟没有崩溃、延迟不变。开启 core dump 可能让驱动改变模块加载方式；这里不开 core dump，只设 CUDA_MODULE_LOADING=EAGER（PyTorch 默认是 LAZY：kernel 第一次用到时才加载），检验运行中延迟加载 kernel 是否与崩溃有关。负载与 z4、z10 相同。
- 运行记录：结果 `20261111_z12_minicpm_crash_eager_loading_run01`：ok，服务退出码 s0=1，w3b_soak 跑了 4/8 档，10 分钟

#### `z10_minicpm_crash_coredump_rep2`
- 手动步骤：srun 返回码 0；资源 1 卡、8 核、90 GB（Slurm 记账）
- 配置：`configs/experiments/a6000/z10_minicpm_crash_coredump_rep2.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s，启动前缀 `env CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_ENABLE_CPU_COREDUMP_ON_EXCEPTION=0 CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1 CUDA_COREDUMP_GENERATION_FLAGS=skip_global_memory,skip_shared_memory,skip_local_memory CUDA_COREDUMP_SHOW_PROGRESS=1 CUDA_COREDUMP_FILE={exp_dir}/cuda_core_%h_%p_%t`
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：定位语音负载下的 CUDA illegal memory access（第 17 节：九次崩溃，退出的都是 talker 进程）。负载与 z4 相同：W3b 开环泊松到达 1.5 请求/秒，每 2 分钟一档，最多 8 档。服务启动时开启 GPU core dump：出现 GPU 异常时驱动把出错的 kernel 和位置写进实验目录，用 cuda-gdb 读取；只保留 kernel 信息，不保存显存内容，也不生成 CPU core dump，以免文件过大。与 CUDA_LAUNCH_BLOCKING 不同，它不改变执行时序。默认配置。 重复运行：第一次 16 分钟、1440 个请求没有崩溃，延迟与默认配置相同；检验这是否偶然。
- 运行记录：结果 `20261111_z10_minicpm_crash_coredump_rep2_run01`：ok，18 分钟

#### `z11b_minicpm_crash_split_process_nodump`
- 手动步骤：srun 返回码 0；资源 1 卡、8 核、90 GB（Slurm 记账）
- 配置：`configs/experiments/a6000/z11b_minicpm_crash_split_process_nodump.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：与 z11 相同（code2wav 放在单独的进程），但不开 GPU core dump。z11 不崩溃的结果混有 core dump 的影响：z10 只开 core dump 也没有崩溃。负载与 z4 相同。
- 运行记录：结果 `20261111_z11b_minicpm_crash_split_process_nodump_run01`：ok，20 分钟

#### `z11b_minicpm_crash_split_process_nodump_rep2`
- 手动步骤：srun 返回码 0；资源 1 卡、8 核、90 GB（Slurm 记账）
- 配置：`configs/experiments/a6000/z11b_minicpm_crash_split_process_nodump_rep2.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：与 z11 相同（code2wav 放在单独的进程），但不开 GPU core dump。z11 不崩溃的结果混有 core dump 的影响：z10 只开 core dump 也没有崩溃。负载与 z4 相同。 重复运行：第一次 16 分钟没有崩溃。
- 运行记录：结果 `20261111_z11b_minicpm_crash_split_process_nodump_rep2_run01`：ok，19 分钟

#### `zp1_minicpm_crash_talker_private_stream`
- 手动步骤：srun 返回码 0；资源 1 卡、8 核、90 GB（Slurm 记账）
- 配置：`configs/experiments/a6000/zp1_minicpm_crash_talker_private_stream.json`（硬件 a6000x1，运行环境 unicorn_diag，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s，启动前缀 `env SGLANG_OMNI_DIAG_PRIVATE_STREAM_STAGES=talker`
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：验证崩溃根因（第 17 节）。代码阅读发现：talker 线程的所有 GPU 工作都在默认队列（stream 0）上；code2wav 的 HiFT 为了绕开 cuFFT 表上传的问题，每次 STFT/iSTFT 都切到 stream 0，并双向等待（token2wav/hift.py:38-50）。这里用 sglang-omni 的诊断副本（同一提交 520b49ee，加 configs/runtime/patches/sglang_omni_diag_private_stream.patch），只让 talker 的调度线程改用一条独立的 CUDA stream；同一进程、CUDA graph 和 torch.compile 都保持默认。负载与 z4 相同。若不再崩溃，说明共用 stream 0 是触发条件。
- 运行记录：结果 `20261112_zp1_minicpm_crash_talker_private_stream_run01`：ok，服务退出码 s0=1，w3b_soak 跑了 3/8 档，9 分钟

#### `zp1_minicpm_crash_talker_private_stream_rep2`
- 手动步骤：srun 返回码 137；资源 1 卡、8 核、90 GB（Slurm 记账）；11-12 00:16 手动取消（步骤 409712.48，开始后 46 秒）。运行目录里记录的理由：zp1 第一次已在约 6 分钟时崩溃，stream 0 共享不是必要条件，重复没有信息量；让 zg1/zg2 尽早开始
- 配置：`configs/experiments/a6000/zp1_minicpm_crash_talker_private_stream_rep2.json`（硬件 a6000x1，运行环境 unicorn_diag，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s，启动前缀 `env SGLANG_OMNI_DIAG_PRIVATE_STREAM_STAGES=talker`
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：验证崩溃根因（第 17 节）。代码阅读发现：talker 线程的所有 GPU 工作都在默认队列（stream 0）上；code2wav 的 HiFT 为了绕开 cuFFT 表上传的问题，每次 STFT/iSTFT 都切到 stream 0，并双向等待（token2wav/hift.py:38-50）。这里用 sglang-omni 的诊断副本（同一提交 520b49ee，加 configs/runtime/patches/sglang_omni_diag_private_stream.patch），只让 talker 的调度线程改用一条独立的 CUDA stream；同一进程、CUDA graph 和 torch.compile 都保持默认。负载与 z4 相同。若不再崩溃，说明共用 stream 0 是触发条件。 重复运行。
- 运行记录：结果 `20261112_zp1_minicpm_crash_talker_private_stream_rep2_run01`：没有 result.json

#### `zg1_minicpm_crash_plan_guard`
- 手动步骤：srun 返回码 0；资源 1 卡、8 核、90 GB（Slurm 记账）
- 配置：`configs/experiments/a6000/zg1_minicpm_crash_plan_guard.json`（硬件 a6000x1，运行环境 unicorn_diag2，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s，启动前缀 `env SGLANG_DIAG_PLAN_GUARD=1`
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：验证崩溃根因（第 17 节）。三份代码阅读都指出 talker 的一个主机端竞争：图模式下 FlashInfer 的 fast_decode_plan 每一步在 CPU 上改写 wrapper 的页锁定计划表，再用非阻塞方式拷到 GPU；talker 提前一步工作，如果上一步的拷贝因为排在 code2wav 的高优先级工作后面还没执行，上一步会读到下一步的分块索引。这里用第二个诊断副本（sglang-omni 520b49ee 无改动 + SGLang 副本加 env 开关的保护补丁），其余与 z4 完全相同。开启保护：在同一个 wrapper 上重新规划之前，如果上一份计划的异步拷贝可能还没执行，就计数并等待它完成。若不再崩溃，说明这个竞争是崩溃的原因。
- 运行记录：结果 `20261112_zg1_minicpm_crash_plan_guard_run01`：ok，服务退出码 s0=1，w3b_soak 跑了 1/8 档，5 分钟

#### `zg2_minicpm_crash_plan_count`
- 手动步骤：srun 返回码 137；资源 1 卡、8 核、90 GB（Slurm 记账）；11-12 00:24 手动取消（步骤 409712.50）。运行目录里记录的理由：zg1（计划表保护）仍在约 2 分钟崩溃，停掉 zg2（只计数），取消尚未开始的第 7 轮；第 8 轮先跑 zc1，再跑 zg3
- 配置：`configs/experiments/a6000/zg2_minicpm_crash_plan_count.json`（硬件 a6000x1，运行环境 unicorn_diag2，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s，启动前缀 `env SGLANG_DIAG_PLAN_GUARD=count`
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：验证崩溃根因（第 17 节）。三份代码阅读都指出 talker 的一个主机端竞争：图模式下 FlashInfer 的 fast_decode_plan 每一步在 CPU 上改写 wrapper 的页锁定计划表，再用非阻塞方式拷到 GPU；talker 提前一步工作，如果上一步的拷贝因为排在 code2wav 的高优先级工作后面还没执行，上一步会读到下一步的分块索引。这里用第二个诊断副本（sglang-omni 520b49ee 无改动 + SGLang 副本加 env 开关的保护补丁），其余与 z4 完全相同。只计数、不等待：检查默认行为下这个竞争窗口是否频繁出现，以及崩溃前是否出现。
- 运行记录：结果 `20261112_zg2_minicpm_crash_plan_count_run01`：没有 result.json

#### `zc1_minicpm_crash_one_hw_queue`
- 手动步骤：srun 返回码 0；资源 1 卡、8 核、90 GB（Slurm 记账）
- 配置：`configs/experiments/a6000/zc1_minicpm_crash_one_hw_queue.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s，启动前缀 `env CUDA_DEVICE_MAX_CONNECTIONS=1`
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：验证崩溃根因（第 17 节）。zp1（talker 用独立 stream）和 zg1（FlashInfer 计划表保护）都没能阻止崩溃；能阻止崩溃的设置（code2wav 单独进程、两种 CUDA graph 都关、CUDA_LAUNCH_BLOCKING=1）都减少或消除了talker 和 code2wav 的 kernel 在同一个 CUDA context 里真正并发执行。这里设 CUDA_DEVICE_MAX_CONNECTIONS=1，让所有 stream 共用一条硬件工作队列，不同 stream 的 kernel 基本只能依次执行；其余与 z4 相同（正式运行环境，无补丁）。
- 运行记录：结果 `20261112_zc1_minicpm_crash_one_hw_queue_run01`：ok，服务退出码 s0=1，w3b_soak 跑了 2/8 档，6 分钟

#### `zg3_minicpm_crash_equal_stream_priority`
- 手动步骤：srun 返回码 0；资源 1 卡、8 核、90 GB（Slurm 记账）
- 配置：`configs/experiments/a6000/zg3_minicpm_crash_equal_stream_priority.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，参数 `--code2wav.factory.decode_stream_priority 0`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：验证崩溃根因（第 17 节）。zp1 让 talker 改用独立 CUDA stream 后仍在约 6 分钟崩溃，所以共用 stream 0 不是必要条件。剩下的进程内耦合是队列优先级：code2wav 的 decode_stream 优先级为 -1（minicpm_o/config.py:26，“runs ahead of the default stream”），会抢在 talker 前面执行。这里只把它设为 0（--code2wav.factory.decode_stream_priority 0），其余与 z4 相同（正式运行环境，无补丁）。
- 运行记录：结果 `20261112_zg3_minicpm_crash_equal_stream_priority_run01`：ok，服务退出码 s0=1，w3b_soak 跑了 1/8 档，4 分钟

#### `zh1_minicpm_crash_hift_caller_stream`
- 手动步骤：srun 返回码 0；资源 1 卡、8 核、90 GB（Slurm 记账）；第一次建步骤时 slurmctld 超时（步骤 409712.54 未运行即取消），srun 自动重试为步骤 409712.55
- 配置：`configs/experiments/a6000/zh1_minicpm_crash_hift_caller_stream.json`（硬件 a6000x1，运行环境 unicorn_diag，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s，启动前缀 `env SGLANG_OMNI_DIAG_HIFT_CALLER_STREAM=1`
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：验证崩溃根因（第 17 节）。zp1（talker 改用独立 stream）、zg1（FlashInfer 计划表保护）、zc1（单硬件队列）、zg3（两条 stream 同优先级）都仍然崩溃。同一进程里唯一明确把 code2wav 线程和其他线程绑在一起的代码，是 HiFT 每次 STFT/iSTFT 都切到 stream 0 并双向等待（token2wav/hift.py:38-50）。这里用诊断副本（同一提交 520b49ee，加 configs/runtime/patches/sglang_omni_diag_hift_caller_stream.patch），设 SGLANG_OMNI_DIAG_HIFT_CALLER_STREAM=1，让 cuFFT 留在 code2wav 自己的 decode_stream 上，code2wav 线程不再碰 stream 0；同一进程、CUDA graph、torch.compile 都保持默认，负载与 z4 相同。代价：新 cuFFT 计划的表可能还没上传完就被读取（上游加这段代码的原因），最坏是个别请求音频出错，不影响判断是否崩溃。
- 运行记录：结果 `20261112_zh1_minicpm_crash_hift_caller_stream_run01`：ok，服务退出码 s0=1，w3b_soak 跑了 1/8 档，5 分钟

#### `zm1_minicpm_crash_code2wav_private_pool`
- 手动步骤：srun 返回码 0；资源 1 卡、8 核、90 GB（Slurm 记账）
- 配置：`configs/experiments/a6000/zm1_minicpm_crash_code2wav_private_pool.json`（硬件 a6000x1，运行环境 unicorn_diag，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s，启动前缀 `env SGLANG_OMNI_DIAG_PRIVATE_POOL_STAGES=code2wav`
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：验证崩溃根因（第 17 节）。同一进程里，talker 线程、主线程和 code2wav 线程共用 PyTorch 的显存缓存分配器；code2wav 单独进程（z11b）不崩，可能是因为分配器分开了。这里用诊断副本（同一提交 520b49ee，加 configs/runtime/patches/sglang_omni_diag_private_stream.patch 里的 MemPool 开关），设 SGLANG_OMNI_DIAG_PRIVATE_POOL_STAGES=code2wav，让 code2wav 调度线程的所有显存分配都走一个私有 MemPool（torch.cuda.use_mem_pool，只作用于当前线程），其他一切保持默认（同一进程、stream 0 上的 HiFT、CUDA graph、torch.compile），负载与 z4 相同。若不再崩溃，说明跨线程复用同一块缓存显存是必要条件；若仍崩溃，分配器共享可以排除。
- 运行记录：结果 `20261112_zm1_minicpm_crash_code2wav_private_pool_run01`：ok，服务退出码 s0=1，w3b_soak 跑了 2/8 档，6 分钟

#### `zs1_minicpm_crash_serialize_gpu`
- 手动步骤：srun 返回码 0；资源 1 卡、8 核、90 GB（Slurm 记账）
- 配置：`configs/experiments/a6000/zs1_minicpm_crash_serialize_gpu.json`（硬件 a6000x1，运行环境 unicorn_diag，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s，启动前缀 `env SGLANG_OMNI_DIAG_SERIALIZE_GPU=1`
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：验证崩溃根因（第 17 节），只在 zm1 仍崩溃时运行。zh1（HiFT 不碰 stream 0）也在 20 秒内崩溃。这里用诊断副本（同一提交 520b49ee，加 configs/runtime/patches/sglang_omni_diag_serialize_gpu.patch），设 SGLANG_OMNI_DIAG_SERIALIZE_GPU=1：code2wav 每个 batch 开始前请求独占，talker 在事件循环顶部看到请求就同步设备并暂停，code2wav 跑完并同步后再放行。这样两个调度线程的 GPU 工作在时间上不重叠，同一进程、同一分配器、CUDA graph、torch.compile 都不变（主线程的 CUDA IPC 拷贝仍可能与 code2wav 重叠）。负载与 z4 相同；talker 被暂停会降低吞吐，可能出现排队超时，判断以日志里是否出现 illegal memory access 为准。
- 运行记录：结果 `20261112_zs1_minicpm_crash_serialize_gpu_run01`：ok，22 分钟

#### `zs2_minicpm_crash_serialize_hift`
- 手动步骤：srun 返回码 0；资源 1 卡、8 核、90 GB（Slurm 记账）
- 配置：`configs/experiments/a6000/zs2_minicpm_crash_serialize_hift.json`（硬件 a6000x1，运行环境 unicorn_diag，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s，启动前缀 `env SGLANG_OMNI_DIAG_SERIALIZE_GPU=hift`
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：验证崩溃根因（第 17 节），只在 zs1（整段 code2wav batch 串行化）不崩时运行。同一补丁，设 SGLANG_OMNI_DIAG_SERIALIZE_GPU=hift：只在 HiFT 阶段（含最后的 .cpu()）让 talker 暂停，flow 阶段仍与 talker 并发。所有崩溃都在某个 code2wav batch 开始后 204–552 ms 报出，大致对应 batch 的后段。若不再崩溃，说明需要的是 HiFT 阶段与 talker 并发；若仍崩溃，说明 flow 阶段的并发就足以触发。负载与 z4 相同。
- 运行记录：结果 `20261112_zs2_minicpm_crash_serialize_hift_run01`：ok，31 分钟

## 8 卡计划 v1

### 来源：`configs/experiments/plan_a6000_8gpu.json`

#### `y0_envcheck`
- 计划条目：优先级 100；依赖 无；申请 1 卡、4 核、32 GB；预计 15 分钟；超时 40 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000/x0_envcheck.json`（硬件 a6000x1，运行环境 unicorn，素材 —，种子 0）
- 阶段：
  1. `idle` 空闲 1 s
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：运行环境检查：导入 torch/sglang/sglang_omni，记录驱动、GPU、拓扑和包版本。不启动服务。
- 运行记录：没有运行记录

#### `y1_minicpm_replicas_n8`
- 计划条目：优先级 90；依赖 `y0_envcheck`；申请 8 卡、56 核、880 GB；预计 50 分钟；超时 125 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu/y1_minicpm_replicas_n8.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，启动时限 2400 s
- 服务 r2：minicpm_o_4_5，GPU 2，启动时限 2400 s
- 服务 r3：minicpm_o_4_5，GPU 3，启动时限 2400 s
- 服务 r4：minicpm_o_4_5，GPU 4，启动时限 2400 s
- 服务 r5：minicpm_o_4_5，GPU 5，启动时限 2400 s
- 服务 r6：minicpm_o_4_5，GPU 6，启动时限 2400 s
- 服务 r7：minicpm_o_4_5，GPU 7，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b` 闭环：W3b（文本提问→文本+语音）；并发 16, 32, 64, 128；每档 256 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
  3. `replay_text_dep` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_text_c4.jsonl`，模式 dependency，速度 1×，会话复制 32 份，会话到达 poisson:0.5，内容 auto
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：副本扩展：8 个完整的 MiniCPM-o 默认语音流水线副本（每卡一个），客户端按在途请求最少路由。并发与回放会话数随副本数等比放大，用来画扩展曲线。
- 运行记录：没有运行记录

#### `y2_minicpm_routing_8rep`
- 计划条目：优先级 89；依赖 `y0_envcheck`；申请 8 卡、56 核、880 GB；预计 50 分钟；超时 125 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu/y2_minicpm_routing_8rep.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 r0：minicpm_o_4_5，GPU 0，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，启动时限 2400 s
- 服务 r2：minicpm_o_4_5，GPU 2，启动时限 2400 s
- 服务 r3：minicpm_o_4_5，GPU 3，启动时限 2400 s
- 服务 r4：minicpm_o_4_5，GPU 4，启动时限 2400 s
- 服务 r5：minicpm_o_4_5，GPU 5，启动时限 2400 s
- 服务 r6：minicpm_o_4_5，GPU 6，启动时限 2400 s
- 服务 r7：minicpm_o_4_5，GPU 7，启动时限 2400 s
- 阶段：
  1. `replay_round_robin` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_text_c4.jsonl`，模式 dependency，速度 1×，会话复制 32 份，会话到达 poisson:2.0，内容 auto；路由 round_robin
  2. `replay_least_outstanding` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_text_c4.jsonl`，模式 dependency，速度 1×，会话复制 32 份，会话到达 poisson:2.0，内容 auto；路由 least_outstanding
  3. `replay_session_affinity` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_text_c4.jsonl`，模式 dependency，速度 1×，会话复制 32 份，会话到达 poisson:2.0，内容 auto；路由 session_affinity
  4. `replay_voice_least` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_voice_c4.jsonl`，模式 dependency，速度 1×，会话复制 16 份，会话到达 poisson:1.0，内容 auto；路由 least_outstanding
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：8 个副本上比较客户端路由策略（轮询 / 在途请求最少 / 按会话固定），负载为 Agent trace 的依赖感知回放（32 倍会话，Poisson 到达）。
- 运行记录：没有运行记录

#### `y1_minicpm_replicas_n4`
- 计划条目：优先级 80；依赖 `y0_envcheck`；申请 4 卡、28 核、440 GB；预计 45 分钟；超时 112 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu/y1_minicpm_replicas_n4.json`（硬件 a6000x4，运行环境 unicorn，素材 synthetic-v1，种子 0，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，启动时限 2400 s
- 服务 r2：minicpm_o_4_5，GPU 2，启动时限 2400 s
- 服务 r3：minicpm_o_4_5，GPU 3，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b` 闭环：W3b（文本提问→文本+语音）；并发 8, 16, 32, 64；每档 128 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
  3. `replay_text_dep` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_text_c4.jsonl`，模式 dependency，速度 1×，会话复制 16 份，会话到达 poisson:0.5，内容 auto
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：副本扩展：4 个完整的 MiniCPM-o 默认语音流水线副本（每卡一个），客户端按在途请求最少路由。并发与回放会话数随副本数等比放大，用来画扩展曲线。
- 运行记录：没有运行记录

#### `y3b_minicpm_4gpu_two_disagg_pairs`
- 计划条目：优先级 79；依赖 `y0_envcheck`；申请 4 卡、28 核、440 GB；预计 45 分钟；超时 112 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu/y3b_minicpm_4gpu_two_disagg_pairs.json`（硬件 a6000x4，运行环境 unicorn，素材 synthetic-v1，种子 0，多服务路由 least_outstanding）
- 服务 p0：minicpm_o_4_5，GPU 0, 1，参数 `--talker.gpu 1 --code2wav.gpu 1`，启动时限 2400 s
- 服务 p1：minicpm_o_4_5，GPU 2, 3，参数 `--talker.gpu 1 --code2wav.gpu 1`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b` 闭环：W3b（文本提问→文本+语音）；并发 8, 16, 32, 64；每档 128 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
  3. `replay_text_dep` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_text_c4.jsonl`，模式 dependency，速度 1×，会话复制 16 份，会话到达 poisson:1.0，内容 auto
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：4 卡预算：两组拆分部署（每组 thinker 一张卡、talker+code2wav 一张卡），与 y1_n4（4 个完整副本）对比。
- 运行记录：没有运行记录

#### `y3c_minicpm_4gpu_talker_replica3`
- 计划条目：优先级 78；依赖 `y0_envcheck`；申请 4 卡、28 核、440 GB；预计 45 分钟；超时 112 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu/y3c_minicpm_4gpu_talker_replica3.json`（硬件 a6000x4，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3，参数 `--config configs/models/minicpm_o_4_5_talker_replica3_4gpu.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b` 闭环：W3b（文本提问→文本+语音）；并发 8, 16, 32, 64；每档 128 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
  3. `replay_text_dep` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_text_c4.jsonl`，模式 dependency，速度 1×，会话复制 16 份，会话到达 poisson:1.0，内容 auto
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：4 卡预算：一个 thinker（GPU 0）配三份 talker+code2wav（GPU 1-3）。Spark 上 thinker 占单请求语音延迟的大部分，这里检验只扩语音阶段的效果。
- 运行记录：没有运行记录

#### `y1_minicpm_replicas_n2`
- 计划条目：优先级 70；依赖 `y0_envcheck`；申请 2 卡、14 核、220 GB；预计 40 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu/y1_minicpm_replicas_n2.json`（硬件 a6000x2，运行环境 unicorn，素材 synthetic-v1，种子 0，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b` 闭环：W3b（文本提问→文本+语音）；并发 4, 8, 16, 32；每档 64 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
  3. `replay_text_dep` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_text_c4.jsonl`，模式 dependency，速度 1×，会话复制 8 份，会话到达 poisson:0.5，内容 auto
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：副本扩展：2 个完整的 MiniCPM-o 默认语音流水线副本（每卡一个），客户端按在途请求最少路由。并发与回放会话数随副本数等比放大，用来画扩展曲线。
- 运行记录：没有运行记录

#### `y4_minicpm_thinker_tp2_text`
- 计划条目：优先级 69；依赖 `y0_envcheck`；申请 2 卡、14 核、220 GB；预计 30 分钟；超时 75 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu/y4_minicpm_thinker_tp2_text.json`（硬件 a6000x2，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0, 1，参数 `--text-only --thinker.process thinker --thinker.tp_size 2 --thinker.gpu [0, 1]`，启动时限 2400 s
- 阶段：
  1. `smoke_w1` 冒烟检查：W1（文本→文本），2 个请求，检查输出非空
  2. `w1` 闭环：W1（文本→文本）；并发 1, 4, 16, 64；每档 64 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：thinker 张量并行：MiniCPM-o text-only，thinker TP=2（GPU 0,1）。与两个 TP=1 副本（y4b）比较延迟与吞吐的取舍。拿到拓扑后，再决定是否在 NVLink 对与非 NVLink 对上各跑一次。
- 运行记录：没有运行记录

#### `y4b_minicpm_two_text_replicas`
- 计划条目：优先级 68；依赖 `y0_envcheck`；申请 2 卡、14 核、220 GB；预计 30 分钟；超时 75 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu/y4b_minicpm_two_text_replicas.json`（硬件 a6000x2，运行环境 unicorn，素材 synthetic-v1，种子 0，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--text-only`，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，参数 `--text-only`，启动时限 2400 s
- 阶段：
  1. `smoke_w1` 冒烟检查：W1（文本→文本），2 个请求，检查输出非空
  2. `w1` 闭环：W1（文本→文本）；并发 1, 4, 16, 64；每档 64 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：y4 的对照：两个 TP=1 的 text-only 副本。
- 运行记录：没有运行记录

#### `y1_minicpm_replicas_n1`
- 计划条目：优先级 60；依赖 `y0_envcheck`；申请 1 卡、7 核、110 GB；预计 40 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu/y1_minicpm_replicas_n1.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b` 闭环：W3b（文本提问→文本+语音）；并发 2, 4, 8, 16；每档 32 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
  3. `replay_text_dep` 回放：trace `latest/x5_minicpm_agents/raw/agent_trace_agents_text_c4.jsonl`，模式 dependency，速度 1×，会话复制 4 份，会话到达 poisson:0.5，内容 auto
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：副本扩展：1 个完整的 MiniCPM-o 默认语音流水线副本（每卡一个），客户端按在途请求最少路由。并发与回放会话数随副本数等比放大，用来画扩展曲线。
- 运行记录：没有运行记录

#### `y6_minicpm_video20s_cpu4`
- 计划条目：优先级 55；依赖 `y0_envcheck`；申请 1 卡、4 核、110 GB；预计 30 分钟；超时 75 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu/y6_minicpm_video20s_cpu4.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，参数 `--text-only`，启动时限 2400 s
- 阶段：
  1. `w4_video_20s` 闭环：W4（视频+文本→文本），视频 20 s（素材标签 count_circles）；并发 1, 2, 4, 8；每档 24 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：CPU 配额扩展：20 秒视频理解，step 分配 4 个 CPU（与 8 个 CPU 的 x4 对照）。
- 运行记录：没有运行记录

#### `y6_minicpm_video20s_cpu16`
- 计划条目：优先级 54；依赖 `y0_envcheck`；申请 1 卡、16 核、110 GB；预计 30 分钟；超时 75 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu/y6_minicpm_video20s_cpu16.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，参数 `--text-only`，启动时限 2400 s
- 阶段：
  1. `w4_video_20s` 闭环：W4（视频+文本→文本），视频 20 s（素材标签 count_circles）；并发 1, 2, 4, 8；每档 24 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：CPU 配额扩展：20 秒视频理解，step 分配 16 个 CPU（与 8 个 CPU 的 x4 对照）。
- 运行记录：没有运行记录

#### `y5_qwen3_omni_two_pipelines_6gpu`
- 计划条目：优先级 50；依赖 `y0_envcheck`；申请 6 卡、42 核、660 GB；预计 70 分钟；超时 175 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu/y5_qwen3_omni_two_pipelines_6gpu.json`（硬件 a6000x6，运行环境 unicorn，素材 synthetic-v1，种子 0，多服务路由 least_outstanding）
- 服务 p0：qwen3_omni_30b_a3b，GPU 0, 1, 2，参数 `--model-name qwen3-omni --thinker.tp_size 2 --thinker.gpu [0, 1] --talker_ar.gpu 2 --thinker.engine.disable_custom_all_reduce true`，启动时限 3600 s，环境变量 `NCCL_P2P_DISABLE=1`
- 服务 p1：qwen3_omni_30b_a3b，GPU 3, 4, 5，参数 `--model-name qwen3-omni --thinker.tp_size 2 --thinker.gpu [0, 1] --talker_ar.gpu 2 --thinker.engine.disable_custom_all_reduce true`，启动时限 3600 s，环境变量 `NCCL_P2P_DISABLE=1`
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b` 闭环：W3b（文本提问→文本+语音）；并发 2, 4, 8, 16；每档 48 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：Qwen3-Omni 扩展：两条完整流水线（每条 thinker TP=2 + talker，3 卡），客户端路由。与 Job 409712 的单条流水线（x9）对比。
- 运行记录：没有运行记录

## plan v2 第一批（8×A6000）

### 来源：`configs/experiments/plan_a6000_8gpu_v2.json`

#### `v2_y0_env_topology`
- 计划条目：优先级 100；依赖 无；申请 8 卡、54 核、780 GB；预计 15 分钟；超时 45 分钟；失败后最多重试 1 次
- 命令：`bash scripts/slurm/env_topology.sh unicorn minicpm_o_4_5`
- 计划里的说明：Y0：GPU 拓扑、NVLink、点对点拷贝带宽、NVLink 对上的 NCCL all-reduce，计时预热 MiniCPM-o 权重文件（P7）
- 运行记录：编排器：成功（尝试 1 次），11-15 01:55 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 0.40 h；结果 `20261115_v2_y0_env_topology_run01`：ok

#### `v2_y0s_r1_default_soak`
- 计划条目：优先级 98；依赖 无；申请 1 卡、7 核、95 GB；预计 45 分钟；超时 120 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_y0s_r1_default_soak.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 20 次）
- 服务 r0：minicpm_o_4_5，GPU 0，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 1 时停止；服务崩溃的档不重跑
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：plan v2 稳定性门槛的对照：单卡默认配置（talker 与 code2wav 同进程），W3b 开环 1.5 请求/秒（约为容量的一半）× 16 档 × 2 分钟。服务崩溃后自动重启、接着跑下一档，用来数每小时崩溃几次（第 17 节平均约 3 分钟一次）。
- 运行记录：编排器：成功（尝试 1 次），11-15 01:58 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 0.94 h；结果 `20261115_v2_y0s_r1_default_soak_run01`：ok，服务重启 4 次，w3b_soak 有 4 档受服务退出影响，52 分钟

#### `v2_y0s_p1_soak`
- 计划条目：优先级 97；依赖 无；申请 2 卡、14 核、190 GB；预计 40 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_y0s_p1_soak.json`（硬件 a6000x2，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 10 次）
- 服务 p0：minicpm_o_4_5，GPU 0, 1，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `probe` 闭环：W3b（文本提问→文本+语音）；并发 8, 16, 32；每档 48 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2
  3. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `probe` 阶段的闭环最高吞吐 × 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 1 时停止；服务崩溃的档不重跑
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：plan v2 稳定性门槛：1 个拆分服务（每个 1 张 thinker 卡 + 1 张 talker 卡，code2wav 单独进程），客户端按在途请求最少路由。先用闭环（并发 8, 16, 32）测容量，再以一半容量开环 14 档 × 2 分钟。崩溃后自动重启并记录，不重跑。
- 运行记录：编排器：成功（尝试 1 次），11-15 01:58 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 1.36 h；结果 `20261115_v2_y0s_p1_soak_run01`：ok，37 分钟

#### `v2_y0s_s1x3_soak`
- 计划条目：优先级 96；依赖 无；申请 4 卡、27 核、390 GB；预计 40 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_y0s_s1x3_soak.json`（硬件 a6000x4，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 10 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3，参数 `--config configs/models/minicpm_o_4_5_v2_s1x3.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `probe` 闭环：W3b（文本提问→文本+语音）；并发 16, 32, 64；每档 48 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2
  3. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `probe` 阶段的闭环最高吞吐 × 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 1 时停止；服务崩溃的档不重跑
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：plan v2 稳定性门槛：一个服务：1 个 thinker（pipeline）副本 + 3 个 talker 副本，code2wav 单独进程、与 talker 同卡。先用闭环（并发 16, 32, 64）测容量，再以一半容量开环 14 档 × 2 分钟。崩溃后自动重启并记录，不重跑。
- 运行记录：编排器：成功（尝试 1 次），11-15 01:58 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 2.59 h；结果 `20261115_v2_y0s_s1x3_soak_run01`：ok，35 分钟

#### `v2_a_r1`
- 计划条目：优先级 95；依赖 无；申请 1 卡、7 核、95 GB；预计 30 分钟；超时 90 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_a_r1.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 1, 2, 4, 8, 16, 32；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `w3b_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 w3b_closed；torch profiler 无；nsys 无
- 配置里的说明：plan v2 A 的单卡基线 R1（与稳定性门槛同时运行；并发 1–16 与 x3 对照）：1 个完整副本（每卡一个服务，code2wav 单独进程），客户端按在途请求最少路由。W3b 闭环并发 1–32（每档 max(96, 2×并发) 个请求，开事件记录），再按闭环测得的容量做开环泊松 50%, 75%, 90%, 100%, 120%，每档 2 分钟。
- 运行记录：编排器：成功（尝试 1 次），11-15 01:58 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 0.49 h；结果 `20261115_v2_a_r1_run01`：ok，25 分钟

#### `v2_a_r8_run1`
- 计划条目：优先级 90；依赖 无；申请 8 卡、54 核、780 GB；预计 28 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_a_r8_run1.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r2：minicpm_o_4_5，GPU 2，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r3：minicpm_o_4_5，GPU 3，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r4：minicpm_o_4_5，GPU 4，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r5：minicpm_o_4_5，GPU 5，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r6：minicpm_o_4_5，GPU 6，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r7：minicpm_o_4_5，GPU 7，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 16, 32, 64, 128, 256；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `w3b_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
  4. `w3b_torch_profile_c64` 闭环：W3b（文本提问→文本+语音）；并发 64；每档 64 个请求（另加预热请求：16 个，不计入统计）；停止条件：错误率 > 0.2；**torch profiler**
- 额外采集：服务端事件 w3b_closed；torch profiler w3b_torch_profile_c64；nsys 无
- 配置里的说明：plan v2 A（8 卡，第 1 次）：8 个完整副本（每卡一个服务，code2wav 单独进程），客户端按在途请求最少路由。W3b 闭环并发 16–256（每档 max(96, 2×并发) 个请求，开事件记录），再按闭环测得的容量做开环泊松 50%, 75%, 90%, 100%, 120%，每档 2 分钟。最后在并发 64 开一个 torch profiler 窗口。
- 运行记录：编排器：成功（尝试 1 次），11-15 02:55 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 2.25 h；结果 `20261115_v2_a_r8_run1_run01`：ok，17 分钟

#### `v2_a_p4_run1`
- 计划条目：优先级 89；依赖 无；申请 8 卡、54 核、780 GB；预计 25 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_a_p4_run1.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 p0：minicpm_o_4_5，GPU 0, 1，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 p1：minicpm_o_4_5，GPU 2, 3，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 p2：minicpm_o_4_5，GPU 4, 5，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 p3：minicpm_o_4_5，GPU 6, 7，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 16, 32, 64, 128, 256；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `w3b_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 w3b_closed；torch profiler 无；nsys 无
- 配置里的说明：plan v2 A（8 卡，第 1 次）：4 个拆分服务（每个 1 张 thinker 卡 + 1 张 talker 卡，code2wav 单独进程），客户端按在途请求最少路由。W3b 闭环并发 16–256（每档 max(96, 2×并发) 个请求，开事件记录），再按闭环测得的容量做开环泊松 50%, 75%, 90%, 100%, 120%，每档 2 分钟。
- 运行记录：编排器：成功（尝试 1 次），11-15 03:12 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 2.01 h；结果 `20261115_v2_a_p4_run1_run01`：ok，15 分钟

#### `v2_a_s1x7_run1`
- 计划条目：优先级 88；依赖 无；申请 8 卡、54 核、780 GB；预计 25 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_a_s1x7_run1.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3, 4, 5, 6, 7，参数 `--config configs/models/minicpm_o_4_5_v2_s1x7.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 16, 32, 64, 128, 256；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `w3b_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 w3b_closed；torch profiler 无；nsys 无
- 配置里的说明：plan v2 A（8 卡，第 1 次）：一个服务：1 个 thinker（pipeline）副本 + 7 个 talker 副本，code2wav 单独进程、与 talker 同卡。W3b 闭环并发 16–256（每档 max(96, 2×并发) 个请求，开事件记录），再按闭环测得的容量做开环泊松 50%, 75%, 90%, 100%, 120%，每档 2 分钟。
- 运行记录：编排器：成功（尝试 1 次），11-15 03:27 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 2.10 h；结果 `20261115_v2_a_s1x7_run1_run01`：ok，16 分钟

#### `v2_a_s2x6_run1`
- 计划条目：优先级 87；依赖 无；申请 8 卡、54 核、780 GB；预计 28 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_a_s2x6_run1.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3, 4, 5, 6, 7，参数 `--config configs/models/minicpm_o_4_5_v2_s2x6.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 16, 32, 64, 128, 256；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `w3b_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
  4. `w3b_torch_profile_c64` 闭环：W3b（文本提问→文本+语音）；并发 64；每档 64 个请求（另加预热请求：16 个，不计入统计）；停止条件：错误率 > 0.2；**torch profiler**
- 额外采集：服务端事件 w3b_closed；torch profiler w3b_torch_profile_c64；nsys 无
- 配置里的说明：plan v2 A（8 卡，第 1 次）：一个服务：2 个 thinker（pipeline）副本 + 6 个 talker 副本，code2wav 单独进程、与 talker 同卡。W3b 闭环并发 16–256（每档 max(96, 2×并发) 个请求，开事件记录），再按闭环测得的容量做开环泊松 50%, 75%, 90%, 100%, 120%，每档 2 分钟。最后在并发 64 开一个 torch profiler 窗口。
- 运行记录：编排器：成功（尝试 1 次），11-15 03:42 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 1.96 h；结果 `20261115_v2_a_s2x6_run1_run01`：ok，15 分钟

#### `v2_a_s3x5_run1`
- 计划条目：优先级 86；依赖 无；申请 8 卡、54 核、780 GB；预计 25 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_a_s3x5_run1.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3, 4, 5, 6, 7，参数 `--config configs/models/minicpm_o_4_5_v2_s3x5.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 16, 32, 64, 128, 256；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `w3b_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 w3b_closed；torch profiler 无；nsys 无
- 配置里的说明：plan v2 A（8 卡，第 1 次）：一个服务：3 个 thinker（pipeline）副本 + 5 个 talker 副本，code2wav 单独进程、与 talker 同卡。W3b 闭环并发 16–256（每档 max(96, 2×并发) 个请求，开事件记录），再按闭环测得的容量做开环泊松 50%, 75%, 90%, 100%, 120%，每档 2 分钟。
- 运行记录：编排器：成功（尝试 1 次），11-15 03:57 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 1.98 h；结果 `20261115_v2_a_s3x5_run1_run01`：ok，15 分钟

#### `v2_a_s4x4_run1`
- 计划条目：优先级 85；依赖 无；申请 8 卡、54 核、780 GB；预计 25 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_a_s4x4_run1.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3, 4, 5, 6, 7，参数 `--config configs/models/minicpm_o_4_5_v2_s4x4.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 16, 32, 64, 128, 256；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `w3b_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 w3b_closed；torch profiler 无；nsys 无
- 配置里的说明：plan v2 A（8 卡，第 1 次）：一个服务：4 个 thinker（pipeline）副本 + 4 个 talker 副本，code2wav 单独进程、与 talker 同卡。W3b 闭环并发 16–256（每档 max(96, 2×并发) 个请求，开事件记录），再按闭环测得的容量做开环泊松 50%, 75%, 90%, 100%, 120%，每档 2 分钟。
- 运行记录：编排器：成功（尝试 1 次），11-15 04:12 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 2.02 h；结果 `20261115_v2_a_s4x4_run1_run01`：ok，15 分钟

#### `v2_a_s4x4_run2`
- 计划条目：优先级 84；依赖 无；申请 8 卡、54 核、780 GB；预计 25 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_a_s4x4_run2.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3, 4, 5, 6, 7，参数 `--config configs/models/minicpm_o_4_5_v2_s4x4.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 16, 32, 64, 128, 256；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `w3b_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 w3b_closed；torch profiler 无；nsys 无
- 配置里的说明：plan v2 A（8 卡，第 2 次）：一个服务：4 个 thinker（pipeline）副本 + 4 个 talker 副本，code2wav 单独进程、与 talker 同卡。W3b 闭环并发 16–256（每档 max(96, 2×并发) 个请求，开事件记录），再按闭环测得的容量做开环泊松 50%, 75%, 90%, 100%, 120%，每档 2 分钟。
- 运行记录：编排器：成功（尝试 1 次），11-15 04:27 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 1.98 h；结果 `20261115_v2_a_s4x4_run2_run01`：ok，15 分钟

#### `v2_a_s3x5_run2`
- 计划条目：优先级 83；依赖 无；申请 8 卡、54 核、780 GB；预计 25 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_a_s3x5_run2.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3, 4, 5, 6, 7，参数 `--config configs/models/minicpm_o_4_5_v2_s3x5.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 16, 32, 64, 128, 256；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `w3b_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 w3b_closed；torch profiler 无；nsys 无
- 配置里的说明：plan v2 A（8 卡，第 2 次）：一个服务：3 个 thinker（pipeline）副本 + 5 个 talker 副本，code2wav 单独进程、与 talker 同卡。W3b 闭环并发 16–256（每档 max(96, 2×并发) 个请求，开事件记录），再按闭环测得的容量做开环泊松 50%, 75%, 90%, 100%, 120%，每档 2 分钟。
- 运行记录：编排器：成功（尝试 1 次），11-15 04:42 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 1.99 h；结果 `20261115_v2_a_s3x5_run2_run01`：ok，15 分钟

#### `v2_a_s2x6_run2`
- 计划条目：优先级 82；依赖 无；申请 8 卡、54 核、780 GB；预计 25 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_a_s2x6_run2.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3, 4, 5, 6, 7，参数 `--config configs/models/minicpm_o_4_5_v2_s2x6.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 16, 32, 64, 128, 256；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `w3b_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 w3b_closed；torch profiler 无；nsys 无
- 配置里的说明：plan v2 A（8 卡，第 2 次）：一个服务：2 个 thinker（pipeline）副本 + 6 个 talker 副本，code2wav 单独进程、与 talker 同卡。W3b 闭环并发 16–256（每档 max(96, 2×并发) 个请求，开事件记录），再按闭环测得的容量做开环泊松 50%, 75%, 90%, 100%, 120%，每档 2 分钟。
- 运行记录：编排器：成功（尝试 1 次），11-15 04:57 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 1.98 h；结果 `20261115_v2_a_s2x6_run2_run01`：ok，15 分钟

#### `v2_a_s1x7_run2`
- 计划条目：优先级 81；依赖 无；申请 8 卡、54 核、780 GB；预计 25 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_a_s1x7_run2.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3, 4, 5, 6, 7，参数 `--config configs/models/minicpm_o_4_5_v2_s1x7.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 16, 32, 64, 128, 256；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `w3b_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 w3b_closed；torch profiler 无；nsys 无
- 配置里的说明：plan v2 A（8 卡，第 2 次）：一个服务：1 个 thinker（pipeline）副本 + 7 个 talker 副本，code2wav 单独进程、与 talker 同卡。W3b 闭环并发 16–256（每档 max(96, 2×并发) 个请求，开事件记录），再按闭环测得的容量做开环泊松 50%, 75%, 90%, 100%, 120%，每档 2 分钟。
- 运行记录：编排器：成功（尝试 1 次），11-15 05:12 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 1.99 h；结果 `20261115_v2_a_s1x7_run2_run01`：ok，15 分钟

#### `v2_a_p4_run2`
- 计划条目：优先级 80；依赖 无；申请 8 卡、54 核、780 GB；预计 25 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_a_p4_run2.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 p0：minicpm_o_4_5，GPU 0, 1，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 p1：minicpm_o_4_5，GPU 2, 3，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 p2：minicpm_o_4_5，GPU 4, 5，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 p3：minicpm_o_4_5，GPU 6, 7，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 16, 32, 64, 128, 256；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `w3b_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 w3b_closed；torch profiler 无；nsys 无
- 配置里的说明：plan v2 A（8 卡，第 2 次）：4 个拆分服务（每个 1 张 thinker 卡 + 1 张 talker 卡，code2wav 单独进程），客户端按在途请求最少路由。W3b 闭环并发 16–256（每档 max(96, 2×并发) 个请求，开事件记录），再按闭环测得的容量做开环泊松 50%, 75%, 90%, 100%, 120%，每档 2 分钟。
- 运行记录：编排器：成功（尝试 1 次），11-15 05:27 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 2.09 h；结果 `20261115_v2_a_p4_run2_run01`：ok，16 分钟

#### `v2_a_r8_run2`
- 计划条目：优先级 79；依赖 无；申请 8 卡、54 核、780 GB；预计 25 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_a_r8_run2.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r2：minicpm_o_4_5，GPU 2，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r3：minicpm_o_4_5，GPU 3，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r4：minicpm_o_4_5，GPU 4，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r5：minicpm_o_4_5，GPU 5，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r6：minicpm_o_4_5，GPU 6，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r7：minicpm_o_4_5，GPU 7，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 16, 32, 64, 128, 256；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `w3b_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 w3b_closed；torch profiler 无；nsys 无
- 配置里的说明：plan v2 A（8 卡，第 2 次）：8 个完整副本（每卡一个服务，code2wav 单独进程），客户端按在途请求最少路由。W3b 闭环并发 16–256（每档 max(96, 2×并发) 个请求，开事件记录），再按闭环测得的容量做开环泊松 50%, 75%, 90%, 100%, 120%，每档 2 分钟。
- 运行记录：编排器：成功（尝试 1 次），11-15 05:42 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 2.16 h；结果 `20261115_v2_a_r8_run2_run01`：ok，16 分钟

#### `v2_b2_r8`
- 计划条目：优先级 78；依赖 无；申请 8 卡、54 核、780 GB；预计 45 分钟；超时 150 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_b2_r8.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r2：minicpm_o_4_5，GPU 2，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r3：minicpm_o_4_5，GPU 3，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r4：minicpm_o_4_5，GPU 4，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r5：minicpm_o_4_5，GPU 5，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r6：minicpm_o_4_5，GPU 6，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r7：minicpm_o_4_5，GPU 7，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `lo_text_c64` Agent（合成任务，真实模型请求，录 trace）：128 个会话，并发 64，种子 7100，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）；路由 least_outstanding
  3. `lo_text_c128` Agent（合成任务，真实模型请求，录 trace）：256 个会话，并发 128，种子 7101，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）；路由 least_outstanding
  4. `lo_text_c256` Agent（合成任务，真实模型请求，录 trace）：512 个会话，并发 256，种子 7102，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）；**服务端事件**，路由 least_outstanding
  5. `lo_text_c512` Agent（合成任务，真实模型请求，录 trace）：1024 个会话，并发 512，种子 7103，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）；路由 least_outstanding
  6. `lo_voice_c64` Agent（合成任务，真实模型请求，录 trace）：128 个会话，并发 64，种子 7150，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）；路由 least_outstanding
  7. `lo_voice_c128` Agent（合成任务，真实模型请求，录 trace）：256 个会话，并发 128，种子 7151，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）；路由 least_outstanding
  8. `lo_voice_c256` Agent（合成任务，真实模型请求，录 trace）：512 个会话，并发 256，种子 7152，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）；路由 least_outstanding
  9. `lo_voice_c512` Agent（合成任务，真实模型请求，录 trace）：1024 个会话，并发 512，种子 7153，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）；路由 least_outstanding
  10. `sa_text_c64` Agent（合成任务，真实模型请求，录 trace）：128 个会话，并发 64，种子 7200，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）；路由 session_affinity
  11. `sa_text_c128` Agent（合成任务，真实模型请求，录 trace）：256 个会话，并发 128，种子 7201，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）；路由 session_affinity
  12. `sa_text_c256` Agent（合成任务，真实模型请求，录 trace）：512 个会话，并发 256，种子 7202，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）；**服务端事件**，路由 session_affinity
  13. `sa_text_c512` Agent（合成任务，真实模型请求，录 trace）：1024 个会话，并发 512，种子 7203，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）；路由 session_affinity
  14. `sa_voice_c64` Agent（合成任务，真实模型请求，录 trace）：128 个会话，并发 64，种子 7250，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）；路由 session_affinity
  15. `sa_voice_c128` Agent（合成任务，真实模型请求，录 trace）：256 个会话，并发 128，种子 7251，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）；路由 session_affinity
  16. `sa_voice_c256` Agent（合成任务，真实模型请求，录 trace）：512 个会话，并发 256，种子 7252，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）；路由 session_affinity
  17. `sa_voice_c512` Agent（合成任务，真实模型请求，录 trace）：1024 个会话，并发 512，种子 7253，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）；路由 session_affinity
- 额外采集：服务端事件 lo_text_c256、sa_text_c256；torch profiler 无；nsys 无
- 配置里的说明：plan v2 B2：8 个完整副本上的大规模 Agent（会话不重复），同时进行的会话 64–512，每档会话总数为并发的 2 倍；文本 Agent 与语音 Agent（文本提问、最后一步语音回答），客户端路由分别用在途请求最少和按会话固定。
- 运行记录：编排器：成功（尝试 1 次），11-15 05:59 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 1.04 h；结果 `20261115_v2_b2_r8_run01`：ok，8 分钟

#### `v2_b2_s2x6`
- 计划条目：优先级 77；依赖 无；申请 8 卡、54 核、780 GB；预计 40 分钟；超时 150 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_b2_s2x6.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3, 4, 5, 6, 7，参数 `--config configs/models/minicpm_o_4_5_v2_s2x6.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `text_c64` Agent（合成任务，真实模型请求，录 trace）：128 个会话，并发 64，种子 7300，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）
  3. `text_c128` Agent（合成任务，真实模型请求，录 trace）：256 个会话，并发 128，种子 7301，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）
  4. `text_c256` Agent（合成任务，真实模型请求，录 trace）：512 个会话，并发 256，种子 7302，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）；**服务端事件**
  5. `text_c512` Agent（合成任务，真实模型请求，录 trace）：1024 个会话，并发 512，种子 7303，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）
  6. `voice_c64` Agent（合成任务，真实模型请求，录 trace）：128 个会话，并发 64，种子 7350，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）
  7. `voice_c128` Agent（合成任务，真实模型请求，录 trace）：256 个会话，并发 128，种子 7351，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）
  8. `voice_c256` Agent（合成任务，真实模型请求，录 trace）：512 个会话，并发 256，种子 7352，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）
  9. `voice_c512` Agent（合成任务，真实模型请求，录 trace）：1024 个会话，并发 512，种子 7353，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）
  10. `toollat0p5_text_c256` Agent（合成任务，真实模型请求，录 trace）：512 个会话，并发 256，种子 7390，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），模拟工具延迟 对数正态（中位数 0.5 s，σ 0.5）
  11. `toollat2_text_c256` Agent（合成任务，真实模型请求，录 trace）：512 个会话，并发 256，种子 7391，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），模拟工具延迟 对数正态（中位数 2.0 s，σ 0.5）
- 额外采集：服务端事件 text_c256；torch profiler 无；nsys 无
- 配置里的说明：plan v2 B2：一个服务：2 个 thinker（pipeline）副本 + 6 个 talker 副本，code2wav 单独进程、与 talker 同卡（按假设 A-H1 预选的池化布局）上的大规模 Agent，同 v2_b2_r8；另加 256 个会话时工具延迟 0.5 s、2 s（对数正态中位数）两档，与 text_c256（无工具延迟）一起看工具等待让各池空闲多少。
- 运行记录：编排器：成功（尝试 1 次），11-15 06:07 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 0.76 h；结果 `20261115_v2_b2_s2x6_run01`：ok，6 分钟

#### `v2_b2_p4`
- 计划条目：优先级 76；依赖 无；申请 8 卡、54 核、780 GB；预计 30 分钟；超时 120 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_b2_p4.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 p0：minicpm_o_4_5，GPU 0, 1，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 p1：minicpm_o_4_5，GPU 2, 3，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 p2：minicpm_o_4_5，GPU 4, 5，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 p3：minicpm_o_4_5，GPU 6, 7，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `text_c64` Agent（合成任务，真实模型请求，录 trace）：128 个会话，并发 64，种子 7400，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）
  3. `text_c128` Agent（合成任务，真实模型请求，录 trace）：256 个会话，并发 128，种子 7401，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）
  4. `text_c256` Agent（合成任务，真实模型请求，录 trace）：512 个会话，并发 256，种子 7402，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）；**服务端事件**
  5. `text_c512` Agent（合成任务，真实模型请求，录 trace）：1024 个会话，并发 512，种子 7403，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）
  6. `voice_c64` Agent（合成任务，真实模型请求，录 trace）：128 个会话，并发 64，种子 7450，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）
  7. `voice_c128` Agent（合成任务，真实模型请求，录 trace）：256 个会话，并发 128，种子 7451，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）
  8. `voice_c256` Agent（合成任务，真实模型请求，录 trace）：512 个会话，并发 256，种子 7452，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）
  9. `voice_c512` Agent（合成任务，真实模型请求，录 trace）：1024 个会话，并发 512，种子 7453，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）
- 额外采集：服务端事件 text_c256；torch profiler 无；nsys 无
- 配置里的说明：plan v2 B2：4 个拆分服务（每个 1 张 thinker 卡 + 1 张 talker 卡，code2wav 单独进程），客户端按在途请求最少路由上的大规模 Agent，同 v2_b2_r8 的在途请求最少路由部分。
- 运行记录：编排器：成功（尝试 1 次），11-15 06:12 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 0.67 h；结果 `20261115_v2_b2_p4_run01`：ok，5 分钟

#### `v2_a_r4_run1`
- 计划条目：优先级 70；依赖 无；申请 4 卡、27 核、390 GB；预计 25 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_a_r4_run1.json`（硬件 a6000x4，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r2：minicpm_o_4_5，GPU 2，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r3：minicpm_o_4_5，GPU 3，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 8, 16, 32, 64, 128；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `w3b_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 w3b_closed；torch profiler 无；nsys 无
- 配置里的说明：plan v2 A（4 卡，第 1 次）：4 个完整副本（每卡一个服务，code2wav 单独进程），客户端按在途请求最少路由。W3b 闭环并发 8–128（每档 max(96, 2×并发) 个请求，开事件记录），再按闭环测得的容量做开环泊松 50%, 75%, 90%, 100%, 120%，每档 2 分钟。
- 运行记录：编排器：成功（尝试 1 次），11-15 06:17 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 1.06 h；结果 `20261115_v2_a_r4_run1_run01`：ok，16 分钟

#### `v2_a_p2_run1`
- 计划条目：优先级 69；依赖 无；申请 4 卡、27 核、390 GB；预计 25 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_a_p2_run1.json`（硬件 a6000x4，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 p0：minicpm_o_4_5，GPU 0, 1，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 p1：minicpm_o_4_5，GPU 2, 3，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 8, 16, 32, 64, 128；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `w3b_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 w3b_closed；torch profiler 无；nsys 无
- 配置里的说明：plan v2 A（4 卡，第 1 次）：2 个拆分服务（每个 1 张 thinker 卡 + 1 张 talker 卡，code2wav 单独进程），客户端按在途请求最少路由。W3b 闭环并发 8–128（每档 max(96, 2×并发) 个请求，开事件记录），再按闭环测得的容量做开环泊松 50%, 75%, 90%, 100%, 120%，每档 2 分钟。
- 运行记录：编排器：成功（尝试 1 次），11-15 06:17 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 0.98 h；结果 `20261115_v2_a_p2_run1_run01`：ok，15 分钟

#### `v2_a_s1x3_run1`
- 计划条目：优先级 68；依赖 无；申请 4 卡、27 核、390 GB；预计 25 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_a_s1x3_run1.json`（硬件 a6000x4，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3，参数 `--config configs/models/minicpm_o_4_5_v2_s1x3.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 8, 16, 32, 64, 128；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `w3b_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 w3b_closed；torch profiler 无；nsys 无
- 配置里的说明：plan v2 A（4 卡，第 1 次）：一个服务：1 个 thinker（pipeline）副本 + 3 个 talker 副本，code2wav 单独进程、与 talker 同卡。W3b 闭环并发 8–128（每档 max(96, 2×并发) 个请求，开事件记录），再按闭环测得的容量做开环泊松 50%, 75%, 90%, 100%, 120%，每档 2 分钟。
- 运行记录：编排器：成功（尝试 1 次），11-15 06:32 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 0.99 h；结果 `20261115_v2_a_s1x3_run1_run01`：ok，15 分钟

#### `v2_a_s2x2_run1`
- 计划条目：优先级 67；依赖 无；申请 4 卡、27 核、390 GB；预计 25 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_a_s2x2_run1.json`（硬件 a6000x4，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3，参数 `--config configs/models/minicpm_o_4_5_v2_s2x2.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 8, 16, 32, 64, 128；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `w3b_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 w3b_closed；torch profiler 无；nsys 无
- 配置里的说明：plan v2 A（4 卡，第 1 次）：一个服务：2 个 thinker（pipeline）副本 + 2 个 talker 副本，code2wav 单独进程、与 talker 同卡。W3b 闭环并发 8–128（每档 max(96, 2×并发) 个请求，开事件记录），再按闭环测得的容量做开环泊松 50%, 75%, 90%, 100%, 120%，每档 2 分钟。
- 运行记录：编排器：成功（尝试 1 次），11-15 06:33 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 0.97 h；结果 `20261115_v2_a_s2x2_run1_run01`：ok，14 分钟

#### `v2_a_s2x2_run2`
- 计划条目：优先级 66；依赖 无；申请 4 卡、27 核、390 GB；预计 25 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_a_s2x2_run2.json`（硬件 a6000x4，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3，参数 `--config configs/models/minicpm_o_4_5_v2_s2x2.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 8, 16, 32, 64, 128；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `w3b_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 w3b_closed；torch profiler 无；nsys 无
- 配置里的说明：plan v2 A（4 卡，第 2 次）：一个服务：2 个 thinker（pipeline）副本 + 2 个 talker 副本，code2wav 单独进程、与 talker 同卡。W3b 闭环并发 8–128（每档 max(96, 2×并发) 个请求，开事件记录），再按闭环测得的容量做开环泊松 50%, 75%, 90%, 100%, 120%，每档 2 分钟。
- 运行记录：编排器：成功（尝试 1 次），11-15 06:47 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 0.94 h；结果 `20261115_v2_a_s2x2_run2_run01`：ok，14 分钟

#### `v2_a_s1x3_run2`
- 计划条目：优先级 65；依赖 无；申请 4 卡、27 核、390 GB；预计 25 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_a_s1x3_run2.json`（硬件 a6000x4，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3，参数 `--config configs/models/minicpm_o_4_5_v2_s1x3.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 8, 16, 32, 64, 128；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `w3b_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 w3b_closed；torch profiler 无；nsys 无
- 配置里的说明：plan v2 A（4 卡，第 2 次）：一个服务：1 个 thinker（pipeline）副本 + 3 个 talker 副本，code2wav 单独进程、与 talker 同卡。W3b 闭环并发 8–128（每档 max(96, 2×并发) 个请求，开事件记录），再按闭环测得的容量做开环泊松 50%, 75%, 90%, 100%, 120%，每档 2 分钟。
- 运行记录：编排器：成功（尝试 1 次），11-15 06:48 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 0.93 h；结果 `20261115_v2_a_s1x3_run2_run01`：ok，14 分钟

#### `v2_a_p2_run2`
- 计划条目：优先级 64；依赖 无；申请 4 卡、27 核、390 GB；预计 25 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_a_p2_run2.json`（硬件 a6000x4，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 p0：minicpm_o_4_5，GPU 0, 1，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 p1：minicpm_o_4_5，GPU 2, 3，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 8, 16, 32, 64, 128；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `w3b_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 w3b_closed；torch profiler 无；nsys 无
- 配置里的说明：plan v2 A（4 卡，第 2 次）：2 个拆分服务（每个 1 张 thinker 卡 + 1 张 talker 卡，code2wav 单独进程），客户端按在途请求最少路由。W3b 闭环并发 8–128（每档 max(96, 2×并发) 个请求，开事件记录），再按闭环测得的容量做开环泊松 50%, 75%, 90%, 100%, 120%，每档 2 分钟。
- 运行记录：编排器：成功（尝试 1 次），11-15 07:01 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 0.93 h；结果 `20261115_v2_a_p2_run2_run01`：ok，14 分钟

#### `v2_a_r4_run2`
- 计划条目：优先级 63；依赖 无；申请 4 卡、27 核、390 GB；预计 25 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_a_r4_run2.json`（硬件 a6000x4，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r2：minicpm_o_4_5，GPU 2，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r3：minicpm_o_4_5，GPU 3，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 8, 16, 32, 64, 128；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `w3b_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 w3b_closed；torch profiler 无；nsys 无
- 配置里的说明：plan v2 A（4 卡，第 2 次）：4 个完整副本（每卡一个服务，code2wav 单独进程），客户端按在途请求最少路由。W3b 闭环并发 8–128（每档 max(96, 2×并发) 个请求，开事件记录），再按闭环测得的容量做开环泊松 50%, 75%, 90%, 100%, 120%，每档 2 分钟。
- 运行记录：编排器：成功（尝试 1 次），11-15 07:02 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 1.00 h；结果 `20261115_v2_a_r4_run2_run01`：ok，15 分钟

#### `v2_b1_r8`
- 计划条目：优先级 60；依赖 无；申请 8 卡、54 核、780 GB；预计 30 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_b1_r8.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r2：minicpm_o_4_5，GPU 2，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r3：minicpm_o_4_5，GPU 3，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r4：minicpm_o_4_5，GPU 4，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r5：minicpm_o_4_5，GPU 5，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r6：minicpm_o_4_5，GPU 6，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r7：minicpm_o_4_5，GPU 7，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `mix_closed` 闭环：混合负载（W1（文本→文本） 40%、W3b（文本提问→文本+语音） 40%、W2（语音提问→文本） 20%，每个请求按权重抽取）；并发 16, 32, 64, 128, 256；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `mix_open` 开环（泊松到达）：混合负载（W1（文本→文本） 40%、W3b（文本提问→文本+语音） 40%、W2（语音提问→文本） 20%，每个请求按权重抽取）；速率 按本实验 `mix_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 mix_closed；torch profiler 无；nsys 无
- 配置里的说明：plan v2 B1：8 个完整副本（每卡一个服务，code2wav 单独进程），客户端按在途请求最少路由上的混合负载（40% W1 文本对话、40% W3b 文本提问语音回答、20% W2 语音提问文本回答，每个请求按权重抽取），闭环并发 16–256，再按测得容量开环 50%–120%。
- 运行记录：编排器：成功（尝试 1 次），11-15 07:17 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 2.05 h；结果 `20261115_v2_b1_r8_run01`：ok，15 分钟

#### `v2_b1_s2x6`
- 计划条目：优先级 59；依赖 无；申请 8 卡、54 核、780 GB；预计 30 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_b1_s2x6.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3, 4, 5, 6, 7，参数 `--config configs/models/minicpm_o_4_5_v2_s2x6.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `mix_closed` 闭环：混合负载（W1（文本→文本） 40%、W3b（文本提问→文本+语音） 40%、W2（语音提问→文本） 20%，每个请求按权重抽取）；并发 16, 32, 64, 128, 256；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `mix_open` 开环（泊松到达）：混合负载（W1（文本→文本） 40%、W3b（文本提问→文本+语音） 40%、W2（语音提问→文本） 20%，每个请求按权重抽取）；速率 按本实验 `mix_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 mix_closed；torch profiler 无；nsys 无
- 配置里的说明：plan v2 B1：一个服务：2 个 thinker（pipeline）副本 + 6 个 talker 副本，code2wav 单独进程、与 talker 同卡上的混合负载（40% W1 文本对话、40% W3b 文本提问语音回答、20% W2 语音提问文本回答，每个请求按权重抽取），闭环并发 16–256，再按测得容量开环 50%–120%。
- 运行记录：编排器：成功（尝试 1 次），11-15 07:32 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 1.79 h；结果 `20261115_v2_b1_s2x6_run01`：ok，13 分钟

#### `v2_b1_p4`
- 计划条目：优先级 58；依赖 无；申请 8 卡、54 核、780 GB；预计 30 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_b1_p4.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 p0：minicpm_o_4_5，GPU 0, 1，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 p1：minicpm_o_4_5，GPU 2, 3，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 p2：minicpm_o_4_5，GPU 4, 5，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 p3：minicpm_o_4_5，GPU 6, 7，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `mix_closed` 闭环：混合负载（W1（文本→文本） 40%、W3b（文本提问→文本+语音） 40%、W2（语音提问→文本） 20%，每个请求按权重抽取）；并发 16, 32, 64, 128, 256；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `mix_open` 开环（泊松到达）：混合负载（W1（文本→文本） 40%、W3b（文本提问→文本+语音） 40%、W2（语音提问→文本） 20%，每个请求按权重抽取）；速率 按本实验 `mix_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 mix_closed；torch profiler 无；nsys 无
- 配置里的说明：plan v2 B1：4 个拆分服务（每个 1 张 thinker 卡 + 1 张 talker 卡，code2wav 单独进程），客户端按在途请求最少路由上的混合负载（40% W1 文本对话、40% W3b 文本提问语音回答、20% W2 语音提问文本回答，每个请求按权重抽取），闭环并发 16–256，再按测得容量开环 50%–120%。
- 运行记录：编排器：成功（尝试 1 次），11-15 07:46 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 1.84 h；结果 `20261115_v2_b1_p4_run01`：ok，14 分钟

#### `v2_a_r2`
- 计划条目：优先级 50；依赖 无；申请 2 卡、14 核、190 GB；预计 25 分钟；超时 90 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_a_r2.json`（硬件 a6000x2，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 4, 8, 16, 32, 64；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `w3b_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 w3b_closed；torch profiler 无；nsys 无
- 配置里的说明：plan v2 A（2 卡，第 1 次）：2 个完整副本（每卡一个服务，code2wav 单独进程），客户端按在途请求最少路由。W3b 闭环并发 4–64（每档 max(96, 2×并发) 个请求，开事件记录），再按闭环测得的容量做开环泊松 50%, 75%, 90%, 100%, 120%，每档 2 分钟。
- 运行记录：编排器：成功（尝试 1 次），11-15 07:59 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 0.56 h；结果 `20261115_v2_a_r2_run01`：ok，17 分钟

#### `v2_a_p1`
- 计划条目：优先级 49；依赖 无；申请 2 卡、14 核、190 GB；预计 25 分钟；超时 90 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2_a_p1.json`（硬件 a6000x2，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 p0：minicpm_o_4_5，GPU 0, 1，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 4, 8, 16, 32, 64；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `w3b_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 w3b_closed；torch profiler 无；nsys 无
- 配置里的说明：plan v2 A（2 卡，第 1 次）：1 个拆分服务（每个 1 张 thinker 卡 + 1 张 talker 卡，code2wav 单独进程），客户端按在途请求最少路由。W3b 闭环并发 4–64（每档 max(96, 2×并发) 个请求，开事件记录），再按闭环测得的容量做开环泊松 50%, 75%, 90%, 100%, 120%，每档 2 分钟。
- 运行记录：编排器：成功（尝试 1 次），11-15 07:59 开始，第一批节点（共享 gpu 分区的 8×A6000），GPU 时长 0.52 h；结果 `20261115_v2_a_p1_run01`：ok，15 分钟

## plan v2 第二批（8×A6000，Job 454290）

### 来源：`configs/experiments/plan_a6000_8gpu_v2_batch2.json`

#### `v2b_y0_env_topology`
- 计划条目：优先级 100；依赖 无；申请 8 卡、54 核、780 GB；预计 10 分钟；超时 45 分钟；失败后最多重试 1 次
- 命令：`bash scripts/slurm/env_topology.sh unicorn minicpm_o_4_5`
- 计划里的说明：第二批 Y0（compute-03）：拓扑、点对点拷贝带宽、NCCL all-reduce（第一批的 Y0 因为 nvidia-smi 表头的 ANSI 码什么都没测），计时预热 MiniCPM-o 权重
- 运行记录：编排器：成功（尝试 1 次），11-16 19:02 开始，compute-03（8×A6000），GPU 时长 0.80 h；结果 `20261116_v2b_y0_env_topology_run01`：ok

#### `v2b_a_s2x6_c03`
- 计划条目：优先级 90；依赖 无；申请 8 卡、54 核、780 GB；预计 18 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_a_s2x6_c03.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3, 4, 5, 6, 7，参数 `--config configs/models/minicpm_o_4_5_v2_s2x6.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 16, 32, 64, 128, 256；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `w3b_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 w3b_closed；torch profiler 无；nsys 无
- 配置里的说明：第二批 A 的跨节点对照（compute-03，与第一批在另一台 8 卡节点上的两次比较）：一个服务：2 个 thinker（pipeline）副本 + 6 个 talker 副本，code2wav 单独进程、与 talker 同卡。W3b 闭环并发 16–256（每档 max(96, 2×并发) 个请求，开事件记录），再按闭环测得的容量做开环泊松 50%, 75%, 90%, 100%, 120%，每档 2 分钟。
- 运行记录：编排器：成功（尝试 1 次），11-16 19:08 开始，compute-03（8×A6000），GPU 时长 2.29 h；结果 `20261116_v2b_a_s2x6_c03_run01`：ok，14 分钟

#### `v2b_a_r8_c03`
- 计划条目：优先级 89；依赖 无；申请 8 卡、54 核、780 GB；预计 18 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_a_r8_c03.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r2：minicpm_o_4_5，GPU 2，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r3：minicpm_o_4_5，GPU 3，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r4：minicpm_o_4_5，GPU 4，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r5：minicpm_o_4_5，GPU 5，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r6：minicpm_o_4_5，GPU 6，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r7：minicpm_o_4_5，GPU 7，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 16, 32, 64, 128, 256；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `w3b_open` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 按本实验 `w3b_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 w3b_closed；torch profiler 无；nsys 无
- 配置里的说明：第二批 A 的跨节点对照（compute-03，与第一批在另一台 8 卡节点上的两次比较）：8 个完整副本（每卡一个服务，code2wav 单独进程），客户端按在途请求最少路由。W3b 闭环并发 16–256（每档 max(96, 2×并发) 个请求，开事件记录），再按闭环测得的容量做开环泊松 50%, 75%, 90%, 100%, 120%，每档 2 分钟。
- 运行记录：编排器：成功（尝试 1 次），11-16 19:25 开始，compute-03（8×A6000），GPU 时长 1.93 h；结果 `20261116_v2b_a_r8_c03_run01`：ok，14 分钟

#### `v2b_b1_s4x4`
- 计划条目：优先级 60；依赖 无；申请 8 卡、54 核、780 GB；预计 17 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_b1_s4x4.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3, 4, 5, 6, 7，参数 `--config configs/models/minicpm_o_4_5_v2_s4x4.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `mix_closed` 闭环：混合负载（W1（文本→文本） 40%、W3b（文本提问→文本+语音） 40%、W2（语音提问→文本） 20%，每个请求按权重抽取）；并发 16, 32, 64, 128, 256；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `mix_open` 开环（泊松到达）：混合负载（W1（文本→文本） 40%、W3b（文本提问→文本+语音） 40%、W2（语音提问→文本） 20%，每个请求按权重抽取）；速率 按本实验 `mix_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 mix_closed；torch profiler 无；nsys 无
- 配置里的说明：第二批 B1：一个服务：4 个 thinker（pipeline）副本 + 4 个 talker 副本，code2wav 单独进程、与 talker 同卡上的混合负载（40% W1 文本对话、40% W3b 文本提问语音回答、20% W2 语音提问文本回答，每个请求按权重抽取），闭环并发 16–256，再按测得容量开环 50%–120%。
- 运行记录：编排器：成功（尝试 1 次），11-16 21:32 开始，compute-03（8×A6000），GPU 时长 1.77 h；结果 `20261116_v2b_b1_s4x4_run01`：ok，13 分钟

#### `v2b_b1_s3x5`
- 计划条目：优先级 59；依赖 无；申请 8 卡、54 核、780 GB；预计 17 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_b1_s3x5.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3, 4, 5, 6, 7，参数 `--config configs/models/minicpm_o_4_5_v2_s3x5.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `mix_closed` 闭环：混合负载（W1（文本→文本） 40%、W3b（文本提问→文本+语音） 40%、W2（语音提问→文本） 20%，每个请求按权重抽取）；并发 16, 32, 64, 128, 256；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `mix_open` 开环（泊松到达）：混合负载（W1（文本→文本） 40%、W3b（文本提问→文本+语音） 40%、W2（语音提问→文本） 20%，每个请求按权重抽取）；速率 按本实验 `mix_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 mix_closed；torch profiler 无；nsys 无
- 配置里的说明：第二批 B1：一个服务：3 个 thinker（pipeline）副本 + 5 个 talker 副本，code2wav 单独进程、与 talker 同卡上的混合负载（40% W1 文本对话、40% W3b 文本提问语音回答、20% W2 语音提问文本回答，每个请求按权重抽取），闭环并发 16–256，再按测得容量开环 50%–120%。
- 运行记录：编排器：成功（尝试 1 次），11-16 21:45 开始，compute-03（8×A6000），GPU 时长 1.77 h；结果 `20261116_v2b_b1_s3x5_run01`：ok，13 分钟

### 来源：`configs/experiments/plan_a6000_8gpu_v2_batch2_part2.json`

#### `v2b_b1_s5x3`
- 计划条目：优先级 58；依赖 无；申请 8 卡、54 核、780 GB；预计 17 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_b1_s5x3.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3, 4, 5, 6, 7，参数 `--config configs/models/minicpm_o_4_5_v2_s5x3.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `mix_closed` 闭环：混合负载（W1（文本→文本） 40%、W3b（文本提问→文本+语音） 40%、W2（语音提问→文本） 20%，每个请求按权重抽取）；并发 16, 32, 64, 128, 256；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `mix_open` 开环（泊松到达）：混合负载（W1（文本→文本） 40%、W3b（文本提问→文本+语音） 40%、W2（语音提问→文本） 20%，每个请求按权重抽取）；速率 按本实验 `mix_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 mix_closed；torch profiler 无；nsys 无
- 配置里的说明：第二批 B1：一个服务：5 个 thinker（pipeline）副本 + 3 个 talker 副本，code2wav 单独进程、与 talker 同卡上的混合负载（40% W1 文本对话、40% W3b 文本提问语音回答、20% W2 语音提问文本回答，每个请求按权重抽取），闭环并发 16–256，再按测得容量开环 50%–120%。
- 运行记录：编排器：成功（尝试 1 次），11-16 21:58 开始，compute-03（8×A6000），GPU 时长 1.82 h；结果 `20261116_v2b_b1_s5x3_run01`：ok，13 分钟

#### `v2b_b1_s6x2`
- 计划条目：优先级 57；依赖 无；申请 8 卡、54 核、780 GB；预计 17 分钟；超时 100 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_b1_s6x2.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3, 4, 5, 6, 7，参数 `--config configs/models/minicpm_o_4_5_v2_s6x2.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `mix_closed` 闭环：混合负载（W1（文本→文本） 40%、W3b（文本提问→文本+语音） 40%、W2（语音提问→文本） 20%，每个请求按权重抽取）；并发 16, 32, 64, 128, 256；每档 max(96, 2×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s；**服务端事件**
  3. `mix_open` 开环（泊松到达）：混合负载（W1（文本→文本） 40%、W3b（文本提问→文本+语音） 40%、W2（语音提问→文本） 20%，每个请求按权重抽取）；速率 按本实验 `mix_closed` 阶段的闭环最高吞吐 × 0.5, 0.75, 0.9, 1, 1.2；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 mix_closed；torch profiler 无；nsys 无
- 配置里的说明：第二批 B1：一个服务：6 个 thinker（pipeline）副本 + 2 个 talker 副本，code2wav 单独进程、与 talker 同卡上的混合负载（40% W1 文本对话、40% W3b 文本提问语音回答、20% W2 语音提问文本回答，每个请求按权重抽取），闭环并发 16–256，再按测得容量开环 50%–120%。
- 运行记录：编排器：成功（尝试 1 次），11-16 22:12 开始，compute-03（8×A6000），GPU 时长 1.80 h；结果 `20261116_v2b_b1_s6x2_run01`：ok，13 分钟

#### `v2b_b2l_r8`
- 计划条目：优先级 50；依赖 无；申请 8 卡、54 核、780 GB；预计 45 分钟；超时 150 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_b2l_r8.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r2：minicpm_o_4_5，GPU 2，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r3：minicpm_o_4_5，GPU 3，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r4：minicpm_o_4_5，GPU 4，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r5：minicpm_o_4_5，GPU 5，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r6：minicpm_o_4_5，GPU 6，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r7：minicpm_o_4_5，GPU 7，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `lo_text_c64` Agent（合成任务，真实模型请求，录 trace）：512 个会话，并发 64，种子 8100，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）；路由 least_outstanding
  3. `lo_text_c128` Agent（合成任务，真实模型请求，录 trace）：1024 个会话，并发 128，种子 8101，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）；路由 least_outstanding
  4. `lo_text_c256` Agent（合成任务，真实模型请求，录 trace）：2048 个会话，并发 256，种子 8102，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）；**服务端事件**，路由 least_outstanding
  5. `lo_text_c512` Agent（合成任务，真实模型请求，录 trace）：2048 个会话，并发 512，种子 8103，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）；路由 least_outstanding
  6. `lo_voice_c64` Agent（合成任务，真实模型请求，录 trace）：512 个会话，并发 64，种子 8150，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）；路由 least_outstanding
  7. `lo_voice_c128` Agent（合成任务，真实模型请求，录 trace）：1024 个会话，并发 128，种子 8151，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）；路由 least_outstanding
  8. `lo_voice_c256` Agent（合成任务，真实模型请求，录 trace）：2048 个会话，并发 256，种子 8152，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）；路由 least_outstanding
  9. `lo_voice_c512` Agent（合成任务，真实模型请求，录 trace）：2048 个会话，并发 512，种子 8153，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）；路由 least_outstanding
  10. `sa_text_c64` Agent（合成任务，真实模型请求，录 trace）：512 个会话，并发 64，种子 8200，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）；路由 session_affinity
  11. `sa_text_c128` Agent（合成任务，真实模型请求，录 trace）：1024 个会话，并发 128，种子 8201，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）；路由 session_affinity
  12. `sa_text_c256` Agent（合成任务，真实模型请求，录 trace）：2048 个会话，并发 256，种子 8202，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）；**服务端事件**，路由 session_affinity
  13. `sa_text_c512` Agent（合成任务，真实模型请求，录 trace）：2048 个会话，并发 512，种子 8203，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）；路由 session_affinity
  14. `sa_voice_c64` Agent（合成任务，真实模型请求，录 trace）：512 个会话，并发 64，种子 8250，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）；路由 session_affinity
  15. `sa_voice_c128` Agent（合成任务，真实模型请求，录 trace）：1024 个会话，并发 128，种子 8251，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）；路由 session_affinity
  16. `sa_voice_c256` Agent（合成任务，真实模型请求，录 trace）：2048 个会话，并发 256，种子 8252，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）；路由 session_affinity
  17. `sa_voice_c512` Agent（合成任务，真实模型请求，录 trace）：2048 个会话，并发 512，种子 8253，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）；路由 session_affinity
- 额外采集：服务端事件 lo_text_c256、sa_text_c256；torch profiler 无；nsys 无
- 配置里的说明：第二批 B2（加长）：8 个完整副本（每卡一个服务，code2wav 单独进程），客户端按在途请求最少路由上的大规模 Agent（会话不重复），同时进行的会话 64–512，每档会话总数为并发的 8 倍（512 并发时 4 倍），每档约 2–3 分钟（第一批每档只有 6–40 秒）；文本 Agent 与语音 Agent（文本提问、最后一步语音回答），路由分别用在途请求最少和按会话固定。
- 运行记录：编排器：运行中（尝试 1 次），11-16 22:25 开始，compute-03（8×A6000）

#### `v2b_b2l_s2x6`
- 计划条目：优先级 49；依赖 无；申请 8 卡、54 核、780 GB；预计 30 分钟；超时 120 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_b2l_s2x6.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3, 4, 5, 6, 7，参数 `--config configs/models/minicpm_o_4_5_v2_s2x6.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `text_c64` Agent（合成任务，真实模型请求，录 trace）：512 个会话，并发 64，种子 8300，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）
  3. `text_c128` Agent（合成任务，真实模型请求，录 trace）：1024 个会话，并发 128，种子 8301，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）
  4. `text_c256` Agent（合成任务，真实模型请求，录 trace）：2048 个会话，并发 256，种子 8302，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）；**服务端事件**
  5. `text_c512` Agent（合成任务，真实模型请求，录 trace）：2048 个会话，并发 512，种子 8303，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）
  6. `voice_c64` Agent（合成任务，真实模型请求，录 trace）：512 个会话，并发 64，种子 8350，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）
  7. `voice_c128` Agent（合成任务，真实模型请求，录 trace）：1024 个会话，并发 128，种子 8351，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）
  8. `voice_c256` Agent（合成任务，真实模型请求，录 trace）：2048 个会话，并发 256，种子 8352，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）
  9. `voice_c512` Agent（合成任务，真实模型请求，录 trace）：2048 个会话，并发 512，种子 8353，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）
  10. `toollat0p5_text_c256` Agent（合成任务，真实模型请求，录 trace）：2048 个会话，并发 256，种子 8390，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），模拟工具延迟 对数正态（中位数 0.5 s，σ 0.5）
  11. `toollat2_text_c256` Agent（合成任务，真实模型请求，录 trace）：2048 个会话，并发 256，种子 8391，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），模拟工具延迟 对数正态（中位数 2.0 s，σ 0.5）
- 额外采集：服务端事件 text_c256；torch profiler 无；nsys 无
- 配置里的说明：第二批 B2（加长）：一个服务：2 个 thinker（pipeline）副本 + 6 个 talker 副本，code2wav 单独进程、与 talker 同卡上的大规模 Agent，每档会话总数为并发的 8 倍（512 并发时 4 倍），每档约 2–3 分钟（第一批每档只有 6–40 秒）；另加 256 个会话时工具延迟 0.5 s、2 s 两档（对数正态中位数，各 2048 个会话）。
- 运行记录：编排器：排队（可运行）（尝试 0 次）

#### `v2b_b2l_p4`
- 计划条目：优先级 48；依赖 无；申请 8 卡、54 核、780 GB；预计 25 分钟；超时 120 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_b2l_p4.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 p0：minicpm_o_4_5，GPU 0, 1，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 p1：minicpm_o_4_5，GPU 2, 3，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 p2：minicpm_o_4_5，GPU 4, 5，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 p3：minicpm_o_4_5，GPU 6, 7，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `text_c64` Agent（合成任务，真实模型请求，录 trace）：512 个会话，并发 64，种子 8400，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）
  3. `text_c128` Agent（合成任务，真实模型请求，录 trace）：1024 个会话，并发 128，种子 8401，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）
  4. `text_c256` Agent（合成任务，真实模型请求，录 trace）：2048 个会话，并发 256，种子 8402，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）；**服务端事件**
  5. `text_c512` Agent（合成任务，真实模型请求，录 trace）：2048 个会话，并发 512，种子 8403，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）
  6. `voice_c64` Agent（合成任务，真实模型请求，录 trace）：512 个会话，并发 64，种子 8450，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）
  7. `voice_c128` Agent（合成任务，真实模型请求，录 trace）：1024 个会话，并发 128，种子 8451，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）
  8. `voice_c256` Agent（合成任务，真实模型请求，录 trace）：2048 个会话，并发 256，种子 8452，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）
  9. `voice_c512` Agent（合成任务，真实模型请求，录 trace）：2048 个会话，并发 512，种子 8453，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）
- 额外采集：服务端事件 text_c256；torch profiler 无；nsys 无
- 配置里的说明：第二批 B2（加长）：4 个拆分服务（每个 1 张 thinker 卡 + 1 张 talker 卡，code2wav 单独进程），客户端按在途请求最少路由上的大规模 Agent，每档会话总数为并发的 8 倍（512 并发时 4 倍），每档约 2–3 分钟（第一批每档只有 6–40 秒）。
- 运行记录：编排器：排队（可运行）（尝试 0 次）

#### `v2b_b2l_s4x4`
- 计划条目：优先级 47；依赖 无；申请 8 卡、54 核、780 GB；预计 25 分钟；超时 120 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_b2l_s4x4.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3, 4, 5, 6, 7，参数 `--config configs/models/minicpm_o_4_5_v2_s4x4.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `text_c64` Agent（合成任务，真实模型请求，录 trace）：512 个会话，并发 64，种子 8500，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）
  3. `text_c128` Agent（合成任务，真实模型请求，录 trace）：1024 个会话，并发 128，种子 8501，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）
  4. `text_c256` Agent（合成任务，真实模型请求，录 trace）：2048 个会话，并发 256，种子 8502，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）；**服务端事件**
  5. `text_c512` Agent（合成任务，真实模型请求，录 trace）：2048 个会话，并发 512，种子 8503，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明）
  6. `voice_c64` Agent（合成任务，真实模型请求，录 trace）：512 个会话，并发 64，种子 8550，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）
  7. `voice_c128` Agent（合成任务，真实模型请求，录 trace）：1024 个会话，并发 128，种子 8551，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）
  8. `voice_c256` Agent（合成任务，真实模型请求，录 trace）：2048 个会话，并发 256，种子 8552，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）
  9. `voice_c512` Agent（合成任务，真实模型请求，录 trace）：2048 个会话，并发 512，种子 8553，每会话最多 5 步，每个会话的任务不同（只共享系统提示和工具说明），最后一步语音回答，用户思考时间 对数正态（中位数 1.0 s，σ 0.5）
- 额外采集：服务端事件 text_c256；torch profiler 无；nsys 无
- 配置里的说明：第二批 B2（加长）：一个服务：4 个 thinker（pipeline）副本 + 4 个 talker 副本，code2wav 单独进程、与 talker 同卡上的大规模 Agent，每档会话总数为并发的 8 倍（512 并发时 4 倍），每档约 2–3 分钟（第一批每档只有 6–40 秒）。
- 运行记录：编排器：排队（可运行）（尝试 0 次）

### 来源：`configs/experiments/plan_a6000_8gpu_v2_batch2_part3.json`

#### `v2b_aoh_s2x6`
- 计划条目：优先级 80；依赖 无；申请 8 卡、54 核、780 GB；预计 15 分钟；超时 90 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_aoh_s2x6.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3, 4, 5, 6, 7，参数 `--config configs/models/minicpm_o_4_5_v2_s2x6.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_open_abs` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 16, 20, 24, 28, 32, 36, 40 请求/秒；每档 90 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：第二批 A 的高负载开环：一个服务：2 个 thinker（pipeline）副本 + 6 个 talker 副本，code2wav 单独进程、与 talker 同卡。W3b 开环泊松，所有布局用同一组绝对速率 16, 20, 24, 28, 32, 36, 40 请求/秒，每档 90 秒；第一批的开环最高只到闭环容量的 120%，池化布局在那里还远没离开 SLO。
- 运行记录：编排器：成功（尝试 1 次），11-16 19:40 开始，compute-03（8×A6000），GPU 时长 1.87 h；结果 `20261116_v2b_aoh_s2x6_run01`：ok，14 分钟

#### `v2b_aoh_s3x5`
- 计划条目：优先级 79；依赖 无；申请 8 卡、54 核、780 GB；预计 15 分钟；超时 90 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_aoh_s3x5.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3, 4, 5, 6, 7，参数 `--config configs/models/minicpm_o_4_5_v2_s3x5.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_open_abs` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 16, 20, 24, 28, 32, 36, 40 请求/秒；每档 90 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：第二批 A 的高负载开环：一个服务：3 个 thinker（pipeline）副本 + 5 个 talker 副本，code2wav 单独进程、与 talker 同卡。W3b 开环泊松，所有布局用同一组绝对速率 16, 20, 24, 28, 32, 36, 40 请求/秒，每档 90 秒；第一批的开环最高只到闭环容量的 120%，池化布局在那里还远没离开 SLO。
- 运行记录：编排器：成功（尝试 1 次），11-16 20:19 开始，compute-03（8×A6000），GPU 时长 1.91 h；结果 `20261116_v2b_aoh_s3x5_run01`：ok，14 分钟

#### `v2b_aoh_s1x7`
- 计划条目：优先级 78；依赖 无；申请 8 卡、54 核、780 GB；预计 15 分钟；超时 90 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_aoh_s1x7.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3, 4, 5, 6, 7，参数 `--config configs/models/minicpm_o_4_5_v2_s1x7.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_open_abs` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 16, 20, 24, 28, 32, 36, 40 请求/秒；每档 90 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：第二批 A 的高负载开环：一个服务：1 个 thinker（pipeline）副本 + 7 个 talker 副本，code2wav 单独进程、与 talker 同卡。W3b 开环泊松，所有布局用同一组绝对速率 16, 20, 24, 28, 32, 36, 40 请求/秒，每档 90 秒；第一批的开环最高只到闭环容量的 120%，池化布局在那里还远没离开 SLO。
- 运行记录：编排器：成功（尝试 1 次），11-16 20:33 开始，compute-03（8×A6000），GPU 时长 1.91 h；结果 `20261116_v2b_aoh_s1x7_run01`：ok，14 分钟

#### `v2b_aoh_r8`
- 计划条目：优先级 77；依赖 无；申请 8 卡、54 核、780 GB；预计 15 分钟；超时 90 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_aoh_r8.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r2：minicpm_o_4_5，GPU 2，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r3：minicpm_o_4_5，GPU 3，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r4：minicpm_o_4_5，GPU 4，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r5：minicpm_o_4_5，GPU 5，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r6：minicpm_o_4_5，GPU 6，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r7：minicpm_o_4_5，GPU 7，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_open_abs` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 16, 20, 24, 28, 32, 36, 40 请求/秒；每档 90 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：第二批 A 的高负载开环：8 个完整副本（每卡一个服务，code2wav 单独进程），客户端按在途请求最少路由。W3b 开环泊松，所有布局用同一组绝对速率 16, 20, 24, 28, 32, 36, 40 请求/秒，每档 90 秒；第一批的开环最高只到闭环容量的 120%，池化布局在那里还远没离开 SLO。
- 运行记录：编排器：成功（尝试 1 次），11-16 20:47 开始，compute-03（8×A6000），GPU 时长 1.92 h；结果 `20261116_v2b_aoh_r8_run01`：ok，14 分钟

#### `v2b_aoh_s4x4`
- 计划条目：优先级 76；依赖 无；申请 8 卡、54 核、780 GB；预计 15 分钟；超时 90 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_aoh_s4x4.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3, 4, 5, 6, 7，参数 `--config configs/models/minicpm_o_4_5_v2_s4x4.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_open_abs` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 16, 20, 24, 28, 32, 36, 40 请求/秒；每档 90 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：第二批 A 的高负载开环：一个服务：4 个 thinker（pipeline）副本 + 4 个 talker 副本，code2wav 单独进程、与 talker 同卡。W3b 开环泊松，所有布局用同一组绝对速率 16, 20, 24, 28, 32, 36, 40 请求/秒，每档 90 秒；第一批的开环最高只到闭环容量的 120%，池化布局在那里还远没离开 SLO。
- 运行记录：编排器：成功（尝试 1 次），11-16 21:02 开始，compute-03（8×A6000），GPU 时长 2.01 h；结果 `20261116_v2b_aoh_s4x4_run01`：ok，15 分钟

#### `v2b_aoh_p4`
- 计划条目：优先级 75；依赖 无；申请 8 卡、54 核、780 GB；预计 15 分钟；超时 90 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_aoh_p4.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 p0：minicpm_o_4_5，GPU 0, 1，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 p1：minicpm_o_4_5，GPU 2, 3，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 p2：minicpm_o_4_5，GPU 4, 5，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 p3：minicpm_o_4_5，GPU 6, 7，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_open_abs` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 16, 20, 24, 28, 32, 36, 40 请求/秒；每档 90 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：第二批 A 的高负载开环：4 个拆分服务（每个 1 张 thinker 卡 + 1 张 talker 卡，code2wav 单独进程），客户端按在途请求最少路由。W3b 开环泊松，所有布局用同一组绝对速率 16, 20, 24, 28, 32, 36, 40 请求/秒，每档 90 秒；第一批的开环最高只到闭环容量的 120%，池化布局在那里还远没离开 SLO。
- 运行记录：编排器：成功（尝试 1 次），11-16 21:17 开始，compute-03（8×A6000），GPU 时长 1.99 h；结果 `20261116_v2b_aoh_p4_run01`：ok，15 分钟

### 来源：`configs/experiments/plan_a6000_8gpu_v2_batch2_part4.json`

#### `v2b_g1a_links`
- 计划条目：优先级 45；依赖 无；申请 8 卡、54 核、780 GB；预计 17 分钟；超时 90 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_g1a_links.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 round_robin）
- 服务 g_node：minicpm_o_4_5，GPU 0, 2，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 g_pix：minicpm_o_4_5，GPU 1, 3，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 g_nv4a：minicpm_o_4_5，GPU 4, 5，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 g_nv4b：minicpm_o_4_5，GPU 6, 7，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 16, 32；每档 max(96, 3×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2；**服务端事件**
  3. `w3b_open_abs` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 6, 10, 13, 15 请求/秒；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止；**服务端事件**
- 额外采集：服务端事件 w3b_closed、w3b_open_abs；torch profiler 无；nsys 无
- 配置里的说明：第二批 G1：4 个 2 卡拆分服务（thinker 卡 → talker 卡，code2wav 单独进程）同时运行，卡对的连接分别是 node（GPU 0→2）、pix（GPU 1→3）、nv4（GPU 4→5）、nv4（GPU 6→7）；客户端轮流分配请求，每个服务承受相同的负载。W3b 闭环并发 16、32，开环 6、10、13、15 请求/秒（合计），开事件记录，按服务比较延迟和跨卡传递时间。
- 运行记录：编排器：排队（可运行）（尝试 0 次）

#### `v2b_g1b_links`
- 计划条目：优先级 44；依赖 无；申请 8 卡、54 核、780 GB；预计 17 分钟；超时 90 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_g1b_links.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 round_robin）
- 服务 g_sysa：minicpm_o_4_5，GPU 0, 4，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 g_sysb：minicpm_o_4_5，GPU 1, 5，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 g_pix：minicpm_o_4_5，GPU 2, 3，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 g_nv4：minicpm_o_4_5，GPU 6, 7，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_closed` 闭环：W3b（文本提问→文本+语音）；并发 16, 32；每档 max(96, 3×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2；**服务端事件**
  3. `w3b_open_abs` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 6, 10, 13, 15 请求/秒；每档 120 s（请求数 = max(96, 速率×时长)）；某档错误率 > 0.2 时停止；**服务端事件**
- 额外采集：服务端事件 w3b_closed、w3b_open_abs；torch profiler 无；nsys 无
- 配置里的说明：第二批 G1：4 个 2 卡拆分服务（thinker 卡 → talker 卡，code2wav 单独进程）同时运行，卡对的连接分别是 sys（GPU 0→4）、sys（GPU 1→5）、pix（GPU 2→3）、nv4（GPU 6→7）；客户端轮流分配请求，每个服务承受相同的负载。W3b 闭环并发 16、32，开环 6、10、13、15 请求/秒（合计），开事件记录，按服务比较延迟和跨卡传递时间。
- 运行记录：编排器：排队（可运行）（尝试 0 次）

#### `v2b_f1_r1`
- 计划条目：优先级 40；依赖 无；申请 1 卡、7 核、95 GB；预计 25 分钟；超时 120 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_f1_r1.json`（硬件 a6000x1，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--text-only`，启动时限 2400 s
- 阶段：
  1. `smoke_w4` 冒烟检查：W4（视频+文本→文本），2 个请求，检查输出非空
  2. `w4_video_10s` 闭环：W4（视频+文本→文本），视频 10 s（素材标签 count_circles）；并发 1, 2, 4, 8；每档 max(24, 3×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
  3. `w4_video_20s` 闭环：W4（视频+文本→文本），视频 20 s（素材标签 count_circles）；并发 1, 2, 4, 8；每档 max(24, 3×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：第二批 F1：1 个完整副本（--text-only，每卡一个服务）上的视频理解（W4，10 秒和 20 秒合成视频，文本输出），闭环并发 1, 2, 4, 8（每个副本 1、2、4、8），每档 max(24, 3×并发) 个请求；每卡 7 个 CPU。看视频吞吐是否随副本数线性增长，以及 GPU 空闲比例。
- 运行记录：编排器：排队（可运行）（尝试 0 次）

#### `v2b_f1_r2`
- 计划条目：优先级 39；依赖 `v2b_f1_r1`；申请 2 卡、14 核、190 GB；预计 25 分钟；超时 120 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_f1_r2.json`（硬件 a6000x2，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--text-only`，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，参数 `--text-only`，启动时限 2400 s
- 阶段：
  1. `smoke_w4` 冒烟检查：W4（视频+文本→文本），2 个请求，检查输出非空
  2. `w4_video_10s` 闭环：W4（视频+文本→文本），视频 10 s（素材标签 count_circles）；并发 2, 4, 8, 16；每档 max(24, 3×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
  3. `w4_video_20s` 闭环：W4（视频+文本→文本），视频 20 s（素材标签 count_circles）；并发 2, 4, 8, 16；每档 max(24, 3×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：第二批 F1：2 个完整副本（--text-only，每卡一个服务）上的视频理解（W4，10 秒和 20 秒合成视频，文本输出），闭环并发 2, 4, 8, 16（每个副本 1、2、4、8），每档 max(24, 3×并发) 个请求；每卡 7 个 CPU。看视频吞吐是否随副本数线性增长，以及 GPU 空闲比例。
- 运行记录：编排器：排队（等依赖）（尝试 0 次）

#### `v2b_f1_r4`
- 计划条目：优先级 38；依赖 `v2b_f1_r2`；申请 4 卡、27 核、390 GB；预计 25 分钟；超时 120 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_f1_r4.json`（硬件 a6000x4，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--text-only`，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，参数 `--text-only`，启动时限 2400 s
- 服务 r2：minicpm_o_4_5，GPU 2，参数 `--text-only`，启动时限 2400 s
- 服务 r3：minicpm_o_4_5，GPU 3，参数 `--text-only`，启动时限 2400 s
- 阶段：
  1. `smoke_w4` 冒烟检查：W4（视频+文本→文本），2 个请求，检查输出非空
  2. `w4_video_10s` 闭环：W4（视频+文本→文本），视频 10 s（素材标签 count_circles）；并发 4, 8, 16, 32；每档 max(24, 3×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
  3. `w4_video_20s` 闭环：W4（视频+文本→文本），视频 20 s（素材标签 count_circles）；并发 4, 8, 16, 32；每档 max(24, 3×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：第二批 F1：4 个完整副本（--text-only，每卡一个服务）上的视频理解（W4，10 秒和 20 秒合成视频，文本输出），闭环并发 4, 8, 16, 32（每个副本 1、2、4、8），每档 max(24, 3×并发) 个请求；每卡 7 个 CPU。看视频吞吐是否随副本数线性增长，以及 GPU 空闲比例。
- 运行记录：编排器：排队（等依赖）（尝试 0 次）

#### `v2b_f1_r8`
- 计划条目：优先级 37；依赖 `v2b_f1_r4`；申请 8 卡、54 核、780 GB；预计 25 分钟；超时 120 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_f1_r8.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--text-only`，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，参数 `--text-only`，启动时限 2400 s
- 服务 r2：minicpm_o_4_5，GPU 2，参数 `--text-only`，启动时限 2400 s
- 服务 r3：minicpm_o_4_5，GPU 3，参数 `--text-only`，启动时限 2400 s
- 服务 r4：minicpm_o_4_5，GPU 4，参数 `--text-only`，启动时限 2400 s
- 服务 r5：minicpm_o_4_5，GPU 5，参数 `--text-only`，启动时限 2400 s
- 服务 r6：minicpm_o_4_5，GPU 6，参数 `--text-only`，启动时限 2400 s
- 服务 r7：minicpm_o_4_5，GPU 7，参数 `--text-only`，启动时限 2400 s
- 阶段：
  1. `smoke_w4` 冒烟检查：W4（视频+文本→文本），2 个请求，检查输出非空
  2. `w4_video_10s` 闭环：W4（视频+文本→文本），视频 10 s（素材标签 count_circles）；并发 8, 16, 32, 64；每档 max(24, 3×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
  3. `w4_video_20s` 闭环：W4（视频+文本→文本），视频 20 s（素材标签 count_circles）；并发 8, 16, 32, 64；每档 max(24, 3×并发) 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2 或 e2e 中位数 > 90 s
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：第二批 F1：8 个完整副本（--text-only，每卡一个服务）上的视频理解（W4，10 秒和 20 秒合成视频，文本输出），闭环并发 8, 16, 32, 64（每个副本 1、2、4、8），每档 max(24, 3×并发) 个请求；每卡 7 个 CPU。看视频吞吐是否随副本数线性增长，以及 GPU 空闲比例。
- 运行记录：编排器：排队（等依赖）（尝试 0 次）

### 来源：`configs/experiments/plan_a6000_8gpu_v2_batch2_part5.json`

#### `v2b_d_dep_orig_1x`
- **注意**：还没有放入 Job 454290。等 `v2b_b2l_r8` 成功并录好 trace 后，由 Mac 上的监视脚本放进 plan.d
- 计划条目：优先级 30；依赖 无；申请 8 卡、54 核、780 GB；预计 10 分钟；超时 60 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_d_dep_orig_1x.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r2：minicpm_o_4_5，GPU 2，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r3：minicpm_o_4_5，GPU 3，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r4：minicpm_o_4_5，GPU 4，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r5：minicpm_o_4_5，GPU 5，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r6：minicpm_o_4_5，GPU 6，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r7：minicpm_o_4_5，GPU 7，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `replay` 回放：trace `latest/v2b_b2l_r8/raw/agent_trace_lo_text_c256.jsonl`，模式 dependency，速度 1×，会话复制 1 份，会话到达 trace，内容 auto，请求超时 300 s；路由 least_outstanding
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：第二批 D（回放方法）：在新启动的 8 个完整副本（每卡一个服务，code2wav 单独进程），客户端按在途请求最少路由上回放 v2b_b2l_r8 的 lo_text_c256（256 个并发文本 Agent 的实时运行）。依赖感知回放，内容用原文，全部 2048 个不重复会话，1 倍速。
- 运行记录：没有运行记录

#### `v2b_d_dep_prefix_1x`
- **注意**：还没有放入 Job 454290。等 `v2b_b2l_r8` 成功并录好 trace 后，由 Mac 上的监视脚本放进 plan.d
- 计划条目：优先级 29；依赖 无；申请 8 卡、54 核、780 GB；预计 10 分钟；超时 60 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_d_dep_prefix_1x.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r2：minicpm_o_4_5，GPU 2，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r3：minicpm_o_4_5，GPU 3，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r4：minicpm_o_4_5，GPU 4，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r5：minicpm_o_4_5，GPU 5，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r6：minicpm_o_4_5，GPU 6，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r7：minicpm_o_4_5，GPU 7，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `replay` 回放：trace `latest/v2b_b2l_r8/raw/agent_trace_lo_text_c256.jsonl`，模式 dependency，速度 1×，会话复制 1 份，会话到达 trace，内容 prefix_surrogate，请求超时 300 s；路由 least_outstanding
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：第二批 D（回放方法）：在新启动的 8 个完整副本（每卡一个服务，code2wav 单独进程），客户端按在途请求最少路由上回放 v2b_b2l_r8 的 lo_text_c256（256 个并发文本 Agent 的实时运行）。依赖感知回放，内容用保留前缀的替代内容，全部 2048 个不重复会话，1 倍速。
- 运行记录：没有运行记录

#### `v2b_d_dep_sur_1x`
- **注意**：还没有放入 Job 454290。等 `v2b_b2l_r8` 成功并录好 trace 后，由 Mac 上的监视脚本放进 plan.d
- 计划条目：优先级 28；依赖 无；申请 8 卡、54 核、780 GB；预计 10 分钟；超时 60 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_d_dep_sur_1x.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r2：minicpm_o_4_5，GPU 2，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r3：minicpm_o_4_5，GPU 3，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r4：minicpm_o_4_5，GPU 4，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r5：minicpm_o_4_5，GPU 5，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r6：minicpm_o_4_5，GPU 6，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r7：minicpm_o_4_5，GPU 7，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `replay` 回放：trace `latest/v2b_b2l_r8/raw/agent_trace_lo_text_c256.jsonl`，模式 dependency，速度 1×，会话复制 1 份，会话到达 trace，内容 surrogate，请求超时 300 s；路由 least_outstanding
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：第二批 D（回放方法）：在新启动的 8 个完整副本（每卡一个服务，code2wav 单独进程），客户端按在途请求最少路由上回放 v2b_b2l_r8 的 lo_text_c256（256 个并发文本 Agent 的实时运行）。依赖感知回放，内容用完全替代，全部 2048 个不重复会话，1 倍速。
- 运行记录：没有运行记录

#### `v2b_d_dep_orig_copy4_1x`
- **注意**：还没有放入 Job 454290。等 `v2b_b2l_r8` 成功并录好 trace 后，由 Mac 上的监视脚本放进 plan.d
- 计划条目：优先级 27；依赖 无；申请 8 卡、54 核、780 GB；预计 10 分钟；超时 60 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_d_dep_orig_copy4_1x.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r2：minicpm_o_4_5，GPU 2，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r3：minicpm_o_4_5，GPU 3，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r4：minicpm_o_4_5，GPU 4，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r5：minicpm_o_4_5，GPU 5，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r6：minicpm_o_4_5，GPU 6，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r7：minicpm_o_4_5，GPU 7，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `replay` 回放：trace `latest/v2b_b2l_r8/raw/agent_trace_lo_text_c256.jsonl`，模式 dependency，速度 1×，会话复制 4 份，会话到达 trace，内容 auto，只取前 512 个会话，请求超时 300 s；路由 least_outstanding
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：第二批 D（回放方法）：在新启动的 8 个完整副本（每卡一个服务，code2wav 单独进程），客户端按在途请求最少路由上回放 v2b_b2l_r8 的 lo_text_c256（256 个并发文本 Agent 的实时运行）。依赖感知回放，内容用原文，前 512 个会话复制 4 份，1 倍速。
- 运行记录：没有运行记录

#### `v2b_d_req_orig_1x`
- **注意**：还没有放入 Job 454290。等 `v2b_b2l_r8` 成功并录好 trace 后，由 Mac 上的监视脚本放进 plan.d
- 计划条目：优先级 26；依赖 无；申请 8 卡、54 核、780 GB；预计 10 分钟；超时 60 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_d_req_orig_1x.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r2：minicpm_o_4_5，GPU 2，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r3：minicpm_o_4_5，GPU 3，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r4：minicpm_o_4_5，GPU 4，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r5：minicpm_o_4_5，GPU 5，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r6：minicpm_o_4_5，GPU 6，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r7：minicpm_o_4_5，GPU 7，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `replay` 回放：trace `latest/v2b_b2l_r8/raw/agent_trace_lo_text_c256.jsonl`，模式 request，速度 1×，会话复制 1 份，会话到达 trace，内容 auto，请求超时 300 s；路由 least_outstanding
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：第二批 D（回放方法）：在新启动的 8 个完整副本（每卡一个服务，code2wav 单独进程），客户端按在途请求最少路由上回放 v2b_b2l_r8 的 lo_text_c256（256 个并发文本 Agent 的实时运行）。请求级回放，内容用原文，全部 2048 个不重复会话，1 倍速。
- 运行记录：没有运行记录

#### `v2b_d_dep_orig_2x`
- **注意**：还没有放入 Job 454290。等 `v2b_b2l_r8` 成功并录好 trace 后，由 Mac 上的监视脚本放进 plan.d
- 计划条目：优先级 25；依赖 无；申请 8 卡、54 核、780 GB；预计 10 分钟；超时 60 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_d_dep_orig_2x.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r2：minicpm_o_4_5，GPU 2，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r3：minicpm_o_4_5，GPU 3，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r4：minicpm_o_4_5，GPU 4，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r5：minicpm_o_4_5，GPU 5，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r6：minicpm_o_4_5，GPU 6，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r7：minicpm_o_4_5，GPU 7，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `replay` 回放：trace `latest/v2b_b2l_r8/raw/agent_trace_lo_text_c256.jsonl`，模式 dependency，速度 2×，会话复制 1 份，会话到达 trace，内容 auto，请求超时 300 s；路由 least_outstanding
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：第二批 D（回放方法）：在新启动的 8 个完整副本（每卡一个服务，code2wav 单独进程），客户端按在途请求最少路由上回放 v2b_b2l_r8 的 lo_text_c256（256 个并发文本 Agent 的实时运行）。依赖感知回放，内容用原文，全部 2048 个不重复会话，2 倍速。
- 运行记录：没有运行记录

#### `v2b_d_dep_orig_4x`
- **注意**：还没有放入 Job 454290。等 `v2b_b2l_r8` 成功并录好 trace 后，由 Mac 上的监视脚本放进 plan.d
- 计划条目：优先级 24；依赖 无；申请 8 卡、54 核、780 GB；预计 10 分钟；超时 60 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_d_dep_orig_4x.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r2：minicpm_o_4_5，GPU 2，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r3：minicpm_o_4_5，GPU 3，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r4：minicpm_o_4_5，GPU 4，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r5：minicpm_o_4_5，GPU 5，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r6：minicpm_o_4_5，GPU 6，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r7：minicpm_o_4_5，GPU 7，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `replay` 回放：trace `latest/v2b_b2l_r8/raw/agent_trace_lo_text_c256.jsonl`，模式 dependency，速度 4×，会话复制 1 份，会话到达 trace，内容 auto，请求超时 300 s；路由 least_outstanding
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：第二批 D（回放方法）：在新启动的 8 个完整副本（每卡一个服务，code2wav 单独进程），客户端按在途请求最少路由上回放 v2b_b2l_r8 的 lo_text_c256（256 个并发文本 Agent 的实时运行）。依赖感知回放，内容用原文，全部 2048 个不重复会话，4 倍速。
- 运行记录：没有运行记录

#### `v2b_d_req_orig_4x`
- **注意**：还没有放入 Job 454290。等 `v2b_b2l_r8` 成功并录好 trace 后，由 Mac 上的监视脚本放进 plan.d
- 计划条目：优先级 23；依赖 无；申请 8 卡、54 核、780 GB；预计 10 分钟；超时 60 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_d_req_orig_4x.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r2：minicpm_o_4_5，GPU 2，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r3：minicpm_o_4_5，GPU 3，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r4：minicpm_o_4_5，GPU 4，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r5：minicpm_o_4_5，GPU 5，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r6：minicpm_o_4_5，GPU 6，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r7：minicpm_o_4_5，GPU 7，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `replay` 回放：trace `latest/v2b_b2l_r8/raw/agent_trace_lo_text_c256.jsonl`，模式 request，速度 4×，会话复制 1 份，会话到达 trace，内容 auto，请求超时 300 s；路由 least_outstanding
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：第二批 D（回放方法）：在新启动的 8 个完整副本（每卡一个服务，code2wav 单独进程），客户端按在途请求最少路由上回放 v2b_b2l_r8 的 lo_text_c256（256 个并发文本 Agent 的实时运行）。请求级回放，内容用原文，全部 2048 个不重复会话，4 倍速。
- 运行记录：没有运行记录

#### `v2b_d_dep_prefix_4x`
- **注意**：还没有放入 Job 454290。等 `v2b_b2l_r8` 成功并录好 trace 后，由 Mac 上的监视脚本放进 plan.d
- 计划条目：优先级 22；依赖 无；申请 8 卡、54 核、780 GB；预计 10 分钟；超时 60 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_d_dep_prefix_4x.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r2：minicpm_o_4_5，GPU 2，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r3：minicpm_o_4_5，GPU 3，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r4：minicpm_o_4_5，GPU 4，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r5：minicpm_o_4_5，GPU 5，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r6：minicpm_o_4_5，GPU 6，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r7：minicpm_o_4_5，GPU 7，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `replay` 回放：trace `latest/v2b_b2l_r8/raw/agent_trace_lo_text_c256.jsonl`，模式 dependency，速度 4×，会话复制 1 份，会话到达 trace，内容 prefix_surrogate，请求超时 300 s；路由 least_outstanding
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：第二批 D（回放方法）：在新启动的 8 个完整副本（每卡一个服务，code2wav 单独进程），客户端按在途请求最少路由上回放 v2b_b2l_r8 的 lo_text_c256（256 个并发文本 Agent 的实时运行）。依赖感知回放，内容用保留前缀的替代内容，全部 2048 个不重复会话，4 倍速。
- 运行记录：没有运行记录

#### `v2b_d_dep_sur_4x`
- **注意**：还没有放入 Job 454290。等 `v2b_b2l_r8` 成功并录好 trace 后，由 Mac 上的监视脚本放进 plan.d
- 计划条目：优先级 21；依赖 无；申请 8 卡、54 核、780 GB；预计 10 分钟；超时 60 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_d_dep_sur_4x.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r2：minicpm_o_4_5，GPU 2，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r3：minicpm_o_4_5，GPU 3，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r4：minicpm_o_4_5，GPU 4，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r5：minicpm_o_4_5，GPU 5，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r6：minicpm_o_4_5，GPU 6，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r7：minicpm_o_4_5，GPU 7，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `replay` 回放：trace `latest/v2b_b2l_r8/raw/agent_trace_lo_text_c256.jsonl`，模式 dependency，速度 4×，会话复制 1 份，会话到达 trace，内容 surrogate，请求超时 300 s；路由 least_outstanding
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：第二批 D（回放方法）：在新启动的 8 个完整副本（每卡一个服务，code2wav 单独进程），客户端按在途请求最少路由上回放 v2b_b2l_r8 的 lo_text_c256（256 个并发文本 Agent 的实时运行）。依赖感知回放，内容用完全替代，全部 2048 个不重复会话，4 倍速。
- 运行记录：没有运行记录

### 来源：`configs/experiments/plan_a6000_8gpu_v2_batch2_part6.json`

#### `v2b_sb_s2x6`
- 计划条目：优先级 88；依赖 无；申请 8 卡、54 核、780 GB；预计 10 分钟；超时 60 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_sb_s2x6.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3, 4, 5, 6, 7，参数 `--config configs/models/minicpm_o_4_5_v2_s2x6.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_small` 闭环：W3b（文本提问→文本+语音）；并发 1, 2, 4, 8；每档 64 个请求（另加预热请求：4 个，不计入统计）；停止条件：错误率 > 0.2；**服务端事件**
- 额外采集：服务端事件 w3b_small；torch profiler 无；nsys 无
- 配置里的说明：第二批小 batch 实时服务：一个服务：2 个 thinker（pipeline）副本 + 6 个 talker 副本，code2wav 单独进程、与 talker 同卡。W3b 闭环并发 1, 2, 4, 8，每档 64 个请求（另有 4 个预热），开事件记录，看小 batch 下的 TTFT、端到端 ÷ 语音秒和各阶段时间。
- 运行记录：编排器：成功（尝试 1 次），11-16 19:54 开始，compute-03（8×A6000），GPU 时长 0.81 h；结果 `20261116_v2b_sb_s2x6_run01`：ok，6 分钟

#### `v2b_sb_r8`
- 计划条目：优先级 87；依赖 无；申请 8 卡、54 核、780 GB；预计 10 分钟；超时 60 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_sb_r8.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 r0：minicpm_o_4_5，GPU 0，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r1：minicpm_o_4_5，GPU 1，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r2：minicpm_o_4_5，GPU 2，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r3：minicpm_o_4_5，GPU 3，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r4：minicpm_o_4_5，GPU 4，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r5：minicpm_o_4_5，GPU 5，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r6：minicpm_o_4_5，GPU 6，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 服务 r7：minicpm_o_4_5，GPU 7，参数 `--code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_small` 闭环：W3b（文本提问→文本+语音）；并发 1, 2, 4, 8；每档 64 个请求（另加预热请求：4 个，不计入统计）；停止条件：错误率 > 0.2；**服务端事件**
- 额外采集：服务端事件 w3b_small；torch profiler 无；nsys 无
- 配置里的说明：第二批小 batch 实时服务：8 个完整副本（每卡一个服务，code2wav 单独进程），客户端按在途请求最少路由。W3b 闭环并发 1, 2, 4, 8，每档 64 个请求（另有 4 个预热），开事件记录，看小 batch 下的 TTFT、端到端 ÷ 语音秒和各阶段时间。
- 运行记录：编排器：成功（尝试 1 次），11-16 20:00 开始，compute-03（8×A6000），GPU 时长 0.88 h；结果 `20261116_v2b_sb_r8_run01`：ok，7 分钟

#### `v2b_sb_p4`
- 计划条目：优先级 86；依赖 无；申请 8 卡、54 核、780 GB；预计 10 分钟；超时 60 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_sb_p4.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次，多服务路由 least_outstanding）
- 服务 p0：minicpm_o_4_5，GPU 0, 1，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 p1：minicpm_o_4_5，GPU 2, 3，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 p2：minicpm_o_4_5，GPU 4, 5，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 服务 p3：minicpm_o_4_5，GPU 6, 7，参数 `--talker.gpu 1 --code2wav.gpu 1 --code2wav.process code2wav`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_small` 闭环：W3b（文本提问→文本+语音）；并发 1, 2, 4, 8；每档 64 个请求（另加预热请求：4 个，不计入统计）；停止条件：错误率 > 0.2；**服务端事件**
- 额外采集：服务端事件 w3b_small；torch profiler 无；nsys 无
- 配置里的说明：第二批小 batch 实时服务：4 个拆分服务（每个 1 张 thinker 卡 + 1 张 talker 卡，code2wav 单独进程），客户端按在途请求最少路由。W3b 闭环并发 1, 2, 4, 8，每档 64 个请求（另有 4 个预热），开事件记录，看小 batch 下的 TTFT、端到端 ÷ 语音秒和各阶段时间。
- 运行记录：编排器：成功（尝试 1 次），11-16 20:06 开始，compute-03（8×A6000），GPU 时长 0.80 h；结果 `20261116_v2b_sb_p4_run01`：ok，6 分钟

#### `v2b_sb_s1x7`
- 计划条目：优先级 85；依赖 无；申请 8 卡、54 核、780 GB；预计 10 分钟；超时 60 分钟；失败后最多重试 1 次
- 配置：`configs/experiments/a6000_8gpu_v2/v2b_sb_s1x7.json`（硬件 a6000x8，运行环境 unicorn，素材 synthetic-v1，种子 0，服务退出后自动重启并重跑这一档，最多 4 次）
- 服务 s0：minicpm_o_4_5，GPU 0, 1, 2, 3, 4, 5, 6, 7，参数 `--config configs/models/minicpm_o_4_5_v2_s1x7.yaml`，启动时限 2400 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_small` 闭环：W3b（文本提问→文本+语音）；并发 1, 2, 4, 8；每档 64 个请求（另加预热请求：4 个，不计入统计）；停止条件：错误率 > 0.2；**服务端事件**
- 额外采集：服务端事件 w3b_small；torch profiler 无；nsys 无
- 配置里的说明：第二批小 batch 实时服务：一个服务：1 个 thinker（pipeline）副本 + 7 个 talker 副本，code2wav 单独进程、与 talker 同卡。W3b 闭环并发 1, 2, 4, 8，每档 64 个请求（另有 4 个预热），开事件记录，看小 batch 下的 TTFT、端到端 ÷ 语音秒和各阶段时间。
- 运行记录：编排器：成功（尝试 1 次），11-16 20:12 开始，compute-03（8×A6000），GPU 时长 0.83 h；结果 `20261116_v2b_sb_s1x7_run01`：ok，6 分钟

## Spark（GB10）功能验证

#### `s1_minicpm_text_smoke`
- 配置：`configs/experiments/spark/s1_minicpm_text_smoke.json`（硬件 spark_gb10，运行环境 spark，素材 —，种子 0）
- 服务 s0：minicpm_o_4_5_spark，GPU 0，参数 `--text-only`，启动时限 3600 s
- 阶段：
  1. `smoke_w1` 冒烟检查：W1（文本→文本），3 个请求，检查输出非空
  2. `w1_small` 闭环：W1（文本→文本）；并发 1, 4；每档 8 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2；**服务端事件**
- 额外采集：服务端事件 w1_small；torch profiler 无；nsys 无
- 配置里的说明：Spark GB10 功能验证：MiniCPM-o 4.5 --text-only 能否启动并正确回答；少量并发只用来检查客户端与服务端，数字不与 A6000 比较。
- 运行记录：结果 `20261111_s1_minicpm_text_smoke_run01`：failed，server_crashed，0 分钟；结果 `20261111_s1_minicpm_text_smoke_run02`：ok，4 分钟

#### `s2_minicpm_speech_smoke`
- 配置：`configs/experiments/spark/s2_minicpm_speech_smoke.json`（硬件 spark_gb10，运行环境 spark，素材 —，种子 0）
- 服务 s0：minicpm_o_4_5_spark，GPU 0，启动时限 3600 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_small` 闭环：W3b（文本提问→文本+语音）；并发 1, 2；每档 4 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2；**服务端事件**
  3. `agents_text_small` Agent（合成任务，真实模型请求，录 trace）：4 个会话，并发 2，种子 2，每会话最多 5 步
- 额外采集：服务端事件 w3b_small；torch profiler 无；nsys 无
- 配置里的说明：Spark GB10 功能验证：MiniCPM-o 4.5 默认语音流水线能否在 GB10 上生成语音；同时检查服务端请求事件记录。
- 运行记录：结果 `20261111_s2_minicpm_speech_smoke_run01`：ok，6 分钟

#### `s3_minicpm_speech_memtune_smoke`
- 配置：`configs/experiments/spark/s3_minicpm_speech_memtune_smoke.json`（硬件 spark_gb10，运行环境 spark，素材 —，种子 0）
- 服务 s0：minicpm_o_4_5_spark，GPU 0，参数 `--thinker.engine.mem_fraction_static 0.6`，启动时限 3600 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_small` 闭环：W3b（文本提问→文本+语音）；并发 1, 2；每档 4 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2；**服务端事件**
- 额外采集：服务端事件 w3b_small；torch profiler 无；nsys 无
- 配置里的说明：Spark GB10 A/B：与 s2 相同，只把 thinker 的 engine.mem_fraction_static 从默认 0.85 降到 0.6，检查 code2wav 能否在剩余显存里捕获 Flow CUDA graph（默认配置下因余量不足而跳过），以及对首个音频块时间的影响。
- 运行记录：结果 `20261111_s3_minicpm_speech_memtune_smoke_run01`：ok，4 分钟

#### `s4_minicpm_agents_smoke`
- 配置：`configs/experiments/spark/s4_minicpm_agents_smoke.json`（硬件 spark_gb10，运行环境 spark，素材 synthetic-spark，种子 0）
- 服务 s0：minicpm_o_4_5_spark，GPU 0，启动时限 3600 s
- 阶段：
  1. `assets` 生成合成素材：pool=synthetic-spark，question_tasks=12，task_seed=11，kb_seed=7，passages=2，source_label=synthetic:tts:openbmb/MiniCPM-o-4_5@fb09628e
  2. `smoke_w2` 冒烟检查：W2（语音提问→文本），2 个请求，检查输出非空
  3. `smoke_w4` 冒烟检查：W4（视频+文本→文本），2 个请求，检查输出非空
  4. `agents_text` Agent（合成任务，真实模型请求，录 trace）：9 个会话，并发 3，种子 2，每会话最多 5 步
  5. `agents_voice` Agent（合成任务，真实模型请求，录 trace）：4 个会话，并发 2，种子 11，每会话最多 5 步，语音提问，最后一步语音回答
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：Spark GB10 功能验证：改进后的 Agent（并行工具调用、参数容错、带示例的提示词）；用 MiniCPM-o 自带 TTS 在 Spark 上生成语音提问，验证语音输入、语音回答的 Agent 流程和 W2/W4 输入路径。
- 运行记录：结果 `20261111_s4_minicpm_agents_smoke_run01`：ok，3 分钟

#### `s5_minicpm_replay_profile_smoke`
- 配置：`configs/experiments/spark/s5_minicpm_replay_profile_smoke.json`（硬件 spark_gb10，运行环境 spark，素材 synthetic-spark，种子 0）
- 服务 s0：minicpm_o_4_5_spark，GPU 0，启动时限 3600 s
- 阶段：
  1. `replay_dep` 回放：trace `latest/s4_minicpm_agents_smoke/raw/agent_trace_agents_text.jsonl`，模式 dependency，速度 1×，会话复制 1 份，会话到达 trace，内容 auto
  2. `replay_req` 回放：trace `latest/s4_minicpm_agents_smoke/raw/agent_trace_agents_text.jsonl`，模式 request，速度 1×，会话复制 1 份，会话到达 trace，内容 auto
  3. `replay_voice_dep` 回放：trace `latest/s4_minicpm_agents_smoke/raw/agent_trace_agents_voice.jsonl`，模式 dependency，速度 1×，会话复制 1 份，会话到达 trace，内容 auto
  4. `w3b_torchprof` 闭环：W3b（文本提问→文本+语音）；并发 2；每档 4 个请求（另加预热请求：1 个，不计入统计）；停止条件：错误率 > 0.2；**torch profiler**
- 额外采集：服务端事件 无；torch profiler w3b_torchprof；nsys 无
- 配置里的说明：Spark GB10 功能验证：用 s4 录下的真实 Agent trace 做请求级与依赖感知回放；用 SGLang-Omni 的 torch profiler 采一小段 kernel 时间线并解析 GPU 忙闲区间。
- 运行记录：结果 `20261111_s5_minicpm_replay_profile_smoke_run01`：ok，5 分钟

#### `s6_minicpm_torchprof_smoke`
- 配置：`configs/experiments/spark/s6_minicpm_torchprof_smoke.json`（硬件 spark_gb10，运行环境 spark，素材 synthetic-spark，种子 0）
- 服务 s0：minicpm_o_4_5_spark，GPU 0，启动时限 3600 s
- 阶段：
  1. `w3b_torchprof` 闭环：W3b（文本提问→文本+语音）；并发 2；每档 4 个请求（另加预热请求：1 个，不计入统计）；停止条件：错误率 > 0.2；**torch profiler**
- 额外采集：服务端事件 无；torch profiler w3b_torchprof；nsys 无
- 配置里的说明：Spark GB10 功能验证：torch profiler 时间窗（修复后的按阶段命名与等待导出），并解析 kernel 忙闲区间。
- 运行记录：结果 `20261111_s6_minicpm_torchprof_smoke_run01`：ok，4 分钟

#### `s7_minicpm_replay_scaling`
- 配置：`configs/experiments/spark/s7_minicpm_replay_scaling.json`（硬件 spark_gb10，运行环境 spark，素材 synthetic-spark，种子 0）
- 服务 s0：minicpm_o_4_5_spark，GPU 0，启动时限 3600 s
- 阶段：
  1. `dep_x1` 回放：trace `latest/s4_minicpm_agents_smoke/raw/agent_trace_agents_text.jsonl`，模式 dependency，速度 1×，会话复制 1 份，会话到达 trace，内容 auto
  2. `req_x1` 回放：trace `latest/s4_minicpm_agents_smoke/raw/agent_trace_agents_text.jsonl`，模式 request，速度 1×，会话复制 1 份，会话到达 trace，内容 auto
  3. `dep_x4_poisson` 回放：trace `latest/s4_minicpm_agents_smoke/raw/agent_trace_agents_text.jsonl`，模式 dependency，速度 1×，会话复制 4 份，会话到达 poisson:0.5，内容 auto
  4. `req_x4_speed2` 回放：trace `latest/s4_minicpm_agents_smoke/raw/agent_trace_agents_text.jsonl`，模式 request，速度 2×，会话复制 4 份，会话到达 trace，内容 auto
  5. `dep_x4_speed2` 回放：trace `latest/s4_minicpm_agents_smoke/raw/agent_trace_agents_text.jsonl`，模式 dependency，速度 2×，会话复制 4 份，会话到达 trace，内容 auto
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：Spark GB10 内部对照：按 A6000 x6 的规模回放 s4 的文本 Agent trace——依赖感知 vs 请求级，会话复制 4 倍、Poisson 到达、2 倍加速。结论只适用于 GB10，同时检验 x6 规模下的分析流程。
- 运行记录：结果 `20261111_s7_minicpm_replay_scaling_run01`：ok，6 分钟

#### `s8_minicpm_nsys_smoke`
- 配置：`configs/experiments/spark/s8_minicpm_nsys_smoke.json`（硬件 spark_gb10，运行环境 spark，素材 —，种子 0）
- 服务 s0：minicpm_o_4_5_spark，GPU 0，参数 `--text-only`，启动时限 3600 s，启动前缀 `/usr/local/bin/nsys profile -t cuda,nvtx --sample=none --delay=200 --duration=40 --kill=none --force-overwrite=true -o <实验目录>/profiling/…`
- 阶段：
  1. `align` 空闲 25 s
  2. `w1_nsys` 闭环：W1（文本→文本）；并发 4；每档 28 个请求（另加预热请求：0 个，不计入统计）；停止条件：错误率 > 0.2
  3. `flush` 空闲 30 s
- 额外采集：服务端事件 无；torch profiler 无；nsys s0
- 配置里的说明：Spark GB10 功能验证：用 Nsight Systems 采集 MiniCPM-o text-only 服务进程的一段 CUDA 时间线（服务就绪后约 40 秒），导出 SQLite，用 analysis.nsys 计算 GPU 忙闲区间。
- 运行记录：结果 `20261111_s8_minicpm_nsys_smoke_run01`：ok，4 分钟

#### `s9_agent_vs_closedloop_pilot`
- 配置：`configs/experiments/spark/s9_agent_vs_closedloop_pilot.json`（硬件 spark_gb10，运行环境 spark，素材 —，种子 0）
- 服务 s0：minicpm_o_4_5_spark，GPU 0，参数 `--text-only`，启动时限 3600 s
- 阶段：
  1. `closed_w1_c4` 闭环：W1（文本→文本）；并发 4；每档 32 个请求（另加预热请求：4 个，不计入统计）；停止条件：错误率 > 0.2
  2. `agents_c4_notool` Agent（合成任务，真实模型请求，录 trace）：24 个会话，并发 4，种子 21，每会话最多 5 步
  3. `agents_c4_toollat` Agent（合成任务，真实模型请求，录 trace）：24 个会话，并发 4，种子 21，每会话最多 5 步，模拟工具延迟 对数正态（中位数 0.5 s，σ 0.5）
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：Spark GB10 内部对照（试点）：同一服务、相同并发 4 下，比较闭环压测与 Agent 会话（无/有模拟工具延迟）的 GPU 采样利用率和空闲比例，观察请求依赖和工具等待带来的 GPU 空闲。只用于检验分析方法，结论不推广到 A6000。
- 运行记录：结果 `20261111_s9_agent_vs_closedloop_pilot_run01`：ok，4 分钟

#### `s10_minicpm_talker_replica_functional`
- 配置：`configs/experiments/spark/s10_minicpm_talker_replica_functional.json`（硬件 spark_gb10，运行环境 spark，素材 —，种子 0）
- 服务 s0：minicpm_o_4_5_spark，GPU 0，参数 `--config configs/models/minicpm_o_4_5_talker_replica2_samegpu.yaml`，启动时限 3600 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_small` 闭环：W3b（文本提问→文本+语音）；并发 1, 4；每档 8 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2；**服务端事件**
- 额外采集：服务端事件 w3b_small；torch profiler 无；nsys 无
- 配置里的说明：功能测试，不是性能实验。x7c（只复制 talker）在 A6000 上因 YAML 缺 model_path 启动失败；修好后在 Spark 上用同类配置（两个 talker 副本都在 GPU 0）检查服务能否启动、请求能否完成、两个副本是否都接到请求。
- 运行记录：结果 `20261111_s10_minicpm_talker_replica_functional_run01`：failed，server_crashed，0 分钟；结果 `20261111_s10_minicpm_talker_replica_functional_run02`：ok，4 分钟

#### `s11_minicpm_replay_payload_fidelity`
- 配置：`configs/experiments/spark/s11_minicpm_replay_payload_fidelity.json`（硬件 spark_gb10，运行环境 spark，素材 —，种子 0）
- 服务 s0：minicpm_o_4_5_spark，GPU 0，参数 `--text-only`，启动时限 3600 s
- 阶段：
  1. `replay_original` 回放：trace `unicorn/experiments/20261111_f3_minicpm_agents_large_run01/raw/agent_trace_agents_text_c16.jsonl`，模式 dependency，速度 1×，会话复制 1 份，会话到达 trace，内容 auto
  2. `replay_surrogate` 回放：trace `unicorn/experiments/20261111_f3_minicpm_agents_large_run01/raw/agent_trace_agents_text_c16.jsonl`，模式 dependency，速度 1×，会话复制 1 份，会话到达 trace，内容 surrogate
  3. `replay_original_again` 回放：trace `unicorn/experiments/20261111_f3_minicpm_agents_large_run01/raw/agent_trace_agents_text_c16.jsonl`，模式 dependency，速度 1×，会话复制 1 份，会话到达 trace，内容 auto
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：Spark 实验（不是 A6000 基准）：同一份 Agent trace（A6000 f3 的文本 Agent，并发 16 采集，128 个会话）用两种请求内容回放。原文：重发 trace 中保存的请求；替代：同样字符数的随机文本、按记录的输出 token 数固定输出长度。两者的依赖结构、时间间隔和输出长度相同，差别只在内容，因此主要反映前缀共享（f3 中 thinker 前缀缓存命中 93% 的输入 token）对测量结果的影响。最后再跑一次原文，检查顺序影响。
- 运行记录：结果 `20261111_s11_minicpm_replay_payload_fidelity_run01`：ok，4 分钟

#### `s12_minicpm_crash_soak_gb10`
- 配置：`configs/experiments/spark/s12_minicpm_crash_soak_gb10.json`（硬件 spark_gb10，运行环境 spark，素材 —，种子 0）
- 服务 s0：minicpm_o_4_5_spark，GPU 0，启动时限 3600 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4 请求/秒；每档 48 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：Spark 实验：A6000 上默认语音流水线在开环负载下平均约 5 分钟崩溃一次（talker 进程 illegal memory access）。这里检验 GB10 上是否复现：默认配置，W3b 开环 0.4 请求/秒（GB10 较慢，按容量约一半设定），每 2 分钟一档，最多 15 档。如果复现，崩溃定位可以在 Spark 上做，不必占用 A6000。
- 运行记录：结果 `20261111_s12_minicpm_crash_soak_gb10_run01`：ok，48 分钟

#### `s13_minicpm_pipeline_replica_functional`
- 配置：`configs/experiments/spark/s13_minicpm_pipeline_replica_functional.json`（硬件 spark_gb10，运行环境 spark，素材 —，种子 0）
- 服务 s0：minicpm_o_4_5_spark，GPU 0，参数 `--config configs/models/minicpm_o_4_5_pipeline2_talker2_samegpu.yaml`，启动时限 3600 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_small` 闭环：W3b（文本提问→文本+语音）；并发 1, 4；每档 8 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2；**服务端事件**
- 额外采集：服务端事件 w3b_small；torch profiler 无；nsys 无
- 配置里的说明：功能测试，不是性能实验。8 卡计划里的“thinker 池 + talker 池”布局需要复制 thinker 所在的 pipeline 进程。离线检查已通过；这里在 Spark 上把 pipeline 和 talker 各复制两份、都放在 GPU 0，检查服务能否启动、请求能否完成、两个 thinker 副本是否都接到请求。
- 运行记录：结果 `20261111_s13_minicpm_pipeline_replica_functional_run01`：ok，6 分钟

#### `s14_minicpm_code2wav_replica_functional`
- 配置：`configs/experiments/spark/s14_minicpm_code2wav_replica_functional.json`（硬件 spark_gb10，运行环境 spark，素材 —，种子 0）
- 服务 s0：minicpm_o_4_5_spark，GPU 0，参数 `--config configs/models/minicpm_o_4_5_c2w_replica2_samegpu.yaml`，启动时限 3600 s
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_small` 闭环：W3b（文本提问→文本+语音）；并发 1, 4；每档 8 个请求（另加预热请求数与并发数相同，不计入统计）；停止条件：错误率 > 0.2；**服务端事件**
- 额外采集：服务端事件 w3b_small；torch profiler 无；nsys 无
- 配置里的说明：功能测试，不是性能实验。plan v2 的 S 布局把 code2wav 放在自己的进程里，并与 talker 一样复制多份（避开第 17 节的崩溃）。离线检查通过，但运行时还没验证过：talker 副本能否把结果交给单独进程的 code2wav 副本。这里在 Spark 上把 talker 和 code2wav 各复制两份、都放在 GPU 0，检查服务能否启动、语音请求能否完成、两个 code2wav 副本是否都接到请求。
- 运行记录：结果 `20261112_s14_minicpm_code2wav_replica_functional_run01`：ok，4 分钟

## 五、写进计划文档、但还没有生成配置的实验

**plan v2（`docs/04_experiments/plan_8gpu_v2.md` 第 4 节）**

| 名称 | 内容 |
|---|---|
| E1 Q-A | Qwen3-Omni：两条流水线，每条是 TP=2 thinker + 1 张 talker 卡（6 卡） |
| E1 Q-B | Qwen3-Omni：一条流水线，TP=2 thinker + 6 个 talker 副本（8 卡） |
| E1 Q-C | Qwen3-Omni：两条流水线，每条是 TP=2 thinker + 2 张 talker 卡（8 卡） |
| E2 | 在 E1 最好的配置上，打开和关闭 `enable_partial_start`，比较首个音频块时间、e2e 和播放卡顿 |
| E3 | TP=2 的 thinker 放在 NVLink 对上并打开 P2P，与放在 PCIe 对上并关闭 P2P 对比 |
| F2 | 单副本，服务 CPU 配额 2、4、8、16 核，客户端固定在配额以外的核上 |
| G2 | MiniCPM-o 文本服务：thinker TP=2（分别放在 NVLink 对和 PCIe 对上），对比 2 个 TP=1 副本 |

v1 8 卡计划里有相近的配置，但都没有运行（见第四节“8 卡计划 v1”）：
- `y4_minicpm_thinker_tp2_text` 和 `y4b_minicpm_two_text_replicas`，接近 G2，但没有区分 NVLink 对和 PCIe 对；
- `y5_qwen3_omni_two_pipelines_6gpu`，与 E1 Q-A 的卡数分法相同；
- `y6_minicpm_video20s_cpu4`、`y6_minicpm_video20s_cpu16`，是视频负载下的 CPU 配额对照，与 F2 部分重合，但客户端没有固定在配额以外的核上。

**第三批候选（第二批中期报告第 6 节，还没有写进计划文档）**

| 名称 | 内容 |
|---|---|
| 量化 | thinker 权重 INT8/INT4，只量化权重 |
| thinker TP=2 | 放在 NVLink 对上，测小 batch 延迟 |
| 投机解码 | sglang-omni 不支持，只在纯 SGLang 上测 thinker 语言模型，作为参考 |
| 流式流水线 | 用 Qwen3-Omni 测，即 E2 |
| code2wav 独立 GPU | 例如 2 个 thinker 卡 + 4 个 talker 卡 + 2 个 code2wav 卡 |
| prefill/decode 拆分 | 放在最后 |


## 六、不在任何计划里的配置

下面的配置写好后没有放进任何计划，也没有手动运行（说明里写了它们是哪个条件下的备选）。

#### `zl1_minicpm_crash_talker_no_lookahead`
- 配置：`configs/experiments/a6000/zl1_minicpm_crash_talker_no_lookahead.json`（硬件 a6000x1，运行环境 unicorn_diag，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s，启动前缀 `env SGLANG_OMNI_DIAG_NO_LOOKAHEAD=1`
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：验证崩溃根因（第 17 节），只在 zs1（整段 code2wav batch 串行化）仍崩溃时运行。诊断副本加 env 开关 SGLANG_OMNI_DIAG_NO_LOOKAHEAD=1：只让 talker 的 decode 走同步路径（不做一步前瞻；z5 想做同样的事，但工厂函数不接受参数），thinker 不变。CUDA_LAUNCH_BLOCKING=1 能避免崩溃而串行化不能时，CPU 跑在 GPU 前面的那一步是主要嫌疑。负载与 z4 相同。
- 运行记录：没有运行记录

#### `zm1_minicpm_crash_code2wav_private_pool_rep2`
- 配置：`configs/experiments/a6000/zm1_minicpm_crash_code2wav_private_pool_rep2.json`（硬件 a6000x1，运行环境 unicorn_diag，素材 synthetic-v1，种子 0）
- 服务 s0：minicpm_o_4_5，GPU 0，启动时限 2400 s，启动前缀 `env SGLANG_OMNI_DIAG_PRIVATE_POOL_STAGES=code2wav`
- 阶段：
  1. `smoke_w3b` 冒烟检查：W3b（文本提问→文本+语音），2 个请求，检查输出非空
  2. `w3b_soak` 开环（泊松到达）：W3b（文本提问→文本+语音）；速率 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 请求/秒；每档 180 个请求；某档错误率 > 0.2 时停止
- 额外采集：服务端事件 无；torch profiler 无；nsys 无
- 配置里的说明：zm1 的重复（只在 zm1 未崩溃时运行），配置完全相同。验证崩溃根因（第 17 节）。同一进程里，talker 线程、主线程和 code2wav 线程共用 PyTorch 的显存缓存分配器；code2wav 单独进程（z11b）不崩，可能是因为分配器分开了。这里用诊断副本（同一提交 520b49ee，加 configs/runtime/patches/sglang_omni_diag_private_stream.patch 里的 MemPool 开关），设 SGLANG_OMNI_DIAG_PRIVATE_POOL_STAGES=code2wav，让 code2wav 调度线程的所有显存分配都走一个私有 MemPool（torch.cuda.use_mem_pool，只作用于当前线程），其他一切保持默认（同一进程、stream 0 上的 HiFT、CUDA graph、torch.compile），负载与 z4 相同。若不再崩溃，说明跨线程复用同一块缓存显存是必要条件；若仍崩溃，分配器共享可以排除。
- 运行记录：没有运行记录

