# P1、P2、P3 的执行设计（CPU 准备，运行之前写定）

> 2026-11-18 晚。三个实验都**还没有 GPU 运行**；本文只写：原登记（`research/next_steps/p0_p3_experiment_plan.md`）里没写到、但执行前必须定下来的细节，以及 CPU 检查的证据。原登记的假设和判定规则一条不改。
> 证据等级：Measured（CPU 检查实际跑过）/ Estimated（时间估计）。

## 0. 总览

| 实验 | 现在的状态 | 可以直接放进队列的计划 | 条件计划 |
|---|---|---|---|
| P1 按长度组批 + 最长等待（2+2+4，饱和区） | 补丁、CPU 检查、计划生成器、分析脚本完成 | `configs/experiments/plan_ppv_p1_prerun.json`（预跑，定 X） | `plan_ppv_p1.json`（预跑后把 X 填进 `make_plan_ppv_p1.py` 的 `P1_X` 再生成）；`plan_ppv_p1_interaction.json`（只在 C1 通过时） |
| P2 隐藏状态旁路复制的逐位检查 | 补丁、CPU 检查、计划、门槛分析脚本完成 | `plan_ppv_p2_verify.json`（1 个运行） | `plan_ppv_p2_perf.json`（实验 11 登记的 9 个性能运行，只在逐位检查通过后） |
| P3 停顿 | gc 日志和调用栈转储进了 P1 的补丁；停顿检测器 `stall_watch.py` 完成（runner 挂钩待加，见第 4 节） | 无 | `plan_ppv_p3.json`（本地盘对 NFS，ABBA，预算有剩余才放） |

## 1. P1：按长度组批 + 最长等待

### 1.1 补丁

- `configs/runtime/patches/sglang_omni_batch_ppv_p1.patch`（sha256 `59f177fa…e91`）= 实验 10 的组批补丁 + 下面三件事。运行时 `configs/models/runtime_unicorn_ppv_maxwait.json`，集群 worktree `repos/sglang-omni-ppv-maxwait`（520b49ee + 这个补丁）。
- **最长等待**（按登记的算法）：`SimpleScheduler.enqueue` 收到消息时记单调时钟（和墙钟，给事件用）；控制文件多一个键 `max_wait_ms`（数字 ≥ 0，或 `null` = 不老化；缺这个键保持原值；文字、负数、无穷、NaN 一律忽略）。取批时（只在 `policy: length` 下）先把窗口里等了 ≥ τ 的请求按到达顺序放进批（不超过批上限），剩下的名额再按“和最老请求的长度最接近”挑；没挑上的照旧按到达顺序放回队首。两个边界在登记里没写，这里定下：**等待恰好等于 τ 算超时**；一个超时请求放不进 `max_batch_cost` 时**这一批就此结束**，不让更年轻的请求顶它的位置（code2wav 默认没有 `max_batch_cost`，这条实际不会触发）。
- **事件**：`exp_batch_formed` 多出 `enqueue_ns`（进入调度器的墙钟）、`waited_ms`、`is_forced`、`forced_count`、`anchor_aged`、`max_wait_ms`。code2wav 排队时间因此可以直接用“入队 → 出队”（P3 也要这个入队时刻）。
- **P3 的两个低开销钩子**（`SGLANG_OMNI_EXP_TRACE` 里写 `gc`、`stack` 才生效，运行时配置默认写上）：每个进程第 2 代垃圾回收的起点和时长（回调里只记到内存，由一个每秒醒一次的线程写日志行 `exp_gc ...`，开事件时也写事件）；`faulthandler` 注册 SIGUSR2，把所有线程的 Python 调用栈写进 `SGLANG_OMNI_EXP_STACK_DIR/stack_<pid>.txt`，文件第一行记 pid 和进程启动时间。

### 1.2 CPU 检查（Measured）

| 检查 | 结果 |
|---|---|
| `check_sglang_omni_ppv_maxwait.py`（控制文件、老化顺序、批上限、两个端点、种子固定的饥饿模拟、成本上限、批事件字段、中止不留痕、gc、SIGUSR2） | Mac（torch 2.14.1）30 项通过（SIGUSR2 两项在 macOS 跳过）；集群登录节点，运行时自己的 python（torch 2.13.0+cu130）32 项全部通过，含 SIGUSR2 |
| 实验 10 原来的检查 `check_sglang_omni_batch.py` 在新补丁上（回归） | 集群上通过（15 项，含 code2wav 的 prompt 长度记录） |
| 新检查在实验 10 的补丁上 | 失败（`SimpleScheduler` 没有 `max_wait_s`）——检查确实依赖新代码 |
| 变异检查（13 个变异：去掉老化、倒序、`>=` 改 `>`、不关批、不清中止、FIFO 不清零、缺键重置、忽略批上限、不标强制、不记入队时刻、接受负数、gc 记所有代、gc 总是注册） | 13/13 被检查发现 |
| 饥饿模拟（400 个请求、到达率 25/s、每批 0.4 s、τ = 1 s） | 每个请求恰好出去一次；没有一个超时请求被更年轻的请求超过；超时请求按到达顺序出去 |
| 补丁重放 | 在干净的 520b49ee 上 `git apply` 后与 Mac worktree 逐文件相同；集群 worktree 的 8 个文件 blob hash 与 Mac 相同 |
| `check_pipeline_config.py`：`t2_k2_c4` 和四组变体用的打包 yaml（`enable_flow_variable_length`、`enable_flow_block_compile`） | 集群上 `"ok": true` |

### 1.3 执行细节（登记补充，看到任何 P1 数据之前写定）

| 项 | 定下来的做法 | 为什么 |
|---|---|---|
| X 的定义 | 预跑闭环（W3b 256、W3L 128、MIX 192，FIFO16，不开事件）的**满额时段吞吐**：测量请求第一次发出到最后一次发出之间完成的请求数 ÷ 这段时间。登记容量（第一次发出 → 最后完成）同时报告 | 登记容量把最后的排空也算进去，实验 9 里比满额时段低 20–36%；用它定 1.1X 可能根本不超载 |
| 速率 | 0.9X、1.0X、1.1X，四舍五入到 0.5 请求/秒（实验 9 的规则） | 同一规则 |
| 每格 | 一个开环段 = 一种负载 × 一种设置 × 一档速率；先 15 s（W3L 20 s）预热，分析时按计划发送时间丢掉，再测登记的 75 s（W3L 90 s）；每格前都有一个控制段（4 个键全写，等 3 s） | 开环记录没有预热标记；控制文件缺键会沿用上一个设置 |
| 顺序和种子 | 每个区组一次启动；负载顺序按区组轮换（W3b/W3L/MIX → W3L/MIX/W3b → MIX/W3b/W3L）；每种负载内 15 格完全随机（种子 20261119）；到达过程的种子 = 20261119 + 区组号（再加速率），所以**同一区组同一速率的 5 种设置看到的是同一串请求和到达时刻** | 配对比较 |
| 事件 | 全部格子开服务端事件（登记）；预跑不开 | — |
| 指标口径 | 提供负载 = 这一格的速率；完成吞吐 = 测量请求中成功完成的数 ÷ 它们的发送窗口；goodput = TTFT ≤ 1 s 且 端到端/语音秒 ≤ 0.5 的测量请求数 ÷ 发送窗口；失败和超时在分位数里算无穷慢 | — |
| 稳态和“稳定容量” | 稳态 = 实验 9 的规则（发送窗口后半段在途请求数的斜率 ≤ 0.05 × 速率）。稳定容量 = 某区组、某负载、某设置里**稳态的最高速率**；最高档也稳态时只是下界；一档都不稳态写 “< 0.9X”。**不稳态的格子的吞吐不叫容量** | 1.1X 预计不稳态（超载） |
| 长度分组 | 每格内按语音秒的中位数分成短、长；P99 只在一组 ≥ 100 个请求时给，否则 N/A 并报样本数 | — |
| 饥饿 | 每组排队 P95/P99/最大值；“排队超过同区组同速率 FIFO16 格同组 P95 的 3 倍”的比例；有强制选入的批的占比 | 登记 |
| H1b、H1c 用哪几档 | 登记没写档位，这里定下：H1a、H1b 用 MIX 的 1.0X 和 1.1X（3 个区组 × 2 档都要满足）；H1c 用 W3b 的全部三档（3 个区组 × 3 档） | H1a 原文是“1.0X、1.1X 两档”；短请求的长尾只在排队时出现 |
| 排除 | 错误 + 超时 > 5% 的格子、服务重启所在的段、控制文件没在每一批生效的格子，不进判定；缺一个区组的判定写“数据不足” | 沿用实验 9、10 |
| 四组变体（只在 C1 通过时） | 稠密/打包 flow × FIFO16/Length8+1 s，MIX 和 W3b 的 1.0X、1.1X；每个区组两个运行（换 flow 要重启），flow 顺序按区组交替，两者的格子和顺序相同；同样的绝对速率（同一提供负载） | 分开报四组，不把两个加速比相加 |

### 1.4 时间（Estimated）

- 预跑：约 25 分钟 × 8 卡 ≈ 3.3 GPU-时。
- 正式：每个区组 45 格 × 每格（控制 3 s + 预热和测量 90/110 s + 排空 10–30 s）≈ 92 分钟，加启动约 6 分钟；3 个区组 ≈ **4.9 小时 × 8 卡 ≈ 39 GPU-时**。原登记估计 3.5 小时偏低，原因是登记的测量时长本身（45 格 × 75–90 s ≈ 1 小时/区组）没算预热、排空和控制。
- 如果预算不够，唯一合理的缩减（必须在放进队列前决定并写进这里）：W3b、W3L 上只跑 FIFO16、Length8、Length8+1 s（MIX 保留 5 种设置，H1a/H1b/H1c 都不受影响），约 4.4 小时。
- 四组变体：6 个运行 × 约 25 分钟 ≈ 2.5 小时 × 8 卡 ≈ 20 GPU-时。

### 1.5 分析

`scripts/analysis/ppv_p1_report.py`：`--prerun` 给出 X 和速率；`--run ×3` 给出每格的客户端指标、code2wav 排队（实验 10 的“分派 → 出队”为登记值，“入队 → 出队”为交叉检查）、补齐比例、CUDA graph 命中（按 flow 求解路径数批次，**不是** kernel 时间占比）、强制选入、饥饿、稳定容量、配对对比和 H1a/H1b/H1c。测试 `tests/test_ppv_p1_report.py`（合成记录，11 项）。

## 2. P2：隐藏状态旁路复制的逐位检查

### 2.1 补丁

- `configs/runtime/patches/sglang_omni_async_ppv_p2.patch`（sha256 `91688e33…a28`）= 实验 11 的补丁 + 模式 `SGLANG_OMNI_SPEECH_LOOKAHEAD=sidecopy_verify`。与实验 11 的补丁相比，**只有** `minicpm_o/thinker_model_runner.py` 变了：两行改动（错误信息；`side_copy` 在 sidecopy 和 sidecopy_verify 下都开），其余全是新增。其他 9 个文件与实验 11 的逐字节相同。运行时 `runtime_unicorn_ppv_verify.json`，集群 worktree `repos/sglang-omni-ppv-verify`（10 个文件的 blob hash 与 Mac 相同）。
- verify 模式做的事（按登记）：旁路流上的拷贝照常做，下游用的就是它；同时每一步在启动流上对同一批快照行再拷一份参照（排在旁路拷贝和它的事件之后，不改变旁路流等什么）；请求结束时：(1) 参照行在启动流上 stack 后拷到 CPU；(2) `torch.cuda.synchronize()` 之后把旁路这边的设备行再读一次；比较 旁路 == 参照、旁路 == 重读（`torch.equal`，同 shape、同 dtype），行数 = 期望行数（没有 known_tts 时 = `len(output_ids)`，依据 `talker_request.py`“第一个隐藏行是最后一个提示位置”，每次前向一行；有 known_tts 时 = `len(known_tts_output_ids)`），每行记下当时的 `len(output_ids)` 检查“一步一行”。不相等时记请求 id、第一处不同的行、三份的哈希、步号、这一步的批大小和这一步结束的请求数，并写一条 warning（不开事件也有）。中止（`reset_request`）清掉参照。

### 2.2 CPU 检查（Measured）

替身设备模型：启动流上的结果只有在旁路流等过它之后记录的事件、或者设备同步之后，才对旁路流可见，否则读到旧内存；没被 `record_stream` 保护的快照，可能被分配器交给后面的启动流工作。

| 检查 | 结果 |
|---|---|
| `check_sglang_omni_ppv_verify.py`（模式；正确代码三份相等、行数和“一步一行”；“读早了（每步）”“被覆盖（没有 record_stream）”“读早了（请求结束）”三类错误都报出、第一处坏行位置正确；known_tts 切片；漏步和行数不符；warning；中止清理） | Mac（同一 sglang 0.5.21 源码树）19 项通过；集群登录节点运行时 python 19 项通过 |
| 实验 11 原来的检查在新补丁上（回归） | Mac 和集群都通过 |
| 新检查在实验 11 的补丁上 | 失败（`sidecopy_verify` 是未知模式） |
| 变异检查：verify 本身 9 个（自己和自己比、参照取旁路行、不切片、中止不清、期望行数减一、对齐恒真、不写 warning、verify 关掉、去掉同步） | 8/9 被发现；“去掉同步”没被发现——这是**等价变异**：旁路这边的 `.to("cpu")` 已经让主机等完旁路流，重读又在启动流上按顺序执行，所以多一次设备同步不改变能观察到的结果（登记要求的同步保留，作为保险） |
| 变异检查：旁路复制本身的 3 个真实 bug（不等快照事件、不 `record_stream`、结束时不等最后一行的事件） | 3/3 被 verify 模式报出 |

### 2.3 门槛（登记的规则，补上口径）

`scripts/analysis/ppv_p2_report.py`：**通过**当且仅当 (1) 每个 `exp_hidden_verify` 事件的四项检查都为真；(2) 每个检查段里成功结束、带语音的请求都恰好有一个 verify 事件（没有事件的请求算“未验证”，不算通过）；(3) 服务日志里没有 `exp_hidden_verify mismatch`（这也覆盖了不开事件的冒烟段）；(4) 除了 `chk_timeout` 段故意的客户端超时以外没有客户端错误，没有服务重启。任何一条不满足 → **不通过**，只报告证据，不放性能运行。
**事先写定的勘误处理**：如果参照本身的行数就不等于期望行数（说明“期望行数”的定义错了），记为 erratum，行数检查改用“旁路行数 = 参照行数”；这**永远不会**把逐位不相等变成通过。

### 2.4 局限

- verify 检查的是**拷贝路径**（旁路流的等待、显存生命周期、请求结束的拷贝），不是快照本身：如果快照之前图缓冲区就被下一步覆盖（实验 8 的 nosnapshot 情形），参照和旁路拷的是同一块错的数据，查不出来。
- verify 每步多一次 clone、每个请求多一次阻塞拷贝和设备同步，时序和纯 sidecopy 不同。通过只说明“在 verify 的时序下逐位正确”；之后的性能运行用实验 11 的运行时（sidecopy 的代码与这里检查的相同）。
- 实验 11 原来“未通过”的结论不改写；P2 是新的检查，结果单独报告。

### 2.5 计划和时间（Estimated）

- `ppv_p2_verify`：2+6，实验 11 的检查段落（贪心负载并发 1/8/32/64，文字/语音、短/长、带/不带 seed 混合，客户端超时中止段和之后的段，W3XL），开事件和 digest。约 20 分钟 × 8 卡 ≈ 2.7 GPU-时。
- `ppv_p2_perf_block{1,2,3}_{A,B,C}`：实验 11 登记的 9 个运行，只换 ID，运行时仍是 `unicorn_async`。约 2.5–3.8 小时 × 8 卡。**只在门槛通过后放**。

## 3. P3：停顿

### 3.1 低开销观测（随其他实验一起收集）

| 观测 | 在哪 | 开销 |
|---|---|---|
| Linux 压力指标、每组进程状态 | 已在 runner 的 CPU 采样器里（父会话加的） | 每秒读 `/proc` |
| 第 2 代 GC 的起点和时长 | P1 补丁的 `gc` 部分（日志行 + 事件） | 回调里只追加内存 |
| code2wav 入队时刻、阶段交接事件 | P1 补丁的批事件；开事件的段落都有 `stage_hop_*` | 只在开事件时 |
| 服务日志断档 > 1 s 时：每个进程的状态和 wchan、所有线程的 Python 调用栈 | `src/omni_research/orchestration/stall_watch.py` + P1 补丁的 `stack` 部分 | 每 0.1 s stat 一次日志；断档时才读 `/proc`、发 SIGUSR2；每 30 s 最多一次、每个运行最多 20 次 |
| 日志 stat 本身慢（> 0.2 s） | 同上 | — |

- **py-spy**：运行时 venv 里有，但集群的 `ptrace_scope = 1`（登录节点和 the 8-GPU node 都是），py-spy 作为 runner 的子进程不能附着到兄弟进程，所以默认不用，改用进程自己注册的 SIGUSR2 调用栈转储（标准库 `faulthandler`）。SIGUSR2 会结束没注册的进程，所以只发给“注册文件里的启动时间和当前进程一致”的 pid。
- 停顿检测只在负载段“上膛”（`arm()`）：控制段、事件开关、段间空闲时日志安静是正常的，不算停顿。

### 3.2 条件对照（本地盘对 NFS）

`plan_ppv_p3.json`：4 个运行，顺序 NFS、本地盘、本地盘、NFS（ABBA，避免一晚上的漂移被当成差别）；每个运行 t2_k2_c4、P1 运行时、不写控制文件（默认 FIFO16）、gc 和停顿检测开（调用栈转储按偏离 D3 去掉）；冒烟后 W3b 开环 21 请求/秒 6 段 × 300 秒，开事件。本地盘运行把 `OMNI_EXP_DIR` 指到 `/scratch/$USER/omni_ppv/<id>`，跑完 `rsync` 回编排器的结果目录再删掉（11-18 时 /scratch 剩 219 GB，一次运行写几 GB）。**局限**：import、模型权重、编译缓存两边都在 NFS 上，对照只隔离“运行自己写的日志、事件和客户端记录”。约 2.5 小时 × 8 卡 ≈ 20 GPU-时，预算有剩余才放。
**结论规则（登记）**：本地盘运行里“负载段内全进程日志断档 > 1 s”消失、NFS 运行里仍出现，或断档时 PSI io / 慢 stat 尖峰 → 支持“共享文件系统”；四个运行都没有停顿 → 写“Root cause remains unknown; no stall was observed during the new measurement window”，并报每个对照的观察时长、停顿次数、负载和日志证据。一次没有停顿不能证明 NFS 是原因。

## 4. 放进队列之前还要做的事

1. **runner 挂钩**（`src/omni_research/orchestration/experiment.py`，本 fork 没改它）——建议加的几行：
   - `ExperimentRunner.__init__` 里：`self.stall_watchers: list = []`
   - `self.start_servers()` 和 `result["server_ready_s"] = ...` 之后：
     ```python
     from .stall_watch import StallWatcher, tree_pids  # P3
     if self.cfg.get("stall_watch", True):
         self.stall_watchers = [StallWatcher(s.log_path, lambda s=s: tree_pids([s.proc.pid] if s.proc and s.proc.poll() is None else []),
                                             self.out / "raw" / f"stalls_{s.name}.jsonl", stack_dir=self.out / "stacks") for s in self.servers]
         for w in self.stall_watchers:
             w.disarm()
             w.start()
     ```
   - `run_phase` 里，`closed_loop`、`open_loop`、`agent`、`replay`、`plugin` 这几种负载段开始前 `for w in self.stall_watchers: w.arm()`，它的 `finally` 里 `w.disarm()`；外层 `finally` 里 `cpu.stop()` 旁边 `for w in self.stall_watchers: w.stop()`。
   - 注意：如果 runner 重启死掉的服务时换了 `Server` 对象，lambda 拿到的是旧对象，快照里的进程列表会是空的；不影响停顿的检测和时长。
2. 提交本节列出的文件，集群上 `~/workspace/omni-ppv` 拉到这个提交（两个 sglang-omni worktree 已经建好，内容与补丁文件一致）。
3. 先放 `plan_ppv_p1_prerun.json`；分析后把 X 写进 `make_plan_ppv_p1.py` 的 `P1_X`，生成并放 `plan_ppv_p1.json`。

## 5. 文件

| 文件 | 作用 |
|---|---|
| `configs/runtime/patches/sglang_omni_batch_ppv_p1.patch`、`check_sglang_omni_ppv_maxwait.py` | P1 补丁和 CPU 检查 |
| `configs/runtime/patches/sglang_omni_async_ppv_p2.patch`、`check_sglang_omni_ppv_verify.py` | P2 补丁和 CPU 检查 |
| `configs/models/runtime_unicorn_ppv_maxwait.json`、`runtime_unicorn_ppv_verify.json` | 运行时 |
| `configs/models/minicpm_o_4_5_ppv_p1_t2_k2_c4_packed.yaml` | 四组变体的打包 flow 放置 |
| `scripts/slurm/make_plan_ppv_p1.py`、`make_plan_ppv_p2.py`、`make_plan_ppv_p3.py` 及 `tests/test_make_plan_ppv_p{1,2,3}.py` | 计划生成器 |
| `configs/experiments/plan_ppv_p1_prerun.json`、`plan_ppv_p2_verify.json`、`plan_ppv_p2_perf.json`、`plan_ppv_p3.json`、`configs/experiments/ppv/ppv_p1_*`、`ppv_p2_*`、`ppv_p3_*` | 生成的计划和配置 |
| `scripts/analysis/ppv_p1_report.py`、`ppv_p2_report.py` 及测试 | 分析和门槛 |
| `src/omni_research/orchestration/stall_watch.py`、`tests/test_stall_watch.py` | 停顿检测器 |
