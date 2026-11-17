# 实验 6–8 用到的 Agent Skills

> 记录每个 Skill 的来源、版本、能否使用，以及具体用在了哪一步。这些 Skill 都**没有安装**到本机的 Agent 环境：
> 本机已安装的插件（mattpocock-skills、codex、clangd-lsp、humanizer）里没有它们。
> 所以按任务要求，从官方仓库**只读获取** `SKILL.md` 和需要的 references，逐个读完后照着做；没有运行这些仓库里的任何安装脚本或附带脚本。

## 来源和版本

| Skill | 来源仓库 | 读取时的 commit | 读了哪些文件 |
|---|---|---|---|
| experimental-design | K-Dense-AI/scientific-agent-skills | ec59a29a03d5（2026-11-16） | `skills/experimental-design/SKILL.md`（v1.5）、`references/randomization_and_blocking.md`、`references/design_types.md` |
| optimize-for-gpu | 同上 | 同上 | `skills/optimize-for-gpu/SKILL.md`（v1.6）。`references/decision_framework.md` 讲的是 RAPIDS 各库怎么选，和 PyTorch 推理服务无关，只浏览 |
| statistical-analysis | 同上 | 同上 | `skills/statistical-analysis/SKILL.md`（v2.0）、`references/effect_sizes_and_power.md`、`references/reporting_standards.md` |
| scientific-visualization | 同上 | 同上 | `skills/scientific-visualization/SKILL.md`（v1.4）、`references/publication_guidelines.md` |
| systematic-debugging | openai/plugins（`plugins/superpowers/skills/`） | d420d13051bf（2026-11-06） | `systematic-debugging/SKILL.md`、`root-cause-tracing.md` |
| test-driven-development | 同上 | 同上 | `test-driven-development/SKILL.md`、`writing-good-tests.md` |
| verification-before-completion | 同上 | 同上 | `verification-before-completion/SKILL.md` |

几个 Skill 的说明里提到可以用 pingouin、pydoe、seaborn 等包，也有附带的脚本。这些都没有装，也没有运行：
- 随机化用 Python 标准库 `random`，种子写进配置；
- 置信区间用 Student t 分布的 97.5% 分位数表（自由度 1–10 的标准值写在分析脚本里）。项目环境里没有 SciPy，也没有为此安装；
- 画图用项目已有的 matplotlib。

## 用在了哪一步

| Skill | 具体用法 | 位置 |
|---|---|---|
| experimental-design | 定实验单位：一次独立启动的服务算一个，请求不算，避免伪重复。用随机完全区组：每个区组把所有方案各跑一次，区组内顺序随机、种子固定。A、B、C 三种测量方式的顺序按区组轮换；实验 8 第三阶段“关”“开”按 ABBA 交替。设了阴性对照（H8d：语音带不带种子应无差别）和阳性对照（“开但不做快照”应出错）。看数据之前写下假设和反证标准；每次补充登记都写明时间，以及当时已经看到哪些数据 | 各实验的 `hypotheses.md`、`scripts/slurm/make_plan_exp06_08.py` |
| optimize-for-gpu | 改动之前先测基线。GPU 时间用 CUDA event 测，在原本就有的同步之后读取，不加额外的 synchronize。kernel 时间和端到端时间分开报告。原型先要求输出逐位正确，再看性能；同时记录额外显存（每步拷贝 批大小 × 4096 × 2 字节） | 实验 6 的打点、实验 8 的原型 |
| statistical-analysis | 以运行为单位汇总，用区组内配对差，报告原始值、均值、t 分布 95% 置信区间和效应大小。只有 3 次重复时不声称显著。分清预先登记的分析和探索性分析（实验 7 的 15、18 请求/秒）。对看过数据后才定的做法做敏感性检查：实验 8 改用“同一并发档直接比较”，结论不变 | 各实验的分析脚本和 README |
| scientific-visualization | 图上画出每个区组的原始点，写明区间的含义和 n；颜色加形状区分方案；对数坐标标明；缺失数据显示为“无数据”；每张图都能从 `results/` 重新生成。看图时发现并改正了实验 6 图 4 写错的标题 | 各实验的作图脚本 |
| systematic-debugging | 遇到异常先找证据、沿链条追到源头，见下面的“排查记录” | 见下 |
| test-driven-development | 严格先写测试（先看到失败，再写实现）的地方，以及没有做到的地方，见下面的“测试先行记录” | 见下 |
| verification-before-completion | 每次说“测试通过”“实验完成”之前，都重新运行命令并读输出；区分“代码写好”“作业已提交”“实验真的完成”。写报告时把凭印象写的数逐条对回数据，改正了几处错误，见下面的“核对记录” | 每次提交前的 `pytest`，以及各 README |

### 排查记录（systematic-debugging）

- **编排器崩溃**：读完整的 traceback，沿调用链往回追（`squeue` 超时 → `slurm_time_left_s` → `time_left` → `step`）。根因是超时异常没有被接住，只在源头改一处（提交 827ec92）。
- **实验 6“逐个发 kernel 慢 50%”**：
  - 复核时发现两组请求长度不同：走 graph 的都 ≤ 1024 帧，逐个发的都 > 1024 帧；
  - 按每帧折算后只慢 4–16%，并查到高负载时 kernel 在 GPU 上排队 12–94 毫秒；
  - 据此把 H2 从“支持”改为“高负载下不支持”。
- **语音路径上 CPU 在等什么**：
  - 时间线上看到一种和纯文字不同的阻塞；
  - 按 CUDA 调用名统计，发现是 `cudaMemcpyAsync`；
  - 按拷贝大小统计，是 8–32 字节的 token id，不是隐藏状态。
  - 由此排除了“原型会被每步拷贝隐藏状态挡住”的担心。
- **code2wav 进程在 profiler 窗口里崩溃**：
  - 4 次都在开 profiler 的段，都是 code2wav，都发生在同段其他进程写完 trace 之后 2–12 秒，崩溃进程自己的 trace 都缺失；
  - 根因没有找到，标为“未知”；第三阶段改为只给 thinker 开 profiler。
- **正确性检查里“关”自己就不一致**：
  - 逐个请求对照，发现不一致的样式在三组里完全相同；
  - 所以它是由请求历史决定的，不是随机误差，于是改为按请求序号逐个比较，同时保留事先登记的按问题分组比较。

### 测试先行记录（test-driven-development）

**严格先写测试**：先运行测试，看到它失败，再写实现。
- 编排器的 squeue 超时（827ec92）；
- 原型补丁 v2：先扩充检查，在 v1 补丁上失败（缺少 speech_lookahead_mode），改代码后在 v2 上通过；
- `exp08_report.py`：写实现之前，在测试里把“补测缺数据时退回去用主区组”明确判为错误；
- `exp08_proto_check.py` 的按请求比较（compare_by_index）；
- `exp07_report.py`：写实现之前，发现一条测试预期和登记的反驳标准不一致，先改正了测试；
- `exp08_proto_perf.py`。

**没有做到**：
- 实验 6–8 的测量补丁是先写代码、后补检查。补救办法是：在没打补丁的副本上运行，确认检查失败；在打补丁的副本上运行，确认通过。
- 实验 6 分析脚本的大部分测试和代码是一起写的，只有两处修正（kernel 排队延迟只算每次发 kernel 的第一个 kernel；用打点的 CPU 区间做归属）是先改测试。
- `exp08_lookahead.py` 和 `exp08_proto_check.py` 最初的测试也是和代码一起写的。

### 核对记录（verification-before-completion）

写报告时把以下内容逐条对回数据或源码：

| 写错的内容 | 改正后 |
|---|---|
| 实验 8 第一阶段的运行时间 | 15:40–16:16、补测 16:19–16:36 |
| 实验 6 的运行时间 | 14:58–15:37 |
| 实验 8 那次崩溃的报错数（23 是日志行数） | 22 |
| 源码行号 | 采样参数检查在第 527–539 行 |
| 六处登记时间（都写快了 1–17 分钟：实验 7、8 最初登记的 14:35、14:40 都晚于提交时间 14:33；实验 8 的 14:58 写成 15:15、15:32 写成 15:33、16:58 写成 17:05；实验 7 的 17:07 写成 17:08） | 按实际时间（以 git 提交时间核对）改正，并在 limitations 里说明 |

另外核实过：thinker 的隐藏维度 4096（模型配置）。

**一次测试失败没有复现**：提交前的一次完整测试有 1 个失败，没记下是哪一个；随后连续 3 次完整运行都全部通过。
