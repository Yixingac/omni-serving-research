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
- 置信区间用 SciPy 的 t 分布，项目环境里已有；
- 画图用项目已有的 matplotlib。

## 用在了哪一步

| Skill | 具体用法 | 位置 |
|---|---|---|
| experimental-design | 定实验单位：一次独立启动的服务算一个，请求不算，避免伪重复。用随机完全区组：每个区组把所有方案各跑一次，区组内顺序随机、种子固定。在 A、B、C 三种测量方式之间轮换顺序，平衡先后影响。在看数据之前写下假设和反证标准 | 各实验的 `hypotheses.md`、`scripts/slurm/make_plan_exp06_08.py` |
| optimize-for-gpu | 改动之前先测基线；GPU 时间用 CUDA event 测，不用 CPU 计时器包住异步调用；kernel 时间和端到端时间分开报告；记录显存；优化是否保留，以正确性和用户关心的指标为准 | 实验 6 的打点、实验 8 的原型判定 |
| statistical-analysis | 以运行为单位汇总；用区组内配对差；报告原始值、均值、t 分布 95% 置信区间和效应大小；只有 3 次重复时不声称显著；分清预先登记的分析和探索性分析 | 各实验的分析脚本和 README |
| scientific-visualization | 图上画出每次运行的原始点，写明区间的含义和 n；用颜色加形状区分方案；柱状图从 0 开始；对数坐标要标明；缺失数据要显示出来；每张图都能从保存的结果文件重新生成 | 各实验的作图脚本 |
| systematic-debugging | 编排器崩溃：先读完整的 traceback，沿调用链往回追（`squeue` 超时 → `slurm_time_left_s` → `time_left` → `step`），找到根因（超时异常没有被接住）后，只在源头改一处 | 提交 827ec92 |
| test-driven-development | 编排器修复：先写一个会失败的测试，看着它因为 `TimeoutExpired` 而失败，再写最小改动让它通过。新增的打点解析和指标计算也先写测试 | `tests/test_orchestrator.py`，以及新增的分析测试 |
| verification-before-completion | 每次声称“测试通过”“实验完成”之前，都重新运行命令并读输出；区分“代码写好”“作业已提交”“实验真的完成” | 每次提交前的 `pytest`，以及最终汇报 |
