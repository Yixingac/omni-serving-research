# 证据审计（中期）：R0、R1、R2、C1、C2、P0、R3 筛选

审计对象：本分支的 Mac 工作区（分支 research/pre-presentation-validation，HEAD fa3a1ee），只读。原始数据核对用 `~/research-staging/unicorn/experiments/` 镜像。图在草稿区重新生成，没有写仓库。
缩写：PV = `research/next_steps/presentation_validation/`；FER = PV/final_experiment_results.md；PE = PV/presentation_evidence.md；PN = PV/presentation_narrative.md；LIM = PV/limitations_and_negative_results.md；RM = PV/README.md；PR = PV/preregistration.md；RI = `research/next_steps/robotics_application_intro.md`；FIG = `scripts/analysis/ppv_figures.py`。

## 总结

- 8 项检查，共 28 个 FAIL（高 4、中 9、低 15），另有 18 个 NOTE。第 1 项核对了 65 组数字（单个数值超过 300 个），其中 15 组不符。
- 第 2 项（重新生成图）和第 7 项（公开仓库卫生）通过。P0 的核心数字全部正确：+30.4/+29.8/+31.9 %，t 区间 +28.0..+33.4 %，H0c −3.7 %。R1 TTFA 1.36 s 和 R2 单元错过率 45.6 % 也都正确。
- 最重要的 5 条更正：
  1. **A-01** C1 的样本量写错了。实际每个方案只回放了 112 批：两个主组各 32 批，分别取自 95 批和 98 批；4 个次组各 12 批。文中写的是“回放 47,484 个真实批”。−24 %/−42 % 这两个数各自只来自 32 批。涉及 FER:97、FIG:385（fig07）、RI:369。
  2. **A-02** R2 的双卡冒烟写成“通过”，但没有说这个冒烟里 240 个单元全部超时（单元延迟 P50 21.5 s、P99 40 s）。R2b 的登记理由和报告里的问答都用到了这句话。涉及 FER:191、PN:80、RM:14、RI:447、PR:237。
  3. **A-03** 重叠标注不全。C1 和 R1 全部 3 个区组都重叠（C1 从 23:40 跑到 00:45），也和 R1 的 3 个参照运行、R2 冒烟、R2 区组 1 重叠；ASR 评估还和 R2 区组 1–2 重叠。文档只写了“R1 区组 1”，而“区组 1 对区组 2、3”的 D1 比较并不能检验 C1 重叠的影响。
  4. **A-04/A-05** 事先登记的时间有问题。R1/R2 分析口径的提交（fbe3e73）在 00:10:25，这时 R1 区组 1、2 已经跑完；文中写的“23:51”无法用 git 验证。PN:28/30 写“比较都在数据存在之前登记”“提交时间早于运行”，这不成立。D1 标的记录时间是 23:50，比它所在的提交（23:45:27）还晚。
  5. **A-08/A-07** R2 的 OOM 数字含义不对。“25/49/74 次 CUDA OOM 事件”其实是日志里匹配到 OOM 的行数：区组 1 的 25 行全是分配器告警，没有异常，也没有失败会话。“5 个超时单元”实际是 3 个。

## 1. 数字重算（JSON 和原始镜像）

| # | 文档里的数字 | 重算值 | 结果 |
|---|---|---|---|
| 1–7 | R0：单元延迟 75/397/479/879 和 75/413/517/695 ms；TTFA 1.46/2.10/2.51 和 1.66/2.44/2.57 s；回答 12/2 和 19/9；说/听/只出字 111/187/2 和 235/360/5；截止未满足 100/71/29 % 和 100/86/43 %；卡顿 13/4/0 和 20/10/1；迟到包 28/3750（1.19 s）和 43/7500（0.75 s）；功耗 143/166 W；0 % 读数 87/71 % | 74.8/397.5/479.0/878.8 等，全部一致 | PASS |
| 8 | R0 显存 “32.7 GB；33.9 GB” | 32,683 / 33,921 MiB = 31.9 / 33.1 GiB | **FAIL A-17** |
| 9–10 | R1 TTFA P50 1,361 ms（区组 1,373/1,169/1,373；t 区间 1,013–1,597），P95 1,708 ms（N 37） | 1361.1；1373.0/1169.4/1372.6；1013.2–1596.7；1708.2 | PASS |
| 11 | R1 (a) 等待 319 ms、(c) 服务 350 ms；PE/RI 写“mean 0.32 s” | P50 分别 319/350；均值 313/350 | P50 PASS；“均值” **FAIL A-14** |
| 12–13 | k=0 有 17 回合、k=1 有 20 回合（54 %）；TTFA 在 k=0/1 时 862/1,472 ms | 17/20；861.9/1472.0 | PASS |
| 14 | “1 unit ≈ 0.96 s” | 0.96 s 是 37 个 b 值（含 17 个 0）的中位数；k=1 的 b 中位数是 1,040 ms，均值 1,016 ms | **FAIL A-15** |
| 15–18 | 单元延迟 72/429 ms（N 900）；首音频服务时间 353 ms（N 41）；卡顿 36/16/0 of 41，所需缓冲 65/165 ms；打断后继续 5.06 s（P90 11.3、max 11.6） | 一致 | PASS |
| 19 | 阶段份额：thinker/talker/speech/perception/交接/其余 = 41/24/24/5/3/2 %；142/88/82/17 ms；单元时间 351 ms；381 个单元 | 40.9/24.2/24.4/5.0/3.1/2.5 % | PASS |
| 20 | 事件开销 +4.9 %；D1 差别 +1.7/+0.5/+8 % | +4.92；+1.67/+0.54/+8.02 % | 数字 PASS（含义见 A-03） |
| 21 | 参照：724 ms（726/709/735；691–756），P95 1,112、P99 1,799 ms；4.1 s 音频、52 字 | 一致 | PASS |
| 22 | N6：1 秒内开口 R1 25 %、R0 29 %、R2 50 % | 25.0/28.6/50.0 % | PASS |
| 23–27 | R2：错过率 0.56/0/0.14/45.6 %；说话单元 353/608、395/744、464/902、707/1,480 ms；只听单元 74/74/77/98 ms；TTFA 935(18)/1,737(22)/1,557(44)/1,755(74)；1 秒截止未满足 50/88/92/94 %；k 计数 12/3/3、6/16/0、17/25/2、34/39/1 | 一致 | PASS（45.6 % 的组成见 N-01） |
| 28–29 | 阶段增量：thinker +89.8、talker +88.7、speech +88.2 ms；talker 领先 7 ms；首个回答 2.8/4.5/2.8 s，之后约 0.9 s | 一致 | PASS |
| 30 | “all five missed units” / “5 个超时单元” | 3 个（2,496/2,176/1,460 ms，原始数据核对） | **FAIL A-07** |
| 31 | 1.34 s 卡顿、16 个迟到包、进程处于 D 状态 | 1,338 ms、16 个、disk-sleep | PASS |
| 31b | “I/O pressure (full, avg10) … 7.15” | full 7.01；7.15 是 some | **FAIL A-18** |
| 32 | 语音进程 13.6 GiB，对份额 7.1 GiB | 服务日志：pid 1072830（speech）13.60 GiB；0.15 × 47.40 = 7.11 GiB | PASS（来源见 N-03） |
| 33 | “25 和 49 次 CUDA OOM 事件”、“74” | 匹配 OOM 的日志行数。区组 1：22 行 mapping 失败 + 3 行 allocation 失败，0 个 RuntimeError；区组 3：49 行，其中 14 行 RuntimeError | **FAIL A-08** |
| 34–37 | P0 容量 17.35/17.37/17.39（17.37）、22.63/22.55/22.94（22.71）、23.05/23.57/24.11（23.58）；W3L 7.24/7.19/7.34、9.50/9.48/9.35；H0a +30.4/+29.8/+31.9（+30.7，+28.0..+33.4）；W3L +31.1/+31.7/+27.3（+30.1）；H0c −1.8/−4.3/−4.9（−3.7，−7.7..+0.3） | 一致，t 区间重算（df 2） | PASS |
| 38–41 | 开环稳定速率；26 req/s 时 A 20.5/17 s/5.4、B 25.0/5.2/25.0、R 25.2/4.0/25.2；31 req/s 时 B 的 P95 8.1–8.3 s、goodput 16–19，R 4.9 s/29.5；A 排队 180–183（8.3–8.7 s）、B 127–129（4.2–4.4 s）、R 15–16 个在跑/0 排队；驻留 73–74 %、60–62 %；GPU 忙 53–58 %、86–89 % | 一致 | PASS |
| 42 | 解码步长 3.79/6.13/4.09 ms；8.5k → 10.4k token/s（+23 %） | 3 个运行中位数的平均：3.786/6.131/4.088 | 数字 PASS（“pooled”的说法见 A-26） |
| 43 | “token usage ≤ 0.13” | 计入 C 触发条件的 B 高负载行最大 0.16（带事件的 256 段：0.16/0.15/0.15） | **FAIL A-16** |
| 44–45 | C 触发 0 %（3,017 行）；M1 “100–106 %” | 3,017/0.0；A、B 方案 100.7–107.9 %，参照在 W3b 256 时 86.8–93.2 % | 44 PASS；45 **FAIL A-21** |
| 46 | P0 实际 GPU-时 20.0 | 按 step.log 计 19.9 | PASS |
| 47–48 | C1 的 E 表 6 组 × 4 个方案；P1 相对 D −24.1/−41.5/−13.0/−25.0/−27.4/−0.8 %；每遍 −24.1/−24.1/−24.0；D_ng +3.3–3.6 % | 一致 | PASS |
| 49 | “P0 +21..+58 %” | 6 组为 +37.8/−0.7/+58.2/+30.8/+20.8/+2.2 % | **FAIL A-11** |
| 50 | “走 CUDA graph 的请求在批处理组 ≤ 3 %（所有方案）” | D 在 MIX length8 为 9.3 %，MIX fifo16 为 3.2 % | **FAIL A-10** |
| 51 | kernel 族：注意力 1.43→0.59（−59 %），mask 0.60→0，GEMM 1.94→1.69，Inductor 融合 +0.22 | 1.426→0.591，0.603→0.019，1.938→1.688，+0.227 | PASS |
| 52 | MCD 4.06 / 5.07（+1.01）；P0、P2 +0.94/+0.95；P1h +0.06；D 对 D_ng 4.09 dB、rel RMS 1.23；mel 0.36 % 对 0.06 % | 一致 | PASS |
| 53 | “回放 47,484 个真实批” | arm_result.json：主组每组 32 批，次组每组 12 批，即每个方案 112 批；47,484 是 38 个组全部导出的总数 | **FAIL A-01** |
| 54–56 | C2：12 个均值；每遍变化 B、C −5..+7 %，D +31..+50/+31..+33/−2..+23 %；rel RMS 9.1e-4 / MCD 0.18 / 0.83；82 个模块 0 ulp；kernel 数 486/404/486/569；kernel 时间占 77 %/35 % | 一致（W3L 的 B 为 1.4950，写作 1.50，属于舍入边界） | PASS |
| 57 | R3 表：8 个格子 × 11 列（送达 5.9/7.7/7.5/7.6/7.4/8.0/9.4/9.5 req/s，丢帧/迟到、P50/P95、编码器、prefill、份额/槽位、利用率/功耗、分类） | 全部一致 | PASS |
| 58–61 | 上限 7.5–7.7 / 9.4–9.5；编码器 16–22×（4 会话格 3.4×）；prefill 2.0–2.2×；排队 ≤ 7 ms；预处理 2–8 %（最多 21 %）；1.5–1.9 核；并发 4 时丢帧 −3.5…+1.2 个点、P95 −14…+9 %、准确率 −0.02…+0.01，18 格无改善；15 个可疑格、140–224 W；一致性检验 117/89 ms → 8.5/11 req/s，89 %/85 % | 一致 | PASS |
| 62 | “每个 16 会话格 283–292 W” | 并发 1：282.9–291.4 W；并发 4：278.7–290.0 W | **FAIL A-20** |
| 63 | SM 时钟 1.64–1.70 GHz | 原始数据（报告同一窗口）：并发 1 为 1.635–1.695 GHz | PASS（来源见 N-03） |
| 64 | “16 × 0.5 fps nothing is dropped … every answer 1.5 s” | frame 形状丢帧 4.4 %，1.50 s 是 P50（P95 1.86 s） | **FAIL A-09** |
| 65 | PE:45 “71–100 % 的读数为 0，功耗 120–220 W” | 被标记的段为 54–99 %（R0 71–87 %，R2 54–99 %），功耗 118–219 W | **FAIL A-19** |

## 2. 重新生成图

| 检查 | 结果 |
|---|---|
| `ppv_figures.py --out <草稿区>` 能否运行 | PASS：exit 0，生成 fig01–fig10；P1、R4 因缺少输入被跳过，符合预期 |
| 与提交版本比较 | PASS：10 张 PNG 与 figures/ 里的版本逐像素相同（差异像素为 0），文件集合一致 |
| 图注是否写了条件和样本数 | 基本 PASS。例外：fig07 的样本量写错（A-01）；fig09 写“74 OOM”（A-08）；fig02 写“pooled”（A-26）；fig10 写“5 个饱和格都是 GPU 受限”，只对并发 1 成立（N-02）；fig08 没写每组的 mel 段数（N-17） |
| 缺失数据有没有画成 0 | PASS：fig03 W3L 的参照、fig07 的 P2/D_ng/P1h 只画了有数据的格子；fig02 参照的“排队 0”是实测值 |

## 3. 状态

| 检查 | 结果 |
|---|---|
| 运行中或排队中的实验是否写成 COMPLETED | PASS：P1 写 RUNNING，R3 写“筛选完成”，R4 写“冒烟通过”，P2、P3、R2b 都不是 COMPLETED |
| P0 第一次尝试失败、P1 区组 2 失败 | PASS：RM:15、RM:17 有记录 |
| R3 第一次筛选崩溃、撤回的第一版确认计划 | **FAIL A-12**：RM:19 两条都没写，GPU-时写的是“—” |
| 清单文件 | **FAIL A-06**：RM:61、FER:3、FER:205、PN:30、PN:33 都引用了 `experiment_manifest.md/.json`，但仓库里没有提交这个文件（只有 inclusion.json 和 ppv_manifest.py） |

## 4. 门槛

| 检查 | 结果 |
|---|---|
| 有没有任何文档声称 C1/C2 带来端到端或服务层面的收益 | PASS：FER、PN、PE、LIM、RI 都写明“微基准、没有做端到端” |
| 微基准数字是否都标了“微基准/局部 GPU 时间” | **FAIL A-27**：PE:39–41 的实验列只写“C1”；RM:16 的 C1 一行没写“微基准” |
| 与“正确性先行”规则是否矛盾 | NOTE N-10 |

## 5. 测量有效性

| 检查 | 结果 |
|---|---|
| 有没有把 profiler 窗口的统计当成整段运行的数据 | PASS（PE:14、FER 的 C1/C2、RI §7 都注明了窗口） |
| CUDA graph 的 kernel 时间占比有没有被叫做命中率 | PASS（PE:14、PN:58、fig05 都写明“时间占比，不是命中率”） |
| 合成负载有没有被说成真实机器人、摄像头或用户 | **FAIL A-28**：RI:156 把合成客户端画成“Robot client” |
| 全双工数字和请求-响应数字并列时有没有说明语义不同 | **FAIL A-13**：RM:12、PN:12 没有说明（FER R1、PE:37、fig06、RI Finding F 都说明了） |
| 周期负载下有没有用 nvidia-smi 利用率作为空闲的证据（R0–R3） | PASS（被标记的读数都没有用作证据），另见 N-16 |
| 其他有效性问题 | **FAIL A-02**（双卡冒烟）、**FAIL A-08**（OOM 是日志行数）、**FAIL A-22**（假设写成了事实）；N-04（“混叠”写成了已证实） |

## 6. 事先登记的先后（git 提交时间 vs 镜像里的开始时间，均为 −0500）

| 项 | 登记提交 | 第一次运行 | 结果 |
|---|---|---|---|
| P0 计划 + 补充 | ade6a17 16:32；1ab92cd 22:14:41 | 22:30:38（第一次尝试）/ 22:35:36 | PASS |
| E-P0-G | 9857c72 23:45:27 | G 在 23:40:50 失败之后，区组 2（01:10）之前；已标注为 Erratum | PASS |
| R0、R1、R2 登记，C1/C2（含 E-C1），R3/R4 | 2699341 22:58:47 | R0 23:14:38；C1/C2、R1、R3 都是 23:40:52；R2 冒烟 00:27:57 | PASS |
| R1/R2 分析口径（标为 23:51） | fbe3e73 **00:10:25** | 提交时 R1 全双工区组 1（23:40–23:52）、区组 2（23:56–00:08）和参照区组 1 已经跑完，参照区组 2 正在跑 | **FAIL A-04** |
| D1（标为 23:50 记录） | 9857c72 23:45:27 | 标注的时间比提交时间晚 | **FAIL A-05** |
| C1/C2 重叠的标注 | — | C1 的 6 个方案（23:40–00:45）与 R1 全部 6 个运行、R2 冒烟、R2 区组 1 重叠；ASR（00:45–00:52）与 R2 区组 1–2 重叠；R3 的崩溃尝试只和 R1 区组 1 的前约 2.5 分钟重叠 | **FAIL A-03** |
| 事后补充：R0 判定修正、R2 描述（a16ab6e 01:35）、R3 功耗和 0 % 列（03:45）、R2b（04:05） | 都在看过数据之后 | 都已标注“事后、描述性、规则不变”；改动核对过，没有改登记的判定 | PASS |
| R3 选格（03:21:30）、D2（03:26） | 筛选结束（03:19:44）之后、确认之前 | 按规则执行 | PASS |

## 7. 公开仓库卫生

PASS。PV 下所有已提交的文件、`scripts/analysis/ppv_*`、`scripts/slurm/make_plan_ppv_*`、SVG 元数据都没有 /home/ 路径、用户名、完整域名、IP、token 或密码。脚本里只有 `~/workspace/omni-ppv`。NOTE N-13：RM:4、audit.md:14、c1_c2_design.md:117、p1_p2_p3_design.md:110 出现了节点短名“compute-03”和作业号 454290（风险低）。

## 8. 内部一致性

核心数字在各文档之间一致：P0 +30.4/+29.8/+31.9 %，TTFA 1.36 s，45.6 %/46 %，−13..−42 %，0.72 s，5.1 s。不一致的有：A-14（0.32 s 在 FER 里是 P50，在 PE/RI 里写成均值）；A-10（“≤ 3 %” 与 PE:15/PN:57 的“2–7 %”冲突）；A-25（PN:30 写“三个 Erratum”却只列了两个，而且没有 D2、D3）；A-23（RI:248 写“3 区组 × 1/2/4/8”）；A-24（FER:265 写确认结果“见下文”，但下文没有）；N-12（PR:231 写“140–220 W”，其他地方写 140–224 W）；N-11（“慢 30 %”和“+31..+50 %”两种写法并存）。

## FAIL 清单与准确的改法

| ID | 级别 | 位置 | 改法 |
|---|---|---|---|
| A-01 | 高 | FER:97；FIG:385（重新生成 fig07）；RI:369 | FER 改为 “Inputs: a seeded sample of the real batches exported from 实验 9 and 10 (`c1_real_shapes.json`, 47,484 batches in 38 groups): 32 per primary group (W3b fifo16 of 95, MIX fifo16 of 98) and 12 per secondary group, i.e. 112 batches per arm (D_ng, P2, P1h: the 64 primary batches only)”；图注改为 “replaying 112 real batches per arm sampled from the 47,484 exported”；RI 改为 “C1 microbenchmark on 112 sampled real batches per arm” |
| A-02 | 高 | FER:191；PN:80；RM:14；RI:447；PR:237（登记文本不改，只附审计注） | 在 “whose smoke passed” 后加：“; the smoke met the registered start-and-no-failure rule but all 240 of its units were late (unit lag P50 21.5 s, P99 40 s, first audio 7.3 s)”；RM 改为 “冒烟按登记规则通过（能启动、无失败会话），但 240 个单元全部超时（单元延迟 P50 21.5 s）” |
| A-03 | 高 | FER:54、79、97、168；LIM:12；inclusion.json:46、58 及其后的 R1/R2 条目 | FER:54 改为 “all three blocks and the three reference runs overlapped the C1 microbenchmark (C1 arms 23:40–00:45; C2 only during block 1); block 1's first ~2.5 min also overlapped the crashed R3 screening — deviation D1”；FER:79 加 “(blocks 2–3 also overlapped C1, so this tests only the C2/R3 overlap)”；FER:97 改为 “its arms overlapped all R1 runs, the R2 smokes and R2 block 1; the ASR pass overlapped R2 blocks 1–2”；FER:168 加 “C1's last arm and ASR pass ran during blocks 1–2 (D1 item 4)”；LIM:12 和 inclusion.json 按同样的事实改 |
| A-04 | 高 | PR:161；FER:52；PN:28、30 | PR 标题后加 “（提交 fbe3e73 于 11-19 00:10:25，此时 R1 区组 1、2 已跑完；‘23:51、分析之前’无法用 git 验证）”；FER 改为 “were committed at 00:10 (fbe3e73), after R1 duplex blocks 1–2 had run, by the author's account before any analysis”；PN:28 改为 “registered before its experiment's first run, except the R1 analysis details (committed after two of three R1 blocks, before analysis)”；PN:30 去掉 “(commit times precede runs)”，或加上同样的例外 |
| A-05 | 中 | PR:150 | “23:50 记录”改为 “23:45 记录（提交 9857c72 23:45:27）” |
| A-06 | 中 | RM:61；FER:3、205；PN:30、33 | 用 `ppv_manifest.py` 生成并提交 `experiment_manifest.md/.json`；提交之前把这几处引用改为 “inclusion.json（清单尚未提交）” |
| A-07 | 中 | FER:182；LIM:15 | 把 “all five missed units” 改为 “all three missed units (lags 2.50, 2.18, 1.46 s)”；“5 个超时单元”改为“3 个超时单元” |
| A-08 | 中 | FER:177；PE:43；PN:78；FIG:452（重新生成 fig09） | FER 改为 “server-log lines matching CUDA OOM: 25 in block 1 (allocator warnings only, no failed session), 49 in block 3 (incl. 14 RuntimeError); 7 failed sessions (block 3)”；PN 改为 “7 sessions failed with CUDA OOM in one of the two blocks”；图的标注改为 “OOM log lines 25+49” |
| A-09 | 中 | FER:267、271 | 英文改为 “(single PNG) only 4.4 % of ticks are dropped, below the 10 % threshold, yet the median answer arrives 1.50 s after its frame (P95 1.86 s; ≈ 10× the 1-session P50 of 0.14 s)”；中文改为 “单张 PNG、16 个会话 × 0.5 帧/秒只丢 4.4%（低于 10% 的饱和线），但回答中位数晚 1.50 秒（P95 1.86 秒）” |
| A-10 | 中 | FER:110；PE:41 | 改为 “Requests on a CUDA graph in the sampled batched groups: 0–3.2 %, except D on MIX length8 (9.3 %, 12 batches)”；PE 改为 “0–9 % of sampled batched requests (rule-based share over all recorded batches: 2–7 %)” |
| A-11 | 中 | FER:126；RI:369 | 改为 “slower in four of six groups (P0 +21..+58 %), equal on MIX fifo16 (−0.7 %) and the open-loop group (+2.2 %)” |
| A-12 | 中 | RM:19 | 状态改为 “筛选完成（第一次尝试 `ppv_r3_screen_c{1,4}` 启动即崩溃，F3）；第一版确认计划撤回（D2），区组串行版排队中”；GPU-时改为 “0.88（筛选）+ 0.08（失败尝试）” |
| A-13 | 中 | RM:12；PN:12 | RM 在 0.72 s 前加 “语义不同：从整段请求发出计时、无端点检测、回答约一半长，不是速度比较”；PN 加 “(request-response TTFA starts when the whole request is sent and has no end-of-speech detection; not comparable to the duplex number)” |
| A-14 | 低 | PE:34；RI:159；FER:81 | 把 “mean parts: 0.32 s” 改为 “median 0.32 s (mean 0.31 s)”；RI 和 FER 同样改 |
| A-15 | 低 | FER:75 | “(1 unit ≈ 0.96 s)” 改为 “(the extra unit adds a median 1.04 s in the k = 1 turns)” |
| A-16 | 低 | FER:217；PE:29；RI:192 | “≤ 0.13” 改为 “≤ 0.16 (0.13 without request events)” |
| A-17 | 低 | FER:32 | 改为 “31.9 GiB (32,683 MiB) ｜ 33.1 GiB (33,921 MiB)” |
| A-18 | 低 | FER:183 | 改为 “rose from 0.18 to 7.01 (full) / 7.15 (some)” |
| A-19 | 低 | PE:45 | 改为 “flagged phases read 0 % in 54–99 % of samples (R0 71–87 %, R2 54–99 %) at 118–219 W” |
| A-20 | 低 | FER:260；PE:46；LIM:31 | 改为 “283–291 W at concurrency 1 (279–290 W at concurrency 4)” |
| A-21 | 低 | LIM:26 | 改为 “方案 A、B 的 talker CPU 在所有负载下为 100–108%（2+2+4 参照在 W3b 256 时 87–93%）” |
| A-22 | 低 | FER:195 | 改为 “客户端卡住 1.34 秒（区组 1；假设：写 NFS 的记录阻塞了事件循环）——这是 P3 要找的那种停顿的一个候选实例” |
| A-23 | 低 | RI:248 | 改为 “R2 (3 blocks; 2/4/8 sessions in 2 of 3 blocks)” |
| A-24 | 低 | FER:265 | 改为 “The 3-block confirmation is queued and will be reported here when it has run.” |
| A-25 | 低 | PN:30 | 改为 “three errata (E-C1 before C1; E-P0-G after the failed G start, before blocks 2–3; E-P1-X before the P1 main runs) and three deviations (D1–D3)” |
| A-26 | 低 | FIG:171（重新生成 fig02）；FER:219 | 改为 “mean over each arm's runs of the per-run median step time (≥ 20 log lines per size and run)”；FER 把 “pooled” 改为 “mean of the 3 runs' medians” |
| A-27 | 低 | PE:39–41；RM:16 | 实验列改为 “C1 (microbenchmark)”；RM 改为 “C1 flow 稠密 vs 打包（微基准，局部 GPU 时间）” |
| A-28 | 低 | RI:156 | 改为 “participant U as Client (synthetic TTS speech standing in for a robot, 80 ms packets)” |

## NOTE

- **N-01**：R2 8 个会话的 45.6 % 是两个区组合并的结果。区组 1 为 3.5 %，没有失败会话；区组 3 为 87.7 %，7/8 的会话失败。建议分区组报告。
- **N-02**：“5 个饱和格都是 GPU 受限”只在并发 1 时成立。并发 4 时 frame_s16_f1（GPU 忙 75 %）和 clip_s16_f2（74 %）都归为“饱和、原因不明”（FER:265、PE:46、fig10）。
- **N-03**：13.6 GiB 只出现在服务日志里，SM 时钟只在原始 gpu_samples 里，而 PE:43/46 引用的是 r2_summary.json 和 r3_screen.json。两者都已用镜像核实。
- **N-04**：“混叠”在 FER:38、FER:260、RI:449 写成了已证实，代码注释里写的是假设。另外 R2 区组 2 在同样负载下读到 45 %。
- **N-05**：RI §11 已经过时（RI:458、RI:462 写着“全双工还没跑过”“改成 64 还只是假设”）。
- **N-06**：RI:189 写 talker 名额“hard-coded to 32”，但 P0 只改了配置就把它改成了 64。
- **N-07**：PR:265 写“每次转储约 0.5 % 崩溃概率”，这只来自 1 次事件，而 P3 还在跑。建议写成“约 200 次里 1 次”。
- **N-08**：LIM:44 的空行把 N7、N8 从表格里断开了。
- **N-09**：PE:16 和 PE:20 引用了 “P1 below”“P2 below”，但表里没有这两行。
- **N-10**：PE 表头写“没通过的实验不给数字”，FER 的 C1 写“不下性能结论”，但 PE:38 和 PN:66 仍然给出了 P1 −13..−42 % 的性能说法（虽然标了微基准）。
- **N-11**：C2 FP16 写“+31..+50 % 更慢”，漏掉了 W3L 组（−2..+23 %）。三个组的均值分别是 +40/+32/+14 %。
- **N-12**：LIM:31 写“整段读 0%”，但其中 1 格的 0 读数只占 77 %；PR:231 写 140–220 W，其他地方写 140–224 W。
- **N-13**：节点短名和作业号，见第 7 节。
- **N-14**：F1 写失败尝试“约 0.15 GPU-时”，按镜像每个运行不超过 17 s，约 0.04 GPU-时，无法核实分配时长。
- **N-15**：RI Finding D 写请求-响应路径“低负载约 2.0 s”，但没提 R1 参照在一句话回答时只要 0.72 s，TTFA 取决于回答长度。
- **N-16**：R2 “第一个先满的阶段”的资源判定用的 GPU 利用率被标为可疑。
- **N-17**：fig08 的图注没写样本数。
- **N-18**：R0 “0.75 % 迟到包”只是 1 个会话那段的数字，整体是 0.63 %。

范围说明：P1、R2b、R3 确认、R4、P2、P3 的结果都没有审计。把它们写得好像已经有结果的地方只有 A-24、N-09、N-07。

## 处理结果（作者，2026-11-19 05:55，提交 `c21e3be`）

所有 FAIL 都先用数据或 git 记录核实过，再改：

- **已改**：A-01 至 A-28 全部；N-01（R2 8 个会话按区组报告）、N-02、N-04（“混叠”改为假设）、N-06、N-07（登记文件里加更正行，原文保留）、N-08、N-10（证据表的规则改为“未通过门槛的数字只和失败的门槛一起出现”）、N-11、N-12（同上，加更正行）、N-14（F1 改为 0.07 GPU-时，按编排器记录）、N-15、N-16（写进 R2 的局限）、N-17、N-18。
- **A-02 的后果**：在任何 R2b 运行之前，给 R2b 补了勘误 E-R2b（判定规则不变，写明冒烟里单元全部超时、双卡放置是否实时事先未知、所用卡对由编排器分配）。我自己从原始数据重算的冒烟单元延迟是 P50 约 22 s、P99 约 40 s（审计写 21.5 s，差别来自分位数算法）。
- **A-04、A-05**：登记文件的原文不改，在对应标题下加了更正说明（提交时间、当时已经跑完的区组）。
- **A-06**：用 `ppv_manifest.py` 生成并提交了中间版的 `experiment_manifest.md/.json`；失败信息里的家目录、节点名和地址先清洗掉。所有实验结束后会再生成一次。
- **留到最后**：N-05（机器人介绍第 11 节要按最终结果重写）、N-09（证据表里的 P1、P2 行等结果出来再加）。
- **不改**：N-13（节点短名和作业号）。仓库其他文档已经在用，不含用户名或完整域名。
