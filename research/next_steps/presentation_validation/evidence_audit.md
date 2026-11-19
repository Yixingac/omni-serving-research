# 最终证据审计：final_experiment_results.md 等文件 vs results/*.json（2026-11-19）

方法：所有数字都用 python3 从 `results/p1_summary.json`、`r3_confirm.json`、`r4_working_set.json`、`p2_verify.json`、`r2b_summary.json` 重新算了一遍。R2b 的基线另外查了 `r2_summary.json`，P2 字段的含义查了 `scripts/analysis/ppv_p2_report.py`。没有修改任何仓库文件。
结果：3 个 FAIL，11 个 WARN。**没有任何文字改变了登记判定**，所有登记判定都和 JSON 一致（P1 H1a/H1b/H1c 不确定；R3 5 格不支持；R4 combined 稳定且与 v1 相同；P2 通过；R2b 反驳）。

## FAIL

**F1 P1“低于饱和时 5 种设置没有差别”被 W3L 数据否定（同一句话出现在 5 处）**
- 出现位置：final_experiment_results.md ## P1 Evidence “At 0.9X and 1.0X all five settings are within a few percent on goodput”；同节 Interpretation “0.9X and 1.0X look the same for all five settings”；## Synthesis RQ1 “below saturation all five P1 settings give the same goodput”；presentation_evidence.md 行 “Below saturation, the code2wav batching policy makes no difference to the full service”；README §1 P1 行 “0.9X、1.0X 时 5 种设置的 goodput 相差在几个百分点以内”。
- 数据：W3L 1.0X 的 goodput，FIFO16 为 12.50/14.03/14.40，Length8 为 14.74/15.03/14.45 req/s：区组 1 低 15 %，三区组均值 13.64 对 14.45–14.80，差 8.5 %。FIFO16 的 W3L 1.0X 在区组 2_a2 和 3 都是 steady=false，这和文中自己写的 stable capacity “W3L FIFO16 14.5 / 13 / 13”一致，也就是说 1.0X 对 FIFO16 已经不算低于饱和。区组 1 里 FIFO16 虽然稳定，仍然更差：W3L 1.0X e2e P95 16.6 s 对 10.8 s；W3L 0.9X goodput 12.92 对 13.78（低 6 %），e2e P95 15.7 s 对 10.0 s。
- 这句话成立的部分：MIX（1.0X 均值 14.57–14.77）✓；W3b 各速率均值差 ≤ 2.8 % ✓；W3L 0.9X 均值差 2.2 % ✓。
- 建议改为：MIX、W3b 在 0.9X/1.0X 各设置相差 ≤ 3 %；W3L 的 1.0X 已经到了 FIFO16 的容量（区组 2_a2、3 不稳定），FIFO16 比 Length8 低 8 %（区组 1 低 15 %）。

**F2 R2b “no first audio arrived”**
- 出现位置：## R2b Evidence “…lag P50 24.3 / 26.2 / 26.2 s), no first audio arrived”。
- 数据（arms.ms8_2gpu.levels[0].pooled）：replies_with_audio = 12（每个会话 1 个），first_audio_latency_ms n=12、P50 2.88 s，audio_s_total = 1116 s。但 84 个回合里 answered = 0、no_response = 84，TTFA n = 0。
- 建议改为：“no user turn was answered (0 of 84; TTFA undefined); each session produced one reply with audio, all of it late”。

**F3 presentation_narrative.md Slide 9 里的 R4 数字只取了区组 1（旧值）**
- 出现位置：Slide 9 Supporting evidence “routing loses 23–33 % of prefill reuse and slows the answer after a plan 1.7–3.4×”。
- 数据（3 个区组）：routing 23.3–33.8 %（其它文件都写 23–34 %），计划→回答 1.70–4.12×（区组 2、3 的 16 会话分别是 2.80、4.12）。只看区组 1 才是 23.3–32.9 %、1.72–3.36×。建议改为 “23–34 %” 和 “1.7–4.1×”。

## WARN

**W1 R4：媒体历史对照档的驱逐损失没有提到**
- 出现位置：## R4 Eviction “≤ 3.0 % of prefill tokens at every level”，Routing “23–34 % of all prefill tokens at every level”，Result “nothing in these runs calls for it”；evidence 中 R4 行 “≤ 3.0 % … at every level up to 128 sessions”。
- 数据：r4_n32_media2 档，驱逐造成的损失占 prefill 的 13.9/12.3/12.2 %，超过登记的 10 % 份额线；routing 34.2/36.0/36.7 %。clean pairs = 0，所以 TTFT 条件无法评估；同副本的 157/141/134 对全部记为“中间有驱逐”。这一档的参照用的是文字历史的 run 级参照（source=run），驱逐份额可能偏高。
- 登记判定不受影响（这一档不在登记档位里），但 “nothing in these runs” 说得过满，“at every level” 应改为 “at every registered level”，并补一句这一档的读数。

**W2 R4：v1 代码判定的描述不准确**
- 原文：“at 64 and 128 sessions for the wrong reason (zero loss from an undefined reference)”。
- 数据：6 个（档位×区组）格子里有 4 个确实是 0 损失。但 128 会话区组 1（routing 18.8 %，1 个 clean pair，比值 42.6）和 64 会话区组 3（routing 18.2 %，2 个 clean pair，比值 10.3）都不是 0，v1 在这两格用 1–2 个 clean pair 判了 routing_cost=true。最终结论确实相同 ✓。

**W3 R4 表 32 会话的驱逐损失写成 “0.1 / 0.1 / 0.1 %”**：数据是 0.14 / 0.05 / 0.05 %（低影响）。

**W4 P1：“Which cells were overloaded … in blocks 1 and 3 under FIFO16” 不完整**
- W3L FIFO16 在 1.1X 三个区组都不稳定（包括 2_a2），在 1.0X 的区组 2_a2、3 也不稳定。
- MIX 和 W3L 在 1.1X 的 Length8 区组 3 也是 steady=false。W3L Length8 在 1.1X 的 goodput 只占实际到达量（16.9/15.2/17.2 req/s）的 82/98/83 %，所以 “keeps goodput near the offered load” 应加限定。这一处同时影响 Result、Synthesis RQ1 和 evidence 中 P1 行。

**W5 P1：“short requests waited longer, not shorter” 与文中自己的数字不完全相符**：MIX 区组 1 是 5.06 < 5.20 s（×0.97），W3L 区组 2 是 5.57 < 5.78 s。准确说法是 6 个格子里 4 个变长。

**W6 P2：“885 finished speech requests verified, each with exactly one verify event”**
- 出现位置：## P2 Evidence/Result、Synthesis RQ2、README P2 行、evidence 中 P2 行。
- 数据：885 = requests_verified，按脚本是 verify 事件的总数。门槛第 2 条覆盖的是非预热、正常完成的语音请求，共 662 个（per_phase.speech_ok 之和），662 个都恰好有 1 个事件。另外 223 个事件来自这个集合之外的请求（脚本按 warmup 过滤掉的）。885 个事件全部逐位相等，没有重复。
- 另外 chk_timeout 段的事件数为 0，中止段本身没有任何请求被验证。
- 建议改为：“885 verify events, all bit-exact; each of the 662 measured speech requests had exactly one”。

**W7 R2b 的单卡基线 “the same 4 sessions missed ≤ 0.6 % of units” / “broke 4 sessions that worked on one GPU”**
- 出现位置：## R2b、README R2b 行、evidence 中 R2b 行、中文小结。
- 数据：R2 ms8_1gpu 第 4 档只跑了区组 1、3，错过率 0.0 / 0.28 %（合并 0.14 %），没有会话失败，但 meets_serving_target=false。0.6 % 实际是 R2 的 1 会话档（合并 0.56 %）。“worked” 是描述性说法，不是登记结论。

**W8 R2b “with 42.6 GB used”**：mem_max_mib = 42,556 MiB，即 41.6 GiB 或 44.6 GB。建议直接写 “42,556 MiB”。

**W9 R3 把推导结论标成了“实测/登记”**
- 出现位置：## Synthesis RQ3 “Main result (Measured) … Camera input is limited by GPU work per frame … never by CPU preprocessing”；evidence 中 R3 行 “saturate the replica's GPU” 以及 “at concurrency 1 all 5 saturated cells GPU-bound by the registered rule”（这是筛选阶段的说法）。
- 数据：确认阶段两个 clip 格在 3/3 区组的登记类别都是 saturated_other（原因不明），只有 3 个 frame 格是 GPU-bound。“不是预处理”有数据支持（没有一格是预处理受限，等待 P95 ≤ 0.77 ms）；但 “clip 受 GPU 限制” 是 Derived（来自 consistency check），应改标。
- 另外 ## R3 Interpretation 里的 “≈ 9.5 clips/s” 是筛选值，确认阶段是 9.2–11.4。

**W10 evidence 行标题 “makes its first token 2–4× slower”**：数据是 1.70–4.12×（32 会话时 1.70–1.74），应与正文一致写 “1.7–4.1×”。

**W11 P1 测量窗口（低影响）**
- 原文 “75 s (W3L 90 s) measured” 和 Limitations 里的 “these 75–90 s”。
- 数据：实际发送跨度 MIX 68.3–79.9 s，W3b 72.5–77.8 s，W3L 83.4–94.8 s，completed_rps 用的是实际跨度。MIX 1.1X 三个区组的请求数几乎一样（1210/1194/1197），区组 2_a2 的“负载偏轻”其实是跨度更长（77.8 s 对 70.7/71.1 s），不是请求更少。

## 私有信息
- README.md 第 4 行 “Job 454290（compute-03，…）”：主机名。
- final_experiment_results.md 的运行时 ID `unicorn_instr` / `unicorn_ppv_kv` / `unicorn_ppv_maxwait` / `unicorn_ppv_verify` / `unicorn_async`（第 205、239、304、354、392、396、453–458 行）带有集群名 unicorn。
- Annotation 里有 “Slurm job 454290”，R2b Setup 里有 GPU UUID 前缀 `d4ff46d0 + 51b87964`（低影响）。
- 这 4 个文件里没有 /home/、/Users/ 路径，也没有用户名。

## 已核对、无误（只列数量）
- P1：约 95 个数字正确，包括 15 个登记比值、音频比值范围、描述表 24 个值、强制选入比例 18 个、τ 1 s 的 goodput/padding/队列值、饥饿比例 0.6/2.5/0.4 %、stable capacity 全部、TTFT P95 最大 0.344 s、135/135 格有效且 0 失败。
- R3 确认阶段：表中 75 个值，加上类别、判定、准确率、GPU 忙碌率 80.7–90.4 %、槽位 1.37–1.46、wait ≤ 0.8 ms、份额 ≤ 11.5 %、板卡差 1–22 %/5.5 %、上限 7.2–8.1 / 9.2–11.4，全部正确。
- R4：表中 75 个值（W3 除外）和约 20 个要点数字正确，包括 127,039、43,069、无 KV 等待、714–724 k、40–57 %、1.21×、1.07–1.14×、0.86–1.27×，以及媒体对照的 0.53–0.55→0.23–0.24、0.78–0.85→1.65–1.87 s。
- P2：7 个计数正确（885 的含义见 W6）。
- R2b：12 个数字正确（F2、W7、W8 除外），包括 24.3/26.2/26.2 s、91–94 %、186–188 W、10–13 %、157–158 W、约 1.7 核、无 OOM、8 会话三区组都被跳过。
- Annotation 表与各节一致。Slide 7、8、10 的数字正确（84–92 % 强制选入、15.3 对 17.1/16.8 等）；Slide 10 第 (2) 条沿用了 F1 的说法。P3 的 “one of the two H1c cells” ✓（W3b 0.9X 区组 2_a2 比值 1.100），14 % 功效 ✓（1−e^(−1/6.5)）。

---

## 作者的处理（2026-11-19，报告前）

- F1（P1“低于饱和时 5 种设置没有差别”）：已改。MIX、W3b 在 0.9X、1.0X 相差 ≤ 3%；W3L 在 1.0X 时 FIFO16 已不稳态、低 2–15%。结果、综合、证据表、README、第 7/10 页、介绍 §9–11 都已同步。
- F2（R2b “no first audio arrived”）：已改为“没有一个用户回合得到按时的回答，TTFA 无定义；有少量音频，远落后于实时”。
- F3（第 9 页的 R4 数字只取了区组 1）：已改为 23–34%、1.7–4.1×；介绍里同样的数字一并改。
- W1：R4 写明“登记的各档 ≤ 3%”，并报告媒体历史对照的驱逐损失 12–14%（不进判定）。W2：改写 v1 判定在 64–128 个会话时为什么不可靠。W6：P2 写成“885 次校验，覆盖 662 个完成的语音请求”。W7：R2 的 4 会话基线改为 0% / 0.28%（两个区组）。W8：41.6 GiB。W9：R3 的短片格子“GPU 受限”标为推导（登记类别是“原因不明”）。W10：1.7–4.1×。W4/W5：P1 的“短请求等得更久”改为“6 格里的 4 格”。小问题：32 会话驱逐 0.14/0.05/0.05%。
- 私有信息：README 和几份设计文档里的主机名已删除；运行时配置名 `unicorn_*`、GPU UUID 前缀和 Slurm 作业号是仓库里已有的标识，保留。
- 没有改动任何登记判定。
