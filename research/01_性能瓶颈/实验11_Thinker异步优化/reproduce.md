# 实验 11：怎样复现

## 需要什么

- 8×RTX A6000 节点；sglang-omni 520b49ee 打上 `configs/runtime/patches/sglang_omni_async_exp11.patch`（测量补丁 + 实验 8 的快照原型 + 旁路复制 + 调度步标注 + digest 里的指纹和输出哈希）。运行配置 `configs/models/runtime_unicorn_async.json`。
- 每个服务用环境变量 `SGLANG_OMNI_SPEECH_LOOKAHEAD` 选组：0（A，同步）、1（B，快照）、sidecopy（C，快照 + 旁路复制）。

## 步骤

1. CPU 检查（替身对象，验证控制流程）：

   ```bash
   cd <打了补丁的 sglang-omni> && PYTHONPATH=$PWD <python> <研究仓库>/configs/runtime/patches/check_sglang_omni_async.py
   PYTHONPATH=$PWD <python> <研究仓库>/configs/runtime/patches/check_sglang_omni_proto.py   # 实验 8 的检查仍应通过
   ```

2. 正确性检查：放 `configs/experiments/plan_exp11_check.json`（A、C、A2 三次运行），然后

   ```bash
   python scripts/analysis/exp11_check.py --data-root <results> --run A=<目录> --run C=<目录> --run A2=<目录> \
       --out research/01_性能瓶颈/实验11_Thinker异步优化
   ```

   打印 PASS 才继续；FAIL 时不做 C 组的性能对比（hypotheses.md）。**2026-11-18 这次是 FAIL，没有做第 3–5 步。**
   - 诊断（补充登记五）：先 `python scripts/slurm/make_plan_exp09_11.py --exp11-diagnostic` 生成配置，放 `configs/experiments/plan_exp11_diag.json`（B、B2 两次运行，各约 6 分钟），然后：

   ```bash
   python scripts/analysis/exp11_check.py --data-root <results> --run A=<A 目录> --run C=<diag B 目录> --run A2=<A2 目录> --out <tmp>/lookahead
   python scripts/analysis/exp11_check.py --data-root <results> --run A=<diag B 目录> --run C=<C 目录> --run A2=<diag B2 目录> --out <tmp>/sidecopy
   cp <tmp>/lookahead/results/check.json research/01_性能瓶颈/实验11_Thinker异步优化/results/diag_lookahead_check.json
   cp <tmp>/sidecopy/results/check.json research/01_性能瓶颈/实验11_Thinker异步优化/results/diag_sidecopy_check.json
   D=research/01_性能瓶颈/实验11_Thinker异步优化
   python scripts/analysis/exp11_diagnostic.py --gate $D/results/check.json --lookahead $D/results/diag_lookahead_check.json \
       --sidecopy $D/results/diag_sidecopy_check.json --out $D
   python scripts/analysis/exp11_figures.py --out $D   # 图 9
   ```

   这次用的目录见 `metadata.json`。
3. 性能对比：放 `configs/experiments/plan_exp11.json`（9 个运行）。
4. 分析：

   ```bash
   python scripts/analysis/exp11_report.py --data-root <results> --run <9 个运行目录> --out research/01_性能瓶颈/实验11_Thinker异步优化
   python scripts/analysis/exp11_ttft.py --data-root <results> --run <9 个运行目录> --phase-re '^w3b_ns_c(\d+)_B$' \
       --arm-re 'block(\d+)_([ABC])' --base A --out research/01_性能瓶颈/实验11_Thinker异步优化 --name ttft
   ```

5. 实测时间线（图 10）：在集群上截窗口，再画图：

   ```bash
   python scripts/analysis/exp11_timeline.py extract --data-root <results> --run A=<区组 1 的 A> --run B=<区组 1 的 B> \
       --run C=<区组 1 的 C> --phase w3b_ns_c64_C --out research/01_性能瓶颈/实验11_Thinker异步优化
   python scripts/analysis/exp11_timeline.py draw --out research/01_性能瓶颈/实验11_Thinker异步优化
   ```

6. 实验 8 旧数据上的 TTFT 拆分（登记前做的，结果已在 hypotheses.md 里说明）：

   ```bash
   python scripts/analysis/exp11_ttft.py --data-root <results> --run <实验 8 第三阶段 6 个运行> --phase-re '^w3b_ns_c(\d+)_B$' \
       --arm-re 'block(\d+)_(off|on)' --out research/01_性能瓶颈/实验11_Thinker异步优化 --name ttft_exp08data
   ```
