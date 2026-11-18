# 实验 19：怎样复现

这个实验没有新的 GPU 运行，复现指的是从原始数据重新生成 `results/` 和 `figures/`。如果要重新跑 GPU 部分，配置是 plan v2 的 `configs/experiments/a6000_8gpu_v2/v2_b1_*.json`、`v2b_b1_*.json`、`v2_b2_*.json`、`v2b_b2l_*.json`（由 `scripts/slurm/make_plan_8gpu_v2.py` 生成），每个都要 8×RTX A6000，运行环境见 `configs/models/runtime_unicorn.json`。

## 需要什么

- 28 个运行目录（列在 [metadata.json](metadata.json)），位于 `PROJECT_DATA_ROOT/experiments/`。原始数据不公开。
- 本仓库的 Python 环境（`.venv`，有 numpy 和 matplotlib）。不需要 GPU。

## 步骤

1. 给还没有 `analysis/analysis.json` 的运行补上分析（用 plan v2 的 SLO；已有的不要覆盖）：

   ```bash
   python scripts/analysis/analyze_experiment.py --slo ttft_s=1.0,e2e_per_audio_s=0.5 --no-figures <运行目录> ...
   ```

2. 汇总（`<results>` 是 `PROJECT_DATA_ROOT`，脚本读 `<results>/experiments/<运行目录>`）：

   ```bash
   args=()
   for l in r8 p4 s2x6; do args+=(--mix 20261115_v2_b1_${l}_run01 --agent-short 20261115_v2_b2_${l}_run01); done
   for l in s3x5 s4x4 s5x3 s6x2; do args+=(--mix 20261116_v2b_b1_${l}_run01); done
   for l in r8 s2x6 p4 s4x4; do args+=(--agent 20261116_v2b_b2l_${l}_run01); done
   for l in r8 p4 s1x7 s2x6 s3x5 s4x4; do for k in 1 2; do args+=(--w3b 20261115_v2_a_${l}_run${k}_run01); done; done
   args+=(--w3b 20261116_v2b_a_s2x6_c03_run01 --w3b 20261116_v2b_a_r8_c03_run01)
   python scripts/analysis/exp19_report.py --data-root <results> "${args[@]}" --out research/01_性能瓶颈/实验19_负载类型和最佳分工
   ```

   最后两行打印两种容差（3%、6%）下的判定，应当是 B-H1 支持、B-H2 支持、B-H3 后半句“命中率小、TTFT 不确定”。

3. 画图：

   ```bash
   python scripts/analysis/exp19_figures.py --out research/01_性能瓶颈/实验19_负载类型和最佳分工
   ```

4. 测试：

   ```bash
   python -m pytest -q tests/test_exp19_report.py
   ```
