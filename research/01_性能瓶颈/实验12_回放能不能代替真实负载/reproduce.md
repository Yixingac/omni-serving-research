# 实验 12：怎样复现

这个实验没有新的 GPU 运行，复现指的是从原始数据重新生成 `results/` 和 `figures/`。如果要重新跑 GPU 部分：先跑实验 19 的 `v2b_b2l_r8`（它录下 trace），再跑 `configs/experiments/a6000_8gpu_v2/v2b_d_*.json`（由 `scripts/slurm/make_plan_8gpu_v2.py` 生成），每个都要 8×RTX A6000。客户端要用 a89165f 以后的版本，否则 4 倍速时会用完文件句柄。

## 需要什么

- 15 个运行目录（列在 [metadata.json](metadata.json)），位于 `PROJECT_DATA_ROOT/experiments/`。原始数据不公开。
- 本仓库的 Python 环境（`.venv`）。不需要 GPU。

## 步骤

1. 给还没有 `analysis/analysis.json` 的运行补上分析（已有的不要覆盖）：

   ```bash
   python scripts/analysis/analyze_experiment.py --slo ttft_s=1.0,e2e_per_audio_s=0.5 --no-figures <运行目录> ...
   ```

2. 汇总：

   ```bash
   args=()
   for x in dep_orig_1x dep_orig_2x dep_orig_copy4_1x dep_prefix_1x dep_sur_1x req_orig_1x; do args+=(--replay 20261116_v2b_d_${x}_run01); done
   for x in dep_orig dep_prefix dep_sur req_orig; do args+=(--replay 20261117_v2b_d_${x}_4x_run01 --replay 20261117_v2b_d_${x}_4x_r2_run01); done
   python scripts/analysis/exp12_report.py --data-root <results> --live 20261116_v2b_b2l_r8_run01:lo_text_c256 "${args[@]}" \
       --out research/01_性能瓶颈/实验12_回放能不能代替真实负载
   ```

   最后三行打印 D-H1 三句话的判定：不支持、不支持、支持。

3. 画图：

   ```bash
   python scripts/analysis/exp12_figures.py --out research/01_性能瓶颈/实验12_回放能不能代替真实负载
   ```

4. 测试：

   ```bash
   python -m pytest -q tests/test_exp12_report.py
   ```
