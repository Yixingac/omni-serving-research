# 实验 14：怎样复现

这个实验没有新的 GPU 运行，复现指的是从原始数据重新生成 `results/` 和 `figures/`。如果要重新跑 GPU 部分，配置是 `configs/experiments/a6000_8gpu_v2/v2b_f1_r{1,2,4,8}.json` 和 `v2b_f1_r1_events.json`（由 `scripts/slurm/make_plan_8gpu_v2.py` 生成），最多要 8×RTX A6000。

## 需要什么

- 5 个运行目录（列在 [metadata.json](metadata.json)），位于 `PROJECT_DATA_ROOT/experiments/`。原始数据不公开。
- 本仓库的 Python 环境（`.venv`）。不需要 GPU。

## 步骤

1. 给还没有 `analysis/analysis.json` 的运行补上分析（已有的不要覆盖）：

   ```bash
   python scripts/analysis/analyze_experiment.py --slo ttft_s=1.0,e2e_per_audio_s=0.5 --no-figures <运行目录> ...
   ```

2. 汇总：

   ```bash
   python scripts/analysis/exp14_report.py --data-root <results> \
       --run 20261116_v2b_f1_r1_run01 --run 20261116_v2b_f1_r2_run01 --run 20261116_v2b_f1_r4_run01 --run 20261116_v2b_f1_r8_run01 \
       --events 20261117_v2b_f1_r1_events_run01 --out research/01_性能瓶颈/实验14_视频请求加副本
   ```

   最后一行打印判定：近似线性“不确定”（对照算法“支持”），GPU 一句整段平均“不确定”。

3. 画图：

   ```bash
   python scripts/analysis/exp14_figures.py --out research/01_性能瓶颈/实验14_视频请求加副本
   ```

4. 测试：

   ```bash
   python -m pytest -q tests/test_exp14_report.py
   ```
