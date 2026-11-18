# 实验 3：怎样复现

这个实验没有新的 GPU 运行，复现指的是从原始数据重新生成 `results/` 和 `figures/`。如果要重新跑 GPU 部分，配置是 plan v2 的 `configs/experiments/a6000_8gpu_v2/v2b_sb_*.json`、`v2b_a_*_c03.json`、`v2_a_*_run1.json`、`v2b_aoh_*.json`（由 `scripts/slurm/make_plan_8gpu_v2.py` 生成），都要 8×RTX A6000。

## 需要什么

- 14 个运行目录（列在 [metadata.json](metadata.json)），位于 `PROJECT_DATA_ROOT/experiments/`。原始数据不公开。
- 本仓库的 Python 环境（`.venv`）。不需要 GPU。

## 步骤

1. 汇总：

   ```bash
   python scripts/analysis/exp03_report.py --data-root <results> \
     --curve s2x6:20261116_v2b_sb_s2x6_run01:w3b_small --curve s2x6:20261116_v2b_a_s2x6_c03_run01:w3b_closed \
     --curve r8:20261116_v2b_sb_r8_run01:w3b_small --curve r8:20261116_v2b_a_r8_c03_run01:w3b_closed \
     --curve p4:20261116_v2b_sb_p4_run01:w3b_small --curve p4:20261115_v2_a_p4_run1_run01:w3b_closed \
     --curve s1x7:20261116_v2b_sb_s1x7_run01:w3b_small --curve s1x7:20261115_v2_a_s1x7_run1_run01:w3b_closed \
     $(for l in s2x6 s1x7 s3x5 s4x4 p4 r8; do echo --open $l:20261116_v2b_aoh_${l}_run01; done) \
     --out research/01_性能瓶颈/实验03_同时处理多少请求最好
   ```

2. 画图：

   ```bash
   python scripts/analysis/exp03_figures.py --out research/01_性能瓶颈/实验03_同时处理多少请求最好
   ```

3. 测试：

   ```bash
   python -m pytest -q tests/test_exp03_report.py
   ```
