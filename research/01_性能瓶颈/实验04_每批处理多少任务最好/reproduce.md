# 实验 4：怎样复现

这个实验没有新的 GPU 运行，复现指的是从原始数据重新生成 `results/` 和 `figures/`。GPU 部分的配置和实验 3 相同。

## 需要什么

- 8 个运行目录（列在 [metadata.json](metadata.json)），位于 `PROJECT_DATA_ROOT/experiments/`，要有 `raw/server_events/` 和 `logs/server_*.log`。原始数据不公开。
- 本仓库的 Python 环境（`.venv`）。不需要 GPU。

## 步骤

1. 汇总：

   ```bash
   python scripts/analysis/exp04_report.py --data-root <results> \
     --curve s2x6:20261116_v2b_sb_s2x6_run01:w3b_small --curve s2x6:20261116_v2b_a_s2x6_c03_run01:w3b_closed \
     --curve r8:20261116_v2b_sb_r8_run01:w3b_small --curve r8:20261116_v2b_a_r8_c03_run01:w3b_closed \
     --curve p4:20261116_v2b_sb_p4_run01:w3b_small --curve p4:20261115_v2_a_p4_run1_run01:w3b_closed \
     --curve s1x7:20261116_v2b_sb_s1x7_run01:w3b_small --curve s1x7:20261115_v2_a_s1x7_run1_run01:w3b_closed \
     --out research/01_性能瓶颈/实验04_每批处理多少任务最好
   ```

2. 画图：

   ```bash
   python scripts/analysis/exp04_figures.py --out research/01_性能瓶颈/实验04_每批处理多少任务最好
   ```

3. 测试：

   ```bash
   python -m pytest -q tests/test_exp04_report.py
   ```
