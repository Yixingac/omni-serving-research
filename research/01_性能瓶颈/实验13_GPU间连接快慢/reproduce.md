# 实验 13：怎样复现

这个实验没有新的 GPU 运行，复现指的是从原始数据重新生成 `results/` 和 `figures/`。如果要重新跑 GPU 部分：拓扑测量是 `configs/experiments/a6000_8gpu_v2/v2b_y0_env_topology.json`（`scripts/slurm/env_topology.sh`），两组对比是 `v2b_g1a_links.json`、`v2b_g1b_links.json`（由 `scripts/slurm/make_plan_8gpu_v2.py` 生成），都要 8×RTX A6000。不同节点的 NVLink 配对不同，要先看拓扑测量的结果再定服务放在哪几张卡上。

## 需要什么

- 3 个运行目录（列在 [metadata.json](metadata.json)），位于 `PROJECT_DATA_ROOT/experiments/`。原始数据不公开。
- 本仓库的 Python 环境（`.venv`）。不需要 GPU。

## 步骤

1. 汇总：

   ```bash
   python scripts/analysis/exp13_report.py --data-root <results> --topology 20261116_v2b_y0_env_topology_run01 \
       --run 20261116_v2b_g1a_links_run01 --run 20261116_v2b_g1b_links_run01 \
       --out research/01_性能瓶颈/实验13_GPU间连接快慢
   ```

   最后三行打印判定：闭环“不确定”，开环“支持”，总体“不确定”。

2. 画图：

   ```bash
   python scripts/analysis/exp13_figures.py --out research/01_性能瓶颈/实验13_GPU间连接快慢
   ```

3. 测试：

   ```bash
   python -m pytest -q tests/test_exp13_report.py
   ```
