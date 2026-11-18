# 实验 9：怎样复现

## 需要什么

- 一台 8×RTX A6000 的节点（本实验在 compute-03，Slurm Job 454290）。
- sglang-omni 520b49ee，打上 `configs/runtime/patches/sglang_omni_instr_exp06_08.patch`（测量补丁，和实验 6、7 相同）。运行配置是 `configs/models/runtime_unicorn_instr.json`。
- MiniCPM-o 4.5（revision fb09628e98b1f55d8d5b863e0eccfcdd11bff499）。

## 步骤

1. 生成配置（已经提交，重新生成应当没有变化）：

   ```bash
   python scripts/slurm/make_plan_exp09_11.py --check
   ```

2. 新放置的离线检查（不占 GPU）：

   ```bash
   python scripts/spark/check_pipeline_config.py --gpus 8 --config configs/models/minicpm_o_4_5_exp_t2_k1_c5.yaml
   ```

3. 预跑：把 `configs/experiments/plan_exp09_pilot.json` 放进作业，例如 `scripts/mac/place_plan.sh <JOBID> <RUN_DIR> configs/experiments/plan_exp09_pilot.json exp09_pilot`。
4. 由预跑定开环速率（规则见 hypotheses.md）：

   ```bash
   python scripts/analysis/exp09_report.py --data-root <results> --pilot --run <4 个预跑目录> --out <目录>
   ```

   把打印出的速率填进 `make_plan_exp09_11.py` 的 `EXP09_RATES_W3B`、`EXP09_RATES_W3L`，重新生成配置。
5. 正式运行：放 `configs/experiments/plan_exp09.json`（12 个运行，每个 23–29 分钟，含约 3 分钟启动；12 个共约 5 小时 15 分钟）。
   这次用的 12 个目录见 `metadata.json` 的 `runs`；方案 2 区组 1 用 `run02`（`run01` 启动失败）。
6. 分析和作图：

   ```bash
   python scripts/analysis/exp09_report.py --data-root <results> --run <12 个正式运行目录> --out research/01_性能瓶颈/实验09_最佳GPU分工
   python scripts/analysis/exp09_stalls.py --data-root <results> --run <同上> --out research/01_性能瓶颈/实验09_最佳GPU分工/results/stalls.json
   python scripts/analysis/exp09_figures.py --out research/01_性能瓶颈/实验09_最佳GPU分工
   python scripts/analysis/exp09_11_diagrams.py   # 示意图
   ```

## 运行目录

`results/experiments/<日期>_exp09_<放置>_block<区组>_run01/`：`raw/requests_<段>.jsonl`（客户端记录）、`raw/server_events/<段>/`（B 段的服务端事件）、`raw/gpu_samples.csv`（nvidia-smi 每 0.2 秒）、`result.json`、`logs/`。原始数据带机器名和路径，不进公开仓库。
