# 实验 10：怎样复现

## 需要什么

- 8×RTX A6000 节点；sglang-omni 520b49ee 打上 `configs/runtime/patches/sglang_omni_batch_exp10.patch`（测量补丁 + code2wav 组批控制 + digest）。运行配置 `configs/models/runtime_unicorn_batch.json`。
- 压测程序需要 control 阶段和服务环境变量里的 `{exp_dir}` 占位符（研究仓库 91c2cdc 起）。

## 步骤

1. 补丁的 CPU 检查（在改动之前的代码上应当失败，之后通过）：

   ```bash
   cd <打了补丁的 sglang-omni> && PYTHONPATH=$PWD <python> <研究仓库>/configs/runtime/patches/check_sglang_omni_batch.py
   ```

2. 关 graph 的配置检查：`python scripts/spark/check_pipeline_config.py --gpus 8 --config configs/models/minicpm_o_4_5_exp10_t2_k4_c2_nograph.yaml`。
3. 先放正确性检查 `configs/experiments/plan_exp10_check.json`，再放性能运行 `configs/experiments/plan_exp10.json`（6 个运行）。
4. 分析和作图：

   ```bash
   python scripts/analysis/exp10_report.py --data-root <results> --run <6 个性能运行目录> --check-run <正确性运行目录> \
       --out research/01_性能瓶颈/实验10_Code2Wav_Batching
   python scripts/analysis/exp10_figures.py --out research/01_性能瓶颈/实验10_Code2Wav_Batching
   ```

## 控制文件怎样起作用

每次运行的服务带环境变量 `SGLANG_OMNI_C2W_CONTROL=<运行目录>/c2w_control.json`。每个测量段之前有一个 control 段，把 `{"max_batch_size": …, "policy": "fifo"|"length", "window": 64}` 原子地写进这个文件，等 3 秒；code2wav 每次组批前发现文件变了就重新读。每批的事件里记下实际生效的批上限和策略，分析时逐批核对。
