# 环境安装

## 研究代码（任何机器，不需要 GPU）

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest -q
```

核心模块只依赖 Python 标准库，所以也可以直接用 SGLang-Omni 运行环境里的 Python 运行（实验执行器就是这么做的），只要把 `src/` 加进 `PYTHONPATH`。

## SGLang-Omni 运行环境

两套环境使用相同的 commit 和依赖版本：

| | Unicorn（A6000，x86_64） | Spark（GB10，aarch64） |
|---|---|---|
| 位置 | `~/workspace/sglang-omni-learning`（NFS） | `~/research/omni-runtime` |
| 构建 | 早先学习项目的 `scripts/setup-unicorn.sh` | 本仓库 `scripts/spark/setup_runtime.sh` |
| sglang-omni | `520b49ee`（fork `Yixingac/sglang-omni` 的 `learning/sglang-omni` 分支 = 上游 `1845546e` + 两个 pre-Hopper 修复） | 同左 |
| sglang | `e00930c5`（v0.5.21），editable | 同左 |
| torch | 2.13.0+cu130 | 同左 |
| 运行时配置 | `configs/models/runtime_unicorn.json` | `configs/models/runtime_spark.json` |

两个 pre-Hopper 修复（A6000 是 sm_86）：
1. 融合的 code predictor Triton kernel 需要 132 KB 共享内存，sm_86/89 每个 block 最多 99 KB，改为在共享内存不够时走普通路径。上游在 #3284 修复（晚于我们固定的版本）。
2. Qwen3-Omni talker 的 MoE 后端在 SM < 90 上改用 triton（`flashinfer_cutlass` 没有 sm_86 的 kernel）。

2026-11-11 在两套环境里补装了 sglang-omni 的 `minicpm-o` 可选依赖（`onnx`、`einops`），否则 MiniCPM-o 无法启动：

```bash
<运行环境>/bin/uv pip install --python <运行环境>/bin/python -c <版本锁定文件> "onnx>=1.18.0" "einops>=0.8.1"
```

实验执行器启动服务前会检查 `required_imports` 里的模块都能导入，缺了就直接失败并写明缺什么。

## 模型权重

| 模型 | revision | 下载 |
|---|---|---|
| MiniCPM-o 4.5 | `fb09628e98b1f55d8d5b863e0eccfcdd11bff499` | `hf download openbmb/MiniCPM-o-4_5 --revision <rev> --local-dir <dir>` |
| Qwen3-Omni-30B-A3B-Instruct | `26291f793822fb6be9555850f06dfe95f2d7e695` | 同上 |
| Qwen3-TTS-12Hz-0.6B-Base | `5d83992436eae1d760afd27aff78a71d676296fc` | 同上 |

权重放在运行时配置指定的目录，不进 git。
