#!/usr/bin/env bash
# Spark 端：构建 SGLang-Omni 运行环境（aarch64 / GB10 / CUDA 13.0），全部装在用户目录，不需要 sudo。
# 版本与 Unicorn A6000 环境一致：sglang-omni 520b49ee（= 上游 1845546e + 两个 pre-Hopper 修复），
# sglang e00930c5 (v0.5.21)，Python 包版本用 configs/runtime/sglang_omni_aarch64_constraints.txt 锁定。
# 每一步都是幂等的，失败后可以直接重跑。
#
#   setup_runtime.sh [repos|runtime|python|check|all]
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
ROOT="${OMNI_RUNTIME_ROOT:-$HOME/research/omni-runtime}"
OMNI_URL=https://github.com/Yixingac/sglang-omni.git
OMNI_COMMIT=520b49ee
SGLANG_URL=https://github.com/sgl-project/sglang.git
SGLANG_COMMIT=e00930c5489053f26d86b179cee0d087f846acbb
PY="$ROOT/.conda-runtime/bin/python"
mkdir -p "$ROOT"/{repos,tools,logs}
cd "$ROOT"
step() { printf '\n==> %s  (%s)\n' "$*" "$(date +%T)"; }

repos() {
  step "clone repositories"
  clone() {
    local url=$1 dir=$2 commit=$3
    [ -d "$dir/.git" ] || git clone -q --filter=blob:none "$url" "$dir"
    git -C "$dir" cat-file -e "$commit^{commit}" 2>/dev/null || git -C "$dir" fetch -q origin "$commit" || git -C "$dir" fetch -q origin
    git -C "$dir" checkout -q -B research/pinned "$commit"
    echo "$dir @ $(git -C "$dir" rev-parse --short HEAD)"
  }
  clone "$OMNI_URL" repos/sglang-omni "$OMNI_COMMIT"
  clone "$SGLANG_URL" repos/sglang "$SGLANG_COMMIT"
}

runtime() {
  step "conda-forge runtime (aarch64)"
  # micromamba 1.5.10: the 2.x 'latest' build segfaulted on Unicorn; keep both platforms on one version.
  [ -x tools/bin/micromamba ] || curl -fsSL https://micro.mamba.pm/api/micromamba/linux-aarch64/1.5.10 | tar -xj -C tools bin/micromamba
  export MAMBA_ROOT_PREFIX="$ROOT/tools/mamba"
  # pynini is pinned: nemo-text-processing 1.2.0 (an sglang-omni dependency) needs exactly
  # 2.1.6.post1, and a newer conda pynini makes pip try (and fail) to compile it from source.
  local packages=(python=3.12 uv "ffmpeg>=4,<9" "pynini=2.1.6.post1" libsndfile
                  cuda-version=13.0 cuda-nvcc cuda-cudart-dev cuda-nvrtc-dev cuda-cccl
                  libcublas-dev libcurand-dev cuda-nvtx-dev cuda-profiler-api)
  # conda-forge has no linux-aarch64 build of sox (2026-11-11); it is only needed by a few TTS
  # utilities, so it is tried separately and skipped when unavailable.
  [ -x "$PY" ] || tools/bin/micromamba create -y -q -p "$ROOT/.conda-runtime" -c conda-forge "${packages[@]}"
  # Idempotent fix-up for an environment created before the pin existed.
  tools/bin/micromamba install -y -q -p "$ROOT/.conda-runtime" -c conda-forge "pynini=2.1.6.post1"
  tools/bin/micromamba install -y -q -p "$ROOT/.conda-runtime" -c conda-forge sox 2>/dev/null || echo "sox: not available on this platform, skipped"
  # JIT builds link with -L$CUDA_HOME/lib64; on aarch64 the conda toolkit keeps its libraries here.
  ln -sfn targets/sbsa-linux/lib .conda-runtime/lib64
  "$PY" --version
  .conda-runtime/bin/nvcc --version | tail -1
}

python_packages() {
  step "Python packages (aarch64 constraints)"
  local c="$REPO/configs/runtime/sglang_omni_aarch64_constraints.txt"
  .conda-runtime/bin/uv pip install --python "$PY" --prerelease=allow -c "$c" -e repos/sglang-omni debugpy
  SGLANG_BUILD_RUST_EXTS=none .conda-runtime/bin/uv pip install --python "$PY" --no-deps -e repos/sglang/python
  .conda-runtime/bin/uv pip freeze --python "$PY" > logs/requirements-spark.txt
}

check() {
  step "import check"
  CUDA_HOME="$ROOT/.conda-runtime" "$PY" - <<'PY'
import torch, sglang, sglang_omni, flashinfer
print("torch", torch.__version__, "cuda", torch.version.cuda, "device", torch.cuda.get_device_name(0),
      "capability", torch.cuda.get_device_capability(0))
print("sglang", sglang.__version__, "sglang_omni", getattr(sglang_omni, "__version__", "?"), "flashinfer", flashinfer.__version__)
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
print("matmul ok", float((x @ x).float().abs().mean()) > 0)
PY
}

case "${1:-all}" in
  repos) repos ;;
  runtime) runtime ;;
  python) python_packages ;;
  check) check ;;
  all) repos; runtime; python_packages; check ;;
  *) echo "usage: $0 [repos|runtime|python|check|all]" >&2; exit 2 ;;
esac
step "done: ${1:-all}"
