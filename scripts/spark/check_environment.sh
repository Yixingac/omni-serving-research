#!/usr/bin/env bash
# Spark 端：只读检查运行环境，输出一行一项，便于记录到文档或日志。
set -uo pipefail
REPO="${OMNI_REPO:-$HOME/research/repos/omni-serving-research}"
echo "host=$(hostname -s) arch=$(uname -m) kernel=$(uname -r)"
grep -h -E "DGX_OTA_VERSION" /etc/dgx-release 2>/dev/null | head -1
nvidia-smi --query-gpu=name,driver_version,temperature.gpu,utilization.gpu --format=csv,noheader 2>&1 | sed 's/^/gpu: /'
nvidia-smi 2>/dev/null | grep -oE "CUDA Version: [0-9.]+" | sed 's/^/cuda_driver: /'
# Compute processes on the GPU (a renter or another user's job would show up here).
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/gpu_compute_app: /'
echo "gpu-agent: $(systemctl is-active gpu-agent 2>/dev/null || echo unknown)"
free -g | awk '/Mem:/{print "memory_gb: total="$2" used="$3" available="$7}'
echo "cpus: $(nproc)  load: $(cut -d' ' -f1-3 /proc/loadavg)"
df -h "$HOME" | awk 'NR==2{print "disk_home: size="$2" used="$3" avail="$4}'
echo "python3: $(python3 --version 2>&1)"
echo "uv: $("$HOME/.local/bin/uv" --version 2>/dev/null || echo missing)"
echo "nsys: $(command -v nsys || echo missing) $(nsys --version 2>/dev/null | head -1)"
echo "tmux_sessions: $(tmux ls -F '#S' 2>/dev/null | tr '\n' ' ')"
if [[ -x "$REPO/.venv/bin/python" ]]; then
  echo "research_venv: $("$REPO/.venv/bin/python" -c 'import omni_research,sys;print(sys.version.split()[0], omni_research.__version__)' 2>&1)"
fi
if [[ -d "$REPO/.git" ]]; then
  echo "repo: $(git -C "$REPO" rev-parse --abbrev-ref HEAD) @ $(git -C "$REPO" rev-parse --short HEAD) dirty=$(git -C "$REPO" status --porcelain | wc -l)"
fi
