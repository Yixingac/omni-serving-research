"""Decide where a task runs: the Mac, the personal DGX Spark, or the lab Slurm cluster.

The rules (docs/03_hardware/task_placement.md explains them in Chinese):

1. Formal A6000 measurements, multi-GPU work and anything tied to the A6000/sm_86 stack go to
   Slurm. Nothing else may displace them, and they never fall back to another host.
2. Other GPU/CUDA work goes to the Spark when it is known to run on ARM64; if ARM64 support is
   unknown, the Spark is tried first and Slurm is the fallback.
3. CPU work goes to the host with the smallest estimated total time
   (setup + transfer + compute + save). Very short tasks stay on the Mac because SSH and copies
   would cost more than the work itself.
4. A CPU-heavy task is never placed on a host that is running a formal benchmark at that time.

Estimates are corrected by history: if tasks of a category on a host ran k times longer than
estimated, later estimates for that category/host are multiplied by k.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Host:
    name: str  # mac | spark | slurm
    cpus: int
    memory_gb: float
    gpu: str | None  # None | "GB10" | "RTX A6000"
    arch: str  # arm64 | x86_64
    cuda: bool
    always_available: bool = True
    setup_min: float = 0.0  # fixed cost to start a task there (ssh, env activation, queueing)
    link_gbps: dict[str, float] = field(default_factory=dict)  # effective transfer rate to other hosts
    running_benchmark: bool = False


@dataclass
class Task:
    task_id: str
    description: str
    category: str = "cpu"  # cpu | gpu_dev | benchmark | io
    requires_gpu: bool = False
    requires_cuda: bool = False
    requires_a6000: bool = False
    multi_gpu: bool = False
    arm64_ok: bool | None = True  # None = unknown
    estimated_cpu_cores: int = 1
    estimated_memory_gb: float = 1.0
    estimated_duration_minutes: float = 5.0  # compute time on a reference host (the Spark for CPU work)
    input_host: str = "mac"
    input_size_gb: float = 0.0
    output_size_gb: float = 0.0
    dependencies: list[str] = field(default_factory=list)
    status: str = "pending"
    pinned_host: str | None = None  # e.g. "mac" when the task needs credentials that live only there
    pin_reason: str = ""
    preferred_host: str | None = None  # set by place()
    fallback_host: str | None = None
    reason: str = ""
    estimated_total_minutes: float | None = None


# CPU tasks at least this long go to the always-on Spark rather than the Mac.
LONG_TASK_MINUTES = 5.0

# Relative CPU speed for single-node data processing (Spark Grace cores = 1.0). These are coarse
# placeholders until calibrated by recorded durations.
CPU_SPEED = {"mac": 1.0, "spark": 1.0, "slurm": 0.9}


def _transfer_min(task: Task, src: str, dst: str, hosts: dict[str, Host]) -> float:
    if src == dst or task.input_size_gb <= 0:
        return 0.0
    gbps = hosts[src].link_gbps.get(dst) or hosts[dst].link_gbps.get(src)
    if not gbps:
        return float("inf")
    return task.input_size_gb * 8 / gbps / 60


def total_minutes(task: Task, host: Host, hosts: dict[str, Host], calibration: dict[str, float]) -> float:
    k = calibration.get(f"{task.category}@{host.name}", 1.0)
    compute = task.estimated_duration_minutes * k / CPU_SPEED.get(host.name, 1.0)
    if task.estimated_cpu_cores > host.cpus:
        compute *= task.estimated_cpu_cores / host.cpus
    save = task.output_size_gb * 8 / max(host.link_gbps.get("mac", 1.0), 0.1) / 60 if host.name != "mac" else 0.0
    return host.setup_min + _transfer_min(task, task.input_host, host.name, hosts) + compute + save


def place(task: Task, hosts: dict[str, Host], calibration: dict[str, float] | None = None) -> Task:
    cal = calibration or {}
    if task.pinned_host:
        task.preferred_host, task.fallback_host = task.pinned_host, None
        task.reason = f"固定在 {task.pinned_host}：{task.pin_reason or '只有这台机器具备所需条件'}。"
        task.estimated_total_minutes = total_minutes(task, hosts[task.pinned_host], hosts, cal)
        return task
    if task.requires_a6000 or task.multi_gpu or task.category == "benchmark":
        task.preferred_host, task.fallback_host = "slurm", None
        task.reason = ("正式 A6000 测量或多卡实验：结果必须来自同一套 A6000 硬件和软件栈，"
                       "不能在 Spark 上替代；等待 Slurm 分配。")
        task.estimated_total_minutes = total_minutes(task, hosts["slurm"], hosts, cal)
        return task
    if task.requires_gpu or task.requires_cuda:
        if task.arm64_ok is False:
            task.preferred_host, task.fallback_host = "slurm", None
            task.reason = "需要 CUDA，且已知不支持 ARM64/GB10，只能在 Slurm 上运行。"
        else:
            task.preferred_host, task.fallback_host = "spark", "slurm"
            unknown = "（ARM64 兼容性未知，先在 Spark 上尝试，失败再交给 Slurm）" if task.arm64_ok is None else ""
            task.reason = f"需要 GPU 但不是正式 A6000 测量：用随时可用的 Spark，不占实验室 GPU 时间{unknown}。"
        task.estimated_total_minutes = total_minutes(task, hosts[task.preferred_host], hosts, cal)
        return task
    # CPU-only work: compare estimated total time on each host that has enough memory.
    options = []
    for name in ("mac", "spark"):
        h = hosts[name]
        if task.estimated_memory_gb > h.memory_gb * 0.8:
            continue
        if h.running_benchmark and task.estimated_cpu_cores > 1:
            continue  # never steal CPU from a formal measurement
        options.append((total_minutes(task, h, hosts, cal), name))
    if not options:
        task.preferred_host, task.fallback_host = None, None
        task.reason = "没有满足内存要求且不干扰正式测量的主机，任务推迟。"
        return task
    options.sort()
    best_t, best = options[0]
    names = [n for _, n in options]
    if best == "mac" and "spark" in names and task.estimated_duration_minutes >= LONG_TASK_MINUTES:
        # The Mac sleeps, loses Wi-Fi and closes terminals; anything that takes minutes runs in a
        # tmux session on the always-on Spark instead, even if the estimate is a little higher.
        best_t, best = next(o for o in options if o[1] == "spark")
        task.preferred_host, task.fallback_host = "spark", "mac"
        task.estimated_total_minutes = best_t
        task.reason = (f"耗时约 {best_t:.0f} 分钟的长任务：Mac 可能休眠或断网，放在常开的 Spark 上用 tmux 运行，"
                       "断开连接后也能继续。")
        return task
    task.preferred_host = best
    task.fallback_host = options[1][1] if len(options) > 1 else None
    task.estimated_total_minutes = best_t
    if best == "mac":
        task.reason = (f"CPU 任务，在 Mac 上估计 {best_t:.1f} 分钟；"
                       "交给 Spark 的连接、同步开销不比计算本身小，所以留在 Mac。")
    else:
        task.reason = (f"CPU 任务，在 Spark 上估计 {best_t:.1f} 分钟（含数据传输），"
                       "比 Mac 快或不占用 Mac；不需要实验室 GPU。")
    return task


def calibrate(records: list[dict[str, Any]], min_samples: int = 3) -> dict[str, float]:
    """Median actual/estimated ratio per (category, host) from finished task records.

    A category/host pair is calibrated only once it has ``min_samples`` records, so a single
    unusual run cannot swing later placement decisions."""
    ratios: dict[str, list[float]] = {}
    for r in records:
        est, act = r.get("estimated_duration_minutes"), r.get("actual_duration_minutes")
        if est and act and est > 0:
            ratios.setdefault(f"{r.get('category', 'cpu')}@{r.get('host')}", []).append(act / est)
    out = {}
    for k, v in ratios.items():
        if len(v) < min_samples:
            continue
        v.sort()
        out[k] = v[len(v) // 2]
    return out


def load_manifest(path: str | Path) -> tuple[dict[str, Host], list[Task], list[dict[str, Any]]]:
    text = Path(path).read_text()
    try:
        import yaml

        doc = yaml.safe_load(text)
    except ImportError:
        doc = json.loads(text)
    hosts = {h["name"]: Host(**h) for h in doc["hosts"]}
    tasks = [Task(**t) for t in doc["tasks"]]
    return hosts, tasks, doc.get("history", [])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Show where each task in the placement manifest should run, and why.")
    p.add_argument("manifest", nargs="?", default="configs/orchestration/task_placement.yaml")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    hosts, tasks, history = load_manifest(args.manifest)
    cal = calibrate(history)
    placed = [place(t, hosts, cal) for t in tasks]
    if args.json:
        print(json.dumps([t.__dict__ for t in placed], ensure_ascii=False, indent=2))
        return 0
    if cal:
        print("校正系数（实际/预估）:", ", ".join(f"{k}={v:.2f}" for k, v in sorted(cal.items())))
    for t in placed:
        est = "-" if t.estimated_total_minutes in (None, float("inf")) else f"{t.estimated_total_minutes:.0f} min"
        print(f"{t.task_id:<34} {str(t.preferred_host):<6} (备选 {str(t.fallback_host):<5}) {est:>8}  {t.status:<10} {t.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
