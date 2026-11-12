"""Allocation discovery (Slurm or local) and remaining-walltime queries.

Nothing here assumes a cluster layout: GPU/CPU/memory counts come from the Slurm environment of
the current job (or ``scontrol show job`` when the orchestrator runs outside the job, e.g. on a
login node targeting ``--jobid``), and from explicit flags in local mode.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class NodeResources:
    name: str
    gpus: int
    cpus: int
    mem_gb: float
    gpu_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "gpus": self.gpus, "cpus": self.cpus, "mem_gb": self.mem_gb, "gpu_type": self.gpu_type}


@dataclass
class Allocation:
    mode: str  # "slurm" | "local"
    job_id: str | None
    nodes: list[NodeResources]
    end_time: float | None = None  # epoch seconds, if known
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "job_id": self.job_id, "nodes": [n.to_dict() for n in self.nodes],
                "end_time": self.end_time, **self.extra}


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False).stdout


def parse_mem_mb(s: str) -> float:
    m = re.fullmatch(r"([\d.]+)([KMGT]?)", s.strip())
    if not m:
        raise ValueError(f"cannot parse memory {s!r}")
    v = float(m.group(1))
    return v * {"K": 1 / 1024, "": 1, "M": 1, "G": 1024, "T": 1024 * 1024}[m.group(2)]


def parse_scontrol_job(text: str) -> dict[str, Any]:
    """Extract per-node TRES from `scontrol show job -d` output (single- or multi-node)."""
    out: dict[str, Any] = {"nodes": []}
    m = re.search(r"\bEndTime=(\S+)", text)
    out["end_time_str"] = m.group(1) if m else None
    m = re.search(r"\bJobState=(\S+)", text)
    out["state"] = m.group(1) if m else None
    # Detail lines: "  Nodes=node1 CPU_IDs=0-7 Mem=65536 GRES=gpu:nvidia_rtx_a6000:2(IDX:0-1)"
    for mm in re.finditer(r"Nodes=(\S+) CPU_IDs=(\S+) Mem=(\d+) GRES=(\S*)", text):
        nodes, cpu_ids, mem, gres = mm.groups()
        ncpu = 0
        for part in cpu_ids.split(","):
            if "-" in part:
                a, b = part.split("-")
                ncpu += int(b) - int(a) + 1
            elif part:
                ncpu += 1
        g = re.search(r"gpu:(?:([^:(]+):)?(\d+)", gres)
        out["nodes"].append({"nodes": nodes, "cpus": ncpu, "mem_mb": int(mem),
                             "gpus": int(g.group(2)) if g else 0, "gpu_type": g.group(1) if g else None})
    return out


def discover_allocation(job_id: str | None = None, local_gpus: int = 0, local_cpus: int | None = None,
                        local_mem_gb: float | None = None) -> Allocation:
    job_id = job_id or os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return Allocation(
            mode="local",
            job_id=None,
            nodes=[NodeResources(os.uname().nodename, local_gpus, local_cpus or os.cpu_count() or 1, local_mem_gb or 16.0)],
        )
    info = parse_scontrol_job(_run(["scontrol", "show", "job", "-d", str(job_id)]))
    nodes = []
    for n in info["nodes"]:
        expanded = _run(["scontrol", "show", "hostnames", n["nodes"]]).split() or [n["nodes"]]
        for host in expanded:
            nodes.append(NodeResources(host, n["gpus"], n["cpus"], n["mem_mb"] / 1024, n["gpu_type"]))
    end = None
    if info.get("end_time_str") and info["end_time_str"] not in ("Unknown", "None"):
        try:
            end = time.mktime(time.strptime(info["end_time_str"], "%Y-%m-%dT%H:%M:%S"))
        except ValueError:
            end = None
    return Allocation(mode="slurm", job_id=str(job_id), nodes=nodes, end_time=end, extra={"state": info.get("state")})


def slurm_time_left_s(job_id: str) -> float | None:
    """Remaining walltime from squeue %L ([d-]hh:mm:ss); None if unknown/unlimited."""
    tl = _run(["squeue", "-h", "-j", str(job_id), "-o", "%L"]).strip()
    if not tl or tl in ("INVALID", "UNLIMITED", "NOT_SET"):
        return None
    days = 0
    if "-" in tl:
        d, tl = tl.split("-", 1)
        days = int(d)
    secs = 0
    for p in tl.split(":"):
        secs = secs * 60 + int(p)
    return days * 86400 + secs


def make_time_left_fn(alloc: Allocation, fixed_deadline: float | None = None) -> Callable[[], float]:
    """Seconds left before the allocation (or an explicit deadline) ends; inf if unbounded."""
    def fn() -> float:
        candidates = []
        if fixed_deadline is not None:
            candidates.append(fixed_deadline - time.time())
        if alloc.mode == "slurm" and alloc.job_id:
            left = slurm_time_left_s(alloc.job_id)
            if left is not None:
                candidates.append(left)
            elif alloc.end_time:
                candidates.append(alloc.end_time - time.time())
        return min(candidates) if candidates else float("inf")
    return fn
