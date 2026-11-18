#!/usr/bin/env python3
"""Parse the raw output of preflight.sh into one JSON topology document (stdlib only).

Every field is taken from a command's output. A value that could not be read stays ``None``
instead of being guessed, and ``sources`` names the file each section came from.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
from pathlib import Path


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def _body(text: str) -> str:
    """Drop the '$ cmd' header line and the '[exit=N]' trailer added by preflight.sh."""
    lines = text.splitlines()
    if lines and lines[0].startswith("$ "):
        lines = lines[1:]
    if lines and lines[-1].startswith("[exit="):
        lines = lines[:-1]
    return "\n".join(lines)


def parse_gpu_query(text: str) -> list[dict]:
    body = _body(text).strip()
    if not body or "," not in body:
        return []
    rows = list(csv.reader(io.StringIO(body)))
    header = [h.strip().split(" [")[0] for h in rows[0]]
    gpus = []
    for row in rows[1:]:
        if len(row) != len(header):
            continue
        gpus.append({k: v.strip() for k, v in zip(header, row)})
    return gpus


def parse_topo(text: str) -> dict:
    """Parse `nvidia-smi topo -m` into a link matrix plus per-GPU CPU/NUMA affinity."""
    body = _body(text)
    lines = [re.sub(r"\x1b\[[0-9;]*m", "", ln) for ln in body.splitlines() if ln.strip()]
    if not lines:
        return {}
    header = lines[0].split()
    gpu_cols = [h for h in header if re.fullmatch(r"GPU\d+", h)]
    matrix: dict[str, dict[str, str]] = {}
    affinity: dict[str, dict[str, str]] = {}
    for ln in lines[1:]:
        parts = ln.split()
        if not parts or not re.fullmatch(r"GPU\d+", parts[0]):
            continue
        name = parts[0]
        links = parts[1 : 1 + len(gpu_cols)]
        matrix[name] = dict(zip(gpu_cols, links))
        rest = parts[1 + len(gpu_cols) :]
        affinity[name] = {
            "cpu_affinity": rest[0] if len(rest) > 0 else None,
            "numa_affinity": rest[1] if len(rest) > 1 else None,
        }
    nvlink_pairs = sorted(
        {tuple(sorted((a, b))) for a, row in matrix.items() for b, v in row.items() if v.startswith("NV")}
    )
    return {
        "gpus": gpu_cols,
        "matrix": matrix,
        "affinity": affinity,
        "nvlink_pairs": [list(p) for p in nvlink_pairs],
    }


def parse_nvlink(text: str) -> dict[str, list[float]]:
    """Per-GPU active NVLink link speeds in GB/s from `nvidia-smi nvlink -s`."""
    result: dict[str, list[float]] = {}
    current = None
    for ln in _body(text).splitlines():
        m = re.match(r"GPU (\d+):", ln)
        if m:
            current = f"GPU{m.group(1)}"
            result[current] = []
            continue
        m = re.search(r"Link \d+: ([\d.]+) GB/s", ln)
        if m and current:
            result[current].append(float(m.group(1)))
    return result


def parse_kv_lines(text: str) -> dict[str, str]:
    out = {}
    for ln in _body(text).splitlines():
        if "=" in ln:
            k, v = ln.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def parse_lscpu(text: str) -> dict[str, str]:
    out = {}
    for ln in _body(text).splitlines():
        if ":" in ln:
            k, v = ln.split(":", 1)
            out[k.strip()] = v.strip()
    keep = ["Model name", "Socket(s)", "Core(s) per socket", "Thread(s) per core", "CPU(s)", "NUMA node(s)", "L3"]
    return {k: out.get(k) for k in keep}


def parse_net(text: str) -> list[dict]:
    nics = []
    for ln in _body(text).splitlines():
        m = re.match(r"(\S+) speed=(\S*) operstate=(\S*)", ln)
        if m:
            speed = m.group(2)
            nics.append(
                {
                    "name": m.group(1),
                    "speed_mbps": int(speed) if speed.lstrip("-").isdigit() and int(speed) > 0 else None,
                    "operstate": m.group(3),
                }
            )
    return nics


def parse_node(node_dir: Path) -> dict:
    r = lambda name: _read(node_dir / f"{name}.txt")  # noqa: E731
    env = parse_kv_lines(r("env_slurm"))
    cpuset = re.search(r"Cpus_allowed_list:\s*(\S+)", r("cpuset"))
    mem_total = re.search(r"Mem:\s+(\d+)", _body(r("free")))
    driver = re.search(r"RmProfilingAdminOnly:\s*(\d+)", r("driver_params"))
    nvcc = re.search(r"release ([\d.]+)", r("nvcc"))
    nsight = {}
    for ln in _body(r("nsight")).splitlines():
        if ":" in ln:
            tool, path = ln.split(":", 1)
            path = path.strip()
            nsight[tool.strip()] = None if not path or path == "none" else path
    net_text = r("net")
    return {
        "hostname": _body(r("hostname")).strip() or node_dir.name,
        "cpu": parse_lscpu(r("lscpu")),
        "cpus_allowed": cpuset.group(1) if cpuset else None,
        "memory_total_bytes": int(mem_total.group(1)) if mem_total else None,
        "cgroup_memory": parse_kv_lines(r("cgroup_mem")),
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        "slurm_env": {k: v for k, v in env.items() if k.startswith("SLURM_")},
        "gpus": parse_gpu_query(r("nvidia_smi_query")),
        "topology": parse_topo(r("nvidia_topo")),
        "nvlink_active_gbps": parse_nvlink(r("nvidia_nvlink")),
        "profiling_admin_only": int(driver.group(1)) if driver else None,
        "nvcc_release": nvcc.group(1) if nvcc else None,
        "nsight": nsight,
        "nics": parse_net(r("net")),
        # `ls /dev/infiniband` prints "No such file or directory" when there is no IB device.
        "infiniband_devices_present": ("No such file" not in net_text) if net_text else None,
        "sources": sorted(p.name for p in node_dir.glob("*.txt")),
    }


def main(out_dir: str) -> None:
    root = Path(out_dir)
    nodes = [parse_node(d) for d in sorted(p for p in root.iterdir() if p.is_dir())]
    job_text = _read(root / "scontrol_job.txt")
    job = {}
    for key in ("JobId", "Partition", "NodeList", "NumNodes", "NumCPUs", "TimeLimit", "StartTime", "EndTime", "TRES"):
        m = re.search(rf"\b{key}=(\S+)", job_text)
        job[key] = m.group(1) if m else None
    doc = {
        "schema": "omni_research.preflight/v1",
        "job": job,
        "time_left": _read(root / "time_left.txt").strip() or None,
        "node_count": len(nodes),
        "gpu_count": sum(len(n["gpus"]) for n in nodes),
        "nodes": nodes,
    }
    json.dump(doc, sys.stdout, indent=2, sort_keys=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
