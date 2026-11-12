"""GPU-to-GPU measurements for plan v2's Y0 step (run with the SGLang-Omni runtime's python, inside
an allocation that owns all the GPUs it tests).

    python scripts/slurm/gpu_p2p_check.py --out raw/p2p.json            # everything below
    python scripts/slurm/gpu_p2p_check.py --pair 0 1                     # one pair, both directions
    python scripts/slurm/gpu_p2p_check.py --allreduce 0 1 [--no-p2p]     # 2-rank NCCL all-reduce

Every measurement runs in its own subprocess with a timeout, because peer traffic has hung on a
PCIe-only node before (NCCL P2P on a PIX pair). A hang becomes a recorded result, not a stuck step.
NVLink pairs are read from `nvidia-smi topo -m`; the NCCL all-reduce with P2P on runs only on such a
pair, and a non-NVLink pair is measured with NCCL_P2P_DISABLE=1 for comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time


def parse_topo(text: str) -> dict[tuple[int, int], str]:
    """Link type for each GPU pair (i < j) from the `nvidia-smi topo -m` matrix, e.g. {(0, 1): "NV4"}."""
    links: dict[tuple[int, int], str] = {}
    header: list[str] | None = None
    for line in text.splitlines():
        cols = line.split()
        if not cols:
            continue
        if header is None:
            if cols[0].startswith("GPU0"):
                header = cols
            continue
        m = re.fullmatch(r"GPU(\d+)", cols[0])
        if not m:
            continue  # NIC rows, the legend
        i = int(m.group(1))
        for name, val in zip(header, cols[1:]):
            mj = re.fullmatch(r"GPU(\d+)", name)
            if mj and int(mj.group(1)) > i:
                links[(i, int(mj.group(1)))] = val
    return links


def _pair(i: int, j: int, mib: int, iters: int) -> dict:
    import torch

    n = mib * 1024 * 1024
    out = {"pair": [i, j], "peer_access": [torch.cuda.can_device_access_peer(i, j), torch.cuda.can_device_access_peer(j, i)]}
    for src, dst in ((i, j), (j, i)):
        a = torch.empty(n, dtype=torch.uint8, device=f"cuda:{src}")
        b = torch.empty(n, dtype=torch.uint8, device=f"cuda:{dst}")
        b.copy_(a)
        torch.cuda.synchronize(src)
        torch.cuda.synchronize(dst)
        t0 = time.perf_counter()
        for _ in range(iters):
            b.copy_(a, non_blocking=True)
        torch.cuda.synchronize(src)
        torch.cuda.synchronize(dst)
        dt = time.perf_counter() - t0
        out[f"{src}->{dst}_GBps"] = round(n * iters / dt / 1e9, 2)
    return out


def _allreduce_rank(rank: int, devs: list[int], mib: int, iters: int, port: int, q) -> None:  # noqa: ANN001
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(devs[rank])
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=len(devs))
    x = torch.ones(mib * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    for _ in range(3):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    if rank == 0:
        size = x.numel() * 4
        world = len(devs)
        q.put({"devices": devs, "MiB": mib, "algbw_GBps": round(size / dt / 1e9, 2),
               "busbw_GBps": round(size / dt / 1e9 * 2 * (world - 1) / world, 2), "p2p_disabled": os.environ.get("NCCL_P2P_DISABLE") == "1"})
    dist.destroy_process_group()


def _allreduce(i: int, j: int, mib: int, iters: int) -> dict:
    import socket

    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    procs = [ctx.Process(target=_allreduce_rank, args=(r, [i, j], mib, iters, port, q)) for r in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    if any(p.exitcode != 0 for p in procs):
        return {"devices": [i, j], "error": f"exit codes {[p.exitcode for p in procs]}"}
    return q.get()


def _sub(args: list[str], timeout_s: float, env: dict | None = None) -> dict:
    t0 = time.monotonic()
    try:
        cp = subprocess.run([sys.executable, os.path.abspath(__file__), *args], capture_output=True, text=True,
                            timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired:
        return {"args": args, "error": f"timed out after {timeout_s:.0f}s (treated as a hang)"}
    lines = [ln for ln in cp.stdout.splitlines() if ln.startswith("{")]
    if cp.returncode != 0 or not lines:
        return {"args": args, "error": f"exit {cp.returncode}: {cp.stderr[-800:]}"}
    res = json.loads(lines[-1])
    res["wall_s"] = round(time.monotonic() - t0, 1)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pair", nargs=2, type=int)
    ap.add_argument("--allreduce", nargs=2, type=int)
    ap.add_argument("--no-p2p", action="store_true")
    ap.add_argument("--mib", type=int, default=256)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--timeout-s", type=float, default=90.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.pair:
        print(json.dumps(_pair(a.pair[0], a.pair[1], a.mib, a.iters)))
        return 0
    if a.allreduce:
        print(json.dumps(_allreduce(a.allreduce[0], a.allreduce[1], a.mib, max(a.iters, 20))))
        return 0

    topo = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout
    links = parse_topo(topo)
    gpus = sorted({g for p in links for g in p}) or [0]
    report: dict = {"links": {f"{i}-{j}": v for (i, j), v in sorted(links.items())}, "pairs": [], "allreduce": []}
    for (i, j) in sorted(links):
        report["pairs"].append({"link": links[(i, j)], **_sub(["--pair", str(i), str(j), "--mib", str(a.mib), "--iters", str(a.iters)], a.timeout_s)})
    nv = [p for p, v in sorted(links.items()) if v.startswith("NV")]
    other = [p for p, v in sorted(links.items()) if not v.startswith("NV")]
    if nv:
        report["allreduce"].append({"link": links[nv[0]], **_sub(["--allreduce", str(nv[0][0]), str(nv[0][1]), "--mib", str(a.mib)], 3 * a.timeout_s)})
    if other:
        env = dict(os.environ, NCCL_P2P_DISABLE="1")
        report["allreduce"].append({"link": links[other[0]], **_sub(["--allreduce", str(other[0][0]), str(other[0][1]), "--mib", str(a.mib), "--no-p2p"],
                                                                    3 * a.timeout_s, env=env)})
    report["gpus"] = gpus
    text = json.dumps(report, indent=1)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
