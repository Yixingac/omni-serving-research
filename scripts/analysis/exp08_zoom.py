#!/usr/bin/env python3
"""A few consecutive thinker decode steps (GPU kernels and CPU CUDA calls) for 实验 8 figure 1.

    python scripts/analysis/exp08_zoom.py --data-root <results root> --run 20261117_exp08_lookahead_block1_run01 \
        --phase w3t_c8_C --phase w3t_ns_c8_C --phase w3b_ns_c8_C --out <dir>/results/zoom.json

Runs where the raw traces are (the cluster). For each phase, the first thinker process trace, cut to the steps
around the middle of the window as exp05_step_sync.zoom does. Times are milliseconds from the cut start. CPU calls
are sync, launch, copy_wait (a device-to-host copy that blocked over 1 ms: in the synchronous speech path the
token-id copy is where the CPU waits for the step) or api.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp05_step_sync as ss  # noqa: E402

COPY_WAIT_US = 1000


def zoom(tr: dict, n_steps: int) -> dict:
    ev = tr.get("traceEvents", [])
    rt = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver")), key=lambda e: e["ts"])
    launches = [e for e in rt if e["name"] == "cudaGraphLaunch"]
    if len(launches) < n_steps + 2:
        return {}
    mid = len(launches) // 2
    t0, t1 = launches[mid]["ts"] - 2000, launches[mid + n_steps]["ts"]

    def kind(e: dict) -> str:
        if e["name"] in ss.SYNC:
            return "sync"
        if e["name"] == "cudaGraphLaunch":
            return "launch"
        return "copy_wait" if e["name"].startswith("cudaMemcpy") and e["dur"] > COPY_WAIT_US else "api"

    gpu = [((e["ts"] - t0) / 1e3, e["dur"] / 1e3, bool((e.get("args") or {}).get("graph id", 0)))
           for e in ev if e.get("ph") == "X" and e.get("cat") in ss.DEVICE_CATS and t0 <= e["ts"] <= t1]
    cpu = [((e["ts"] - t0) / 1e3, e["dur"] / 1e3, kind(e)) for e in rt if t0 <= e["ts"] <= t1]
    return {"span_ms": (t1 - t0) / 1e3, "gpu": gpu, "cpu": cpu}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--phase", action="append", required=True)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    exp = Path(a.data_root).expanduser() / "experiments" / a.run
    out = {"run": a.run, "steps": a.steps, "phases": {}}
    for phase in a.phase:
        files = ss.procs_of(exp / "profiling" / "torch" / phase, "thinker")
        if not files:
            out["phases"][phase] = None
            continue
        tr = ss.load_trace(files[0])
        z = zoom(tr, n_steps=a.steps)
        s = ss.summarize(ss.steps_of(tr))
        out["phases"][phase] = {**z, "summary": s}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out) + "\n")
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
