#!/usr/bin/env python3
"""实验 8 第三阶段：thinker 的 CPU 在哪些调用上停下来等 GPU。

    python scripts/analysis/exp08_wait_causes.py --data-root <results root> \
        --run 20261117_exp08_proto_perf_block1_off_run01 --run ... --out research/01_性能瓶颈/实验08_Thinker_Lookahead

Descriptive, not part of any registered verdict. In the C-phase torch traces of the thinker processes, a CPU call
"blocks" when it is a synchronize call or a cudaMemcpy* that took over 1 ms. Copies are sorted by the size of the
matching GPU copy (correlation id): small (<= 1 KB, token ids), large (> 1 KB, e.g. a request's hidden states),
unknown (no GPU record). With lookahead on, the thinker waits on the previous step's event while the GPU already
runs the next step, so a sync call there is not GPU idle; a blocking copy drains the queue first. Writes
results/wait_causes.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp05_step_sync as ss  # noqa: E402

BLOCKING_US = 1000
SMALL_BYTES = 1024
KINDS = ("sync", "copy_small", "copy_large", "copy_unknown")
PHASE_RE = re.compile(r"^w3b_ns_c(\d+)_C$")
RUN_RE = re.compile(r"exp08_proto_perf_block(\d+)_(off|on)")


def blocking_calls(trace: dict[str, Any]) -> list[dict[str, Any]]:
    events = [e for e in trace.get("traceEvents", []) if e.get("ph") == "X"]
    gpu_bytes = {e["args"].get("correlation"): e["args"].get("bytes") for e in events if e.get("cat") == "gpu_memcpy" and e.get("args")}
    out = []
    for e in sorted((e for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver")), key=lambda e: e["ts"]):
        if e.get("dur", 0) <= BLOCKING_US:
            continue
        if e["name"] in ss.SYNC:
            out.append({"kind": "sync", "ms": e["dur"] / 1000, "bytes": None})
        elif e["name"].startswith("cudaMemcpy"):
            n = gpu_bytes.get((e.get("args") or {}).get("correlation"))
            kind = "copy_unknown" if n is None else "copy_small" if n <= SMALL_BYTES else "copy_large"
            out.append({"kind": kind, "ms": e["dur"] / 1000, "bytes": n})
    return out


def summarize(calls: list[dict[str, Any]], launches: int) -> dict[str, dict[str, Any]]:
    out = {}
    for kind in KINDS:
        ms = [c["ms"] for c in calls if c["kind"] == kind]
        out[kind] = {"n": len(ms), "total_ms": float(sum(ms)), "per_launch": len(ms) / launches if launches else None,
                     "ms_p50": float(np.median(ms)) if ms else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    rows = []
    for name in a.run:
        m = RUN_RE.search(name)
        for phase_dir in sorted((root / name / "profiling" / "torch").iterdir()):
            pm = PHASE_RE.match(phase_dir.name)
            if not pm:
                continue
            for i, f in enumerate(ss.procs_of(phase_dir, "thinker")):
                tr = ss.load_trace(f)
                launches = sum(1 for e in tr.get("traceEvents", []) if e.get("ph") == "X" and e.get("name") == "cudaGraphLaunch")
                calls = blocking_calls(tr)
                rows.append({"run": name, "block": int(m.group(1)), "arm": m.group(2), "concurrency": int(pm.group(1)), "process": i,
                             "graph_launches": launches, "kinds": summarize(calls, launches),
                             "large_copy_bytes_p50": float(np.median([c["bytes"] for c in calls if c["kind"] == "copy_large"]))
                             if any(c["kind"] == "copy_large" for c in calls) else None})
    by = {}
    for arm in ("off", "on"):
        for c in sorted({r["concurrency"] for r in rows}):
            mine = [r for r in rows if r["arm"] == arm and r["concurrency"] == c]
            launches = sum(r["graph_launches"] for r in mine)
            by[f"{arm}_c{c}"] = {kind: {"per_launch": sum(r["kinds"][kind]["n"] for r in mine) / launches if launches else None,
                                        "blocked_ms_per_launch": sum(r["kinds"][kind]["total_ms"] for r in mine) / launches if launches else None}
                                 for kind in KINDS}
            by[f"{arm}_c{c}"]["graph_launches"] = launches
    out = Path(a.out) / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "wait_causes.json").write_text(json.dumps({"generated_by": "scripts/analysis/exp08_wait_causes.py", "blocking_us": BLOCKING_US,
                                                      "small_bytes": SMALL_BYTES, "per_process": rows, "by_arm_level": by},
                                                     ensure_ascii=False, indent=1) + "\n")
    print(out / "wait_causes.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
