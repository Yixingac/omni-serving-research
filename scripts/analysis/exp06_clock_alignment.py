#!/usr/bin/env python3
"""Offset between the patch's wall-clock events and the torch trace clock: per code2wav batch in a profiler
window, download_end_ns (taken right after .cpu() returns) minus the end of that batch's DtoH copy call."""
import glob, gzip, json, re, sys
import numpy as np
diffs = []
for run in sys.argv[1:]:
    for ph in glob.glob(f"{run}/profiling/torch/w3b_c*_C"):
        phase = ph.rsplit("/", 1)[1]
        for f in glob.glob(f"{ph}/s0/code2wav@r*_trace_pid*.gz"):
            pid = re.search(r"_pid(\d+)_", f).group(1)
            ev_files = glob.glob(f"{run}/raw/server_events/{phase}/s0/events_code2wav@r*_{pid}.jsonl")
            if not ev_files:
                continue
            tr = json.load(gzip.open(f, "rt"))
            base = int(tr["baseTimeNanoseconds"])
            ev = [e for e in tr["traceEvents"] if e.get("ph") == "X"]
            d2h_corr = {(e.get("args") or {}).get("correlation") for e in ev if e.get("cat") == "gpu_memcpy" and "DtoH" in e.get("name", "")}
            ends = np.array(sorted(base + (e["ts"] + e.get("dur", 0)) * 1000 for e in ev
                                   if e.get("cat") == "cuda_runtime" and (e.get("args") or {}).get("correlation") in d2h_corr))
            if ends.size == 0:
                continue
            seen = set()
            for line in open(ev_files[0]):
                if "exp_vocode" not in line:
                    continue
                m = json.loads(line)["metadata"]
                if m["batch_id"] in seen:
                    continue
                seen.add(m["batch_id"])
                t = m["download_end_ns"]
                i = np.searchsorted(ends, t)
                cands = [ends[j] for j in (i - 1, i) if 0 <= j < ends.size]
                near = min(cands, key=lambda x: abs(t - x))
                if abs(t - near) < 20e6:
                    diffs.append((t - near) / 1e3)
d = np.array(diffs)
print(json.dumps({"batches": int(d.size), "median_us": float(np.median(d)), "p05_us": float(np.percentile(d, 5)),
                  "p95_us": float(np.percentile(d, 95)), "min_us": float(d.min()), "max_us": float(d.max())}))
