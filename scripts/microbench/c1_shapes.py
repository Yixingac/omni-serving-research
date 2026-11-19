#!/usr/bin/env python3
"""C1 准备：从已有运行的 code2wav 批事件导出真实批形状（每批几行、每行多长），并汇总补齐、注意力形状和 graph 资格。

    # on the cluster, where the raw runs are (pure Python, no numpy/torch):
    python3 scripts/microbench/c1_shapes.py export --data-root <PROJECT_DATA_ROOT> \
        --run 20261118_exp10_graph_block1_run01 ... --out /tmp/c1_batches.jsonl
    # anywhere:
    python3 scripts/microbench/c1_shapes.py summarize --batches /tmp/c1_batches.jsonl \
        --out research/next_steps/results/c1_real_shapes.json --replay-out research/next_steps/results/c1_replay_batches.json

Sources (one exp_vocode event per batch, deduplicated by process and batch id):
- 实验 10 closed loop on t2_k4_c2 (code2wav-bound), phases <w3b|mix>_<setting>, graph and nograph runs;
- 实验 10 validation on t2_k2_c4 (W3b open loop up to 23.5 req/s), phases val_fifo16 / val_length8;
- 实验 9 event phases (w3b_closed_B at 64/128/256, w3l_closed_B at 128) with the stock fifo16 batching.
实验 9 used the instrumented runtime, whose events lack prompt_token_lengths; the prompt length is then derived from
the recorded padded frames (frames / 2 - longest row) and marked as derived.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flow_shapes as fs  # noqa: E402

RUN_RES = [
    (re.compile(r"exp10_(graph|nograph)_block(\d+)"), lambda m: (f"exp10_t2_k4_c2_{m.group(1)}", int(m.group(2)))),
    (re.compile(r"exp10_validation_(t2_k\d_c\d)_\w+?_block(\d+)"), lambda m: (f"exp10val_{m.group(1)}", int(m.group(2)))),
    (re.compile(r"exp09_(t2_k\d_(?:c\d|shared))_block(\d+)"), lambda m: (f"exp09_{m.group(1)}", int(m.group(2)))),
]
PHASE_RES = [
    re.compile(r"^(?P<w>w3b|mix|w3l)_(?P<s>(?:fifo|length)\d+)$"),  # 实验 10 closed loop
    re.compile(r"^val_(?P<s>(?:fifo|length)\d+)$"),  # 实验 10 validation (W3b open loop)
    re.compile(r"^(?P<w>w3b|w3l)_closed_B$"),  # 实验 9 events phases, stock batching
]


def classify_run(run: str) -> tuple[str, int] | None:
    for rx, fn in RUN_RES:
        m = rx.search(run)
        if m:
            return fn(m)
    return None


def classify_phase(phase: str) -> tuple[str, str] | None:
    """(workload, setting) of a measured phase, or None for phases that are not exported."""
    for rx in PHASE_RES:
        m = rx.match(phase)
        if m:
            g = m.groupdict()
            return g.get("w") or "w3b", g.get("s") or "fifo16"
    return None


def vocode_batches(lines: Iterable[str]) -> list[dict[str, Any]]:
    """One record per (pid, batch_id) from exp_vocode events (every request of a batch carries the same metadata)."""
    seen: dict[tuple[Any, Any], dict[str, Any]] = {}
    formed: dict[tuple[Any, Any], dict[str, Any]] = {}
    for line in lines:
        if '"exp_vocode"' not in line and '"exp_batch_formed"' not in line:
            continue
        e = json.loads(line)
        md = e.get("metadata") or {}
        key = (e.get("pid"), md.get("batch_id"))
        if md.get("batch_id") is None:
            continue
        if e.get("event_name") == "exp_batch_formed":
            formed.setdefault(key, {k: md.get(k) for k in ("batch_cap", "batch_policy", "window_size")})
            continue
        if e.get("event_name") != "exp_vocode" or key in seen or not md.get("token_lengths"):
            continue
        toks = [int(t) for t in md["token_lengths"]]
        prompts = md.get("prompt_token_lengths")
        derived = False
        if not prompts and md.get("flow_solve_frames"):
            p = int(md["flow_solve_frames"]) // fs.UP_RATE - max(toks)
            prompts, derived = [p] * len(toks), True
        seen[key] = {"pid": e.get("pid"), "batch_id": md["batch_id"], "rows": len(toks), "token_lengths": toks,
                     "prompt_token_lengths": [int(p) for p in prompts] if prompts else None, "prompt_derived": derived,
                     "flow_solve_path": md.get("flow_solve_path"), "flow_solve_frames": md.get("flow_solve_frames"),
                     "gpu_flow_ms": md.get("gpu_flow_start_to_flow_end_ms"), "gpu_hift_ms": md.get("gpu_flow_end_to_hift_end_ms"),
                     "t_ns": e.get("timestamp_ns")}
    for key, rec in seen.items():
        rec.update(formed.get(key, {}))
    return sorted(seen.values(), key=lambda r: (r["t_ns"] or 0, str(r["pid"]), r["batch_id"]))


def export(data_root: Path, runs: list[str], out: Path) -> int:
    n = 0
    with open(out, "w") as fh:
        for run in runs:
            src = classify_run(run)
            if src is None:
                print(f"skip {run}: unknown run name", file=sys.stderr)
                continue
            source, block = src
            ev_root = data_root / "experiments" / run / "raw" / "server_events"
            if not ev_root.is_dir():
                print(f"skip {run}: no server events", file=sys.stderr)
                continue
            for phase_dir in sorted(p for p in ev_root.iterdir() if p.is_dir()):
                wl = classify_phase(phase_dir.name)
                if wl is None:
                    continue
                files = sorted(phase_dir.glob("*/events_code2wav*.jsonl")) or sorted(phase_dir.glob("*/events_talker*.jsonl"))

                def lines() -> Iterable[str]:
                    for f in files:
                        with open(f, errors="ignore") as h:
                            yield from h

                for rec in vocode_batches(lines()):
                    rec.update(run=run, source=source, block=block, phase=phase_dir.name, workload=wl[0], setting=wl[1])
                    rec.pop("pid", None)
                    fh.write(json.dumps(rec) + "\n")
                    n += 1
            print(f"{run}: done ({n} batches so far)", file=sys.stderr)
    return n


def replay_sets(records: list[dict[str, Any]], per_group: int, seed: int) -> dict[str, Any]:
    """Seeded uniform samples of real batches for the microbenchmark: token lengths only (prompt = default reference)."""
    out: dict[str, Any] = {}
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(fs.group_key(r), []).append(r)
    for k, v in sorted(groups.items()):
        picked = fs.sample_batches(v, per_group, seed)
        out[k] = {"population_batches": len(v), "sampled": len(picked),
                  "batches": [b["token_lengths"] for b in picked]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--data-root", required=True)
    e.add_argument("--run", action="append", required=True)
    e.add_argument("--out", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--batches", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--replay-out")
    s.add_argument("--per-group", type=int, default=200)
    s.add_argument("--seed", type=int, default=20261119)
    a = ap.parse_args()
    if a.cmd == "export":
        n = export(Path(a.data_root).expanduser(), a.run, Path(a.out))
        print(f"wrote {n} batches to {a.out}")
        return 0
    records = [json.loads(x) for x in open(a.batches)]
    prompts = sorted({p for r in records for p in (r.get("prompt_token_lengths") or [])})
    summary = {"generated_by": "scripts/microbench/c1_shapes.py summarize", "source_runs": sorted({r["run"] for r in records}),
               "prompt_token_lengths_seen": prompts, "groups": fs.summarize(records),
               "notes": ["frames = 2 x (prompt + codec tokens); padding ratio is frame-weighted (实验 10's registered definition)",
                         "request_share_by_path: share of requests whose batch would take each flow path at 520b49ee",
                         "recorded_dense_path_reproduced: share of dense-run batches whose recorded path the rules reproduce"]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=1) + "\n")
    if a.replay_out:
        rep = {"generated_by": "scripts/microbench/c1_shapes.py summarize --replay-out", "seed": a.seed,
               "per_group": a.per_group, "prompt_tokens": fs.DEFAULT_PROMPT_TOKENS, "groups": replay_sets(records, a.per_group, a.seed)}
        Path(a.replay_out).write_text(json.dumps(rep) + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
