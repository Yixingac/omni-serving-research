"""Rehearse an experiment plan with the CPU-only fake server (plan v2, P6).

    .venv/bin/python scripts/slurm/dryrun_plan.py configs/experiments/plan_a6000_8gpu_v2.json \
        --gpus 8 --cpus 56 --mem-gb 800 [--work DIR]

Every experiment config is shrunk (two short levels, a few requests, one-second open-loop steps, a
handful of agent sessions), the plan is rewritten to use the shrunk copies, and the real orchestrator
runs it on a local allocation of the given size with OMNI_FAKE_SERVER=1. This catches what only shows
at run time in our own code: unknown phase options, a capacity_from that names no earlier phase,
workload mixes, agent options, resource packing, result handling. It does not exercise SGLang-Omni
(check those configs with scripts/spark/check_pipeline_config.py) and its numbers mean nothing.
Commands other than run_experiment.sh (e.g. the GPU topology step) are replaced by `true`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from omni_research.orchestration.plan import load_plan  # noqa: E402
from omni_research.replay.backends import silence_wav  # noqa: E402

RUN_EXP = re.compile(r"^bash scripts/slurm/run_experiment\.sh (\S+\.json)$")


def shrink_phase(ph: dict) -> dict:
    ph = dict(ph)
    kind = ph.get("kind")
    if kind == "closed_loop":
        ph["concurrency"] = [min(int(c), 4) for c in ph.get("concurrency", [1])[:2]]
        ph["requests_per_level"] = 3
        ph.pop("requests_per_concurrency", None)
        ph["warmup"] = 1
    elif kind == "open_loop":
        if "rates_rps" in ph:
            ph["rates_rps"] = ph["rates_rps"][:2]
            ph["requests_per_rate"] = 3
        if "capacity_fractions" in ph:
            ph["capacity_fractions"] = ph["capacity_fractions"][:2]
        if "duration_per_rate_s" in ph:
            ph["duration_per_rate_s"] = 1
            ph["min_requests_per_rate"] = 2
    elif kind == "agent":
        ph["sessions"] = min(int(ph.get("sessions", 4)), 4)
        ph["concurrency"] = min(int(ph.get("concurrency", 2)), 2)
        if ph.get("tool_latency"):
            ph["tool_latency"] = [0.01, 0.5]
        agent = dict(ph.get("agent", {}))
        if agent.get("think_time_s"):
            agent["think_time_s"] = [0.01, 0.1]
        ph["agent"] = agent
    elif kind == "smoke":
        ph["requests"] = min(int(ph.get("requests", 2)), 2)
    return ph


def shrink_config(cfg: dict) -> dict:
    cfg = json.loads(json.dumps(cfg))
    for s in cfg.get("servers", []):
        s["ready_timeout_s"] = 60
    cfg["phases"] = [shrink_phase(p) for p in cfg.get("phases", [])]
    return cfg


def synthetic_pool(data_root: Path, name: str) -> None:
    """A tiny asset pool with the kinds the workloads ask for (silent audio tagged agent_question)."""
    pool = data_root / "assets" / name
    pool.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(3):
        p = pool / f"q{i}.wav"
        p.write_bytes(silence_wav(2.0 + i))
        rows.append({"asset_id": f"dry-q{i}", "kind": "audio", "path": p.name, "duration_s": 2.0 + i,
                     "source": "synthetic:silence", "tags": ["agent_question"], "task_id": f"task-dry-{i}"})
    (pool / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("plan")
    ap.add_argument("--gpus", type=int, required=True)
    ap.add_argument("--cpus", type=int, required=True)
    ap.add_argument("--mem-gb", type=float, required=True)
    ap.add_argument("--work", default=None, help="working directory (default: a new temporary one)")
    a = ap.parse_args()

    load_plan(a.plan)  # the real plan must load and validate as it is
    work = Path(a.work or tempfile.mkdtemp(prefix="dryrun-"))
    (work / "configs").mkdir(parents=True, exist_ok=True)
    doc = json.loads(Path(a.plan).read_text())
    pools = set()
    replaced = []
    for e in doc["experiments"]:
        m = RUN_EXP.match(e["command"])
        if m:
            cfg = json.loads((REPO / m.group(1)).read_text())
            if cfg.get("assets"):
                pools.add(cfg["assets"])
            p = work / "configs" / f"{e['experiment_id']}.json"
            p.write_text(json.dumps(shrink_config(cfg), ensure_ascii=False, indent=1))
            e["command"] = f"bash scripts/slurm/run_experiment.sh {shlex.quote(str(p))}"
        else:
            replaced.append(e["experiment_id"])
            e["command"] = "true"
        e["estimated_duration_s"] = 30
        e["timeout_s"] = 900
        e["max_retries"] = 0
    (work / "plan.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    data = work / "data"
    for name in pools:
        synthetic_pool(data, name)

    env = dict(os.environ, OMNI_FAKE_SERVER="1", PROJECT_DATA_ROOT=str(data), OMNI_REPO=str(REPO),
               PYTHONPATH=str(REPO / "src"), PATH=f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}")
    env.pop("SLURM_JOB_ID", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    cmd = [sys.executable, "-m", "omni_research.orchestration.cli", "run", "--plan", str(work / "plan.json"),
           "--run-dir", str(work / "orchestrator"), "--local-gpus", str(a.gpus), "--local-cpus", str(a.cpus),
           "--local-mem-gb", str(a.mem_gb), "--poll-s", "1", "--port-base", "41000"]
    print(f"dry run in {work} (commands replaced by `true`: {', '.join(replaced) or 'none'})", flush=True)
    with open(work / "orchestrator.log", "w") as fh:
        rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    state = json.loads((work / "orchestrator" / "state.json").read_text())
    exps = state.get("experiments", {})
    bad = {k: v for k, v in exps.items() if v.get("status") != "SUCCEEDED"}
    for k, v in sorted(exps.items(), key=lambda kv: kv[1].get("started_at") or 0):
        print(f"{v.get('status'):<10} {k}" + (f"  {v.get('reason')}" if v.get("status") != "SUCCEEDED" else ""))
    print(f"orchestrator exit {rc}; {len(exps) - len(bad)}/{len(exps)} succeeded")
    return 0 if rc == 0 and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
