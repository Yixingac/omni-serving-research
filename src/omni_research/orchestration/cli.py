"""Command line for the experiment orchestrator.

    python -m omni_research.orchestration.cli run    --plan PLAN --run-dir DIR [--jobid ID]
    python -m omni_research.orchestration.cli status --run-dir DIR
    python -m omni_research.orchestration.cli check  --plan PLAN          # validate a plan

``run`` inside a Slurm batch job uses that job; with ``--jobid`` it targets an existing
allocation from outside (e.g. a login node), launching every experiment as a step of it.
Without Slurm it runs locally with ``--local-gpus/--local-cpus/--local-mem-gb``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

from .launcher import LocalLauncher, SlurmStepLauncher
from .plan import load_plan
from .resources import discover_allocation, make_time_left_fn
from .scheduler import Orchestrator
from .state import StateStore


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def cmd_run(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    store = StateStore(args.run_dir)
    alloc = discover_allocation(args.jobid, args.local_gpus, args.local_cpus, args.local_mem_gb)
    if args.max_gpus is not None:
        for n in alloc.nodes:
            n.gpus = min(n.gpus, args.max_gpus)
    if alloc.mode == "slurm":
        gpu_type = next((n.gpu_type for n in alloc.nodes if n.gpu_type), None)
        launcher = SlurmStepLauncher(alloc.job_id, args.cwd or _repo_root(), gpu_type)
    else:
        launcher = LocalLauncher(args.cwd or _repo_root())
    # Experiment data lives under PROJECT_DATA_ROOT (persistent storage), separate from code.
    data_root = Path(os.path.expanduser(args.data_root or os.environ.get("PROJECT_DATA_ROOT") or _repo_root() / "results")).resolve()
    deadline = None
    if args.deadline_s:
        import time

        deadline = time.time() + args.deadline_s
    orch = Orchestrator(
        plan,
        store,
        alloc,
        launcher,
        time_left_fn=make_time_left_fn(alloc, deadline),
        stop_margin_s=args.stop_margin_s,
        poll_s=args.poll_s,
        grace_s=args.grace_s,
        port_base=args.port_base,
        env={"OMNI_RUN_DIR": str(Path(args.run_dir).resolve()), "OMNI_REPO": str(_repo_root()),
             "PROJECT_DATA_ROOT": str(data_root)},
        experiments_root=data_root / "experiments",
    )

    def handle(signum, _frame):  # noqa: ANN001
        orch.request_stop(f"signal {signal.Signals(signum).name}")

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(s, handle)
    stop_file = Path(args.run_dir) / "STOP"
    original_step = orch.step

    def step_with_stop_file() -> bool:
        if stop_file.exists():
            orch.request_stop("STOP file present")
        return original_step()

    orch.step = step_with_stop_file  # type: ignore[method-assign]
    print(f"orchestrator: plan={plan.plan_id} experiments={len(plan.order)} allocation={alloc.to_dict()}", flush=True)
    state = orch.run()
    counts = Orchestrator.status_counts(state)
    print(f"orchestrator: finished {counts} shutdown={state.shutdown_reason}", flush=True)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    store = StateStore(args.run_dir)
    st = store.load()
    if st is None:
        print(f"no state in {args.run_dir}")
        return 1
    rows = []
    for eid, es in st.experiments.items():
        rows.append((eid, es.status, es.attempts, round(es.gpu_seconds / 3600, 3), (es.reason or "")[:80]))
    if args.json:
        print(json.dumps({"plan_id": st.plan_id, "experiments": [dict(zip(("id", "status", "attempts", "gpu_h", "reason"), r)) for r in rows]}, indent=2))
        return 0
    w = max((len(r[0]) for r in rows), default=10)
    print(f"plan {st.plan_id}  shutdown={st.shutdown_reason}")
    for r in rows:
        print(f"  {r[0]:<{w}}  {r[1]:<9}  attempts={r[2]}  gpu_h={r[3]:<6}  {r[4]}")
    total = sum(r[3] for r in rows)
    print(f"  total GPU-hours used by experiments: {total:.2f}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    print(f"plan {plan.plan_id}: {len(plan.order)} experiments, DAG ok")
    for eid in plan.order:
        e = plan.experiments[eid]
        print(f"  {eid}: prio={e.priority} gpus={e.required_gpus} cpus={e.required_cpus} mem={e.required_memory_gb}G "
              f"est={e.estimated_duration_s:.0f}s deps={e.dependencies}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="omni_research.orchestration")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--plan", required=True)
    r.add_argument("--run-dir", required=True)
    r.add_argument("--jobid", default=os.environ.get("SLURM_JOB_ID"))
    r.add_argument("--cwd", default=None)
    r.add_argument("--data-root", default=None, help="PROJECT_DATA_ROOT (default: env or <repo>/results)")
    r.add_argument("--stop-margin-s", type=float, default=900)
    r.add_argument("--poll-s", type=float, default=5)
    r.add_argument("--grace-s", type=float, default=60)
    r.add_argument("--port-base", type=int, default=21000)
    r.add_argument("--deadline-s", type=float, default=None, help="extra wall-clock budget from now")
    r.add_argument("--max-gpus", type=int, default=None, help="use at most this many GPUs per node")
    # Local mode only (no Slurm job): the resources to schedule against. The env defaults let
    # run_all.sh be exercised on a CPU-only host, e.g. OMNI_LOCAL_GPUS=3 with OMNI_FAKE_SERVER=1.
    r.add_argument("--local-gpus", type=int, default=int(os.environ.get("OMNI_LOCAL_GPUS", 0)))
    r.add_argument("--local-cpus", type=int, default=int(os.environ["OMNI_LOCAL_CPUS"]) if os.environ.get("OMNI_LOCAL_CPUS") else None)
    r.add_argument("--local-mem-gb", type=float, default=float(os.environ["OMNI_LOCAL_MEM_GB"]) if os.environ.get("OMNI_LOCAL_MEM_GB") else None)
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("status")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_status)
    c = sub.add_parser("check")
    c.add_argument("--plan", required=True)
    c.set_defaults(fn=cmd_check)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
