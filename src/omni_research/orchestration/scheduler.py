"""Hardware-aware experiment scheduler: DAG + resources + deadlines + retries + checkpoints.

Policy, in order:
1. An experiment becomes READY when all its dependencies SUCCEEDED, and BLOCKED as soon as one
   of them ends FAILED/BLOCKED/SKIPPED.
2. READY experiments are considered by priority (higher first), then plan order.
3. The highest-priority READY experiment that does not fit right now gets a reservation: lower-
   priority experiments may start only if they are expected to finish before the reservation
   could start (EASY backfill on estimated durations), so a large experiment is not starved.
4. Nothing starts unless its estimated duration fits in the remaining walltime minus the
   shutdown margin; such experiments are SKIPPED with reason "insufficient_time".
5. A failed attempt is retried only if attempts <= max_retries AND its failure class is in
   ``retry_on`` (transient by default). Deterministic failures are never retried unchanged.
6. When the remaining walltime reaches the stop margin (or a stop is requested), nothing new
   starts, running steps get SIGTERM and a grace period, and state is checkpointed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .launcher import Assignment, Handle, Launcher
from .plan import TERMINAL, Experiment, Plan, load_additions
from .resources import Allocation
from .state import ExperimentState, RunState, StateStore


@dataclass
class _Free:
    gpus: int
    cpus: int
    mem_gb: float


class Orchestrator:
    def __init__(
        self,
        plan: Plan,
        store: StateStore,
        alloc: Allocation,
        launcher: Launcher,
        *,
        time_left_fn: Callable[[], float],
        stop_margin_s: float = 900.0,
        poll_s: float = 2.0,
        grace_s: float = 60.0,
        port_base: int = 21000,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        additions_dir: str | Path | None = None,
        env: dict[str, str] | None = None,
        experiments_root: str | Path | None = None,
    ) -> None:
        self.plan = plan
        self.store = store
        self.alloc = alloc
        self.launcher = launcher
        self.time_left = time_left_fn
        self.stop_margin_s = stop_margin_s
        self.poll_s = poll_s
        self.grace_s = grace_s
        self.port_base = port_base
        self.clock = clock
        self.sleep = sleep
        self.additions_dir = Path(additions_dir) if additions_dir else store.run_dir / "plan.d"
        self.experiments_root = Path(experiments_root) if experiments_root else store.run_dir / "experiments"
        self.env = env or {}
        self.handles: dict[str, Handle] = {}
        self._seen_additions: set[str] = set()
        self._stop_reason: str | None = None
        self._ports_in_use: set[int] = set()
        self._port_counter = 0
        self._local_gpus: dict[str, set[int]] = {n.name: set(range(n.gpus)) for n in alloc.nodes}
        self.state = self._init_state()

    # -- state ----------------------------------------------------------------------------------
    def _init_state(self) -> RunState:
        prev = self.store.load()
        state = prev or RunState(plan_id=self.plan.plan_id, run_dir=str(self.store.run_dir))
        state.allocation = self.alloc.to_dict()
        for eid in self.plan.order:
            state.experiments.setdefault(eid, ExperimentState())
        for eid, es in state.experiments.items():
            if es.status == "RUNNING":  # left over from a previous orchestrator that died
                self._record_failure(eid, es, "interrupted", "orchestrator restarted while running", None)
            elif es.status == "READY":
                es.status = "PENDING"
        self.store.save(state)
        if prev:
            self.store.event("resume", previous_statuses=self.status_counts(state))
        return state

    @staticmethod
    def status_counts(state: RunState) -> dict[str, int]:
        out: dict[str, int] = {}
        for es in state.experiments.values():
            out[es.status] = out.get(es.status, 0) + 1
        return out

    def request_stop(self, reason: str) -> None:
        if not self._stop_reason:
            self._stop_reason = reason
            self.store.event("stop_requested", reason=reason)

    # -- resources ------------------------------------------------------------------------------
    def _free(self) -> dict[str, _Free]:
        free = {n.name: _Free(n.gpus, n.cpus, n.mem_gb) for n in self.alloc.nodes}
        for h in self.handles.values():
            f = free[h.assignment.node]
            f.gpus -= h.assignment.gpus
            f.cpus -= h.assignment.cpus
            f.mem_gb -= h.assignment.mem_gb
        return free

    def _fits_allocation(self, e: Experiment) -> bool:
        return any(e.required_gpus <= n.gpus and e.required_cpus <= n.cpus and e.required_memory_gb <= n.mem_gb
                   for n in self.alloc.nodes)

    def _node_for(self, e: Experiment, free: dict[str, _Free]) -> str | None:
        # Best fit: the node with the fewest free GPUs that still fits, to keep big holes open.
        cands = [(f.gpus, name) for name, f in free.items()
                 if e.required_gpus <= f.gpus and e.required_cpus <= f.cpus and e.required_memory_gb <= f.mem_gb]
        return min(cands)[1] if cands else None

    def _shadow_time(self, e: Experiment, now: float) -> float:
        """Earliest time ``e`` could start if running experiments end at their estimates."""
        events = sorted(
            ((h.started_mono + self.plan.experiments[h.experiment_id].estimated_duration_s, h) for h in self.handles.values()),
            key=lambda x: x[0],
        )
        free = self._free()
        if self._node_for(e, free):
            return now
        for t, h in events:
            f = free[h.assignment.node]
            f.gpus += h.assignment.gpus
            f.cpus += h.assignment.cpus
            f.mem_gb += h.assignment.mem_gb
            if self._node_for(e, free):
                return max(t, now)
        return float("inf")

    def _next_port(self) -> int:
        """Ports rotate (base, base+10, ...) instead of being reused right away, so a new server
        never lands on a port whose previous owner is still shutting down. Each experiment may
        use port..port+9 (e.g. several replicas)."""
        for _ in range(1000):
            p = self.port_base + 10 * (self._port_counter % 1000)
            self._port_counter += 1
            if p not in self._ports_in_use:
                self._ports_in_use.add(p)
                return p
        raise RuntimeError("no free port block")

    # -- transitions ----------------------------------------------------------------------------
    def _record_failure(self, eid: str, es: ExperimentState, klass: str, detail: str, rc: int | None) -> None:
        e = self.plan.experiments.get(eid)
        es.failures.append({"attempt": es.attempts, "class": klass, "detail": detail[:2000], "exit_code": rc, "t": time.time()})
        es.exit_code = rc
        retry = e is not None and es.attempts <= e.max_retries and klass in e.retry_on
        es.status = "PENDING" if retry else "FAILED"
        es.reason = f"{klass}: {detail[:300]}" + (" (will retry)" if retry else "")
        self.store.event("experiment_failed", experiment_id=eid, attempt=es.attempts, failure_class=klass,
                         retry=retry, exit_code=rc)

    def _update_ready(self) -> None:
        st = self.state.experiments
        changed = True
        while changed:
            changed = False
            for eid in self.plan.order:
                es = st[eid]
                if es.status != "PENDING":
                    continue
                deps = self.plan.experiments[eid].dependencies
                bad = [d for d in deps if st[d].status in ("FAILED", "BLOCKED", "SKIPPED")]
                if bad:
                    es.status = "BLOCKED"
                    es.reason = f"dependency {bad[0]} is {st[bad[0]].status}"
                    self.store.event("experiment_blocked", experiment_id=eid, reason=es.reason)
                    changed = True
                elif all(st[d].status == "SUCCEEDED" for d in deps):
                    es.status = "READY"
                    changed = True

    def _launch(self, e: Experiment, node: str) -> None:
        es = self.state.experiments[e.experiment_id]
        es.attempts += 1
        a = Assignment(node=node, gpus=e.required_gpus, cpus=e.required_cpus, mem_gb=e.required_memory_gb,
                       port=self._next_port())
        if self.alloc.mode == "local" and e.required_gpus:
            free_idx = sorted(self._local_gpus[node])[: e.required_gpus]
            self._local_gpus[node] -= set(free_idx)
            a.gpu_indices = free_idx
        out_dir = self._new_experiment_dir(e)
        h = self.launcher.launch(e, a, out_dir, self.env)
        self.handles[e.experiment_id] = h
        es.status = "RUNNING"
        es.started_at = time.time()
        es.ended_at = None
        es.assigned = a.to_dict() | {"step_name": h.step_name}
        es.output_dir = str(out_dir)
        self.store.event("experiment_started", experiment_id=e.experiment_id, attempt=es.attempts, assigned=es.assigned)

    def _new_experiment_dir(self, e: Experiment) -> Path:
        """``<experiments_root>/<YYYYMMDD>_<name>_runNN``: every attempt is a new run id, and an
        existing directory is never reused, so a retry cannot overwrite earlier data."""
        name = e.output_path or e.experiment_id
        stamp = time.strftime("%Y%m%d")
        n = 1
        while True:
            d = self.experiments_root / f"{stamp}_{name}_run{n:02d}"
            if not d.exists():
                d.mkdir(parents=True)
                return d
            n += 1

    def _release(self, eid: str) -> Handle:
        h = self.handles.pop(eid)
        self._ports_in_use.discard(h.assignment.port)
        if self.alloc.mode == "local":
            self._local_gpus[h.assignment.node] |= set(h.assignment.gpu_indices)
        es = self.state.experiments[eid]
        wall = self.clock() - h.started_mono
        es.gpu_seconds += h.assignment.gpus * wall
        es.ended_at = time.time()
        return h

    def _finish(self, eid: str, rc: int) -> None:
        h = self._release(eid)
        es = self.state.experiments[eid]
        es.exit_code = rc
        result = _read_result(h.out_dir)
        if rc == 0 and (result is None or result.get("status") in ("ok", "succeeded", "success")):
            es.status = "SUCCEEDED"
            es.reason = None
            self.store.event("experiment_succeeded", experiment_id=eid, attempt=es.attempts)
            return
        klass = (result or {}).get("error_class") or ("nonzero_exit" if rc != 0 else "bad_output")
        detail = (result or {}).get("error") or f"exit code {rc}"
        self._record_failure(eid, es, klass, detail, rc)

    def _poll_running(self) -> None:
        now = self.clock()
        for eid in list(self.handles):
            h = self.handles[eid]
            rc = self.launcher.poll(h)
            if rc is not None:
                self._finish(eid, rc)
                continue
            e = self.plan.experiments[eid]
            if now - h.started_mono > e.timeout_s:
                self.store.event("experiment_timeout", experiment_id=eid, timeout_s=e.timeout_s)
                self.launcher.terminate(h, self.grace_s)
                self._release(eid)
                es = self.state.experiments[eid]
                self._record_failure(eid, es, "timeout", f"exceeded timeout_s={e.timeout_s}", None)

    def _schedule(self) -> None:
        now = self.clock()
        left = self.time_left()
        ready = [eid for eid in self.plan.order if self.state.experiments[eid].status == "READY"]
        ready.sort(key=lambda eid: (-self.plan.experiments[eid].priority, self.plan.order.index(eid)))
        reservation: float | None = None
        for eid in ready:
            e = self.plan.experiments[eid]
            es = self.state.experiments[eid]
            if not self._fits_allocation(e):
                es.status, es.reason = "SKIPPED", "requires more resources than the allocation has"
                self.store.event("experiment_skipped", experiment_id=eid, reason=es.reason)
                continue
            if e.estimated_duration_s > left - self.stop_margin_s:
                es.status, es.reason = "SKIPPED", f"insufficient_time: needs ~{e.estimated_duration_s:.0f}s, {left - self.stop_margin_s:.0f}s usable"
                self.store.event("experiment_skipped", experiment_id=eid, reason=es.reason)
                continue
            if reservation is not None and now + e.estimated_duration_s > reservation:
                continue  # would delay the reserved higher-priority experiment
            node = self._node_for(e, self._free())
            if node:
                self._launch(e, node)
            elif reservation is None:
                reservation = self._shadow_time(e, now)
                self.store.event("reservation", experiment_id=eid, starts_in_s=reservation - now)

    def _shutdown(self, reason: str) -> None:
        self.state.shutdown_reason = reason
        self.store.event("shutdown", reason=reason, running=list(self.handles))
        for eid in list(self.handles):
            h = self.handles[eid]
            rc = self.launcher.terminate(h, self.grace_s)
            self._release(eid)
            es = self.state.experiments[eid]
            if rc == 0 and _read_result(h.out_dir) is not None:
                es.status = "SUCCEEDED"
            else:
                self._record_failure(eid, es, "interrupted", f"stopped at shutdown ({reason})", rc)
        self.store.save(self.state)

    # -- main loop ------------------------------------------------------------------------------
    def step(self) -> bool:
        """One scheduling round. Returns False when the run is finished."""
        try:
            added = load_additions(self.plan, self.additions_dir, self._seen_additions)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            self.store.event("plan_addition_rejected", error=str(exc))
            added = []
        for eid in added:
            self.state.experiments.setdefault(eid, ExperimentState())
            self.store.event("experiment_added", experiment_id=eid)
        self._poll_running()
        if self._stop_reason is None and self.time_left() <= self.stop_margin_s:
            self.request_stop("walltime margin reached")
        if self._stop_reason is not None:
            self._shutdown(self._stop_reason)
            return False
        self._update_ready()
        self._schedule()
        self.store.save(self.state)
        active = any(es.status in ("RUNNING",) for es in self.state.experiments.values())
        pending = [eid for eid, es in self.state.experiments.items() if es.status in ("PENDING", "READY")]
        if not active and pending:
            # Nothing runs and nothing could start: these will never become runnable.
            for eid in pending:
                es = self.state.experiments[eid]
                es.status, es.reason = "SKIPPED", es.reason or "could not be scheduled"
                self.store.event("experiment_skipped", experiment_id=eid, reason=es.reason)
            self.store.save(self.state)
            return False
        return active or bool(pending)

    def run(self) -> RunState:
        self.store.event("run_started", plan_id=self.plan.plan_id, allocation=self.alloc.to_dict(),
                         experiments=len(self.plan.order))
        try:
            while self.step():
                self.sleep(self.poll_s)
        finally:
            if self.handles:
                self._shutdown(self._stop_reason or "orchestrator exiting")
            self.store.save(self.state)
            self.store.event("run_finished", statuses=self.status_counts(self.state),
                             gpu_hours=sum(es.gpu_seconds for es in self.state.experiments.values()) / 3600)
        return self.state


def _read_result(out_dir: Path) -> dict[str, Any] | None:
    p = Path(out_dir) / "result.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {"status": "failed", "error_class": "bad_output", "error": "unreadable result.json"}
