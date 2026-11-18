"""Experiment plans: a dependency-aware DAG of experiments with resource needs.

A plan is JSON (or YAML when PyYAML is importable)::

    {"plan_id": "main", "defaults": {...},
     "experiments": [{"experiment_id": "...", "priority": 50, "required_gpus": 1, ...}]}

Extra experiments can be appended while a run is in progress by dropping JSON files into
``<run_dir>/plan.d/``; existing experiments are never redefined (ids are immutable).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STATUSES = ("PENDING", "READY", "RUNNING", "SUCCEEDED", "FAILED", "BLOCKED", "SKIPPED")
TERMINAL = ("SUCCEEDED", "FAILED", "BLOCKED", "SKIPPED")

# Failure classes produced by run_experiment / the launcher. Only "transient" ones are retried
# by default: retrying a deterministic failure (OOM, unsupported config) without changing
# anything wastes GPU time and is not allowed.
TRANSIENT_FAILURES = ("port_in_use", "server_start_timeout", "interrupted", "node_failure", "client_connection")
DETERMINISTIC_FAILURES = ("oom", "unsupported", "config_error", "assertion", "timeout", "nonzero_exit", "bad_output")


@dataclass
class Experiment:
    experiment_id: str
    command: str
    priority: int = 0
    required_gpus: int = 0
    required_cpus: int = 1
    required_memory_gb: float = 8.0
    required_nodes: int = 1
    estimated_duration_s: float = 600.0
    dependencies: list[str] = field(default_factory=list)
    timeout_s: float = 3600.0
    max_retries: int = 0
    retry_on: list[str] = field(default_factory=lambda: list(TRANSIENT_FAILURES))
    output_path: str = ""
    tags: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Plan:
    plan_id: str
    experiments: dict[str, Experiment]
    order: list[str]  # definition order, used as the final tie-breaker
    source: str = ""

    def add(self, exp: Experiment) -> bool:
        if exp.experiment_id in self.experiments:
            return False
        self.experiments[exp.experiment_id] = exp
        self.order.append(exp.experiment_id)
        return True


def _load_doc(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml  # optional dependency

        return yaml.safe_load(text)
    return json.loads(text)


def experiment_from_dict(d: dict[str, Any], defaults: dict[str, Any] | None = None) -> Experiment:
    merged = dict(defaults or {})
    merged.update(d)
    if "experiment_id" not in merged or "command" not in merged:
        raise ValueError(f"experiment needs experiment_id and command: {d}")
    names = set(Experiment.__dataclass_fields__)
    unknown = set(merged) - names
    if unknown:
        raise ValueError(f"experiment {merged['experiment_id']}: unknown fields {sorted(unknown)}")
    exp = Experiment(**merged)
    if not exp.output_path:
        exp.output_path = exp.experiment_id
    return exp


def load_plan(path: str | Path) -> Plan:
    path = Path(path)
    doc = _load_doc(path)
    defaults = doc.get("defaults", {})
    exps: dict[str, Experiment] = {}
    order: list[str] = []
    for d in doc.get("experiments", []):
        e = experiment_from_dict(d, defaults)
        if e.experiment_id in exps:
            raise ValueError(f"duplicate experiment_id {e.experiment_id}")
        exps[e.experiment_id] = e
        order.append(e.experiment_id)
    plan = Plan(plan_id=doc.get("plan_id", path.stem), experiments=exps, order=order, source=str(path))
    validate_plan(plan)
    return plan


def load_additions(plan: Plan, directory: Path, seen: set[str]) -> list[str]:
    """Merge new experiment files from ``directory`` (each a plan-shaped doc). Returns new ids."""
    added: list[str] = []
    if not directory.is_dir():
        return added
    for f in sorted(directory.glob("*.json")) + sorted(directory.glob("*.y*ml")):
        if f.name in seen:
            continue
        doc = _load_doc(f)
        for d in doc.get("experiments", []):
            e = experiment_from_dict(d, doc.get("defaults", {}))
            if plan.add(e):
                added.append(e.experiment_id)
        seen.add(f.name)
    if added:
        validate_plan(plan)
    return added


def validate_plan(plan: Plan) -> None:
    """Unknown dependencies and cycles are plan errors (raised), not runtime surprises."""
    for e in plan.experiments.values():
        for d in e.dependencies:
            if d not in plan.experiments:
                raise ValueError(f"{e.experiment_id} depends on unknown experiment {d}")
        if e.required_nodes != 1:
            raise ValueError(f"{e.experiment_id}: only single-node experiments are supported (required_nodes=1)")
    state: dict[str, int] = {}

    def visit(n: str, path: list[str]) -> None:
        if state.get(n) == 1:
            raise ValueError(f"dependency cycle: {' -> '.join(path + [n])}")
        if state.get(n) == 2:
            return
        state[n] = 1
        for d in plan.experiments[n].dependencies:
            visit(d, path + [n])
        state[n] = 2

    for n in plan.experiments:
        visit(n, [])
