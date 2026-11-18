"""Persistent orchestrator state: atomic JSON checkpoints plus an append-only event log."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def atomic_write_json(path: str | Path, obj: Any) -> None:
    """Write JSON so readers see either the old or the new file, never a partial one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


@dataclass
class ExperimentState:
    status: str = "PENDING"
    attempts: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)  # one entry per failed attempt
    started_at: float | None = None
    ended_at: float | None = None
    assigned: dict[str, Any] | None = None  # node, gpu count, cpus, mem, port, step id
    exit_code: int | None = None
    reason: str | None = None  # why BLOCKED/SKIPPED/FAILED
    output_dir: str | None = None
    gpu_seconds: float = 0.0  # allocated GPUs x wall time, summed over attempts


@dataclass
class RunState:
    plan_id: str
    run_dir: str
    created_at: float = field(default_factory=time.time)
    experiments: dict[str, ExperimentState] = field(default_factory=dict)
    allocation: dict[str, Any] = field(default_factory=dict)
    shutdown_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "run_dir": self.run_dir,
            "created_at": self.created_at,
            "allocation": self.allocation,
            "shutdown_reason": self.shutdown_reason,
            "experiments": {k: asdict(v) for k, v in self.experiments.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunState":
        rs = cls(plan_id=d["plan_id"], run_dir=d["run_dir"], created_at=d.get("created_at", time.time()))
        rs.allocation = d.get("allocation", {})
        rs.shutdown_reason = d.get("shutdown_reason")
        for k, v in d.get("experiments", {}).items():
            rs.experiments[k] = ExperimentState(**v)
        return rs


class StateStore:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "state.json"
        self.events_path = self.run_dir / "events.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> RunState | None:
        if not self.path.exists():
            return None
        return RunState.from_dict(json.loads(self.path.read_text()))

    def save(self, state: RunState) -> None:
        atomic_write_json(self.path, state.to_dict())

    def event(self, kind: str, **fields: Any) -> None:
        rec = {"t": time.time(), "event": kind, **fields}
        with open(self.events_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
