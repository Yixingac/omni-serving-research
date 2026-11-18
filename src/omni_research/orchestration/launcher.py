"""Launch experiments as Slurm job steps (or local processes) and stop them again.

Slurm mode runs every experiment as its own ``srun --exact`` step with an explicit GPU, CPU
and memory request, so Slurm (not this code) decides which devices the step sees and
remaps ``CUDA_VISIBLE_DEVICES`` accordingly. A CPU-only step asks for ``--gres=none``,
because a step without a GRES request would otherwise inherit all of the job's GPUs.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .plan import Experiment


@dataclass
class Assignment:
    node: str
    gpus: int
    cpus: int
    mem_gb: float
    port: int
    gpu_indices: list[int] = field(default_factory=list)  # local mode only; Slurm picks in slurm mode

    def to_dict(self) -> dict[str, Any]:
        return {"node": self.node, "gpus": self.gpus, "cpus": self.cpus, "mem_gb": self.mem_gb,
                "port": self.port, "gpu_indices": self.gpu_indices}


@dataclass
class Handle:
    experiment_id: str
    proc: subprocess.Popen
    started_mono: float
    out_dir: Path
    assignment: Assignment
    step_name: str
    log_fh: Any = None


class Launcher(Protocol):
    def launch(self, exp: Experiment, a: Assignment, out_dir: Path, env: dict[str, str]) -> Handle: ...
    def poll(self, h: Handle) -> int | None: ...
    def terminate(self, h: Handle, grace_s: float = 60.0) -> int | None: ...


def _experiment_env(exp: Experiment, a: Assignment, out_dir: Path, extra: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    env.update(extra)
    env.update(exp.env)
    env.update(
        OMNI_EXP_ID=exp.experiment_id,
        OMNI_EXP_DIR=str(out_dir),
        OMNI_PORT=str(a.port),
        OMNI_GPUS=str(a.gpus),
        OMNI_CPUS=str(a.cpus),
        OMNI_TIMEOUT_S=str(int(exp.timeout_s)),
    )
    return env


class LocalLauncher:
    """Run experiments as local process groups with explicit CUDA_VISIBLE_DEVICES."""

    def __init__(self, cwd: str | Path) -> None:
        self.cwd = str(cwd)

    def launch(self, exp: Experiment, a: Assignment, out_dir: Path, env: dict[str, str]) -> Handle:
        out_dir.mkdir(parents=True, exist_ok=True)
        e = _experiment_env(exp, a, out_dir, env)
        e["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in a.gpu_indices)
        fh = open(out_dir / "step.log", "ab")
        proc = subprocess.Popen(["bash", "-c", exp.command], cwd=self.cwd, env=e, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
        return Handle(exp.experiment_id, proc, time.monotonic(), out_dir, a, step_name=f"local:{proc.pid}", log_fh=fh)

    def poll(self, h: Handle) -> int | None:
        rc = h.proc.poll()
        if rc is not None and h.log_fh:
            h.log_fh.close()
            h.log_fh = None
        return rc

    def terminate(self, h: Handle, grace_s: float = 60.0) -> int | None:
        if h.proc.poll() is None:
            try:
                os.killpg(h.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                h.proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(h.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                h.proc.wait(timeout=30)
        return self.poll(h)


class SlurmStepLauncher:
    """Run each experiment as one `srun --exact` step inside job ``job_id``."""

    def __init__(self, job_id: str, cwd: str | Path, gpu_type: str | None = None) -> None:
        self.job_id = str(job_id)
        self.cwd = str(cwd)
        self.gpu_type = gpu_type

    def srun_argv(self, exp: Experiment, a: Assignment, out_dir: Path) -> list[str]:
        gres = "none" if a.gpus == 0 else (f"gpu:{self.gpu_type}:{a.gpus}" if self.gpu_type else f"gpu:{a.gpus}")
        return [
            "srun",
            f"--jobid={self.job_id}",
            "--nodes=1",
            "--ntasks=1",
            f"--nodelist={a.node}",
            "--exact",
            f"--gres={gres}",
            f"--cpus-per-task={a.cpus}",
            f"--mem={int(a.mem_gb * 1024)}M",
            f"--job-name=omni:{exp.experiment_id}",
            "--kill-on-bad-exit=1",
            f"--output={out_dir / 'step.log'}",
            "bash",
            "-c",
            exp.command,
        ]

    def launch(self, exp: Experiment, a: Assignment, out_dir: Path, env: dict[str, str]) -> Handle:
        out_dir.mkdir(parents=True, exist_ok=True)
        argv = self.srun_argv(exp, a, out_dir)
        (out_dir / "srun_argv.txt").write_text(" ".join(argv[:-1]) + " <command>\n" + exp.command + "\n")
        fh = open(out_dir / "srun.log", "ab")  # srun's own messages; the step's output goes to step.log
        proc = subprocess.Popen(argv, cwd=self.cwd, env=_experiment_env(exp, a, out_dir, env), stdout=fh,
                                stderr=subprocess.STDOUT, start_new_session=True)
        return Handle(exp.experiment_id, proc, time.monotonic(), out_dir, a, step_name=f"omni:{exp.experiment_id}", log_fh=fh)

    def poll(self, h: Handle) -> int | None:
        rc = h.proc.poll()
        if rc is not None and h.log_fh:
            h.log_fh.close()
            h.log_fh = None
        return rc

    def _step_ids(self, name: str) -> list[str]:
        out = subprocess.run(["squeue", "-h", "-s", "-j", self.job_id, "-o", "%i|%j"], capture_output=True, text=True,
                             timeout=30, check=False).stdout
        return [ln.split("|", 1)[0] for ln in out.splitlines() if ln.endswith(f"|{name}")]

    def terminate(self, h: Handle, grace_s: float = 60.0) -> int | None:
        if h.proc.poll() is None:
            # srun forwards SIGTERM to the step's tasks; scancel is the fallback if it hangs.
            try:
                os.killpg(h.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                h.proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                for step in self._step_ids(h.step_name):
                    subprocess.run(["scancel", step], check=False, timeout=30)
                try:
                    h.proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(h.proc.pid, signal.SIGKILL)
                    h.proc.wait(timeout=30)
        return self.poll(h)
