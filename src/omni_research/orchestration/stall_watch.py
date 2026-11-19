"""Low-overhead pause detector for a running server (P3): watch the shared server log for silence.

Every stage process of a server writes to one log, so a stretch with no new line at all is a pause of the whole
server (实验 9's stall 2: 3.09 s without a line). The watcher stats the log every ``poll_s``; when it has not grown for
more than ``gap_s`` it records, once per pause and at most every ``min_interval_s``:
- each server process's state and wait channel from /proc (R running, S sleeping, D uninterruptible I/O wait);
- the Python stacks of every thread, by sending SIGUSR2 to processes that registered a stack dump (the P1 patch's
  exp_trace "stack" part writes ``stack_<pid>.txt`` with the pid and start time; SIGUSR2 would end any other
  process, so only pids whose current start time matches the registration are signalled);
- optionally a py-spy dump (``py_spy``); with /proc/sys/kernel/yama/ptrace_scope > 0 (the cluster) py-spy cannot
  attach to a sibling process, so the default is off and the reason is recorded.
It also records a stat of the log that itself took longer than ``slow_stat_s`` (a blocked network file system shows
there first) and the end of each pause with its length. Records go to a JSONL file; nothing here touches the GPU.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable


def proc_state(pid: int, proc_root: str = "/proc") -> dict[str, Any]:
    """State letter, wait channel and start time (clock ticks since boot) of one process; empty if it is gone."""
    try:
        stat = Path(proc_root, str(pid), "stat").read_text()
    except OSError:
        return {}
    fields = stat.rsplit(")", 1)[1].split()
    out: dict[str, Any] = {"pid": pid, "state": fields[0], "starttime": int(fields[19])}
    try:
        out["wchan"] = Path(proc_root, str(pid), "wchan").read_text().strip() or "0"
    except OSError:
        out["wchan"] = None
    return out


def registered_stack_pids(stack_dir: Path, proc_root: str = "/proc") -> dict[int, Path]:
    """Pids whose stack file's registration line matches the process now running under that pid."""
    out: dict[int, Path] = {}
    for path in sorted(stack_dir.glob("stack_*.txt")) if stack_dir.is_dir() else []:
        try:
            first = path.read_text(errors="ignore").splitlines()[0]
            fields = dict(part.split("=", 1) for part in first.split()[1:])
            pid, start = int(fields["pid"]), int(fields["starttime"])
        except (OSError, IndexError, KeyError, ValueError):
            continue
        if proc_state(pid, proc_root).get("starttime") == start:
            out[pid] = path
    return out


def tree_pids(root_pids: list[int]) -> list[int]:
    """The given processes and all their descendants (the stage processes of each server)."""
    try:
        import psutil
    except ImportError:
        return list(root_pids)
    out: list[int] = []
    for pid in root_pids:
        try:
            root = psutil.Process(pid)
            out += [root.pid] + [c.pid for c in root.children(recursive=True)]
        except psutil.Error:
            continue
    return sorted(set(out))


def ptrace_scope(path: str = "/proc/sys/kernel/yama/ptrace_scope") -> int | None:
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


class StallWatcher(threading.Thread):
    def __init__(self, log_path: Path, pids_fn: Callable[[], list[int]], out_path: Path, *, stack_dir: Path | None = None,
                 gap_s: float = 1.0, poll_s: float = 0.1, min_interval_s: float = 30.0, max_dumps: int = 20,
                 slow_stat_s: float = 0.2, py_spy: str | None = None, proc_root: str = "/proc",
                 signal_fn: Callable[[int, int], None] = os.kill, clock: Callable[[], float] = time.monotonic,
                 run_fn: Callable[..., Any] = subprocess.run) -> None:
        super().__init__(daemon=True, name="stall-watch")
        self.log_path, self.pids_fn, self.out_path, self.stack_dir = Path(log_path), pids_fn, Path(out_path), stack_dir
        self.gap_s, self.poll_s, self.min_interval_s, self.max_dumps = gap_s, poll_s, min_interval_s, max_dumps
        self.slow_stat_s, self.py_spy, self.proc_root = slow_stat_s, py_spy, proc_root
        self.signal_fn, self.clock, self.run_fn = signal_fn, clock, run_fn
        self._stop = threading.Event()
        self.last_size: int | None = None
        self.last_growth = clock()
        self.in_pause = False
        self.pause_started_wall_ns = 0
        self.last_dump = -1e18
        self.dumps = 0
        self.pauses = 0
        self.records: list[dict[str, Any]] = []
        # The runner arms the watcher only while a load phase runs: between phases (control files, event toggles,
        # idle drains) a silent log is not a pause.
        self.armed = True

    def write(self, rec: dict[str, Any]) -> None:
        self.records.append(rec)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def snapshot(self, silent_s: float) -> dict[str, Any]:
        """Process states, stack dumps (SIGUSR2 to registered pids) and an optional py-spy dump, once per pause."""
        pids = list(self.pids_fn())
        rec: dict[str, Any] = {"event": "pause_snapshot", "t_wall_ns": time.time_ns(), "silent_s": round(silent_s, 3),
                               "processes": [s for s in (proc_state(p, self.proc_root) for p in pids) if s]}
        if self.stack_dir is not None:
            signalled = []
            for pid, path in registered_stack_pids(self.stack_dir, self.proc_root).items():
                if pid in pids:
                    try:
                        self.signal_fn(pid, signal.SIGUSR2)
                        signalled.append({"pid": pid, "file": path.name})
                    except OSError as exc:
                        signalled.append({"pid": pid, "error": str(exc)})
            rec["stack_signals"] = signalled
        if self.py_spy:
            scope = ptrace_scope()
            if scope not in (0, None):
                rec["py_spy"] = f"skipped: ptrace_scope={scope} does not let py-spy attach to a sibling process"
            else:
                dumps = {}
                for pid in pids:
                    try:
                        res = self.run_fn([self.py_spy, "dump", "--pid", str(pid), "--nonblocking"], capture_output=True,
                                          text=True, timeout=10)
                        dumps[str(pid)] = (res.stdout or res.stderr)[-20000:]
                    except (OSError, subprocess.TimeoutExpired) as exc:
                        dumps[str(pid)] = f"error: {exc}"
                rec["py_spy"] = dumps
        return rec

    def poll_once(self) -> None:
        t0 = self.clock()
        try:
            size = self.log_path.stat().st_size
        except OSError:
            size = None
        now = self.clock()
        stat_s = now - t0
        if stat_s > self.slow_stat_s:
            self.write({"event": "slow_stat", "t_wall_ns": time.time_ns(), "stat_s": round(stat_s, 3)})
        if size is None:
            return
        if self.last_size is None or size != self.last_size:
            if self.in_pause:
                self.write({"event": "pause_end", "t_wall_ns": time.time_ns(), "start_wall_ns": self.pause_started_wall_ns,
                            "silent_s": round(now - self.last_growth, 3)})
                self.in_pause = False
            self.last_size, self.last_growth = size, now
            return
        silent = now - self.last_growth
        if silent > self.gap_s and not self.in_pause and self.armed:
            self.in_pause = True
            self.pauses += 1
            self.pause_started_wall_ns = time.time_ns() - int(silent * 1e9)
            rec = {"event": "pause_start", "t_wall_ns": time.time_ns(), "silent_s": round(silent, 3), "pause": self.pauses}
            self.write(rec)
            if self.dumps < self.max_dumps and now - self.last_dump >= self.min_interval_s:
                self.last_dump, self.dumps = now, self.dumps + 1
                self.write(self.snapshot(silent))

    def arm(self) -> None:
        """Start counting silence from now (a load phase begins)."""
        self.last_growth = self.clock()
        self.armed = True

    def disarm(self) -> None:
        self.armed = False

    def run(self) -> None:
        while not self._stop.wait(self.poll_s):
            try:
                self.poll_once()
            except Exception as exc:  # noqa: BLE001 - the watcher must never take the run down
                try:
                    self.write({"event": "watcher_error", "t_wall_ns": time.time_ns(), "error": f"{type(exc).__name__}: {exc}"[:300]})
                except OSError:
                    pass

    def stop(self) -> None:
        self._stop.set()
