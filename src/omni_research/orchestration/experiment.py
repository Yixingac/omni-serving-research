"""Run one experiment inside its allocation (a Slurm step or a local process).

    python -m omni_research.orchestration.experiment CONFIG.json [--out DIR]

Layout written under the experiment directory (``$OMNI_EXP_DIR`` when launched by the
orchestrator)::

    manifest.json        ids, git revisions, hardware, timing, file list
    config.json          the exact configuration that ran
    result.json          status / error_class (read by the orchestrator)
    logs/                server_<name>.log, client.log, env/*.txt
    raw/                 requests_<phase>.jsonl, agent_trace.jsonl, bench_trace.jsonl,
                         server_events/<phase>/*.jsonl, gpu_samples.csv, cpu_samples.jsonl
    profiling/           profiler outputs (only when a phase asks for them)
    analysis/metrics.json, analysis/summary.csv

The runner never touches GPUs outside the step: servers get subsets of the step's own
CUDA_VISIBLE_DEVICES list.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..analysis import metrics as M
from ..replay.backends import BackendPool, OpenAIChatBackend
from ..replay.drivers import JsonlSink, closed_loop, open_loop
from ..replay.payloads import AssetPool, Workload
from ..tracing.collector import TraceCollector
from ..tracing.schema import SourceKind
from .state import atomic_write_json

FAILURE_PATTERNS = [
    ("oom", r"CUDA out of memory|OutOfMemoryError|torch\.OutOfMemoryError|cudaErrorMemoryAllocation|Not enough memory"),
    ("port_in_use", r"Address already in use|address already in use"),
    ("unsupported", r"NotImplementedError|is not supported|Unsupported|no kernel image is available|Invalid backend"),
    ("config_error", r"BadParameter|ValidationError|unrecognized arguments|No such option"),
]


class ExperimentFailure(Exception):
    def __init__(self, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


def classify_log(text: str, default: str) -> str:
    for klass, pat in FAILURE_PATTERNS:
        if re.search(pat, text):
            return klass
    return default


def _tail(path: Path, n: int = 60) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def _http_json(url: str, body: dict[str, Any] | None = None, timeout: float = 10.0) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read() or b"null")
        except (json.JSONDecodeError, ValueError):
            return exc.code, None


def _port_free(port: int) -> bool:
    """True unless something is *listening* on the port. SO_REUSEADDR (as servers set it) lets
    the probe ignore connections from a previous server that are still in TIME_WAIT."""
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


# ------------------------------------------------------------------------------------------
# Samplers


class GpuSampler:
    """nvidia-smi sampling of the step's GPUs (NVML 'utilization.gpu' = fraction of time at least
    one kernel was running in the sample period; it is *not* SM occupancy)."""

    FIELDS = "timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm,pstate"

    def __init__(self, path: Path, interval_ms: int = 200) -> None:
        self.path = path
        self.interval_ms = interval_ms
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        if not shutil.which("nvidia-smi"):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "w")
        fh.write(self.FIELDS + "\n")
        fh.flush()
        self.proc = subprocess.Popen(["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits",
                                      f"-lms={self.interval_ms}"], stdout=fh, stderr=subprocess.DEVNULL)

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class CpuSampler(threading.Thread):
    """Process-tree CPU usage of the servers (and the client itself) once per second."""

    def __init__(self, path: Path, pids_fn, interval_s: float = 1.0) -> None:  # noqa: ANN001
        super().__init__(daemon=True)
        self.path, self.pids_fn, self.interval_s = path, pids_fn, interval_s
        self._stop = threading.Event()

    def run(self) -> None:
        try:
            import psutil
        except ImportError:
            return
        procs: dict[int, Any] = {}
        with open(self.path, "w") as fh:
            psutil.cpu_percent(None)
            while not self._stop.wait(self.interval_s):
                rec: dict[str, Any] = {"t_wall_ns": time.time_ns(), "system_cpu_percent": psutil.cpu_percent(None),
                                       "groups": {}}
                for name, root in self.pids_fn().items():
                    total = 0.0
                    n = 0
                    try:
                        tree = [psutil.Process(root)] + psutil.Process(root).children(recursive=True)
                    except psutil.Error:
                        continue
                    for p in tree:
                        if p.pid not in procs:
                            procs[p.pid] = p
                            try:
                                p.cpu_percent(None)
                            except psutil.Error:
                                pass
                            continue
                        try:
                            total += procs[p.pid].cpu_percent(None)
                            n += 1
                        except psutil.Error:
                            pass
                    rec["groups"][name] = {"cpu_percent": total, "processes": n}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()

    def stop(self) -> None:
        self._stop.set()


# ------------------------------------------------------------------------------------------
# Servers


class Server:
    def __init__(self, spec: dict[str, Any], runtime: dict[str, Any], gpus: list[str], port: int, logs: Path,
                 ready_timeout_s: float) -> None:
        self.spec = spec
        self.runtime = runtime
        self.gpus = gpus
        self.port = port
        self.name = spec.get("name", f"s{port}")
        self.log_path = logs / f"server_{self.name}.log"
        self.ready_timeout_s = ready_timeout_s
        self.proc: subprocess.Popen | None = None
        self.started_wall = None
        self.ready_after_s: float | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def command(self) -> str:
        rt = self.runtime
        if self.spec.get("command"):
            # Explicit server command (used for the CPU-only fake server in tests); {port} is filled in.
            return substitute(str(self.spec["command"]), rt, self.port)
        args = " ".join(_quote(substitute(a, rt, self.port)) for a in self.spec.get("args", []))
        model = _quote(os.path.expanduser(self.spec["model_path"]))
        serve = (f"{rt.get('python', 'python')} -m sglang_omni.cli serve --model-path {model} "
                 f"--host 127.0.0.1 --port {self.port} {args}")
        prefix = self.spec.get("prefix", "")  # e.g. an nsys command line for profiling runs
        env_script = rt.get("env_script")
        parts = []
        if env_script:
            parts.append(f". {_quote(os.path.expanduser(env_script))}")
        if rt.get("cwd"):
            parts.append(f"cd {_quote(os.path.expanduser(rt['cwd']))}")
        parts.append(f"exec {prefix} {serve}".strip())
        return " && ".join(parts)

    def start(self) -> None:
        if not _port_free(self.port):
            raise ExperimentFailure("port_in_use", f"port {self.port} is busy before server {self.name} started")
        env = dict(os.environ)
        env.update({k: os.path.expanduser(str(v)) for k, v in self.runtime.get("env", {}).items()})
        env.update({k: str(v) for k, v in self.spec.get("env", {}).items()})
        env["CUDA_VISIBLE_DEVICES"] = ",".join(self.gpus)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.log_path, "ab")
        fh.write(f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n# {self.command()}\n".encode())
        fh.flush()
        self.started_wall = time.time()
        self.proc = subprocess.Popen(["bash", "-c", self.command()], env=env, stdout=fh, stderr=subprocess.STDOUT,
                                     start_new_session=True)

    def wait_ready(self) -> float:
        t0 = time.monotonic()
        while True:
            if self.proc is None or self.proc.poll() is not None:
                code = None if self.proc is None else self.proc.returncode
                tail = _tail(self.log_path)
                raise ExperimentFailure(classify_log(tail, "server_crashed"), f"server {self.name} exited ({code}) before ready:\n{tail[-1500:]}")
            try:
                status, body = _http_json(f"{self.base_url}/health", timeout=5)
                if status == 200 and isinstance(body, dict) and body.get("status") == "healthy":
                    self.ready_after_s = time.monotonic() - t0
                    return self.ready_after_s
            except (urllib.error.URLError, ConnectionError, OSError, TimeoutError):
                pass
            if time.monotonic() - t0 > self.ready_timeout_s:
                tail = _tail(self.log_path)
                raise ExperimentFailure(classify_log(tail, "server_start_timeout"),
                                        f"server {self.name} not healthy after {self.ready_timeout_s:.0f}s:\n{tail[-1500:]}")
            time.sleep(5)

    def stop(self, grace_s: float = 45.0) -> int | None:
        if self.proc is None or self.proc.poll() is not None:
            return None if self.proc is None else self.proc.returncode
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return self.proc.poll()
        try:
            return self.proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            return self.proc.wait(timeout=30)


def _quote(s: str) -> str:
    import shlex

    return shlex.quote(str(s))


def substitute(s: str, runtime: dict[str, Any], port: int | None = None) -> str:
    """Fill ``{research_repo}``, ``{sglang_omni_repo}`` and ``{port}`` placeholders in config strings."""
    repo = os.environ.get("OMNI_REPO") or str(Path(__file__).resolve().parents[3])
    s = s.replace("{research_repo}", repo)
    s = s.replace("{sglang_omni_repo}", os.path.expanduser(runtime.get("cwd", "")))
    if port is not None:
        s = s.replace("{port}", str(port))
    return s


def stage_model(src: str, scratch_root: str | None) -> str:
    """Copy model weights to node-local scratch once per node (a lock serializes concurrent steps)."""
    src = os.path.expanduser(src)
    if not scratch_root:
        return src
    root = Path(os.path.expanduser(scratch_root))
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        return src
    dst = root / Path(src).name
    done = dst.with_name(dst.name + ".complete")
    if done.exists():
        return str(dst)
    import fcntl

    with open(root / f".{dst.name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not done.exists():
            subprocess.run(["rsync", "-a", "--exclude", ".cache/", f"{src}/", f"{dst}/"], check=True)
            done.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    return str(dst)


# ------------------------------------------------------------------------------------------
# Runner


class ExperimentRunner:
    def __init__(self, cfg: dict[str, Any], out: Path) -> None:
        self.cfg = cfg
        self.out = out
        for d in ("logs/env", "raw/server_events", "profiling", "analysis", "figures"):
            (out / d).mkdir(parents=True, exist_ok=True)
        self.data_root = Path(os.path.expanduser(cfg.get("data_root") or os.environ.get("PROJECT_DATA_ROOT")
                                                 or os.environ.get("OMNI_REPO", ".") + "/results"))
        self.port_base = int(os.environ.get("OMNI_PORT") or cfg.get("port_base", 30000))
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self.step_gpus = [g for g in vis.split(",") if g.strip()] if vis else []
        self.servers: list[Server] = []
        self.phase_summaries: list[dict[str, Any]] = []
        self.stop_requested = False
        self.client_log = open(out / "logs" / "client.log", "a")
        self.t_start = time.time()

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        print(line, flush=True)
        self.client_log.write(line + "\n")
        self.client_log.flush()

    # -- environment fingerprint ------------------------------------------------------------
    def fingerprint(self) -> dict[str, Any]:
        env_dir = self.out / "logs" / "env"
        def cap(name: str, cmd: list[str]) -> str:
            try:
                txt = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
            except (OSError, subprocess.TimeoutExpired) as exc:
                txt = f"error: {exc}"
            (env_dir / f"{name}.txt").write_text(txt)
            return txt
        cap("nvidia_smi_L", ["nvidia-smi", "-L"])
        cap("nvidia_topo", ["nvidia-smi", "topo", "-m"])
        cap("nvidia_query", ["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,memory.total,ecc.mode.current,"
                             "pcie.link.gen.current,pcie.link.width.current,driver_version", "--format=csv"])
        cap("nvlink", ["nvidia-smi", "nvlink", "-s"])
        repo = Path(os.environ.get("OMNI_REPO", Path(__file__).resolve().parents[3]))
        def git(path: Path) -> str | None:
            try:
                return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                                      timeout=30).stdout.strip() or None
            except OSError:
                return None
        rt = self.cfg.get("runtime", {})
        omni_repo = Path(os.path.expanduser(rt.get("cwd", ""))) if rt.get("cwd") else None
        fp = {
            "host": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_") and "TOKEN" not in k},
            "research_repo_commit": git(repo),
            "research_repo_dirty": bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True,
                                                       text=True).stdout.strip()) if git(repo) else None,
            "sglang_omni_commit": git(omni_repo) if omni_repo else None,
            "models": {s.get("name", "s"): {"path": s.get("model_path"), "revision": s.get("model_revision")}
                       for s in self.cfg.get("servers", [])},
        }
        py = rt.get("python")
        if py and rt.get("collect_versions", True):
            code = ("import json,importlib.metadata as m;out={}\n"
                    "for p in ['torch','sglang','sglang-omni','transformers','flashinfer-python','triton']:\n"
                    "  try: out[p]=m.version(p)\n  except Exception: out[p]=None\nprint(json.dumps(out))")
            try:
                txt = subprocess.run([os.path.expanduser(py), "-c", code], capture_output=True, text=True, timeout=300).stdout
                fp["packages"] = json.loads(txt.strip().splitlines()[-1]) if txt.strip() else None
            except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, IndexError):
                fp["packages"] = None
        return fp

    def check_imports(self) -> None:
        """Fail fast when the runtime lacks a package the servers need (e.g. MiniCPM-o's onnx)."""
        rt = self.cfg.get("runtime", {})
        mods = rt.get("required_imports") or []
        py = rt.get("python")
        if not mods or not py or self.cfg.get("fake_server"):
            return
        code = "import importlib, sys\nmissing=[]\nfor m in sys.argv[1:]:\n  try: importlib.import_module(m)\n  except Exception as e: missing.append(f'{m}: {type(e).__name__}: {e}')\nprint('\\n'.join(missing)); sys.exit(1 if missing else 0)"
        env = dict(os.environ)
        env.update({k: os.path.expanduser(str(v)) for k, v in rt.get("env", {}).items()})
        proc = subprocess.run([os.path.expanduser(py), "-c", code, *mods], capture_output=True, text=True, timeout=900, env=env)
        if proc.returncode != 0:
            raise ExperimentFailure("config_error", "runtime is missing required modules:\n" + (proc.stdout or proc.stderr)[-2000:])
        self.log(f"runtime imports ok: {', '.join(mods)}")

    # -- servers ----------------------------------------------------------------------------
    def start_servers(self) -> None:
        rt = self.cfg.get("runtime", {})
        for i, spec in enumerate(self.cfg.get("servers", [])):
            gpus_idx = spec.get("gpus", [])
            if any(g >= len(self.step_gpus) for g in gpus_idx) and self.step_gpus:
                raise ExperimentFailure("config_error", f"server {spec.get('name')} wants GPUs {gpus_idx}, step has {self.step_gpus}")
            gpus = [self.step_gpus[g] for g in gpus_idx] if self.step_gpus else [str(g) for g in gpus_idx]
            spec = dict(spec)
            if spec.get("stage_to_scratch") and rt.get("scratch_root"):
                t0 = time.monotonic()
                spec["model_path"] = stage_model(spec["model_path"], rt["scratch_root"])
                self.log(f"staged {spec.get('name')} weights to {spec['model_path']} in {time.monotonic() - t0:.0f}s")
            srv = Server(spec, rt, gpus, self.port_base + i, self.out / "logs", float(spec.get("ready_timeout_s", 2400)))
            self.log(f"starting server {srv.name} on GPUs {gpus} port {srv.port}")
            srv.start()
            self.servers.append(srv)
        # Servers start in parallel; wait for all of them.
        for srv in self.servers:
            took = srv.wait_ready()
            self.log(f"server {srv.name} healthy after {took:.0f}s")

    def stop_servers(self) -> dict[str, Any]:
        codes = {}
        for srv in self.servers:
            codes[srv.name] = srv.stop()
        return codes

    def backend(self, keep_text: bool = False):  # noqa: ANN201
        model = self.cfg.get("served_model_name")
        backs = [OpenAIChatBackend(s.base_url, model=model, keep_text=keep_text, name=s.name) for s in self.servers]
        if len(backs) == 1:
            return backs[0]
        return BackendPool(backs, policy=self.cfg.get("routing", "least_outstanding"))

    def server_events(self, phase: str, start: bool) -> None:
        for s in self.servers:
            if start:
                body = {"run_id": f"{self.cfg['experiment_id']}-{phase}-{s.name}",
                        "event_dir": str(self.out / "raw" / "server_events" / phase / s.name)}
                code, _ = _http_json(f"{s.base_url}/start_request_profile", body, timeout=30)
            else:
                code, _ = _http_json(f"{s.base_url}/stop_request_profile", {}, timeout=60)
            self.log(f"request profiler {'start' if start else 'stop'} on {s.name}: HTTP {code}")

    # -- phases -----------------------------------------------------------------------------
    def _pool(self, ph: dict[str, Any]) -> AssetPool | None:
        name = ph.get("assets") or self.cfg.get("assets")
        if not name:
            return None
        p = Path(os.path.expanduser(name))
        if not p.is_absolute():
            p = self.data_root / "assets" / name / "manifest.jsonl"
        return AssetPool(p)

    def _workload(self, ph: dict[str, Any]) -> Workload:
        p = Path(ph["workload"])
        if not p.is_absolute():
            p = Path(os.environ.get("OMNI_REPO", Path(__file__).resolve().parents[3])) / p
        wl = Workload.from_file(p)
        for k, v in ph.get("workload_overrides", {}).items():
            setattr(wl, k, v)
        return wl

    async def run_phase(self, ph: dict[str, Any], bench_trace: TraceCollector) -> dict[str, Any]:
        kind = ph["kind"]
        name = ph.get("name", kind)
        raw = self.out / "raw"
        seed = int(ph.get("seed", self.cfg.get("seed", 0)))
        summary: dict[str, Any] = {"phase": name, "kind": kind, "started": time.time()}
        if ph.get("server_events"):
            self.server_events(name, True)
        try:
            if kind == "smoke":
                summary.update(await self._phase_smoke(ph, name, seed, bench_trace))
            elif kind == "closed_loop":
                summary.update(await self._phase_closed(ph, name, seed, bench_trace))
            elif kind == "open_loop":
                summary.update(await self._phase_open(ph, name, seed, bench_trace))
            elif kind == "agent":
                summary.update(await self._phase_agent(ph, name, seed))
            elif kind == "replay":
                summary.update(await self._phase_replay(ph, name, seed))
            elif kind == "generate_assets":
                summary.update(await self._phase_assets(ph, name, seed))
            elif kind == "idle":
                await asyncio.sleep(float(ph.get("seconds", 10)))
            else:
                raise ExperimentFailure("config_error", f"unknown phase kind {kind!r}")
        finally:
            if ph.get("server_events"):
                self.server_events(name, False)
            summary["ended"] = time.time()
        _ = raw
        return summary

    async def _phase_smoke(self, ph, name, seed, bench_trace) -> dict[str, Any]:  # noqa: ANN001
        wl = self._workload(ph)
        sink = JsonlSink(self.out / "raw" / f"requests_{name}.jsonl")
        be = self.backend(keep_text=True)
        try:
            await closed_loop(be, wl, self._pool(ph), concurrency=1, requests=int(ph.get("requests", 3)), warmup=0,
                              sink=sink, collector=bench_trace, seed=seed, phase=name, keep_text=True)
        finally:
            await be.close()
            sink.close()
        recs = sink.records
        bad = []
        for r in recs:
            if r["status"] != "ok":
                bad.append(f"{r['request_id']}: {r['status']} {r.get('error')}")
            elif "text" in r["output_modalities"] and not r.get("text_chars"):
                bad.append(f"{r['request_id']}: empty text")
            elif "audio" in r["output_modalities"] and not r.get("output_audio_s"):
                bad.append(f"{r['request_id']}: no audio")
        samples = [{"request_id": r["request_id"], "text": (r.get("text") or "")[:400], "audio_s": r.get("output_audio_s"),
                    "expected": (r.get("payload") or {}).get("expected_answer")} for r in recs]
        if bad:
            raise ExperimentFailure("bad_output", "smoke validation failed: " + "; ".join(bad[:5]))
        return {"requests": len(recs), "validation": "ok", "samples": samples}

    async def _phase_closed(self, ph, name, seed, bench_trace) -> dict[str, Any]:  # noqa: ANN001
        wl = self._workload(ph)
        pool = self._pool(ph)
        levels = []
        stop_reason = None
        max_err = float(ph.get("max_error_rate", 0.2))
        e2e_slo = ph.get("stop_if_p50_e2e_above_s")
        for c in ph.get("concurrency", [1, 2, 4, 8]):
            if self.stop_requested:
                stop_reason = "stop requested"
                break
            for rep in range(int(ph.get("repeats", 1))):
                sink = JsonlSink(self.out / "raw" / f"requests_{name}.jsonl")
                be = self.backend()
                try:
                    info = await closed_loop(be, wl, pool, concurrency=int(c), requests=int(ph.get("requests_per_level", 32)),
                                             warmup=int(ph.get("warmup", c)), sink=sink, collector=bench_trace,
                                             seed=seed + 1000 * rep + int(c), phase=name)
                finally:
                    await be.close()
                    sink.close()
                meas = [r for r in sink.records if not r.get("warmup")]
                errors = sum(1 for r in meas if r["status"] != "ok")
                lv = {"concurrency": c, "repeat": rep, **info, "errors": errors, "n": len(meas),
                      "e2e": M.summarize([M.request_metrics(r).get("e2e_s") for r in meas if r["status"] == "ok"], ci=False),
                      "ttft": M.summarize([M.request_metrics(r).get("ttft_s") for r in meas if r["status"] == "ok"], ci=False),
                      "throughput": M.throughput(meas)}
                levels.append(lv)
                self.log(f"{name}: c={c} rep={rep} n={len(meas)} errors={errors} p50_e2e={lv['e2e'].get('median')}")
                if meas and errors / len(meas) > max_err:
                    stop_reason = f"error rate {errors}/{len(meas)} at concurrency {c}"
                    break
                if e2e_slo and lv["e2e"].get("median") and lv["e2e"]["median"] > float(e2e_slo):
                    stop_reason = f"median e2e {lv['e2e']['median']:.2f}s > {e2e_slo}s at concurrency {c}"
                    break
            if stop_reason:
                break
        return {"levels": levels, "stopped_reason": stop_reason}

    async def _phase_open(self, ph, name, seed, bench_trace) -> dict[str, Any]:  # noqa: ANN001
        wl = self._workload(ph)
        pool = self._pool(ph)
        out = []
        for rate in ph.get("rates_rps", [0.5, 1.0]):
            sink = JsonlSink(self.out / "raw" / f"requests_{name}.jsonl")
            be = self.backend()
            try:
                info = await open_loop(be, wl, pool, rate_rps=float(rate), requests=int(ph.get("requests_per_rate", 40)),
                                       sink=sink, collector=bench_trace, seed=seed + int(rate * 1000), phase=name)
            finally:
                await be.close()
                sink.close()
            errors = sum(1 for r in sink.records if r["status"] != "ok")
            out.append({"rate_rps": rate, **info, "errors": errors})
            self.log(f"{name}: rate={rate} errors={errors}")
            if errors > len(sink.records) * float(ph.get("max_error_rate", 0.2)):
                break
        return {"rates": out}

    async def _phase_agent(self, ph, name, seed) -> dict[str, Any]:  # noqa: ANN001
        from ..tracing.agent_harness import AgentConfig, ToolBox, make_knowledge_base, make_tasks, run_agents

        kb = make_knowledge_base(int(ph.get("kb_seed", 7)))
        tasks = make_tasks(int(ph.get("sessions", 8)), seed, kb)
        cfg = AgentConfig(**ph.get("agent", {}))
        if cfg.input_modality == "audio":
            pool = self._pool(ph)
            by_task = {a.meta.get("task_id"): a for a in (pool.of_kind("audio", "agent_question") if pool else [])}
            for t in tasks:
                a = by_task.get(t.task_id)
                if a is None:
                    raise ExperimentFailure("config_error", f"no spoken question for {t.task_id} in asset pool")
                t.audio_path = a.path
        lat = ph.get("tool_latency")  # [median_s, sigma] or None
        tools = ToolBox(kb, simulated_latency=tuple(lat) if lat else None, rng=random.Random(seed))
        # Tasks, prompts and media are synthetic, so the trace may keep request content
        # (content_policy=public); that lets a replay resend the original requests exactly.
        tc = TraceCollector(self.out / "raw" / f"agent_trace_{name}.jsonl", source_kind=SourceKind.OBSERVED,
                            generator="omni_research.tracing.agent_harness", seed=seed, content_policy="public",
                            workload={"phase": name, "sessions": len(tasks), "concurrency": ph.get("concurrency", 4),
                                      "input_modality": cfg.input_modality, "speak_final": cfg.speak_final,
                                      "tool_latency": lat, "experiment_id": self.cfg["experiment_id"]},
                            notes="Tasks, knowledge base and simulated tool latency are synthetic; model requests are real.")
        results: list[dict[str, Any]] = []
        be = self.backend(keep_text=True)
        try:
            outcomes = await run_agents(be, tc, tasks, tools, cfg, int(ph.get("concurrency", 4)), seed=seed, results=results,
                                        arrival_rate_per_s=ph.get("session_arrival_rate_per_s"))
        finally:
            await be.close()
            tc.close()
        sink = JsonlSink(self.out / "raw" / f"requests_{name}.jsonl")
        for r in results:
            r.pop("text", None)  # keep the trace free of model text by default; outcomes record success
            sink.write(r)
        sink.close()
        (self.out / "raw" / f"agent_outcomes_{name}.jsonl").write_text("".join(json.dumps(o) + "\n" for o in outcomes))
        ok = sum(1 for o in outcomes if o["success"])
        return {"sessions": len(outcomes), "task_success": ok, "requests": len(results),
                "trace": f"raw/agent_trace_{name}.jsonl"}

    async def _phase_replay(self, ph, name, seed) -> dict[str, Any]:  # noqa: ANN001
        from ..replay.engine import ReplayConfig, ReplayEngine, build_plan, replay_collector
        from ..tracing.io import load_trace

        tp = Path(os.path.expanduser(ph["trace"]))
        if not tp.is_absolute():
            tp = self.data_root / tp
        trace = load_trace(tp)
        plan = build_plan(trace)
        rcfg = ReplayConfig(mode=ph.get("mode", "dependency"), speed=float(ph.get("speed", 1.0)),
                            session_copies=int(ph.get("session_copies", 1)), session_arrival=ph.get("session_arrival", "trace"),
                            seed=seed, request_timeout_s=ph.get("request_timeout_s"))
        be = self.backend()
        tc = replay_collector(self.out / "raw" / f"replay_trace_{name}.jsonl", plan, rcfg, getattr(be, "name", "?"))
        try:
            eng = ReplayEngine(plan, be, rcfg, pool=self._pool(ph), results_path=self.out / "raw" / f"requests_{name}.jsonl",
                               collector=tc)
            summary = await eng.run()
        finally:
            await be.close()
            tc.close()
        return {"replay": summary, "source_trace": str(tp)}

    async def _phase_assets(self, ph, name, seed) -> dict[str, Any]:  # noqa: ANN001
        from ..tracing.agent_harness import make_knowledge_base, make_tasks
        from ..replay import assets as A

        pool_dir = self.data_root / "assets" / ph.get("pool", "synthetic-v1")
        rows: list[dict[str, Any]] = []
        if ph.get("videos", True):
            ff = ph.get("ffmpeg") or shutil.which("ffmpeg") or os.path.expanduser(self.cfg.get("runtime", {}).get("ffmpeg", ""))
            rows += A.generate_counting_videos(pool_dir, seed=seed, ffmpeg=ff or None)
        if ph.get("speech", True) and self.servers:
            tasks = make_tasks(int(ph.get("question_tasks", 24)), int(ph.get("task_seed", seed)), make_knowledge_base(int(ph.get("kb_seed", 7))))
            extra = json.loads(substitute(json.dumps(ph.get("tts_extra", {})), self.cfg.get("runtime", {})))
            rows += await A.generate_spoken_questions(self.servers[0].base_url, pool_dir, tasks, source_label=ph["source_label"],
                                                      model=ph.get("tts_model", self.cfg.get("served_model_name")),
                                                      passages=int(ph.get("passages", 6)), seed=seed, extra_body=extra)
        manifest = A.write_manifest(pool_dir, rows)
        return {"assets": len(rows), "manifest": str(manifest)}

    # -- main -------------------------------------------------------------------------------
    def run(self) -> dict[str, Any]:
        result: dict[str, Any] = {"status": "ok", "experiment_id": self.cfg["experiment_id"]}
        (self.out / "config.json").write_text(json.dumps(self.cfg, indent=2))
        fp = self.fingerprint()
        manifest = {"experiment_id": self.cfg["experiment_id"], "run_dir": str(self.out), "started": self.t_start,
                    "environment": fp, "hardware_label": self.cfg.get("hardware_label"),
                    "description": self.cfg.get("description", "")}
        atomic_write_json(self.out / "manifest.json", manifest)
        gpu = GpuSampler(self.out / "raw" / "gpu_samples.csv", int(self.cfg.get("gpu_sample_ms", 200)))
        cpu = CpuSampler(self.out / "raw" / "cpu_samples.jsonl",
                         lambda: {s.name: s.proc.pid for s in self.servers if s.proc and s.proc.poll() is None} | {"client": os.getpid()})
        gpu.start()
        cpu.start()

        def on_term(signum, _frame):  # noqa: ANN001
            self.stop_requested = True
            self.log(f"received signal {signum}; finishing the current phase and saving results")
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, on_term)
        signal.signal(signal.SIGUSR1, on_term)
        try:
            self.check_imports()
            if self.cfg.get("servers"):
                self.start_servers()
                result["server_ready_s"] = {s.name: s.ready_after_s for s in self.servers}
            bench_trace = TraceCollector(self.out / "raw" / "bench_trace.jsonl", source_kind=SourceKind.OBSERVED,
                                         generator="omni_research.replay.drivers", seed=self.cfg.get("seed", 0),
                                         workload={"experiment_id": self.cfg["experiment_id"]},
                                         notes="Benchmark load: synthetic prompts / synthetic media, real server.")
            try:
                for ph in self.cfg.get("phases", []):
                    if self.stop_requested:
                        break
                    self.log(f"phase {ph.get('name', ph['kind'])} starting")
                    s = asyncio.run(self.run_phase(ph, bench_trace))
                    self.phase_summaries.append(s)
                    atomic_write_json(self.out / "analysis" / "phases.json", self.phase_summaries)
                    self.log(f"phase {ph.get('name', ph['kind'])} done")
            finally:
                bench_trace.close()
        except ExperimentFailure as exc:
            result.update(status="failed", error_class=exc.error_class, error=str(exc)[:4000])
            self.log(f"FAILED ({exc.error_class}): {str(exc)[:500]}")
        except KeyboardInterrupt:
            result.update(status="failed", error_class="interrupted", error="stopped by signal; partial results saved")
        except Exception as exc:  # noqa: BLE001 - every failure must end up in result.json
            import traceback

            result.update(status="failed", error_class="runner_error", error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-3000:]}")
            self.log(f"runner error: {exc!r}")
        finally:
            result["server_exit_codes"] = self.stop_servers()
            gpu.stop()
            cpu.stop()
            result["phases"] = self.phase_summaries
            result["gpu"] = summarize_gpu_samples(self.out / "raw" / "gpu_samples.csv")
            result["wall_s"] = time.time() - self.t_start
            manifest.update(ended=time.time(), status=result["status"],
                            files=sorted(str(p.relative_to(self.out)) for p in self.out.rglob("*") if p.is_file()))
            atomic_write_json(self.out / "manifest.json", manifest)
            atomic_write_json(self.out / "analysis" / "metrics.json", {"phases": self.phase_summaries, "gpu": result["gpu"]})
            atomic_write_json(self.out / "result.json", result)
            self.client_log.close()
        return result


def summarize_gpu_samples(path: Path) -> dict[str, Any]:
    """Per-GPU peak memory and mean/peak sampled utilization from nvidia-smi CSV samples."""
    if not path.exists():
        return {}
    per: dict[str, dict[str, Any]] = {}
    lines = path.read_text().splitlines()[1:]
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 9:
            continue
        idx = parts[1]
        try:
            util = float(parts[3])
        except ValueError:
            continue
        def num(x: str) -> float | None:
            try:
                return float(x)
            except ValueError:
                return None  # e.g. [N/A] memory on GB10 unified memory
        mem, total = num(parts[5]), num(parts[6])
        d = per.setdefault(idx, {"samples": 0, "util_sum": 0.0, "util_max": 0.0, "mem_max_mib": None, "mem_total_mib": total, "uuid": parts[2]})
        d["samples"] += 1
        d["util_sum"] += util
        d["util_max"] = max(d["util_max"], util)
        if mem is not None:
            d["mem_max_mib"] = mem if d["mem_max_mib"] is None else max(d["mem_max_mib"], mem)
    for d in per.values():
        d["util_mean"] = d.pop("util_sum") / d["samples"] if d["samples"] else None
    return per


def resolve_config(cfg: dict[str, Any], repo: Path) -> dict[str, Any]:
    """Expand ``runtime_profile`` and per-server ``model`` references from configs/models/."""
    cfg = json.loads(json.dumps(cfg))
    models_dir = repo / "configs" / "models"
    if cfg.get("runtime_profile"):
        prof = json.loads((models_dir / f"runtime_{cfg['runtime_profile']}.json").read_text())
        prof.update(cfg.get("runtime", {}))
        cfg["runtime"] = prof
    for spec in cfg.get("servers", []):
        if spec.get("model"):
            m = json.loads((models_dir / f"{spec['model']}.json").read_text())
            spec.setdefault("model_path", m["model_path"])
            spec.setdefault("model_revision", m.get("revision"))
            cfg.setdefault("served_model_name", m.get("served_model_name"))
    if os.environ.get("OMNI_FAKE_SERVER") == "1":
        # Dry run of a real plan on a CPU-only host: every server becomes the fake server.
        for spec in cfg.get("servers", []):
            spec["command"] = f"{sys.executable} -m omni_research.replay.fake_server --port {{port}} --startup-s 2 --agent-json"
        cfg.setdefault("runtime", {})["collect_versions"] = False
        cfg["runtime"].pop("env_script", None)
        cfg["runtime"].pop("cwd", None)
        cfg["runtime"].pop("scratch_root", None)
        cfg["fake_server"] = True
    return cfg


def update_latest_link(data_root: Path, name: str, out: Path) -> None:
    """``<data_root>/latest/<name>`` -> the newest successful run of an experiment (atomic swap)."""
    latest = data_root / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    tmp = latest / f".{name}.tmp"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(out.resolve())
    os.replace(tmp, latest / name)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    repo = Path(os.environ.get("OMNI_REPO") or Path(__file__).resolve().parents[3])
    cfg = resolve_config(json.loads(Path(args.config).read_text()), repo)
    out = Path(os.environ.get("OMNI_EXP_DIR") or args.out or f"results/experiments/{cfg['experiment_id']}")
    runner = ExperimentRunner(cfg, out)
    res = runner.run()
    if res["status"] == "ok":
        update_latest_link(runner.data_root, cfg["experiment_id"], out)
    return 0 if res["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
