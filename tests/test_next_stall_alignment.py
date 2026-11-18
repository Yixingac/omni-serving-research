"""Next-step research P3: aligning a short stall with the server log and CPU samples (synthetic data only)."""

import datetime as dt
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("next_stall_alignment", HERE / "next_stall_alignment.py")
sa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sa)

T = dt.datetime(2026, 11, 18, 4, 13, 39, 656000)
STAGES = "[INFO] sglang_omni.models.minicpm_o.stages: "
METRICS = "[INFO] sglang.srt.managers.scheduler_components.metrics_reporter: "


def stamp(sec: float) -> str:
    t = T + dt.timedelta(seconds=sec)
    return t.strftime("%Y-%m-%d %H:%M:%S") + f",{t.microsecond // 1000:03d}"


def c2w(sec: float, size: int) -> str:
    return f"{stamp(sec)} {STAGES}minicpm_code2wav_batch size={size} max_codec_tokens=400\n"


def decode(sec: float, running: int, per_req: int, queue: int) -> str:
    return (f"{stamp(sec)} {METRICS}Decode batch, #running-req: {running}, #token: {running * per_req}, "
            f"token usage: 0.00, cuda graph: True, gen throughput (token/s): 2000.00, #queue-req: {queue}\n")


def test_scan_counts_lines_batches_and_talker_state():
    lines = [c2w(10.2, 1), c2w(10.6, 4), decode(10.5, 6, 450, 0), decode(10.7, 40, 80, 0),  # 2nd is a thinker line
             c2w(14.1, 15), "a line without a time stamp\n", c2w(30.0, 1)]                   # last one is outside
    s = sa.scan_log(lines, T.timestamp(), 10, 15, 200)
    assert s["log_lines"][10] == 4 and s["log_lines"][11] == 0 and s["log_lines"][14] == 1
    assert s["code2wav_batches"][10] == {"batches": 2, "requests": 5}
    assert s["code2wav_batches"][14] == {"batches": 1, "requests": 15}
    assert s["talker"] == {10: {"running_p50": 6, "queue_max": 0}}
    assert sa.log_gaps(s["stamps"], 0.5) == [{"at_s": 10.7, "gap_s": 3.4}]


def test_cpu_rows_window_and_groups():
    base = T.timestamp()
    samples = [{"t_wall_ns": int((base + t) * 1e9), "system_cpu_percent": cpu,
                "groups": {"s0": {"cpu_percent": srv, "by_pid": {}}, "client": {"cpu_percent": 20.0}}}
               for t, cpu, srv in ((53.7, 12.0, 684.0), (54.8, 2.0, 35.0), (70.0, 13.0, 800.0))]
    rows = sa.cpu_rows(samples, base, 50, 60)
    assert [r["t_s"] for r in rows] == [53.7, 54.8]
    assert rows[1]["groups"] == {"s0": 35.0, "client": 20.0} and rows[1]["system_cpu_percent"] == 2.0
