"""实验 4: how large the batches of each stage get, and whether a larger batch makes a stage faster or slower."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("exp04_report", HERE / "exp04_report.py")
r04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r04)

LOG = [
    "2026-11-16 19:13:41,447 [INFO] sglang_omni.models.minicpm_o.stages: minicpm_code2wav_batch size=1 max_codec_tokens=408",
    "2026-11-16 19:13:41,600 [INFO] something else",
    "2026-11-16 19:13:45,017 [INFO] sglang_omni.models.minicpm_o.stages: minicpm_code2wav_batch size=3 max_codec_tokens=385",
    "2026-11-16 19:13:47,668 [INFO] sglang_omni.models.minicpm_o.stages: minicpm_code2wav_batch size=16 max_codec_tokens=293",
]


def test_parse_code2wav_batches_with_the_node_offset():
    b = r04.parse_c2w_log(LOG, utc_offset_s=-5 * 3600)
    assert [x["size"] for x in b] == [1, 3, 16] and b[0]["max_codec_tokens"] == 408
    # 19:13:41.447 local at UTC-5 is 00:13:41.447 UTC the next day
    assert abs(b[0]["t"] - 1794874421.447) < 1e-3


def test_batch_stats_in_a_window():
    batches = [{"t": 10.0, "size": 1}, {"t": 11.0, "size": 3}, {"t": 12.0, "size": 16}, {"t": 99.0, "size": 5}]
    st = r04.batch_stats(batches, 9.0, 20.0, cap=16)
    assert st == {"batches": 3, "requests": 20, "mean_size": 20 / 3, "max_size": 16, "share_at_cap": 1 / 3}


def test_per_request_time_from_back_to_back_batches():
    # One replica: batch starts 0, 2.0 (after a batch of 10), 4.4 (after 12), then a 30 s gap (idle, left out).
    batches = [{"t": 0.0, "size": 10}, {"t": 2.0, "size": 12}, {"t": 4.4, "size": 4}, {"t": 34.4, "size": 1}]
    est = r04.per_request_from_gaps(batches, max_gap_s=5.0)
    assert est["pairs"] == 2 and abs(est["median_ms"] - 200.0) < 1e-9  # 2.0/10 and 2.4/12 -> 200 ms each


def test_tokens_per_second_per_replica():
    assert r04.rate_per_replica(eff_batch=25.0, ms_per_unit=34.0) == 25.0 * 1000.0 / 34.0
    assert r04.rate_per_replica(eff_batch=None, ms_per_unit=34.0) is None


def test_gap_estimate_only_where_code2wav_is_backlogged():
    # Batches of about one request mean code2wav waited for work: the gaps are idle time, not service time.
    assert r04.backlogged({"mean_size": 1.2}) is False
    assert r04.backlogged({"mean_size": 2.4}) is True
    assert r04.backlogged({"mean_size": None}) is False
