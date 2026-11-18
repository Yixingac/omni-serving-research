"""实验 13: does the link between the thinker GPU and the talker GPU change anything."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("exp13_report", HERE / "exp13_report.py")
r13 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r13)


def test_link_of_service_names():
    assert [r13.link_of(x) for x in ("g_nv4a", "g_nv4", "g_pix", "g_node", "g_sysb")] == ["NV4", "NV4", "PIX", "NODE", "SYS"]


def test_bandwidth_by_link_type():
    topo = {"pairs": [{"link": "NV4", "pair": [0, 3], "0->3_GBps": 52.0, "3->0_GBps": 53.0},
                      {"link": "PIX", "pair": [1, 3], "1->3_GBps": 2.7, "3->1_GBps": 2.7},
                      {"link": "SYS", "pair": [0, 4], "0->4_GBps": 2.6, "4->0_GBps": 2.8}]}
    bw = r13.bandwidth_by_link(topo)
    assert bw["NV4"] == {"pairs": 1, "median_GBps": 52.5, "min_GBps": 52.0, "max_GBps": 53.0}
    assert bw["SYS"]["median_GBps"] == 2.7


def test_verdict_supports_when_link_time_and_latency_differences_are_small():
    hop = {"NV4": 3.8, "PIX": 4.0, "SYS": 3.9}          # ms
    e2e = {"g_nv4a": 2.45, "g_nv4b": 2.86, "g_pix": 2.65, "g_sysa": 2.82}  # s
    v = r13.link_verdict(hop, e2e)
    assert v["verdict"] == "支持"
    assert abs(v["hop_spread_share_of_e2e"] - 0.2e-3 / ((2.65 + 2.82) / 2)) < 1e-12  # spread / median end-to-end
    assert abs(v["e2e_rel_non_nvlink"] - ((2.65 + 2.82) / 2 / ((2.45 + 2.86) / 2) - 1)) < 1e-9


def test_verdict_rejects_a_large_latency_gap_beyond_same_link_spread():
    hop = {"NV4": 3.8, "SYS": 3.9}
    e2e = {"g_nv4a": 2.0, "g_nv4b": 2.02, "g_sysa": 2.6, "g_sysb": 2.7}
    assert r13.link_verdict(hop, e2e)["verdict"] == "不支持"


def test_verdict_uncertain_when_gap_is_within_same_link_spread():
    hop = {"NV4": 3.8, "SYS": 3.9}
    e2e = {"g_nv4a": 2.0, "g_nv4b": 2.6, "g_sysa": 2.5, "g_sysb": 2.6}  # 11 % gap, but the two NVLink services differ by 30 %
    assert r13.link_verdict(hop, e2e)["verdict"] == "不确定"


def test_link_of_ignores_a_run_suffix():
    # Both G1 runs have a service called g_pix; keys carry the run after '#'.
    assert r13.link_of("g_pix#g1b_links") == "PIX" and r13.link_of("g_nv4a#g1a_links") == "NV4"
