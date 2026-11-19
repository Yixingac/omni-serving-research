"""R4 analysis on a synthetic server log and session records (no GPU, no data files)."""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
spec = importlib.util.spec_from_file_location("ppv_r4_report", HERE / "ppv_r4_report.py")
r4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r4)

T = "2026-11-19 03:00:{s:02d},{ms:03d} [INFO] sglang_omni.scheduling.omni_scheduler: "


def pool(sec, rep, evicted, available=100000, evictable=1000, total=127039):
    pid, gpu = rep.split(":")
    return (T.format(s=sec, ms=0) + f"exp_kv_pool pid={pid} gpu={gpu} stage=thinker total_tokens={total} "
            f"available_tokens={available} evictable_tokens={evictable} protected_tokens=10 running_reqs=1 waiting_reqs=0 "
            f"evicted_tokens_total={evicted} evict_calls_total=0")


def prefill(sec, ms, rep, rid, inp, cached, evicted=None):
    pid, gpu = rep.split(":")
    tail = f" evicted_tokens_total={evicted}" if evicted is not None else ""
    return (T.format(s=sec, ms=ms) + f"exp_kv_prefill pid={pid} gpu={gpu} stage=thinker rid={rid} input_tokens={inp} "
            f"cached_tokens={cached} extend_tokens={inp - cached}" + tail)


def test_parse_log_accepts_the_unset_extend_range_sentinel():
    line = (T.format(s=1, ms=0) + "exp_kv_prefill pid=1 gpu=0 stage=thinker rid=x input_tokens=50 cached_tokens=10 "
            "extend_tokens=-1 evicted_tokens_total=0")
    prefills, _ = r4.parse_log([line])
    assert len(prefills) == 1 and prefills[0]["extend_tokens"] == -1


def test_prefill_tokens_are_input_minus_cached():
    reqs, prefills, pools = _scenario()
    att = r4.attribute(reqs, prefills, pools)
    assert att["prefill_tokens"] == sum(p["input_tokens"] - p["cached_tokens"] for p in prefills)


def test_eviction_bracketing_from_pool_lines_without_prefill_counters():
    _, pools = r4.parse_log([pool(0, "11:0", 0), pool(2, "11:0", 0), pool(4, "11:0", 700)])
    t0 = pools[0]["ts"]
    assert r4.evicted_between(pools, "11:0", t0 + 0.5, t0 + 1.5) == 0
    assert r4.evicted_between(pools, "11:0", t0 + 2.5, t0 + 3.5) == 700
    assert r4.evicted_between(pools, "11:0", t0 + 4.5, t0 + 5.0) is None


def test_parse_log_reads_both_line_kinds_and_ignores_others():
    lines = [pool(1, "11:0", 0), prefill(1, 500, "11:0", "r4-s000-t00-answer", 400, 120),
             "2026-11-19 03:00:02,000 [INFO] sglang.srt...: Prefill batch, #new-seq: 1, #new-token: 280, #cached-token: 120"]
    prefills, pools = r4.parse_log(lines, utc_offset_s=-5 * 3600)
    assert len(prefills) == 1 and len(pools) == 1
    assert prefills[0]["replica"] == "11:0" and prefills[0]["cached_tokens"] == 120 and prefills[0]["extend_tokens"] == 280
    assert pools[0]["total"] == 127039 and pools[0]["evicted"] == 0
    assert abs(prefills[0]["ts"] - pools[0]["ts"] - 0.5) < 1e-6


def _scenario():
    # session 0 stays on replica A (clean); session 1 switches replica (routing); session 2 stays on A after an eviction
    lines = [pool(0, "11:0", 0), pool(0, "22:1", 0)]
    lines += [prefill(1, 0, "11:0", "p-s000-t00-answer", 400, 100, 0), prefill(1, 0, "22:1", "p-s001-t00-answer", 420, 100, 0),
              prefill(1, 100, "11:0", "p-s002-t00-answer", 380, 100, 0)]
    lines += [pool(2, "11:0", 0), pool(2, "22:1", 0)]
    lines += [prefill(3, 0, "11:0", "p-s000-t01-answer", 600, 360, 0), prefill(3, 0, "11:0", "p-s001-t01-answer", 640, 100, 0)]
    lines += [pool(4, "11:0", 5000), pool(4, "22:1", 0)]
    lines += [prefill(5, 0, "11:0", "p-s002-t01-answer", 560, 100, 5000)]
    lines += [pool(6, "11:0", 5000), pool(6, "22:1", 0)]
    prefills, pools = r4.parse_log(lines)
    reqs = []
    for rid in ("p-s000-t00-answer", "p-s001-t00-answer", "p-s002-t00-answer", "p-s000-t01-answer", "p-s001-t01-answer",
                "p-s002-t01-answer"):
        s, t = int(rid.split("-s")[1][:3]), int(rid.split("-t")[1][:2])
        slow = rid in ("p-s001-t01-answer", "p-s002-t01-answer")
        reqs.append({"request_id": rid, "session": s, "turn": t, "step": "answer", "sent_ns": 1_000, "status": "ok",
                     "first_text_ns": 1_000 + (900_000_000 if slow else 300_000_000), "completion_tokens": 20})
    return reqs, prefills, pools


def test_attribution_separates_routing_and_eviction_losses():
    reqs, prefills, pools = _scenario()
    att = r4.attribute(reqs, prefills, pools)
    assert att["shared_floor_tokens"] == 100
    assert att["reference_reuse"]["next_turn"]["n"] == 1  # only session 0 is a clean pair
    ref = att["reference_reuse"]["next_turn"]["median"]
    assert abs(ref - 360 / 400) < 1e-9
    assert att["counts"] == {"pairs": 3, "cross_replica": 1, "same_replica_evicted": 1, "clean": 1, "unknown_eviction": 0}
    assert att["lost_reuse_tokens"]["routing"] == round(ref * 420 - 100)
    assert att["lost_reuse_tokens"]["eviction"] == round(ref * 380 - 100)
    assert att["ttft_s_by_cause"]["routing"]["p50"] > 1.2 * att["ttft_s_by_cause"]["clean"]["p50"]


def test_working_set_follows_each_sessions_latest_request():
    reqs, prefills, _ = _scenario()
    sessions = [{"session": s} for s in range(3)]
    ws = r4.working_set(reqs, sessions, prefills)
    a = dict(ws["11:0"])
    first = min(a)
    assert a[first] == 420  # at the first prefill only session 0 has a request on replica A
    assert a[round(first + 1.0, 3)] == 420 + 400  # one second later session 2 too (prompt + 20 completion tokens)


def test_verdict_needs_losses_that_cost_time_not_just_a_ratio():
    calm = {"phase": "n8", "sessions": 8, "logical_ratio_peak": {"a": 0.9}, "pool": {"a": {"resident_peak_ratio": 0.9}},
            "attribution": {"lost_share_of_prefill": {"eviction": 0.01, "routing": 0.02},
                            "ttft_s_by_cause": {"clean": {"p50": 0.3}, "eviction": {"p50": 0.31}}}}
    v = r4.verdict([calm])
    assert v["levels"][0]["over_registered_ratio"] and not v["levels"][0]["eviction_pressure"]
    assert v["conclusion"].startswith("当前实验条件下")
    hot = {**calm, "attribution": {"lost_share_of_prefill": {"eviction": 0.25, "routing": 0.0},
                                   "ttft_s_by_cause": {"clean": {"p50": 0.3}, "eviction": {"p50": 0.6}}}}
    assert r4.verdict([calm, hot])["conclusion"].startswith("eviction pressure")
    routed = {**calm, "attribution": {"lost_share_of_prefill": {"eviction": 0.0, "routing": 0.3},
                                      "ttft_s_by_cause": {"clean": {"p50": 0.3}, "routing": {"p50": 0.5}}}}
    assert "session-affinity" in r4.verdict([routed])["conclusion"]


def test_smoke_check_joins_requests_and_measures_media_tokens_by_difference():
    lines = [prefill(1, 0, "11:0", "sm-s000-t00-answer", 900, 100), prefill(1, 0, "11:0", "sm-s001-t00-answer", 950, 100),
             prefill(2, 0, "11:0", "sm-s000-t01-answer", 1300, 700),
             prefill(1, 0, "22:1", "tx-s000-t00-answer", 400, 100), prefill(1, 0, "22:1", "tx-s001-t00-answer", 430, 100)]
    prefills, _ = r4.parse_log(lines)
    rq = lambda rid, s, t: {"request_id": rid, "session": s, "turn": t, "step": "answer", "tool_turn": False}  # noqa: E731
    phases = {"r4_smoke_n4": [rq("sm-s000-t00-answer", 0, 0), rq("sm-s001-t00-answer", 1, 0), rq("sm-s000-t01-answer", 0, 1),
                              rq("sm-s001-t01-answer", 1, 1)],
              "r4_smoke_n4_textonly": [rq("tx-s000-t00-answer", 0, 0), rq("tx-s001-t00-answer", 1, 0)]}
    out = r4.smoke_check(phases, prefills)
    assert out["r4_smoke_n4"]["joined_share"] == 0.75 and out["r4_smoke_n4"]["input_tokens_p50_by_turn"] == {0: 925, 1: 1300}
    assert out["media_tokens_per_turn"]["values"] == [500, 520] and out["media_tokens_per_turn"]["p50"] == 510


def test_structural_gap_pairs_the_media_phase_with_the_text_phase_of_the_same_size():
    lv = lambda name, n, ref, ttft: {"phase": name, "sessions": n, "ttft_s": {"p50": ttft},  # noqa: E731
                                     "attribution": {"reference_reuse": {"next_turn": {"median": ref}}}}
    out = r4.structural_gap([lv("r4_n32", 32, 0.9, 0.4), lv("r4_n64", 64, 0.85, 0.5), lv("r4_n32_media2", 32, 0.3, 0.7)])
    assert out == {"r4_n32_media2": {"vs": "r4_n32", "next_turn_reuse_text": 0.9, "next_turn_reuse_media": 0.3,
                                     "next_turn_reuse_all_pairs_text": None, "next_turn_reuse_all_pairs_media": None,
                                     "ttft_p50_text": 0.4, "ttft_p50_media": 0.7}}


def test_combined_verdict_needs_the_same_conclusion_in_all_three_blocks():
    none = "当前实验条件下，尚无证据支持引入更复杂的 Agent-aware Cache Management。"
    ev = "eviction pressure observed: design the session-affinity / cost-aware eviction A/B"
    same = {f"b{i}": {"conclusion": none} for i in (1, 2, 3)}
    assert r4.combined_verdict(same) == {"conclusion": none, "stable": True, "by_block": {f"b{i}": none for i in (1, 2, 3)}}
    mixed = {**same, "b3": {"conclusion": ev}}
    v = r4.combined_verdict(mixed)
    assert v["conclusion"] == "不确定" and not v["stable"] and v["by_block"]["b3"] == ev
    assert r4.combined_verdict({"b1": {"conclusion": none}})["conclusion"] == "数据不足"


def test_erratum_reference_falls_back_to_the_run_level_below_ten_clean_pairs():
    reqs, prefills, pools = _scenario()
    registered = r4.attribute(reqs, prefills, pools)
    assert registered["reference_reuse"]["next_turn"]["source"] == "level"  # the code as registered: one pair is enough
    fixed = r4.attribute(reqs, prefills, pools, run_ref={"within_turn": None, "next_turn": 0.5})
    ref = fixed["reference_reuse"]["next_turn"]
    assert ref["source"] == "run" and ref["median"] == 0.5 and ref["n"] == 1
    assert fixed["lost_reuse_tokens"]["routing"] == round(0.5 * 420 - 100)
    assert fixed["reference_reuse"]["within_turn"]["source"] == "none"
    assert fixed["reuse_all_pairs"]["next_turn"]["n"] == 3


def test_run_reference_pools_the_clean_pairs_of_the_main_levels_only():
    lv = lambda name, ratios: {"phase": name, "attribution": {"reference_reuse": {  # noqa: E731
        "next_turn": {"_clean_ratios": ratios}, "within_turn": {"_clean_ratios": []}}}}
    ref = r4.run_reference([lv("r4_n8", [0.6, 0.62]), lv("r4_n16", [0.64]), lv("r4_n32_media2", [0.1, 0.1, 0.1])])
    assert ref == {"within_turn": None, "next_turn": 0.62}
    assert r4.strip_private({"a": {"_clean_ratios": [1], "median": 2}, "b": [{"_x": 1, "y": 2}]}) == {"a": {"median": 2}, "b": [{"y": 2}]}


def _lv(name, n, ev, rt, clean_n, clean_p50=0.3, ev_p50=0.6, rt_p50=0.6):
    return {"phase": name, "sessions": n, "logical_ratio_peak": {"a": 0.2}, "pool": {"a": {"resident_peak_ratio": 1.0}},
            "attribution": {"lost_share_of_prefill": {"eviction": ev, "routing": rt},
                            "ttft_s_by_cause": {"clean": {"p50": clean_p50, "n": clean_n}, "eviction": {"p50": ev_p50, "n": 50},
                                                "routing": {"p50": rt_p50, "n": 50}}}}


def test_corrected_verdict_calls_a_high_loss_without_comparable_ttft_undetermined():
    v = r4.verdict_corrected([_lv("r4_n8", 8, 0.0, 0.25, 40), _lv("r4_n64", 64, 0.30, 0.20, 0)])
    assert [(x["eviction_pressure"], x["routing_cost"]) for x in v["levels"]] == [("no", "yes"), ("undetermined", "undetermined")]
    assert v["conclusion"] == r4.UNDETERMINED_EVICTION and "8 sessions" in v["routing_note"]
    # a low loss share settles "no" even when first-token time cannot be compared
    low = r4.verdict_corrected([_lv("r4_n8", 8, 0.0, 0.25, 40), _lv("r4_n64", 64, 0.05, 0.02, 0)])
    assert low["levels"][1]["eviction_pressure"] == "no" and "session-affinity" in low["conclusion"]
    # what the registered code produced for such a level: no reference -> zero loss and no clean TTFT -> "no evidence"
    v1_level = {**_lv("r4_n64", 64, 0.0, 0.0, 0), "attribution": {"lost_share_of_prefill": {"eviction": 0.0, "routing": 0.0},
                                                                  "ttft_s_by_cause": {"eviction": {"p50": 1.1, "n": 305}}}}
    v1 = r4.verdict([v1_level])
    assert not v1["levels"][0]["eviction_pressure"] and v1["conclusion"].startswith("当前实验条件下")


def test_corrected_verdict_keeps_the_registered_order_and_sentence():
    calm = r4.verdict_corrected([_lv("r4_n8", 8, 0.01, 0.02, 40), _lv("r4_n16", 16, 0.02, 0.05, 30)])
    assert calm["conclusion"].startswith("当前实验条件下，尚无证据支持引入更复杂的 Agent-aware Cache Management")
    hot = r4.verdict_corrected([_lv("r4_n8", 8, 0.2, 0.2, 40), _lv("r4_n64", 64, 0.3, 0.3, 0)])
    assert hot["conclusion"].startswith("eviction pressure")
    cheap = r4.verdict_corrected([_lv("r4_n8", 8, 0.0, 0.3, 40, rt_p50=0.32)])  # loss but not slower: no routing cost
    assert cheap["levels"][0]["routing_cost"] == "no" and cheap["conclusion"].startswith("当前实验条件下")
