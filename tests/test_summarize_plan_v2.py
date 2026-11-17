"""SLO rules and layout roles of scripts/analysis/summarize_plan_v2.py (no data files needed)."""

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("summarize_plan_v2", Path(__file__).resolve().parents[1] / "scripts/analysis/summarize_plan_v2.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def rec(ttft_s, done_s, audio_s=0.0, speech=False, status="ok"):
    t0 = 1_000_000_000
    return {"status": status, "sent_ns": t0, "first_text_ns": t0 + int(ttft_s * 1e9), "done_ns": t0 + int(done_s * 1e9),
            "first_audio_ns": t0 + int(done_s * 1e9) if speech else None, "last_audio_ns": t0 + int(done_s * 1e9) if speech else None,
            "output_audio_s": audio_s, "output_modalities": ["text", "audio"] if speech else ["text"]}


def test_slo_speech_uses_e2e_per_audio_second_and_ttft():
    assert mod.slo_ok(rec(0.1, 6.0, audio_s=14.0, speech=True))       # 0.43 s per audio second
    assert not mod.slo_ok(rec(0.1, 8.0, audio_s=14.0, speech=True))   # 0.57
    assert not mod.slo_ok(rec(1.5, 3.0, audio_s=14.0, speech=True))   # TTFT over 1 s
    assert not mod.slo_ok(rec(0.1, 3.0, audio_s=14.0, speech=True, status="error"))


def test_slo_text_only_needs_ttft():
    assert mod.slo_ok(rec(0.9, 30.0))
    assert not mod.slo_ok(rec(1.1, 1.2))


def test_slo_capacity_stops_at_first_failing_level():
    assert mod.slo_capacity([(10, 1.0), (20, 0.99), (30, 0.7), (40, 0.96)]) == (20, True)
    assert mod.slo_capacity([(10, 1.0), (20, 1.0)]) == (20, False)
    assert mod.slo_capacity([(10, 0.5)]) == (None, True)


def test_layout_roles_and_names():
    assert mod.layout_roles("s2x6") == {"thinker": [0, 1], "talker": [2, 3, 4, 5, 6, 7]}
    assert mod.layout_roles("p4") == {"thinker": [0, 2, 4, 6], "talker": [1, 3, 5, 7]}
    assert mod.layout_roles("r8") == {"full replica": list(range(8))}
    assert [mod.eid_layout(e) for e in ("v2_a_s2x6_run1", "v2b_aoh_r8", "v2_b1_p4", "v2b_a_s2x6_c03", "v2_a_r8_run2")] == ["s2x6", "r8", "p4", "s2x6", "r8"]
