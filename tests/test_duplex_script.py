"""Duplex input timeline: seeded, fixed length, turns where the script says, 80 ms packets."""

import json

import numpy as np

from omni_research.duplex.audio import write_wav
from omni_research.duplex.script import PACKET_SAMPLES, SAMPLE_RATE, Clip, build_script, load_clips


def _clips():
    rng = np.random.default_rng(0)
    return [Clip(f"q{i}", (rng.standard_normal(int(SAMPLE_RATE * (2 + i))) * 2000).astype(np.int16), f"question {i}")
            for i in range(3)]


def test_script_is_seeded_fixed_length_and_places_turns_in_silence():
    clips = _clips()
    s = build_script(clips, duration_s=40.0, seed=5, session_index=1, lead_in_s=2.0, answer_window_s=5.0)
    assert len(s.pcm) == 40 * SAMPLE_RATE
    assert s.turns[0].start_sample == 2 * SAMPLE_RATE
    for a, b in zip(s.turns, s.turns[1:]):
        assert b.start_sample - a.end_sample == 5 * SAMPLE_RATE  # one answer window between turns
    assert s.turns[-1].end_sample + 5 * SAMPLE_RATE <= len(s.pcm)
    for t in s.turns:
        clip = next(c for c in clips if c.asset_id == t.asset_id)
        assert np.array_equal(s.pcm[t.start_sample:t.end_sample], clip.samples)
    again = build_script(clips, duration_s=40.0, seed=5, session_index=1, lead_in_s=2.0, answer_window_s=5.0)
    assert again.sha256() == s.sha256() and [t.asset_id for t in again.turns] == [t.asset_id for t in s.turns]
    other = build_script(clips, duration_s=40.0, seed=5, session_index=2, lead_in_s=2.0, answer_window_s=5.0)
    assert [t.asset_id for t in other.turns] != [t.asset_id for t in s.turns] or other.sha256() != s.sha256()


def test_packets_cover_the_input_and_turn_packets_are_right():
    s = build_script(_clips(), duration_s=12.3, seed=1, answer_window_s=3.0)
    pk = s.packets()
    assert sum(len(b) for _, _, b in pk) == len(s.pcm) * 2
    assert [p[0] for p in pk] == list(range(len(pk))) and pk[1][1] == 80.0
    assert all(len(b) == PACKET_SAMPLES * 2 for _, _, b in pk[:-1])
    t = s.turns[0]
    assert t.last_packet == (t.end_sample - 1) // PACKET_SAMPLES and t.first_packet == t.start_sample // PACKET_SAMPLES
    j = s.to_json()
    assert j["turns"][0]["last_packet"] == t.last_packet and j["pcm_sha256"] == s.sha256()
    json.dumps(j)


def test_barge_in_turns_follow_their_question_after_the_offset():
    s = build_script(_clips(), duration_s=60.0, seed=2, answer_window_s=4.0, barge_in_every=2, barge_in_offset_s=1.5)
    kinds = [t.kind for t in s.turns]
    assert "barge_in" in kinds
    for i, t in enumerate(s.turns):
        if t.kind == "barge_in":
            q = s.turns[i - 1]
            assert q.kind == "question" and t.interrupts_turn == q.index
            assert t.start_sample - q.end_sample == int(1.5 * SAMPLE_RATE)
    questions = [t for t in s.turns if t.kind == "question"]
    assert all((k % 2 == 1) == any(b.interrupts_turn == q.index for b in s.turns if b.kind == "barge_in")
               for k, q in enumerate(questions))


def test_load_clips_reads_a_16k_pool(tmp_path):
    rows = []
    for i, c in enumerate(_clips()):
        write_wav(tmp_path / f"{c.asset_id}.wav", c.samples, SAMPLE_RATE)
        rows.append({"asset_id": c.asset_id, "kind": "audio", "path": f"{c.asset_id}.wav", "tags": ["agent_question"],
                     "transcript": c.transcript})
    rows.append({"asset_id": "p0", "kind": "audio", "path": "q0.wav", "tags": ["passage"]})
    (tmp_path / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    got = load_clips(tmp_path)
    assert [c.asset_id for c in got] == ["q0", "q1", "q2"] and got[1].transcript == "question 1"
