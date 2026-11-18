"""实验 8 phase-3 correctness analysis: per-question comparison across arms (synthetic records and events only)."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("exp08_proto_check", HERE / "exp08_proto_check.py")
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)


def rec(rid, prompt, text="t", tokens=10, status="ok", audio=8.0):
    return {"request_id": rid, "status": status, "text": text, "completion_tokens": tokens, "output_audio_s": audio,
            "warmup": False, "payload": {"prompt_index": prompt}}


def hidden(rid, digest, rows=10):
    return {"request_id": rid, "event_name": "exp_thinker_hidden", "metadata": {"rows": rows, "hidden_sha256": digest}}


def vocode(rids, digests):
    return {"request_id": rids[0], "event_name": "exp_vocode", "metadata": {"request_ids": rids, "request_codec_sha256": digests}}


def arm(prompts, hidden_of=lambda p, i: f"h{p}", codec_of=lambda p, i: f"c{p}", text_of=lambda p, i: f"text{p}"):
    recs, events = [], []
    for i, p in enumerate(prompts):
        rid = f"r{i}"
        recs.append(rec(rid, p, text=text_of(p, i)))
        events.append(hidden(rid, hidden_of(p, i)))
        events.append(vocode([rid], [codec_of(p, i)]))
    return pc.outputs_by_request(recs, events)


PROMPTS = [0, 1, 2, 0, 1, 2]


def test_identical_arms_match_and_a_shifted_arm_does_not():
    arms = {"off": arm(PROMPTS), "on": arm(PROMPTS), "nosnap": arm(PROMPTS, hidden_of=lambda p, i: f"shifted{p}")}
    c = pc.compare(arms)
    assert c["per_arm"]["off"]["consistent_prompts"] == {"text": 3, "hidden": 3, "codec": 3}
    assert c["vs_off"]["on"] == {f: {"same": 3, "different": 0, "not_comparable": 0, "different_prompts": []} for f in pc.FIELDS}
    assert c["vs_off"]["nosnap"]["hidden"] == {"same": 0, "different": 3, "not_comparable": 0, "different_prompts": [0, 1, 2]}
    assert c["vs_off"]["nosnap"]["text"]["same"] == 3


def test_a_baseline_that_disagrees_with_itself_is_not_compared():
    # The two repeats of question 1 in the off arm differ, so question 1 cannot be judged in any arm.
    off = arm(PROMPTS, hidden_of=lambda p, i: f"h{p}-{i}" if p == 1 else f"h{p}")
    c = pc.compare({"off": off, "on": arm(PROMPTS)})
    assert c["per_arm"]["off"]["consistent_prompts"]["hidden"] == 2
    assert c["vs_off"]["on"]["hidden"] == {"same": 2, "different": 0, "not_comparable": 1, "different_prompts": []}


def test_missing_events_and_failed_requests_count_as_missing_not_as_equal():
    recs = [rec("r0", 0), rec("r1", 1, status="error"), rec("r2", 2)]
    events = [hidden("r0", "h0"), vocode(["r0", "r2"], ["c0", "c2"])]  # r2 has no hidden-state event
    out = pc.outputs_by_request(recs, events)
    assert out["r0"] == {"index": None, "prompt": 0, "status": "ok", "text": "t", "hidden": "h0", "codec": "c0", "rows": 10, "tokens": 10,
                         "audio_s": 8.0}
    assert out["r1"]["status"] == "error" and out["r1"]["hidden"] is None
    assert out["r2"]["hidden"] is None and out["r2"]["codec"] == "c2"
    s = pc.mixed_batch_summary(out)
    assert s["requests"] == 3 and s["errors"] == 1 and s["missing_hidden"] == 1 and s["missing_codec"] == 0
    assert s["rows_minus_tokens"] == {"0": 1}  # only requests with both numbers


def test_warmup_records_are_ignored():
    recs = [rec("r0", 0), {**rec("w0", 0), "warmup": True}]
    assert set(pc.outputs_by_request(recs, [])) == {"r0"}


def test_request_by_request_comparison_matches_the_kth_request_of_each_arm():
    # Both arms send the same question sequence; the baseline's own repeats differ, but the same way in every arm.
    def arm_seq(hidden):
        recs = [{**rec(f"r{i}", p), "index": i} for i, p in enumerate([0, 1, 0])]
        events = [hidden_ev for i, h in enumerate(hidden) for hidden_ev in (hidden_event(f"r{i}", h), vocode([f"r{i}"], [f"c{i}"]))]
        return pc.outputs_by_request(recs, events)

    off, on, shifted = arm_seq(["a", "b", "a2"]), arm_seq(["a", "b", "a2"]), arm_seq(["x", "b", "a2"])
    by_index = pc.compare_by_index({"off": off, "on": on, "nosnap": shifted})
    assert by_index["on"] == {"requests": 3, "same": {"text": 3, "hidden": 3, "codec": 3}, "different_indices": {"text": [], "hidden": [], "codec": []}}
    assert by_index["nosnap"]["same"]["hidden"] == 2 and by_index["nosnap"]["different_indices"]["hidden"] == [0]


def hidden_event(rid, digest):
    return hidden(rid, digest)
