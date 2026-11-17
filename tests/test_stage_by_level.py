"""scripts/analysis/stage_by_level.py on a tiny made-up experiment: two servers, two levels."""

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts/analysis"
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location("stage_by_level", HERE / "stage_by_level.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)

MS = 1_000_000  # ns per ms
T0 = 1_800_000_000 * 1_000_000_000


def events(rid, stage, start_ms, model_ms):
    """Server events of one request in one thinker or talker stage, starting at T0 + start_ms."""
    t = T0 + start_ms * MS
    names = [("stage_dispatch", 0), ("scheduler_queue_enter", 0), ("scheduler_prefill_start", 2), ("model_path_start", 2),
             ("scheduler_prefill_end", 12), ("model_path_end", 2 + model_ms), ("stage_complete", 3 + model_ms)]
    return [{"request_id": rid, "stage": stage, "event_name": n, "timestamp_ns": t + d * MS} for n, d in names]


def test_role_and_label_names():
    assert mod.role_of("s3/thinker@r1") == "thinker"
    assert mod.role_of("talker") == "talker"
    assert mod.label_role("s0/talker@r2.queue") == ("talker", "queue")
    assert mod.label_role("hop.s0/thinker@r0->s0/talker@r2") == ("hop", "thinker->talker")


def test_breakdown_groups_by_level_and_counts_replicas_per_server(tmp_path):
    exp = tmp_path / "exp"
    reqs, by_server = [], {"s0": [], "s1": []}
    # Level 1: one request at a time on s0. Level 2: two overlapping requests, one per server.
    plan = [("a", 1, "s0", 0, 100, 10, 1.0), ("b", 2, "s0", 1000, 200, 20, 2.0), ("c", 2, "s1", 1000, 200, 40, 2.0)]
    for rid, conc, srv, start, model, toks, audio in plan:
        by_server[srv] += events(rid, "thinker", start, model)
        reqs.append({"request_id": rid, "concurrency": conc, "status": "ok", "sent_ns": T0 + start * MS,
                     "first_text_ns": T0 + (start + 20) * MS, "done_ns": T0 + (start + model + 5) * MS,
                     "completion_tokens": toks, "output_audio_s": audio, "output_modalities": ["text"]})
    for srv, evs in by_server.items():
        d = exp / "raw" / "server_events" / "w" / srv
        d.mkdir(parents=True)
        (d / "events_thinker_1.jsonl").write_text("\n".join(json.dumps(e) for e in evs) + "\n")
    (exp / "raw" / "requests_w.jsonl").write_text("\n".join(json.dumps(r) for r in reqs) + "\n")
    rows = mod.breakdown(exp, "w")
    assert [r["level"] for r in rows] == [1.0, 2.0]
    one, two = rows
    assert one["n"] == 1 and two["n"] == 2
    assert one["stage_ms"][("thinker", "model_path")] == [100.0]
    assert sorted(two["stage_ms"][("thinker", "model_path")]) == [200.0, 200.0]
    # (model path - prefill) / (tokens - 1): (200 - 10) / 19 and (200 - 10) / 39.
    assert sorted(round(v, 3) for v in two["stage_ms"][("thinker", "ms_per_token")]) == [round(190 / 39, 3), 10.0]
    # Two thinkers (one per server); level 2 window = 203 ms with 200 ms in each replica -> ~0.99.
    assert two["replicas"] == {"thinker": 2}
    assert abs(two["eff_batch"]["thinker"] - 200 / 203) < 1e-6
