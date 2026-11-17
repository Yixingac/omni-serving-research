#!/usr/bin/env python3
"""实验 8 第三阶段的正确性检查：同样的确定性请求，原型关（off）、开（on）、开但不做快照（nosnap），逐个问题比较输出。

    python scripts/analysis/exp08_proto_check.py --data-root <results root> \
        --run off=20261117_exp08_proto_check_off_run01 --run on=... --run nosnap=... --out research/.../实验08_Thinker_Lookahead

check_c1 (concurrency 1, thinker and talker greedy): for every question, the text, the hash of the thinker hidden
states handed to the talker (exp_thinker_hidden) and the hash of the talker's codec tokens (exp_vocode) must be the
same in every repeat of the off arm (else the question is not comparable) and then equal in each other arm.
check_c8 (mixed batches): errors, missing hashes, and hidden rows minus completion tokens per request.
Writes results/proto_check.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

FIELDS = ("text", "hidden", "codec")


def outputs_by_request(recs: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Measured requests of one phase with their text, hidden-state hash and codec hash (None when missing)."""
    hidden: dict[str, dict[str, Any]] = {}
    codec: dict[str, str] = {}
    for e in events:
        m = e.get("metadata") or {}
        if e.get("event_name") == "exp_thinker_hidden":
            hidden[e["request_id"]] = m
        elif e.get("event_name") == "exp_vocode" and m.get("request_codec_sha256") is not None:
            codec.update(zip(m["request_ids"], m["request_codec_sha256"]))
    out = {}
    for r in recs:
        if r.get("warmup"):
            continue
        rid = r["request_id"]
        ok = r.get("status") == "ok"
        h = hidden.get(rid) or {}
        out[rid] = {"prompt": (r.get("payload") or {}).get("prompt_index"), "status": r.get("status"),
                    "text": r.get("text") if ok else None, "hidden": h.get("hidden_sha256"), "codec": codec.get(rid),
                    "rows": h.get("rows"), "tokens": r.get("completion_tokens"), "audio_s": r.get("output_audio_s")}
    return out


def by_prompt(outputs: dict[str, dict[str, Any]], field: str) -> dict[int, set]:
    values: dict[int, set] = defaultdict(set)
    for o in outputs.values():
        values[o["prompt"]].add(o[field])
    return values


def compare(arms: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Per arm: questions whose repeats agree (no missing value). Versus off: a question is comparable when the
    off arm's repeats agree; it is the same when the other arm's repeats all equal that value."""
    per_arm = {}
    for name, outputs in arms.items():
        per_arm[name] = {"requests": len(outputs), "errors": sum(1 for o in outputs.values() if o["status"] != "ok"),
                         "prompts": len({o["prompt"] for o in outputs.values()}),
                         "consistent_prompts": {f: sum(1 for vs in by_prompt(outputs, f).values() if len(vs) == 1 and None not in vs)
                                                for f in FIELDS}}
    vs_off: dict[str, Any] = {}
    for name, outputs in arms.items():
        if name == "off":
            continue
        vs_off[name] = {}
        for f in FIELDS:
            ref, mine = by_prompt(arms["off"], f), by_prompt(outputs, f)
            res = {"same": 0, "different": 0, "not_comparable": 0, "different_prompts": []}
            for p in sorted(set(ref) | set(mine), key=lambda x: (x is None, x)):
                if len(ref.get(p, set())) != 1 or None in ref[p] or None in mine.get(p, {None}):
                    res["not_comparable"] += 1
                elif mine[p] == ref[p]:
                    res["same"] += 1
                else:
                    res["different"] += 1
                    res["different_prompts"].append(p)
            vs_off[name][f] = res
    return {"per_arm": per_arm, "vs_off": vs_off}


def mixed_batch_summary(outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ok = [o for o in outputs.values() if o["status"] == "ok"]
    return {"requests": len(outputs), "errors": len(outputs) - len(ok), "no_audio": sum(1 for o in ok if not o["audio_s"]),
            "missing_hidden": sum(1 for o in ok if o["hidden"] is None), "missing_codec": sum(1 for o in ok if o["codec"] is None),
            "rows_minus_tokens": {str(k): v for k, v in sorted(Counter(o["rows"] - o["tokens"] for o in ok
                                                                       if o["rows"] is not None and o["tokens"] is not None).items())}}


def read_phase(exp: Path, phase: str) -> dict[str, dict[str, Any]]:
    from omni_research.analysis.experiment_analysis import read_jsonl

    recs = read_jsonl(exp / "raw" / f"requests_{phase}.jsonl")
    events = []
    for path in sorted((exp / "raw" / "server_events" / phase).rglob("*.jsonl")):
        with open(path) as fh:
            for line in fh:
                if "exp_thinker_hidden" in line or "exp_vocode" in line:
                    events.append(json.loads(line))
    return outputs_by_request(recs, events)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", action="append", required=True, help="arm=run directory name (arms: off, on, nosnap)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    runs = dict(r.split("=", 1) for r in a.run)
    c1 = {arm: read_phase(root / name, "check_c1") for arm, name in runs.items()}
    c8 = {arm: read_phase(root / name, "check_c8") for arm, name in runs.items()}
    summary = {"generated_by": "scripts/analysis/exp08_proto_check.py", "runs": runs, "check_c1": compare(c1),
               "check_c8": {arm: mixed_batch_summary(o) for arm, o in c8.items()},
               "c1_requests": {arm: sorted(o.values(), key=lambda x: (x["prompt"] is None, x["prompt"])) for arm, o in c1.items()}}
    out = Path(a.out) / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "proto_check.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    print(out / "proto_check.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
