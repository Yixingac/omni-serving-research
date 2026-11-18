#!/usr/bin/env python3
"""实验 11 正确性检查：A（同步）、C（快照 + 旁路复制）、A2（A 的重复），按登记的规则判定是否通过。

    python scripts/analysis/exp11_check.py --data-root <results root> --run A=20261118_exp11_check_A_run01 \
        --run C=20261118_exp11_check_C_run01 --run A2=20261118_exp11_check_A2_run01 --out research/01_性能瓶颈/实验11_Thinker异步优化

Per request (paired across runs by its position in the phase): status, text tokens, the thinker's hidden-state rows,
hash and fingerprint (row norms and 4 random projections), the hash of the output token ids, and the codec token hash.
Rules (hypotheses.md): at concurrency 1, C equals A request by request (text, hidden hash, codec hash); C has no
errors outside the timeout phase; every finished speech request of C has as many hidden rows as text tokens; for
requests with identical output tokens, the share whose fingerprints differ by more than 0.05 (relative to the row
norm) is not more than 5 points above the same share for A2 against A.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

TOLERANCE = 0.05
MARGIN = 0.05
HIDDEN_SIZE = 4096  # the MiniCPM-o 4.5 thinker's hidden size
TIMEOUT_PHASE = "chk_timeout"
EXACT_PHASE = "chk_c1"
EXACT_REQUESTS = 18
# every phase of the check runs (scripts/slurm/make_plan_exp09_11.py, exp11_check_entries)
EXPECTED_PHASES = ("chk_c1", "chk_c8", "chk_c32", "chk_c64", "chk_mix_text", "chk_mix_long", "chk_mix_seed", "chk_timeout",
                   "chk_after", "chk_w3xl")


def fp_deviation(fa: dict[str, Any], fb: dict[str, Any]) -> float:
    """Largest difference over rows: the norm relative to the reference's (fb) row norm, each projection relative to
    the size of a projection on a random unit-variance direction, norm / sqrt(hidden size) (supplementary
    registration 4). Relative to the norm itself, a row swapped for another of similar norm would pass."""
    na, nb = fa["norm"], fb["norm"]
    if len(na) != len(nb):
        return math.inf
    dev = 0.0
    for i, (x, y) in enumerate(zip(na, nb)):
        if y <= 0:
            continue
        scale = y / math.sqrt(HIDDEN_SIZE)
        d = max(abs(x - y) / y, max(abs(u - v) for u, v in zip(fa["proj"][i], fb["proj"][i])) / scale)
        dev = max(dev, d)
    return dev


def by_index(reqs: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {r["index"]: r for r in reqs}


def exact(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, int]:
    ia, ib = by_index(a), by_index(b)
    keys = sorted(set(ia) & set(ib))
    same = lambda k, f: ia[k].get(f) is not None and ia[k].get(f) == ib[k].get(f)  # noqa: E731
    text = sum(same(k, "text") if ia[k].get("text") is not None else same(k, "out_sha") for k in keys)
    return {"pairs": len(keys), "text": text, "hidden": sum(same(k, "hidden") for k in keys), "codec": sum(same(k, "codec") for k in keys)}


def fingerprints(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    ia, ib = by_index(a), by_index(b)
    devs = []
    for k in sorted(set(ia) & set(ib)):
        x, y = ia[k], ib[k]
        if x.get("status") != "ok" or y.get("status") != "ok" or not x.get("fp") or not y.get("fp"):
            continue
        if x.get("out_sha") is None or x.get("out_sha") != y.get("out_sha"):
            continue
        devs.append(fp_deviation(y["fp"], x["fp"]))  # a (the first argument) is the reference
    over = sum(d > TOLERANCE for d in devs)
    return {"pairs": len(devs), "over_tolerance": over, "share_over": over / len(devs) if devs else None,
            "max_deviation": max(devs) if devs else None}


def rows_check(reqs: list[dict[str, Any]]) -> dict[str, int]:
    """Every finished speech request needs its hidden-state event, with as many rows as text tokens; a missing event
    is a mismatch (the runner returns early when the hidden states are lost)."""
    seen = [r for r in reqs if r.get("status") == "ok" and r.get("speech", True)]
    return {"checked": len(seen), "mismatched": sum(r.get("rows") is None or r["rows"] != r.get("tokens") for r in seen)}


def gate(runs: dict[str, dict[str, list[dict[str, Any]]]], meta: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """The registered rules, with evidence required (supplementary registration 4): all three runs complete, with
    status ok and no server restart, every phase present; 18 exact pairs at concurrency 1; no error in C except
    client timeouts in the timeout phase; every finished speech request of C with its hidden rows; and in every
    other phase, request pairs with identical text for both C vs A and A2 vs A before the shares are compared."""
    reasons: list[str] = []
    for arm in ("A", "C", "A2"):
        if arm not in runs:
            reasons.append(f"{arm}: run missing")
            continue
        missing = [p for p in EXPECTED_PHASES if p not in runs[arm]]
        if missing:
            reasons.append(f"{arm}: phases missing {missing}")
        m = (meta or {}).get(arm)
        if m is not None and m.get("status") != "ok":
            reasons.append(f"{arm}: run status {m.get('status')}")
        if m is not None and m.get("restarts"):
            reasons.append(f"{arm}: server restarted ({[x.get('where') for x in m['restarts']]})")
    a, c, a2 = runs.get("A", {}), runs.get("C", {}), runs.get("A2", {})
    out: dict[str, Any] = {"phases": {}}
    if EXACT_PHASE in a and EXACT_PHASE in c:
        e = exact(a[EXACT_PHASE], c[EXACT_PHASE])
        out["c1_exact"] = e
        if not (e["pairs"] == EXACT_REQUESTS and e["text"] == e["hidden"] == e["codec"] == e["pairs"]):
            reasons.append(f"{EXACT_PHASE}: C differs from A or not all {EXACT_REQUESTS} requests paired ({e})")
    for name in EXPECTED_PHASES:
        if name not in c:
            continue
        reqs = c[name]
        errors = sum(r.get("status") != "ok" and not (name == TIMEOUT_PHASE and r.get("status") == "timeout") for r in reqs)
        row: dict[str, Any] = {"errors": errors, "rows": rows_check(reqs)}
        if errors:
            reasons.append(f"{name}: {errors} errors in C")
        if row["rows"]["mismatched"]:
            reasons.append(f"{name}: {row['rows']['mismatched']} finished speech requests without hidden rows == text tokens in C")
        if name not in (EXACT_PHASE, TIMEOUT_PHASE):
            fc = fingerprints(a[name], reqs) if name in a else None
            fa2 = fingerprints(a[name], a2[name]) if name in a and name in a2 else None
            row["C_vs_A"], row["A2_vs_A"] = fc, fa2
            if not fc or not fc["pairs"]:
                reasons.append(f"{name}: no request pairs with identical text for C vs A")
            elif not fa2 or not fa2["pairs"]:
                reasons.append(f"{name}: no request pairs with identical text for A2 vs A")
            elif fc["share_over"] > fa2["share_over"] + MARGIN:
                reasons.append(f"{name}: fingerprints over tolerance {fc['share_over']:.2f} for C vs {fa2['share_over']:.2f} for A2")
        out["phases"][name] = row
    out["pass"] = not reasons
    out["reasons"] = reasons
    return out


def load_run(exp: Path) -> dict[str, list[dict[str, Any]]]:
    res = json.loads((exp / "result.json").read_text())
    phases: dict[str, list[dict[str, Any]]] = {}
    for ph in res.get("phases", []):
        name = ph["phase"]
        if not name.startswith("chk_"):
            continue
        recs = [json.loads(x) for x in open(exp / "raw" / f"requests_{name}.jsonl") if x.strip()]
        recs = [r for r in recs if not r.get("warmup")]
        recs.sort(key=lambda r: (r.get("index") if r.get("index") is not None else 0, r.get("sent_ns") or 0))
        hidden: dict[str, dict[str, Any]] = {}
        output: dict[str, dict[str, Any]] = {}
        codec: dict[str, str] = {}
        for path in sorted((exp / "raw" / "server_events" / name).rglob("*.jsonl")):
            with open(path) as fh:
                for line in fh:
                    if "exp_thinker_hidden" not in line and "exp_thinker_output" not in line and "request_codec_sha256" not in line:
                        continue
                    e = json.loads(line)
                    md = e.get("metadata") or {}
                    if e.get("event_name") == "exp_thinker_hidden":
                        hidden[e["request_id"]] = md
                    elif e.get("event_name") == "exp_thinker_output":  # set after the scheduler assigns the output ids
                        output[e["request_id"]] = md
                    elif e.get("event_name") == "exp_vocode":
                        codec.update(dict(zip(md.get("request_ids", []), md.get("request_codec_sha256", []))))
        rows = []
        for i, r in enumerate(recs):
            h = hidden.get(r.get("request_id")) or {}
            o = output.get(r.get("request_id")) or {}
            rows.append({"index": r.get("index") if r.get("index") is not None else i, "rid": r.get("request_id"), "status": r.get("status"),
                         "tokens": r.get("completion_tokens"), "audio_s": r.get("output_audio_s"), "text": r.get("text"),
                         "speech": "audio" in (r.get("output_modalities") or []),
                         "rows": h.get("rows"), "hidden": h.get("hidden_sha256"), "fp": h.get("hidden_fingerprint"),
                         "out_sha": o.get("output_ids_sha256"), "out_len": o.get("output_len"), "codec": codec.get(r.get("request_id")),
                         "workload": r.get("workload_id")})
        phases[name] = rows
    return phases


def lengths(reqs: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in reqs if r.get("status") == "ok"]
    toks = sorted(r["tokens"] for r in ok if r.get("tokens"))
    audio = sorted(r["audio_s"] for r in ok if r.get("audio_s"))
    return {"n": len(ok), "tokens_min": toks[0] if toks else None, "tokens_max": toks[-1] if toks else None,
            "tokens_mean": sum(toks) / len(toks) if toks else None, "audio_s_max": audio[-1] if audio else None,
            "audio_s_mean": sum(audio) / len(audio) if audio else None,
            "codec_at_talker_cap": sum(1 for a in audio if a >= 2048 * 0.04 - 1e-6)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run", action="append", required=True, help="ARM=run_dir, ARM in A, C, A2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root).expanduser() / "experiments"
    runs, names, meta = {}, {}, {}
    for spec in a.run:
        arm, name = spec.split("=", 1)
        runs[arm], names[arm] = load_run(root / name), name
        res = json.loads((root / name / "result.json").read_text())
        meta[arm] = {"status": res.get("status"), "restarts": res.get("server_restarts") or []}
    g = gate(runs, meta)
    summary = {"generated_by": "scripts/analysis/exp11_check.py", "runs": names, "run_meta": meta, "gate": g,
               "lengths": {arm: {ph: lengths(r) for ph, r in phases.items() if ph in ("chk_w3xl", "chk_mix_long", "chk_c32")}
                           for arm, phases in runs.items()},
               "errors": {arm: {ph: sum(x.get("status") != "ok" for x in r) for ph, r in phases.items()} for arm, phases in runs.items()}}
    out = Path(a.out) / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "check.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str) + "\n")
    print("PASS" if g["pass"] else f"FAIL: {g['reasons']}")
    print(out / "check.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
