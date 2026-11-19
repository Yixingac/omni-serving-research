#!/usr/bin/env python3
"""C1/C2 质量门槛：比较各方案在质量集（带文字稿的合成语音）上的波形误差、MCD 和 ASR 回读，按事先登记的规则判定。

    python scripts/microbench/quality_eval.py c1 --root <c1 dir> [--asr-url http://127.0.0.1:18090 --asr-model NAME] --out c1_quality.json
    python scripts/microbench/quality_eval.py c2 --root <c2 dir> [--asr-url ...] --out c2_quality.json

The rules are registered in research/next_steps/presentation_validation/c1_c2_design.md before any run:

C1 (flow): reference = REF (FP32 dense flow, one request per call); baseline = D (production); floors = D's literal
repeat (same batches again; the registered rule) and D alone (the same requests one per call: the variation that batch
composition already causes in production).
  - registered rule, reported as written: P1 vs D energy and RMS differences <= those of D's repeat;
  - gate (erratum E-C1, see the design note): every waveform 960 samples per codec token; mean MCD(P1, REF) <= mean
    MCD(D, REF) + 0.1 dB; ASR exact matches(P1) >= min(D, D alone) - 1 and pooled WER(P1) <= max(D, D alone) + 0.5 pt.
C2 (HiFT): reference = A (production HiFT) with the same excitation noise (seed s1); floor = A with seed s2.
  - B (weight-norm folding): folded weights within 1 ulp and eager outputs bit-identical (or rel RMS <= 1e-6 if the
    weights differ by 1 ulp);
  - C (channels-last): eager rel RMS <= 1e-4 and MCD <= 0.01 dB against eager A;
  - D (FP16 body): all samples finite; mean MCD(D, A) < 0.1 dB; ASR exact matches(D) >= min(A, A s2) - 1 and pooled WER(D)
    <= max(A, A s2) + 0.5 pt. Combinations take the gate of their least exact part.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from omni_research.analysis import audio_quality as aq  # noqa: E402

MCD_MARGIN_DB = 0.1
WER_MARGIN = 0.005
EXACT_MARGIN = 1
C2_LAYOUT_REL_RMS, C2_LAYOUT_MCD = 1e-4, 0.01
C2_FOLD_REL_RMS = 1e-6


def wav_bytes(x: np.ndarray, sr: int = aq.SAMPLE_RATE) -> bytes:
    pcm = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((pcm * 32767.0).round().astype("<i2").tobytes())
    return buf.getvalue()


def transcribe(url: str, model: str, audio: bytes, timeout: float = 300.0) -> str:
    """POST /v1/audio/transcriptions (OpenAI style multipart) to an sglang-omni ASR server; returns the text."""
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in (("model", model), ("response_format", "json")):
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
                 "Content-Type: audio/wav\r\n\r\n".encode() + audio + b"\r\n")
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{url.rstrip('/')}/v1/audio/transcriptions", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode()).get("text", "")


def load_arm(root: Path, arm: str) -> dict[str, Any] | None:
    p = root / arm / "arm_result.json"
    return json.loads(p.read_text()) if p.exists() else None


def items_of(res: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    return {f["asset_id"]: f for f in (res.get("quality") or {}).get(label, [])}


def compare(root: Path, cand_arm: str, cand: dict[str, dict[str, Any]], ref_arm: str, ref: dict[str, dict[str, Any]],
            key: str = "wave") -> dict[str, Any]:
    """Per-request waveform (or mel) errors and MCD of candidate against reference, and their means."""
    rows = []
    for aid, f in sorted(cand.items()):
        if aid not in ref:
            continue
        x = np.load(root / cand_arm / f[key])
        y = np.load(root / ref_arm / ref[aid][key])
        if x.shape != y.shape:
            rows.append({"asset_id": aid, "shape_mismatch": [list(x.shape), list(y.shape)]})
            continue
        row = {"asset_id": aid, **aq.waveform_errors(x, y), "finite": bool(np.isfinite(x).all())}
        if key == "wave":
            row["mcd_db"] = aq.mcd_db(x, y)
            row["length_ok"] = x.size == 960 * f["tokens"]
        rows.append(row)
    ok = [r for r in rows if "shape_mismatch" not in r]
    agg = {k: aq.mean(r[k] for r in ok) for k in ("rel_rms", "energy_rel_diff", "max_abs", "mcd_db") if ok and k in ok[0]}
    agg.update(n=len(ok), shape_mismatches=len(rows) - len(ok), all_finite=all(r["finite"] for r in ok),
               bit_identical_all=all(r["bit_identical"] for r in ok) if ok else False,
               length_ok_all=all(r.get("length_ok", True) for r in ok), max_rel_rms=max((r["rel_rms"] for r in ok), default=None))
    return {"summary": agg, "rows": rows}


def asr_block(root: Path, arm: str, files: dict[str, dict[str, Any]], url: str | None, model: str, cache: dict[str, str]) -> dict[str, Any] | None:
    if not url:
        return None
    hyps, refs, texts = [], [], {}
    for aid, f in sorted(files.items()):
        k = f"{arm}/{f['wave']}"
        if k not in cache:
            cache[k] = transcribe(url, model, wav_bytes(np.load(root / arm / f["wave"])))
        hyps.append(cache[k])
        refs.append(f["transcript"])
        texts[aid] = cache[k]
    return {**aq.asr_agreement(hyps, refs), "texts": texts}


def asr_pass(cand: dict[str, Any] | None, floors: list[dict[str, Any] | None]) -> bool | None:
    floors = [f for f in floors if f]
    if cand is None or not floors:
        return None
    return (cand["exact_matches"] >= min(f["exact_matches"] for f in floors) - EXACT_MARGIN
            and cand["wer"] <= max(f["wer"] for f in floors) + WER_MARGIN)


def c1(a: argparse.Namespace) -> dict[str, Any]:
    root, cache = Path(a.root), {}
    ref_res, d_res = load_arm(root, "REF"), load_arm(root, "D")
    if ref_res is None or d_res is None:
        raise SystemExit("C1 needs the REF and D arms")
    ref, d = items_of(ref_res, "one"), items_of(d_res, "batched")
    out: dict[str, Any] = {"vs_ref": {}, "vs_d": {}, "asr": {}}
    out["vs_ref"]["D"] = compare(root, "D", d, "REF", ref)
    out["vs_ref"]["D_alone"] = compare(root, "D", items_of(d_res, "alone"), "REF", ref)
    out["floor_repeat_vs_d"] = compare(root, "D", items_of(d_res, "repeat"), "D", d)
    out["floor_alone_vs_d"] = compare(root, "D", items_of(d_res, "alone"), "D", d)
    out["mel_vs_ref"] = {"D": compare(root, "D", d, "REF", ref, key="mel")["summary"]}
    out["asr"]["REF"] = asr_block(root, "REF", ref, a.asr_url, a.asr_model, cache)
    out["asr"]["D"] = asr_block(root, "D", d, a.asr_url, a.asr_model, cache)
    out["asr"]["D_alone"] = asr_block(root, "D", items_of(d_res, "alone"), a.asr_url, a.asr_model, cache)
    gates: dict[str, Any] = {}
    for arm in ("D_ng", "P0", "P1", "P2", "P1h"):
        res = load_arm(root, arm)
        if res is None:
            continue
        c = items_of(res, "batched")
        out["vs_ref"][arm] = compare(root, arm, c, "REF", ref)
        out["vs_d"][arm] = compare(root, arm, c, "D", d)
        out["mel_vs_ref"][arm] = compare(root, arm, c, "REF", ref, key="mel")["summary"]
        out["asr"][arm] = asr_block(root, arm, c, a.asr_url, a.asr_model, cache)
        vs_ref, base = out["vs_ref"][arm]["summary"], out["vs_ref"]["D"]["summary"]
        lit = out["floor_repeat_vs_d"]["summary"]
        registered = (out["vs_d"][arm]["summary"]["energy_rel_diff"] <= lit["energy_rel_diff"]
                      and out["vs_d"][arm]["summary"]["rel_rms"] <= lit["rel_rms"])
        mcd_ok = vs_ref["mcd_db"] <= base["mcd_db"] + MCD_MARGIN_DB
        asr_ok = asr_pass(out["asr"][arm], [out["asr"]["D"], out["asr"]["D_alone"]])
        gates[arm] = {"registered_rule_literal": registered, "length_ok": vs_ref["length_ok_all"], "finite": vs_ref["all_finite"],
                      "mcd_db": vs_ref["mcd_db"], "mcd_db_D": base["mcd_db"], "mcd_ok": mcd_ok, "asr_ok": asr_ok,
                      "pass": bool(vs_ref["length_ok_all"] and vs_ref["all_finite"] and mcd_ok and asr_ok is True)}
    out["gates"] = gates
    out["c1_gate"] = {"candidate": "P1", **gates.get("P1", {"pass": False, "reason": "P1 not run"})}
    return out


def c2(a: argparse.Namespace) -> dict[str, Any]:
    root, cache = Path(a.root), {}
    a_res = load_arm(root, "A")
    if a_res is None:
        raise SystemExit("C2 needs arm A")
    a1, a2, a_eager = items_of(a_res, "s1"), items_of(a_res, "s2"), items_of(a_res, "eager_s1")
    out: dict[str, Any] = {"floor_seed": compare(root, "A", a2, "A", a1), "arms": {}, "asr": {}}
    out["asr"]["A"] = asr_block(root, "A", a1, a.asr_url, a.asr_model, cache)
    out["asr"]["A_s2"] = asr_block(root, "A", a2, a.asr_url, a.asr_model, cache)
    gates: dict[str, Any] = {}
    for p in sorted(root.glob("*/arm_result.json")):
        arm = p.parent.name
        if arm == "A":
            continue
        res = json.loads(p.read_text())
        comp = compare(root, arm, items_of(res, "s1"), "A", a1)
        entry: dict[str, Any] = {"vs_A": comp["summary"]}
        eager = items_of(res, "eager_s1")
        if eager and a_eager:
            entry["eager_vs_A_eager"] = compare(root, arm, eager, "A", a_eager)["summary"]
        parts = set(arm)
        g: dict[str, Any] = {"finite": comp["summary"]["all_finite"]}
        if "D" in parts:
            out["asr"][arm] = asr_block(root, arm, items_of(res, "s1"), a.asr_url, a.asr_model, cache)
            g.update(kind="half", mcd_db=comp["summary"]["mcd_db"], mcd_ok=comp["summary"]["mcd_db"] < MCD_MARGIN_DB,
                     asr_ok=asr_pass(out["asr"][arm], [out["asr"]["A"], out["asr"]["A_s2"]]))
            g["pass"] = bool(g["finite"] and g["mcd_ok"] and g["asr_ok"] is True)
        elif "C" in parts:
            e = entry.get("eager_vs_A_eager") or {}
            g.update(kind="layout", eager_rel_rms=e.get("rel_rms"), eager_mcd_db=e.get("mcd_db"))
            g["pass"] = bool(g["finite"] and e and e["rel_rms"] <= C2_LAYOUT_REL_RMS and e["mcd_db"] <= C2_LAYOUT_MCD)
        elif "B" in parts:
            e = entry.get("eager_vs_A_eager") or {}
            ulp = (res.get("fold_check") or {}).get("max_ulp")
            g.update(kind="fold", weights_max_ulp=ulp, eager_bit_identical=e.get("bit_identical_all"), eager_rel_rms=e.get("rel_rms"))
            g["pass"] = bool(g["finite"] and ulp is not None and ulp <= 1 and e
                             and (e["bit_identical_all"] or (ulp <= 1 and e["rel_rms"] <= C2_FOLD_REL_RMS)))
        entry["gate"] = g
        gates[arm] = g
        out["arms"][arm] = entry
    out["gates"] = gates
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("which", choices=("c1", "c2"))
    ap.add_argument("--root", required=True)
    ap.add_argument("--asr-url")
    ap.add_argument("--asr-model", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = c1(a) if a.which == "c1" else c2(a)
    res["rules"] = {"mcd_margin_db": MCD_MARGIN_DB, "wer_margin": WER_MARGIN, "exact_margin": EXACT_MARGIN,
                    "c2_layout": [C2_LAYOUT_REL_RMS, C2_LAYOUT_MCD], "c2_fold_rel_rms": C2_FOLD_REL_RMS,
                    "asr": {"url_given": bool(a.asr_url), "model": a.asr_model}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o)))
    print(json.dumps(res.get("c1_gate") or res.get("gates"), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
