#!/usr/bin/env python3
"""Build the duplex input pool (16 kHz mono PCM16) from the spoken questions of an existing asset pool.

    python scripts/assets/make_duplex_speech.py --data-root <PROJECT_DATA_ROOT> [--src-pool synthetic-v1] \
        [--out-pool duplex-v1] [--tag agent_question]

The synthetic-v1 pool already holds 24 spoken questions (24 kHz PCM16 WAVs, made on 2026-11-11 by
Qwen/Qwen3-TTS-12Hz-0.6B-Base@5d83992 through the runner's generate_assets phase; the W2/W3 chat
workloads use the same files). The full-duplex endpoint takes 16 kHz, so each clip is resampled by 2/3
with the deterministic numpy filter of omni_research.duplex.audio and its leading/trailing silence below
-45 dBFS is cut (100 ms kept on each side). No GPU, no network, no randomness: the same source files give
the same bytes, recorded as SHA-256 per clip in the new manifest. An existing output pool is never
overwritten with different content (--force replaces it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from omni_research.duplex.audio import read_wav, resample_pcm16, trim_silence, wav_bytes

OUT_RATE = 16000


def build(src_pool: Path, tag: str, trim_dbfs: float, margin_ms: float) -> tuple[list[dict], dict[str, bytes]]:
    rows, files = [], {}
    for line in (src_pool / "manifest.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") != "audio" or tag not in (r.get("tags") or []):
            continue
        src_bytes = (src_pool / r["path"]).read_bytes()
        samples, rate = read_wav(src_bytes)
        y = resample_pcm16(samples, rate, OUT_RATE)
        y, start, end = trim_silence(y, OUT_RATE, threshold_dbfs=trim_dbfs, margin_ms=margin_ms)
        data = wav_bytes(y, OUT_RATE)
        rel = f"audio16k/{r['asset_id']}.wav"
        files[rel] = data
        rows.append({"asset_id": r["asset_id"], "kind": "audio", "path": rel, "duration_s": round(len(y) / OUT_RATE, 4),
                     "sample_rate": OUT_RATE, "tags": list(r.get("tags") or []), "transcript": r.get("transcript", ""),
                     "question": r.get("question"), "source": f"{r.get('source')}+resampled16k+trimmed",
                     "src_asset_id": r["asset_id"], "src_sha256": hashlib.sha256(src_bytes).hexdigest(),
                     "src_duration_s": round(len(samples) / rate, 4), "trim_samples_16k": [start, end],
                     "pcm_sha256": hashlib.sha256(y.astype("<i2").tobytes()).hexdigest()})
    rows.sort(key=lambda x: x["asset_id"])
    return rows, files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--src-pool", default="synthetic-v1")
    ap.add_argument("--out-pool", default="duplex-v1")
    ap.add_argument("--tag", default="agent_question")
    ap.add_argument("--trim-dbfs", type=float, default=-45.0)
    ap.add_argument("--margin-ms", type=float, default=100.0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    assets = Path(a.data_root).expanduser() / "assets"
    rows, files = build(assets / a.src_pool, a.tag, a.trim_dbfs, a.margin_ms)
    if not rows:
        print(f"no {a.tag!r} audio in {assets / a.src_pool}", file=sys.stderr)
        return 1
    manifest = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    out = assets / a.out_pool
    if out.exists():
        same = (out / "manifest.jsonl").exists() and (out / "manifest.jsonl").read_text() == manifest and all(
            (out / p).exists() and (out / p).read_bytes() == b for p, b in files.items())
        if same:
            print(f"{out} already holds these {len(rows)} clips; nothing to do")
            return 0
        if not a.force:
            print(f"{out} exists with different content; use --force to replace it", file=sys.stderr)
            return 2
        shutil.rmtree(out)
    (out / "audio16k").mkdir(parents=True)
    for p, b in files.items():
        (out / p).write_bytes(b)
    (out / "manifest.jsonl").write_text(manifest)
    (out / "generation.json").write_text(json.dumps({
        "generated_by": "scripts/assets/make_duplex_speech.py", "src_pool": a.src_pool, "tag": a.tag,
        "sample_rate": OUT_RATE, "resampler": "omni_research.duplex.audio.resample_pcm16 (Kaiser-windowed sinc, beta 8.6)",
        "trim_dbfs": a.trim_dbfs, "margin_ms": a.margin_ms, "clips": len(rows),
        "total_s": round(sum(r["duration_s"] for r in rows), 3)}, indent=1) + "\n")
    print(f"wrote {len(rows)} clips ({sum(r['duration_s'] for r in rows):.1f} s) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
