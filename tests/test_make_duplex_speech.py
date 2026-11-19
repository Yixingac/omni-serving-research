"""Duplex input pool builder: 24 kHz spoken questions -> deterministic, trimmed 16 kHz clips (no GPU)."""

import importlib.util
import json
from pathlib import Path

import numpy as np

from omni_research.duplex.audio import read_wav, wav_bytes

HERE = Path(__file__).resolve().parents[1] / "scripts/assets"
spec = importlib.util.spec_from_file_location("make_duplex_speech", HERE / "make_duplex_speech.py")
mk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mk)


def _src_pool(root: Path) -> None:
    pool = root / "assets" / "synthetic-v1"
    (pool / "audio").mkdir(parents=True)
    rate = 24000
    t = np.arange(int(1.2 * rate)) / rate
    voice = (6000 * np.sin(2 * np.pi * 300 * t)).astype(np.int16)
    clip = np.concatenate([np.zeros(rate // 2, np.int16), voice, np.zeros(rate, np.int16)])
    rows = []
    for i in range(2):
        (pool / "audio" / f"q-{i}.wav").write_bytes(wav_bytes(clip, rate))
        rows.append({"asset_id": f"q-{i}", "kind": "audio", "path": f"audio/q-{i}.wav", "sample_rate": rate,
                     "tags": ["agent_question", "arith"], "transcript": f"question {i}", "source": "synthetic:tts:test"})
    rows.append({"asset_id": "passage-00", "kind": "audio", "path": "audio/q-0.wav", "tags": ["passage"]})
    rows.append({"asset_id": "v0", "kind": "video", "path": "video/v0.mp4", "tags": ["agent_question"]})
    (pool / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_builds_trimmed_16k_clips_deterministically_and_refuses_to_overwrite(tmp_path):
    _src_pool(tmp_path)
    assert mk.main(["--data-root", str(tmp_path)]) == 0
    out = tmp_path / "assets" / "duplex-v1"
    rows = [json.loads(x) for x in (out / "manifest.jsonl").read_text().splitlines()]
    assert [r["asset_id"] for r in rows] == ["q-0", "q-1"]
    samples, rate = read_wav(out / rows[0]["path"])
    assert rate == 16000 and abs(len(samples) / 16000 - (1.2 + 0.2)) < 0.02  # speech plus 100 ms margins
    assert rows[0]["src_sha256"] and rows[0]["pcm_sha256"] == rows[1]["pcm_sha256"]
    assert json.loads((out / "generation.json").read_text())["clips"] == 2
    first = (out / "manifest.jsonl").read_text()
    assert mk.main(["--data-root", str(tmp_path)]) == 0  # same content: a no-op
    assert (out / "manifest.jsonl").read_text() == first
    assert mk.main(["--data-root", str(tmp_path), "--margin-ms", "50"]) == 2  # different content: refused
    assert mk.main(["--data-root", str(tmp_path), "--margin-ms", "50", "--force"]) == 0
    assert json.loads((out / "manifest.jsonl").read_text().splitlines()[0])["duration_s"] < rows[0]["duration_s"]
