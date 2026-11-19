"""The input one simulated user streams to /v1/realtime: a fixed, seeded timeline of spoken questions.

A session's input is one continuous 16 kHz PCM16 signal of exactly ``duration_s`` seconds:

    [lead-in silence] [question 1] [answer window] [question 2] [answer window] ... [silence to the end]

The full-duplex model listens to every second of it (the server cuts it into 1 s units) and decides per
unit whether to keep listening or to speak, so the client keeps sending silence while the model answers.
The timeline does not react to the server: the same seed gives the same bytes, so arms and session counts
receive identical input. A barge-in turn (every ``barge_in_every``-th turn, 0 = none) places a second
question ``barge_in_offset_s`` after the first one ends, i.e. while the model is probably answering;
whether it was a real interruption (assistant audio arriving at that moment) is decided afterwards from
the trace, never assumed.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .audio import read_wav

SAMPLE_RATE = 16000
PACKET_MS = 80  # the upstream benchmark client's packet (benchmarks/duplex/client.py); 1 s unit = 12.5 packets
PACKET_SAMPLES = SAMPLE_RATE * PACKET_MS // 1000


@dataclass(frozen=True)
class Clip:
    asset_id: str
    samples: np.ndarray = field(repr=False)
    transcript: str = ""

    @property
    def duration_s(self) -> float:
        return len(self.samples) / SAMPLE_RATE


@dataclass
class Turn:
    index: int
    kind: str  # "question" or "barge_in"
    asset_id: str
    transcript: str
    start_sample: int
    end_sample: int  # exclusive
    interrupts_turn: int | None = None

    @property
    def last_packet(self) -> int:
        """Sequence number of the packet that carries the utterance's last sample."""
        return (self.end_sample - 1) // PACKET_SAMPLES

    @property
    def first_packet(self) -> int:
        return self.start_sample // PACKET_SAMPLES


@dataclass
class SessionScript:
    session_index: int
    seed: int
    duration_s: float
    pcm: np.ndarray = field(repr=False)
    turns: list[Turn]
    params: dict[str, Any]

    def packets(self) -> list[tuple[int, float, bytes]]:
        """(seq, t_start_ms, PCM16 bytes) in 80 ms packets; the last one may be shorter."""
        raw = np.asarray(self.pcm, dtype="<i2").tobytes()
        step = PACKET_SAMPLES * 2
        return [(i, i * PACKET_MS * 1.0, raw[off:off + step]) for i, off in enumerate(range(0, len(raw), step))]

    def sha256(self) -> str:
        return hashlib.sha256(np.asarray(self.pcm, dtype="<i2").tobytes()).hexdigest()

    def to_json(self) -> dict[str, Any]:
        return {"session_index": self.session_index, "seed": self.seed, "duration_s": self.duration_s,
                "sample_rate": SAMPLE_RATE, "packet_ms": PACKET_MS, "pcm_sha256": self.sha256(),
                "params": self.params, "turns": [asdict(t) | {"last_packet": t.last_packet, "first_packet": t.first_packet}
                                                 for t in self.turns]}


def load_clips(pool_dir: str | Path, tag: str = "agent_question") -> list[Clip]:
    """16 kHz clips of an asset pool written by scripts/assets/make_duplex_speech.py, sorted by asset id."""
    pool = Path(pool_dir)
    clips = []
    for line in (pool / "manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("kind") != "audio" or tag not in (row.get("tags") or []):
            continue
        samples, rate = read_wav(pool / row["path"])
        if rate != SAMPLE_RATE:
            raise ValueError(f"{row['asset_id']}: {rate} Hz, the duplex pool must be {SAMPLE_RATE} Hz")
        clips.append(Clip(row["asset_id"], samples, row.get("transcript", "")))
    if not clips:
        raise LookupError(f"no {tag!r} audio in {pool}")
    return sorted(clips, key=lambda c: c.asset_id)


def build_script(clips: list[Clip], *, duration_s: float, seed: int, session_index: int = 0,
                 lead_in_s: float = 2.0, answer_window_s: float = 12.0, barge_in_every: int = 0,
                 barge_in_offset_s: float = 3.0) -> SessionScript:
    """Seeded order of the clips (reshuffled each time they are used up); turns are appended while a turn
    and its answer window still fit, and the rest of the session is silence."""
    if not clips:
        raise ValueError("no clips")
    total = round(duration_s * SAMPLE_RATE)
    pcm = np.zeros(total, dtype=np.int16)
    rng = random.Random(seed * 1000 + session_index)
    bag: list[Clip] = []

    def draw() -> Clip:
        if not bag:
            bag.extend(clips)
            rng.shuffle(bag)
        return bag.pop()

    turns: list[Turn] = []
    cursor = round(lead_in_s * SAMPLE_RATE)
    window = round(answer_window_s * SAMPLE_RATE)
    offset = round(barge_in_offset_s * SAMPLE_RATE)
    while True:
        q = draw()
        k = len([t for t in turns if t.kind == "question"])
        barge = barge_in_every > 0 and k % barge_in_every == barge_in_every - 1
        need = len(q.samples) + window
        b = None
        if barge:
            b = draw()
            need = len(q.samples) + offset + len(b.samples) + window
        if cursor + need > total:
            break
        pcm[cursor:cursor + len(q.samples)] = q.samples
        turns.append(Turn(len(turns), "question", q.asset_id, q.transcript, cursor, cursor + len(q.samples)))
        cursor += len(q.samples)
        if b is not None:
            cursor += offset
            pcm[cursor:cursor + len(b.samples)] = b.samples
            turns.append(Turn(len(turns), "barge_in", b.asset_id, b.transcript, cursor, cursor + len(b.samples),
                              interrupts_turn=turns[-1].index))
            cursor += len(b.samples)
        cursor += window
    params = {"lead_in_s": lead_in_s, "answer_window_s": answer_window_s, "barge_in_every": barge_in_every,
              "barge_in_offset_s": barge_in_offset_s, "clips": len(clips)}
    return SessionScript(session_index, seed, duration_s, pcm, turns, params)
