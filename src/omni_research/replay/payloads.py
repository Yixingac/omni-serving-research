"""Request payloads: workload templates, public/synthetic asset pools, replay surrogates.

Assets are listed in a JSONL manifest; each line is one input file::

    {"asset_id": "q-arith-007", "kind": "audio", "path": "audio/q-arith-007.wav",
     "duration_s": 3.42, "source": "synthetic:tts:<model>@<rev>",
     "transcript": "What is 17 plus 25?", "answer": "42"}

Paths are relative to the manifest. Only public or synthetic files belong in a pool, and every
asset says where it came from in ``source``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backends import ModelRequest

# A small fixed vocabulary for synthetic filler text (original word list, no external corpus).
_WORDS = (
    "system latency request stage model audio video speech token stream buffer queue batch "
    "kernel memory thread process signal sample frame window encoder decoder thinker talker "
    "vocoder schedule replica device budget transfer channel packet vector matrix tensor cache "
    "policy session agent tool result answer question summary report measure observe compare "
    "describe explain review detail simple careful brief clear accurate useful public synthetic"
).split()


def synthetic_text(n_tokens: int, rng: random.Random) -> str:
    """Filler text of roughly ``n_tokens`` tokens (about 0.75 words per token for English)."""
    n_words = max(1, int(round(n_tokens * 0.75)))
    words = [rng.choice(_WORDS) for _ in range(n_words)]
    sentences = []
    for i in range(0, len(words), 12):
        chunk = words[i : i + 12]
        sentences.append(" ".join(chunk).capitalize() + ".")
    return " ".join(sentences)


@dataclass
class Asset:
    asset_id: str
    kind: str
    path: str
    duration_s: float | None = None
    source: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)


class AssetPool:
    def __init__(self, manifest: str | Path | None) -> None:
        self.assets: list[Asset] = []
        self.root: Path | None = None
        if manifest:
            manifest = Path(manifest)
            self.root = manifest.parent
            for line in manifest.read_text().splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                path = d.pop("path")
                full = str((self.root / path).resolve()) if not path.startswith(("/", "http://", "https://")) else path
                self.assets.append(
                    Asset(
                        asset_id=d.pop("asset_id"),
                        kind=d.pop("kind"),
                        path=full,
                        duration_s=d.pop("duration_s", None),
                        source=d.pop("source", "unknown"),
                        meta=d,
                    )
                )

    def of_kind(self, kind: str, tag: str | None = None) -> list[Asset]:
        out = [a for a in self.assets if a.kind == kind]
        if tag:
            out = [a for a in out if tag in a.meta.get("tags", [])]
        return out

    def pick(self, kind: str, rng: random.Random, target_s: float | None = None, tag: str | None = None) -> Asset:
        cands = self.of_kind(kind, tag)
        if not cands:
            raise LookupError(f"asset pool has no {kind!r} assets" + (f" tagged {tag!r}" if tag else ""))
        if target_s is None:
            return rng.choice(cands)
        best = min(abs((a.duration_s or 0.0) - target_s) for a in cands)
        closest = [a for a in cands if abs((a.duration_s or 0.0) - target_s) <= best + 1e-9]
        return rng.choice(closest)


@dataclass
class Workload:
    """A request template (one of W1..W6), loaded from configs/workloads/*.json."""

    workload_id: str
    input_modalities: list[str]
    output_modalities: list[str]
    max_tokens: int | None = 128
    stream: bool = True
    system_prompt: str | None = None
    prompt: dict[str, Any] = field(default_factory=lambda: {"kind": "synthetic_text", "tokens": 128})
    audio: dict[str, Any] | None = None  # {"tag": ..., "target_s": ...}
    video: dict[str, Any] | None = None
    image: dict[str, Any] | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    temperature: float | None = 0.0
    timeout_s: float = 300.0
    description: str = ""

    @classmethod
    def from_file(cls, path: str | Path) -> "Workload":
        d = json.loads(Path(path).read_text())
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in names})


def _text_from_prompt_spec(spec: dict[str, Any], rng: random.Random, asset: Asset | None) -> tuple[str, dict[str, Any]]:
    kind = spec.get("kind", "synthetic_text")
    if kind == "fixed":
        return spec["text"], {"prompt_kind": "fixed"}
    if kind == "choice":
        i = rng.randrange(len(spec["texts"]))
        return spec["texts"][i], {"prompt_kind": "choice", "prompt_index": i}
    if kind == "asset_question":
        # Text that goes with a media asset (e.g. "How many circles appear?" for a video).
        q = (asset.meta.get("question") if asset else None) or spec.get("fallback", "Describe the input.")
        return q, {"prompt_kind": "asset_question"}
    if kind == "synthetic_text":
        n = int(spec.get("tokens", 128))
        prefix = spec.get("prefix", "Summarize the following notes in two sentences: ")
        return prefix + synthetic_text(n, rng), {"prompt_kind": "synthetic_text", "prompt_tokens_target": n}
    raise ValueError(f"unknown prompt kind {kind!r}")


def build_request(
    wl: Workload,
    rng: random.Random,
    request_id: str,
    pool: AssetPool | None = None,
    metadata: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> ModelRequest:
    """Instantiate one request of workload ``wl``; payload details go into ``metadata.payload``."""
    payload: dict[str, Any] = {"workload_id": wl.workload_id}
    audios = videos = images = None
    media_asset = None
    if "audio" in wl.input_modalities:
        spec = wl.audio or {}
        a = pool.pick("audio", rng, spec.get("target_s"), spec.get("tag")) if pool else None
        if a is None:
            raise LookupError("audio workload needs an asset pool")
        audios = [a.path]
        media_asset = a
        payload.update(audio_asset=a.asset_id, audio_s=a.duration_s, audio_source=a.source)
    if "video" in wl.input_modalities:
        spec = wl.video or {}
        v = pool.pick("video", rng, spec.get("target_s"), spec.get("tag")) if pool else None
        if v is None:
            raise LookupError("video workload needs an asset pool")
        videos = [v.path]
        media_asset = v
        payload.update(video_asset=v.asset_id, video_s=v.duration_s, video_source=v.source)
    if "image" in wl.input_modalities:
        spec = wl.image or {}
        im = pool.pick("image", rng, None, spec.get("tag")) if pool else None
        if im is None:
            raise LookupError("image workload needs an asset pool")
        images = [im.path]
        media_asset = im
        payload.update(image_asset=im.asset_id, image_source=im.source)
    text, pmeta = _text_from_prompt_spec(wl.prompt, rng, media_asset)
    payload.update(pmeta, prompt_chars=len(text))
    if media_asset is not None and "answer" in media_asset.meta:
        payload["expected_answer"] = media_asset.meta["answer"]
    messages: list[dict[str, Any]] = []
    if wl.system_prompt:
        messages.append({"role": "system", "content": wl.system_prompt})
    messages.extend(history or [])
    messages.append({"role": "user", "content": text})
    md = dict(metadata or {})
    md["payload"] = payload
    md["workload_id"] = wl.workload_id
    return ModelRequest(
        request_id=request_id,
        messages=messages,
        input_modalities=list(wl.input_modalities),
        output_modalities=list(wl.output_modalities),
        max_tokens=wl.max_tokens,
        audios=audios,
        videos=videos,
        images=images,
        stream=wl.stream,
        temperature=wl.temperature,
        extra_body=dict(wl.extra_body),
        timeout_s=wl.timeout_s,
        metadata=md,
    )


def surrogate_request(
    attrs: dict[str, Any],
    rng: random.Random,
    request_id: str,
    pool: AssetPool | None,
    metadata: dict[str, Any] | None = None,
    default_max_tokens: int = 128,
) -> ModelRequest:
    """Rebuild a request for replay from a trace span's size attributes (content is not stored).

    Records the difference between the original and the surrogate (``metadata.payload``) so an
    analysis can tell a faithful replay from an approximate one.
    """
    in_mod = list(attrs.get("input_modalities", ["text"]))
    out_mod = list(attrs.get("output_modalities", ["text"]))
    payload: dict[str, Any] = {"kind": "surrogate"}
    audios = videos = images = None
    if "audio" in in_mod:
        target = attrs.get("audio_input_s")
        a = pool.pick("audio", rng, target) if pool else None
        if a is None:
            raise LookupError("replaying an audio request needs an audio asset pool")
        audios = [a.path]
        payload.update(audio_asset=a.asset_id, audio_s=a.duration_s, target_audio_s=target)
    if "video" in in_mod:
        target = attrs.get("video_input_s")
        v = pool.pick("video", rng, target) if pool else None
        if v is None:
            raise LookupError("replaying a video request needs a video asset pool")
        videos = [v.path]
        payload.update(video_asset=v.asset_id, video_s=v.duration_s, target_video_s=target)
    if "image" in in_mod:
        im = pool.pick("image", rng) if pool else None
        if im is None:
            raise LookupError("replaying an image request needs an image asset pool")
        images = [im.path]
        payload.update(image_asset=im.asset_id)
    target_chars = int(attrs.get("input_text_chars") or 400)
    text = synthetic_text(max(1, target_chars // 5), rng)[:target_chars]
    payload.update(target_text_chars=target_chars, text_chars=len(text))
    md = dict(metadata or {})
    md["payload"] = payload
    max_tokens = attrs.get("max_tokens") or attrs.get("output_text_tokens") or default_max_tokens
    return ModelRequest(
        request_id=request_id,
        messages=[{"role": "user", "content": text}],
        input_modalities=in_mod,
        output_modalities=out_mod,
        max_tokens=int(max_tokens),
        audios=audios,
        videos=videos,
        images=images,
        metadata=md,
        # Without the recorded text the model's output length cannot be reproduced exactly, so
        # ignore_eos keeps generation at the recorded length when the server supports it.
        extra_body={"stage_params": {"thinker": {"ignore_eos": True}}} if attrs.get("force_length") else {},
    )
