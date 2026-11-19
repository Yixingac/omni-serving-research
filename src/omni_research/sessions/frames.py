"""Reproducible synthetic camera input: frames, short clips and per-session unique media (stdlib only).

A camera sees N coloured circles moving over a gradient background, 640x360 by default. Every frame also
carries a strip of black/white blocks that encode (session, frame index), so no two frames of a run are
byte-identical: the server's media cache keys (a hash of the decoded pixels) and its prefix cache can
then never hit on a repeated frame. Frames are PNG (zlib, deterministic); clips are H.264 MP4 made by
piping raw frames into ffmpeg, like the counting videos of replay.assets.
"""

from __future__ import annotations

import base64
import math
import random
import struct
import subprocess
import zlib
from dataclasses import dataclass, field

COLORS = [(230, 57, 70), (29, 53, 87), (69, 123, 157), (42, 157, 143), (233, 196, 106), (244, 162, 97)]
CODE_BITS = 32  # 16 bits session, 16 bits frame index
CODE_ROWS = 8


@dataclass
class Camera:
    """One session's camera: the scene and its motion are fixed by the seed."""

    seed: int
    session: int
    width: int = 640
    height: int = 360
    capture_fps: float = 8.0
    circles: list[dict] = field(default_factory=list)
    n_circles: int = 0

    def __post_init__(self) -> None:
        rng = random.Random(self.seed * 100_003 + self.session)
        self.n_circles = rng.choice([1, 2, 3, 4, 5])
        lo, hi = min(self.width, self.height) // 14, min(self.width, self.height) // 9
        self.circles = []
        for i in range(self.n_circles):
            r = rng.randint(lo, hi)
            self.circles.append({"x0": rng.uniform(r, self.width - r), "y0": rng.uniform(CODE_ROWS + r, self.height - r),
                                 "r": r, "vx": rng.uniform(-90, 90), "vy": rng.uniform(-90, 90),
                                 "c": COLORS[(i + self.session) % len(COLORS)]})
        self.bg_top = (235 - (self.session * 7) % 40, 240, 230 + (self.session * 3) % 20)

    def positions(self, t_s: float) -> list[tuple[int, int, int, tuple[int, int, int]]]:
        """Circle centres at time t (bouncing off the walls), as (x, y, r, colour)."""
        out = []
        for c in self.circles:
            x = _bounce(c["x0"] + c["vx"] * t_s, c["r"], self.width - c["r"])
            y = _bounce(c["y0"] + c["vy"] * t_s, CODE_ROWS + c["r"], self.height - c["r"])
            out.append((int(x), int(y), c["r"], c["c"]))
        return out

    def background(self) -> bytes:
        if not hasattr(self, "_background"):
            w, h = self.width, self.height
            top, bottom = self.bg_top, (200, 205, 215)
            rows = []
            for y in range(h):
                a = y / max(1, h - 1)
                rows.append(bytes(int(top[k] + (bottom[k] - top[k]) * a) for k in range(3)) * w)
            self._background = b"".join(rows)
        return self._background

    def rgb(self, frame_index: int) -> bytes:
        """Raw RGB24 bytes of frame ``frame_index`` (captured at frame_index / capture_fps seconds)."""
        w, h = self.width, self.height
        frame = bytearray(self.background())
        for cx, cy, r, colour in self.positions(frame_index / self.capture_fps):
            col = bytes(colour)
            for y in range(max(CODE_ROWS, cy - r), min(h, cy + r + 1)):
                half = int(math.sqrt(max(0, r * r - (y - cy) ** 2)))
                x0, x1 = max(0, cx - half), min(w, cx + half + 1)
                if x1 > x0:
                    start = (y * w + x0) * 3
                    frame[start:start + (x1 - x0) * 3] = col * (x1 - x0)
        code = ((self.session & 0xFFFF) << 16) | (frame_index & 0xFFFF)
        block = w // CODE_BITS
        for bit in range(CODE_BITS):
            col = b"\x00\x00\x00" if (code >> (CODE_BITS - 1 - bit)) & 1 else b"\xff\xff\xff"
            for y in range(CODE_ROWS):
                start = (y * w + bit * block) * 3
                frame[start:start + block * 3] = col * block
        return bytes(frame)

    def png(self, frame_index: int) -> bytes:
        return encode_png(self.rgb(frame_index), self.width, self.height)


def content_seed(seed: int, content_key: str) -> int:
    """Seed of a phase's media and sessions: the experiment seed mixed with a key (by default the phase name), so
    two phases of one run never send the same frame or session (no cache hits across load levels), while the same
    phase in another block or another arm sends exactly the same content (paired comparisons)."""
    return (seed * 1_000_003 + zlib.crc32(content_key.encode())) & 0x7FFFFFFF


def _bounce(v: float, lo: float, hi: float) -> float:
    span = hi - lo
    if span <= 0:
        return lo
    m = (v - lo) % (2 * span)
    return lo + (m if m <= span else 2 * span - m)


def encode_png(rgb: bytes, width: int, height: int, level: int = 6) -> bytes:
    """Minimal RGB PNG (filter 0 on every row); deterministic for the same bytes."""
    stride = width * 3
    raw = b"".join(b"\x00" + rgb[y * stride:(y + 1) * stride] for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, level)) + chunk(b"IEND", b"")


def decode_png_size(data: bytes) -> tuple[int, int]:
    """(width, height) from a PNG header, for checks."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    return struct.unpack(">II", data[16:24])


def data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def encode_clip(camera: Camera, first_frame: int, n_frames: int, ffmpeg: str,
                frames: dict[int, bytes] | None = None) -> bytes:
    """H.264 MP4 of frames [first_frame, first_frame + n_frames) at the camera's capture rate. ``frames`` is an
    optional cache of already rendered RGB frames (consecutive clips of a sliding window share most frames)."""
    rgb = []
    for k in range(first_frame, first_frame + n_frames):
        if frames is not None and k in frames:
            rgb.append(frames[k])
            continue
        data = camera.rgb(k)
        if frames is not None:
            frames[k] = data
        rgb.append(data)
    proc = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{camera.width}x{camera.height}",
         "-r", f"{camera.capture_fps:g}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
         "-movflags", "frag_keyframe+empty_moov", "-f", "mp4", "pipe:1"],
        input=b"".join(rgb), capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-400:]!r}")
    return proc.stdout


def wav_with_padding(wav: bytes, pad_samples: int) -> bytes:
    """A PCM WAV with ``pad_samples`` frames of silence appended (header sizes fixed up).

    Appending a few milliseconds of silence leaves the spoken content unchanged but changes the decoded
    waveform, so a clip reused by several sessions or turns gets its own media cache key."""
    if wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos, fmt, data_at = 12, None, None
    while pos + 8 <= len(wav):
        cid, size = wav[pos:pos + 4], struct.unpack("<I", wav[pos + 4:pos + 8])[0]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", wav[pos + 8:pos + 24])
        elif cid == b"data":
            data_at = pos
            break
        pos += 8 + size + (size & 1)
    if fmt is None or data_at is None:
        raise ValueError("WAV without fmt or data chunk")
    channels, bits = fmt[1], fmt[5]
    old_size = min(struct.unpack("<I", wav[data_at + 4:data_at + 8])[0], len(wav) - data_at - 8)
    body = wav[data_at + 8:data_at + 8 + old_size] + b"\x00" * (pad_samples * channels * (bits // 8))
    head = bytearray(wav[:data_at])
    head[4:8] = struct.pack("<I", len(head) - 8 + 8 + len(body))
    return bytes(head) + b"data" + struct.pack("<I", len(body)) + body
