"""Synthetic camera frames and per-session media for R3/R4 (stdlib only, no GPU, no network)."""

import shutil
import struct
import zlib

import pytest

from omni_research.sessions.frames import Camera, data_url, decode_png_size, encode_clip, encode_png, wav_with_padding


def _png_pixels(png: bytes, width: int, height: int) -> bytes:
    pos, idat = 8, b""
    while pos < len(png):
        size = struct.unpack(">I", png[pos:pos + 4])[0]
        kind = png[pos + 4:pos + 8]
        if kind == b"IDAT":
            idat += png[pos + 8:pos + 8 + size]
        pos += 12 + size
    raw = zlib.decompress(idat)
    stride = width * 3 + 1
    assert all(raw[y * stride] == 0 for y in range(height))  # filter 0 on every row
    return b"".join(raw[y * stride + 1:(y + 1) * stride] for y in range(height))


def test_png_round_trips_the_rgb_bytes_and_is_deterministic():
    cam = Camera(seed=20261119, session=3, width=64, height=36)
    rgb = cam.rgb(5)
    png = encode_png(rgb, 64, 36)
    assert decode_png_size(png) == (64, 36)
    assert _png_pixels(png, 64, 36) == rgb
    assert Camera(seed=20261119, session=3, width=64, height=36).png(5) == png


def test_every_frame_of_every_session_is_unique_and_the_scene_is_seeded():
    cams = [Camera(seed=1, session=s, width=128, height=72) for s in range(4)]
    frames = {c.png(k) for c in cams for k in range(20)}
    assert len(frames) == 80  # the code strip makes even identical scenes byte-different
    assert Camera(seed=1, session=2).n_circles == Camera(seed=1, session=2).n_circles
    assert {Camera(seed=1, session=s).n_circles for s in range(30)} <= {1, 2, 3, 4, 5}


def test_circles_stay_inside_the_frame():
    cam = Camera(seed=7, session=0)
    for k in range(0, 2000, 37):
        for x, y, r, _ in cam.positions(k / cam.capture_fps):
            assert r <= x <= cam.width - r and r <= y <= cam.height - r


def test_data_url_and_wav_padding():
    assert data_url(b"\x00\x01", "image/png") == "data:image/png;base64,AAE="
    pcm = b"\x01\x00" * 160
    fmt = struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
    wav = b"RIFF" + struct.pack("<I", 4 + 8 + 16 + 8 + len(pcm)) + b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt + \
        b"data" + struct.pack("<I", len(pcm)) + pcm
    out = wav_with_padding(wav, 40)
    assert out[:4] == b"RIFF" and struct.unpack("<I", out[4:8])[0] == len(out) - 8
    data_size = struct.unpack("<I", out[40:44])[0]
    assert data_size == len(pcm) + 80 and out[44:44 + len(pcm)] == pcm and out[-80:] == b"\x00" * 80
    assert wav_with_padding(wav, 41) != out
    with pytest.raises(ValueError):
        wav_with_padding(b"not a wav", 1)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_clip_is_an_mp4():
    clip = encode_clip(Camera(seed=1, session=0, width=64, height=36), 0, 4, shutil.which("ffmpeg"))
    assert clip[4:8] == b"ftyp"
