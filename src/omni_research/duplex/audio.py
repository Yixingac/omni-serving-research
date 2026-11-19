"""PCM16 audio helpers: WAV I/O, a deterministic polyphase resampler and end trimming (numpy only).

The duplex endpoint takes 16 kHz mono PCM16 (``input_audio_format`` of /v1/realtime/capabilities) and
returns 24 kHz mono PCM16. The spoken questions of the synthetic-v1 pool are 24 kHz PCM16 WAVs, so they
are resampled by 2/3 here. The filter is a Kaiser-windowed sinc computed in float64, so the same input
always gives the same bytes on every machine (no scipy/librosa version dependence).
"""

from __future__ import annotations

import io
import math
import wave
from pathlib import Path

import numpy as np

PCM16_MAX = 32767


def read_wav(path_or_bytes: str | Path | bytes) -> tuple[np.ndarray, int]:
    """PCM16 WAV -> (int16 mono samples, sample rate); several channels are averaged."""
    raw = bytes(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else Path(path_or_bytes).read_bytes()
    with wave.open(io.BytesIO(raw), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM, got {8 * w.getsampwidth()}-bit")
        rate, channels = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    if channels > 1:
        data = np.round(data.reshape(-1, channels).astype(np.float64).mean(axis=1)).astype(np.int16)
    return data.astype(np.int16, copy=False), rate


def wav_bytes(samples: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return buf.getvalue()


def write_wav(path: str | Path, samples: np.ndarray, rate: int) -> None:
    Path(path).write_bytes(wav_bytes(samples, rate))


def lowpass_kernel(up: int, down: int, half_taps_per_phase: int = 16, beta: float = 8.6) -> np.ndarray:
    """Windowed-sinc low-pass for the up-sampled rate, cut at the lower of the two Nyquist frequencies."""
    cutoff = 1.0 / max(up, down)  # as a fraction of the up-sampled Nyquist frequency
    half = half_taps_per_phase * max(up, down)
    n = np.arange(-half, half + 1, dtype=np.float64)
    h = np.sinc(cutoff * n) * np.kaiser(2 * half + 1, beta)
    return h * (up / h.sum())  # DC gain = up, so the zero-stuffed signal keeps its level


def resample_poly(x: np.ndarray, up: int, down: int) -> np.ndarray:
    """Resample by up/down (zero-stuff, filter, decimate), output length ceil(len * up / down)."""
    g = math.gcd(up, down)
    up, down = up // g, down // g
    xf = np.asarray(x, dtype=np.float64)
    if up == down == 1:
        return xf.copy()
    h = lowpass_kernel(up, down)
    stuffed = np.zeros(len(xf) * up, dtype=np.float64)
    stuffed[::up] = xf
    y = np.convolve(stuffed, h, mode="full")
    delay = (len(h) - 1) // 2
    n_out = -(-len(xf) * up // down)
    return y[delay:delay + n_out * down:down][:n_out]


def to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x), -PCM16_MAX - 1, PCM16_MAX).astype(np.int16)


def resample_pcm16(samples: np.ndarray, rate_in: int, rate_out: int) -> np.ndarray:
    if rate_in == rate_out:
        return np.asarray(samples, dtype=np.int16).copy()
    return to_int16(resample_poly(samples, rate_out, rate_in))


def frame_dbfs(samples: np.ndarray, rate: int, frame_ms: float = 10.0) -> np.ndarray:
    """RMS level per frame in dBFS (full scale = 32768)."""
    n = max(1, int(rate * frame_ms / 1000))
    x = np.asarray(samples, dtype=np.float64)
    frames = len(x) // n
    if frames == 0:
        return np.zeros(0)
    ms = (x[:frames * n].reshape(frames, n) ** 2).mean(axis=1)
    return 10.0 * np.log10(ms / 32768.0 ** 2 + 1e-12)


def trim_silence(samples: np.ndarray, rate: int, threshold_dbfs: float = -45.0,
                 margin_ms: float = 100.0) -> tuple[np.ndarray, int, int]:
    """Cut leading and trailing frames below the threshold, keeping a margin. Returns (audio, start, end)
    in samples of the input; an all-silent input is returned unchanged."""
    frame = max(1, int(rate * 0.010))
    level = frame_dbfs(samples, rate)
    active = np.flatnonzero(level > threshold_dbfs)
    if active.size == 0:
        return np.asarray(samples, dtype=np.int16).copy(), 0, len(samples)
    margin = int(rate * margin_ms / 1000)
    start = max(0, int(active[0]) * frame - margin)
    end = min(len(samples), (int(active[-1]) + 1) * frame + margin)
    return np.asarray(samples[start:end], dtype=np.int16).copy(), start, end


def pcm16_seconds(n_bytes: int, rate: int) -> float:
    return n_bytes / 2 / rate


def rms_dbfs(pcm: bytes) -> float:
    """Level of a PCM16 byte string; -inf-like (-240) for silence or an empty string."""
    if len(pcm) < 2:
        return -240.0
    x = np.frombuffer(pcm[:len(pcm) // 2 * 2], dtype="<i2").astype(np.float64)
    return float(10.0 * np.log10((x ** 2).mean() / 32768.0 ** 2 + 1e-24))
