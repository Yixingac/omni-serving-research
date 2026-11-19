"""Objective audio-quality measures for vocoder/flow changes (numpy only; registered for C1/C2).

Every comparison is between two time-aligned waveforms of the same request (same codec tokens, same length), so no
alignment (DTW) is needed. Definitions (fixed before any C1/C2 run, research/next_steps/presentation_validation/
c1_c2_design.md):

- rel_rms = ||x - y||_2 / ||y||_2, max_abs = max |x - y|, energy_rel_diff = |sum x^2 - sum y^2| / sum y^2;
- MCD (dB): 24 kHz, 25 ms Hann window (600 samples), 5 ms hop (120 samples), n_fft 1024, 80 HTK-mel bands 0-12 kHz on
  the power spectrum, cepstrum = orthonormal DCT-II of the natural-log mel *amplitude* (0.5 ln(power + 1e-10)),
  coefficients 1..24 (c0, the energy, excluded); per frame (10 / ln 10) * sqrt(2 * sum_d (c_d - c'_d)^2), averaged
  over frames whose reference energy is within 60 dB of the reference's loudest frame. This is an MFCC-style MCD; its
  absolute values are not comparable with SPTK mel-cepstrum MCDs in the literature, only arms against each other;
- ASR agreement: text lower-cased, apostrophes kept, every other non-alphanumeric character a space, whitespace
  collapsed; WER = word-level Levenshtein distance / reference words; exact match = normalized strings equal.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

import numpy as np

SAMPLE_RATE = 24000
MCD_N_FFT, MCD_WIN, MCD_HOP, MCD_MELS, MCD_COEFFS = 1024, 600, 120, 80, 24
MCD_FLOOR_DB = 60.0
MCD_CONST = 10.0 / math.log(10.0) * math.sqrt(2.0)


def _pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"waveforms differ in length: {x.shape[0]} vs {y.shape[0]}")
    return x, y


def waveform_errors(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """x: candidate, y: reference."""
    x, y = _pair(x, y)
    d = x - y
    ref = float(np.sqrt(np.sum(y * y)))
    err = float(np.sqrt(np.sum(d * d)))
    ey = float(np.sum(y * y))
    return {"rel_rms": err / ref if ref > 0 else (0.0 if err == 0 else math.inf),
            "max_abs": float(np.max(np.abs(d))) if d.size else 0.0,
            "energy_rel_diff": abs(float(np.sum(x * x)) - ey) / ey if ey > 0 else 0.0,
            "snr_db": 20 * math.log10(ref / err) if err > 0 and ref > 0 else math.inf,
            "bit_identical": bool(np.array_equal(x, y))}


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr: int = SAMPLE_RATE, n_fft: int = MCD_N_FFT, n_mels: int = MCD_MELS,
                   fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    """Triangular HTK-mel filters, shape (n_mels, n_fft // 2 + 1)."""
    fmax = sr / 2 if fmax is None else fmax
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = _mel_to_hz(np.linspace(_hz_to_mel(np.array(fmin)), _hz_to_mel(np.array(fmax)), n_mels + 2))
    fb = np.zeros((n_mels, freqs.size))
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (freqs - lo) / (c - lo) if c > lo else np.zeros_like(freqs)
        down = (hi - freqs) / (hi - c) if hi > c else np.zeros_like(freqs)
        fb[i] = np.clip(np.minimum(up, down), 0, None)
    return fb


def _dct_ortho(x: np.ndarray, n: int) -> np.ndarray:
    """Orthonormal DCT-II along the last axis, first n coefficients."""
    m = x.shape[-1]
    k = np.arange(n)[:, None]
    basis = np.cos(np.pi / m * (np.arange(m)[None, :] + 0.5) * k)
    scale = np.full(n, math.sqrt(2.0 / m))
    scale[0] = math.sqrt(1.0 / m)
    return (x @ basis.T) * scale


def mel_cepstrum(x: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, np.ndarray]:
    """(cepstra (frames, 25) including c0, frame energy in dB)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < MCD_WIN:
        x = np.pad(x, (0, MCD_WIN - x.size))
    n_frames = 1 + (x.size - MCD_WIN) // MCD_HOP
    idx = np.arange(MCD_WIN)[None, :] + MCD_HOP * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hanning(MCD_WIN + 2)[1:-1]
    spec = np.abs(np.fft.rfft(frames, n=MCD_N_FFT, axis=1)) ** 2
    mel = spec @ mel_filterbank(sr).T
    cep = _dct_ortho(0.5 * np.log(mel + 1e-10), MCD_COEFFS + 1)
    energy_db = 10 * np.log10(np.sum(spec, axis=1) + 1e-20)
    return cep, energy_db


def mcd_db(x: np.ndarray, y: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Mean mel-cepstral distortion of candidate x against reference y (definition in the module docstring)."""
    x, y = _pair(x, y)
    cx, _ = mel_cepstrum(x, sr)
    cy, ey = mel_cepstrum(y, sr)
    keep = ey >= ey.max() - MCD_FLOOR_DB
    diff = cx[keep, 1:] - cy[keep, 1:]
    return float(np.mean(MCD_CONST * np.sqrt(np.sum(diff * diff, axis=1)))) if keep.any() else 0.0


def normalize_text(s: str) -> str:
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9']+", " ", s)
    return " ".join(w.strip("'") for w in s.split() if w.strip("'"))


def word_errors(hyp: str, ref: str) -> tuple[int, int]:
    """(edit distance in words, reference word count) after normalization."""
    h, r = normalize_text(hyp).split(), normalize_text(ref).split()
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, start=1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[-1], len(r)


def asr_agreement(hyps: Sequence[str], refs: Sequence[str]) -> dict[str, float | int]:
    """Pooled WER and the number of exact (normalized) matches of hypotheses against references."""
    if len(hyps) != len(refs):
        raise ValueError("hypotheses and references differ in count")
    errs = words = exact = 0
    for h, r in zip(hyps, refs):
        e, n = word_errors(h, r)
        errs, words = errs + e, words + n
        exact += normalize_text(h) == normalize_text(r)
    return {"n": len(refs), "wer": errs / words if words else 0.0, "exact_matches": exact,
            "exact_share": exact / len(refs) if refs else 0.0}


def float32_ulp_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Largest distance in units in the last place between two float32 arrays (0 = bit-identical)."""
    a = np.ascontiguousarray(a, dtype=np.float32).view(np.int32).astype(np.int64)
    b = np.ascontiguousarray(b, dtype=np.float32).view(np.int32).astype(np.int64)
    if a.shape != b.shape:
        raise ValueError("arrays differ in shape")
    # map the sign-magnitude float order onto a monotone integer line
    a = np.where(a < 0, -(a & 0x7FFFFFFF), a)
    b = np.where(b < 0, -(b & 0x7FFFFFFF), b)
    return int(np.max(np.abs(a - b))) if a.size else 0


def mean(values: Iterable[float]) -> float | None:
    v = [x for x in values if x is not None and not math.isnan(x)]
    return sum(v) / len(v) if v else None
