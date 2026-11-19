"""Objective audio-quality measures used by the C1/C2 gates (numpy only)."""

import math

import numpy as np
import pytest

from omni_research.analysis import audio_quality as aq

SR = aq.SAMPLE_RATE


def _tone(freqs, seconds=1.0, seed=0):
    """Harmonics on top of broadband noise, so every mel band carries energy (as speech does)."""
    t = np.arange(int(SR * seconds)) / SR
    rng = np.random.default_rng(seed)
    x = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f in freqs)
    return 0.1 * x + 0.02 * rng.standard_normal(t.size)


def _lowpassed(x, taps):
    return np.convolve(x, np.ones(taps) / taps, mode="same")


def test_waveform_errors_identical_scaled_and_mismatched():
    y = _tone([220, 660])
    same = aq.waveform_errors(y.copy(), y)
    assert same["rel_rms"] == 0 and same["bit_identical"] and same["max_abs"] == 0 and same["snr_db"] == math.inf
    scaled = aq.waveform_errors(1.001 * y, y)
    assert scaled["rel_rms"] == pytest.approx(1e-3, rel=1e-6) and not scaled["bit_identical"]
    assert scaled["energy_rel_diff"] == pytest.approx(1.001 ** 2 - 1, rel=1e-6)
    with pytest.raises(ValueError):
        aq.waveform_errors(y[:-1], y)


def test_mcd_is_zero_for_identical_and_gain_only_changes():
    y = _tone([200, 500, 1200])
    assert aq.mcd_db(y, y) == 0.0
    # a gain change moves only c0 (the energy), which the definition excludes
    assert aq.mcd_db(2.0 * y, y) < 1e-6


def test_mcd_grows_with_distortion_and_separates_different_sounds():
    y = _tone([200, 500, 1200])
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(y.size)
    small = aq.mcd_db(y + 1e-3 * noise, y)
    large = aq.mcd_db(y + 3e-2 * noise, y)
    assert 0 < small < large
    other = aq.mcd_db(_lowpassed(_tone([300], seed=2), 9), y)  # a darker, different sound
    assert other > large


def test_mel_filterbank_shape_and_partition():
    fb = aq.mel_filterbank()
    assert fb.shape == (aq.MCD_MELS, aq.MCD_N_FFT // 2 + 1)
    assert (fb >= 0).all() and (fb.max(axis=1) > 0.5).all()  # every band has a peak


def test_text_normalization_wer_and_exact_matches():
    assert aq.normalize_text("How many steel-bolts? It’s 81.") == "how many steel bolts it's 81"
    assert aq.word_errors("how many steel bolts", "How many steel bolts?") == (0, 4)
    assert aq.word_errors("how many bolts", "how many steel bolts") == (1, 4)  # one deletion
    out = aq.asr_agreement(["a b c", "x y"], ["a b c", "x z"])
    assert out["n"] == 2 and out["exact_matches"] == 1 and out["wer"] == pytest.approx(1 / 5)
    with pytest.raises(ValueError):
        aq.asr_agreement(["a"], [])


def test_float32_ulp_distance():
    a = np.array([1.0, -2.0, 0.0], dtype=np.float32)
    b = np.nextafter(a, np.float32(np.inf)).astype(np.float32)
    assert aq.float32_ulp_distance(a, a) == 0
    assert aq.float32_ulp_distance(a, b) == 1
    assert aq.float32_ulp_distance(np.array([-0.0], np.float32), np.array([0.0], np.float32)) == 0
    assert aq.float32_ulp_distance(np.array([1.0], np.float32), np.array([1.0 + 4 * 2 ** -23], np.float32)) == 4
