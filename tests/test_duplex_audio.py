"""Duplex input audio helpers: resampling, trimming and WAV round trips (numpy only, no GPU)."""

import numpy as np

from omni_research.duplex.audio import frame_dbfs, read_wav, resample_pcm16, rms_dbfs, trim_silence, wav_bytes


def _tone(hz, seconds, rate, amp=8000.0):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.int16)


def test_resample_24k_to_16k_keeps_length_level_and_frequency():
    x = _tone(440.0, 2.0, 24000)
    y = resample_pcm16(x, 24000, 16000)
    assert len(y) == 32000 and y.dtype == np.int16
    core = y[1600:-1600].astype(np.float64)  # away from the filter's edges
    assert abs(np.sqrt((core ** 2).mean()) - 8000 / np.sqrt(2)) < 0.01 * 8000
    spec = np.abs(np.fft.rfft(core * np.hanning(len(core))))
    peak_hz = np.argmax(spec) * 16000 / len(core)
    assert abs(peak_hz - 440.0) < 2.0


def test_resample_removes_content_above_the_new_nyquist_and_is_deterministic():
    x = _tone(10000.0, 1.0, 24000)  # above 8 kHz: must not alias into the 16 kHz output
    y = resample_pcm16(x, 24000, 16000)
    assert np.sqrt((y[800:-800].astype(np.float64) ** 2).mean()) < 0.01 * 8000
    assert np.array_equal(resample_pcm16(x, 24000, 16000), y)
    assert np.array_equal(resample_pcm16(y, 16000, 16000), y)


def test_trim_keeps_speech_and_a_margin():
    rate = 16000
    x = np.concatenate([np.zeros(rate, np.int16), _tone(300.0, 1.5, rate), np.zeros(2 * rate, np.int16)])
    y, start, end = trim_silence(x, rate, margin_ms=100.0)
    assert start == rate - 1600 and abs(end - (rate + 24000 + 1600)) <= 160
    assert len(y) == end - start
    silent = np.zeros(rate, np.int16)
    assert trim_silence(silent, rate)[1:] == (0, rate)


def test_wav_round_trip_and_levels():
    x = _tone(200.0, 0.5, 24000)
    y, rate = read_wav(wav_bytes(x, 24000))
    assert rate == 24000 and np.array_equal(x, y)
    assert rms_dbfs(b"") == -240.0 and rms_dbfs(np.zeros(100, np.int16).tobytes()) < -200
    assert -15.5 < rms_dbfs(x.tobytes()) < -15.0  # 20 log10(8000 / sqrt(2) / 32768) = -15.26 dBFS
    assert len(frame_dbfs(x, 24000)) == 50
