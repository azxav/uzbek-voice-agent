"""Decode uploaded audio and resample to 16 kHz mono float32."""

from __future__ import annotations

import io

import numpy as np

TARGET_SAMPLE_RATE = 16_000


def to_mono_float32(samples: np.ndarray) -> np.ndarray:
    audio = np.asarray(samples)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1:
        raise ValueError(f"expected mono or stereo audio, got shape {audio.shape}")
    return audio.astype(np.float32, copy=False)


def resample_linear(
    samples: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE
) -> np.ndarray:
    audio = to_mono_float32(samples)
    if orig_sr <= 0:
        raise ValueError("sample rate must be positive")
    if orig_sr == target_sr or audio.size == 0:
        return audio
    target_len = max(1, int(round(audio.size * target_sr / orig_sr)))
    source_x = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    target_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(target_x, source_x, audio).astype(np.float32)


def decode_audio_bytes(payload: bytes) -> tuple[np.ndarray, int]:
    """Read wav/flac/ogg bytes with libsndfile. Returns mono float32 and sample rate."""
    if not payload:
        raise ValueError("empty audio payload")
    import soundfile as sf

    try:
        audio, sample_rate = sf.read(io.BytesIO(payload), always_2d=False, dtype="float32")
    except Exception as exc:  # soundfile raises LibsndfileError subclasses
        raise ValueError("could not decode audio; upload wav, flac, or ogg") from exc
    return to_mono_float32(audio), int(sample_rate)


def prepare_for_asr(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    return resample_linear(samples, sample_rate, TARGET_SAMPLE_RATE)
