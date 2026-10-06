import math
import wave
from pathlib import Path

import numpy as np

from uzbek_voice_agent.audio import decode_audio_bytes, prepare_for_asr, resample_linear

FIXTURE = Path(__file__).parent / "fixtures" / "tone.wav"


def test_fixture_wav_decodes() -> None:
    payload = FIXTURE.read_bytes()
    samples, sample_rate = decode_audio_bytes(payload)
    assert sample_rate == 16_000
    assert samples.dtype == np.float32
    assert samples.size == 4_800
    assert float(np.max(np.abs(samples))) > 0.1


def test_resample_changes_length() -> None:
    source = np.ones(8_000, dtype=np.float32)
    resampled = resample_linear(source, 8_000, 16_000)
    assert resampled.size == 16_000


def test_prepare_for_asr_is_16k() -> None:
    samples = np.sin(np.linspace(0, math.pi, 800, dtype=np.float32))
    prepared = prepare_for_asr(samples, 8_000)
    assert prepared.size == 1_600


def test_rejects_empty_payload() -> None:
    try:
        decode_audio_bytes(b"")
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def _write_tone(path: Path) -> None:
    rate = 16_000
    seconds = 0.3
    frames = int(rate * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        pcm = bytearray()
        for i in range(frames):
            value = int(0.4 * 32767 * math.sin(2 * math.pi * 440 * i / rate))
            pcm += int(value).to_bytes(2, "little", signed=True)
        handle.writeframes(bytes(pcm))


if __name__ == "__main__":
    _write_tone(FIXTURE)
