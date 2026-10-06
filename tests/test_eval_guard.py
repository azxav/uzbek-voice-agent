from pathlib import Path

import pytest

from uzbek_voice_agent.eval_asr import _audio_array, main, unique_rows

FIXTURE = Path(__file__).parent / "fixtures" / "tone.wav"


def test_eval_refuses_train_split() -> None:
    with pytest.raises(SystemExit, match="refusing split"):
        main(["--split", "train", "--limit", "1"])


def test_eval_refuses_validation_split() -> None:
    with pytest.raises(SystemExit, match="refusing split"):
        main(["--split", "validation"])


def test_unique_rows_drops_repeated_id() -> None:
    kept, dropped = unique_rows([{"id": 1}, {"id": 2}, {"id": 1}])
    assert [row["id"] for _, row in kept] == [1, 2]
    assert dropped == [{"index": 2, "id": 1}]


def test_audio_row_decodes_wav_bytes() -> None:
    samples, sample_rate = _audio_array(
        {"audio": {"bytes": FIXTURE.read_bytes(), "path": "tone.wav"}}
    )
    assert sample_rate == 16_000
    assert len(samples) == 4_800
