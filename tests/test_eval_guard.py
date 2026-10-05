import pytest

from uzbek_voice_agent.eval_asr import main


def test_eval_refuses_train_split() -> None:
    with pytest.raises(SystemExit, match="refusing split"):
        main(["--split", "train", "--limit", "1"])


def test_eval_refuses_validation_split() -> None:
    with pytest.raises(SystemExit, match="refusing split"):
        main(["--split", "validation"])
