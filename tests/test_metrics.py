from uzbek_voice_agent.metrics import score_utterances
from uzbek_voice_agent.normalize import local_normalize


def test_perfect_transcript_is_zero() -> None:
    rates = score_utterances(["Salom, dunyo!"], ["salom dunyo"])
    assert rates.wer == 0.0
    assert rates.cer == 0.0
    assert rates.n_utterances == 1
    assert rates.n_ref_words == 2


def test_one_word_substitution() -> None:
    rates = score_utterances(["salom dunyo"], ["salom olam"])
    assert rates.wer == 0.5
    assert rates.n_ref_words == 2


def test_hyphen_and_apostrophe_fold() -> None:
    assert local_normalize("O'zbek-tili!") == "oʻzbek tili"


def test_empty_reference_is_skipped() -> None:
    rates = score_utterances(["...", "salom"], ["", "salom"])
    assert rates.skipped_empty_reference == 1
    assert rates.n_utterances == 1
    assert rates.wer == 0.0
