"""Corpus WER and CER. References and hypotheses are normalized first."""

from __future__ import annotations

from dataclasses import dataclass

import jiwer

from uzbek_voice_agent.normalize import normalize_pair


@dataclass(frozen=True)
class ErrorRates:
    wer: float
    cer: float
    n_utterances: int
    n_ref_words: int
    n_ref_chars: int
    skipped_empty_reference: int


def _normalize_pair(reference: str, hypothesis: str) -> tuple[str, str]:
    return normalize_pair(reference, hypothesis)


def score_utterances(references: list[str], hypotheses: list[str]) -> ErrorRates:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")

    refs: list[str] = []
    hyps: list[str] = []
    skipped = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        norm_ref, norm_hyp = _normalize_pair(reference, hypothesis)
        if not norm_ref:
            skipped += 1
            continue
        refs.append(norm_ref)
        hyps.append(norm_hyp if norm_hyp else "")

    if not refs:
        return ErrorRates(
            wer=1.0,
            cer=1.0,
            n_utterances=0,
            n_ref_words=0,
            n_ref_chars=0,
            skipped_empty_reference=skipped,
        )

    n_words = sum(len(ref.split()) for ref in refs)
    n_chars = sum(len(ref) for ref in refs)
    return ErrorRates(
        wer=float(jiwer.wer(refs, hyps)),
        cer=float(jiwer.cer(refs, hyps)),
        n_utterances=len(refs),
        n_ref_words=n_words,
        n_ref_chars=n_chars,
        skipped_empty_reference=skipped,
    )
