"""Text normalization for WER/CER.

NavAI scores with the Apache-2.0 ``uzbek_text_norm`` package (Cyrillic folding,
spelled-out numbers, okina apostrophes). When that package is installed, this
module uses it. Otherwise it applies a smaller, explicit local normalizer and
the eval report says so. Numbers are not spelled out by the local fallback.
"""

from __future__ import annotations

import re
import unicodedata

_HYPHEN = re.compile("[" + re.escape("-\u2010\u2011\u2012\u2013\u2014") + "]+")
_NON_WORD = re.compile(r"[^\w\sʻ]", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")

_APOSTROPHES = str.maketrans(
    {
        "'": "ʻ",
        "’": "ʻ",
        "‘": "ʻ",
        "`": "ʻ",
        "ʼ": "ʻ",
        "ʹ": "ʻ",
        "´": "ʻ",
    }
)


def local_normalize(text: str) -> str:
    """Lowercase, strip punctuation, fold apostrophes, split hyphens."""
    folded = unicodedata.normalize("NFC", text).lower().translate(_APOSTROPHES)
    folded = _HYPHEN.sub(" ", folded)
    folded = _NON_WORD.sub(" ", folded)
    return _SPACES.sub(" ", folded).strip()


def normalization_backend() -> str:
    try:
        import uzbek_text_norm

        version = getattr(uzbek_text_norm, "__version__", "unknown")
        return f"uzbek_text_norm@{version}"
    except Exception:
        return "repo-local-v1"


def normalize_pair(reference: str, hypothesis: str) -> tuple[str, str]:
    """Normalize a gold/hypothesis pair with the same rules on both sides."""
    if normalization_backend().startswith("uzbek_text_norm"):
        try:
            import uzbek_text_norm

            return (
                uzbek_text_norm.normalize_reference(reference),
                uzbek_text_norm.normalize_hypothesis(hypothesis),
            )
        except Exception:
            pass
    return local_normalize(reference), local_normalize(hypothesis)


def normalize_transcript(text: str) -> str:
    """Single-string helper. Scoring uses normalize_pair so refs and hyps stay aligned."""
    return normalize_pair(text, text)[0]
