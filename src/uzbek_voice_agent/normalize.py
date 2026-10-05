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


def _navai_normalize(text: str) -> str:
    import uzbek_text_norm

    normalize = getattr(uzbek_text_norm, "normalize", None)
    if normalize is None:
        raise AttributeError("uzbek_text_norm.normalize is missing")
    return str(normalize(text))


def normalization_backend() -> str:
    try:
        import uzbek_text_norm

        version = getattr(uzbek_text_norm, "__version__", "unknown")
        return f"uzbek_text_norm@{version}"
    except Exception:
        return "repo-local-v1"


def normalize_transcript(text: str) -> str:
    if normalization_backend().startswith("uzbek_text_norm"):
        try:
            return _navai_normalize(text)
        except Exception:
            return local_normalize(text)
    return local_normalize(text)
