"""Local reply stub. This is not a language model and needs no API key."""

from __future__ import annotations


def reply_to_transcript(transcript: str) -> str:
    cleaned = " ".join(transcript.split())
    if not cleaned:
        return "Kechirasiz, ovozingizni eshitolmadim. Qaytadan ayting."
    return f"Siz aytdingiz: {cleaned}"
