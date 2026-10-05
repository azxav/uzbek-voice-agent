"""NavAI Whisper-small Uzbek transcription.

Weights: https://huggingface.co/navai-uz/whisper-small-uzbek (Apache-2.0).
Default device is CPU. Set ASR_DEVICE=cuda to use a GPU when one is present.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from uzbek_voice_agent import MODEL_ID
from uzbek_voice_agent.audio import prepare_for_asr

# Greedy decode. NavAI's published generation_config also sets this repetition guard.
GENERATE_KWARGS: dict[str, Any] = {
    "language": "uz",
    "task": "transcribe",
    "num_beams": 1,
    "no_repeat_ngram_size": 4,
}


def resolve_device(requested: str | None = None) -> str:
    device = (requested or os.environ.get("ASR_DEVICE") or "cpu").strip().lower()
    if device in {"gpu", "cuda"}:
        return "cuda"
    return "cpu"


class UzbekASR:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = resolve_device(device)
        self._pipeline: Any | None = None

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        import torch
        from transformers import pipeline

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("ASR_DEVICE=cuda but torch.cuda.is_available() is false")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=self.device,
            dtype=dtype,
        )
        return self._pipeline

    def transcribe_array(self, samples: np.ndarray, sample_rate: int) -> str:
        audio = prepare_for_asr(samples, sample_rate)
        if audio.size == 0:
            return ""
        asr = self.load()
        result = asr(
            {"array": audio, "sampling_rate": 16_000},
            generate_kwargs=GENERATE_KWARGS,
        )
        text = result["text"] if isinstance(result, dict) else str(result)
        return text.strip()
