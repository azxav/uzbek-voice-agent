"""Plugin checks that run only when livekit-agents is installed. CI skips them."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

pytest.importorskip("livekit.agents")

from livekit import rtc  # noqa: E402
from livekit.agents import llm  # noqa: E402

from uzbek_voice_agent.agent import _frames_to_float, build_server  # noqa: E402


class FakeASR:
    model_id = "fake-navai"
    device = "cpu"

    def transcribe_array(self, samples: np.ndarray, sample_rate: int) -> str:
        assert sample_rate == 16_000
        assert samples.size == 3
        return "salom"


def _frame() -> rtc.AudioFrame:
    pcm = np.array([0, 16384, -16384], dtype=np.int16)
    return rtc.AudioFrame(
        data=pcm.tobytes(),
        sample_rate=16_000,
        num_channels=1,
        samples_per_channel=3,
    )


def test_frames_to_float_scales_pcm() -> None:
    samples, sample_rate = _frames_to_float(_frame())
    assert sample_rate == 16_000
    assert samples.shape == (3,)
    assert samples[1] == pytest.approx(0.5, abs=1e-3)


def test_stt_recognize_uses_injected_asr() -> None:
    server = build_server()
    stt = server.stt_factory(FakeASR())

    async def _run():
        return await stt.recognize(_frame())

    event = asyncio.run(_run())
    assert event.alternatives[0].text == "salom"
    assert stt.model == "fake-navai"
    assert stt.provider == "navai-uz"


def test_echo_llm_repeats_user_text() -> None:
    server = build_server()
    engine = server.llm_factory()
    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="user", content="salom")

    async def _run() -> str:
        stream = engine.chat(chat_ctx=chat_ctx)
        chunks = [chunk async for chunk in stream]
        return "".join(chunk.delta.content or "" for chunk in chunks if chunk.delta)

    assert asyncio.run(_run()) == "Siz aytdingiz: salom"
