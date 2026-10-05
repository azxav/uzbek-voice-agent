"""LiveKit worker for a local Uzbek ASR pipeline.

The session shape follows LiveKit's MIT-licensed agent-starter-python
(AgentServer, rtc_session, AgentSession, room connect). Speech-to-text is
NavAI Whisper running in this process. The language-model slot is a local
echo stub, so the demo does not need a paid LLM key. Spoken playback needs
a TTS plugin; this worker leaves TTS unset and still publishes the transcript
path through the agent session.

Run against LiveKit Cloud or the compose profile:

    LIVEKIT_URL=ws://localhost:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret \\
        python -m uzbek_voice_agent.agent dev
"""

from __future__ import annotations

import logging

from uzbek_voice_agent import MODEL_ID
from uzbek_voice_agent.asr import UzbekASR
from uzbek_voice_agent.reply import reply_to_transcript

logger = logging.getLogger("uzbek_voice_agent.agent")

INSTRUCTIONS = (
    "You are a short Uzbek voice demo. Repeat the caller's words clearly in Latin-script "
    "Uzbek and do not invent facts. Speak in plain sentences."
)


def pipeline_spec() -> dict[str, str | bool]:
    """Dependency-free description of the worker. Unit tests assert this wiring."""
    return {
        "stt_model": MODEL_ID,
        "stt_provider": "navai-uz",
        "llm": "local-echo-stub",
        "tts": "unset",
        "paid_llm_required": False,
        "livekit_server_required_for_realtime": True,
        "pattern": "livekit-agents AgentServer rtc_session",
    }


def _last_user_text(chat_ctx: object) -> str:
    items = getattr(chat_ctx, "items", None) or getattr(chat_ctx, "messages", [])
    text = ""
    for item in items:
        role = getattr(item, "role", None)
        if role not in {"user", "human"}:
            continue
        content = getattr(item, "text_content", None)
        if content is None:
            raw = getattr(item, "content", "")
            if isinstance(raw, str):
                content = raw
            elif isinstance(raw, list):
                content = " ".join(part for part in raw if isinstance(part, str))
            else:
                content = ""
        text = str(content or "")
    return text


def _frames_to_float(buffer: object) -> tuple[object, int]:
    import numpy as np

    frames = buffer if isinstance(buffer, list) else [buffer]
    if not frames:
        return np.zeros(0, dtype=np.float32), 16_000
    sample_rate = int(frames[0].sample_rate)
    channels = int(getattr(frames[0], "num_channels", 1) or 1)
    chunks = []
    for frame in frames:
        pcm = np.frombuffer(bytes(frame.data), dtype=np.int16).astype(np.float32)
        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1)
        chunks.append(pcm / 32768.0)
    return np.concatenate(chunks), sample_rate


def build_server() -> object:
    """Construct the LiveKit AgentServer. Imports livekit only when called."""
    import asyncio

    from livekit.agents import (
        Agent,
        AgentServer,
        AgentSession,
        JobContext,
        cli,
    )
    from livekit.agents import llm as lk_llm
    from livekit.agents import stt as lk_stt
    from livekit.agents.types import NOT_GIVEN

    class UzbekWhisperSTT(lk_stt.STT):
        def __init__(self, asr: UzbekASR | None = None) -> None:
            super().__init__(
                capabilities=lk_stt.STTCapabilities(streaming=False, interim_results=False)
            )
            self._asr = asr or UzbekASR()

        @property
        def model(self) -> str:
            return self._asr.model_id

        @property
        def provider(self) -> str:
            return "navai-uz"

        async def _recognize_impl(self, buffer, *, language=NOT_GIVEN, conn_options):
            del language, conn_options
            samples, sample_rate = _frames_to_float(buffer)
            text = await asyncio.to_thread(self._asr.transcribe_array, samples, sample_rate)
            return lk_stt.SpeechEvent(
                type=lk_stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[lk_stt.SpeechData(language="uz", text=text)],
            )

    class EchoLLM(lk_llm.LLM):
        @property
        def model(self) -> str:
            return "local-echo-stub"

        @property
        def provider(self) -> str:
            return "uzbek-voice-agent"

        def chat(self, *, chat_ctx, tools=None, conn_options, **kwargs):
            del tools, kwargs
            return EchoStream(self, chat_ctx=chat_ctx, conn_options=conn_options)

    class EchoStream(lk_llm.LLMStream):
        def __init__(self, llm_engine: EchoLLM, *, chat_ctx, conn_options) -> None:
            super().__init__(
                llm_engine,
                chat_ctx=chat_ctx,
                tools=[],
                conn_options=conn_options,
            )
            self._source_ctx = chat_ctx

        async def _run(self) -> None:
            reply = reply_to_transcript(_last_user_text(self._source_ctx))
            self._event_ch.send_nowait(
                lk_llm.ChatChunk(
                    id="echo",
                    delta=lk_llm.ChoiceDelta(role="assistant", content=reply),
                )
            )

    class UzbekAssistant(Agent):
        def __init__(self) -> None:
            super().__init__(instructions=INSTRUCTIONS, llm=EchoLLM())

    server = AgentServer()

    @server.rtc_session(agent_name="uzbek-voice-agent")
    async def uzbek_session(ctx: JobContext) -> None:
        ctx.log_context_fields = {"room": ctx.room.name}
        session = AgentSession(stt=UzbekWhisperSTT())
        await session.start(agent=UzbekAssistant(), room=ctx.room)
        await ctx.connect()

    server.cli = cli  # type: ignore[attr-defined]
    return server


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(".env.local")
    load_dotenv(".env")
    server = build_server()
    cli = server.cli  # type: ignore[attr-defined]
    cli.run_app(server)


if __name__ == "__main__":
    main()
