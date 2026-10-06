from uzbek_voice_agent.agent import pipeline_spec
from uzbek_voice_agent.reply import reply_to_transcript


def test_echo_reply() -> None:
    assert reply_to_transcript("  salom   dunyo ") == "Siz aytdingiz: salom dunyo"


def test_empty_reply_asks_again() -> None:
    assert "eshitolmadim" in reply_to_transcript("   ")


def test_pipeline_does_not_require_paid_llm() -> None:
    spec = pipeline_spec()
    assert spec["stt_model"] == "navai-uz/whisper-small-uzbek"
    assert spec["llm"] == "local-echo-stub"
    assert spec["paid_llm_required"] is False
    assert spec["livekit_server_required_for_realtime"] is True
