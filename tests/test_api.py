from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from uzbek_voice_agent.api import create_app

FIXTURE = Path(__file__).parent / "fixtures" / "tone.wav"


class FakeASR:
    model_id = "fake-navai"
    device = "cpu"
    loaded = False

    def transcribe_array(self, samples: np.ndarray, sample_rate: int) -> str:
        self.loaded = True
        assert sample_rate == 16_000
        assert samples.size > 0
        return "salom"


def test_health() -> None:
    client = TestClient(create_app(FakeASR()))
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "fake-navai"
    assert body["model_loaded"] is False


def test_transcribe_fixture_wav() -> None:
    client = TestClient(create_app(FakeASR()))
    with FIXTURE.open("rb") as handle:
        response = client.post("/transcribe", files={"file": ("tone.wav", handle, "audio/wav")})
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "salom"
    assert body["reply"] == "Siz aytdingiz: salom"
    assert body["reply_kind"] == "echo-stub"
    assert body["model"] == "fake-navai"
    assert body["duration_seconds"] == 0.3


def test_transcribe_rejects_garbage() -> None:
    client = TestClient(create_app(FakeASR()))
    response = client.post("/transcribe", files={"file": ("bad.txt", b"not audio", "text/plain")})
    assert response.status_code == 400


def test_index_mentions_upload() -> None:
    client = TestClient(create_app(FakeASR()))
    response = client.get("/")
    assert response.status_code == 200
    assert "Transcribe" in response.text
