"""FastAPI demo: health and file transcription. No LiveKit room required."""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from uzbek_voice_agent import MODEL_ID, __version__
from uzbek_voice_agent.asr import UzbekASR, resolve_device
from uzbek_voice_agent.audio import TARGET_SAMPLE_RATE, decode_audio_bytes, prepare_for_asr
from uzbek_voice_agent.reply import reply_to_transcript

MAX_UPLOAD_BYTES = 25 * 1024 * 1024

_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Uzbek voice agent</title>
  <style>
    body { font-family: Georgia, serif; max-width: 40rem; margin: 2rem auto; padding: 0 1rem; color: #1c1917; }
    code { font-family: ui-monospace, monospace; }
    button { font: inherit; padding: 0.4rem 0.8rem; }
    pre { white-space: pre-wrap; background: #f5f5f4; padding: 1rem; }
  </style>
</head>
<body>
  <h1>Uzbek voice agent</h1>
  <p>Upload a wav, flac, or ogg file. Transcription uses NavAI Whisper-small Uzbek on this machine. The reply line is a local echo, not a paid language model.</p>
  <form id="form">
    <input id="file" type="file" accept="audio/*,.wav,.flac,.ogg" required>
    <button type="submit">Transcribe</button>
  </form>
  <pre id="out">Waiting for a file.</pre>
  <script>
    const form = document.getElementById("form");
    const out = document.getElementById("out");
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const data = new FormData();
      data.append("file", document.getElementById("file").files[0]);
      out.textContent = "Transcribing…";
      const response = await fetch("/transcribe", { method: "POST", body: data });
      const body = await response.text();
      out.textContent = response.ok ? body : ("Request failed\\n" + body);
    });
  </script>
</body>
</html>
"""


class Transcriber(Protocol):
    model_id: str
    device: str

    def transcribe_array(self, samples: np.ndarray, sample_rate: int) -> str: ...


def create_app(transcriber: Transcriber | None = None) -> FastAPI:
    app = FastAPI(title="Uzbek voice agent", version=__version__)
    app.state.transcriber = transcriber or UzbekASR()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _PAGE

    @app.get("/health")
    def health() -> dict[str, str | bool]:
        engine: Transcriber = app.state.transcriber
        loaded = bool(getattr(engine, "loaded", False))
        return {
            "status": "ok",
            "version": __version__,
            "model": getattr(engine, "model_id", MODEL_ID),
            "device": getattr(engine, "device", resolve_device()),
            "model_loaded": loaded,
        }

    @app.post("/transcribe")
    async def transcribe(file: UploadFile = File(...)) -> dict[str, object]:  # noqa: B008
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="file exceeds 25 MB")
        try:
            samples, sample_rate = decode_audio_bytes(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        prepared = prepare_for_asr(samples, sample_rate)
        duration = float(prepared.size / TARGET_SAMPLE_RATE) if prepared.size else 0.0
        engine: Transcriber = app.state.transcriber
        try:
            text = engine.transcribe_array(samples, sample_rate)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"transcription failed: {exc}") from exc
        return {
            "text": text,
            "reply": reply_to_transcript(text),
            "reply_kind": "echo-stub",
            "model": engine.model_id,
            "device": engine.device,
            "duration_seconds": round(duration, 3),
            "sample_rate": sample_rate,
        }

    @app.post("/livekit/token")
    def livekit_token(room: str = "uzbek-demo", identity: str = "guest") -> dict[str, str]:
        """Mint a local-dev token. Requires the agent extra and a running LiveKit server."""
        api_key = os.environ.get("LIVEKIT_API_KEY", "devkey")
        api_secret = os.environ.get("LIVEKIT_API_SECRET", "secret")
        livekit_url = os.environ.get("LIVEKIT_URL", "ws://localhost:7880")
        try:
            from livekit.api import AccessToken, VideoGrants
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail="livekit-agents is not installed; pip install -e '.[agent]'",
            ) from exc
        if not room.strip() or not identity.strip():
            raise HTTPException(status_code=400, detail="room and identity are required")
        token = (
            AccessToken(api_key, api_secret)
            .with_identity(identity.strip())
            .with_name(identity.strip())
            .with_grants(VideoGrants(room_join=True, room=room.strip()))
            .to_jwt()
        )
        return {"url": livekit_url, "token": token, "room": room.strip()}

    return app


app = create_app()


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("uzbek_voice_agent.api:app", host=host, port=port, reload=False)
