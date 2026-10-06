# Uzbek voice agent

Personal project by Azizbek Xasanov (azxav).

An Uzbek speech demo: a FastAPI upload path, plus an optional LiveKit worker that transcribes with `navai-uz/whisper-small-uzbek`. The file API runs with no LiveKit account. Realtime audio needs a LiveKit server. Compose starts one locally; LiveKit Cloud is optional.

## Try it

1. `POST /transcribe` with a wav, flac, or ogg file. The response includes the transcript and an echo line, so the turn shape is visible without an LLM key.
2. `GET /health` for process status. It does not download the model.
3. `make eval-asr` for word and character error on a stated prefix of a public Uzbek test set. The command refuses every split except `test` and does not train.

## Architecture

```mermaid
flowchart LR
  caller[Caller audio]
  api[FastAPI /transcribe]
  worker[LiveKit worker]
  asr[NavAI whisper-small-uzbek]
  echo[Local echo stub]
  room[LiveKit server optional]

  caller -->|file upload| api
  api --> asr
  asr --> echo
  caller -->|WebRTC| room
  room --> worker
  worker --> asr
  worker --> echo
```

The worker session is `AgentServer`, an `rtc_session` entrypoint, `AgentSession`, then `ctx.connect()`. Speech-to-text is `navai-uz/whisper-small-uzbek` running in process. The LLM slot is an in-process echo (`Siz aytdingiz: ...`). TTS is unset, so the room gets the agent text path and does not synthesize speech.

## Run the file API

CPU is the default. A GPU is optional: install a CUDA PyTorch build and set `ASR_DEVICE=cuda`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[asr,dev]"
uvicorn uzbek_voice_agent.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` and upload a clip, or:

```bash
curl -s -F "file=@tests/fixtures/tone.wav" http://localhost:8000/transcribe
curl -s http://localhost:8000/health
```

The first transcription downloads `navai-uz/whisper-small-uzbek` from Hugging Face (about 1 GB of weights).

### Docker Compose

```bash
docker compose up --build api
```

That publishes the API on port 8000 and caches weights in a volume. It does not start LiveKit.

## LiveKit

Start a local server with no Cloud account:

```bash
docker compose --profile realtime up --build
```

That starts `livekit-server` with the demo key pair in `deploy/livekit.yaml` (`devkey` / `secret`), plus the agent worker. Those credentials are for a machine under local control. Change them before any public port forward.

Mint a join token from the API once the `agent` extra is installed:

```bash
curl -s -X POST "http://localhost:8000/livekit/token?room=uzbek-demo&identity=guest"
```

Point a LiveKit client at `ws://localhost:7880` with that token. The worker loads the same Whisper model on CPU, so the first utterance is slow.

Without Docker, run the same worker this way:

```bash
export LIVEKIT_URL=ws://localhost:7880
export LIVEKIT_API_KEY=devkey
export LIVEKIT_API_SECRET=secret
python -m uzbek_voice_agent.agent dev
```

LiveKit Cloud is optional. Set `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET` from the Cloud project and run the worker the same way. Those values are not committed.

## Eval

Install the scorer first so the numbers match the table. `uzbek-text-norm` is Apache-2.0 and is not on PyPI under that name:

```bash
pip install "uzbek-text-norm @ git+https://github.com/NavAI-pro/uzbek-text-norm.git"
make eval-asr LIMIT=8
```

Measured on CPU with greedy decode (`num_beams=1`, `no_repeat_ngram_size=4`). Source file: `eval/fleurs_uz_prefix.json`. The model was not trained for this run, and the test split was not used to fit weights.

| Item | Value |
| --- | --- |
| Model | `navai-uz/whisper-small-uzbek` |
| Set | `google/fleurs`, config `uz_uz`, split `test` |
| Slice | first 8 streamed rows; id `1882` was repeated, so **7 unique utterances** were scored |
| Reference field | `transcription` |
| Normalizer | `uzbek_text_norm` 0.3.0, reference and hypothesis |
| Reference words / chars | 133 / 1057 |
| **WER** | **14.29%** |
| **CER** | **7.00%** |

Scoring the repeated row as well would count one perfect utterance twice and print 12.50% WER on 8 rows. The table drops that repeat.

The model card reports **16.96 WER** on the full FLEURS Uzbek test set, **9.58 WER** on Common Voice 22 Uzbek test, and **11.57 macro WER** across FLEURS, Common Voice, USC, and FeruzaSpeech. Those are full-set figures. The 14.29% on 7 utterances is a different result. One clip in this slice (id `1685`, about 35 mm film) is badly wrong; several others match.

## Tests

CI installs `.[dev]` only. It lints with Ruff and runs pytest against `tests/fixtures/tone.wav`. It does not download Hugging Face weights.

```bash
pip install -e ".[dev]"
make ci
```

## Limits

- Whisper-small on CPU is the supported default. It is not a realtime call-center stack.
- The echo reply does not answer questions. The core demo does not include a paid LLM.
- TTS is unwired, so the LiveKit worker does not speak audio back.
- Telephony and SIP are out of scope.
- This repo does not fine-tune Whisper.
- The subset WER/CER covers 7 utterances. It is not the full-set FLEURS number on the model card.
- Without `uzbek_text_norm`, `make eval-asr` falls back to a smaller local normalizer and the JSON says so. Digit spelling will not match the table above.
- Far-field noise, strong dialect, and rare names are weak spots called out on the model card.
- Demo LiveKit keys in `deploy/livekit.yaml` are not a production secret.

## Credits and licenses

| Piece | License |
| --- | --- |
| This repository's code | MIT, see `LICENSE` |
| [livekit-examples/agent-starter-python](https://github.com/livekit-examples/agent-starter-python) | MIT, Copyright (c) 2025 LiveKit, Inc. |
| [livekit/agents](https://github.com/livekit/agents) | Apache-2.0 |
| [navai-uz/whisper-small-uzbek](https://huggingface.co/navai-uz/whisper-small-uzbek) | Apache-2.0 weights |
| `google/fleurs` `uz_uz` test audio, used only at eval time | CC-BY-4.0 |

Details are in `NOTICE`.
