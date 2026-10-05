FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src

# CPU wheels. GPU hosts can rebuild with the default PyTorch index and ASR_DEVICE=cuda.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ".[asr,agent]"

ENV ASR_DEVICE=cpu \
    HF_HOME=/cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "uzbek_voice_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
