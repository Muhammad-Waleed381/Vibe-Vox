FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    NLTK_DATA=/app/.cache/nltk \
    TTS_HOST=0.0.0.0 \
    TTS_PORT=5000 \
    VIBEVOX_HOST=0.0.0.0 \
    VIBEVOX_PORT=8000 \
    LOAD_BASE_MODEL=true \
    LOAD_CUSTOM_VOICE_MODEL=false \
    LOAD_DESIGN_MODEL=true \
    MODEL_SIZE=1.7B

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN sed '/^flash-attn/d' requirements.txt > /tmp/requirements-nofa.txt && \
    pip install --no-cache-dir -r /tmp/requirements-nofa.txt && \
    rm /tmp/requirements-nofa.txt

RUN python -m nltk.downloader punkt

COPY . .

RUN chmod +x docker-entrypoint.sh

EXPOSE 5000 8000

VOLUME ["/app/.cache", "/app/exports"]

ENTRYPOINT ["/app/docker-entrypoint.sh"]
