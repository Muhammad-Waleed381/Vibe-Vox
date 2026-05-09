#!/bin/bash
set -euo pipefail

echo "============================================"
echo "  VibeVox — starting services"
echo "  Model size: ${MODEL_SIZE:-1.7B}"
echo "  Base: ${LOAD_BASE_MODEL:-true}, CustomVoice: ${LOAD_CUSTOM_VOICE_MODEL:-false}, Design: ${LOAD_DESIGN_MODEL:-true}"
echo "============================================"

# ── 1. Start TTS server in background ──
echo "[1/3] Starting TTS server..."
python /app/tts_server.py &
TTS_PID=$!

# ── 2. Wait for TTS to be healthy ──
echo "[2/3] Waiting for TTS server to become ready..."
for i in $(seq 1 300); do
    if curl -sf http://127.0.0.1:5000/health > /dev/null 2>&1; then
        echo "      TTS server ready (attempt $i)"
        break
    fi
    if [ $i -eq 300 ]; then
        echo "ERROR: TTS server failed to start after ~10 minutes"
        kill $TTS_PID 2>/dev/null || true
        exit 1
    fi
    sleep 2
done

# ── 3. Start web server in foreground ──
echo "[3/3] Starting web server..."
echo "      UI at http://0.0.0.0:${VIBEVOX_PORT:-8000}"
echo ""
exec python -m web.app
