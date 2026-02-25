#!/bin/bash

# VibeVox Startup Script
# Runs both the TTS server and the web API server

set -euo pipefail

echo "🎙️ Starting VibeVox System..."
echo ""

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "⚠️  Warning: Virtual environment not activated"
    echo "   Continuing with system Python as requested"
    echo ""
fi

# Check GROQ API key
if [[ -z "${GROQ_API_KEY:-}" ]]; then
    echo "❌ Error: GROQ_API_KEY environment variable not set"
    echo "   Please set it: export GROQ_API_KEY='your-api-key-here'"
    exit 1
fi

# Configuration
TTS_PORT=${TTS_PORT:-5000}
WEB_PORT=${VIBEVOX_PORT:-8000}
TTS_HOST=${TTS_HOST:-127.0.0.1}
WEB_HOST=${VIBEVOX_HOST:-127.0.0.1}

echo "🔧 Configuration:"
echo "   TTS Server: http://$TTS_HOST:$TTS_PORT"
echo "   Web Server: http://$WEB_HOST:$WEB_PORT"
echo ""

# Create logs directory
mkdir -p logs

# Ensure required tools are present
command -v python >/dev/null 2>&1 || { echo "❌ python not found"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "❌ curl not found"; exit 1; }

# Cleanup stale PID files if needed
for pidfile in logs/tts_server.pid logs/web_server.pid; do
    if [[ -f "$pidfile" ]]; then
        existing_pid=$(cat "$pidfile" || true)
        if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
            echo "⚠️  Existing VibeVox process detected (PID: $existing_pid). Run ./stop_vibevox.sh first."
            exit 1
        fi
        rm -f "$pidfile"
    fi
done

wait_for_http() {
    local url="$1"
    local timeout_sec="$2"
    local waited=0
    while [[ "$waited" -lt "$timeout_sec" ]]; do
        if curl -fsS "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done
    return 1
}

# Start TTS Server in background
echo "🚀 Starting TTS Server (Qwen3-TTS)..."
TTS_HOST=$TTS_HOST TTS_PORT=$TTS_PORT python tts_server.py > logs/tts_server.log 2>&1 &
TTS_PID=$!
echo "   TTS Server PID: $TTS_PID"

# Wait for TTS server to start
echo "   Waiting for TTS server to initialize..."

# Check if TTS server is running
if ! kill -0 "$TTS_PID" 2>/dev/null; then
    echo "❌ TTS Server failed to start. Check logs/tts_server.log"
    exit 1
fi

if ! wait_for_http "http://$TTS_HOST:$TTS_PORT/health" 30; then
    echo "❌ TTS health endpoint did not become ready in time"
    kill "$TTS_PID" 2>/dev/null || true
    exit 1
fi

# Start Web API Server in background
echo "🚀 Starting Web API Server..."
VIBEVOX_HOST=$WEB_HOST VIBEVOX_PORT=$WEB_PORT python -m web.app > logs/web_server.log 2>&1 &
WEB_PID=$!
echo "   Web Server PID: $WEB_PID"

# Check if web server is running
if ! kill -0 "$WEB_PID" 2>/dev/null; then
    echo "❌ Web Server failed to start. Check logs/web_server.log"
    kill "$TTS_PID" 2>/dev/null || true
    exit 1
fi

if ! wait_for_http "http://$WEB_HOST:$WEB_PORT/api/health" 30; then
    echo "❌ Web API health endpoint did not become ready in time"
    kill "$WEB_PID" 2>/dev/null || true
    kill "$TTS_PID" 2>/dev/null || true
    exit 1
fi

echo ""
echo "✅ VibeVox is running!"
echo ""
echo "📊 Services:"
echo "   • TTS Server:  http://$TTS_HOST:$TTS_PORT/health"
echo "   • Web UI:      http://$WEB_HOST:$WEB_PORT/"
echo "   • API Health:  http://$WEB_HOST:$WEB_PORT/api/health"
echo ""
echo "📝 Logs:"
echo "   • TTS Server:  tail -f logs/tts_server.log"
echo "   • Web Server:  tail -f logs/web_server.log"
echo ""
echo "⏹️  To stop: kill $TTS_PID $WEB_PID"
echo "   Or use: pkill -f 'python.*tts_server' && pkill -f 'python.*web.app'"
echo ""

# Save PIDs to file for later cleanup
echo "$TTS_PID" > logs/tts_server.pid
echo "$WEB_PID" > logs/web_server.pid

echo "Press Ctrl+C to view logs in real-time, or use the commands above"
echo ""

# Keep script running and show logs
trap "echo ''; echo 'Shutting down...'; kill $WEB_PID $TTS_PID 2>/dev/null || true; exit 0" INT TERM

# Follow both logs
tail -f logs/tts_server.log logs/web_server.log
