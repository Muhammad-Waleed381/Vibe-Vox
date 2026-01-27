#!/bin/bash

# VibeVox Startup Script
# Runs both the TTS server and the web API server

set -e

echo "🎙️ Starting VibeVox System..."
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Warning: Virtual environment not activated"
    echo "   It's recommended to activate venv first: source venv/bin/activate"
    echo ""
fi

# Check GROQ API key
if [[ -z "$GROQ_API_KEY" ]]; then
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

# Start TTS Server in background
echo "🚀 Starting TTS Server (Qwen3-TTS)..."
TTS_HOST=$TTS_HOST TTS_PORT=$TTS_PORT python tts_server.py > logs/tts_server.log 2>&1 &
TTS_PID=$!
echo "   TTS Server PID: $TTS_PID"

# Wait a bit for TTS server to start
echo "   Waiting for TTS server to initialize..."
sleep 5

# Check if TTS server is running
if ! kill -0 $TTS_PID 2>/dev/null; then
    echo "❌ TTS Server failed to start. Check logs/tts_server.log"
    exit 1
fi

# Start Web API Server in background
echo "🚀 Starting Web API Server..."
VIBEVOX_HOST=$WEB_HOST VIBEVOX_PORT=$WEB_PORT python -m web.app > logs/web_server.log 2>&1 &
WEB_PID=$!
echo "   Web Server PID: $WEB_PID"

# Wait a bit for web server to start
sleep 3

# Check if web server is running
if ! kill -0 $WEB_PID 2>/dev/null; then
    echo "❌ Web Server failed to start. Check logs/web_server.log"
    kill $TTS_PID 2>/dev/null
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
trap "echo ''; echo 'Shutting down...'; kill $TTS_PID $WEB_PID 2>/dev/null; exit 0" INT TERM

# Follow both logs
tail -f logs/tts_server.log logs/web_server.log
