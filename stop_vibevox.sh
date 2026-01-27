#!/bin/bash

# VibeVox Stop Script
# Stops all running VibeVox servers

echo "🛑 Stopping VibeVox..."

# Check for PID files
if [ -f logs/tts_server.pid ]; then
    TTS_PID=$(cat logs/tts_server.pid)
    if kill -0 "$TTS_PID" 2>/dev/null; then
        echo "  Stopping TTS Server (PID: $TTS_PID)..."
        kill "$TTS_PID"
        echo "  ✓ TTS Server stopped"
    fi
    rm -f logs/tts_server.pid
fi

if [ -f logs/web_server.pid ]; then
    WEB_PID=$(cat logs/web_server.pid)
    if kill -0 "$WEB_PID" 2>/dev/null; then
        echo "  Stopping Web Server (PID: $WEB_PID)..."
        kill "$WEB_PID"
        echo "  ✓ Web Server stopped"
    fi
    rm -f logs/web_server.pid
fi

# Fallback: kill by process name
echo "  Checking for any remaining processes..."
pkill -f 'python.*tts_server' 2>/dev/null && echo "  ✓ Killed remaining TTS processes"
pkill -f 'python.*web.app' 2>/dev/null && echo "  ✓ Killed remaining Web processes"

echo ""
echo "✅ VibeVox stopped successfully"
