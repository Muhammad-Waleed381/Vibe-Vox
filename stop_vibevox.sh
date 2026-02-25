#!/bin/bash

set -euo pipefail

# VibeVox Stop Script
# Stops all running VibeVox servers

echo "🛑 Stopping VibeVox..."

stop_pid() {
    local pid="$1"
    local name="$2"

    if ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi

    echo "  Stopping $name (PID: $pid)..."
    kill "$pid" 2>/dev/null || true

    for _ in {1..10}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "  ✓ $name stopped"
            return 0
        fi
        sleep 0.5
    done

    echo "  ⚠️  $name still running, forcing stop"
    kill -9 "$pid" 2>/dev/null || true
}

# Check for PID files
if [ -f logs/tts_server.pid ]; then
    TTS_PID=$(cat logs/tts_server.pid)
    stop_pid "$TTS_PID" "TTS Server"
    rm -f logs/tts_server.pid
fi

if [ -f logs/web_server.pid ]; then
    WEB_PID=$(cat logs/web_server.pid)
    stop_pid "$WEB_PID" "Web Server"
    rm -f logs/web_server.pid
fi

# Fallback: kill by process name
echo "  Checking for any remaining processes..."
pkill -f 'python.*tts_server' 2>/dev/null && echo "  ✓ Killed remaining TTS processes"
pkill -f 'python.*web.app' 2>/dev/null && echo "  ✓ Killed remaining Web processes"

echo ""
echo "✅ VibeVox stopped successfully"
