#!/bin/bash
# SysMonitor Startup Script
# Usage: ./start.sh [port]

PORT=${1:-7070}
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║          SysMonitor v1.0                 ║"
echo "║  Linux CPU/GPU/Memory/Process Monitor    ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found. Please install python3."
    exit 1
fi

# Install dependencies
echo "[*] Checking dependencies..."
pip3 install flask flask-cors psutil --break-system-packages -q 2>/dev/null || \
pip3 install flask flask-cors psutil -q 2>/dev/null || \
pip install flask flask-cors psutil -q 2>/dev/null

echo "[*] Starting server on port $PORT..."
echo "[*] Open browser: http://localhost:$PORT"
echo ""
echo "    Press Ctrl+C to stop"
echo ""

cd "$DIR"
python3 server.py "$PORT"
