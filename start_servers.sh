#!/usr/bin/env bash
# ============================================================
# CardioPulse AI - Server Launch Script (Linux / macOS / WSL)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if present
if [ -d "venv" ]; then
    echo "[INFO] Activating virtual environment..."
    source venv/bin/activate
fi

echo "[INFO] Starting FastAPI application on port 8000..."
uvicorn api:app --reload --port 8000 &
UVICORN_PID=$!

echo "[INFO] Starting static web server on port 8080..."
cd "$SCRIPT_DIR/static"
python3 -m http.server 8080 &
STATIC_PID=$!

echo "$UVICORN_PID" > "$SCRIPT_DIR/.uvicorn.pid"
echo "$STATIC_PID" > "$SCRIPT_DIR/.static.pid"

sleep 2

echo "============================================================"
echo "  CardioPulse AI Services Active!"
echo "  - Unified Dashboard: http://127.0.0.1:8000/dashboard"
echo "  - Swagger Docs:      http://127.0.0.1:8000/docs"
echo "  - Welcome Portal:    http://127.0.0.1:8080/intro.html"
echo "============================================================"
echo "To terminate, run: ./stop_servers.sh"

wait
