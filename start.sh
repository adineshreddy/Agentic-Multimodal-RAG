#!/usr/bin/env bash
# start.sh — export .env vars into the shell, then start backend + frontend
set -e

if [ ! -f ".env" ]; then
  echo "ERROR: .env file not found. Copy .env.example → .env and fill in your keys."
  exit 1
fi

echo "Stopping any previous backend/frontend processes..."
pkill -f "uvicorn app.main" 2>/dev/null || true
pkill -f "streamlit run"    2>/dev/null || true
sleep 1

# ── Export every variable from .env into this shell session ────────────────
# set -a makes all subsequent variable assignments auto-exported to child processes
set -a
# shellcheck disable=SC1091
source .env
set +a
echo "✓ Loaded $(grep -c '^[A-Z]' .env) variables from .env"

echo "Starting FastAPI backend on http://localhost:8000 ..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

sleep 2

echo "Starting Streamlit frontend on http://localhost:8501 ..."
streamlit run frontend/streamlit_app.py --server.port 8501 &
FRONTEND_PID=$!

echo ""
echo "========================================="
echo "  Backend:  http://localhost:8000/docs"
echo "  Frontend: http://localhost:8501"
echo "========================================="
echo "Press Ctrl+C to stop both services."

trap "kill $API_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'" EXIT INT TERM
wait
