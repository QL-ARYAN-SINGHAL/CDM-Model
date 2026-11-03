#!/bin/bash
# Access-key protected live tracker (Verity + KnowLine tabs) behind its own Cloudflare Quick Tunnel.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
TRACKER_PORT="${TRACKER_PORT:-8012}"
export TRACKER_PORT
command -v cloudflared >/dev/null || { echo "cloudflared is required" >&2; exit 1; }
[ -x venv/bin/python ] || { echo "Missing venv; run ./run.sh once first" >&2; exit 1; }
venv/bin/python tracker_server.py >> tracker_server.log 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT INT TERM
sleep 1
echo "Access key (also in tracker_access_key): $(cat tracker_access_key)"
cloudflared tunnel --url "http://127.0.0.1:${TRACKER_PORT}"
