#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared is required; install it from Cloudflare first." >&2
  exit 1
fi
if lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port 8000 is already in use. Stop the existing Verity server, then retry." >&2
  exit 1
fi
if lsof -nP -iTCP:8001 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port 8001 is already in use. Stop the existing public proxy, then retry." >&2
  exit 1
fi
if [ ! -x venv/bin/python ]; then
  echo "Missing venv/bin/python. Start Verity once with ./run.sh to create its environment." >&2
  exit 1
fi

export VERITY_MODEL_PATH="${VERITY_MODEL_PATH:-$ROOT/models/verity/best}"
export VERITY_MODEL_NAME="${VERITY_MODEL_NAME:-Verity V4}"
if [ ! -f "$VERITY_MODEL_PATH/model.safetensors" ]; then
  echo "Missing checkpoint: $VERITY_MODEL_PATH/model.safetensors" >&2
  exit 1
fi

./ensure_kev.sh >> kev_server.log 2>&1 &
KEV_PID=$!

venv/bin/python api_server.py >> verity_server.log 2>&1 &
API_PID=$!
venv/bin/python public_proxy.py >> public_proxy.log 2>&1 &
PROXY_PID=$!
cleanup() {
  kill "$PROXY_PID" 2>/dev/null || true
  wait "$PROXY_PID" 2>/dev/null || true
  kill "$API_PID" 2>/dev/null || true
  wait "$API_PID" 2>/dev/null || true
  kill "$KEV_PID" 2>/dev/null || true
  wait "$KEV_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ready=false
for _ in $(seq 1 90); do
  if curl --noproxy '*' -fsS http://127.0.0.1:8000/health | venv/bin/python -c 'import json,sys; h=json.load(sys.stdin); sys.exit(0 if h.get("model_loaded") else 1)' >/dev/null 2>&1; then
    ready=true
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    echo "Verity API exited during startup. Check verity_server.log." >&2
    exit 1
  fi
  sleep 2
done
if [ "$ready" != true ]; then
  echo "Verity API did not become ready. Check verity_server.log." >&2
  exit 1
fi

echo "Verity is loaded. Open the tunnel URL to start testing."
echo "Keep this process running; stop it with Ctrl-C to close the public tunnel."
cloudflared tunnel --url http://127.0.0.1:8001
