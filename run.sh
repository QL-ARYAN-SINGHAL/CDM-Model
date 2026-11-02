#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd "$(dirname "$0")" && pwd)"
export VERITY_MODEL_PATH="${VERITY_MODEL_PATH:-$ROOT/models/verity/best}"
export VERITY_MODEL_NAME="${VERITY_MODEL_NAME:-Verity V4}"
if [ ! -f "$VERITY_MODEL_PATH/model.safetensors" ]; then
  echo "Missing checkpoint: $VERITY_MODEL_PATH/model.safetensors" >&2
  exit 1
fi
if [ ! -x venv/bin/python ]; then
  python3 -m venv venv
fi
if ! venv/bin/python -c 'import gliner2, fastapi, uvicorn, torch' >/dev/null 2>&1; then
  venv/bin/python -m pip install -r requirements.txt
fi
./ensure_kev.sh >> kev_server.log 2>&1 &
KEV_PID=$!
cleanup() {
  kill "$KEV_PID" 2>/dev/null || true
  wait "$KEV_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
echo "Verity: http://127.0.0.1:8000/"
venv/bin/python api_server.py
