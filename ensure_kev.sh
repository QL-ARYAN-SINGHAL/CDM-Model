#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
KEV_MODEL_PATH="${KEV_MODEL_PATH:-$ROOT/models/kev/Kev-0.8B-Q8_0.gguf}"
KEV_PORT="${KEV_PORT:-8009}"
LLAMA_SERVER="${LLAMA_SERVER:-/opt/homebrew/bin/llama-server}"
if [ ! -f "$KEV_MODEL_PATH" ]; then
  echo "Missing Kev checkpoint: $KEV_MODEL_PATH" >&2
  exit 1
fi
if [ ! -x "$LLAMA_SERVER" ]; then
  echo "Missing llama-server: $LLAMA_SERVER" >&2
  exit 1
fi
if curl --noproxy '*' -fsS "http://127.0.0.1:${KEV_PORT}/health" >/dev/null 2>&1; then
  echo "Kev server already running on :${KEV_PORT}"
  exit 0
fi
exec "$LLAMA_SERVER" -m "$KEV_MODEL_PATH" --host 127.0.0.1 --port "$KEV_PORT" -c 8192 -np 1 -b 2048 -ub 2048
