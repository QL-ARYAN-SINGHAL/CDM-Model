#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
KNOWLINE_MODEL_PATH="${KNOWLINE_MODEL_PATH:-$ROOT/models/knowline/KnowLine-2B-Gen1.Q8_0.gguf}"
KNOWLINE_PORT="${KNOWLINE_PORT:-8009}"
LLAMA_SERVER="${LLAMA_SERVER:-/opt/homebrew/bin/llama-server}"
if [ ! -f "$KNOWLINE_MODEL_PATH" ]; then
  echo "Missing KnowLine-2B checkpoint: $KNOWLINE_MODEL_PATH" >&2
  exit 1
fi
if [ ! -x "$LLAMA_SERVER" ]; then
  echo "Missing llama-server: $LLAMA_SERVER" >&2
  exit 1
fi
if curl --noproxy '*' -fsS "http://127.0.0.1:${KNOWLINE_PORT}/health" >/dev/null 2>&1; then
  echo "KnowLine server already running on :${KNOWLINE_PORT}"
  exit 0
fi
exec "$LLAMA_SERVER" -m "$KNOWLINE_MODEL_PATH" --host 127.0.0.1 --port "$KNOWLINE_PORT" -c 8192 -np 1 -b 2048 -ub 64 --checkpoint-min-step 0 --ctx-checkpoints 32
