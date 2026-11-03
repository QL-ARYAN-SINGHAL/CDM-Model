#!/bin/bash
# usage: drive_config.sh LABEL "<llama-server flags>" CACHE WARM [PILOT] [EXTRA_RUN_ARGS]
# Starts ONE isolated KnowLine server on 8038, runs run_config.py, stops only the PID it started (verified by its port).
cd "$(dirname "$0")/.." || exit 1
LABEL=$1; FLAGS=$2; CACHE=$3; WARM=$4; PILOT=${5:-0}
M="/Users/apple/Quantization Model/CDM-Model/models/knowline/KnowLine-4B-Gen5.5.Q5_K_M.gguf"
mkdir -p knowline_opt/logs knowline_opt/results
if lsof -iTCP:8038 -sTCP:LISTEN >/dev/null 2>&1; then echo "port 8038 busy, abort"; exit 2; fi
if ps aux | grep -E 'llama-server' | grep -v grep | grep -v 'port 8009' | grep -q .; then echo "another experimental llama-server is running, abort"; exit 3; fi
/opt/homebrew/bin/llama-server -m "$M" --host 127.0.0.1 --port 8038 -c 8192 -np 1 --slot-save-path knowline_opt/slotsave $FLAGS > "knowline_opt/logs/server_$LABEL.log" 2>&1 &
SP=$!
echo "started server pid $SP"
for _ in $(seq 1 120); do curl --noproxy '*' -fsS http://127.0.0.1:8038/health >/dev/null 2>&1 && break; sleep 2; done
curl --noproxy '*' -fsS http://127.0.0.1:8038/health >/dev/null 2>&1 || { echo "server did not become healthy"; }
venv/bin/python knowline_opt/run_config.py http://127.0.0.1:8038 "$LABEL" $SP --cache $CACHE --warm $WARM --pilot $PILOT 2>&1 | grep -v -i warn | tail -20
if ps -o command= -p $SP | grep -q -- "--port 8038"; then kill $SP; wait $SP 2>/dev/null; echo "stopped my process $SP"; fi
echo "DRIVE DONE $LABEL"
