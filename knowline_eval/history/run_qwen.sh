#!/bin/bash
cd "$(dirname "$0")/.."
pkill llama-server; sleep 1
nohup venv/bin/optiq serve --model "$PWD/models/qwen08-decision" --host 127.0.0.1 --port 8012 > optiq_08b.log 2>&1 &
for i in $(seq 120); do curl -s localhost:8012/health >/dev/null 2>&1 && break; sleep 2; done
date +%T > knowline_eval/results/qwen08.start
(cd knowline_eval && ../venv/bin/python run_sysone.py --url http://127.0.0.1:8012 --tag qwen08 --model "$PWD/../models/qwen08-decision" --schemas A,B,C,D,E --out results/qwen08.jsonl > results/qwen08.log 2>&1)
date +%T > knowline_eval/results/qwen08.end
pkill -f "optiq serve"
echo QWEN DONE
