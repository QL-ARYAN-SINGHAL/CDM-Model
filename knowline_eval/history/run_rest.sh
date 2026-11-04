#!/bin/bash
# After Aurelis finishes: Qwen3.5-0.8B-decision (optiq serve, MLX), then KnowLine 4B (-ub 128). One model at a time.
cd "$(dirname "$0")/.."
while pgrep -f "run_sysone.py --url http://127.0.0.1:8011" >/dev/null; do sleep 5; done
pkill llama-server; sleep 2
until grep -q DONE /tmp/dlq08.log; do sleep 5; done
nohup venv/bin/optiq serve --model "$PWD/models/qwen08-decision" --host 127.0.0.1 --port 8012 > optiq_08b.log 2>&1 &
for i in $(seq 90); do curl -s localhost:8012/health >/dev/null 2>&1 && break; sleep 2; done
cd knowline_eval
../venv/bin/python run_sysone.py --url http://127.0.0.1:8012 --tag qwen08 --model "$PWD/../models/qwen08-decision" --schemas A,B,C,D,E --out results/qwen08.jsonl > results/qwen08.log 2>&1
cd ..; pkill -f "optiq serve"; sleep 3
./switch_knowline.sh 4b && ps -o args= -p $(pgrep llama-server) && venv/bin/python knowline_eval/run_bench.py --model 4b --schemas A,B,C,D,E --out knowline_eval/results/4b.jsonl
echo REST DONE
