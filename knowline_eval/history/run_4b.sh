#!/bin/bash
cd "$(dirname "$0")/.."
while pgrep -f "run_bench.py --model 2b" >/dev/null; do sleep 5; done
./switch_knowline.sh 4b && ps -o args= -p $(pgrep llama-server) && venv/bin/python knowline_eval/run_bench.py --model 4b --schemas A,B,C,D,E --out knowline_eval/results/4b.jsonl
echo ALL DONE
