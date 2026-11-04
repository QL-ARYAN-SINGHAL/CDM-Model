#!/bin/bash
# Full benchmark: one model at a time (8 GB Mac). Results -> knowline_eval/results/*.jsonl
cd "$(dirname "$0")/.."
mkdir -p knowline_eval/results
./switch_knowline.sh 2b && venv/bin/python knowline_eval/run_bench.py --model 2b --schemas A,B,C,D,E --out knowline_eval/results/2b.jsonl
./switch_knowline.sh 4b && venv/bin/python knowline_eval/run_bench.py --model 4b --schemas A,B,C,D,E --out knowline_eval/results/4b.jsonl
venv/bin/python knowline_eval/run_bench.py --model 4b --schemas A --read-twice --out knowline_eval/results/4b_rt.jsonl
echo ALL DONE
