#!/bin/bash
# Read-only experiments for the latency investigation, one after another, against the isolated server on :8034.
# This script stops nothing.
cd "$(dirname "$0")"
while [ ! -f eval_results/latency_investigation/baseline_summary_warm.json ]; do sleep 5; done
U=http://127.0.0.1:8034
for step in "tokens" "cache $U" "nprobs $U" "twice $U" "concurrency $U $U" "suite-twice $U"; do
  echo "##### $step"; venv/bin/python experiments_knowline.py $step 2>&1 | grep -v -i warn | tail -40
done
echo EXPERIMENTS DONE
