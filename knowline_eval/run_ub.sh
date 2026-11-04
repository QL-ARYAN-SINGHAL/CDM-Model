#!/bin/bash
# Same model, same suite, schema A, different llama-server batch settings. Port 8009 (the app's KnowLine port).
cd "$(dirname "$0")/.."
M=models/knowline/KnowLine-2B-Gen1.Q8_0.gguf
rm -f knowline_eval/results/ub_*.jsonl
run() {  # name, cache-flag, llama-server extra flags
  N=$1; C=$2; shift 2
  pkill llama-server; sleep 3
  nohup llama-server -m $M --host 127.0.0.1 --port 8009 -c 8192 -np 1 "$@" > knowline_eval/results/ub_$N.server.log 2>&1 &
  for i in $(seq 60); do curl -s localhost:8009/health | grep -q ok && break; sleep 2; done
  echo "== $N: $*"; ps -o args= -p $(pgrep llama-server) | cut -c1-200
  venv/bin/python knowline_eval/run_bench.py --model 2b --schemas A --tag ub_$N $C --out knowline_eval/results/ub_$N.jsonl
}
run base2048_nocache --no-cache -b 2048 -ub 2048
run ub128_nocache    --no-cache -b 2048 -ub 128
run ub128_cache      ""         -b 2048 -ub 128 --checkpoint-min-step 0 --ctx-checkpoints 32
run ub256_cache      ""         -b 2048 -ub 256 --checkpoint-min-step 0 --ctx-checkpoints 32
run ub64_cache       ""         -b 2048 -ub 64  --checkpoint-min-step 0 --ctx-checkpoints 32
pkill llama-server
echo UB DONE
