#!/bin/bash
cd "$(dirname "$0")/.."
pkill -f "qev serve"; pkill llama-server; sleep 3
for Q in Q8_0 Q4_K_M; do
  nohup venv/bin/qev serve --gguf "$PWD/models/joey1/joey1-$Q.gguf" --tokenizer "$PWD/models/joey1-hf" --llama-server /opt/homebrew/bin/llama-server --parallel 1 --host 127.0.0.1 --port 8013 > qev_joey_$Q.log 2>&1 &
  for i in $(seq 150); do curl -s localhost:8013/health >/dev/null 2>&1 && break; sleep 2; done
  rm -f knowline_eval/results/joey_$Q.jsonl
  (cd knowline_eval && ../venv/bin/python run_sysone.py --url http://127.0.0.1:8013 --tag joey_$Q --model jev-latest --schemas A,B,C,D,E --out results/joey_$Q.jsonl > results/joey_$Q.log 2>&1)
  pkill -f "qev serve"; pkill llama-server; sleep 3
done
echo JOEY DONE
