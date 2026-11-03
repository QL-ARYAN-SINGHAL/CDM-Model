#!/bin/bash
# Run KnowLine-4B and Jeffan-4B one at a time on eval_set.json (120 labelled prompts).
cd "$(dirname "$0")"
export EVAL_SET="$PWD/eval_set.json"
PORT=8031
run(){ # label mode gguf
  /opt/homebrew/bin/llama-server -m "$3" --host 127.0.0.1 --port $PORT -c 8192 -np 1 -b 2048 -ub 2048 > "eval_big_$1.server.log" 2>&1 &
  pid=$!
  for _ in $(seq 1 90); do curl --noproxy '*' -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 2; done
  echo "=== $1"; venv/bin/python eval_alt_models.py "$2" "$1" $PORT
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
}
run big-knowline knowline models/knowline/KnowLine-4B-Gen5.5.Q5_K_M.gguf
run big-jeffan jeffan-gguf models/candidates/jeffan-4b/Jeffan-4B-2010.Q5_K_M.gguf
echo ALL DONE
