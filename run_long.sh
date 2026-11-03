#!/bin/bash
# Long prompts (eval_long.json): StartLux 4B Q4, StartLux 2B Q8, then KnowLine 4B Q5 read once and read twice. One at a time.
cd "$(dirname "$0")"
while ! grep -q "STARTLUX DONE" startlux_run.log; do sleep 10; done
export EVAL_SET="$PWD/eval_long.json"
C=models/candidates
for entry in "startlux-4b:StartLux-Decision-v1.1-4B-Instant-Q4_K_M.gguf:startlux-4b-q4" "startlux-2b:StartLux-Decision-v1.1-2B-Instant-Q8_0.gguf:startlux-2b-q8"; do
  IFS=: read dir gguf label <<< "$entry"
  echo "##### long $label"
  /opt/homebrew/bin/llama-server -m "$C/$dir/$gguf" -ngl 99 -c 8192 --parallel 1 --ctx-checkpoints 0 -b 4096 -ub 4096 --host 127.0.0.1 --port 8081 > "eval_long_$label.llama.log" 2>&1 &
  LP=$!
  for _ in $(seq 1 90); do curl --noproxy '*' -fsS http://127.0.0.1:8081/health >/dev/null 2>&1 && break; sleep 2; done
  STARTLUX_LLAMACPP_BIDIR=server venv/bin/startlux-decide serve --model "$C/$dir" --backend llamacpp --llamacpp-url http://127.0.0.1:8081 --port 8095 > "eval_long_$label.decide.log" 2>&1 &
  DP=$!
  for _ in $(seq 1 60); do curl --noproxy '*' -fsS http://127.0.0.1:8095/health >/dev/null 2>&1 && break; sleep 2; done
  venv/bin/python bench_startlux.py "$label" 8095 2>&1 | grep -E "RESULT|Error|Traceback"
  kill $DP $LP 2>/dev/null; wait $DP $LP 2>/dev/null
done
for cfg in q5-once-compact q5-twice-compact; do
  echo "##### long knowline $cfg"; venv/bin/python benchmark_knowline.py single $cfg "$EVAL_SET" 2>&1 | grep -E "acc@|Error|Traceback"
done
echo LONG DONE
