#!/bin/bash
# The 100-prompt, 3-intent suite on KnowLine 4B Q5, StartLux 4B Q4_K_M and StartLux 2B Q8_0 (llama.cpp), one at a time.
cd "$(dirname "$0")"
C=models/candidates
echo "##### knowline-4b-q5"
/opt/homebrew/bin/llama-server -m models/knowline/KnowLine-4B-Gen5.5.Q5_K_M.gguf --host 127.0.0.1 --port 8033 -c 8192 -np 1 -b 2048 -ub 2048 > eval_suite_knowline.llama.log 2>&1 &
LP=$!; for _ in $(seq 1 90); do curl --noproxy '*' -fsS http://127.0.0.1:8033/health >/dev/null 2>&1 && break; sleep 2; done
venv/bin/python bench_suite100.py knowline-4b-q5 knowline 8033 2>&1 | grep -E "RESULT|Error|Traceback"
kill $LP; wait $LP 2>/dev/null; sleep 3
for entry in "startlux-4b:StartLux-Decision-v1.1-4B-Instant-Q4_K_M.gguf:startlux-4b-q4" "startlux-2b:StartLux-Decision-v1.1-2B-Instant-Q8_0.gguf:startlux-2b-q8"; do
  IFS=: read dir gguf label <<< "$entry"
  echo "##### $label"
  /opt/homebrew/bin/llama-server -m "$C/$dir/$gguf" -ngl 99 -c 8192 --parallel 1 --ctx-checkpoints 0 -b 4096 -ub 4096 --host 127.0.0.1 --port 8081 > "eval_suite_$label.llama.log" 2>&1 &
  LP=$!; for _ in $(seq 1 90); do curl --noproxy '*' -fsS http://127.0.0.1:8081/health >/dev/null 2>&1 && break; sleep 2; done
  STARTLUX_LLAMACPP_BIDIR=server venv/bin/startlux-decide serve --model "$C/$dir" --backend llamacpp --llamacpp-url http://127.0.0.1:8081 --port 8095 > "eval_suite_$label.decide.log" 2>&1 &
  DP=$!; for _ in $(seq 1 60); do curl --noproxy '*' -fsS http://127.0.0.1:8095/health >/dev/null 2>&1 && break; sleep 2; done
  venv/bin/python bench_suite100.py "$label" startlux 8095 2>&1 | grep -E "RESULT|Error|Traceback"
  kill $DP $LP 2>/dev/null; wait $DP $LP 2>/dev/null; sleep 3
done
echo SUITE DONE
