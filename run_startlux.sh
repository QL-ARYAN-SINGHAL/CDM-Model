#!/bin/bash
# For each StartLux GGUF: llama-server (non-causal prompts, one slot) + startlux-decide serve, then the 500-prompt benchmark.
cd "$(dirname "$0")"
C=models/candidates
for entry in "startlux-4b:StartLux-Decision-v1.1-4B-Instant-Q4_K_M.gguf:startlux-4b-q4" "startlux-2b:StartLux-Decision-v1.1-2B-Instant-Q8_0.gguf:startlux-2b-q8"; do
  IFS=: read dir gguf label <<< "$entry"
  while [ ! -f "$C/$dir/$gguf" ]; do sleep 5; done
  echo "##### $label"
  /opt/homebrew/bin/llama-server -m "$C/$dir/$gguf" -ngl 99 -c 8192 --parallel 1 --ctx-checkpoints 0 -b 4096 -ub 4096 --host 127.0.0.1 --port 8081 > "eval_$label.llama.log" 2>&1 &
  LP=$!
  for _ in $(seq 1 90); do curl --noproxy '*' -fsS http://127.0.0.1:8081/health >/dev/null 2>&1 && break; sleep 2; done
  STARTLUX_LLAMACPP_BIDIR=server venv/bin/startlux-decide serve --model "$C/$dir" --backend llamacpp --llamacpp-url http://127.0.0.1:8081 --port 8095 > "eval_$label.decide.log" 2>&1 &
  DP=$!
  for _ in $(seq 1 60); do curl --noproxy '*' -fsS http://127.0.0.1:8095/health >/dev/null 2>&1 && break; sleep 2; done
  curl --noproxy '*' -s http://127.0.0.1:8095/v1/systemone -H 'Content-Type: application/json' -d '{"state":"Book a flight to Delhi tomorrow.","questions":{"q":{"type":"noul","instructions":"Does this line request travel booking?","criteria":{"true":"Actual request to book a flight or hotel.","false":"General discussion of travel or no booking request."}}}}' | head -c 300; echo
  venv/bin/python bench_startlux.py "$label" 8095 2>&1 | grep -E "RESULT|Error|Traceback"
  kill $DP $LP 2>/dev/null; wait $DP $LP 2>/dev/null
done
echo STARTLUX DONE
