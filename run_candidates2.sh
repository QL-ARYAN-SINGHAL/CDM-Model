#!/bin/bash
# After the KnowLine experiments: StartLux 2B/4B (GGUF, llama.cpp) and StartLux MLX builds against current KnowLine.
# Starts only its own servers on free ports and stops only those (by recorded PID, after checking the command line).
cd "$(dirname "$0")"
C=models/candidates
stop_mine(){ # pid expected-substring
  if ps -o command= -p "$1" 2>/dev/null | grep -q -- "$2"; then kill "$1" 2>/dev/null; wait "$1" 2>/dev/null; echo "stopped my process $1 ($2)"; fi
}
bench(){ # label mode port
  for suite in eval_suite100.json eval_target250.json; do
    tag=$([ $suite = eval_target250.json ] && echo "-t250" || echo "")
    echo "--- $1$tag"; SUITE=$suite venv/bin/python bench_suite100.py "$1$tag" "$2" "$3" 2>&1 | grep -E "RESULT|Error|Traceback"
  done
}
while ! grep -q "EXPERIMENTS DONE" latency_experiments.log; do sleep 10; done
echo "##### knowline current (target size)"; SUITE=eval_target250.json venv/bin/python bench_suite100.py knowline-4b-q5-t250 knowline 8034 2>&1 | grep -E "RESULT|Error|Traceback"
stop_mine 29846 "port 8034"       # my isolated profiling server, idle now
venv/bin/pip install -q -e "tools/StartLux-Decision[mlx]" 2>&1 | tail -2
for entry in "startlux-2b:StartLux-Decision-v1.1-2B-Instant-Q8_0.gguf:startlux-2b-q8" "startlux-4b:StartLux-Decision-v1.1-4B-Instant-Q4_K_M.gguf:startlux-4b-q4"; do
  IFS=: read dir gguf label <<< "$entry"
  echo "##### $label (llama.cpp)"
  /opt/homebrew/bin/llama-server -m "$C/$dir/$gguf" -ngl 99 -c 8192 --parallel 1 --ctx-checkpoints 0 -b 4096 -ub 4096 --host 127.0.0.1 --port 8081 > "eval_c2_$label.llama.log" 2>&1 &
  LP=$!; for _ in $(seq 1 90); do curl --noproxy '*' -fsS http://127.0.0.1:8081/health >/dev/null 2>&1 && break; sleep 2; done
  STARTLUX_LLAMACPP_BIDIR=server venv/bin/startlux-decide serve --model "$C/$dir" --backend llamacpp --llamacpp-url http://127.0.0.1:8081 --port 8095 > "eval_c2_$label.decide.log" 2>&1 &
  DP=$!; for _ in $(seq 1 60); do curl --noproxy '*' -fsS http://127.0.0.1:8095/health >/dev/null 2>&1 && break; sleep 2; done
  if [ $label = startlux-4b-q4 ]; then
    SUITE=eval_target250.json venv/bin/python bench_suite100.py "$label-t250" startlux 8095 2>&1 | grep -E "RESULT|Error|Traceback"
  else bench "$label" startlux 8095; fi
  stop_mine $DP "port 8095"; stop_mine $LP "port 8081"; sleep 3
done
for entry in "mlx-4b-4bit:startlux-mlx-4b-4bit" "mlx-2b-8bit:startlux-mlx-2b-8bit"; do
  IFS=: read dir label <<< "$entry"
  while [ ! -f "$C/$dir/model.safetensors" ]; do sleep 10; done
  echo "##### $label (MLX)"
  venv/bin/startlux-decide serve --model "$C/$dir" --backend mlx --port 8096 > "eval_c2_$label.decide.log" 2>&1 &
  DP=$!; for _ in $(seq 1 120); do curl --noproxy '*' -fsS http://127.0.0.1:8096/health >/dev/null 2>&1 && break; sleep 2; done
  bench "$label" startlux 8096
  stop_mine $DP "port 8096"; sleep 3
done
echo CANDIDATES2 DONE
