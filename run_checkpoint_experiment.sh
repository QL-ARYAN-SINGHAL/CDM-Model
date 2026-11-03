#!/bin/bash
# Starts its own isolated KnowLine server per variant (port 8037), tests, and stops only that process.
cd "$(dirname "$0")"
while ! grep -q "CANDIDATES2 DONE" candidates2_run.log; do sleep 10; done
M=models/knowline/KnowLine-4B-Gen5.5.Q5_K_M.gguf
for variant in "ub64:-b 512 -ub 64" "ub128:-b 512 -ub 128" "ub256:-b 1024 -ub 256"; do
  label=${variant%%:*}; flags=${variant#*:}
  echo "##### $label ($flags --checkpoint-min-step 0)"
  /opt/homebrew/bin/llama-server -m $M --host 127.0.0.1 --port 8037 -c 8192 -np 1 $flags --checkpoint-min-step 0 --ctx-checkpoints 32 > "latency_ckpt_$label.log" 2>&1 &
  SP=$!; for _ in $(seq 1 90); do curl --noproxy '*' -fsS http://127.0.0.1:8037/health >/dev/null 2>&1 && break; sleep 2; done
  venv/bin/python experiment_checkpoint.py http://127.0.0.1:8037 $label 2>&1 | grep -v -i warn | tail -3
  if ps -o command= -p $SP | grep -q -- "port 8037"; then kill $SP; wait $SP 2>/dev/null; echo "stopped my process $SP"; fi
  sleep 3
done
echo CHECKPOINT DONE
