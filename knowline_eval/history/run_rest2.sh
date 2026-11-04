#!/bin/bash
# Order: (Aurelis running) -> convert joey-1 to GGUF -> Qwen3.5-0.8B-decision -> joey-1 Q8_0 -> joey-1 Q4_K_M -> KnowLine 4B. One model at a time.
cd "$(dirname "$0")/.."
LOG=knowline_eval/results
while pgrep -f "run_sysone.py --url http://127.0.0.1:8011" >/dev/null; do sleep 5; done
pkill llama-server; sleep 2
# --- convert joey-1 (needs download finished)
until grep -q DONE /tmp/dljoey.log; do sleep 5; done
mkdir -p models/joey1
venv/bin/python /private/tmp/claude-501/-Users-app--claude-spares-spare-G0z7Kj/fee269a0-d9fe-444d-b2cc-96cc3853d90f/scratchpad/llcpp/convert_hf_to_gguf.py models/joey1-hf --outfile models/joey1/joey1-f16.gguf --outtype f16 > $LOG/joey_convert.log 2>&1 && rm -f models/joey1-hf/model.safetensors
llama-quantize models/joey1/joey1-f16.gguf models/joey1/joey1-Q8_0.gguf Q8_0 >> $LOG/joey_convert.log 2>&1
llama-quantize models/joey1/joey1-f16.gguf models/joey1/joey1-Q4_K_M.gguf Q4_K_M >> $LOG/joey_convert.log 2>&1
ls -l models/joey1/ >> $LOG/joey_convert.log; rm -f models/joey1/joey1-f16.gguf
# --- Qwen3.5-0.8B-decision (MLX)
until grep -q DONE /tmp/dlq08.log; do sleep 5; done
nohup venv/bin/optiq serve --model "$PWD/models/qwen08-decision" --host 127.0.0.1 --port 8012 > optiq_08b.log 2>&1 &
for i in $(seq 120); do curl -s localhost:8012/health >/dev/null 2>&1 && break; sleep 2; done
(cd knowline_eval && ../venv/bin/python run_sysone.py --url http://127.0.0.1:8012 --tag qwen08 --model "$PWD/../models/qwen08-decision" --schemas A,B,C,D,E --out results/qwen08.jsonl > results/qwen08.log 2>&1)
pkill -f "optiq serve"; sleep 3
# --- joey-1 Q8_0 and Q4_K_M via OneJev's llama.cpp engine (qev serve --gguf)
for Q in Q8_0 Q4_K_M; do
  nohup venv/bin/qev serve --gguf "$PWD/models/joey1/joey1-$Q.gguf" --tokenizer "$PWD/models/joey1-hf" --llama-server /opt/homebrew/bin/llama-server --parallel 1 --host 127.0.0.1 --port 8013 > qev_joey_$Q.log 2>&1 &
  for i in $(seq 150); do curl -s localhost:8013/health >/dev/null 2>&1 && break; sleep 2; done
  (cd knowline_eval && ../venv/bin/python run_sysone.py --url http://127.0.0.1:8013 --tag joey_$Q --schemas A,B,C,D,E --out results/joey_$Q.jsonl > results/joey_$Q.log 2>&1)
  pkill -f "qev serve"; pkill llama-server; sleep 3
done
# --- KnowLine 4B at -ub 128
./switch_knowline.sh 4b && ps -o args= -p $(pgrep llama-server) && venv/bin/python knowline_eval/run_bench.py --model 4b --schemas A,B,C,D,E --out $LOG/4b.jsonl
echo REST DONE
