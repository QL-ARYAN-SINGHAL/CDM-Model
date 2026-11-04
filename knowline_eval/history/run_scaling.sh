#!/bin/bash
cd "$(dirname "$0")/.."
R=knowline_eval/results/scaling.jsonl; rm -f $R
for M in 2b 4b; do
  ./switch_knowline.sh $M >/dev/null
  PID=$(pgrep llama-server | head -1)
  echo "{\"model\":\"knowline$M\",\"r\":$(venv/bin/python knowline_eval/scaling.py --kind knowline$M --pid $PID)}" >> $R
done
pkill llama-server; sleep 3
nohup llama-server -m models/aurelis/aurelis-2b-q8_0.gguf --host 127.0.0.1 --port 8011 -c 8192 -np 1 > aurelis_server.log 2>&1 &
for i in $(seq 60); do curl -s localhost:8011/health | grep -q ok && break; sleep 2; done
PID=$(pgrep llama-server | head -1)
echo "{\"model\":\"aurelis2b\",\"r\":$(venv/bin/python knowline_eval/scaling.py --kind aurelis --pid $PID)}" >> $R
pkill llama-server
echo SCALING DONE
