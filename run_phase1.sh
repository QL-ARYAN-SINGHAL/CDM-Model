#!/bin/bash
# Phase 1: serving options on the 500 single prompts, one config at a time (each starts/stops its own llama-server).
cd "$(dirname "$0")"
for c in q5-twice q5-once q5-twice-compact q5-once-compact q4-once-compact q4-twice-compact; do
  echo "##### $c"; venv/bin/python benchmark_knowline.py single $c 2>&1 | grep -v -i warn
done
echo PHASE1 DONE
