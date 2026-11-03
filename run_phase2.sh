#!/bin/bash
# Phase 2: five intents on one big prompt (eval_big.json), whole-prompt vs line-by-line scoring.
cd "$(dirname "$0")"
b(){ echo "##### big $*"; venv/bin/python benchmark_knowline.py big "$@" 2>&1 | grep -v -i warn | grep -E "intent-level|Error|Traceback"; }
b q5-once-compact whole - paragraph
b q5-once-compact-cache whole - paragraph
b q5-twice-compact whole - paragraph
b q5-once-compact whole - multiline
b q5-once-compact line 8 multiline
b q5-twice line 8 multiline
echo PHASE2 DONE
