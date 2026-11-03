#!/bin/bash
# Convert KnowLine 2B and 0.8B (HF weights) to Q5_K_M GGUF, then run the 500-prompt benchmark on each.
cd "$(dirname "$0")"
for pair in "08b:0.8B-Gen1:k08b-once" "2b:2B-Gen1:k2b-once"; do
  IFS=: read short name cfg <<< "$pair"
  src=models/candidates/hf-$short; out=models/candidates/knowline-$short; mkdir -p $out
  while [ ! -f $src/model.safetensors ]; do sleep 5; done
  echo "##### convert $name"
  venv/bin/python tools/llama.cpp-src/convert_hf_to_gguf.py $src --outfile $out/KnowLine-$name.f16.gguf --outtype f16 2>&1 | tail -3
  /opt/homebrew/bin/llama-quantize $out/KnowLine-$name.f16.gguf $out/KnowLine-$name.Q5_K_M.gguf Q5_K_M 2>&1 | tail -2
  rm -f $out/KnowLine-$name.f16.gguf
  ls -la $out | awk '{print $5, $9}' | grep gguf
  echo "##### bench $cfg"; venv/bin/python benchmark_knowline.py single $cfg 2>&1 | grep -v -i warn | grep -E "acc@|Error|Traceback"
done
echo SMALL DONE
