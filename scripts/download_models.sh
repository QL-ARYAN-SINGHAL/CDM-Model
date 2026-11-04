#!/bin/bash
# Download the model files for this repo from the GitHub Release "models-v1" into ./models/ and verify checksums.
# usage: scripts/download_models.sh [knowline|aurelis|all]     (default: knowline)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="${REPO:-QL-ARYAN-SINGHAL/CDM-Model}"; TAG="${TAG:-models-v1}"; WHAT="${1:-knowline}"
BASE="https://github.com/$REPO/releases/download/$TAG"
mkdir -p models/knowline models/aurelis
get() { echo "-> $2"; curl -L --fail --retry 5 -C - -o "$2" "$BASE/$1"; }   # -C - resumes an interrupted download
curl -sL --fail -o /tmp/CDM_SHA256SUMS "$BASE/SHA256SUMS"
check() { want=$(grep " $2\$" /tmp/CDM_SHA256SUMS | cut -d' ' -f1); got=$(shasum -a 256 "$1/$2" | cut -d' ' -f1); [ "$want" = "$got" ] && echo "OK  $2" || { echo "BAD CHECKSUM $2"; exit 1; }; }
if [ "$WHAT" = knowline ] || [ "$WHAT" = all ]; then
  get KnowLine-2B-Gen1.Q8_0.gguf models/knowline/KnowLine-2B-Gen1.Q8_0.gguf
  check models/knowline KnowLine-2B-Gen1.Q8_0.gguf
fi
if [ "$WHAT" = aurelis ] || [ "$WHAT" = all ]; then       # 2.7 GB > GitHub's 2 GB asset limit, so it is stored in two parts
  get aurelis-2b-q8_0.gguf.part-aa models/aurelis/aurelis-2b-q8_0.gguf.part-aa
  get aurelis-2b-q8_0.gguf.part-ab models/aurelis/aurelis-2b-q8_0.gguf.part-ab
  check models/aurelis aurelis-2b-q8_0.gguf.part-aa; check models/aurelis aurelis-2b-q8_0.gguf.part-ab
  cat models/aurelis/aurelis-2b-q8_0.gguf.part-aa models/aurelis/aurelis-2b-q8_0.gguf.part-ab > models/aurelis/aurelis-2b-q8_0.gguf
  rm models/aurelis/aurelis-2b-q8_0.gguf.part-a?
  check models/aurelis aurelis-2b-q8_0.gguf
fi
echo "done"
