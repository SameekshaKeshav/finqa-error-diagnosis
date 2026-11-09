#!/usr/bin/env bash
# Download the public FinQA data (train/dev/test json) into data/raw/.
# Source: https://github.com/czyssrs/FinQA  (Chen et al., EMNLP 2021)
# Only the three dataset files are fetched; no upstream code is used.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
for split in train dev test; do
  if [ ! -s "data/raw/${split}.json" ]; then
    echo "downloading ${split}.json"
    curl -fsSL -o "data/raw/${split}.json" \
      "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/${split}.json"
  else
    echo "${split}.json already present"
  fi
done
ls -la data/raw
