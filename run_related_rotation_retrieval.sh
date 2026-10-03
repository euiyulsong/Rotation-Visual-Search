#!/usr/bin/env bash
set -euo pipefail

mkdir -p "$HOME/ckpt" "$HOME/results"

python3 "$HOME/rotation_related_retrieval.py" \
  --category Apparel \
  --rot-train-n 8000 \
  --rot-val-n 1000 \
  --gallery-n 10000 \
  --query-n 1000 \
  --rot-epochs 3 \
  --rot-batch 128 \
  --eval-batch 64 \
  --angles 0 90 180 270 \
  --ks 5 10 20 \
  --report-k 10 \
  --rot-ckpt "$HOME/ckpt/related_rotation_resnet18.pt" \
  --out "$HOME/results/related_rotation_retrieval.csv"
