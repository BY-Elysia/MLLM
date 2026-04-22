#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASET_ROOT="${1:-/home/by/workspace/ureader_tools/dataset_existing_local}"
TRAIN_JSONL="${DATASET_ROOT}/annotations/train.jsonl"
DEMO_DIR="${REPO_ROOT}/outputs/clip_small_demo"
SMALL_JSONL="${DEMO_DIR}/train_small.jsonl"

python3 "${SCRIPT_DIR}/make_subset.py" \
  --input "${TRAIN_JSONL}" \
  --output "${SMALL_JSONL}" \
  --limit 64 \
  --seed 42

python3 -m clip.main \
  --train-annotations "${SMALL_JSONL}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${DEMO_DIR}/run" \
  --model-name "openai/clip-vit-base-patch32" \
  --text-mode assistant \
  --epochs 1 \
  --batch-size 8 \
  --eval-batch-size 8 \
  --val-ratio 0.2 \
  --num-workers 0 \
  --log-interval 1 \
  --save-every-epoch
