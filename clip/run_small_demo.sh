#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1

DATASET_ROOT="${1:-/home/by/workspace/ureader_tools/dataset_existing_local}"
TRAIN_JSONL="${DATASET_ROOT}/annotations/train.jsonl"
DEMO_DIR="${REPO_ROOT}/outputs/clip_small_demo"
SMALL_JSONL="${DEMO_DIR}/train_small.jsonl"
CONFIG_PATH="${2:-${REPO_ROOT}/clip/configs/small_demo.json}"

python3 "${SCRIPT_DIR}/make_subset.py" \
  --input "${TRAIN_JSONL}" \
  --output "${SMALL_JSONL}" \
  --limit 64 \
  --seed 42

python3 -u -m clip.main --config "${CONFIG_PATH}"
