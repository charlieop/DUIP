#!/usr/bin/env bash
# End-to-end DUIP pipeline on Amazon Reviews 2023 (Games) with Qwen3.5-2B.

set -euo pipefail

CONFIG="${1:-configs/games.yaml}"

echo "[1/3] Preparing data ..."
python -m scripts.prepare_data --config "$CONFIG"

echo "[2/3] Training ..."
python -m scripts.run_training --config "$CONFIG"

echo "[3/3] Evaluating ..."
python -m scripts.run_eval --config "$CONFIG" \
    --checkpoint checkpoints/games/best.pt

echo "Done. See results/games.json"
