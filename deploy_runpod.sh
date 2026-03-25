#!/usr/bin/env bash
set -euo pipefail

DATASET_SIZE="${1:-standard}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"

echo "=========================================="
echo "VisualPRM RunPod Deploy"
echo "=========================================="
echo "Dataset preset: $DATASET_SIZE"
echo ""

cd "$WORKSPACE_DIR"

if [ ! -d "visualprm" ]; then
  git clone https://github.com/Youngkwon-Lee/visualprm-medical-prm.git visualprm
fi

cd visualprm

echo "[1/3] CPU setup"
bash setup_runpod.sh

echo "[2/3] Verify environment"
bash verify_setup.sh

echo "[3/3] Reminder"
echo "CPU phase is complete."
echo "Attach a GPU pod next, then run:"
echo "  bash train_runpod.sh $DATASET_SIZE"
