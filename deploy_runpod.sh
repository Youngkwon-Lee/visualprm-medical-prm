#!/usr/bin/env bash
set -euo pipefail

# VisualPRM RunPod A100 Deployment Script
# Usage: bash deploy_runpod.sh [dataset_size] [gpu_count]
# Examples:
#   bash deploy_runpod.sh standard
#   bash deploy_runpod.sh large
#   bash deploy_runpod.sh mvp

DATASET_SIZE="${1:-standard}"  # mvp, standard, large
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "VisualPRM RunPod A100 Deployment"
echo "=========================================="
echo "Dataset: $DATASET_SIZE"
echo "Workspace: $WORKSPACE_DIR"
echo ""

# 1. Setup workspace
echo "[1/6] Creating workspace directories..."
mkdir -p "$WORKSPACE_DIR"/{data,models,.cache,logs}
cd "$WORKSPACE_DIR"

# 2. Clone/copy repository
echo "[2/6] Setting up repository..."
if [ ! -d "visualprm" ]; then
    if [ -d "$REPO_DIR" ]; then
        cp -r "$REPO_DIR" visualprm
    else
        echo "Error: Repository not found at $REPO_DIR"
        exit 1
    fi
fi
cd visualprm

# 3. Install dependencies
echo "[3/6] Installing dependencies..."
python -m pip install -q --upgrade pip setuptools wheel
pip install -q -r requirements.txt
pip install -q huggingface-hub

# 4. Download model (if not cached)
echo "[4/6] Downloading Qwen3-VL-30B model..."
python << 'EOF'
import os
from huggingface_hub import snapshot_download
from pathlib import Path

cache_dir = Path("/workspace/.cache/huggingface")
cache_dir.mkdir(parents=True, exist_ok=True)

model_id = "Qwen/Qwen3-VL-30B-Instruct"
print(f"Downloading {model_id}...")
try:
    snapshot_download(
        model_id,
        cache_dir=str(cache_dir),
        local_dir_use_symlinks=False,
        revision="main",
    )
    print(f"Model downloaded to {cache_dir}")
except Exception as e:
    print(f"Download failed (may already be cached): {e}")
EOF

# 5. Start Qwen server in background
echo "[5/6] Starting Qwen OpenAI-compatible server..."
export QWEN_MODEL_ID="Qwen/Qwen3-VL-30B-Instruct"
export QWEN_SERVER_HOST="0.0.0.0"
export QWEN_SERVER_PORT="8000"
export QWEN_MAX_NEW_TOKENS="512"
export QWEN_LOAD_IN_4BIT="0"

python runpod_qwen_openai_server.py > "$WORKSPACE_DIR/logs/server.log" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to start
echo "Waiting for server to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    echo "  Waiting... ($i/30)"
    sleep 2
done

# 6. Start training
echo "[6/6] Starting training..."
export DATASET_NAME="$DATASET_SIZE"
export CUDA_VISIBLE_DEVICES="0"
export MIXED_PRECISION="fp16"
export GRADIENT_CHECKPOINTING="1"
export TRAINING_BATCH_SIZE="16"
export TRAINING_EPOCHS="3"
export TRAINING_LEARNING_RATE="2e-5"

python train_visual_prm.py \
    --model_name "Qwen/Qwen3-VL-30B-Instruct" \
    --dataset "$DATASET_SIZE" \
    --batch_size 16 \
    --epochs 3 \
    --learning_rate 2e-5 \
    --use_lora true \
    --use_mixed_precision true \
    --save_interval 500 \
    2>&1 | tee "$WORKSPACE_DIR/logs/training.log"

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
echo "Model saved to: $WORKSPACE_DIR/models/"
echo "Logs: $WORKSPACE_DIR/logs/"
echo ""

# Cleanup
kill $SERVER_PID 2>/dev/null || true

echo "Done!"
