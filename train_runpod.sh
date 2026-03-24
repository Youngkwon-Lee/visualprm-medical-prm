#!/usr/bin/env bash
set -euo pipefail

# VisualPRM Training Script (GPU A100 전용)
# 사전 조건: setup_runpod.sh가 완료되었어야 함
# 이 스크립트는 GPU 비용이 발생함

DATASET_SIZE="${1:-standard}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"

echo "=========================================="
echo "VisualPRM Training (GPU A100-80GB)"
echo "=========================================="
echo "Dataset: $DATASET_SIZE"
echo "Time: ~11 hours (standard) = $5.30 cost"
echo ""

cd "$WORKSPACE_DIR/visualprm"

# Check GPU availability
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
    echo "❌ No GPU found! Make sure you're on A100 instance"
    exit 1
}
echo "✅ GPU ready"
echo ""

# Load environment
export $(cat .env.production | grep -v '^#' | xargs)

# 1. Start Qwen server in background
echo "[1/3] Starting Qwen OpenAI-compatible server..."
python runpod_qwen_openai_server.py > "$WORKSPACE_DIR/logs/server.log" 2>&1 &
SERVER_PID=$!
echo "  PID: $SERVER_PID"

# Wait for server
echo "  Waiting for server to start..."
for i in {1..60}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "  ✅ Server ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "  ❌ Server startup timeout"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
    echo -n "."
    sleep 1
done
echo ""

# 2. Start training
echo "[2/3] Starting training..."
echo "  Dataset: $DATASET_SIZE"
echo "  Batch size: 16"
echo "  Epochs: 3"
echo "  Learning rate: 2e-5"
echo ""

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

TRAIN_EXIT_CODE=$?

# 3. Cleanup
echo ""
echo "[3/3] Cleaning up..."
kill $SERVER_PID 2>/dev/null || true
echo "  ✅ Server stopped"

# Results
echo ""
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully! ✅"
    echo ""
    echo "Results:"
    echo "  Model: $WORKSPACE_DIR/models/final/"
    echo "  Logs: $WORKSPACE_DIR/logs/"
    echo ""
    echo "Next: Download model and run evaluation"
else
    echo "Training failed ❌"
    echo "Check logs: tail -f $WORKSPACE_DIR/logs/training.log"
    exit 1
fi
echo "=========================================="
