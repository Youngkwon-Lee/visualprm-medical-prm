#!/usr/bin/env bash
set -euo pipefail

DATASET_SIZE="${1:-standard}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
PROJECT_DIR="$WORKSPACE_DIR/visualprm"
VENV_DIR="${VENV_DIR:-/root/visualprm-venv}"

echo "=========================================="
echo "VisualPRM GPU Training"
echo "=========================================="
echo "Dataset preset: $DATASET_SIZE"
echo ""

cd "$PROJECT_DIR"

if [ -f "$VENV_DIR/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
else
  echo "Python virtual environment not found at $VENV_DIR"
  echo "Run: bash setup_runpod.sh"
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "No GPU found. Attach a GPU pod first."
  exit 1
fi

export $(grep -v '^#' .env.production | xargs)
export HF_HOME="${HF_HOME:-$WORKSPACE_DIR/.cache/huggingface}"
START_LOCAL_QWEN_SERVER="${START_LOCAL_QWEN_SERVER:-0}"
SERVER_PID=""

echo "[1/3] Optional local Qwen OpenAI-compatible server"
if [ "$START_LOCAL_QWEN_SERVER" = "1" ]; then
  python runpod_qwen_openai_server.py > "$WORKSPACE_DIR/logs/server.log" 2>&1 &
  SERVER_PID=$!

  SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-300}"
  for ((i=1; i<=SERVER_START_TIMEOUT; i++)); do
    if curl -s http://127.0.0.1:8000/health >/dev/null 2>&1; then
      echo "  Qwen server ready"
      break
    fi
    if [ "$i" -eq "$SERVER_START_TIMEOUT" ]; then
      echo "  Qwen server failed to start"
      kill "$SERVER_PID" 2>/dev/null || true
      exit 1
    fi
    sleep 1
  done
else
  echo "  Skipping local Qwen server; train_visual_prm.py loads the model directly."
fi

echo "[2/3] Select training data preset"
case "$DATASET_SIZE" in
  mvp)
    TRAIN_FILE="$WORKSPACE_DIR/data/train_mvp.jsonl"
    VAL_FILE="$WORKSPACE_DIR/data/val_mvp.jsonl"
    ;;
  standard)
    TRAIN_FILE="$WORKSPACE_DIR/data/train_standard.jsonl"
    VAL_FILE="$WORKSPACE_DIR/data/val_standard.jsonl"
    ;;
  large)
    TRAIN_FILE="$WORKSPACE_DIR/data/train_large.jsonl"
    VAL_FILE="$WORKSPACE_DIR/data/val_large.jsonl"
    ;;
  *)
    echo "Unknown dataset preset: $DATASET_SIZE"
    if [ -n "$SERVER_PID" ]; then
      kill "$SERVER_PID" 2>/dev/null || true
    fi
    exit 1
    ;;
esac

if [ ! -f "$TRAIN_FILE" ]; then
  echo "Training file not found: $TRAIN_FILE"
  echo "Prepare step-level JSONL files under /workspace/data first."
  if [ -n "$SERVER_PID" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
  fi
  exit 1
fi

echo "[3/3] Train step-level PRM"
python train_visual_prm.py \
  --model_name "${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}" \
  --train_file "$TRAIN_FILE" \
  --val_file "$VAL_FILE" \
  --batch_size "${TRAINING_BATCH_SIZE:-1}" \
  --grad_accum "${TRAINING_GRAD_ACCUM:-8}" \
  --epochs "${TRAINING_EPOCHS:-3}" \
  --learning_rate "${TRAINING_LEARNING_RATE:-2e-5}" \
  --save_interval "${TRAINING_SAVE_INTERVAL:-500}" \
  2>&1 | tee "$WORKSPACE_DIR/logs/training.log"

if [ -n "$SERVER_PID" ]; then
  kill "$SERVER_PID" 2>/dev/null || true
fi

echo ""
echo "Training finished."
echo "Models: $WORKSPACE_DIR/models"
echo "Logs:   $WORKSPACE_DIR/logs"
