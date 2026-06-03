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
TRAINING_TASK="${TRAINING_TASK:-prm_scorer}"
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
case "$TRAINING_TASK" in
  prm_scorer)
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
    TRAIN_SCRIPT="train_visual_prm.py"
    ;;
  rationale_generator)
    TRAIN_FILE="${RATIONALE_TRAIN_FILE:-$WORKSPACE_DIR/data/rationale_train_sample.jsonl}"
    VAL_FILE="${RATIONALE_VAL_FILE:-$WORKSPACE_DIR/data/rationale_val_sample.jsonl}"
    TRAIN_SCRIPT="train_rationale_generator.py"
    ;;
  visual_distill)
    TRAIN_FILE="${VISUAL_DISTILL_TRAIN_FILE:-$WORKSPACE_DIR/data/gpt_visual_distill_train.jsonl}"
    VAL_FILE="${VISUAL_DISTILL_VAL_FILE:-$WORKSPACE_DIR/data/gpt_visual_distill_val.jsonl}"
    TRAIN_SCRIPT="train_multimodal_distill.py"
    DEFAULT_MODEL_NAME="google/gemma-4-E4B-it"
    IMAGE_ROOT="${IMAGE_ROOT:-$WORKSPACE_DIR/data}"
    ;;
  *)
    echo "Unknown TRAINING_TASK: $TRAINING_TASK"
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

echo "[3/3] Train model"
EXTRA_TRAIN_ARGS=()
if [ -n "${IMAGE_ROOT:-}" ]; then
  EXTRA_TRAIN_ARGS+=(--image_root "$IMAGE_ROOT")
fi

python "$TRAIN_SCRIPT" \
  --model_name "${MODEL_NAME:-${DEFAULT_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}}" \
  --train_file "$TRAIN_FILE" \
  --val_file "$VAL_FILE" \
  "${EXTRA_TRAIN_ARGS[@]}" \
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
