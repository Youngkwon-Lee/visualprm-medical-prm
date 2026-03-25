#!/usr/bin/env bash
set -euo pipefail

# CPU-first setup for RunPod.
# Goal: prepare environment and download the model before attaching a GPU pod.

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$WORKSPACE_DIR/visualprm"
CACHE_DIR="$WORKSPACE_DIR/.cache/huggingface"
VENV_DIR="${VENV_DIR:-/root/visualprm-venv}"
WORKSPACE_VENV_LINK="$WORKSPACE_DIR/.venv"
VENV_CONFIG="$VENV_DIR/pyvenv.cfg"

echo "=========================================="
echo "VisualPRM RunPod Setup (CPU-first)"
echo "=========================================="
echo "Workspace: $WORKSPACE_DIR"
echo "Project:   $PROJECT_DIR"
echo ""

mkdir -p "$WORKSPACE_DIR"/{data,models,logs,.cache}
cd "$WORKSPACE_DIR"

if [ ! -d "$PROJECT_DIR" ]; then
  cp -r "$REPO_DIR" "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"

if [ -f "$PROJECT_DIR/.env.runpod" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/.env.runpod"
  set +a
fi

SERVE_MODEL_ID="${QWEN_SERVE_MODEL_ID:-${QWEN_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}}"

echo "[1/5] Creating Python virtual environment"
if [ ! -f "$VENV_CONFIG" ] || ! grep -qi '^include-system-site-packages = true' "$VENV_CONFIG"; then
  if [ -d "$VENV_DIR" ]; then
    mv "$VENV_DIR" "${VENV_DIR}.old.$(date +%s)"
  fi
  python -m venv --system-site-packages "$VENV_DIR"
fi

if [ -e "$WORKSPACE_VENV_LINK" ] || [ -L "$WORKSPACE_VENV_LINK" ]; then
  CURRENT_LINK_TARGET="$(readlink "$WORKSPACE_VENV_LINK" 2>/dev/null || true)"
  if [ "$CURRENT_LINK_TARGET" != "$VENV_DIR" ]; then
    if [ -L "$WORKSPACE_VENV_LINK" ]; then
      rm -f "$WORKSPACE_VENV_LINK"
    else
      mv "$WORKSPACE_VENV_LINK" "${WORKSPACE_VENV_LINK}.old.$(date +%s)"
    fi
  fi
fi

if [ ! -L "$WORKSPACE_VENV_LINK" ]; then
  ln -s "$VENV_DIR" "$WORKSPACE_VENV_LINK"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[2/5] Installing Python dependencies"
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  datasets \
  Pillow \
  tqdm \
  flask \
  flask-cors \
  openai \
  python-dotenv \
  huggingface-hub \
  filelock==3.19.1 \
  fsspec==2025.9.0 \
  transformers==4.57.3 \
  accelerate==0.33.0 \
  tokenizers \
  sentencepiece \
  peft \
  qwen-vl-utils

if [[ "$SERVE_MODEL_ID" == *"-VL-"* ]]; then
  python -m pip install --upgrade accelerate>=1.13.0
  python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

echo ""
echo "[3/5] Downloading model cache"
export HF_HOME="$CACHE_DIR"
export SERVE_MODEL_ID
python - << 'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(
    os.environ["SERVE_MODEL_ID"],
    cache_dir="/workspace/.cache/huggingface",
    local_dir_use_symlinks=False,
    revision="main",
)
print("Model download complete")
PY

echo ""
echo "[4/5] Writing runpod environment file"
cp .env.runpod .env.production
echo "Created $PROJECT_DIR/.env.production"

echo ""
echo "[5/5] Verifying setup"
bash verify_setup.sh

echo ""
echo "=========================================="
echo "Setup complete"
echo "=========================================="
echo "Next:"
echo "  1. Attach or switch to a GPU pod"
echo "  2. Prepare step-level training JSONL under /workspace/data"
echo "  3. Run: bash train_runpod.sh standard"
