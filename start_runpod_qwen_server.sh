#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
VENV_DIR="${VENV_DIR:-/root/visualprm-venv}"
HF_HOME="${HF_HOME:-$WORKSPACE_DIR/.cache/huggingface}"

if [ -f "$VENV_DIR/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
else
  echo "Python virtual environment not found at $VENV_DIR"
  echo "Run: bash setup_runpod.sh"
  exit 1
fi

export HF_HOME
export QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
export QWEN_SERVER_HOST="${QWEN_SERVER_HOST:-0.0.0.0}"
export QWEN_SERVER_PORT="${QWEN_SERVER_PORT:-8000}"
export QWEN_LOAD_IN_4BIT="${QWEN_LOAD_IN_4BIT:-0}"

python runpod_qwen_openai_server.py
