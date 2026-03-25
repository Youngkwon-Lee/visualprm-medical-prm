#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
PROJECT_DIR="$WORKSPACE_DIR/visualprm"
VENV_DIR="${VENV_DIR:-/root/visualprm-venv}"
PASS=0
FAIL=0

echo "=========================================="
echo "VisualPRM RunPod Verification"
echo "=========================================="

check_ok() {
  echo "  OK  $1"
  PASS=$((PASS + 1))
}

check_fail() {
  echo "  FAIL  $1"
  FAIL=$((FAIL + 1))
}

echo "[1/7] Virtual environment"
if [ -f "$VENV_DIR/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  check_ok "$VENV_DIR"
else
  check_fail "virtualenv missing at $VENV_DIR"
fi

echo "[2/7] Python"
if python --version >/dev/null 2>&1; then
  check_ok "$(python --version 2>&1)"
else
  check_fail "python not found"
fi

echo "[3/7] Required packages"
for pkg in torch transformers accelerate flask openai peft; do
  if python -c "import $pkg" >/dev/null 2>&1; then
    check_ok "$pkg"
  else
    check_fail "$pkg missing"
  fi
done

echo "[4/7] Project files"
for file in api_backend.py train_visual_prm.py runpod_qwen_openai_server.py .env.runpod; do
  if [ -f "$PROJECT_DIR/$file" ]; then
    check_ok "$file"
  else
    check_fail "$file missing"
  fi
done

echo "[5/7] Model cache"
if [ -d "$WORKSPACE_DIR/.cache/huggingface" ]; then
  check_ok "huggingface cache exists"
else
  check_fail "huggingface cache missing"
fi

echo "[6/7] Environment file"
if [ -f "$PROJECT_DIR/.env.production" ]; then
  check_ok ".env.production exists"
  grep -q "OPEN_MODEL_BASE_URL=http://127.0.0.1:8000/v1" "$PROJECT_DIR/.env.production" && check_ok "base URL uses /v1" || check_fail "OPEN_MODEL_BASE_URL missing /v1"
else
  check_fail ".env.production missing"
fi

echo "[7/7] Step-level data expectation"
if [ -f "$WORKSPACE_DIR/data/train_standard.jsonl" ]; then
  check_ok "train_standard.jsonl exists"
else
  echo "  INFO  train_standard.jsonl not found yet (expected before GPU training)"
fi

if [ -f "$WORKSPACE_DIR/data/val_standard.jsonl" ]; then
  check_ok "val_standard.jsonl exists"
else
  echo "  INFO  val_standard.jsonl not found yet (expected before GPU training)"
fi

echo ""
echo "=========================================="
echo "Passed: $PASS"
echo "Failed: $FAIL"
echo "=========================================="

if [ "$FAIL" -gt 0 ]; then
  exit 1
fi
