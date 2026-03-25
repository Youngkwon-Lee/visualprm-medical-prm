#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
PROJECT_DIR="$WORKSPACE_DIR/visualprm"
VENV_DIR="${VENV_DIR:-/root/visualprm-venv}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"

mkdir -p "$WORKSPACE_DIR/logs"

cat > "$PROJECT_DIR/.env" <<EOF
MODEL_PROVIDER=open_model
OPEN_MODEL_BASE_URL=http://127.0.0.1:8000/v1
OPEN_MODEL_API_KEY=EMPTY
OPEN_MODEL_GENERATE_MODEL=$MODEL_PATH
OPEN_MODEL_VERIFY_MODEL=$MODEL_PATH
QWEN_SERVE_MODEL_ID=$MODEL_PATH
QWEN_SERVER_HOST=0.0.0.0
QWEN_SERVER_PORT=8000
QWEN_MAX_NEW_TOKENS=512
QWEN_LOAD_IN_4BIT=0
MC_BACKEND_URL=http://127.0.0.1:8764
HF_HOME=$WORKSPACE_DIR/.cache/huggingface
EOF

cp "$PROJECT_DIR/.env" "$PROJECT_DIR/.env.production"
set -a
# shellcheck disable=SC1090
source "$PROJECT_DIR/.env"
set +a

pkill -f runpod_qwen_openai_server.py || true
pkill -f api_backend.py || true

nohup "$VENV_DIR/bin/python" "$PROJECT_DIR/runpod_qwen_openai_server.py" > "$WORKSPACE_DIR/logs/open_model_server.log" 2>&1 < /dev/null &
echo $! > "$WORKSPACE_DIR/logs/open_model_server.pid"

for ((i=1; i<=900; i++)); do
  if curl -s http://127.0.0.1:8000/health >/dev/null 2>&1; then
    break
  fi
  if [ "$i" -eq 900 ]; then
    echo "open model server failed to start"
    exit 1
  fi
  sleep 1
done

nohup "$VENV_DIR/bin/python" "$PROJECT_DIR/api_backend.py" > "$WORKSPACE_DIR/logs/api_backend.log" 2>&1 < /dev/null &
echo $! > "$WORKSPACE_DIR/logs/api_backend.pid"

for ((i=1; i<=180; i++)); do
  if curl -s http://127.0.0.1:8764/health >/dev/null 2>&1; then
    break
  fi
  if [ "$i" -eq 180 ]; then
    echo "api backend failed to start"
    exit 1
  fi
  sleep 1
done

echo "stack_ready"
