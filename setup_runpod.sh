#!/usr/bin/env bash
set -euo pipefail

# VisualPRM RunPod Setup (CPU/저가 인스턴스용)
# GPU 시작 전에 모든 세팅을 완료
# 모델 다운로드, 의존성 설치, 환경 설정
# GPU 비용 0!

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "VisualPRM Setup (CPU Only)"
echo "=========================================="
echo "Workspace: $WORKSPACE_DIR"
echo ""

# 1. Create workspace
echo "[1/5] Creating workspace directories..."
mkdir -p "$WORKSPACE_DIR"/{data,models,.cache,logs}
cd "$WORKSPACE_DIR"

# 2. Copy repository
echo "[2/5] Setting up repository..."
if [ ! -d "visualprm" ]; then
    if [ -d "$REPO_DIR" ]; then
        cp -r "$REPO_DIR" visualprm
    else
        echo "Error: Repository not found at $REPO_DIR"
        exit 1
    fi
fi
cd visualprm

# 3. Install dependencies (Python packages only, no GPU)
echo "[3/5] Installing Python dependencies..."
python -m pip install -q --upgrade pip setuptools wheel
pip install -q -r requirements.txt
pip install -q huggingface-hub

echo "  ✅ Dependencies installed"

# 4. Download model to cache (큰 작업, 10-20분 소요)
echo "[4/5] Downloading Qwen3-VL-30B model to cache..."
echo "  This may take 10-20 minutes (depends on internet speed)"
echo "  Model size: ~65GB"
echo ""

python << 'EOF'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

cache_dir = Path("/workspace/.cache/huggingface")
cache_dir.mkdir(parents=True, exist_ok=True)

model_id = "Qwen/Qwen3-VL-30B-Instruct"
print(f"Downloading {model_id}...")
print(f"Cache: {cache_dir}")
print("")

try:
    downloaded_path = snapshot_download(
        model_id,
        cache_dir=str(cache_dir),
        local_dir_use_symlinks=False,
        revision="main",
    )
    print(f"\n✅ Model downloaded successfully!")
    print(f"   Path: {downloaded_path}")

    # 모델 구조 확인
    model_files = list(Path(downloaded_path).iterdir())
    print(f"   Files: {len(model_files)} files")
    for f in sorted(model_files)[:5]:
        print(f"     - {f.name}")

except Exception as e:
    print(f"❌ Download failed: {e}")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo "  ✅ Model cache ready"
else
    echo "  ❌ Model download failed"
    exit 1
fi

# 5. Environment setup
echo ""
echo "[5/5] Creating environment configuration..."

# .env 파일 생성 (운영용)
cat > "$WORKSPACE_DIR/visualprm/.env.production" << 'ENVFILE'
# RunPod A100 Production Configuration

MODEL_PROVIDER=open_model
OPEN_MODEL_BASE_URL=http://localhost:8000
OPEN_MODEL_API_KEY=EMPTY
OPEN_MODEL_GENERATE_MODEL=Qwen/Qwen3-VL-30B-Instruct

QWEN_MODEL_ID=Qwen/Qwen3-VL-30B-Instruct
QWEN_SERVER_HOST=0.0.0.0
QWEN_SERVER_PORT=8000
QWEN_MAX_NEW_TOKENS=512
QWEN_LOAD_IN_4BIT=0

CUDA_VISIBLE_DEVICES=0
MIXED_PRECISION=fp16
GRADIENT_CHECKPOINTING=1

TRAINING_BATCH_SIZE=16
TRAINING_EPOCHS=3
TRAINING_LEARNING_RATE=2e-5
DATASET_NAME=standard

WORKSPACE_DIR=/workspace
DATA_DIR=/workspace/data
OUTPUT_DIR=/workspace/models
CACHE_DIR=/workspace/.cache
ENVFILE

echo "  ✅ Configuration created: .env.production"

# Setup verification
echo ""
echo "=========================================="
echo "Setup Complete! ✅"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. VERIFY SETUP (test server without GPU)"
echo "   bash verify_setup.sh"
echo ""
echo "2. START GPU INSTANCE (A100-80GB)"
echo "   - Go to RunPod console"
echo "   - Pause current instance"
echo "   - Select A100-80GB GPU"
echo "   - Resume"
echo ""
echo "3. RUN TRAINING (on GPU instance)"
echo "   bash train_runpod.sh standard"
echo ""
echo "=========================================="
echo "Setup saved in: $WORKSPACE_DIR"
echo "Cache size: $(du -sh $WORKSPACE_DIR/.cache 2>/dev/null | cut -f1)"
echo "=========================================="
