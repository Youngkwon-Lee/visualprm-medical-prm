#!/usr/bin/env bash
set -euo pipefail

# VisualPRM Setup Verification Script
# GPU 없이도 실행 가능
# 모든 준비가 완료되었는지 확인

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"

echo "=========================================="
echo "VisualPRM Setup Verification"
echo "=========================================="
echo ""

PASS=0
FAIL=0

# 1. Check Python
echo "[1/6] Checking Python..."
if python --version > /dev/null 2>&1; then
    PYTHON_VERSION=$(python --version)
    echo "  ✅ $PYTHON_VERSION"
    ((PASS++))
else
    echo "  ❌ Python not found"
    ((FAIL++))
fi

# 2. Check dependencies
echo "[2/6] Checking Python packages..."
PACKAGES=("torch" "transformers" "accelerate" "flask" "openai")
for pkg in "${PACKAGES[@]}"; do
    if python -c "import $pkg" 2>/dev/null; then
        VERSION=$(python -c "import $pkg; print(getattr($pkg, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
        echo "  ✅ $pkg ($VERSION)"
        ((PASS++))
    else
        echo "  ❌ $pkg missing"
        ((FAIL++))
    fi
done

# 3. Check model cache
echo "[3/6] Checking model cache..."
MODEL_CACHE="$WORKSPACE_DIR/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-Instruct"
if [ -d "$MODEL_CACHE" ]; then
    SIZE=$(du -sh "$MODEL_CACHE" 2>/dev/null | cut -f1)
    echo "  ✅ Model cached ($SIZE)"
    ((PASS++))
else
    echo "  ⚠️  Model not cached yet"
    echo "     Run: bash setup_runpod.sh"
fi

# 4. Check workspace directories
echo "[4/6] Checking workspace structure..."
for dir in data models .cache logs; do
    if [ -d "$WORKSPACE_DIR/$dir" ]; then
        echo "  ✅ $dir/"
        ((PASS++))
    else
        echo "  ⚠️  $dir/ not found"
    fi
done

# 5. Check configuration
echo "[5/6] Checking configuration..."
CONFIG_FILE="$WORKSPACE_DIR/visualprm/.env.production"
if [ -f "$CONFIG_FILE" ]; then
    echo "  ✅ Configuration exists"
    ((PASS++))

    # Check critical vars
    if grep -q "QWEN_MODEL_ID" "$CONFIG_FILE"; then
        echo "    ✅ Model configured"
    fi
    if grep -q "TRAINING_BATCH_SIZE" "$CONFIG_FILE"; then
        echo "    ✅ Training config set"
    fi
else
    echo "  ❌ Configuration not found"
    ((FAIL++))
fi

# 6. Test imports
echo "[6/6] Testing critical imports..."
python << 'EOF'
import sys
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("  ✅ HuggingFace transformers")

    import torch
    print("  ✅ PyTorch")

    from flask import Flask
    print("  ✅ Flask")

    import openai
    print("  ✅ OpenAI SDK")

    print("\n  All imports successful!")
except Exception as e:
    print(f"  ❌ Import error: {e}")
    sys.exit(1)
EOF

echo ""
echo "=========================================="
echo "Verification Summary"
echo "=========================================="
echo "✅ Passed: $PASS"
echo "❌ Failed: $FAIL"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "✅ Setup is ready for GPU training!"
    echo ""
    echo "Next steps:"
    echo "  1. Switch to A100-80GB GPU instance"
    echo "  2. Run: bash train_runpod.sh standard"
    echo ""
else
    echo "⚠️  Some checks failed"
    echo "Run: bash setup_runpod.sh"
    exit 1
fi

echo "=========================================="
