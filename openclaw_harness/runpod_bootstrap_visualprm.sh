#!/usr/bin/env bash
set -euo pipefail

# Run this inside the RunPod pod after SSH connects.
# It installs system deps, Node/OpenClaw, Ollama, and Python packages needed by
# the local VisualPRM/OpenClaw harness. No secrets are written here.

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y \
  build-essential \
  ca-certificates \
  curl \
  git \
  git-lfs \
  jq \
  python3-dev \
  python3-venv \
  unzip

if ! command -v node >/dev/null 2>&1 || ! node -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 22 ? 0 : 1)'; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
  apt-get install -y nodejs
fi

npm install -g openclaw@2026.4.24

if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi

mkdir -p /workspace/visualprm_openclaw_harness
cd /workspace/visualprm_openclaw_harness

python3 -m venv .venv312
. .venv312/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch \
  torchvision
python -m pip install \
  accelerate \
  gdown \
  huggingface_hub \
  peft \
  requests \
  safetensors \
  sentencepiece \
  timm \
  transformers

cat <<'EOF'

Bootstrap finished.

Next manual steps:
1. Copy or git-clone the harness into /workspace/visualprm_openclaw_harness.
2. Run `hf auth login` for SNUH-C/medvisualprm-branch-b if needed.
3. Start Ollama if it is not already running:
     ollama serve
4. Pull evaluation models:
     ollama pull qwen2.5:7b-instruct
     ollama pull gemma3:4b

EOF
