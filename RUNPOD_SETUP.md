# RunPod A100 Setup Guide

Complete guide to train VisualPRM with Qwen3-VL-30B on RunPod.

## Prerequisites

- RunPod account with credits
- A100-40GB or A100-80GB GPU (recommended: A100-80GB)
- 500GB+ available disk space

## Quick Start (5 minutes)

### 1. Create RunPod Pod

1. Go to [runpod.io](https://www.runpod.io/console/pods)
2. Click "GPU Pods" → Select A100-80GB
3. Choose template: **PyTorch 2.0** or **RunPod Pytorch**
4. Set storage: 500GB+
5. Click "Connect" → Copy SSH command

### 2. SSH into Pod

```bash
# Replace with your actual SSH command
ssh -p YOUR_PORT root@YOUR_IP
```

### 3. Deploy VisualPRM

```bash
cd /workspace

# Clone repository (or use your git clone)
git clone https://github.com/YOUR_ORG/visualprm.git
cd visualprm

# Run deployment script
bash deploy_runpod.sh standard
```

That's it! Training will start automatically.

---

## Manual Setup (if automated script fails)

### 1. Install Dependencies

```bash
cd /workspace
python -m pip install -U pip setuptools wheel

# Option A: Using requirements.txt
pip install -r visualprm/requirements.txt

# Option B: Manual install
pip install torch transformers accelerate bitsandbytes peft
pip install datasets Pillow tqdm flask flask-cors openai python-dotenv
```

### 2. Download Model

```bash
huggingface-cli download Qwen/Qwen3-VL-30B-Instruct --cache-dir /workspace/.cache/huggingface
```

### 3. Start Qwen Server (Terminal 1)

```bash
export QWEN_MODEL_ID="Qwen/Qwen3-VL-30B-Instruct"
export QWEN_SERVER_HOST="0.0.0.0"
export QWEN_SERVER_PORT="8000"
export QWEN_LOAD_IN_4BIT="0"

cd /workspace/visualprm
python runpod_qwen_openai_server.py
```

Wait for message: `"Starting local Qwen OpenAI-compatible server on http://0.0.0.0:8000"`

### 4. Test Server (Terminal 2)

```bash
curl http://localhost:8000/health | python -m json.tool

# Expected output:
# {
#   "status": "ok",
#   "mode": "qwen_local_server",
#   "model": "Qwen/Qwen3-VL-30B-Instruct"
# }
```

### 5. Start Training (Terminal 2)

```bash
cd /workspace/visualprm

# Option 1: Standard (162K cases) - Recommended
python train_visual_prm.py --dataset standard --batch_size 16 --epochs 3

# Option 2: MVP (35K cases) - Quick test
python train_visual_prm.py --dataset mvp --batch_size 8 --epochs 1

# Option 3: Large (389K cases) - Full benchmark
python train_visual_prm.py --dataset large --batch_size 16 --epochs 3
```

---

## Dataset Configuration

| Dataset | Cases | Size | Time (A100-80GB) | Cost |
|---------|-------|------|-----------------|------|
| MVP | 35.6K | 143K | 2.5h | $1.20 |
| Standard | 162.5K | 650K | 11h | $5.30 |
| Large | 389.5K | 1.56M | 27h | $13 |
| Mega | 1170K | 4.68M | 81h | $39 |

**Recommended: Standard** (good balance of cost and data)

---

## Environment Variables

Create `.env` file in project root:

```bash
# Model Configuration
MODEL_PROVIDER=open_model
OPEN_MODEL_BASE_URL=http://localhost:8000
OPEN_MODEL_GENERATE_MODEL=Qwen/Qwen3-VL-30B-Instruct

# Server Configuration
QWEN_MODEL_ID=Qwen/Qwen3-VL-30B-Instruct
QWEN_SERVER_PORT=8000
QWEN_MAX_NEW_TOKENS=512

# Training Configuration
TRAINING_BATCH_SIZE=16
TRAINING_EPOCHS=3
TRAINING_LEARNING_RATE=2e-5
DATASET_NAME=standard

# Compute
CUDA_VISIBLE_DEVICES=0
MIXED_PRECISION=fp16
GRADIENT_CHECKPOINTING=1
```

---

## Monitoring Training

### View Logs in Real-time

```bash
# Terminal 1: Server logs
tail -f /workspace/logs/server.log

# Terminal 2: Training logs
tail -f /workspace/logs/training.log
```

### Check GPU Usage

```bash
nvidia-smi
nvidia-smi dmon 1  # Monitor continuously
```

### Expected Training Performance

**A100-80GB with Standard (162K) dataset:**
- Epoch 1: ~4 hours
- Epoch 2: ~4 hours
- Epoch 3: ~3 hours
- **Total: ~11 hours**

---

## Saving & Retrieving Results

### Download Model from RunPod

```bash
# On your local machine
scp -P YOUR_PORT -r root@YOUR_IP:/workspace/models ./models
```

### Model Structure After Training

```
/workspace/models/
├── final/                 # Final trained model
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── config.json
├── checkpoint-500/        # Checkpoints
├── checkpoint-1000/
└── ...
```

---

## Troubleshooting

### "Out of Memory" Error

```bash
# Reduce batch size
python train_visual_prm.py --batch_size 8 --dataset standard

# Or enable 4-bit quantization in server
export QWEN_LOAD_IN_4BIT="1"
python runpod_qwen_openai_server.py
```

### Server Not Starting

```bash
# Check if port 8000 is already in use
lsof -i :8000

# Kill existing process
pkill -f "runpod_qwen_openai_server"
```

### Dataset Files Not Found

```bash
# Download dataset files manually
cd /workspace/visualprm

# Copy your JSON dataset files to:
cp your_data.json /workspace/data/
```

### CUDA Out of Memory during Download

```bash
# Use CPU instead
export CUDA_VISIBLE_DEVICES=""
pip install huggingface-hub
huggingface-cli download Qwen/Qwen3-VL-30B-Instruct
```

---

## Cost Breakdown

**Standard (162K, 11 hours on A100-80GB):**

```
GPU:         $0.48/hr × 11h  = $5.30
Data:        Included
API:         $0 (local model)
────────────────────────────
Total:       $5.30
```

vs. Cloud API approach:

```
GPU:         $5.30
API:         $3,250
────────────────────────────
Total:       $3,255 (641x more expensive)
```

---

## Next Steps

1. **Training complete?**
   - Download models: `scp -r root@pod:/workspace/models ./`

2. **Evaluate results:**
   - Run `test_mc_pipeline.py` with trained model

3. **Submit to NeurIPS:**
   - Use results in paper
   - Deadline: May 15, 2026

---

## Support

- RunPod docs: https://docs.runpod.io/
- Qwen model: https://huggingface.co/Qwen/Qwen3-VL-30B-Instruct
- Issues: Create GitHub issue or email

---

**Good luck! 🚀**
