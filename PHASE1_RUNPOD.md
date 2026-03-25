# Phase 1: RunPod CPU Setup

Goal:
- install dependencies
- download the base model
- prepare `.env.production`
- verify the environment before attaching a GPU pod

This phase does **not** train the model yet.

## Recommended Base Model

- `Qwen/Qwen2.5-7B-Instruct`

Reason:
- lighter and more realistic than 30B for first bring-up
- matches the current text-only OpenAI-compatible local server

## Steps

### 1. Create a cheap CPU pod

Use any low-cost CPU pod on RunPod.

### 2. SSH into the pod

```bash
ssh -p YOUR_PORT root@YOUR_IP
```

### 3. Clone the repository

```bash
cd /workspace
git clone https://github.com/Youngkwon-Lee/visualprm-medical-prm.git visualprm
cd visualprm
```

### 4. Run setup

```bash
bash setup_runpod.sh
```

### 5. Verify setup

```bash
bash verify_setup.sh
```

## Expected Output

You should see:
- Python available
- required packages installed
- model cache prepared
- `.env.production` created

## Important

This phase does **not** prove PRM training is complete.
It only proves the RunPod environment is ready for GPU training.

## Before Phase 2

Prepare step-level PRM training data under `/workspace/data`, for example:
- `train_standard.jsonl`
- `val_standard.jsonl`

These files should come from the local pipeline exports, not dummy data.
