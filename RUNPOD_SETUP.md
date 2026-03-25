# RunPod Setup Guide

This guide covers the current supported RunPod workflow for this repository.

## What Works Today

- CPU-first environment preparation
- model download and dependency installation
- text-only OpenAI-compatible local server for Qwen
- step-level PRM training from exported JSONL

## What Does Not Yet Work Automatically

- full multimodal vision server validation on RunPod
- one-click end-to-end dataset generation + training

## Recommended Workflow

### Phase 1: CPU pod

```bash
cd /workspace
git clone https://github.com/Youngkwon-Lee/visualprm-medical-prm.git visualprm
cd visualprm
bash setup_runpod.sh
bash verify_setup.sh
```

### Phase 2: GPU pod

After CPU setup is complete:

```bash
cd /workspace/visualprm
bash train_runpod.sh standard
```

## Required Data

The training script expects step-level JSONL files, for example:

- `/workspace/data/train_standard.jsonl`
- `/workspace/data/val_standard.jsonl`

These should be generated from the local PRM pipeline outputs before training.

## Backend Contract

The local Qwen server must expose:

- `GET /health`
- `POST /v1/chat/completions`

The repository backend then points to:

- `OPEN_MODEL_BASE_URL=http://127.0.0.1:8000/v1`

## Recommended Model

- `Qwen/Qwen2.5-7B-Instruct`

This is the safest current baseline for bring-up.
