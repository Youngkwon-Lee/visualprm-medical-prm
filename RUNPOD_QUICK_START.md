# RunPod Quick Start

## 1. CPU Pod

```bash
cd /workspace
git clone https://github.com/Youngkwon-Lee/visualprm-medical-prm.git visualprm
cd visualprm
bash setup_runpod.sh
bash verify_setup.sh
```

## 2. Copy Training Data

Place step-level training files in:

- `/workspace/data/train_standard.jsonl`
- `/workspace/data/val_standard.jsonl`

## 3. GPU Pod

Attach a GPU pod, then run:

```bash
cd /workspace/visualprm
bash train_runpod.sh standard
```

## Notes

- Current baseline model: `Qwen/Qwen2.5-7B-Instruct`
- Current local server is text-only OpenAI-compatible
- Current PRM training script uses step-level JSON/JSONL rows
