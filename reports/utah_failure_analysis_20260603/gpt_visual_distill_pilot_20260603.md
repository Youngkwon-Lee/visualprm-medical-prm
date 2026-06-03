# GPT Visual Evidence Distillation Pilot - 2026-06-03

## What Was Prepared
- Source teacher file: `utah_webpath_closed_openai_gpt54_direct_0_97_newkey.jsonl`
- Teacher rows: 97
- Kept rows: 86 GPT-correct rows
- Distill records: 172 total
- Descriptor records: 86
- Option-verifier records: 86
- Train/val split: 155 train, 17 val

## Output Files
- `runpod_data/gpt_visual_distill_utah_debug_20260603/gpt_visual_distill_train.jsonl`
- `runpod_data/gpt_visual_distill_utah_debug_20260603/gpt_visual_distill_val.jsonl`
- `runpod_data/gpt_visual_distill_utah_debug_20260603/gpt_visual_distill_descriptor.jsonl`
- `runpod_data/gpt_visual_distill_utah_debug_20260603/gpt_visual_distill_verifier.jsonl`
- `runpod_data/gpt_visual_distill_utah_debug_20260603/gpt_visual_distill_manifest.json`

## Important Caveat
This is a pilot/debug distillation set because the teacher source is Utah WebPath closed test output. It should not be used to train and then report fair accuracy on the same Utah 97 samples.

Use it for:
- debugging whether Gemma4 can learn GPT-style visual evidence phrasing
- checking if visual descriptor quality improves
- smoke-testing the multimodal SFT pipeline

For fair evaluation:
- train on non-test or separate teacher-generated data
- evaluate on held-out Utah samples or another benchmark split

## RunPod Training Command
Copy the distill JSONL files and image folder under `/workspace/data`, then run:

```bash
cd /workspace/visualprm

TRAINING_TASK=visual_distill \
MODEL_NAME=google/gemma-4-E4B-it \
VISUAL_DISTILL_TRAIN_FILE=/workspace/data/gpt_visual_distill_train.jsonl \
VISUAL_DISTILL_VAL_FILE=/workspace/data/gpt_visual_distill_val.jsonl \
IMAGE_ROOT=/workspace/data \
TRAINING_EPOCHS=1 \
TRAINING_BATCH_SIZE=1 \
TRAINING_GRAD_ACCUM=8 \
TRAINING_LEARNING_RATE=1e-5 \
bash train_runpod.sh
```

## Next Check After Training
Run the same next-10 debug set with the trained adapter and compare:
- baseline Gemma4+PRM: 0/10 on this hard set
- option-specific verification: 2/10
- GPT visual hint diagnostic: 4/10

The first target is not full benchmark accuracy. The first target is whether the trained student writes better visual descriptors for cases like thrombocytopenia, Trisomy 21, dense collagen bundles, and suicidal gunshot wound.
