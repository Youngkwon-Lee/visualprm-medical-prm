# VisualPRM-Style Training Data Schema

This document explains how teammates should read and use the current VisualPRM-style rollout data.

## Files

- Raw rollout result:
  - `runpod_artifacts/pathvqa_test_results_vlm100_2s_4k.json`
- VisualPRM-style JSONL:
  - `runpod_artifacts/pathvqa_visualprm_style_vlm100_2s_4k.jsonl`

## What One JSONL Row Means

One row represents:
- one case
- one generated solution for that case
- the full multi-step rationale for that solution
- one supervision value for each step

So the unit is:
- `case x solution`

Not:
- one row per step

## Core Fields

- `image`
  - relative path to the image used for the case
- `question`
  - the VQA question
- `options`
  - answer choices
- `gold_index`, `gold_letter`
  - ground-truth answer
- `solution_index`
  - which sampled solution this is
- `final_answer_index`, `final_answer_letter`
  - model answer for that sampled solution
- `solution_expected_accuracy`
  - final-step Monte Carlo score for the whole solution
- `solution_label`
  - `+` if final-step expected accuracy is greater than `0`, otherwise `-`
- `steps`
  - array of step-level supervision entries

## Step Supervision

Each item in `steps` has:
- `step_index`
- `title`
- `text`
- `expected_accuracy`
- `label`
- `rollout_success`
- `rollout_total`

Interpretation:
- `expected_accuracy = rollout_success / rollout_total`
- if `expected_accuracy > 0`, that step is treated as positive
- if `expected_accuracy = 0`, that step is treated as negative

Important:
- zero-score steps are still useful training signals
- do not drop them by default if the training objective needs positive and negative supervision

## Example

```json
{
  "id": "PathVQA::pathvqa_000001::solution_1",
  "image": "images/pathvqa/pathvqa_000001.jpg",
  "question": "is normal palmar creases present?",
  "final_answer_letter": "B",
  "solution_expected_accuracy": 1.0,
  "steps": [
    {
      "step_index": 1,
      "title": "Examine the Image",
      "expected_accuracy": 1.0,
      "label": "+",
      "rollout_success": 4,
      "rollout_total": 4
    },
    {
      "step_index": 2,
      "title": "Identify Palmar Creases",
      "expected_accuracy": 1.0,
      "label": "+",
      "rollout_success": 4,
      "rollout_total": 4
    }
  ]
}
```

## How A Teammate Can Train With It

Typical learning target:
- input:
  - image
  - question
  - options
  - step text
  - optionally prefix steps
- target:
  - `expected_accuracy`
  - or a derived binary label from that score

Two common ways to use the file:

1. Keep one row as `case x solution`
   - model reads the whole solution
   - predicts one score per step

2. Flatten each row into step-level samples
   - one training sample per step
   - keep image/question/options
   - attach `prefix_steps`, `current_step`, `expected_accuracy`

The second form is usually easier for PRM scorer training.

## Flatten Helper

This repository now includes:

- `flatten_visualprm_style_steps.py`

Use it like this:

```bash
python flatten_visualprm_style_steps.py \
  --input runpod_artifacts/pathvqa_visualprm_style_vlm100_2s_4k.jsonl \
  --output runpod_data/pathvqa_visualprm_style_vlm100_step_level.jsonl
```

Output shape:
- one row per step
- keeps `image`, `question`, `options`
- adds `prefix_steps`
- adds `current_step`
- keeps `expected_accuracy`, `label`, `rollout_success`, `rollout_total`

## Current Run Summary

- completed cases: `99`
- solutions per case: `2`
- rollout `k`: `4`
- total rows in JSONL: `198`
- average primary final-step MC: `0.75`
- average best-solution final-step MC: `0.798`
- primary positive rate: `78.8%`

## Recommended Use

- treat this file as a research-grade training dataset
- use it for scorer training first
- evaluate the trained model on a separate physician-reviewed benchmark

Do not use the physician-reviewed benchmark as training data unless that is an explicit experimental choice.
