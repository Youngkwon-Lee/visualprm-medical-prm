#!/usr/bin/env bash
set -euo pipefail

HARNESS_ROOT="${HARNESS_ROOT:-/workspace/visualprm_openclaw_harness}"
STATE_DIR="${OPENCLAW_STATE_DIR:-/root/.openclaw}"
CONFIG_PATH="${OPENCLAW_CONFIG_PATH:-/root/.openclaw/openclaw.json}"
SAMPLES_JSON="${SAMPLES_JSON:-$HARNESS_ROOT/data/medical_visual_process_bench/openclaw/full_0_0_for_openclaw.json}"
OUT_JSONL="${OUT_JSONL:-$HARNESS_ROOT/results_native_openclaw/runpod_vision_direct_full_subset_0_100.jsonl}"

cd "$HARNESS_ROOT"

export OPENCLAW_STATE_DIR="$STATE_DIR"
export OPENCLAW_CONFIG_PATH="$CONFIG_PATH"
export VISUALPRM_HARNESS_ROOT="$HARNESS_ROOT"
export OPENCLAW_THINKING="${OPENCLAW_THINKING:-off}"

python3 run_openclaw_pathvqa_native.py \
  --samples-json "$SAMPLES_JSON" \
  --agent pathvqa-vision-direct \
  --openclaw-mode gateway \
  --input-mode normal \
  --start-index 0 \
  --max-samples 100 \
  --timeout 240 \
  --votes 1 \
  --retry-invalid 0 \
  --parallelism 1 \
  --out-jsonl "$OUT_JSONL"
