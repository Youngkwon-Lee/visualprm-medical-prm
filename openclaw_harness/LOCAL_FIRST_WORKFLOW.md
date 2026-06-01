# Local-First Workflow

Use the MacBook for day-to-day OpenClaw work, and send only GPU-heavy jobs to
RunPod.

## Recommendation

- Fastest smoke test: local direct Ollama
- OpenClaw behavior test: local OpenClaw + local gateway
- GPU-only work: RunPod
  - MedVisualPRM Branch B reranking
  - large multimodal models
  - 100+ sample benchmark runs
  - full 669 / 1,148 sample evals

This split avoids remote path friction and battery-heavy long sessions when you
only need a small interactive check.

## Current Paths

```bash
REPO=/Users/youngkwon/projects/visualprm-medical-prm/openclaw_harness
HARNESS=/Users/youngkwon/projects/visualprm_openclaw_harness
SAMPLES=$HARNESS/data/medical_visual_process_bench/openclaw/r_pathvqa_closed_0_10_for_openclaw.json
OUT=$HARNESS/results_native_openclaw
```

## 1. Fastest Local Path

Use this when you want the shortest turnaround and do not need OpenClaw tool
planning.

```bash
ollama serve

python3 $REPO/run_ollama_vqa_direct.py \
  --samples-json "$SAMPLES" \
  --model gemma3:4b \
  --start-index 0 \
  --max-samples 10 \
  --num-ctx 2048 \
  --num-predict 128 \
  --out-jsonl "$OUT/local_direct_r_pathvqa_0_10.jsonl"
```

Use this for:

- prompt tuning
- image path sanity checks
- answer-format debugging
- quick 1 to 10 sample correctness checks

## 2. Local OpenClaw Path

Use this when you want the OpenClaw agent itself, but still want to stay on the
MacBook.

First, ensure the local OpenClaw config contains the PathVQA agent profiles:

```bash
python3 $REPO/setup_pathvqa_openclaw_agents.py \
  --workspace /Users/youngkwon/.openclaw/workspace
```

Then start the local gateway and run the harness through `gateway` mode, not
embedded `--local` mode:

```bash
nohup openclaw gateway --force run > /tmp/openclaw-gateway.log 2>&1 &

OPENCLAW_STATE_DIR=/Users/youngkwon/.openclaw \
OPENCLAW_CONFIG_PATH=/Users/youngkwon/.openclaw/openclaw.json \
python3 $REPO/run_openclaw_pathvqa_native.py \
  --samples-json "$SAMPLES" \
  --agent pathvqa-vision-direct \
  --openclaw-mode gateway \
  --input-mode normal \
  --start-index 0 \
  --max-samples 1 \
  --timeout 180 \
  --votes 1 \
  --retry-invalid 0 \
  --parallelism 1 \
  --out-jsonl "$OUT/local_openclaw_gateway_r_pathvqa_0_1.jsonl"
```

Use this for:

- OpenClaw-native agent experiments
- tool allow/deny policy checks
- one-question, one-agent-turn evaluation
- comparing direct model vs agent wrapper behavior

Important:

- Prefer `--openclaw-mode gateway`
- Avoid `--openclaw-mode local` for speed-sensitive tests
- Local OpenClaw is still slower than direct Ollama because the agent wrapper
  adds session and prompt overhead

## 3. RunPod Path

Send the job to RunPod only when the GPU materially helps.

Use RunPod for:

- `SNUH-C/medvisualprm-branch-b`
- long PRM reranking runs
- full-dataset batch evaluation
- larger future fine-tuned checkpoints

RunPod setup and current measurements are in
[RUNPOD_HANDOFF.md](RUNPOD_HANDOFF.md).

## Practical Rule

- Need the answer fastest: local direct Ollama
- Need OpenClaw behavior: local OpenClaw gateway
- Need PRM or scale: RunPod

If a test is only 1 to 10 samples, start local first. Promote to RunPod only
after the local result looks worth scaling.
