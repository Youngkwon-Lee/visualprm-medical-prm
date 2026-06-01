# RunPod OpenClaw Handoff

This note records the current RunPod/OpenClaw state so another machine or agent
can continue the VisualPRM + OpenClaw work without relying on local chat memory.

For the recommended day-to-day split between MacBook and RunPod, see
[LOCAL_FIRST_WORKFLOW.md](LOCAL_FIRST_WORKFLOW.md).

Do not commit Hugging Face, RunPod, or S3 tokens. The user pasted credentials in
chat during setup; rotate them if this environment will be shared.

## Active Pod

- Pod ID used during the latest optimization pass: `fp8nzcdkersqy5`
- Name: `visualprm-openclaw-a40`
- SSH:

```bash
ssh -i /Users/youngkwon/.runpod/ssh/RunPod-Key-Go -p 22034 root@194.68.245.144
```

- GPU observed: `NVIDIA A40`, about `46 GiB` VRAM
- Persistent workspace: `/workspace`
- Harness path on pod: `/workspace/visualprm_openclaw_harness`
- OpenClaw state path on pod: `/root/.openclaw`

If this pod is stopped/replaced, re-run `runpod_create_a40.sh` and
`runpod_bootstrap_visualprm.sh`, then rsync or clone this harness into
`/workspace/visualprm_openclaw_harness`.

## What Is Installed

- `openclaw@2026.4.24`
- Ollama server
- Ollama models pulled:
  - `qwen2.5:7b-instruct`
  - `gemma3:4b`
- Python venv:
  - `/workspace/visualprm_openclaw_harness/.venv312`

Important: the first bootstrap installed a CUDA 13 PyTorch wheel, which did not
match the RunPod driver (`CUDA driver 12.8`). Reinstall PyTorch with `cu128`
before running PRM/Transformers workloads:

```bash
cd /workspace/visualprm_openclaw_harness
. .venv312/bin/activate
python -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch torchvision
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

Ollama itself already detected the A40 with CUDA and a direct Ollama request used
GPU memory successfully.

## Dataset Counts

Medical Visual Process Bench data found in Google Drive / local converted data:

- R-PathVQA closed: 300 QA, 268 unique images
- R-PathVQA open: 300 QA, 275 unique images
- R-PathVQA total: 600 QA, 494 unique images
- R-RAD closed: 272 QA, 146 unique images
- R-RAD open: 179 QA, 111 unique images
- R-RAD total: 451 QA, 203 unique images
- Utah-WebPath closed: 97 QA/images
- Closed-only total: 669 QA
- All total: 1,148 QA, 794 unique images

The local 10-sample JSONs were generated from the Drive data, but they contained
Mac absolute paths. On RunPod, rewrite `/Users/youngkwon/projects/...` to
`/workspace/...` before running.

## OpenClaw Config Gotcha

Do not run the harness with `OPENCLAW_HOME=/root/.openclaw`.

OpenClaw CLI interprets `OPENCLAW_HOME` as a parent state root and then searches
for config at `$OPENCLAW_HOME/.openclaw/openclaw.json`. That caused:

```text
Unknown agent id "pathvqa-web"
```

Use the default root home path, or set the explicit state/config variables:

```bash
export OPENCLAW_STATE_DIR=/root/.openclaw
export OPENCLAW_CONFIG_PATH=/root/.openclaw/openclaw.json
```

The harness script now strips inherited `OPENCLAW_HOME` before invoking the
OpenClaw CLI.

## Current Smoke Status

Validated:

- SSH to RunPod works.
- `ollama list` shows both models.
- Direct Ollama API call to `qwen2.5:7b-instruct` works.
- Direct Ollama OpenAI-compatible `/v1/chat/completions` works.
- OpenClaw sees `pathvqa-web` after removing the bad `OPENCLAW_HOME`.
- One R-PathVQA OpenClaw image-agent smoke completed correctly, but slowly:
  - sample: `r_pathvqa_closed_1`
  - correct: `true`
  - latency: about `117s`
  - tools: `image`, `web_search`, `image`, `image`, `web_search`, `web_fetch`
- A partial warmed OpenClaw run over indices 1-4 completed 4 more rows:
  - correct: `3/4`
  - latency range: about `101s` to `119s` per sample

Not yet validated:

- Full benchmark run.

The first image smoke attempted before fixing the `OPENCLAW_HOME` issue should
be discarded. Direct exact-JSON OpenClaw text prompts sometimes produced empty
assistant payloads, and OpenClaw text turns remained slow even after warmup. To
force a thinking mode, set:

```bash
export OPENCLAW_THINKING=off
```

## Latency Finding

The A40 and Ollama are not the bottleneck. On the RunPod A40:

- Direct Ollama `qwen2.5:7b-instruct` text call: about `0.43s`
- Direct Ollama `gemma3:4b` image+question call: about `1.35s`
- Direct Ollama 6k-token text prompt with `num_ctx=16000`: about `1.94s`
- OpenClaw text turn through the agent harness: about `85-90s`
- OpenClaw R-PathVQA image-agent samples: about `101-119s` each
- OpenClaw `pathvqa-vision-direct` with `--local` embedded mode: about `84-96s`
- OpenClaw `pathvqa-vision-direct` through the persistent gateway: about
  `21-42s` in current tests

The GPU and Ollama are fast; the slow part is the OpenClaw agent runtime path.
The harness originally used `openclaw agent --local`, which OpenClaw documents
as the embedded agent path, not the persistent gateway path. That was the main
reason RunPod looked much slower than direct/manual model use. The harness now
has `--openclaw-mode gateway` as the default, and `--openclaw-mode local` only
for comparison.

OpenClaw still adds a large agent/system/tool harness turn even for short
prompts, and its CLI/gateway wrapper performs session bookkeeping and occasional
compaction-safeguard work. For OpenClaw-native evaluation, prefer the
`pathvqa-vision-direct` agent below: it uses `gemma3:4b` as the primary vision
model and denies all tools, so each question is answered in one OpenClaw agent
turn without repeated `image` tool calls.

Validated `pathvqa-vision-direct` one-sample result:

- Sample: `r_pathvqa_closed_1`
- Correct: `true`
- Attempts: `1`
- Tool calls: `[]`
- Image tool calls: `0`
- Wall time:
  - embedded `--local`: about `84-96s`
  - gateway mode: about `21-42s`

Do not raise the OpenClaw model context window to `32768` just to suppress the
low-context compaction warning. In testing it removed `compactionCount: 1`, but
made the end-to-end turn slower (`~40s`) than the 16k gateway path.

## Next Run Commands

Start/verify Ollama:

```bash
nohup ollama serve > /workspace/ollama.log 2>&1 &
curl -s http://127.0.0.1:11434/api/tags | jq
```

Rewrite sample JSON paths if needed:

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path("/workspace/visualprm_openclaw_harness")
d = root / "data/medical_visual_process_bench/openclaw"
old = "/Users/youngkwon/projects/visualprm_openclaw_harness"
new = str(root)

for p in sorted(d.glob("*_for_openclaw.json")):
    data = json.loads(p.read_text())
    missing = 0
    for row in data:
        for key in ("image_path", "image_url"):
            if isinstance(row.get(key), str):
                row[key] = row[key].replace(old, new)
        img = row.get("image_path") or row.get("image_url")
        if isinstance(img, str) and not Path(img).exists():
            missing += 1
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(p.name, "rows", len(data), "missing", missing)
PY
```

One-sample OpenClaw smoke:

```bash
cd /workspace/visualprm_openclaw_harness
. .venv312/bin/activate

VISUALPRM_HARNESS_ROOT=/workspace/visualprm_openclaw_harness \
OPENCLAW_STATE_DIR=/root/.openclaw \
OPENCLAW_CONFIG_PATH=/root/.openclaw/openclaw.json \
python run_openclaw_pathvqa_native.py \
  --samples-json /workspace/visualprm_openclaw_harness/data/medical_visual_process_bench/openclaw/r_pathvqa_closed_0_10_for_openclaw.json \
  --agent pathvqa-web \
  --input-mode normal \
  --start-index 0 \
  --max-samples 1 \
  --timeout 240 \
  --votes 1 \
  --retry-invalid 0 \
  --out-jsonl /workspace/visualprm_openclaw_harness/results_native_openclaw/runpod_r_pathvqa_smoke_0_1.jsonl
```

One-sample OpenClaw vision-direct smoke:

```bash
cd /workspace/visualprm_openclaw_harness

unset OPENCLAW_HOME
python3 run_openclaw_pathvqa_native.py \
  --samples-json /workspace/visualprm_openclaw_harness/data/medical_visual_process_bench/openclaw/r_pathvqa_closed_0_10_for_openclaw.json \
  --agent pathvqa-vision-direct \
  --openclaw-mode gateway \
  --input-mode normal \
  --start-index 0 \
  --max-samples 1 \
  --timeout 180 \
  --votes 1 \
  --retry-invalid 0 \
  --parallelism 1 \
  --out-jsonl /workspace/visualprm_openclaw_harness/results_native_openclaw/runpod_vision_direct_r_pathvqa_0_1.jsonl
```

The `pathvqa-vision-direct` agent must exist in `/root/.openclaw/openclaw.json`
with primary model `ollama/gemma3:4b` and all tools denied. This keeps the
experiment aligned to: one problem, one OpenClaw agent turn, one answer.

If that passes, run the same command with `--max-samples 10` first. Only after
that should the closed-only 669 QA benchmark be launched.

Fast direct Ollama multimodal run:

```bash
cd /workspace/visualprm_openclaw_harness

python3 run_ollama_vqa_direct.py \
  --samples-json /workspace/visualprm_openclaw_harness/data/medical_visual_process_bench/openclaw/r_pathvqa_closed_0_10_for_openclaw.json \
  --model gemma3:4b \
  --start-index 0 \
  --max-samples 10 \
  --num-ctx 2048 \
  --num-predict 128 \
  --out-jsonl /workspace/visualprm_openclaw_harness/results_native_openclaw/runpod_direct_ollama_r_pathvqa_0_10.jsonl
```

## PRM Reranker

The PRM critic/reranker target is:

- `SNUH-C/medvisualprm-branch-b`
- base model in current script: `google/gemma-4-E4B-it`

Use Hugging Face auth on RunPod before loading private/gated weights:

```bash
hf auth login
```

Do not copy local token values into git.
