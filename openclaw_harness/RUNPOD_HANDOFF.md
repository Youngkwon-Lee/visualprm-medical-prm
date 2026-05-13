# RunPod OpenClaw Handoff

This note records the current RunPod/OpenClaw state so another machine or agent
can continue the VisualPRM + OpenClaw work without relying on local chat memory.

Do not commit Hugging Face, RunPod, or S3 tokens. The user pasted credentials in
chat during setup; rotate them if this environment will be shared.

## Active Pod

- Pod ID: `9ub57bq7dfzwu5`
- Name: `visualprm-openclaw-a40`
- SSH:

```bash
ssh -i /Users/youngkwon/.runpod/ssh/RunPod-Key-Go -p 22155 root@194.68.245.144
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
- A simple OpenClaw text turn worked:
  - provider: `ollama`
  - model: `qwen2.5:7b-instruct`
  - output: `Hello! How can I assist you today?`

Not yet validated:

- Full PathVQA image-agent sample on RunPod.
- Full benchmark run.

The first image smoke was attempted before fixing the `OPENCLAW_HOME` issue, so
that result should be discarded. A direct exact-JSON text prompt with
`--thinking off` also produced an empty assistant payload once; the harness no
longer forces `--thinking off`. To force a thinking mode, set:

```bash
export OPENCLAW_THINKING=off
```

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

If that passes, run the same command with `--max-samples 10` first. Only after
that should the closed-only 669 QA benchmark be launched.

## PRM Reranker

The PRM critic/reranker target is:

- `SNUH-C/medvisualprm-branch-b`
- base model in current script: `google/gemma-4-E4B-it`

Use Hugging Face auth on RunPod before loading private/gated weights:

```bash
hf auth login
```

Do not copy local token values into git.
