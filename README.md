# Medical PRM Pipeline

Medical multi-step PRM dataset builder for VQA-style tasks.

This repository supports:
- `commercial` backend mode using the OpenAI API
- `gemini` backend mode using Google's OpenAI-compatible Gemini API
- `open_model` backend mode using an OpenAI-compatible local server
- `demo` mode in the UI for meetings and offline previews

## Quick Start

Install dependencies:

```powershell
pip install -r requirements.txt
npm install
```

Choose one provider mode by copying an example config:

```powershell
Copy-Item .env.commercial.example .env
```

or

```powershell
Copy-Item .env.open_model.example .env
```

or

```powershell
Copy-Item .env.gemini.example .env
```

Run:

```powershell
python -m http.server 8765
python api_backend.py
```

Open:

- Frontend: `http://127.0.0.1:8765/app.html`
- Backend health: `http://127.0.0.1:8764/health`

## Meeting Demo

The frontend defaults to `Demo Mock (Meeting)` mode so you can preview:
- generated solutions
- Monte Carlo rollout results
- labeling / dataset review screens
- training JSON / JSONL preview

without spending live API quota.

## What This Repo Does

The pipeline is organized as:

1. Generate multiple reasoning candidates per question
2. Score each step prefix with rollout-based Monte Carlo
3. Confirm / override labels in the UI
4. Export:
   - source/result JSON
   - VisualPRM-style JSON
   - step-level training JSON / JSONL

## Repository Layout

- `app.html`, `app.js`, `app.css`
  Frontend UI
- `api_backend.py`
  Backend API for generation and verification
- `test_mc_pipeline.py`
  Offline VisualPRM-style dataset generation test harness
- `build_step_training_json.py`
  Converts nested result JSON into step-level training JSON / JSONL
- `.env.commercial.example`
  Example config for OpenAI API mode
- `.env.open_model.example`
  Example config for local open-model mode
- `BACKEND_PROVIDERS.md`
  Provider setup notes

## Environment Configuration

Choose one provider mode.

### 1. Commercial / OpenAI API

Copy:

```powershell
Copy-Item .env.commercial.example .env
```

Then edit `.env`:

```env
MODEL_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_GENERATE_MODEL=gpt-4o-mini
OPENAI_VERIFY_MODEL=gpt-4o-mini
```

### 2. Open Model / Local Server

Copy:

```powershell
Copy-Item .env.open_model.example .env
```

Then edit `.env`:

```env
MODEL_PROVIDER=open_model
OPEN_MODEL_BASE_URL=http://127.0.0.1:8000/v1
OPEN_MODEL_API_KEY=EMPTY
OPEN_MODEL_GENERATE_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
OPEN_MODEL_VERIFY_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
```

### 3. Gemini / OpenAI-Compatible Commercial Endpoint

Copy:

```powershell
Copy-Item .env.gemini.example .env
```

Then edit `.env`:

```env
MODEL_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_GENERATE_MODEL=gemini-2.5-flash
GEMINI_VERIFY_MODEL=gemini-2.5-flash
```

## Training Data

The UI can export:

- source/result JSON
- step-level training JSON
- step-level training JSONL

In the training format, each row corresponds to one reasoning step.

Core fields:
- `image_url`
- `question`
- `options`
- `prefix_steps`
- `current_step`
- `label`
- `mc_score`

## Notes

- `.env` is ignored and should never be committed.
- `images/` is currently ignored because local assets exceed 1GB.
- Demo result JSON files are kept so the UI can run without live quota.
