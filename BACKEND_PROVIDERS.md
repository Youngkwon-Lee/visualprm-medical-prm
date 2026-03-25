# Backend Providers

This project supports two backend modes:

## 1. Commercial / OpenAI API

Use:
- `.env.commercial.example`

Key variables:
- `MODEL_PROVIDER=openai`
- `OPENAI_API_KEY`
- `OPENAI_GENERATE_MODEL`
- `OPENAI_VERIFY_MODEL`

Recommended default for commercial text-first experiments:
- `OPENAI_GENERATE_MODEL=gpt-4o-mini`
- `OPENAI_VERIFY_MODEL=gpt-4o-mini`

## 2. Commercial / Gemini API

Use:
- `.env.gemini.example`

Key variables:
- `MODEL_PROVIDER=gemini`
- `GEMINI_API_KEY`
- `GEMINI_BASE_URL`
- `GEMINI_GENERATE_MODEL`
- `GEMINI_VERIFY_MODEL`

Recommended default:
- `GEMINI_GENERATE_MODEL=gemini-2.5-flash`
- `GEMINI_VERIFY_MODEL=gemini-2.5-flash`

Notes:
- Gemini uses the official OpenAI-compatible endpoint:
  - `https://generativelanguage.googleapis.com/v1beta/openai/`

## 3. Open-model / Local server

Use:
- `.env.open_model.example`

Key variables:
- `MODEL_PROVIDER=open_model`
- `OPEN_MODEL_BASE_URL`
- `OPEN_MODEL_API_KEY`
- `OPEN_MODEL_GENERATE_MODEL`
- `OPEN_MODEL_VERIFY_MODEL`

Notes:
- The open-model backend assumes an OpenAI-compatible local endpoint.
- A typical setup is a local Qwen2.5-VL deployment behind `vLLM` or another compatible server.
- The Flask API surface stays the same:
  - `POST /generate-steps`
  - `POST /verify-steps`
  - `GET /health`
