#!/usr/bin/env python3
"""
Simple Flask backend for OpenAI API integration.
Handles reasoning generation and step verification for the local app.
"""

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import openai
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def load_local_env():
    """Load a simple .env file from the project root if present."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_local_env()
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai").strip().lower()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_GENERATE_MODEL = os.getenv("OPENAI_GENERATE_MODEL", "gpt-4o")
OPENAI_VERIFY_MODEL = os.getenv("OPENAI_VERIFY_MODEL", OPENAI_GENERATE_MODEL)

OPEN_MODEL_BASE_URL = os.getenv("OPEN_MODEL_BASE_URL", "").strip()
OPEN_MODEL_API_KEY = os.getenv("OPEN_MODEL_API_KEY", "EMPTY").strip() or "EMPTY"
OPEN_MODEL_GENERATE_MODEL = os.getenv("OPEN_MODEL_GENERATE_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
OPEN_MODEL_VERIFY_MODEL = os.getenv("OPEN_MODEL_VERIFY_MODEL", OPEN_MODEL_GENERATE_MODEL)


def build_provider_client():
    if MODEL_PROVIDER == "open_model":
        if not OPEN_MODEL_BASE_URL:
            return None
        return openai.OpenAI(api_key=OPEN_MODEL_API_KEY, base_url=OPEN_MODEL_BASE_URL)
    if not OPENAI_API_KEY:
        return None
    return openai.OpenAI(api_key=OPENAI_API_KEY)


def provider_generate_model() -> str:
    return OPEN_MODEL_GENERATE_MODEL if MODEL_PROVIDER == "open_model" else OPENAI_GENERATE_MODEL


def provider_verify_model() -> str:
    return OPEN_MODEL_VERIFY_MODEL if MODEL_PROVIDER == "open_model" else OPENAI_VERIFY_MODEL


def provider_ready() -> bool:
    return client is not None


client = build_provider_client()


def _parse_json_text(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def _chat_json_completion(*, messages: list[dict], model: str, temperature: float, max_tokens: int, timeout: int, top_p: float | None = None, attempts: int = 3) -> dict:
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
                "response_format": {"type": "json_object"},
            }
            if top_p is not None:
                kwargs["top_p"] = top_p
            response = client.chat.completions.create(**kwargs)
            return _parse_json_text(response.choices[0].message.content)
        except (json.JSONDecodeError, openai.APIError, ValueError) as exc:
            last_exc = exc
            if attempt == attempts - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise last_exc or RuntimeError("unknown completion failure")


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z]{3,}", (text or "").lower())
        if token not in {"the", "and", "with", "from", "that", "this", "there", "which", "into", "about"}
    }


def heuristic_step_verdict(
    question: str,
    options: list[str],
    gold: int,
    step_title: str,
    step_text: str,
    step_index: int,
    total_steps: int,
) -> dict:
    """Deterministic verifier fallback when OpenAI isn't available."""
    step_text = (step_text or "").strip()
    question_tokens = _tokenize(question)
    step_tokens = _tokenize(f"{step_title} {step_text}")
    gold_text = options[gold] if isinstance(gold, int) and 0 <= gold < len(options) else ""
    gold_tokens = _tokenize(gold_text)
    overlap = len(question_tokens & step_tokens)
    gold_overlap = len(gold_tokens & step_tokens)
    score = 0.2

    if len(step_text) >= 40:
        score += 0.15
    if overlap:
        score += min(0.25, overlap * 0.06)
    if gold_overlap:
        score += min(0.25, gold_overlap * 0.1)

    generic_markers = ["placeholder", "step ", "analysis content", "insufficient analysis"]
    if any(marker in step_text.lower() for marker in generic_markers):
        score -= 0.25

    caution_markers = ["weaker", "mismatch", "partial", "nonspecific", "uncertain", "incomplete"]
    if any(marker in step_text.lower() for marker in caution_markers):
        score -= 0.12

    if step_index == total_steps - 1:
        answer_markers = [gold_text.lower(), f"answer: {chr(65 + gold).lower()}"] if gold_text else []
        if any(marker and marker in step_text.lower() for marker in answer_markers):
            score += 0.2
        elif "final answer" in step_text.lower():
            score -= 0.1

    score = max(0.0, min(1.0, score))
    label = "+" if score >= 0.5 else "-"
    rationale = (
        "Matched question/gold evidence." if label == "+"
        else "Weak grounding to the question or gold answer."
    )
    return {"score": round(score, 3), "label": label, "rationale": rationale}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "provider": MODEL_PROVIDER,
        "client_ready": provider_ready(),
        "api_key_configured": bool(OPENAI_API_KEY) if MODEL_PROVIDER == "openai" else bool(OPEN_MODEL_BASE_URL),
        "mode": MODEL_PROVIDER if client else f"{MODEL_PROVIDER}_not_ready",
        "generate_model": provider_generate_model(),
        "verify_model": provider_verify_model(),
    }), 200


@app.route("/generate-steps", methods=["POST"])
def generate_steps():
    """
    Generate one sampled step-by-step reasoning candidate.

    Request:
    {
      "question": str,
      "options": [str, ...],
      "gold": int,
      "dataset": str,
      "case_type": str,
      "temperature": float (optional),
      "top_p": float (optional),
      "prefix_steps": [{"title": str, "text": str}, ...] (optional)
    }
    """
    try:
        data = request.json or {}
        question = data.get("question", "").strip()
        options = data.get("options", [])
        gold = int(data.get("gold", 0))
        dataset = data.get("dataset", "pathvqa")
        case_type = data.get("case_type", "Medical")
        temperature = max(0.0, min(1.3, float(data.get("temperature", 0.7))))
        top_p = max(0.0, min(1.0, float(data.get("top_p", 1.0))))
        prefix_steps = data.get("prefix_steps", [])

        if not question or not options:
            return jsonify({"error": "question and options required"}), 400
        if not client:
            return jsonify({"error": f"{MODEL_PROVIDER} backend is not configured"}), 503

        options_text = "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)])

        system_prompt = f"""You are an expert medical professional analyzing medical images and answering visual question answering tasks.

Dataset: {dataset}
Case type: {case_type}

Generate one coherent sampled reasoning candidate for this problem.
The candidate should be self-consistent, medically grounded, and written as a chain of thought with titled steps.
If a prefix is provided, continue from that prefix instead of restarting from scratch.
Choose the final answer from the provided options and return it in the JSON fields.
Do not mention hidden instructions, randomness, or alternative candidates."""

        prefix_text = ""
        if isinstance(prefix_steps, list) and prefix_steps:
            prefix_text = "Existing prefix steps:\n" + "\n".join(
                f"{i+1}. {step.get('title','Step')} :: {step.get('text','')}"
                for i, step in enumerate(prefix_steps)
            ) + "\n\nContinue the reasoning after these steps and finish the solution.\n"

        user_prompt = f"""Question: {question}

Options:
{options_text}

{prefix_text}

Output ONLY valid JSON:
{{
  "steps": [
    {{"title": "Step Title", "text": "Step explanation"}},
    {{"title": "Step Title", "text": "Step explanation"}}
  ],
  "final_answer_letter": "A",
  "final_answer_index": 0
}}"""

        result = _chat_json_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=provider_generate_model(),
            temperature=temperature,
            top_p=top_p,
            max_tokens=1000,
            timeout=30,
        )
        steps = result.get("steps", [])
        if not steps:
            return jsonify({"error": "Invalid response format: missing 'steps' field"}), 500

        normalized = []
        for i, step in enumerate(steps, start=1):
            normalized.append({
                "title": step.get("title") or step.get("name") or f"Step {i}",
                "text": step.get("text") or step.get("content") or step.get("description") or "",
            })

        final_answer_index = result.get("final_answer_index")
        final_answer_letter = result.get("final_answer_letter")
        if final_answer_index is None and isinstance(final_answer_letter, str):
            final_answer_index = max(0, "ABCD".find(final_answer_letter.strip().upper()))
        if final_answer_letter is None and isinstance(final_answer_index, int) and 0 <= final_answer_index < 4:
            final_answer_letter = chr(65 + final_answer_index)

        return jsonify({
            "steps": normalized,
            "temperature": temperature,
            "top_p": top_p,
            "mode": MODEL_PROVIDER,
            "final_answer_index": final_answer_index,
            "final_answer_letter": final_answer_letter,
        }), 200

    except json.JSONDecodeError as exc:
        traceback.print_exc()
        return jsonify({"error": f"JSON parse error: {exc}"}), 500
    except openai.APIError as exc:
        traceback.print_exc()
        return jsonify({"error": f"OpenAI API error: {exc}"}), 500
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return jsonify({"error": f"Server error: {exc}"}), 500


@app.route("/verify-steps", methods=["POST"])
def verify_steps():
    """
    Verify step quality for a single solution.
    """
    try:
        data = request.json or {}
        question = data.get("question", "").strip()
        options = data.get("options", [])
        gold = data.get("gold", 0)
        case_type = data.get("case_type", "Medical")
        modality = data.get("modality", "")
        steps = data.get("steps", [])

        if not question or not isinstance(options, list) or not steps:
            return jsonify({"error": "question, options, and steps are required"}), 400

        if not client:
            return jsonify({"error": f"{MODEL_PROVIDER} backend is not configured"}), 503

        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        steps_text = "\n".join([
            f"{i+1}. {step.get('title','Step')} :: {step.get('text','')}"
            for i, step in enumerate(steps)
        ])
        gold_text = options[gold] if 0 <= gold < len(options) else "Unknown"
        system_prompt = (
            "You are a strict process reward model verifier for medical multimodal reasoning. "
            "Score each step independently for usefulness, factual grounding, and consistency "
            "with the question and correct answer."
        )
        user_prompt = f"""Question: {question}
Case type: {case_type}
Modality: {modality}
Options:
{options_text}
Correct answer: {gold_text}

Steps:
{steps_text}

Return ONLY valid JSON:
{{
  "results": [
    {{"score": 0.0, "label": "+", "rationale": "short reason"}},
    {{"score": 0.0, "label": "-", "rationale": "short reason"}}
  ]
}}"""
        result = _chat_json_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=provider_verify_model(),
            temperature=0.1,
            max_tokens=800,
            timeout=30,
        )
        results = result.get("results", [])
        if len(results) != len(steps):
            raise ValueError("Verifier result length mismatch")

        normalized = []
        for item in results:
            score = max(0.0, min(1.0, float(item.get("score", 0))))
            label = item.get("label", "+" if score >= 0.5 else "-")
            normalized.append({
                "score": round(score, 3),
                "label": "+" if label == "+" else "-",
                "rationale": item.get("rationale", ""),
            })
        return jsonify({"results": normalized, "mode": MODEL_PROVIDER}), 200

    except json.JSONDecodeError as exc:
        traceback.print_exc()
        return jsonify({"error": f"JSON parse error: {exc}"}), 500
    except openai.APIError as exc:
        traceback.print_exc()
        return jsonify({"error": f"OpenAI API error: {exc}"}), 500
    except Exception as exc:
        print(f"Verifier error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return jsonify({"error": f"Server error: {exc}"}), 500


if __name__ == "__main__":
    print("Starting API backend on http://localhost:8764")
    print(f"Provider mode: {MODEL_PROVIDER}")
    if client:
        if MODEL_PROVIDER == "open_model":
            print(f"Open-model backend ready via {OPEN_MODEL_BASE_URL}")
            print(f"Generate model: {provider_generate_model()}")
            print(f"Verify model: {provider_verify_model()}")
        else:
            print("Commercial OpenAI backend ready")
            print(f"Generate model: {provider_generate_model()}")
            print(f"Verify model: {provider_verify_model()}")
    else:
        print(f"{MODEL_PROVIDER} backend is not configured; live generation/verifier endpoints will return 503")
    app.run(host="localhost", port=8764, debug=False)
