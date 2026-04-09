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
from collections import Counter
from pathlib import Path
from typing import Any

import openai
from flask import Flask, jsonify, request
from flask_cors import CORS

from medical_agent_graph import run_agent_graph
from medical_agent_policy import build_agent_policy
from medical_rag import retrieve_support
from medical_vector_store import embedding_backend_name, get_retrieval_runtime_status, qdrant_available

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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/").strip()
GEMINI_GENERATE_MODEL = os.getenv("GEMINI_GENERATE_MODEL", "gemini-2.5-flash")
GEMINI_VERIFY_MODEL = os.getenv("GEMINI_VERIFY_MODEL", GEMINI_GENERATE_MODEL)

OPEN_MODEL_BASE_URL = os.getenv("OPEN_MODEL_BASE_URL", "").strip()
OPEN_MODEL_API_KEY = os.getenv("OPEN_MODEL_API_KEY", "EMPTY").strip() or "EMPTY"
OPEN_MODEL_GENERATE_MODEL = os.getenv("OPEN_MODEL_GENERATE_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
OPEN_MODEL_VERIFY_MODEL = os.getenv("OPEN_MODEL_VERIFY_MODEL", OPEN_MODEL_GENERATE_MODEL)


def build_provider_client():
    if MODEL_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            return None
        return openai.OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)
    if MODEL_PROVIDER == "open_model":
        if not OPEN_MODEL_BASE_URL:
            return None
        return openai.OpenAI(api_key=OPEN_MODEL_API_KEY, base_url=OPEN_MODEL_BASE_URL)
    if not OPENAI_API_KEY:
        return None
    return openai.OpenAI(api_key=OPENAI_API_KEY)


def provider_generate_model() -> str:
    if MODEL_PROVIDER == "gemini":
        return GEMINI_GENERATE_MODEL
    if MODEL_PROVIDER == "open_model":
        return OPEN_MODEL_GENERATE_MODEL
    return OPENAI_GENERATE_MODEL


def provider_verify_model() -> str:
    if MODEL_PROVIDER == "gemini":
        return GEMINI_VERIFY_MODEL
    if MODEL_PROVIDER == "open_model":
        return OPEN_MODEL_VERIFY_MODEL
    return OPENAI_VERIFY_MODEL


client = None
_client_initialized = False


def get_provider_client():
    global client, _client_initialized
    if not _client_initialized:
        client = build_provider_client()
        _client_initialized = True
    return client


def provider_ready() -> bool:
    return get_provider_client() is not None


def _parse_json_text(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Common model issues: trailing commas, extra prose around a JSON object/array.
        candidate = text
        obj_start = candidate.find("{")
        obj_end = candidate.rfind("}")
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            candidate = candidate[obj_start:obj_end + 1]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        return json.loads(candidate)


def _chat_json_completion(*, messages: list[dict], model: str, temperature: float, max_tokens: int, timeout: int, top_p: float | None = None, attempts: int = 3) -> dict:
    provider_client = get_provider_client()
    if provider_client is None:
        raise RuntimeError(f"{MODEL_PROVIDER} backend is not configured")
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
            response = provider_client.chat.completions.create(**kwargs)
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


def build_retrieval_context(hits: list, *, include_answers: bool = False) -> str:
    if not hits:
        return "No retrieval context available."

    lines = []
    for idx, hit in enumerate(hits, start=1):
        options_text = "; ".join(
            f"{chr(65 + i)}. {option}" for i, option in enumerate(hit.options)
        )
        answer_line = f"\ngold_answer={hit.gold_text or 'Unknown'}" if include_answers else ""
        lines.append(
            f"[{idx}] dataset={hit.dataset} case_type={hit.case_type or 'Unknown'} "
            f"modality={hit.modality or 'Unknown'} score={hit.score}\n"
            f"question={hit.question}\n"
            f"options={options_text}"
            f"{answer_line}"
        )
    return "\n\n".join(lines)


def choose_retrieval_answer(question: str, options: list[str], hits: list) -> tuple[int, str]:
    if not options:
        return 0, "A"

    scores = [0.0 for _ in options]
    question_tokens = _tokenize(question)
    total_hit_score = sum(max(0.0, float(hit.score)) for hit in hits) or 1.0

    for hit_rank, hit in enumerate(hits):
        hit_question_tokens = _tokenize(hit.question)
        hit_option_tokens = _tokenize(" ".join(hit.options))
        rank_weight = 1.0 / (hit_rank + 1)
        score_weight = max(0.0, float(hit.score)) / total_hit_score
        support_weight = (0.7 * rank_weight) + (1.8 * score_weight)
        if hit.modality:
            support_weight += 0.15

        for option_idx, option in enumerate(options):
            option_tokens = _tokenize(option)
            option_text = str(option).strip().lower()
            option_overlap = len(option_tokens & hit_option_tokens)
            question_overlap = len(question_tokens & hit_question_tokens)
            if option_overlap:
                scores[option_idx] += support_weight * (1.0 + 0.4 * option_overlap)
            if option_text and option_text in " ".join(hit.options).lower():
                scores[option_idx] += support_weight * 0.5
            scores[option_idx] += support_weight * 0.08 * question_overlap

    answer_index = 0 if len(set(round(score, 6) for score in scores)) == 1 else max(range(len(options)), key=lambda idx: scores[idx])
    answer_letter = chr(65 + answer_index) if 0 <= answer_index < len(options) else "A"
    return answer_index, answer_letter


def fallback_agentic_response(
    *,
    question: str,
    options: list[str],
    dataset: str,
    case_type: str,
    modality: str,
    policy,
    hits: list,
) -> dict:
    answer_index, answer_letter = choose_retrieval_answer(question, options, hits)
    answer_text = options[answer_index] if 0 <= answer_index < len(options) else ""

    query_tokens = _tokenize(question)
    top_hits = hits[:3] if hits else []
    evidence_lines: list[str] = []

    for hit in top_hits:
        hit_q = (hit.question or "").strip()
        hit_tokens = _tokenize(" ".join([hit.question, hit.case_type, hit.modality, " ".join(hit.options)]))
        overlap = len(query_tokens & hit_tokens)
        evidence_lines.append(
            f"- [{hit.dataset}:{hit.case_id}] score={hit.score:.3f}, overlap={overlap}, "
            f"case={hit.case_type or 'n/a'}, modality={hit.modality or 'n/a'} :: {hit_q[:120]}"
        )

    confidence = "low"
    if top_hits:
        top_score = float(getattr(top_hits[0], "score", 0.0))
        if top_score >= 0.82:
            confidence = "high"
        elif top_score >= 0.70:
            confidence = "medium"

    steps = [
        {
            "title": "Route Case",
            "text": (
                f"Route this case to {policy.specialist} for a {policy.question_type} question in "
                f"{case_type or 'medical'}"
                + (f" with modality {modality}." if modality else ".")
            ),
        },
        {
            "title": "Retrieve Support",
            "text": (
                f"Retrieved {len(hits)} nearby examples from sources {', '.join(policy.retrieval_sources)} "
                f"within datasets {', '.join(policy.allowed_datasets)}.\n"
                + "\n".join(evidence_lines)
                if hits
                else "No close retrieved examples were found under the dataset-aware retrieval policy."
            ),
        },
        {
            "title": "Choose Answer",
            "text": (
                f"Selected option {answer_letter} ({answer_text}) from retrieval-weighted evidence "
                f"with {confidence} confidence; this remains a grounded heuristic, not a copied gold label."
                if hits
                else f"Default to option {answer_letter} ({answer_text}) because no support cases were found."
            ),
        },
    ]

    return {
        "steps": steps,
        "final_answer_index": answer_index,
        "final_answer_letter": answer_letter,
        "mode": "retrieval_fallback",
    }


def verify_or_heuristic(
    *,
    question: str,
    options: list[str],
    gold: int,
    case_type: str,
    modality: str,
    steps: list[dict],
) -> tuple[list[dict], str]:
    if MODEL_PROVIDER == "gemini":
        return [
            heuristic_step_verdict(
                question,
                options,
                gold,
                step.get("title", f"Step {idx+1}"),
                step.get("text", ""),
                idx,
                len(steps),
            )
            for idx, step in enumerate(steps)
        ], "heuristic"

    if get_provider_client():
        try:
            options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            steps_text = "\n".join([
                f"{i+1}. {step.get('title','Step')} :: {step.get('text','')}"
                for i, step in enumerate(steps)
            ])
            gold_text = options[gold] if 0 <= gold < len(options) else "Unknown"
            result = _chat_json_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict process reward model verifier for medical multimodal reasoning. "
                            "Score each step independently for usefulness, factual grounding, and consistency."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n"
                            f"Case type: {case_type}\n"
                            f"Modality: {modality}\n"
                            f"Options:\n{options_text}\n"
                            f"Correct answer: {gold_text}\n\n"
                            f"Steps:\n{steps_text}\n\n"
                            "Return ONLY valid JSON:\n"
                            "{\n"
                            '  "results": [\n'
                            '    {"score": 0.0, "label": "+", "rationale": "short reason"}\n'
                            "  ]\n"
                            "}"
                        ),
                    },
                ],
                model=provider_verify_model(),
                temperature=0.1,
                max_tokens=800,
                timeout=30,
            )
            return result.get("results", []), MODEL_PROVIDER
        except Exception:
            pass

    return [
        heuristic_step_verdict(
            question,
            options,
            gold,
            step.get("title", f"Step {idx+1}"),
            step.get("text", ""),
            idx,
            len(steps),
        )
        for idx, step in enumerate(steps)
    ], "heuristic"


@app.route("/health", methods=["GET"])
def health():
    retrieval_status = get_retrieval_runtime_status()
    return jsonify({
        "status": "ok",
        "provider": MODEL_PROVIDER,
        "client_ready": provider_ready(),
        "api_key_configured": (
            bool(OPENAI_API_KEY)
            if MODEL_PROVIDER == "openai"
            else bool(GEMINI_API_KEY)
            if MODEL_PROVIDER == "gemini"
            else bool(OPEN_MODEL_BASE_URL)
        ),
        "mode": MODEL_PROVIDER if get_provider_client() else f"{MODEL_PROVIDER}_not_ready",
        "generate_model": provider_generate_model(),
        "verify_model": provider_verify_model(),
        "embedding_backend": retrieval_status.get("embedding_backend"),
        "retrieval_runtime": retrieval_status,
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
        image_url = str(data.get("image_url", "")).strip()
        temperature = max(0.0, min(1.3, float(data.get("temperature", 0.7))))
        top_p = max(0.0, min(1.0, float(data.get("top_p", 1.0))))
        prefix_steps = data.get("prefix_steps", [])

        if not question or not options:
            return jsonify({"error": "question and options required"}), 400
        if not get_provider_client():
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

        user_content: list[dict[str, Any]] = []
        if image_url:
            user_content.append({"type": "image_url", "image_url": {"url": image_url}})
        user_content.append({"type": "text", "text": user_prompt})

        result = _chat_json_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
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
        try:
            fallback = fallback_agentic_response(
                question=question,
                options=[str(option) for option in options],
                dataset=dataset,
                case_type=case_type,
                modality=modality,
                policy=policy if "policy" in locals() else build_agent_policy(
                    dataset=dataset,
                    case_type=case_type,
                    modality=modality,
                    question=question,
                    options=[str(option) for option in options],
                ),
                hits=hits if "hits" in locals() else [],
            )
            scores, score_mode = verify_or_heuristic(
                question=question,
                options=[str(option) for option in options],
                gold=gold,
                case_type=case_type,
                modality=modality,
                steps=fallback["steps"],
            )
            return jsonify({
                "router": {
                    "dataset": dataset,
                    "case_type": case_type,
                    "modality": modality,
                    "question_type": policy.question_type if "policy" in locals() else "unknown",
                    "specialist": policy.specialist if "policy" in locals() else "fallback_reasoner",
                },
                "retrieval_sources": list(policy.retrieval_sources) if "policy" in locals() else [],
                "retrieval_mode": graph_state.get("retrieval_mode", "fallback") if "graph_state" in locals() else "fallback",
                "vector_db_ready": qdrant_available(),
                "retrieval_hits": [],
                "steps": fallback["steps"],
                "step_scores": scores,
                "average_step_score": round(sum(float(item.get("score", 0.0)) for item in scores) / len(scores), 3) if scores else 0.0,
                "final_answer_index": fallback["final_answer_index"],
                "final_answer_letter": fallback["final_answer_letter"],
                "mode": "retrieval_fallback",
                "score_mode": score_mode,
            }), 200
        except Exception:
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

        if not get_provider_client():
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


@app.route("/agent-answer", methods=["POST"])
def agent_answer():
    """
    Agent + RAG experimental endpoint.

    Request:
    {
      "question": str,
      "options": [str, ...],
      "gold": int (optional),
      "dataset": str (optional),
      "case_type": str (optional),
      "modality": str (optional),
      "image_url": str (optional),
      "top_k": int (optional)
    }
    """
    try:
        data = request.json or {}
        question = str(data.get("question", "")).strip()
        options = data.get("options", [])
        gold = int(data.get("gold", 0))
        dataset = str(data.get("dataset", "")).strip()
        case_type = str(data.get("case_type", "Medical")).strip()
        modality = str(data.get("modality", "")).strip()
        image_url = str(data.get("image_url", "")).strip()
        current_case_id = str(data.get("id", "")).strip()
        top_k = max(1, min(5, int(data.get("top_k", 3))))

        if not question or not isinstance(options, list) or not options:
            return jsonify({"error": "question and options are required"}), 400

        graph_state = run_agent_graph(
            {
                "id": current_case_id,
                "question": question,
                "options": [str(option) for option in options],
                "dataset": dataset,
                "case_type": case_type,
                "modality": modality,
                "top_k": top_k,
            }
        )
        policy = graph_state["policy"]
        hits = graph_state.get("hits", [])
        reranked_hits = graph_state.get("reranked_hits", [])
        document_hits = graph_state.get("document_hits", [])
        retrieval_debug = graph_state.get("retrieval_debug", {})
        retrieval_context = build_retrieval_context(
            hits,
            include_answers=policy.allow_answer_exposure,
        )
        if document_hits:
            retrieval_context += "\n\nDocument support:\n" + "\n\n".join(
                f"[DOC {idx+1}] domain={doc['domain']} modality={doc['modality']} score={doc['score']}\n"
                f"title={doc['title']}\n"
                f"text={doc['text']}"
                for idx, doc in enumerate(document_hits)
            )

        if get_provider_client():
            try:
                options_text = "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)])
                system_prompt = (
                    "You are a medical multi-agent coordinator. "
                    "Follow this order: (1) draft concise step reasoning from the case/question, "
                    "(2) ground/refine the reasoning with retrieved support and document evidence, "
                    "(3) output final judged answer. "
                    "Avoid unsupported claims and keep steps clinically coherent."
                )
                user_prompt = f"""Question: {question}
Dataset: {dataset or 'unknown'}
Case type: {case_type}
Modality: {modality or 'unknown'}
Question type: {policy.question_type}
Assigned specialist: {policy.specialist}

Options:
{options_text}

Retrieved support:
{retrieval_context}

Return ONLY valid JSON:
{{
  "steps": [
    {{"title": "Route", "text": "..." }},
    {{"title": "Retrieve", "text": "..." }},
    {{"title": "Reason", "text": "..." }},
    {{"title": "Judge", "text": "..." }}
  ],
  "final_answer_letter": "A",
  "final_answer_index": 0
}}"""
                user_content: list[dict[str, Any]] = []
                if image_url:
                    user_content.append({"type": "image_url", "image_url": {"url": image_url}})
                user_content.append({"type": "text", "text": user_prompt})
                result = _chat_json_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    model=provider_generate_model(),
                    temperature=0.3,
                    top_p=0.9,
                    max_tokens=1000,
                    timeout=30,
                )
                steps = result.get("steps", [])
                normalized_steps = [
                    {
                        "title": step.get("title") or f"Step {idx+1}",
                        "text": step.get("text") or "",
                    }
                    for idx, step in enumerate(steps)
                ]
                answer_index = result.get("final_answer_index")
                answer_letter = result.get("final_answer_letter")
                if answer_index is None and isinstance(answer_letter, str):
                    answer_index = max(0, "ABCD".find(answer_letter.strip().upper()))
                if answer_letter is None and isinstance(answer_index, int) and 0 <= answer_index < len(options):
                    answer_letter = chr(65 + answer_index)
                pipeline_result = {
                    "steps": normalized_steps,
                    "final_answer_index": answer_index,
                    "final_answer_letter": answer_letter,
                    "mode": MODEL_PROVIDER,
                }
            except Exception:
                pipeline_result = fallback_agentic_response(
                    question=question,
                    options=[str(option) for option in options],
                    dataset=dataset,
                    case_type=case_type,
                    modality=modality,
                    policy=policy,
                    hits=hits,
                )
        else:
            pipeline_result = fallback_agentic_response(
                question=question,
                options=[str(option) for option in options],
                dataset=dataset,
                case_type=case_type,
                modality=modality,
                policy=policy,
                hits=hits,
            )

        scores, score_mode = verify_or_heuristic(
            question=question,
            options=[str(option) for option in options],
            gold=gold,
            case_type=case_type,
            modality=modality,
            steps=pipeline_result["steps"],
        )

        avg_score = round(
            sum(float(item.get("score", 0.0)) for item in scores) / len(scores),
            3,
        ) if scores else 0.0
        top_hit_score = max((float(hit.score) for hit in hits), default=0.0)

        retrieval_items = [
            {
                "dataset": hit.dataset,
                "id": hit.case_id,
                "case_type": hit.case_type,
                "modality": hit.modality,
                "question": hit.question,
                "score": hit.score,
                "image_url": hit.image_url,
                "rerank_score": next(
                    (item["rerank_score"] for item in reranked_hits if item["hit"].case_id == hit.case_id),
                    None,
                ),
                "rerank_reasons": next(
                    (item["reasons"] for item in reranked_hits if item["hit"].case_id == hit.case_id),
                    {},
                ),
            }
            for hit in hits
        ]
        retrieval_runtime = get_retrieval_runtime_status()

        return jsonify(
            {
                "router": {
                    "dataset": policy.dataset,
                    "case_type": policy.case_type,
                    "modality": policy.modality,
                    "question_type": policy.question_type,
                    "specialist": policy.specialist,
                },
                "retrieval_sources": list(policy.retrieval_sources),
                "retrieval_mode": graph_state.get("retrieval_mode", "lexical"),
                "retrieval_debug": retrieval_debug,
                "embedding_backend": embedding_backend_name(),
                "retrieval_runtime": {
                    "auto_warm": retrieval_runtime.get("auto_warm"),
                    "warmed": retrieval_runtime.get("warmed"),
                    "qdrant_mode": retrieval_runtime.get("qdrant_mode"),
                    "qdrant_ready": retrieval_runtime.get("qdrant_ready"),
                },
                "vector_db_ready": qdrant_available(),
                "retrieval_hits": retrieval_items,
                "retrieval_gated": False,
                "retrieval_top_hit_score": top_hit_score,
                "document_hits": document_hits,
                "steps": pipeline_result["steps"],
                "step_scores": scores,
                "average_step_score": avg_score,
                "final_answer_index": pipeline_result["final_answer_index"],
                "final_answer_letter": pipeline_result["final_answer_letter"],
                "mode": pipeline_result["mode"],
                "score_mode": score_mode,
            }
        ), 200
    except json.JSONDecodeError as exc:
        traceback.print_exc()
        try:
            safe_policy = policy if "policy" in locals() else build_agent_policy(
                dataset=dataset,
                case_type=case_type,
                modality=modality,
                question=question,
                options=[str(option) for option in options],
            )
            safe_hits = hits if "hits" in locals() else []
            fallback = fallback_agentic_response(
                question=question,
                options=[str(option) for option in options],
                dataset=dataset,
                case_type=case_type,
                modality=modality,
                policy=safe_policy,
                hits=safe_hits,
            )
            scores, score_mode = verify_or_heuristic(
                question=question,
                options=[str(option) for option in options],
                gold=gold,
                case_type=case_type,
                modality=modality,
                steps=fallback["steps"],
            )
            return jsonify({
                "router": {
                    "dataset": safe_policy.dataset,
                    "case_type": safe_policy.case_type,
                    "modality": safe_policy.modality,
                    "question_type": safe_policy.question_type,
                    "specialist": safe_policy.specialist,
                },
                "retrieval_sources": list(safe_policy.retrieval_sources),
                "retrieval_mode": graph_state.get("retrieval_mode", "fallback") if "graph_state" in locals() else "fallback",
                "vector_db_ready": qdrant_available(),
                "retrieval_hits": [],
                "steps": fallback["steps"],
                "step_scores": scores,
                "average_step_score": round(sum(float(item.get("score", 0.0)) for item in scores) / len(scores), 3) if scores else 0.0,
                "final_answer_index": fallback["final_answer_index"],
                "final_answer_letter": fallback["final_answer_letter"],
                "mode": "retrieval_fallback",
                "score_mode": score_mode,
            }), 200
        except Exception:
            return jsonify({"error": f"JSON parse error: {exc}"}), 500
    except openai.APIError as exc:
        traceback.print_exc()
        return jsonify({"error": f"OpenAI API error: {exc}"}), 500
    except Exception as exc:
        print(f"Agent answer error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return jsonify({"error": f"Server error: {exc}"}), 500


if __name__ == "__main__":
    print("Starting API backend on http://localhost:8764")
    print(f"Provider mode: {MODEL_PROVIDER}")
    retrieval_status = get_retrieval_runtime_status()
    print(f"Embedding backend: {retrieval_status.get('embedding_backend')}")
    print(f"Qdrant mode: {retrieval_status.get('qdrant_mode')}")
    if retrieval_status.get("warmed"):
        print(f"RAG warmup complete: {retrieval_status.get('cases')} cases / {retrieval_status.get('vectors')} vectors")
    if get_provider_client():
        if MODEL_PROVIDER == "gemini":
            print(f"Gemini backend ready via {GEMINI_BASE_URL}")
            print(f"Generate model: {provider_generate_model()}")
            print(f"Verify model: {provider_verify_model()}")
        elif MODEL_PROVIDER == "open_model":
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
