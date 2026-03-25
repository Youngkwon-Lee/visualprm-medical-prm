#!/usr/bin/env python3
"""Tiny OpenAI-compatible mock server for local contract testing."""

from __future__ import annotations

import json
import re

from flask import Flask, jsonify, request

app = Flask(__name__)


def make_generate_response() -> dict:
    return {
        "id": "chatcmpl-mock-generate",
        "object": "chat.completion",
        "created": 0,
        "model": "mock-qwen",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "steps": [
                                {"title": "Visual Survey", "text": "Review the image and identify the main medical finding relevant to the question."},
                                {"title": "Option Comparison", "text": "Compare the observed finding against the answer choices and remove inconsistent options."},
                                {"title": "Final Decision", "text": "Select the answer choice that best matches the image and question."},
                            ],
                            "final_answer_letter": "A",
                            "final_answer_index": 0,
                        }
                    ),
                },
                "finish_reason": "stop",
            }
        ],
    }


def make_verify_response(prompt: str) -> dict:
    step_count = len(re.findall(r"^\d+\.\s", prompt, flags=re.MULTILINE))
    if step_count <= 0:
        step_count = 3
    results = [
        {"score": 1.0, "label": "+", "rationale": "Mock verifier: step is accepted."}
        for _ in range(step_count)
    ]
    return {
        "id": "chatcmpl-mock-verify",
        "object": "chat.completion",
        "created": 0,
        "model": "mock-qwen",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"results": results}),
                },
                "finish_reason": "stop",
            }
        ],
    }


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json(force=True, silent=True) or {}
    messages = data.get("messages", [])
    prompt = "\n".join(str(msg.get("content", "")) for msg in messages if isinstance(msg, dict))

    if '"results"' in prompt or "Verify step quality" in prompt or "Score each step independently" in prompt:
        return jsonify(make_verify_response(prompt))
    return jsonify(make_generate_response())


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "mode": "mock_open_model"}), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
