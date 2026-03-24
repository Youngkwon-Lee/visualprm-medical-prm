#!/usr/bin/env python3
"""OpenAI-compatible chat completions server for Qwen Vision models on RunPod.

Supports Qwen3-VL-30B and other vision models.
This server is intended for the existing `open_model` backend path:
`api_backend.py` -> OpenAI-compatible endpoint -> local Qwen model.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch
from flask import Flask, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3-VL-30B-Instruct")
HOST = os.getenv("QWEN_SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("QWEN_SERVER_PORT", "8000"))
MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "768"))
LOAD_IN_4BIT = os.getenv("QWEN_LOAD_IN_4BIT", "0") == "1"

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=LOAD_IN_4BIT,
)
model.eval()


def flatten_messages(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            content = "\n".join(text_parts)
        parts.append(f"{role.upper()}:\n{content}")
    return "\n\n".join(parts)


def generate_text(prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
    chat_messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "mode": "qwen_local_server",
            "model": MODEL_ID,
            "load_in_4bit": LOAD_IN_4BIT,
        }
    )


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.get_json(force=True) or {}
    messages = data.get("messages", [])
    model_name = data.get("model", MODEL_ID)
    temperature = float(data.get("temperature", 0.7))
    top_p = float(data.get("top_p", 1.0))
    max_tokens = int(data.get("max_tokens", MAX_NEW_TOKENS))

    prompt = flatten_messages(messages)
    content = generate_text(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    return jsonify(
        {
            "id": "chatcmpl-qwen-local",
            "object": "chat.completion",
            "created": 0,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
    )


if __name__ == "__main__":
    print(f"Starting local Qwen OpenAI-compatible server on http://{HOST}:{PORT}")
    print(f"Model: {MODEL_ID}")
    app.run(host=HOST, port=PORT, debug=False)
