#!/usr/bin/env python3
"""OpenAI-compatible chat completions server for Qwen text models on RunPod.

This server is intended for the existing `open_model` backend path:
`api_backend.py` -> OpenAI-compatible endpoint -> local Qwen model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from flask import Flask, jsonify, request
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration


MODEL_ID = os.getenv("QWEN_SERVE_MODEL_ID", os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"))
HOST = os.getenv("QWEN_SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("QWEN_SERVER_PORT", "8000"))
MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "768"))
LOAD_IN_4BIT = os.getenv("QWEN_LOAD_IN_4BIT", "0") == "1"
IS_VLM = "VL" in MODEL_ID.upper()

app = Flask(__name__)

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

if IS_VLM:
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer = None
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
else:
    processor = None
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
        load_in_4bit=LOAD_IN_4BIT,
    )
model.eval()


def resolve_image_path(url: str) -> str:
    if url.startswith("file://"):
        return url[len("file://") :]
    if url.startswith(("http://", "https://")):
        return url
    path = Path(url)
    if path.is_absolute():
        return str(path)
    return str((Path(__file__).resolve().parent / path).resolve())


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            items: list[dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    items.append({"type": "text", "text": item.get("text", "")})
                elif item_type == "image_url":
                    image_url = item.get("image_url", {})
                    if isinstance(image_url, dict):
                        image_url = image_url.get("url", "")
                    if image_url:
                        items.append({"type": "image", "image": resolve_image_path(str(image_url))})
        else:
            items = [{"type": "text", "text": str(content)}]
        normalized.append({"role": role, "content": items})
    return normalized


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


def generate_vlm(messages: list[dict[str, Any]], temperature: float, top_p: float, max_tokens: int) -> str:
    normalized = normalize_messages(messages)
    chat_template = processor.apply_chat_template(
        normalized,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(normalized)
    inputs = processor(
        text=[chat_template],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    do_sample = temperature > 0
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            max_new_tokens=max_tokens,
        )
    trimmed_ids = [
        output_ids[input_ids.shape[0] :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


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

    if IS_VLM:
        content = generate_vlm(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    else:
        prompt = "\n\n".join(
            f"{msg.get('role', 'user').upper()}:\n{msg.get('content', '')}"
            for msg in messages
        )
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
