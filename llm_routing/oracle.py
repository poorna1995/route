"""Run one model on queries; write QueryResult rows."""

from __future__ import annotations

import gc
import os
import time
from typing import Any

import torch

from llm_routing.corpus import Query, QueryResult
from llm_routing.prompts import build_messages, grade_query, parse_answer


def hf_token() -> str | None:
    """HF read token for gated models (Llama). Set HF_TOKEN on RunPod."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def _load_kwargs(model_id: str) -> dict[str, Any]:
    token = hf_token()
    if token is None and model_id.startswith("meta-llama/"):
        raise RuntimeError(
            "HF_TOKEN is not set. Gated Llama weights require a Hugging Face read token.\n"
            "  1. Accept licenses: meta-llama/Llama-3.2-3B-Instruct and Llama-3.1-8B-Instruct\n"
            "  2. export HF_TOKEN=hf_...\n"
            "  3. Re-run (or: huggingface-cli login --token \"$HF_TOKEN\")"
        )
    return {"token": token} if token else {}


def run_model_mock(
    model_id: str,
    queries: list[Query],
    protocol: dict[str, Any],
) -> list[QueryResult]:
    """Deterministic fake outputs for local pipeline checks (no GPU/HF weights)."""
    del protocol
    is_hi = "8B" in model_id or "70B" in model_id
    results: list[QueryResult] = []
    for i, query in enumerate(queries):
        lo_ok = i % 2 == 0
        hi_ok = i % 3 != 0
        correct = hi_ok if is_hi else lo_ok
        pred = query.answer_index if correct else (query.answer_index + 1) % len(query.choices)
        results.append(
            QueryResult(
                query_id=query.query_id,
                model=model_id,
                raw_output=chr(ord("A") + pred),
                parsed_answer=pred,
                correct=int(correct),
                latency_ms=1.0,
                token_count=1,
            )
        )
    return results


def run_model(
    model_id: str,
    queries: list[Query],
    protocol: dict[str, Any],
    *,
    device: str | None = None,
) -> list[QueryResult]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    max_tokens = int(protocol["decoding"].get("max_tokens", 16))

    hf_kw = _load_kwargs(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, **hf_kw)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        **hf_kw,
    )
    if device != "cuda":
        model.to(device)

    results: list[QueryResult] = []
    for query in queries:
        messages = build_messages(query, protocol)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - t0) * 1000
        new_ids = out[0, inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        results.append(
            QueryResult(
                query_id=query.query_id,
                model=model_id,
                raw_output=text,
                parsed_answer=parse_answer(text, len(query.choices)),
                correct=grade_query(query, text),
                latency_ms=latency_ms,
                token_count=int(new_ids.shape[0]),
            )
        )

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results
