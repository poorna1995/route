"""Oracle inference: MCQ prompts, grading, and GPU forward passes."""

from __future__ import annotations

import gc
import os
import re
import time
from typing import Any

from llm_routing.corpus import Query, QueryResult
from llm_routing.signals.psi.protocol import (
    build_protocol_artifact,
    capture_protocol_trace,
    inference_capture_metadata,
    trace_generated_text,
)

_LETTER = re.compile(r"^([A-Za-z])\s*$")
_FIRST_LETTER = re.compile(r"\b([A-Za-z])\b")


# --- MCQ prompt + grading ---


def _choice_index(letter: str, n: int) -> int | None:
    idx = ord(letter.upper()) - ord("A")
    return idx if 0 <= idx < n else None


def format_choices(choices: tuple[str, ...]) -> str:
    return "\n".join(f"{chr(ord('A') + i)}. {c.strip()}" for i, c in enumerate(choices))


def format_question(query: Query) -> str:
    subject = query.metadata.get("subject")
    text = query.text.strip()
    return f"Subject: {subject}\n\n{text}" if subject else text


def render_user_message(query: Query, protocol: dict[str, Any]) -> str:
    return protocol["user_template"].format(
        question=format_question(query),
        choices=format_choices(query.choices),
    )


def build_messages(query: Query, protocol: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": protocol["system_prompt"]},
        {"role": "user", "content": render_user_message(query, protocol)},
    ]


def parse_answer(text: str, n_choices: int) -> int | None:
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if m := _LETTER.match(line):
            return _choice_index(m.group(1), n_choices)
        if m := _FIRST_LETTER.search(line):
            return _choice_index(m.group(1), n_choices)
        break
    return None


def grade_query(query: Query, model_output: str) -> int:
    pred = parse_answer(model_output, len(query.choices))
    return int(pred == query.answer_index) if pred is not None else 0


# --- GPU inference ---


def huggingface_read_token() -> str | None:
    """HF read token for gated models (Llama). Set HF_TOKEN on RunPod."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def _huggingface_load_kwargs(model_id: str) -> dict[str, Any]:
    token = huggingface_read_token()
    if token is None and model_id.startswith("meta-llama/"):
        raise RuntimeError(
            "HF_TOKEN is not set. Gated Llama weights require a Hugging Face read token.\n"
            "  1. Accept licenses: meta-llama/Llama-3.2-3B-Instruct and Llama-3.1-8B-Instruct\n"
            "  2. export HF_TOKEN=hf_...\n"
            "  3. Re-run (or: huggingface-cli login --token \"$HF_TOKEN\")"
        )
    return {"token": token} if token else {}


def run_oracle_inference(
    model_id: str,
    queries: list[Query],
    protocol: dict[str, Any],
    *,
    device: str | None = None,
) -> list[QueryResult]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    max_tokens = int(protocol["decoding"].get("max_tokens", 16))

    hf_load_kwargs = _huggingface_load_kwargs(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, **hf_load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        **hf_load_kwargs,
    )
    if device != "cuda":
        model.to(device)

    capture_meta = inference_capture_metadata(model, tokenizer, torch_dtype=dtype)
    results: list[QueryResult] = []
    for query in queries:
        messages = build_messages(query, protocol)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        start_time = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        latency_ms = (time.perf_counter() - start_time) * 1000
        prompt_len = inputs["input_ids"].shape[1]
        new_token_ids = outputs.sequences[0, prompt_len:]
        generated_token_ids = [int(token_id) for token_id in new_token_ids.tolist()]
        trace = capture_protocol_trace(
            outputs,
            tokenizer,
            generated_token_ids,
            protocol,
            prompt=prompt,
            n_choices=len(query.choices),
            model=model,
        )
        generated_text = trace_generated_text(trace)
        results.append(
            QueryResult(
                query_id=query.query_id,
                model=model_id,
                raw_output=generated_text,
                parsed_answer=parse_answer(generated_text, len(query.choices)),
                correct=grade_query(query, generated_text),
                latency_ms=latency_ms,
                token_count=len(generated_token_ids),
                model_response=build_protocol_artifact(
                    protocol["protocol_version"],
                    trace,
                    model_id=model_id,
                    prompt=prompt,
                    prompt_token_count=prompt_len,
                    **capture_meta,
                ),
            )
        )

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results
