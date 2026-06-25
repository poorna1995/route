"""Hugging Face model loading, device selection, and memory cleanup."""

from __future__ import annotations

import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from routing.constants import DEFAULT_TOKENIZER_ID


def load_tokenizer(model_id: str = DEFAULT_TOKENIZER_ID):
    """Load HF tokenizer only (no model weights) — for query complexity features."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str, requested: str | None) -> torch.dtype:
    if requested:
        return getattr(torch, requested)
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def resolve_input_device(model) -> torch.device:
    if hasattr(model, "hf_device_map"):
        bad = {d for d in model.hf_device_map.values() if d in ("meta", "disk")}
        if bad:
            raise RuntimeError(
                "Model has weights on meta/disk offload — generation is not supported. "
                "Use --device cuda or --device mps on a machine with enough memory."
            )
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


def load_model_and_tokenizer(
    model_id: str,
    *,
    device: str = "auto",
    dtype: str | None = None,
):
    device = pick_device(device)
    torch_dtype = pick_dtype(device, dtype)
    tokenizer = load_tokenizer(model_id)
    print(f"loading {model_id} → {device} ({torch_dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return model, tokenizer, device


def release_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
