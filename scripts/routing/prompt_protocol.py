"""Frozen task formatting and chat-template wrapping (protocol v1)."""

from __future__ import annotations

import hashlib
from typing import Any, TypedDict

GSM8K_SUFFIX = (
    "Reply with only the final numeric answer.\n"
    "Do not include units, commas, or explanations."
)
LETTER_SUFFIX = (
    "Reply with exactly one uppercase letter: A, B, C, or D.\n"
    "Do not provide any explanation."
)
BOOLQ_SUFFIX = "Reply with exactly one word: yes or no."

PROTOCOL_VERSION = "v1"
ARC_CHOICE_LETTERS = ("A", "B", "C", "D")


def normalize_arc_choice_labels(labels: list) -> list[str]:
    """Map ARC numeric choice labels (1–4) to A–D for protocol v1 letter scoring."""
    raw = [str(lab).strip().upper() for lab in labels]
    if len(raw) == 4 and all(lab in "1234" for lab in raw):
        return list(ARC_CHOICE_LETTERS)
    return raw


def arc_gold_letter(row: dict) -> str:
    """Canonical A–D gold for ARC letter-match scoring."""
    key = str(row["answerKey"]).strip().upper()
    if key in ARC_CHOICE_LETTERS:
        return key
    if key in "1234":
        return ARC_CHOICE_LETTERS[int(key) - 1]
    raise ValueError(f"Unexpected ARC answerKey: {row['answerKey']!r}")


class ChatPromptResult(TypedDict):
    chat_prompt: str
    prompt_tokens: int
    protocol_version: str
    tokenizer_id: str
    prompt_hash: str
    chat_template: bool


def format_gsm8k_question(question: str) -> str:
    return f"{question.strip()}\n\n{GSM8K_SUFFIX}"


def format_arc_question(row: dict) -> str:
    labels = normalize_arc_choice_labels(row["choices"]["label"])
    texts = row["choices"]["text"]
    opts = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, texts))
    return (
        f"Question:\n{row['question'].strip()}\n\n"
        f"Choices:\n{opts}\n\n"
        f"{LETTER_SUFFIX}"
    )


def format_mmlu_question(row: dict) -> str:
    labels = ("A", "B", "C", "D")
    choices = row["choices"]
    opts = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, choices))
    return (
        f"Question:\n{row['question'].strip()}\n\n"
        f"Choices:\n{opts}\n\n"
        f"{LETTER_SUFFIX}"
    )


def format_boolq_question(row: dict) -> str:
    return (
        f"Passage:\n{row['passage'].strip()}\n\n"
        f"Question:\n{row['question'].strip()}\n\n"
        f"{BOOLQ_SUFFIX}"
    )


def build_chat_prompt(
    tokenizer,
    user_content: str,
    *,
    messages: list[dict[str, Any]] | None = None,
) -> ChatPromptResult:
    """Wrap task-formatted user_content with the model's native chat template."""
    if messages is None:
        messages = [{"role": "user", "content": user_content}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer(chat_prompt, return_tensors="pt")["input_ids"]
    return {
        "chat_prompt": chat_prompt,
        "prompt_tokens": int(ids.shape[-1]),
        "protocol_version": PROTOCOL_VERSION,
        "tokenizer_id": getattr(tokenizer, "name_or_path", "unknown"),
        "prompt_hash": hashlib.sha256(chat_prompt.encode()).hexdigest(),
        "chat_template": getattr(tokenizer, "chat_template", None) is not None,
    }
