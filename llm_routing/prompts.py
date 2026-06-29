"""Render MCQ prompts and grade letter answers."""

from __future__ import annotations

import re
from typing import Any

from llm_routing.corpus import Query

_LETTER = re.compile(r"^([A-Za-z])\s*$")
_FIRST_LETTER = re.compile(r"\b([A-Za-z])\b")


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
