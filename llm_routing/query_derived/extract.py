"""Per-query extraction: φ_load and φ_ambiguity (no corpus fitting)."""

from __future__ import annotations

import re
import statistics
import zlib
from typing import Any

from llm_routing.query_derived.config import load_section

_WORD = re.compile(r"\b[\w']+\b", re.UNICODE)


# --- text helpers ---


def word_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD.finditer(text)]


def mattr(tokens: list[str], window: int) -> float:
    n = len(tokens)
    if n == 0:
        return float("nan")
    if n < window:
        return len(set(tokens)) / n
    ratios: list[float] = []
    for start in range(n - window + 1):
        chunk = tokens[start : start + window]
        ratios.append(len(set(chunk)) / window)
    return statistics.fmean(ratios)


def compression_ratio(text: str, level: int) -> float:
    raw = text.encode("utf-8")
    if not raw:
        return float("nan")
    return len(zlib.compress(raw, level=level)) / len(raw)


def word_jaccard(a: str, b: str) -> float:
    sa, sb = set(word_tokens(a)), set(word_tokens(b))
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


# --- token counter ---


class TokenCounter:
    """HF tokenizer optional (count only, no forward pass)."""

    def __init__(self, tokenizer_id: str | None = None) -> None:
        self._tokenizer_id = tokenizer_id
        self._tokenizer: Any = None

    def _hf(self) -> Any:
        if self._tokenizer is None:
            if not self._tokenizer_id:
                raise RuntimeError("HF tokenizer requested but tokenizer_id is unset")
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_id)
        return self._tokenizer

    def count(self, text: str) -> int:
        if self._tokenizer_id:
            return len(self._hf().encode(text, add_special_tokens=False))
        return len(word_tokens(text))

    def chat_token_count(self, messages: list[dict[str, str]]) -> int:
        if self._tokenizer_id:
            tok = self._hf()
            if getattr(tok, "chat_template", None):
                return len(
                    tok.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True
                    )
                )
            return sum(self.count(m["content"]) for m in messages)
        return sum(self.count(m["content"]) for m in messages)


# --- canonical prompt ---


def canonical_user(query: Any, protocol: dict[str, Any]) -> str:
    from llm_routing.prompts import render_user_message

    return render_user_message(query, protocol)


# --- φ_load ---


def extract_structural(
    query: Any,
    canonical: str,
    counter: TokenCounter,
) -> dict[str, float | int]:
    from llm_routing.prompts import format_question

    question = format_question(query)
    opt_lens = [counter.count(c) for c in query.choices]
    q_len = counter.count(question)
    opt_sum = sum(opt_lens) or 1
    return {
        "prompt_token_len": counter.count(canonical),
        "question_token_len": q_len,
        "option_count": len(query.choices),
        "mean_option_token_len": statistics.fmean(opt_lens) if opt_lens else 0.0,
        "std_option_token_len": statistics.pstdev(opt_lens) if len(opt_lens) > 1 else 0.0,
        "question_option_ratio": q_len / opt_sum,
    }


def extract_lexical(canonical: str, config: dict[str, Any]) -> dict[str, float]:
    lcfg = load_section(config)
    tokens = word_tokens(canonical)
    window = int(lcfg.get("mattr_window", 50))
    level = int(lcfg.get("zlib_level", 6))
    return {
        "mattr": mattr(tokens, window),
        "compression_ratio": compression_ratio(canonical, level),
    }


def extract_load(
    query: Any,
    canonical: str,
    counter: TokenCounter,
    config: dict[str, Any],
) -> dict[str, float | int]:
    out = extract_structural(query, canonical, counter)
    out.update(extract_lexical(canonical, config))
    return out


# --- φ_ambiguity ---


def extract_ambiguity(query: Any) -> dict[str, float]:
    stem = query.text.strip()
    choices = [c.strip() for c in query.choices]
    stem_j = [word_jaccard(stem, c) for c in choices]
    pairs: list[float] = []
    for i in range(len(choices)):
        for j in range(i + 1, len(choices)):
            pairs.append(word_jaccard(choices[i], choices[j]))
    char_lens = [len(c) for c in choices]
    return {
        "stem_choice_overlap_max": max(stem_j) if stem_j else 0.0,
        "stem_choice_overlap_mean": statistics.fmean(stem_j) if stem_j else 0.0,
        "choice_choice_overlap": statistics.fmean(pairs) if pairs else 0.0,
        "choice_length_range": (max(char_lens) - min(char_lens)) if char_lens else 0.0,
    }


extract_mcq = extract_ambiguity
