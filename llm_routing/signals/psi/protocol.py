"""Protocol traces (GPU capture) and ψ metrics (CPU) for model_response artifacts.

Immutable trace (never discard — GPU is expensive, storage is cheap):
  generated_text, generated_token, finish_reason
  generated_token_ids, generated_token_logprobs, generated_logits (sparse top-k per step)
  candidate_token_logprobs, predicted_letter, n_choices  (mcq_letter)

Immutable artifact envelope (artifact_version + extractor_version + feature_version + metrics_version):
  model_id, tokenizer_id, prompt_sha256, model_revision,
  transformers_version, torch_version, dtype, capture_time, trace
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

# Per generated step, keep this many vocab entries in generated_logits (sparse, JSON-safe).
GENERATION_TOP_K_LOGPROBS = 32


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def infer_finish_reason(
    generated_token_ids: list[int],
    tokenizer: Any,
    *,
    max_new_tokens: int,
) -> str:
    """Best-effort finish reason from generated ids (HF does not expose a single field)."""
    if not generated_token_ids:
        return "unknown"
    eos_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    for extra_id in getattr(tokenizer, "additional_special_tokens_ids", ()) or ():
        eos_ids.add(int(extra_id))
    if generated_token_ids[-1] in eos_ids:
        return "eos"
    if len(generated_token_ids) >= max_new_tokens:
        return "length"
    return "unknown"


def pack_generation_trace(
    generation_scores: tuple[Any, ...],
    generated_token_ids: list[int],
    *,
    top_k: int = GENERATION_TOP_K_LOGPROBS,
) -> dict[str, Any]:
    import torch

    if len(generation_scores) != len(generated_token_ids):
        raise ValueError(
            f"scores steps={len(generation_scores)} != token_ids={len(generated_token_ids)}"
        )
    chosen_logprobs: list[float] = []
    step_logits: list[dict[str, float]] = []
    for score_tensor, token_id in zip(generation_scores, generated_token_ids):
        log_probs = torch.log_softmax(score_tensor[0], dim=-1)
        chosen_logprobs.append(float(log_probs[token_id].item()))
        k = min(top_k, int(log_probs.shape[0]))
        top_values, top_indices = torch.topk(log_probs, k)
        step_logits.append(
            {str(int(idx)): float(val) for idx, val in zip(top_indices.tolist(), top_values.tolist())}
        )
    return {
        "generated_token_ids": generated_token_ids,
        "generated_token_logprobs": chosen_logprobs,
        "generated_logits": step_logits,
    }


def _mock_generation_trace(*, n_steps: int = 1) -> dict[str, Any]:
    token_ids = [100 + step for step in range(max(1, n_steps))]
    chosen_logprobs = [-0.1 - 0.05 * step for step in range(len(token_ids))]
    step_logits = [
        {str(token_ids[step]): chosen_logprobs[step], str(token_ids[step] + 1): chosen_logprobs[step] - 2.0}
        for step in range(len(token_ids))
    ]
    return {
        "generated_token_ids": token_ids,
        "generated_token_logprobs": chosen_logprobs,
        "generated_logits": step_logits,
    }


def _generation_scores(outputs: Any) -> tuple[Any, ...]:
    scores = getattr(outputs, "scores", None)
    if scores is None:
        raise ValueError("generate outputs missing scores (need output_scores=True)")
    return scores


# --- mcq_letter extractor ---

PROTOCOL_MCQ_LETTER = "mcq_letter"
_PROTOCOL_ALIASES = {"mcq_letter_v1": PROTOCOL_MCQ_LETTER}


def _letter_token_variants(letter: str) -> tuple[str, ...]:
    return (f" {letter}", letter, f" {letter.lower()}", letter.lower())


def _single_token_logprob(tokenizer: Any, log_probs: Any, text: str) -> float | None:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        return None
    return float(log_probs[token_ids[0]].item())


def _choice_letter_labels(n_choices: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(n_choices)]


def aggregate_choice_logprobs(trace: dict[str, Any]) -> dict[str, float]:
    candidates = trace.get("candidates")
    if isinstance(candidates, dict):
        labels = candidates.get("labels")
        scores = candidates.get("scores")
        if isinstance(labels, list) and isinstance(scores, list) and len(labels) == len(scores):
            return {str(label): float(score) for label, score in zip(labels, scores)}
    if "choice_logprobs" in trace:
        return dict(trace["choice_logprobs"])
    candidate_logprobs = trace["candidate_token_logprobs"]
    per_letter: dict[str, float] = {}
    for letter in _choice_letter_labels(trace["n_choices"]):
        best_logprob = float("-inf")
        for variant in _letter_token_variants(letter):
            if variant in candidate_logprobs:
                best_logprob = max(best_logprob, candidate_logprobs[variant])
        per_letter[letter] = best_logprob
    return per_letter


def _softmax_probabilities(scores: list[float]) -> list[float]:
    logit_peak = max(scores)
    exp_logits = [math.exp(value - logit_peak) for value in scores]
    probability_sum = sum(exp_logits)
    return [value / probability_sum for value in exp_logits]


def validate_protocol_trace(trace: dict[str, Any]) -> None:
    n_choices = trace["n_choices"]
    choice_logprobs = aggregate_choice_logprobs(trace)
    if len(choice_logprobs) != n_choices:
        raise ValueError(f"trace n_choices={n_choices} but got {len(choice_logprobs)} choice letters")
    predicted_letter = trace["predicted_letter"]
    if predicted_letter not in choice_logprobs:
        raise ValueError(f"predicted_letter {predicted_letter!r} not in choice logprobs")


@dataclass(frozen=True)
class McqLetterProtocolExtractor:
    protocol_version: str = PROTOCOL_MCQ_LETTER
    name: str = "choice_letter_v1"
    version: int = 2

    def capture_trace(
        self,
        outputs: Any,
        tokenizer: Any,
        generated_token_ids: list[int],
        protocol: dict[str, Any],
        *,
        prompt: str,
        n_choices: int,
    ) -> dict[str, Any]:
        import torch

        generation_scores = _generation_scores(outputs)
        if not generation_scores:
            raise ValueError("generation_scores is empty")
        generation = pack_generation_trace(generation_scores, generated_token_ids)
        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        generated_token = (
            tokenizer.decode(generated_token_ids[:1], skip_special_tokens=True)
            if generated_token_ids
            else ""
        )
        max_new_tokens = int(protocol.get("decoding", {}).get("max_tokens", 16))
        finish_reason = infer_finish_reason(
            generated_token_ids, tokenizer, max_new_tokens=max_new_tokens,
        )
        first_token_scores = generation_scores[0][0]
        token_logprobs = torch.log_softmax(first_token_scores, dim=-1)
        candidate_token_logprobs: dict[str, float] = {}
        for letter in _choice_letter_labels(n_choices):
            for variant in _letter_token_variants(letter):
                logprob = _single_token_logprob(tokenizer, token_logprobs, variant)
                if logprob is not None:
                    candidate_token_logprobs[variant] = logprob
        choice_logprobs = aggregate_choice_logprobs(
            {"candidate_token_logprobs": candidate_token_logprobs, "n_choices": n_choices},
        )
        labels = sorted(choice_logprobs)
        scores = [choice_logprobs[label] for label in labels]
        probs = _softmax_probabilities(scores)
        predicted_letter = max(choice_logprobs, key=choice_logprobs.get)
        return {
            "extractor_version": self.version,
            "feature_version": FEATURE_VERSION,
            "protocol": {
                "protocol_version": self.protocol_version,
            },
            "generated_text": generated_text,
            "generated_token": generated_token,
            "finish_reason": finish_reason,
            "n_choices": n_choices,
            "predicted_letter": predicted_letter,
            "choice_logprobs": choice_logprobs,
            "candidate_token_logprobs": candidate_token_logprobs,
            "candidates": {
                "labels": labels,
                "scores": scores,
                "probabilities": probs,
                "option_count": n_choices,
                "scoring": {
                    "method": "first_token_letter",
                    "version": 1,
                    "temperature": 1.0,
                },
                "scoring_method": "first_token_letter_variants_v1",
            },
            **generation,
        }

    def mock_trace(
        self,
        n_choices: int,
        peak_choice_index: int,
        *,
        generated_text: str | None = None,
    ) -> dict[str, Any]:
        predicted_letter = chr(ord("A") + peak_choice_index)
        text = generated_text if generated_text is not None else predicted_letter
        candidate_token_logprobs: dict[str, float] = {}
        for choice_index in range(n_choices):
            letter = chr(ord("A") + choice_index)
            base_logprob = -0.1 if choice_index == peak_choice_index else -2.5 - 0.1 * choice_index
            for variant in _letter_token_variants(letter):
                candidate_token_logprobs[variant] = base_logprob - (
                    0.01 if variant.startswith(" ") else 0.0
                )
        n_steps = max(1, len(text.split()))
        choice_logprobs = aggregate_choice_logprobs(
            {"candidate_token_logprobs": candidate_token_logprobs, "n_choices": n_choices},
        )
        labels = sorted(choice_logprobs)
        scores = [choice_logprobs[label] for label in labels]
        probs = _softmax_probabilities(scores)
        return {
            "extractor_version": self.version,
            "feature_version": FEATURE_VERSION,
            "protocol": {
                "protocol_version": self.protocol_version,
            },
            "generated_text": text,
            "generated_token": text.splitlines()[0].strip()[:8] or predicted_letter,
            "finish_reason": "mock",
            "n_choices": n_choices,
            "predicted_letter": predicted_letter,
            "choice_logprobs": choice_logprobs,
            "candidate_token_logprobs": candidate_token_logprobs,
            "candidates": {
                "labels": labels,
                "scores": scores,
                "probabilities": probs,
                "option_count": n_choices,
                "scoring": {
                    "method": "first_token_letter",
                    "version": 1,
                    "temperature": 1.0,
                },
                "scoring_method": "first_token_letter_variants_v1",
            },
            **_mock_generation_trace(n_steps=n_steps),
        }

    def has_protocol_trace(self, trace: dict[str, Any]) -> bool:
        has_mcq = bool(trace.get("candidate_token_logprobs") or trace.get("choice_logprobs"))
        has_generation = bool(trace.get("generated_token_ids") and trace.get("generated_token_logprobs"))
        return has_mcq or has_generation

    def validate(self, trace: dict[str, Any]) -> None:
        validate_protocol_trace(trace)

    def compute_metrics(self, trace: dict[str, Any], *, temperature: float = 1.0) -> dict[str, float]:
        self.validate(trace)
        choice_logprobs = aggregate_choice_logprobs(trace)
        letters = sorted(choice_logprobs)
        logits = [choice_logprobs[letter] for letter in letters]
        if temperature != 1.0:
            scale = 1.0 / temperature
            logits = [value * scale for value in logits]
        choice_probs = _softmax_probabilities(logits)
        entropy = -sum(prob * math.log(prob) for prob in choice_probs if prob > 0.0)
        sorted_probs = sorted(choice_probs, reverse=True)
        predicted_letter = trace["predicted_letter"]
        option_count = len(choice_probs)
        predicted_logprob = choice_logprobs[predicted_letter]
        return {
            "entropy": entropy,
            "normalized_entropy": entropy / math.log(option_count) if option_count > 1 else 0.0,
            "msp": sorted_probs[0],
            "margin": sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0],
            "top2_gap": sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0],
            "predicted_logprob": predicted_logprob,
            # Backward-compatible alias used by existing chi / scripts.
            "mean_logprob": predicted_logprob,
            "option_count": float(option_count),
        }


class ProtocolExtractor(Protocol):
    protocol_version: str
    name: str
    version: int

    def capture_trace(
        self,
        outputs: Any,
        tokenizer: Any,
        generated_token_ids: list[int],
        protocol: dict[str, Any],
        *,
        prompt: str,
        **kwargs: Any,
    ) -> dict: ...

    def mock_trace(self, n_choices: int, peak_choice_index: int, /) -> dict: ...

    def has_protocol_trace(self, trace: dict) -> bool: ...

    def validate(self, trace: dict) -> None: ...

    def compute_metrics(self, trace: dict, *, temperature: float = 1.0) -> dict[str, float]: ...


_MCQ_LETTER_EXTRACTOR = McqLetterProtocolExtractor()
_PROTOCOL_EXTRACTORS: dict[str, ProtocolExtractor] = {
    PROTOCOL_MCQ_LETTER: _MCQ_LETTER_EXTRACTOR,
    "mcq_letter_v1": _MCQ_LETTER_EXTRACTOR,
}


def _normalize_protocol_version(protocol_version: str) -> str:
    return _PROTOCOL_ALIASES.get(protocol_version, protocol_version)


def get_protocol_extractor(protocol_version: str) -> ProtocolExtractor:
    version = _normalize_protocol_version(protocol_version)
    try:
        return _PROTOCOL_EXTRACTORS[version]
    except KeyError as exc:
        raise KeyError(f"no ProtocolExtractor for protocol_version={protocol_version!r}") from exc


ARTIFACT_VERSION = 3
# Versioning contract:
# - artifact_version: top-level JSON envelope schema.
# - extractor_version: trace extraction implementation for a protocol extractor.
# - feature_version: feature definitions exposed by this protocol family.
# - metrics_version: formulas/transformations used in compute_metrics().
FEATURE_VERSION = "v1"
METRICS_VERSION = "v3"

MODEL_RESPONSE_METRIC_KEYS = frozenset({
    "entropy",
    "normalized_entropy",
    "msp",
    "margin",
    "top2_gap",
    "predicted_logprob",
    "mean_logprob",
    "option_count",
})


def model_response_prediction(trace: dict[str, Any], metrics: dict[str, float]) -> dict[str, Any]:
    """Stage 5B view: protocol-parsed answer + MSP as confidence field.

    MSP is a routing signal (max softmax over choice letters), not a calibrated P(correct).
    """
    return {
        "parsed_answer": trace["predicted_letter"],
        "confidence": float(metrics["msp"]),
    }


def model_response_raw(query: Any) -> dict[str, str]:
    """Stage 5B view: question text + gold answer letter (no corpus join later)."""
    return {
        "query": query.text.strip(),
        "answer": chr(ord("A") + int(query.answer_index)),
    }


def capture_protocol_trace(
    outputs: Any,
    tokenizer: Any,
    generated_token_ids: list[int],
    protocol: dict[str, Any],
    *,
    prompt: str,
    **extractor_kwargs: Any,
) -> dict[str, Any]:
    """Let the protocol extractor pull what it needs from raw generate() outputs."""
    protocol_version = protocol["protocol_version"]
    return get_protocol_extractor(protocol_version).capture_trace(
        outputs,
        tokenizer,
        generated_token_ids,
        protocol,
        prompt=prompt,
        **extractor_kwargs,
    )


def mock_protocol_trace(
    protocol_version: str,
    n_choices: int,
    peak_choice_index: int,
    *,
    generated_text: str | None = None,
) -> dict[str, Any]:
    extractor = get_protocol_extractor(protocol_version)
    if _normalize_protocol_version(protocol_version) == PROTOCOL_MCQ_LETTER:
        return extractor.mock_trace(n_choices, peak_choice_index, generated_text=generated_text)
    return extractor.mock_trace(n_choices, peak_choice_index)


def inference_capture_metadata(
    model: Any,
    tokenizer: Any,
    *,
    torch_dtype: Any,
) -> dict[str, str | None]:
    """Reproducibility fields stored once per model load in each artifact."""
    import torch
    import transformers

    revision: str | None = None
    if hasattr(model, "config"):
        config = model.config
        revision = (
            getattr(config, "_commit_hash", None)
            or getattr(config, "_name_or_path", None)
            or getattr(config, "name_or_path", None)
        )
    return {
        "model_revision": revision,
        "tokenizer_id": getattr(tokenizer, "name_or_path", None),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "dtype": str(torch_dtype).removeprefix("torch."),
    }


def model_capture_metadata(model: Any, *, torch_dtype: Any) -> dict[str, str | None]:
    """Deprecated alias — prefer inference_capture_metadata(model, tokenizer, ...)."""
    return {
        "model_revision": (
            getattr(getattr(model, "config", None), "_commit_hash", None)
            or getattr(getattr(model, "config", None), "_name_or_path", None)
        ),
        "transformers_version": __import__("transformers").__version__,
        "torch_version": __import__("torch").__version__,
        "dtype": str(torch_dtype).removeprefix("torch."),
        "tokenizer_id": None,
    }


def build_protocol_artifact(
    protocol_version: str,
    trace: dict[str, Any],
    *,
    model_id: str,
    prompt: str,
    prompt_token_count: int | None = None,
    tokenizer_id: str | None = None,
    model_revision: str | None = None,
    transformers_version: str | None = None,
    torch_version: str | None = None,
    dtype: str | None = None,
) -> dict[str, Any]:
    extractor = get_protocol_extractor(protocol_version)
    artifact: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "feature_version": FEATURE_VERSION,
        "protocol_version": protocol_version,
        "extractor": extractor.name,
        "extractor_version": extractor.version,
        "capture_time": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "prompt_sha256": prompt_sha256(prompt),
        "trace": trace,
    }
    if prompt_token_count is not None:
        artifact["prompt_token_count"] = int(prompt_token_count)
    for key, val in (
        ("tokenizer_id", tokenizer_id),
        ("model_revision", model_revision),
        ("transformers_version", transformers_version),
        ("torch_version", torch_version),
        ("dtype", dtype),
    ):
        if val is not None:
            artifact[key] = val
    return artifact


def has_protocol_trace(model_response: dict[str, Any] | None) -> bool:
    if not model_response or "trace" not in model_response:
        return False
    try:
        extractor = get_protocol_extractor(model_response["protocol_version"])
    except KeyError:
        return False
    return extractor.has_protocol_trace(model_response["trace"])
