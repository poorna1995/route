"""Protocol traces (GPU capture) and ψ metrics (CPU) for model_response artifacts.

Three-artifact contract:
  1. Oracle trace (Stage 4, immutable): canonical candidate scores + generation provenance.
     Probabilities and uncertainty features are derived later — never stored here.
  2. SignalRecord (Stage 5): slim ψ metrics + prediction for ML stages.
  3. Analysis table (Stage 6): flat φ/ψ/χ + labels for router training.

Oracle trace (mcq_letter) stores:
  protocol, generation{decoding, generated_text, ...},
  candidates{labels, scores, ranking, option_count, scoring{method, version}}
  predicted_letter, n_choices

Temperature contract (three independent concepts):
  - generation.decoding.temperature — how model.generate was configured
  - candidates.scores — raw log-probs; no analysis temperature applied
  - Stage 5 analysis_temperature — softmax scaling when deriving ψ metrics

Artifact envelope: artifact_version, extractor_version, model_id, prompt_sha256,
  tokenizer_id, model_revision, transformers_version, torch_version, dtype, capture_time
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


def _rank_labels(choice_scores: dict[str, float]) -> list[str]:
    return [
        label
        for label, _ in sorted(choice_scores.items(), key=lambda item: item[1], reverse=True)
    ]


def _resolve_scoring_method(protocol: dict[str, Any]) -> str:
    scoring = protocol.get("scoring") or {}
    return str(scoring.get("method", "first_token_letter"))


def _first_token_letter_scores(
    generation_scores: tuple[Any, ...],
    tokenizer: Any,
    n_choices: int,
) -> dict[str, float]:
    import torch

    if not generation_scores:
        raise ValueError("generation_scores is empty")
    first_token_scores = generation_scores[0][0]
    token_logprobs = torch.log_softmax(first_token_scores, dim=-1)
    variant_logprobs: dict[str, float] = {}
    for letter in _choice_letter_labels(n_choices):
        for variant in _letter_token_variants(letter):
            logprob = _single_token_logprob(tokenizer, token_logprobs, variant)
            if logprob is not None:
                variant_logprobs[variant] = logprob
    return aggregate_choice_scores(
        {"candidate_token_logprobs": variant_logprobs, "n_choices": n_choices},
    )


def _teacher_forced_letter_scores(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    n_choices: int,
    protocol: dict[str, Any],
) -> dict[str, float]:
    """ψ₂: score full candidate continuations P(answer | prompt).

    Not implemented yet — requires a teacher-forced forward pass per option.
    Produces the same canonical candidates.scores block as first_token_letter.
    """
    raise NotImplementedError(
        "scoring.method=teacher_forced is not implemented yet. "
        "Use first_token_letter for the L1 baseline, or implement teacher-forced "
        "scoring in protocol.py without changing Stage 5/6."
    )


_SCORING_BACKENDS: dict[str, int] = {
    "first_token_letter": 1,
    "teacher_forced": 1,
}


def _generation_decoding_config(protocol: dict[str, Any]) -> dict[str, Any]:
    """Record how the oracle answer was generated (not Stage 5 analysis temperature)."""
    decoding = protocol.get("decoding") or {}
    return {
        "do_sample": bool(decoding.get("do_sample", False)),
        "temperature": float(decoding.get("temperature", 0.0)),
        "max_new_tokens": int(decoding.get("max_tokens", 16)),
    }


def capture_candidate_scores(
    method: str,
    *,
    tokenizer: Any,
    n_choices: int,
    generation_scores: tuple[Any, ...] | None = None,
    protocol: dict[str, Any] | None = None,
    prompt: str = "",
    model: Any | None = None,
) -> dict[str, Any]:
    """Score MCQ options — the single extension point for ψ signal quality.

    Returns the canonical candidates block: labels, scores, ranking, option_count, scoring.
    Stage 5+ only reads this block; swap backends without touching downstream stages.
    """
    if method not in _SCORING_BACKENDS:
        raise ValueError(f"unknown scoring.method={method!r}; expected one of {sorted(_SCORING_BACKENDS)}")
    if method == "first_token_letter":
        if generation_scores is None:
            raise ValueError("first_token_letter scoring requires generation_scores")
        choice_scores = _first_token_letter_scores(generation_scores, tokenizer, n_choices)
    elif method == "teacher_forced":
        if model is None or not prompt:
            raise ValueError("teacher_forced scoring requires model and prompt")
        choice_scores = _teacher_forced_letter_scores(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            n_choices=n_choices,
            protocol=protocol or {},
        )
    else:
        raise ValueError(f"scoring backend not wired: {method!r}")
    labels = sorted(choice_scores)
    scores = [choice_scores[label] for label in labels]
    ranking = _rank_labels(choice_scores)
    return {
        "labels": labels,
        "scores": scores,
        "ranking": ranking,
        "option_count": n_choices,
        "scoring": {
            "method": method,
            "version": _SCORING_BACKENDS[method],
        },
    }


def aggregate_choice_scores(trace: dict[str, Any]) -> dict[str, float]:
    """Canonical per-option log-probabilities (aggregated over token variants)."""
    candidates = trace.get("candidates")
    if isinstance(candidates, dict):
        labels = candidates.get("labels")
        scores = candidates.get("scores")
        if isinstance(labels, list) and isinstance(scores, list) and len(labels) == len(scores):
            return {str(label): float(score) for label, score in zip(labels, scores)}
    if "choice_logprobs" in trace:
        return dict(trace["choice_logprobs"])
    candidate_logprobs = trace.get("candidate_token_logprobs")
    if not candidate_logprobs:
        raise KeyError("trace missing candidates.scores and legacy choice fields")
    per_letter: dict[str, float] = {}
    for letter in _choice_letter_labels(trace["n_choices"]):
        best_logprob = float("-inf")
        for variant in _letter_token_variants(letter):
            if variant in candidate_logprobs:
                best_logprob = max(best_logprob, candidate_logprobs[variant])
        per_letter[letter] = best_logprob
    return per_letter


# Backward-compatible alias — scores are log-probabilities.
aggregate_choice_logprobs = aggregate_choice_scores


def choice_probabilities(
    trace: dict[str, Any],
    *,
    analysis_temperature: float = 1.0,
) -> dict[str, float]:
    """Derive softmax probabilities from canonical candidate scores.

    analysis_temperature scales logits before softmax in Stage 5 only; scores are unchanged.
    """
    choice_scores = aggregate_choice_scores(trace)
    letters = sorted(choice_scores)
    logits = [choice_scores[letter] for letter in letters]
    if analysis_temperature != 1.0:
        scale = 1.0 / analysis_temperature
        logits = [value * scale for value in logits]
    probs = _softmax_probabilities(logits)
    return {letter: prob for letter, prob in zip(letters, probs)}


def _softmax_probabilities(scores: list[float]) -> list[float]:
    logit_peak = max(scores)
    exp_logits = [math.exp(value - logit_peak) for value in scores]
    probability_sum = sum(exp_logits)
    return [value / probability_sum for value in exp_logits]


def validate_protocol_trace(trace: dict[str, Any]) -> None:
    n_choices = trace["n_choices"]
    choice_scores = aggregate_choice_scores(trace)
    if len(choice_scores) != n_choices:
        raise ValueError(f"trace n_choices={n_choices} but got {len(choice_scores)} choice letters")
    predicted_letter = trace["predicted_letter"]
    if predicted_letter not in choice_scores:
        raise ValueError(f"predicted_letter {predicted_letter!r} not in choice scores")


@dataclass(frozen=True)
class McqLetterProtocolExtractor:
    protocol_version: str = PROTOCOL_MCQ_LETTER
    name: str = "choice_letter_v1"
    version: int = 5

    def capture_trace(
        self,
        outputs: Any,
        tokenizer: Any,
        generated_token_ids: list[int],
        protocol: dict[str, Any],
        *,
        prompt: str,
        n_choices: int,
        model: Any | None = None,
    ) -> dict[str, Any]:
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
        scoring_method = _resolve_scoring_method(protocol)
        candidates = capture_candidate_scores(
            scoring_method,
            tokenizer=tokenizer,
            n_choices=n_choices,
            generation_scores=generation_scores,
            protocol=protocol,
            prompt=prompt,
            model=model,
        )
        predicted_letter = candidates["ranking"][0]
        return {
            "extractor_version": self.version,
            "feature_version": FEATURE_VERSION,
            "protocol": {
                "protocol_version": self.protocol_version,
            },
            "generation": {
                "decoding": _generation_decoding_config(protocol),
                "generated_text": generated_text,
                "generated_token": generated_token,
                "finish_reason": finish_reason,
                **generation,
            },
            "n_choices": n_choices,
            "predicted_letter": predicted_letter,
            "candidates": candidates,
        }

    def has_protocol_trace(self, trace: dict[str, Any]) -> bool:
        candidates = trace.get("candidates")
        has_scores = (
            isinstance(candidates, dict)
            and isinstance(candidates.get("scores"), list)
            and bool(candidates["scores"])
        )
        has_legacy_mcq = bool(trace.get("candidate_token_logprobs") or trace.get("choice_logprobs"))
        generation = trace.get("generation") if isinstance(trace.get("generation"), dict) else trace
        has_generation = bool(
            generation.get("generated_token_ids") and generation.get("generated_token_logprobs")
        )
        return has_scores or has_legacy_mcq or has_generation

    def validate(self, trace: dict[str, Any]) -> None:
        validate_protocol_trace(trace)

    def compute_metrics(
        self,
        trace: dict[str, Any],
        *,
        analysis_temperature: float = 1.0,
    ) -> dict[str, float]:
        self.validate(trace)
        choice_scores = aggregate_choice_scores(trace)
        choice_probs_map = choice_probabilities(trace, analysis_temperature=analysis_temperature)
        choice_probs = [choice_probs_map[letter] for letter in sorted(choice_probs_map)]
        entropy = -sum(prob * math.log(prob) for prob in choice_probs if prob > 0.0)
        sorted_probs = sorted(choice_probs, reverse=True)
        predicted_letter = trace["predicted_letter"]
        option_count = len(choice_probs)
        predicted_logprob = choice_scores[predicted_letter]
        top2 = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        if top2 > 1e-12:
            top_probability_ratio = sorted_probs[0] / top2
        else:
            top_probability_ratio = 1e6  # JSON-safe cap when top-2 collapses
        return {
            "entropy": entropy,
            "normalized_entropy": entropy / math.log(option_count) if option_count > 1 else 0.0,
            "msp": sorted_probs[0],
            "margin": sorted_probs[0] - top2 if len(sorted_probs) > 1 else sorted_probs[0],
            "top2_gap": sorted_probs[0] - top2 if len(sorted_probs) > 1 else sorted_probs[0],
            "top_probability_ratio": top_probability_ratio,
            "effective_support": math.exp(entropy),
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

    def has_protocol_trace(self, trace: dict) -> bool: ...

    def validate(self, trace: dict) -> None: ...

    def compute_metrics(
        self, trace: dict, *, analysis_temperature: float = 1.0,
    ) -> dict[str, float]: ...


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
METRICS_VERSION = "v4"

MODEL_RESPONSE_METRIC_KEYS = frozenset({
    "entropy",
    "normalized_entropy",
    "msp",
    "margin",
    "top2_gap",
    "top_probability_ratio",
    "effective_support",
    "predicted_logprob",
    "mean_logprob",
    "option_count",
})


def trace_generated_text(trace: dict[str, Any]) -> str:
    generation = trace.get("generation")
    if isinstance(generation, dict) and "generated_text" in generation:
        return str(generation["generated_text"])
    return str(trace["generated_text"])


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
