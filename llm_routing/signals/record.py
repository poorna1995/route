"""Save and load signal files — one row per question.

Methodology (signal artifacts only; CPU unless noted):

  Part I — Experimental design
    M1  prepare           corpus + holdout partition
    M2  oracle (GPU)      pilot traces in oracle/*.jsonl; scorecard gates
    M3  eval              lock R_c / R_t split in corpus/partition.json

  Part II — Development
    4   oracle (GPU)      calib/test traces in oracle/*.jsonl
    5A  model-independent φ(q) → signals/model_independent.jsonl
    5B  model-response    ψ(q) → signals/model_response_{M_lo,M_hi}.jsonl
  5C  cross-model       χ(q) → signals/cross_model_comparative.jsonl (join only)

Each row is self-describing via signal_type. Oracle traces (canonical scores) stay in oracle/*.jsonl.
Stage 5 stores derived ψ metrics + prediction only — not the full option distribution.

Depends on:
  - corpus.read_jsonl / write_jsonl — file I/O
  - signals.psi.protocol.MODEL_RESPONSE_METRIC_KEYS — names for legacy flat rows
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from llm_routing.corpus import read_jsonl, write_jsonl
from llm_routing.signals.psi.protocol import MODEL_RESPONSE_METRIC_KEYS

# χ continuous deltas — promoted from legacy flat rows. prediction_disagreement is bool (see below).
KNOWN_FLAT_METRIC_KEYS = MODEL_RESPONSE_METRIC_KEYS | frozenset({
    "token_entropy",
    "delta_entropy",
    "delta_margin",
    "delta_msp",
    "delta_mean_logprob",
    "prediction_disagreement",
    "answer_disagreement",
})

CATEGORICAL_METRIC_KEYS = frozenset({"prediction_disagreement"})

SignalLayer = Literal["model_independent", "model_dependent", "cross_model"]
SignalType = Literal["model_independent", "model_response", "cross_model_comparative"]

SIGNAL_TYPE_MODEL_INDEPENDENT = "model_independent"
SIGNAL_TYPE_QUERY_DERIVED = SIGNAL_TYPE_MODEL_INDEPENDENT  # deprecated alias
SIGNAL_TYPE_MODEL_RESPONSE = "model_response"
SIGNAL_TYPE_CROSS_MODEL = "cross_model_comparative"

SIGNAL_LAYERS: tuple[dict[str, str], ...] = (
    {
        "layer": "model_independent",
        "signal_type": SIGNAL_TYPE_MODEL_INDEPENDENT,
        "representation_id": "query_combined",
        "description": "Uses only the question text. No model run needed.",
    },
    {
        "layer": "model_dependent",
        "signal_type": SIGNAL_TYPE_MODEL_RESPONSE,
        "representation_id": "model_response",
        "description": "Uses one model's answer and confidence on that question.",
    },
    {
        "layer": "cross_model",
        "signal_type": SIGNAL_TYPE_CROSS_MODEL,
        "representation_id": "cross_model",
        "description": "Compares low vs high model on the same question.",
    },
)

VALID_SIGNAL_TYPES = frozenset(
    {SIGNAL_TYPE_MODEL_INDEPENDENT, "query_derived", SIGNAL_TYPE_MODEL_RESPONSE, SIGNAL_TYPE_CROSS_MODEL}
)

ORACLE_ONLY_KEYS = frozenset({
    "correct",
    "latency_ms",
    "model",
    "model_response",
    "parsed_answer",
    "raw_output",
    "token_count",
})

_LEGACY_DEFAULT_SIGNAL_TYPE = SIGNAL_TYPE_MODEL_RESPONSE


def _normalize_prediction(prediction: dict[str, Any] | None) -> dict[str, Any] | None:
    """Map legacy letter/probability keys to parsed_answer/confidence."""
    if prediction is None:
        return None
    out = dict(prediction)
    if "parsed_answer" not in out and "letter" in out:
        out["parsed_answer"] = out.pop("letter")
    if "confidence" not in out and "probability" in out:
        out["confidence"] = out.pop("probability")
    return out


def _normalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Coerce legacy encodings; keep continuous scalars as float."""
    out = dict(metrics)
    if "answer_disagreement" in out and "prediction_disagreement" not in out:
        out["prediction_disagreement"] = out.pop("answer_disagreement")
    out.pop("predicted_letter_disagreement", None)
    if "prediction_disagreement" in out:
        v = out["prediction_disagreement"]
        if isinstance(v, bool):
            pass
        elif v in (0, 0.0):
            out["prediction_disagreement"] = False
        elif v in (1, 1.0):
            out["prediction_disagreement"] = True
        else:
            out["prediction_disagreement"] = bool(v)
    for key, val in list(out.items()):
        if key in CATEGORICAL_METRIC_KEYS:
            continue
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            out[key] = float(val)
    return out


@dataclass(frozen=True)
class SignalRecord:
    """One question's saved signal numbers (ψ, χ, …). Oracle traces stay in oracle/*.jsonl."""

    query_id: str
    signal_type: str
    metrics: dict[str, float | bool] = field(default_factory=dict)
    prediction: dict[str, Any] | None = None
    raw: dict[str, str] | None = None
    metrics_version: str | None = None
    extractor_version: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        if out.get("prediction") is None:
            out.pop("prediction", None)
        if out.get("raw") is None:
            out.pop("raw", None)
        if not out.get("extra"):
            out.pop("extra", None)
        return out

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> SignalRecord:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        metrics = _normalize_metrics(dict(row.get("metrics") or {}))
        for key in KNOWN_FLAT_METRIC_KEYS:
            if key in row and key not in metrics:
                metrics[key] = row[key]
        metrics = _normalize_metrics(metrics)
        extra = dict(row.get("extra") or {})
        for k, v in row.items():
            if k in ORACLE_ONLY_KEYS:
                continue
            if k not in known and k not in metrics:
                extra[k] = v
        signal_type = row.get("signal_type") or _LEGACY_DEFAULT_SIGNAL_TYPE
        if signal_type == "query_derived":
            signal_type = SIGNAL_TYPE_MODEL_INDEPENDENT
        if signal_type not in VALID_SIGNAL_TYPES:
            raise ValueError(f"unknown signal_type={signal_type!r}")
        optional = {
            k: row[k]
            for k in ("metrics_version", "extractor_version", "raw")
            if k in row
        }
        if "prediction" in row:
            optional["prediction"] = _normalize_prediction(row["prediction"])
        return cls(
            query_id=row["query_id"],
            signal_type=signal_type,
            metrics=metrics,
            extra=extra,
            **optional,
        )


def save_signals(path: Path, records: list[SignalRecord]) -> None:
    write_jsonl(path, records, SignalRecord.to_dict)


def load_signals(path: Path) -> list[SignalRecord]:
    return read_jsonl(path, SignalRecord.from_dict)
