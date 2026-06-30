"""Static φ/ψ/χ column names and signal-layer taxonomy (shared by deploy and analysis)."""

from __future__ import annotations

from llm_routing.signals.chi.stage import CROSS_MODEL_METRIC_KEYS
from llm_routing.signals.psi.protocol import MODEL_RESPONSE_METRIC_KEYS
from llm_routing.signals.phi.core import JSONL_BLOCKS, query_all_columns, query_block_columns

REP_QUERY_STRUCTURAL = "query_structural"
REP_QUERY_AMBIGUITY = "query_ambiguity"
REP_QUERY_GEOMETRY = "query_geometry"
REP_QUERY_COMBINED = "query_combined"
REP_MODEL_RESPONSE = "model_response"
REP_CROSS_MODEL = "cross_model"

SIGNAL_LAYER_MODEL_INDEPENDENT = "model_independent"
SIGNAL_LAYER_MODEL_DEPENDENT = "model_dependent"
SIGNAL_LAYER_CROSS_MODEL = "cross_model"

SIGNAL_LAYER_QUERY = SIGNAL_LAYER_MODEL_INDEPENDENT
SIGNAL_LAYER_RESPONSE = SIGNAL_LAYER_MODEL_DEPENDENT
SIGNAL_LAYER_CROSS = SIGNAL_LAYER_CROSS_MODEL

_LEGACY_SIGNAL_LAYER_ALIASES: dict[str, str] = {
    "query": SIGNAL_LAYER_MODEL_INDEPENDENT,
    "model_response": SIGNAL_LAYER_MODEL_DEPENDENT,
}


def normalize_signal_layer(layer: str) -> str:
    return _LEGACY_SIGNAL_LAYER_ALIASES.get(layer, layer)


def normalize_signal_layers(layers: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(normalize_signal_layer(layer) for layer in layers)


BLOCK_REPRESENTATION: dict[str, str] = {
    "structural": REP_QUERY_STRUCTURAL,
    "ambiguity": REP_QUERY_AMBIGUITY,
    "embedding_geometry": REP_QUERY_GEOMETRY,
}

QUERY_BLOCK_COLUMNS: dict[str, tuple[str, ...]] = {
    block: query_block_columns(block) for block in JSONL_BLOCKS
}
QUERY_COLUMNS: tuple[str, ...] = query_all_columns()

RESPONSE_COLUMNS: tuple[str, ...] = tuple(
    f"psi.{key}" for key in sorted(MODEL_RESPONSE_METRIC_KEYS)
)
CROSS_COLUMNS: tuple[str, ...] = tuple(f"chi.{key}" for key in sorted(CROSS_MODEL_METRIC_KEYS))

LABEL_COLUMNS = ("query_id", "r", "bucket", "y_lo", "y_hi")

FEATURE_COLUMNS: tuple[str, ...] = QUERY_COLUMNS + RESPONSE_COLUMNS + CROSS_COLUMNS

SCHEMA_VERSION = "v1"
