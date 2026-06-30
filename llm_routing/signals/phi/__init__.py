"""Query-derived φ(q): core features + run orchestration."""

from __future__ import annotations

from typing import Any

__all__ = [
    "DEFAULTS_PATH",
    "GeometryModel",
    "JSONL_BLOCKS",
    "MAX_PCA_COMPONENTS",
    "NoveltyModel",
    "QueryDerivedRecord",
    "TokenCounter",
    "ZScoreModel",
    "windowed_type_token_ratio",
    "canonical_user",
    "encode_canonical_texts",
    "extract_ambiguity",
    "extract_lexical",
    "extract_load",
    "extract_mcq",
    "extract_structural",
    "extract_token_stats",
    "flatten_blocks",
    "embedding_geometry_section",
    "load_model_independent_defaults",
    "load_query_derived_defaults",
    "load_section",
    "novelty_section",
    "resolve_tokenizer_id",
    "run_model_independent",
    "run_query_derived",
    "save_embedding",
    "structural_section",
    "word_tokens",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "DEFAULTS_PATH": ("llm_routing.signals.phi.core", "DEFAULTS_PATH"),
    "GeometryModel": ("llm_routing.signals.phi.core", "GeometryModel"),
    "JSONL_BLOCKS": ("llm_routing.signals.phi.core", "JSONL_BLOCKS"),
    "MAX_PCA_COMPONENTS": ("llm_routing.signals.phi.core", "MAX_PCA_COMPONENTS"),
    "NoveltyModel": ("llm_routing.signals.phi.core", "NoveltyModel"),
    "QueryDerivedRecord": ("llm_routing.signals.phi.core", "QueryDerivedRecord"),
    "TokenCounter": ("llm_routing.signals.phi.core", "TokenCounter"),
    "ZScoreModel": ("llm_routing.signals.phi.core", "ZScoreModel"),
    "windowed_type_token_ratio": ("llm_routing.signals.phi.core", "windowed_type_token_ratio"),
    "canonical_user": ("llm_routing.signals.phi.core", "canonical_user"),
    "encode_canonical_texts": ("llm_routing.signals.phi.core", "encode_canonical_texts"),
    "extract_ambiguity": ("llm_routing.signals.phi.core", "extract_ambiguity"),
    "extract_lexical": ("llm_routing.signals.phi.core", "extract_lexical"),
    "extract_load": ("llm_routing.signals.phi.core", "extract_load"),
    "extract_mcq": ("llm_routing.signals.phi.core", "extract_mcq"),
    "extract_structural": ("llm_routing.signals.phi.core", "extract_structural"),
    "extract_token_stats": ("llm_routing.signals.phi.core", "extract_token_stats"),
    "flatten_blocks": ("llm_routing.signals.phi.core", "flatten_blocks"),
    "embedding_geometry_section": ("llm_routing.signals.phi.core", "embedding_geometry_section"),
    "geometry_section": ("llm_routing.signals.phi.core", "embedding_geometry_section"),
    "load_model_independent_defaults": ("llm_routing.signals.phi.core", "load_model_independent_defaults"),
    "load_query_derived_defaults": ("llm_routing.signals.phi.core", "load_query_derived_defaults"),
    "load_section": ("llm_routing.signals.phi.core", "load_section"),
    "novelty_section": ("llm_routing.signals.phi.core", "novelty_section"),
    "resolve_tokenizer_id": ("llm_routing.signals.phi.core", "resolve_tokenizer_id"),
    "run_model_independent": ("llm_routing.signals.phi.run", "run_model_independent"),
    "run_query_derived": ("llm_routing.signals.phi.run", "run_query_derived"),
    "save_embedding": ("llm_routing.signals.phi.core", "save_embedding"),
    "structural_section": ("llm_routing.signals.phi.core", "structural_section"),
    "word_tokens": ("llm_routing.signals.phi.core", "word_tokens"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr = _EXPORTS[name]
    import importlib

    return getattr(importlib.import_module(module_path), attr)
