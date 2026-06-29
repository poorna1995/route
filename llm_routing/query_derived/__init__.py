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
    "load_query_derived_defaults",
    "load_section",
    "novelty_section",
    "resolve_tokenizer_id",
    "run_query_derived",
    "save_embedding",
    "structural_section",
    "word_tokens",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "DEFAULTS_PATH": ("llm_routing.query_derived.core", "DEFAULTS_PATH"),
    "GeometryModel": ("llm_routing.query_derived.core", "GeometryModel"),
    "JSONL_BLOCKS": ("llm_routing.query_derived.core", "JSONL_BLOCKS"),
    "MAX_PCA_COMPONENTS": ("llm_routing.query_derived.core", "MAX_PCA_COMPONENTS"),
    "NoveltyModel": ("llm_routing.query_derived.core", "NoveltyModel"),
    "QueryDerivedRecord": ("llm_routing.query_derived.core", "QueryDerivedRecord"),
    "TokenCounter": ("llm_routing.query_derived.core", "TokenCounter"),
    "ZScoreModel": ("llm_routing.query_derived.core", "ZScoreModel"),
    "windowed_type_token_ratio": ("llm_routing.query_derived.core", "windowed_type_token_ratio"),
    "canonical_user": ("llm_routing.query_derived.core", "canonical_user"),
    "encode_canonical_texts": ("llm_routing.query_derived.core", "encode_canonical_texts"),
    "extract_ambiguity": ("llm_routing.query_derived.core", "extract_ambiguity"),
    "extract_lexical": ("llm_routing.query_derived.core", "extract_lexical"),
    "extract_load": ("llm_routing.query_derived.core", "extract_load"),
    "extract_mcq": ("llm_routing.query_derived.core", "extract_mcq"),
    "extract_structural": ("llm_routing.query_derived.core", "extract_structural"),
    "extract_token_stats": ("llm_routing.query_derived.core", "extract_token_stats"),
    "flatten_blocks": ("llm_routing.query_derived.core", "flatten_blocks"),
    "embedding_geometry_section": ("llm_routing.query_derived.core", "embedding_geometry_section"),
    "geometry_section": ("llm_routing.query_derived.core", "embedding_geometry_section"),
    "load_query_derived_defaults": ("llm_routing.query_derived.core", "load_query_derived_defaults"),
    "load_section": ("llm_routing.query_derived.core", "load_section"),
    "novelty_section": ("llm_routing.query_derived.core", "novelty_section"),
    "resolve_tokenizer_id": ("llm_routing.query_derived.core", "resolve_tokenizer_id"),
    "run_query_derived": ("llm_routing.query_derived.run", "run_query_derived"),
    "save_embedding": ("llm_routing.query_derived.core", "save_embedding"),
    "structural_section": ("llm_routing.query_derived.core", "structural_section"),
    "word_tokens": ("llm_routing.query_derived.core", "word_tokens"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr = _EXPORTS[name]
    import importlib

    return getattr(importlib.import_module(module_path), attr)
