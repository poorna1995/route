"""Query-derived φ(q): [load, ambiguity, semantic, novelty].

Modules
-------
config   — manifest + record schema
extract  — per-query φ_load, φ_ambiguity
engineer — φ_semantic encoder + R_c engineering
run      — pipeline orchestration
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DEFAULTS_PATH",
    "GeometryModel",
    "JSONL_BLOCKS",
    "NoveltyModel",
    "QueryDerivedRecord",
    "TokenCounter",
    "ZScoreModel",
    "_mattr",
    "canonical_user",
    "encode_canonical_texts",
    "extract_ambiguity",
    "extract_lexical",
    "extract_load",
    "extract_mcq",
    "extract_structural",
    "flatten_blocks",
    "load_query_derived_defaults",
    "resolve_tokenizer_id",
    "run_query_derived",
    "save_embedding",
    "word_tokens",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "DEFAULTS_PATH": ("llm_routing.query_derived.config", "DEFAULTS_PATH"),
    "GeometryModel": ("llm_routing.query_derived.engineer", "GeometryModel"),
    "JSONL_BLOCKS": ("llm_routing.query_derived.config", "JSONL_BLOCKS"),
    "NoveltyModel": ("llm_routing.query_derived.engineer", "NoveltyModel"),
    "QueryDerivedRecord": ("llm_routing.query_derived.config", "QueryDerivedRecord"),
    "TokenCounter": ("llm_routing.query_derived.extract", "TokenCounter"),
    "ZScoreModel": ("llm_routing.query_derived.engineer", "ZScoreModel"),
    "_mattr": ("llm_routing.query_derived.extract", "mattr"),
    "canonical_user": ("llm_routing.query_derived.extract", "canonical_user"),
    "encode_canonical_texts": ("llm_routing.query_derived.engineer", "encode_canonical_texts"),
    "extract_ambiguity": ("llm_routing.query_derived.extract", "extract_ambiguity"),
    "extract_lexical": ("llm_routing.query_derived.extract", "extract_lexical"),
    "extract_load": ("llm_routing.query_derived.extract", "extract_load"),
    "extract_mcq": ("llm_routing.query_derived.extract", "extract_mcq"),
    "extract_structural": ("llm_routing.query_derived.extract", "extract_structural"),
    "flatten_blocks": ("llm_routing.query_derived.config", "flatten_blocks"),
    "load_query_derived_defaults": (
        "llm_routing.query_derived.config",
        "load_query_derived_defaults",
    ),
    "resolve_tokenizer_id": ("llm_routing.query_derived.config", "resolve_tokenizer_id"),
    "run_query_derived": ("llm_routing.query_derived.run", "run_query_derived"),
    "save_embedding": ("llm_routing.query_derived.engineer", "save_embedding"),
    "word_tokens": ("llm_routing.query_derived.extract", "word_tokens"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr = _EXPORTS[name]
    import importlib

    return getattr(importlib.import_module(module_path), attr)
