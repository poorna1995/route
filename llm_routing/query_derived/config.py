"""Manifest, schema, and run-time configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_PATH = ROOT / "experiments/query_derived_defaults.yaml"

JSONL_BLOCKS = ("load", "ambiguity", "novelty")


@dataclass
class QueryDerivedRecord:
    query_id: str
    split: str
    load: dict[str, Any]
    ambiguity: dict[str, Any]
    novelty: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def flatten_blocks(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for block in JSONL_BLOCKS:
        for key, val in (record.get(block) or {}).items():
            out[f"{block}.{key}"] = val
    return out


def load_query_derived_defaults(path: Path | None = None) -> dict[str, Any]:
    path = path or DEFAULTS_PATH
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected mapping")
    return data


def load_section(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("load") or config.get("lexical") or {}


def novelty_section(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("novelty") or config.get("geometry") or {}


def resolve_tokenizer_id(setting: dict[str, Any], config: dict[str, Any]) -> str | None:
    source = config.get("tokenizer", {}).get("source", "pool_M_lo")
    if source == "pool_M_lo":
        return setting.get("pool", {}).get("M_lo")
    if source == "none":
        return None
    return str(source)
