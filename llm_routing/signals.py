"""Signal artifacts x(q) — independent of oracle QueryResult."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from llm_routing.corpus import read_jsonl, write_jsonl


@dataclass(frozen=True)
class SignalRecord:
    """One row per query: frozen feature vector for analysis / policy fit."""

    query_id: str
    query_length: int | None = None
    entropy: float | None = None
    margin: float | None = None
    msp: float | None = None
    token_entropy: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> SignalRecord:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        core = {k: row[k] for k in known if k in row and k not in ("extra", "query_id")}
        extra = dict(row.get("extra") or {})
        for k, v in row.items():
            if k not in known:
                extra[k] = v
        return cls(query_id=row["query_id"], extra=extra, **core)


def save_signals(path: Path, records: list[SignalRecord]) -> None:
    write_jsonl(path, records, SignalRecord.to_dict)


def load_signals(path: Path) -> list[SignalRecord]:
    return read_jsonl(path, SignalRecord.from_dict)
