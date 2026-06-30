"""Runtime signal extraction — φ and ψ only (no cross-model χ)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_routing.paths import model_independent_jsonl_path
from llm_routing.signals.phi.core import flatten_query_row
from llm_routing.signal_schema import QUERY_COLUMNS, RESPONSE_COLUMNS
from llm_routing.signals import load_signals


def _load_jsonl_row(path: Path, query_id: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("query_id") == query_id:
                return row
    return None


def extract_from_run(
    run_root: Path | str,
    query_id: str,
    *,
    include_query: bool = True,
    response_role: str = "M_lo",
) -> dict[str, float]:
    """Load precomputed φ and ψ for one query from a research run directory.

    ψ features always come from the M_lo trace (cheap-model run before routing).
    Policies are trained on psi.* columns from model_response_M_lo.jsonl; do not
    pass M_hi here unless you have retrained the policy on M_hi ψ features.
    """
    run_root = Path(run_root)
    out: dict[str, float] = {}

    if include_query:
        phi_row = _load_jsonl_row(model_independent_jsonl_path(run_root), query_id)
        if phi_row is None:
            raise FileNotFoundError(f"model_independent missing {query_id!r}")
        for key, val in flatten_query_row(phi_row).items():
            if val is not None:
                out[key] = float(val)

    psi_path = run_root / "signals" / f"model_response_{response_role}.jsonl"
    psi_found = False
    for rec in load_signals(psi_path):
        if rec.query_id != query_id:
            continue
        psi_found = True
        for key in RESPONSE_COLUMNS:
            raw_key = key.removeprefix("psi.")
            raw = rec.metrics.get(raw_key)
            if raw is None:
                continue
            out[key] = float(raw) if not isinstance(raw, bool) else float(raw)
        break
    if not psi_found:
        raise FileNotFoundError(f"model_response_{response_role} missing {query_id!r}")

    return out


def vectorize_features(
    features: dict[str, float],
    columns: tuple[str, ...],
) -> list[float]:
    """Build a feature vector in policy column order; missing values raise."""
    missing = [col for col in columns if col not in features]
    if missing:
        raise KeyError(f"runtime features missing columns: {missing[:3]}")
    return [float(features[col]) for col in columns]


def default_runtime_columns(*, include_query: bool = False) -> tuple[str, ...]:
    if include_query:
        return QUERY_COLUMNS + RESPONSE_COLUMNS
    return RESPONSE_COLUMNS
