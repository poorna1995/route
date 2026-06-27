"""Load and merge oracle labels with probe signals and query features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from routing.constants import (
    BUCKET_ORDER,
    COL_COMPLEXITY,
    COL_COMPLEXITY_CANDIDATE,
    COL_ROW_UID,
    COMPLEXITY_SELECTION_SCHEMA,
    FEATURE_NAMES,
    PROBE_DERIVED_MARGIN,
    PROBE_DERIVED_SLOPE,
    PROBE_DERIVED_STAB,
    PROBE_FORMATION_BASES,
    PROBE_METHOD,
    PROBE_METRIC_BASES,
    SCREEN_TAG,
)
from routing.prompt_protocol import PROTOCOL_VERSION

MERGED_TABLE_SCHEMA = "merged_v2"

# Delta conventions (documented for readers of merged CSV):
#   delta_entropy, delta_* (except margin): weak − strong
#   delta_margin_gain: strong − weak (margin gain on strong vs weak)


def load_complexity_column(path: Path, *, source_column: str) -> pd.DataFrame:
    """Map D46 winning candidate column to unified c_q."""
    feats = pd.read_csv(path)
    if source_column not in feats.columns:
        raise ValueError(f"features CSV missing column {source_column!r}")
    out = feats[["query_id", source_column]].copy()
    out = out.rename(columns={source_column: COL_COMPLEXITY})
    out[COL_COMPLEXITY_CANDIDATE] = source_column
    return out


def write_complexity_selection(
    path: Path,
    *,
    selected_feature: str,
    calib_locked: bool,
    n: int,
    screen_report: Path,
    features_path: Path | None = None,
    oracle_path: Path | None = None,
) -> None:
    """Persist frozen D46 winner for downstream merge (run once on CALIB)."""
    if selected_feature not in FEATURE_NAMES:
        raise ValueError(f"selected_feature {selected_feature!r} not in {FEATURE_NAMES}")
    payload: dict[str, Any] = {
        "schema": COMPLEXITY_SELECTION_SCHEMA,
        "screen_version": SCREEN_TAG,
        "selected_feature": selected_feature,
        "calib_locked": calib_locked,
        "n": n,
        "screen_report": str(screen_report.resolve()),
    }
    if features_path is not None:
        payload["features_csv"] = str(features_path.resolve())
    if oracle_path is not None:
        payload["oracle"] = str(oracle_path.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_complexity_selection(path: Path) -> dict[str, Any]:
    """Load frozen D46 selection; merge must use this instead of ad-hoc --c-q-column."""
    payload = json.loads(path.read_text())
    schema = payload.get("schema")
    if schema != COMPLEXITY_SELECTION_SCHEMA:
        raise ValueError(
            f"unsupported complexity selection schema {schema!r} in {path} "
            f"(expected {COMPLEXITY_SELECTION_SCHEMA!r})"
        )
    selected = payload.get("selected_feature")
    if selected not in FEATURE_NAMES:
        raise ValueError(f"invalid selected_feature {selected!r} in {path}")
    if not payload.get("calib_locked"):
        raise ValueError(
            f"D46 selection in {path} is not calib_locked — preview only; "
            "re-run screen on full ARC validation (n=299) without --allow-preview before merge"
        )
    return payload


def load_complexity_from_selection(
    features_path: Path,
    selection_path: Path,
) -> pd.DataFrame:
    """Map frozen D46 column → unified c_q for any split's features CSV.

    ``selected_feature.json`` records which CALIB file was screened (provenance);
    merge on TEST (or CALIB) passes that split's features file with the same column.
    """
    selection = load_complexity_selection(selection_path)
    return load_complexity_column(features_path, source_column=selection["selected_feature"])


def normalize_signals_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"empty signal CSV: {path}")

    if "confidence" in df.columns and "max_prob" not in df.columns:
        df = df.rename(columns={"confidence": "max_prob"})
    if "probe_mode" in df.columns and "extraction_method" not in df.columns:
        df = df.rename(columns={"probe_mode": "extraction_method"})

    for col in list(PROBE_METRIC_BASES) + ["vocab_size"]:
        if col not in df.columns:
            df[col] = np.nan

    if df["n_eff"].isna().all() and df["entropy"].notna().all():
        df["n_eff"] = np.exp(df["entropy"])
    if df["entropy_norm"].isna().all() and df["vocab_size"].notna().any():
        vocab = df["vocab_size"].astype(float)
        denom = np.log(vocab.where(vocab > 1, np.nan))
        df["entropy_norm"] = df["entropy"] / denom

    df["schema"] = _detect_schema(df)
    return df


def _detect_schema(df: pd.DataFrame) -> str:
    has_new = any(col in df.columns and df[col].notna().any() for col in ("entropy_norm", "top5_mass", "vocab_size"))
    has_legacy = (
        ("confidence" in df.columns or "probe_mode" in df.columns)
        or df.get("extraction_method", pd.Series(dtype=object)).eq("prefill").any()
    )
    if has_new:
        return "current"
    if has_legacy:
        return "legacy"
    return "unknown"


def load_oracle(path: Path, *, include_gap: bool = True) -> pd.DataFrame:
    """Load oracle JSON rows as a DataFrame keyed by query_id."""
    payload = json.loads(path.read_text())
    rows = payload.get("rows")
    if not rows:
        raise ValueError(f"oracle JSON has no rows: {path}")
    oracle = pd.DataFrame(rows).rename(columns={"id": "query_id"})
    oracle["y_opp"] = (oracle["bucket"] == "opportunity").astype(int)
    if include_gap:
        oracle["oracle_gap"] = oracle["strong_ok"].astype(int) - oracle["weak_ok"].astype(int)
    return oracle


def _assert_unique_query_ids(df: pd.DataFrame, label: str) -> None:
    if not df["query_id"].is_unique:
        dupes = df.loc[df["query_id"].duplicated(), "query_id"].unique()[:5]
        raise ValueError(f"{label}: duplicate query_id values: {list(dupes)}")


def _verify_oracle_probe_prompt_hashes(merged: pd.DataFrame) -> None:
    """Oracle generation and probe extraction must use the same chat prompt."""
    pairs = [
        ("prompt_hash_w", "prompt_hash_weak", "weak"),
        ("prompt_hash_s", "prompt_hash_strong", "strong"),
    ]
    for probe_col, oracle_col, label in pairs:
        if probe_col not in merged.columns or oracle_col not in merged.columns:
            continue
        both = merged[probe_col].notna() & merged[oracle_col].notna()
        if not both.any():
            continue
        mismatch = both & (merged[probe_col] != merged[oracle_col])
        if mismatch.any():
            ids = merged.loc[mismatch, "query_id"].head(5).tolist()
            raise ValueError(
                f"prompt_hash mismatch for {label} model — oracle vs probe used different "
                f"chat prompts (check protocol version / user_content): {ids}"
            )


def _verify_prompt_hashes(merged: pd.DataFrame) -> None:
    if "prompt_hash_w" not in merged.columns or "prompt_hash_s" not in merged.columns:
        return
    both = merged["prompt_hash_w"].notna() & merged["prompt_hash_s"].notna()
    if not both.any():
        return
    mismatch = both & (merged["prompt_hash_w"] != merged["prompt_hash_s"])
    if mismatch.any():
        ids = merged.loc[mismatch, "query_id"].head(5).tolist()
        raise ValueError(
            "prompt_hash mismatch between weak and strong probes — "
            f"different chat prompts for same query_id: {ids}"
        )


def _suffix_columns(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    keep = ["query_id"] + [c for c in PROBE_METRIC_BASES if c in df.columns]
    keep += [c for c in PROBE_FORMATION_BASES if c in df.columns]
    meta = [
        c
        for c in ("model_id", "prompt_hash", "protocol_version", "extraction_method", "schema")
        if c in df.columns
    ]
    out = df[keep + meta].copy()
    rename = {c: f"{c}_{suffix}" for c in PROBE_METRIC_BASES if c in out.columns}
    rename.update({c: f"{c}_{suffix}" for c in PROBE_FORMATION_BASES if c in out.columns})
    rename.update({c: f"{c}_{suffix}" for c in meta if c != "query_id"})
    return out.rename(columns=rename)


def _attach_table_metadata(
    merged: pd.DataFrame,
    *,
    weak: pd.DataFrame,
    strong: pd.DataFrame,
    query_features: pd.DataFrame | None,
) -> pd.DataFrame:
    complexity_candidate = None
    if query_features is not None and COL_COMPLEXITY_CANDIDATE in query_features.columns:
        complexity_candidate = query_features[COL_COMPLEXITY_CANDIDATE].iloc[0]

    protocol_w = merged.get("protocol_version_w", pd.Series([None])).iloc[0]
    protocol_s = merged.get("protocol_version_s", pd.Series([None])).iloc[0]
    protocol = protocol_w if protocol_w == protocol_s else protocol_w

    merged.attrs["table_schema"] = MERGED_TABLE_SCHEMA
    merged.attrs["protocol_version"] = str(protocol or PROTOCOL_VERSION)
    merged.attrs["probe_method"] = PROBE_METHOD
    merged.attrs["schema_weak"] = merged.get("schema_w", pd.Series(["unknown"])).iloc[0]
    merged.attrs["schema_strong"] = merged.get("schema_s", pd.Series(["unknown"])).iloc[0]
    if complexity_candidate is not None:
        merged.attrs["complexity_candidate"] = str(complexity_candidate)
    merged.attrs["delta_conventions"] = {
        "default": "weak_minus_strong",
        PROBE_DERIVED_MARGIN: "strong_minus_weak",
        PROBE_DERIVED_SLOPE: "strong_minus_weak",
        PROBE_DERIVED_STAB: "strong_minus_weak",
    }
    return merged


def merge_tables(
    weak: pd.DataFrame,
    strong: pd.DataFrame,
    oracle: pd.DataFrame,
    *,
    query_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    _assert_unique_query_ids(oracle, "oracle")
    _assert_unique_query_ids(weak, "weak signals")
    _assert_unique_query_ids(strong, "strong signals")
    if query_features is not None:
        _assert_unique_query_ids(query_features, "query features")

    w = _suffix_columns(weak, "w")
    s = _suffix_columns(strong, "s")
    merged = oracle.merge(w, on="query_id", how="inner").merge(s, on="query_id", how="inner")

    if query_features is not None:
        merged = merged.merge(query_features, on="query_id", how="inner")

    if len(merged) == 0:
        raise ValueError("no overlapping query_id across oracle and signal CSVs")

    if len(merged) != len(oracle):
        raise ValueError(
            f"merge row count {len(merged)} != oracle {len(oracle)} — "
            "check query_id alignment across oracle and probe CSVs"
        )

    _verify_prompt_hashes(merged)
    _verify_oracle_probe_prompt_hashes(merged)

    for base in PROBE_METRIC_BASES:
        wc, sc = f"{base}_w", f"{base}_s"
        if wc not in merged.columns or sc not in merged.columns:
            continue
        if base == "margin":
            merged[PROBE_DERIVED_MARGIN] = merged[sc] - merged[wc]
        else:
            merged[f"delta_{base}"] = merged[wc] - merged[sc]

    for base in PROBE_FORMATION_BASES:
        wc, sc = f"{base}_w", f"{base}_s"
        if wc not in merged.columns or sc not in merged.columns:
            continue
        col = PROBE_DERIVED_SLOPE if base == "slope_margin" else PROBE_DERIVED_STAB
        merged[col] = merged[sc] - merged[wc]

    return _attach_table_metadata(merged, weak=weak, strong=strong, query_features=query_features)


def list_signal_columns(merged: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    if COL_COMPLEXITY in merged.columns:
        candidates.append(COL_COMPLEXITY)
    for base in PROBE_METRIC_BASES:
        candidates.extend([f"{base}_w", f"{base}_s"])
    for base in PROBE_FORMATION_BASES:
        candidates.extend([f"{base}_w", f"{base}_s"])
    candidates.extend([c for c in merged.columns if c.startswith("delta_")])

    seen: set[str] = set()
    signal_cols: list[str] = []
    for col in candidates:
        if col in merged.columns and col not in seen:
            seen.add(col)
            signal_cols.append(col)
    return signal_cols


def bucket_summary(merged: pd.DataFrame) -> dict[str, Any]:
    counts = merged["bucket"].value_counts().to_dict()
    n = len(merged)
    rates = {k: v / n for k, v in counts.items()}
    return {"n": n, "counts": counts, "rates": rates}


def distribution_by_bucket(merged: pd.DataFrame, signal_col: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for bucket in BUCKET_ORDER:
        mask = merged["bucket"] == bucket
        out[bucket] = _summary_stats(merged.loc[mask, signal_col].to_numpy(dtype=float))
    return out


def _summary_stats(values: np.ndarray) -> dict[str, Any]:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return {"n": 0, "mean": None, "std": None, "median": None, "q25": None, "q75": None}
    return {
        "n": int(len(v)),
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
        "median": float(np.median(v)),
        "q25": float(np.quantile(v, 0.25)),
        "q75": float(np.quantile(v, 0.75)),
    }
