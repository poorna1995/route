"""Pre-flight checks for oracle, probe, feature, and merged artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from routing.constants import (
    BUCKET_ORDER,
    COL_ROW_UID,
    COMPLEXITY_SELECTION_SCHEMA,
    DEFAULT_TOKENIZER_ID,
    FEATURE_NAMES,
    PROBE_METRIC_BASES,
)
from routing.data import (
    MERGED_TABLE_SCHEMA,
    load_complexity_selection,
    load_oracle,
    merge_tables,
    normalize_signals_csv,
)
from routing.model_utils import load_tokenizer
from routing.prompt_protocol import PROTOCOL_VERSION as PROMPT_PROTOCOL_VERSION

CheckStatus = Literal["pass", "warn", "fail", "skip"]


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    detail: str = ""


@dataclass
class DoctorReport:
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, status: CheckStatus, detail: str = "") -> None:
        self.checks.append(CheckResult(name=name, status=status, detail=detail))

    @property
    def ok(self) -> bool:
        return all(c.status in ("pass", "warn", "skip") for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail}
                for c in self.checks
            ],
        }


def _read_ids(path: Path, *, col: str = "query_id") -> pd.Series:
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        rows = payload.get("rows", [])
        return pd.Series([r.get("id", r.get("query_id")) for r in rows], dtype=str)
    df = pd.read_csv(path)
    if col not in df.columns and "id" in df.columns:
        col = "id"
    return df[col].astype(str)


def _check_unique_ids(report: DoctorReport, label: str, ids: pd.Series) -> None:
    if ids.is_unique:
        report.add(f"{label}_ids_unique", "pass", f"n={len(ids)}")
    else:
        dupes = ids[ids.duplicated()].head(5).tolist()
        report.add(f"{label}_ids_unique", "fail", f"duplicate ids: {dupes}")


def run_doctor(
    *,
    oracle: Path | None = None,
    weak_csv: Path | None = None,
    strong_csv: Path | None = None,
    merged_csv: Path | None = None,
    features_csv: Path | None = None,
    complexity_selection: Path | None = None,
    tokenizer_model: str = DEFAULT_TOKENIZER_ID,
    strict: bool = False,
) -> DoctorReport:
    report = DoctorReport()

    # Tokenizer
    try:
        tok = load_tokenizer(tokenizer_model)
        report.add("tokenizer", "pass", getattr(tok, "name_or_path", tokenizer_model))
    except Exception as exc:
        report.add("tokenizer", "fail", str(exc))
        return report

    report.add("protocol_version", "pass", PROMPT_PROTOCOL_VERSION)

    oracle_df = None
    weak_df = None
    strong_df = None

    if oracle is not None:
        if not oracle.exists():
            report.add("oracle_exists", "fail", str(oracle))
        else:
            try:
                oracle_df = load_oracle(oracle)
                _check_unique_ids(report, "oracle", oracle_df["query_id"])
                counts = oracle_df["bucket"].value_counts().to_dict()
                report.add("oracle_buckets", "pass", ", ".join(f"{k}={counts.get(k, 0)}" for k in BUCKET_ORDER))
                required = {"weak_ok", "strong_ok", "bucket", "gold"}
                missing = required - set(oracle_df.columns)
                if missing:
                    report.add("oracle_schema", "fail", f"missing columns: {sorted(missing)}")
                else:
                    report.add("oracle_schema", "pass", f"n={len(oracle_df)}")
                if COL_ROW_UID in oracle_df.columns:
                    _check_unique_ids(report, "oracle_row_uid", oracle_df[COL_ROW_UID].astype(str))
                else:
                    report.add("oracle_row_uid", "warn", "column missing — re-run oracle to add row_uid")
            except Exception as exc:
                report.add("oracle_load", "fail", str(exc))

    for label, path in (("weak", weak_csv), ("strong", strong_csv)):
        if path is None:
            continue
        if not path.exists():
            report.add(f"{label}_csv_exists", "fail", str(path))
            continue
        try:
            df = normalize_signals_csv(path)
            if label == "weak":
                weak_df = df
            else:
                strong_df = df
            _check_unique_ids(report, label, df["query_id"])
            method = df.get("extraction_method", pd.Series(dtype=object)).iloc[0]
            protocol = df.get("protocol_version", pd.Series(dtype=object)).iloc[0]
            protocol_s = str(protocol)
            if protocol_s != PROMPT_PROTOCOL_VERSION:
                report.add(
                    f"{label}_protocol",
                    "warn" if not strict else "fail",
                    f"protocol_version={protocol!r} (expected {PROMPT_PROTOCOL_VERSION!r})",
                )
            else:
                report.add(f"{label}_protocol", "pass", protocol_s)
            if "prompt_hash" in df.columns and df["prompt_hash"].notna().all():
                report.add(f"{label}_prompt_hash", "pass", "present on all rows")
            else:
                report.add(f"{label}_prompt_hash", "warn", "missing or partial prompt_hash")
            headline = [c for c in ("entropy", "margin") if c in df.columns]
            if len(headline) == 2 and df["entropy"].notna().all() and df["margin"].notna().all():
                report.add(f"{label}_headline_signals", "pass", f"entropy, margin ({method})")
            else:
                report.add(f"{label}_headline_signals", "fail", "entropy or margin missing")
        except Exception as exc:
            report.add(f"{label}_csv_load", "fail", str(exc))

    if oracle_df is not None and weak_df is not None:
        oset = set(oracle_df["query_id"])
        wset = set(weak_df["query_id"])
        if oset == wset:
            report.add("oracle_weak_id_overlap", "pass", f"n={len(oset)}")
        else:
            report.add(
                "oracle_weak_id_overlap",
                "fail",
                f"oracle-only={len(oset - wset)} weak-only={len(wset - oset)}",
            )

    if oracle_df is not None and strong_df is not None:
        oset = set(oracle_df["query_id"])
        sset = set(strong_df["query_id"])
        if oset == sset:
            report.add("oracle_strong_id_overlap", "pass", f"n={len(oset)}")
        else:
            report.add(
                "oracle_strong_id_overlap",
                "fail",
                f"oracle-only={len(oset - sset)} strong-only={len(sset - oset)}",
            )

    if weak_df is not None and strong_df is not None:
        wset = set(weak_df["query_id"])
        sset = set(strong_df["query_id"])
        if wset == sset:
            report.add("weak_strong_id_overlap", "pass", f"n={len(wset)}")
        else:
            report.add("weak_strong_id_overlap", "fail", f"symmetric diff={len(wset ^ sset)}")

        if "prompt_hash" in weak_df.columns and "prompt_hash" in strong_df.columns:
            merged_ids = weak_df.merge(strong_df, on="query_id", suffixes=("_w", "_s"))
            mismatch = merged_ids["prompt_hash_w"] != merged_ids["prompt_hash_s"]
            if mismatch.any():
                ids = merged_ids.loc[mismatch, "query_id"].head(5).tolist()
                report.add(
                    "probe_prompt_hash_parity",
                    "warn",
                    f"weak≠strong prompt_hash on {mismatch.sum()} rows (expected if tokenizers differ): {ids}",
                )
            else:
                report.add("probe_prompt_hash_parity", "pass", "weak == strong per query")

    if oracle_df is not None and weak_df is not None and "prompt_hash_weak" in oracle_df.columns:
        probe_map = weak_df.set_index("query_id")["prompt_hash"]
        oracle_map = oracle_df.set_index("query_id")["prompt_hash_weak"]
        common = probe_map.index.intersection(oracle_map.index)
        if len(common):
            bad = common[probe_map.loc[common] != oracle_map.loc[common]]
            if len(bad):
                report.add(
                    "oracle_weak_prompt_hash",
                    "fail",
                    f"mismatch on {list(bad[:5])}",
                )
            else:
                report.add("oracle_weak_prompt_hash", "pass", f"verified n={len(common)}")

    if oracle_df is not None and strong_df is not None and "prompt_hash_strong" in oracle_df.columns:
        probe_map = strong_df.set_index("query_id")["prompt_hash"]
        oracle_map = oracle_df.set_index("query_id")["prompt_hash_strong"]
        common = probe_map.index.intersection(oracle_map.index)
        if len(common):
            bad = common[probe_map.loc[common] != oracle_map.loc[common]]
            if len(bad):
                report.add(
                    "oracle_strong_prompt_hash",
                    "fail",
                    f"mismatch on {list(bad[:5])}",
                )
            else:
                report.add("oracle_strong_prompt_hash", "pass", f"verified n={len(common)}")

    if features_csv is not None:
        if not features_csv.exists():
            report.add("features_csv_exists", "fail", str(features_csv))
        else:
            feats = pd.read_csv(features_csv)
            _check_unique_ids(report, "features", feats["query_id"])
            if COL_ROW_UID in feats.columns:
                _check_unique_ids(report, "features_row_uid", feats[COL_ROW_UID].astype(str))
            else:
                report.add("features_row_uid", "warn", "column missing — re-run features to add row_uid")
            missing_feats = [c for c in FEATURE_NAMES if c not in feats.columns]
            if missing_feats:
                report.add("features_columns", "fail", f"missing {missing_feats}")
            else:
                report.add("features_columns", "pass", f"n={len(feats)}")

    if complexity_selection is not None:
        if not complexity_selection.exists():
            report.add("complexity_selection_exists", "fail", str(complexity_selection))
        else:
            try:
                sel = load_complexity_selection(complexity_selection)
                report.add(
                    "complexity_selection",
                    "pass",
                    f"{sel['selected_feature']!r} (schema={COMPLEXITY_SELECTION_SCHEMA})",
                )
                if features_csv is not None:
                    feats = pd.read_csv(features_csv)
                    if sel["selected_feature"] not in feats.columns:
                        report.add(
                            "selection_feature_in_csv",
                            "fail",
                            f"{sel['selected_feature']!r} not in features CSV",
                        )
                    else:
                        report.add("selection_feature_in_csv", "pass", sel["selected_feature"])
            except Exception as exc:
                report.add("complexity_selection", "fail", str(exc))

    if merged_csv is not None:
        if not merged_csv.exists():
            report.add("merged_csv_exists", "fail", str(merged_csv))
        else:
            merged = pd.read_csv(merged_csv)
            _check_unique_ids(report, "merged", merged["query_id"])
            if "bucket" in merged.columns:
                counts = merged["bucket"].value_counts().to_dict()
                report.add(
                    "merged_buckets",
                    "pass",
                    ", ".join(f"{k}={counts.get(k, 0)}" for k in BUCKET_ORDER),
                )
            for col in ("entropy_w", "margin_w", "entropy_s", "margin_s"):
                if col not in merged.columns:
                    report.add("merged_headline_signals", "fail", f"missing {col}")
                    break
            else:
                report.add("merged_headline_signals", "pass", "entropy and margin present")
            if "c_q" in merged.columns:
                report.add("merged_c_q", "pass", "c_q present")
            elif complexity_selection is not None:
                report.add("merged_c_q", "warn", "c_q absent (expected when selection provided)")

    # Dry-run merge when all inputs present
    if oracle_df is not None and weak_df is not None and strong_df is not None:
        try:
            qf = None
            if features_csv is not None and complexity_selection is not None:
                from routing.data import load_complexity_from_selection

                qf = load_complexity_from_selection(features_csv, complexity_selection)
            merge_tables(weak_df, strong_df, oracle_df, query_features=qf)
            report.add("merge_dry_run", "pass", f"schema={MERGED_TABLE_SCHEMA}")
        except Exception as exc:
            report.add("merge_dry_run", "fail", str(exc))

    return report
