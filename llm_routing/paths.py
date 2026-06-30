"""Repository paths and shared partition identifiers — single source of truth."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT  # alias used by CLI and tests

EXPERIMENTS_DIR = REPO_ROOT / "experiments"
CANDIDATES_DIR = EXPERIMENTS_DIR / "candidates"
SETTING_DEFAULTS_PATH = EXPERIMENTS_DIR / "defaults.yaml"
MODEL_INDEPENDENT_DEFAULTS_PATH = EXPERIMENTS_DIR / "model_independent_defaults.yaml"
LEGACY_MODEL_INDEPENDENT_DEFAULTS_PATH = EXPERIMENTS_DIR / "query_derived_defaults.yaml"

RUNS_ROOT = EXPERIMENTS_DIR / "runs"
STABLE_RUNS_ROOT = RUNS_ROOT / "permanent"  # grouped paper-citeable runs
SELECTION_REPORT_PATH = RUNS_ROOT / "selection_report.json"
MODEL_INDEPENDENT_INDEX_PATH = RUNS_ROOT / "model_independent_index.json"
LEGACY_MODEL_INDEPENDENT_INDEX_PATH = RUNS_ROOT / "query_derived_index.json"
ROUTER_PACKAGE_DIRNAME = "router_package"
FEATURE_SPEC_FILENAME = "feature_spec.yaml"

SPLITS = ("selection_holdout", "calib", "test")
SPLIT_SUFFIX: dict[str, str] = {
    "selection_holdout": "pilot",
    "calib": "val",
    "test": "test",
}

# Back-compat aliases (deprecated names)
PERMANENT_RUNS_ROOT = STABLE_RUNS_ROOT
QUERY_DERIVED_DEFAULTS_PATH = MODEL_INDEPENDENT_DEFAULTS_PATH
QUERY_DERIVED_INDEX_PATH = MODEL_INDEPENDENT_INDEX_PATH


def resolve_model_independent_defaults_path() -> Path:
    if MODEL_INDEPENDENT_DEFAULTS_PATH.exists():
        return MODEL_INDEPENDENT_DEFAULTS_PATH
    return LEGACY_MODEL_INDEPENDENT_DEFAULTS_PATH


def resolve_model_independent_index_path() -> Path:
    if MODEL_INDEPENDENT_INDEX_PATH.exists():
        return MODEL_INDEPENDENT_INDEX_PATH
    return LEGACY_MODEL_INDEPENDENT_INDEX_PATH


def model_independent_jsonl_path(run_root: Path | str) -> Path:
    """Prefer model_independent.jsonl; fall back to legacy query_derived.jsonl."""
    run_root = Path(run_root)
    new = run_root / "signals" / "model_independent.jsonl"
    legacy = run_root / "signals" / "query_derived.jsonl"
    if new.exists() or not legacy.exists():
        return new
    return legacy
