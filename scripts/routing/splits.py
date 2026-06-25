"""CALIB / TEST split helpers for holdout evaluation and stability analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from routing.constants import BOOTSTRAP_SEED
from routing.datasets import (
    dataset_split_policy,
    load_queries,
    normalize_dataset_name,
    policy_subjects,
    split_source_metadata,
)

SPLITS_SCHEMA_VERSION = "splits_v1"


def load_splits(path: Path) -> dict[str, Any]:
    """Load frozen split manifest: calib + test query_id lists."""
    payload = json.loads(path.read_text())
    calib = payload.get("calib")
    test = payload.get("test")
    if not calib or not test:
        raise ValueError(f"splits JSON must contain 'calib' and 'test' lists: {path}")
    calib_set = set(str(x) for x in calib)
    test_set = set(str(x) for x in test)
    overlap = calib_set & test_set
    if overlap:
        raise ValueError(f"calib and test overlap in {path}: {list(overlap)[:5]}")
    calib_source_split = (
        payload.get("calib_source_split")
        or payload.get("calib_split")
    )
    eval_source_split = (
        payload.get("eval_source_split")
        or payload.get("test_split")
    )
    return {
        "calib": calib_set,
        "test": test_set,
        "seed": payload.get("seed"),
        "calib_size": payload.get("calib_size", len(calib_set)),
        "test_size": payload.get("test_size", len(test_set)),
        "protocol": payload.get("protocol") or payload.get("policy"),
        "policy": payload.get("policy"),
        "calib_source_split": calib_source_split,
        "eval_source_split": eval_source_split,
        # Backward-compatible aliases
        "calib_split": calib_source_split,
        "test_split": eval_source_split,
        "dataset": payload.get("dataset"),
        "source_dataset": payload.get("source_dataset"),
        "policy_config": payload.get("policy_config"),
        "schema_version": payload.get("schema_version") or payload.get("schema"),
    }


def write_splits(
    path: Path,
    *,
    calib: list[str],
    test: list[str],
    seed: int,
    dataset: str | None = None,
    protocol: str | None = None,
    policy: str | None = None,
    calib_split: str | None = None,
    test_split: str | None = None,
    source_dataset: dict[str, Any] | None = None,
    policy_config: dict[str, Any] | None = None,
) -> None:
    overlap = set(calib) & set(test)
    if overlap:
        raise ValueError(f"calib/test overlap: {list(overlap)[:5]}")
    payload: dict[str, Any] = {
        "schema_version": SPLITS_SCHEMA_VERSION,
        "seed": seed,
        "calib_size": len(calib),
        "test_size": len(test),
        "calib": calib,
        "test": test,
    }
    if dataset:
        payload["dataset"] = dataset
    if protocol:
        payload["protocol"] = protocol
    if policy:
        payload["policy"] = policy
    # Canonical names.
    if calib_split:
        payload["calib_source_split"] = calib_split
    if test_split:
        payload["eval_source_split"] = test_split
    # Backward-compatible aliases.
    if calib_split:
        payload["calib_split"] = calib_split
    if test_split:
        payload["test_split"] = test_split
    if source_dataset:
        payload["source_dataset"] = source_dataset
    if policy_config:
        payload["policy_config"] = policy_config
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_policy_splits(
    path: Path,
    *,
    dataset: str,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Create splits using dataset-specific calibration/evaluation policy."""
    ds = normalize_dataset_name(dataset)
    policy = dataset_split_policy(ds)
    calib_split = policy["calib_split"]
    eval_split = policy["eval_split"]
    subjects = policy_subjects(ds, policy)
    calib_meta = split_source_metadata(ds, calib_split, subjects)
    eval_meta = split_source_metadata(ds, eval_split, subjects)
    calib_limit = int(calib_meta["num_rows"])
    eval_limit = int(eval_meta["num_rows"])
    calib_q = load_queries(ds, calib_split, calib_limit, seed, policy=policy)
    test_q = load_queries(ds, eval_split, eval_limit, seed, policy=policy)
    calib_ids = [q["id"] for q in calib_q]
    test_ids = [q["id"] for q in test_q]
    overlap = set(calib_ids) & set(test_ids)
    if overlap:
        raise ValueError(
            f"Policy produced overlapping ids for {ds}: {len(overlap)}"
        )

    write_splits(
        path,
        calib=calib_ids,
        test=test_ids,
        seed=seed,
        dataset=ds,
        protocol="dataset_split_policy_v1",
        policy=policy["policy"],
        calib_split=calib_split,
        test_split=eval_split,
        source_dataset={
            "dataset_key": ds,
            "calib": calib_meta,
            "eval": eval_meta,
        },
        policy_config=policy,
    )
    return {
        "dataset": ds,
        "calib": calib_ids,
        "test": test_ids,
        "calib_split": calib_split,
        "test_split": eval_split,
        "policy": policy["policy"],
        "policy_config": policy,
        "protocol": "dataset_split_policy_v1",
    }


def resolve_hf_split_and_limit(
    *,
    splits: dict[str, Any] | None,
    split_role: str | None,
    default_split: str,
    default_limit: int,
) -> tuple[str, int]:
    """Pick HF split + limit when --splits-json and --split-role are set."""
    if splits is None or split_role is None:
        return default_split, default_limit
    if split_role == "calib":
        hf_split = (
            splits.get("calib_source_split")
            or splits.get("calib_split")
            or default_split
        )
        limit = int(splits.get("calib_size", len(splits["calib"])))
    elif split_role == "test":
        hf_split = (
            splits.get("eval_source_split")
            or splits.get("test_split")
            or default_split
        )
        limit = int(splits.get("test_size", len(splits["test"])))
    else:
        raise ValueError(f"split_role must be 'calib' or 'test', got {split_role!r}")
    return hf_split, limit


def draw_calib_ids(
    pool: list[str],
    *,
    calib_size: int,
    seed: int,
    exclude: set[str] | None = None,
) -> set[str]:
    """Random CALIB subset from pool (optionally excluding fixed TEST ids)."""
    candidates = [qid for qid in pool if exclude is None or qid not in exclude]
    if len(candidates) < calib_size:
        raise ValueError(
            f"cannot draw calib_size={calib_size} from pool n={len(candidates)} "
            f"(exclude={len(exclude or set())})"
        )
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(candidates), size=calib_size, replace=False)
    return {candidates[int(i)] for i in idx}


def query_ids_for_role(splits: dict[str, Any], role: str) -> set[str]:
    """Return query_id set for split role: calib | test."""
    if role == "calib":
        return splits["calib"]
    if role == "test":
        return splits["test"]
    raise ValueError(f"split_role must be 'calib' or 'test', got {role!r}")
