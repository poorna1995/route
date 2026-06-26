"""Benchmark dataset loaders → frozen `user_content` rows."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import numpy as np
from datasets import load_dataset

from routing.prompt_protocol import (
    arc_gold_letter,
    format_arc_question,
    format_mmlu_question,
    mmlu_gold_letter,
)

MMLU_DATASET_ID = "cais/mmlu"
DEFAULT_MMLU_SUBJECTS = ("high_school_physics", "logical_fallacies")

SUPPORTED_DATASETS = frozenset({"arc_challenge", "arc-challenge", "mmlu"})


@lru_cache(maxsize=None)
def _load_hf_split(dataset_id: str, config: str | None, split: str):
    """Cached Hugging Face split load (shared by metadata + query loaders)."""
    if config is None:
        return load_dataset(dataset_id, split=split)
    return load_dataset(dataset_id, config, split=split)


def normalize_dataset_name(dataset: str) -> str:
    """Canonical dataset key used across loaders/split policy."""
    return dataset.lower().replace("-", "_")


def dataset_split_policy(dataset: str) -> dict[str, Any]:
    """Calibration/evaluation split policy by dataset.

    Priority:
    1) official validation/dev -> calibration, official test -> evaluation
    2) official train -> calibration, official eval split -> evaluation
    3) single split -> internal partition (handled by splits.py)
    """
    ds = normalize_dataset_name(dataset)
    if ds == "arc_challenge":
        return {
            "policy": "official_validation_to_calib",
            "calib_split": "validation",
            "eval_split": "test",
            "calib_size": None,
            "eval_size": None,
        }
    if ds == "gsm8k":
        return {
            "policy": "train_to_calib",
            "calib_split": "train",
            "eval_split": "test",
            "calib_size": None,
            "eval_size": None,
        }
    if ds == "mmlu":
        return {
            "policy": "test_only_transfer",
            "calib_split": None,
            "eval_split": "test",
            "calib_size": None,
            "eval_size": None,
            "subjects": list(DEFAULT_MMLU_SUBJECTS),
        }
    raise ValueError(
        f"Unsupported dataset: {dataset}. Supported: {', '.join(sorted(SUPPORTED_DATASETS))}"
    )


def policy_subjects(
    dataset: str,
    policy: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    """Subject list for multi-subject benchmarks; empty for single-config datasets."""
    cfg = policy or dataset_split_policy(normalize_dataset_name(dataset))
    subjects = cfg.get("subjects")
    if not subjects:
        return ()
    return tuple(str(s) for s in subjects)


def _arc_query_id(row: dict) -> str:
    """Stable ARC id from the dataset's native row id."""
    raw_id = row.get("id")
    if raw_id is None:
        raise ValueError("ARC row missing native id")
    return f"arc_{raw_id}"


def make_row_uid(dataset_key: str, split: str, source_id: str) -> str:
    """Globally unique row id from upstream benchmark (distinct from prefixed query_id)."""
    return f"{dataset_key}/{split}/{source_id}"


def _normalize_number(num: str | None) -> str | None:
    if num is None:
        return None
    try:
        return str(int(float(num)))
    except ValueError:
        return num.strip()


def _extract_gsm8k_gold(text: str) -> str | None:
    if "####" in text:
        tail = text.split("####")[-1].strip()
        nums = re.findall(r"-?\d+", tail.replace(",", ""))
        return nums[0] if nums else None
    nums = re.findall(r"-?\d+", text.replace(",", ""))
    return nums[-1] if nums else None


def load_arc_queries(split: str, limit: int, seed: int) -> list[dict]:
    ds = _load_hf_split("allenai/ai2_arc", "ARC-Challenge", split)
    if limit < len(ds):
        ds = ds.shuffle(seed=seed).select(range(limit))
    else:
        ds = ds.select(range(len(ds)))
    return [
        {
            "id": _arc_query_id(row),
            "row_uid": make_row_uid("arc_challenge", split, str(row["id"])),
            "user_content": format_arc_question(row),
            "gold": arc_gold_letter(row),
            "eval": "letter",
        }
        for row in ds
    ]


def load_mmlu_queries(
    split: str,
    limit: int,
    seed: int,
    *,
    subjects: tuple[str, ...] = DEFAULT_MMLU_SUBJECTS,
) -> list[dict]:
    """Load pooled MMLU subjects with frozen MCQ formatting (transfer eval on ``test``)."""
    rows: list[dict] = []
    for subject in subjects:
        ds = _load_hf_split(MMLU_DATASET_ID, subject, split)
        for idx, row in enumerate(ds):
            rows.append(
                {
                    "id": f"mmlu_{subject}_{idx}",
                    "row_uid": make_row_uid("mmlu", split, f"{subject}/{idx}"),
                    "user_content": format_mmlu_question(row),
                    "gold": mmlu_gold_letter(row),
                    "eval": "letter",
                    "subject": subject,
                }
            )
    if not rows:
        raise ValueError(f"No MMLU rows for split={split!r} subjects={subjects}")

    order = np.random.default_rng(seed).permutation(len(rows))
    if limit < len(rows):
        order = order[:limit]
    return [rows[int(i)] for i in order]


def load_queries(
    dataset: str,
    split: str,
    limit: int,
    seed: int,
    *,
    policy: dict[str, Any] | None = None,
) -> list[dict]:
    """Load benchmark rows with frozen task formatting in `user_content`."""
    dataset = normalize_dataset_name(dataset)
    if dataset == "arc_challenge":
        return load_arc_queries(split, limit, seed)
    if dataset == "mmlu":
        cfg = policy or dataset_split_policy(dataset)
        return load_mmlu_queries(split, limit, seed, subjects=policy_subjects(dataset, cfg))

    raise ValueError(
        f"Unsupported dataset: {dataset}. Supported: {', '.join(sorted(SUPPORTED_DATASETS))}"
    )


@lru_cache(maxsize=None)
def split_source_metadata(
    dataset: str,
    split: str,
    subjects: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Reproducibility metadata for a dataset split.

    Includes source dataset id/config, split row counts, and fingerprints.
    Cached per (dataset, split) for the process lifetime.
    """
    ds = normalize_dataset_name(dataset)
    if ds == "arc_challenge":
        raw = _load_hf_split("allenai/ai2_arc", "ARC-Challenge", split)
        version = getattr(getattr(raw, "info", None), "version", None)
        return {
            "dataset_key": ds,
            "dataset_id": "allenai/ai2_arc",
            "dataset_config": "ARC-Challenge",
            "split": split,
            "num_rows": int(len(raw)),
            "revision": str(version) if version is not None else None,
            "fingerprint": getattr(raw, "_fingerprint", None),
        }
    if ds == "gsm8k":
        raw = _load_hf_split("openai/gsm8k", "main", split)
        version = getattr(getattr(raw, "info", None), "version", None)
        return {
            "dataset_key": ds,
            "dataset_id": "openai/gsm8k",
            "dataset_config": "main",
            "split": split,
            "num_rows": int(len(raw)),
            "revision": str(version) if version is not None else None,
            "fingerprint": getattr(raw, "_fingerprint", None),
        }
    if ds == "mmlu":
        subs = subjects or DEFAULT_MMLU_SUBJECTS
        subject_row_counts: dict[str, int] = {}
        total = 0
        for subject in subs:
            raw = _load_hf_split(MMLU_DATASET_ID, subject, split)
            n = int(len(raw))
            subject_row_counts[subject] = n
            total += n
        return {
            "dataset_key": ds,
            "dataset_id": MMLU_DATASET_ID,
            "dataset_config": list(subs),
            "split": split,
            "num_rows": total,
            "subject_row_counts": subject_row_counts,
            "revision": None,
            "fingerprint": None,
        }
    raise ValueError(
        f"Unsupported dataset: {dataset}. Supported: {', '.join(sorted(SUPPORTED_DATASETS))}"
    )


def filter_queries(queries: list[dict], query_ids: set[str]) -> list[dict]:
    """Keep only queries whose id is in query_ids (preserves order)."""
    out = [q for q in queries if q["id"] in query_ids]
    if not out:
        raise ValueError(f"no queries matched filter (n_ids={len(query_ids)})")
    return out


def infer_row_uid(
    query_id: str,
    *,
    split: str | None = None,
) -> str | None:
    """Best-effort row_uid for legacy rows that only stored query_id."""
    if query_id.startswith("arc_"):
        native = query_id.removeprefix("arc_")
        hf_split = split or "validation"
        return make_row_uid("arc_challenge", hf_split, native)
    if query_id.startswith("mmlu_"):
        rest = query_id.removeprefix("mmlu_")
        parts = rest.rsplit("_", 1)
        if len(parts) == 2:
            subject, idx = parts
            hf_split = split or "test"
            return make_row_uid("mmlu", hf_split, f"{subject}/{idx}")
    return None
