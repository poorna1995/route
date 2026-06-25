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
    format_boolq_question,
    format_gsm8k_question,
    format_mmlu_question,
)

SUPPORTED_DATASETS = frozenset({"gsm8k", "arc_challenge", "arc-challenge", "mmlu", "boolq"})


@lru_cache(maxsize=None)
def _load_hf_split(dataset_id: str, config: str | None, split: str):
    """Cached Hugging Face split load (shared by metadata + query loaders)."""
    if config is None:
        return load_dataset(dataset_id, split=split)
    return load_dataset(dataset_id, config, split=split)


@lru_cache(maxsize=None)
def _load_mmlu_subject(subject: str, split: str):
    """Cached per-subject MMLU split load."""
    return load_dataset("cais/mmlu", subject, split=split)


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
    if ds == "mmlu":
        return {
            "policy": "official_dev_to_calib",
            "calib_split": "dev",
            "eval_split": "test",
            "calib_size": None,
            "eval_size": None,
            "subjects": [
                "high_school_physics",
                "abstract_algebra",
                "high_school_us_history",
            ],
        }
    if ds == "gsm8k":
        return {
            "policy": "train_to_calib",
            "calib_split": "train",
            "eval_split": "test",
            "calib_size": None,
            "eval_size": None,
        }
    if ds == "boolq":
        return {
            "policy": "train_to_calib_validation_to_eval",
            "calib_split": "train",
            "eval_split": "validation",
            "calib_size": None,
            "eval_size": None,
        }
    raise ValueError(
        f"Unsupported dataset: {dataset}. Supported: {', '.join(sorted(SUPPORTED_DATASETS))}"
    )


def policy_subjects(
    dataset: str,
    policy: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    """MMLU subjects from policy; empty tuple for other datasets."""
    ds = normalize_dataset_name(dataset)
    cfg = policy or dataset_split_policy(ds)
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


def _mmlu_gold_letter(row: dict) -> str:
    answer = row["answer"]
    if isinstance(answer, int):
        return ("A", "B", "C", "D")[answer]
    letter = str(answer).strip().upper()
    if letter in {"A", "B", "C", "D"}:
        return letter
    if letter.isdigit():
        return ("A", "B", "C", "D")[int(letter)]
    raise ValueError(f"Unexpected MMLU answer: {answer!r}")


def load_gsm8k_queries(split: str, limit: int, seed: int) -> list[dict]:
    ds = _load_hf_split("openai/gsm8k", "main", split)
    n = min(limit, len(ds))
    perm = np.random.default_rng(seed).permutation(len(ds))[:n]
    rows = []
    for pos, hf_i in enumerate(perm):
        row = ds[int(hf_i)]
        rows.append(
            {
                "id": f"gsm8k_{split}_{pos}",
                "row_uid": make_row_uid("gsm8k", split, str(int(hf_i))),
                "user_content": format_gsm8k_question(row["question"]),
                "gold": _normalize_number(_extract_gsm8k_gold(row["answer"])),
                "eval": "numeric",
            }
        )
    return rows


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
    subjects: tuple[str, ...],
) -> list[dict]:
    if not subjects:
        raise ValueError("MMLU load requires policy subjects")
    pool: list[dict[str, Any]] = []
    for subject in subjects:
        part = _load_mmlu_subject(subject, split)
        for hf_i, row in enumerate(part):
            pool.append(
                {
                    "row_uid": make_row_uid("mmlu", f"{subject}/{split}", str(hf_i)),
                    "user_content": format_mmlu_question(row),
                    "gold": _mmlu_gold_letter(row),
                    "eval": "letter",
                    "subject": subject,
                }
            )
    n = min(limit, len(pool))
    perm = np.random.default_rng(seed).permutation(len(pool))[:n]
    rows = []
    for pos, pool_i in enumerate(perm):
        item = pool[int(pool_i)]
        subject = item["subject"]
        rows.append(
            {
                "id": f"mmlu_{subject}_{split}_{pos}",
                **item,
            }
        )
    return rows


def load_boolq_queries(split: str, limit: int, seed: int) -> list[dict]:
    ds = _load_hf_split("google/boolq", None, split)
    n = min(limit, len(ds))
    perm = np.random.default_rng(seed).permutation(len(ds))[:n]
    rows = []
    for pos, hf_i in enumerate(perm):
        row = ds[int(hf_i)]
        gold = "yes" if bool(row["answer"]) else "no"
        rows.append(
            {
                "id": f"boolq_{split}_{pos}",
                "row_uid": make_row_uid("boolq", split, str(int(hf_i))),
                "user_content": format_boolq_question(row),
                "gold": gold,
                "eval": "bool",
            }
        )
    return rows


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

    if dataset == "gsm8k":
        return load_gsm8k_queries(split, limit, seed)

    if dataset == "arc_challenge":
        return load_arc_queries(split, limit, seed)

    if dataset == "mmlu":
        if split not in ("test", "validation", "dev"):
            split = "test"
        subjects = policy_subjects(dataset, policy)
        return load_mmlu_queries(split, limit, seed, subjects=subjects)

    if dataset == "boolq":
        if split == "test":
            split = "validation"
        return load_boolq_queries(split, limit, seed)

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
    if ds == "boolq":
        raw = _load_hf_split("google/boolq", None, split)
        version = getattr(getattr(raw, "info", None), "version", None)
        return {
            "dataset_key": ds,
            "dataset_id": "google/boolq",
            "dataset_config": None,
            "split": split,
            "num_rows": int(len(raw)),
            "revision": str(version) if version is not None else None,
            "fingerprint": getattr(raw, "_fingerprint", None),
        }
    if ds == "mmlu":
        if not subjects:
            subjects = policy_subjects(ds)
        parts: list[dict[str, Any]] = []
        total = 0
        for subject in subjects:
            part = _load_mmlu_subject(subject, split)
            version = getattr(getattr(part, "info", None), "version", None)
            n = int(len(part))
            total += n
            parts.append(
                {
                    "subject": subject,
                    "num_rows": n,
                    "revision": str(version) if version is not None else None,
                    "fingerprint": getattr(part, "_fingerprint", None),
                }
            )
        return {
            "dataset_key": ds,
            "dataset_id": "cais/mmlu",
            "dataset_config": f"subjects={','.join(subjects)}",
            "split": split,
            "num_rows": total,
            "subjects": list(subjects),
            "fingerprint": None,
            "parts": parts,
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
    """Best-effort row_uid for legacy ARC rows that only stored query_id."""
    if query_id.startswith("arc_"):
        native = query_id.removeprefix("arc_")
        hf_split = split or "validation"
        return make_row_uid("arc_challenge", hf_split, native)
    return None
