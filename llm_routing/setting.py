"""Load/save setting YAML; merge experiments/phase_a_defaults.yaml for candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from llm_routing.corpus import DatasetSpec, get_dataset, resolve_holdout_size, resolve_test_size

ROOT = Path(__file__).resolve().parents[1]
PHASE_A_DEFAULTS_PATH = ROOT / "experiments/phase_a_defaults.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _load_phase_a_defaults() -> dict[str, Any]:
    data = yaml.safe_load(PHASE_A_DEFAULTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{PHASE_A_DEFAULTS_PATH}: expected mapping")
    return data


def load_phase_a_defaults() -> dict[str, Any]:
    return _load_phase_a_defaults()


def load_setting(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "setting" not in data:
        raise ValueError(f"{path}: expected top-level 'setting'")
    setting = data["setting"]
    # Candidates are dataset-only; runs store full snapshots.
    if PHASE_A_DEFAULTS_PATH.exists() and "pool" not in setting:
        setting = _deep_merge(_load_phase_a_defaults(), setting)
    return setting


def save_setting(path: Path | str, setting: dict[str, Any]) -> None:
    Path(path).write_text(
        yaml.safe_dump({"setting": setting}, sort_keys=False),
        encoding="utf-8",
    )


def corpus_spec_from_setting(setting: dict[str, Any]) -> DatasetSpec:
    corpus = setting["corpus"]
    base = get_dataset(corpus["dataset"])
    if corpus.get("source", "registry_default") != "registry_default":
        raise NotImplementedError("only registry_default corpus source supported")
    overrides = corpus.get("overrides") or {}
    return DatasetSpec(
        base.name,
        overrides.get("repo", base.repo),
        overrides.get("config", base.config),
        base.corpus_splits,
        base.exclude_splits,
    )


def get_protocol(setting: dict[str, Any]) -> dict[str, Any]:
    block = setting["protocol"]
    template = block.get("user_template") or block.get("dataset_template", "")
    return {
        "protocol_version": block.get("protocol_version", "inline"),
        "system_prompt": block["system_prompt"].rstrip("\n"),
        "user_template": template.rstrip("\n"),
        "decoding": block.get("decoding", {}),
        "grading": block.get("grading", {}),
    }


def get_gates(setting: dict[str, Any]) -> dict[str, float]:
    block = setting["gates"]
    return {
        "min_accuracy_gap": float(block["min_accuracy_gap"]),
        "opportunity_min": float(block["opportunity_min"]),
        "too_hard_max": float(block["too_hard_max"]),
    }


def get_tie_break(setting: dict[str, Any]) -> dict[str, Any]:
    return dict(setting.get("selection", {}).get("tie_break") or {})


def holdout_size(setting: dict[str, Any], corpus_size: int) -> int:
    return resolve_holdout_size(corpus_size, setting["partition"])


def test_size(setting: dict[str, Any], *, eval_pool_n: int) -> int:
    return resolve_test_size(setting["partition"], eval_pool_n=eval_pool_n)


def eval_partition_complete(setting: dict[str, Any]) -> bool:
    ids = setting.get("partition", {}).get("ids") or {}
    return bool(ids.get("calib") and ids.get("test"))


def frozen_holdout_ids(setting: dict[str, Any]) -> list[str] | None:
    ids = setting.get("partition", {}).get("ids")
    if not ids or not ids.get("selection_holdout"):
        return None
    return ids["selection_holdout"]


def freeze_partition_ids(setting: dict[str, Any], partition: dict[str, list[str]]) -> dict[str, Any]:
    updated = dict(setting)
    part = dict(updated["partition"])
    part["ids"] = partition
    updated["partition"] = part
    meta = dict(updated.get("meta") or {})
    meta["holdout_frozen"] = True
    updated["meta"] = meta
    return updated


def freeze_eval_partition(
    setting: dict[str, Any],
    partition: dict[str, list[str]],
    *,
    resolved_test_n: int,
) -> dict[str, Any]:
    updated = freeze_partition_ids(setting, partition)
    part = dict(updated["partition"])
    part["test_n"] = resolved_test_n
    updated["partition"] = part
    meta = dict(updated.get("meta") or {})
    meta["eval_ids_frozen"] = True
    updated["meta"] = meta
    return updated
