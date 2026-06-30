"""Read and write experiment settings from YAML files.

A "setting" is one experiment config: which dataset, which models, how to prompt,
how to split data, and pass/fail gates for benchmark selection.

This file does NOT run experiments. It only loads, merges, saves, and reads
pieces of the setting dict that other modules need.

Depends on:
  - yaml                          (parse/write YAML)
  - corpus.get_dataset, etc.    (dataset names and split sizes)

Example — candidate vs run setting
-----------------------------------

Candidate file (experiments/candidates/arc.yaml) — dataset only:

    setting:
      corpus:
        dataset: ARC-Challenge

At load time, load_setting() merges in experiments/defaults.yaml (pool, protocol,
gates, partition) because the candidate has no "pool" key yet.

After prepare/eval, the run snapshot (experiments/runs/.../setting.yaml) is
fully expanded and frozen — it stores partition ids so splits never change.

    setting = load_setting("experiments/candidates/arc.yaml")
    protocol = get_protocol(setting)   # prompts + decoding for oracle
    gates = get_gates(setting)         # {min_accuracy_gap, opportunity_min, too_hard_max}
    spec = corpus_spec_from_setting(setting)  # DatasetSpec for loading ARC
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from llm_routing.corpus import DatasetSpec, get_dataset, resolve_holdout_size, resolve_test_size

from llm_routing.paths import SETTING_DEFAULTS_PATH as DEFAULTS_PATH


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Combine two dicts. Values in override win; nested dicts are merged recursively."""
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _load_defaults() -> dict[str, Any]:
    """Read experiments/defaults.yaml (pool, protocol, gates, partition policy)."""
    data = yaml.safe_load(DEFAULTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{DEFAULTS_PATH}: expected mapping")
    return data


def corpus_preparation_cfg(setting: dict[str, Any]) -> dict[str, Any]:
    """Merge defaults + run overrides for M1 corpus preparation."""
    defaults = _load_defaults().get("corpus_preparation") or {}
    override = setting.get("corpus_preparation") or {}
    if isinstance(defaults, dict) and isinstance(override, dict):
        return _deep_merge(defaults, override)
    return dict(defaults) if defaults else {}


def load_defaults() -> dict[str, Any]:
    """Public wrapper for _load_defaults()."""
    return _load_defaults()


def load_setting(path: Path | str) -> dict[str, Any]:
    """Load a setting YAML file and return the inner "setting" dict.

    Input:  path to a YAML file with top-level key "setting"
    Output: dict with keys like corpus, pool, protocol, partition, gates, …

    If the file is a thin candidate (no "pool" key), merges experiments/defaults.yaml
    on top so pool/protocol/gates are filled in automatically.

    Used by: run, signals.phi.run, tests
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "setting" not in data:
        raise ValueError(f"{path}: expected top-level 'setting'")
    setting = data["setting"]
    if DEFAULTS_PATH.exists() and "pool" not in setting:
        setting = _deep_merge(_load_defaults(), setting)
    return setting


def save_setting(path: Path | str, setting: dict[str, Any]) -> None:
    """Write a setting dict back to YAML (wraps it under top-level "setting").

    Used when freezing partition ids after prepare or M3 eval.
    """
    Path(path).write_text(
        yaml.safe_dump({"setting": setting}, sort_keys=False),
        encoding="utf-8",
    )


def corpus_spec_from_setting(setting: dict[str, Any]) -> DatasetSpec:
    """Turn setting["corpus"] into a DatasetSpec the corpus loader understands.

    Input:  setting dict (must have setting["corpus"]["dataset"])
    Output: DatasetSpec(name, hf repo, splits, …)

    Example: dataset "ARC-Challenge" → allenai/ai2_arc on HuggingFace
    """
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
    """Extract prompt and decoding config for the oracle.

    Input:  setting with a "protocol" block
    Output: flat dict used by oracle.py and query_derived

        {
          "protocol_version": "mcq_letter",
          "system_prompt": "You answer multiple-choice questions...",
          "user_template": "{question}\\n\\n{choices}\\n\\nAnswer:",
          "decoding": {"temperature": 0.0, "max_tokens": 16},
          "grading": {"method": "mcq"},
        }
    """
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
    """Extract benchmark selection thresholds (Gates C and D).

    Output:
      min_accuracy_gap  — M_hi must beat M_lo by at least this much
      opportunity_min   — enough questions where only M_hi is correct
      too_hard_max      — not too many questions where both models fail
    """
    block = setting["gates"]
    return {
        "min_accuracy_gap": float(block["min_accuracy_gap"]),
        "opportunity_min": float(block["opportunity_min"]),
        "too_hard_max": float(block["too_hard_max"]),
    }


def get_tie_break(setting: dict[str, Any]) -> dict[str, Any]:
    """How to pick the winning benchmark when multiple pass gates (M2 selection)."""
    return dict(setting.get("selection", {}).get("tie_break") or {})


def holdout_size(setting: dict[str, Any], corpus_size: int) -> int:
    """How many questions go into the selection holdout H at M1.

    Input:  setting + total corpus size |C|
    Output: integer count (e.g. 150 from selection_holdout_n in defaults)
    """
    return resolve_holdout_size(corpus_size, setting["partition"])


def test_size(setting: dict[str, Any], *, eval_pool_n: int) -> int:
    """How many questions go into test split R_t at M3 lock-eval.

    Input:  setting + size of C\\H (everything not in holdout)
    Output: integer count from test_fraction / test_min / test_max policy
    """
    return resolve_test_size(setting["partition"], eval_pool_n=eval_pool_n)


def partition_cfg_for_m3_lock(setting: dict[str, Any]) -> dict[str, Any]:
    """Partition config for M3, with legacy fallbacks from defaults.yaml.

    Old run snapshots may only have a fixed test_n; this fills in test_fraction
    from defaults when needed.
    """
    part = dict(setting.get("partition") or {})
    defaults = _load_defaults().get("partition") or {}
    if "test_fraction" in defaults:
        for key in ("test_fraction", "test_min", "test_max"):
            if key in defaults:
                part[key] = defaults[key]
        part.pop("test_n", None)
    return part


def eval_partition_complete(setting: dict[str, Any]) -> bool:
    """True if M3 already ran — calib and test query id lists are frozen in setting."""
    ids = setting.get("partition", {}).get("ids") or {}
    return bool(ids.get("calib") and ids.get("test"))


def frozen_holdout_ids(setting: dict[str, Any]) -> list[str] | None:
    """Return frozen holdout H query ids if M1 prepare already ran, else None."""
    ids = setting.get("partition", {}).get("ids")
    if not ids or not ids.get("selection_holdout"):
        return None
    return ids["selection_holdout"]


def freeze_partition_ids(setting: dict[str, Any], partition: dict[str, list[str]]) -> dict[str, Any]:
    """Attach partition id lists to setting and mark holdout as frozen (M1).

    Input:  setting + {"selection_holdout": [...], ...}
    Output: new setting dict (does not write to disk — call save_setting after)
    """
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
    """Attach calib + test id lists and mark eval partition frozen (M3 lock-eval).

    Input:  setting + full partition (holdout, calib, test) + resolved |R_t|
    Output: new setting dict with meta.eval_ids_frozen = True
    """
    updated = freeze_partition_ids(setting, partition)
    part = dict(updated["partition"])
    part["test_n"] = resolved_test_n
    updated["partition"] = part
    meta = dict(updated.get("meta") or {})
    meta["eval_ids_frozen"] = True
    updated["meta"] = meta
    return updated
