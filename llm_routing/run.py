"""Run directory layout, stage orchestration, and pipeline entry points."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import llm_routing.paths as paths
from llm_routing.paths import ROOT, ROUTER_PACKAGE_DIRNAME, SPLIT_SUFFIX
from llm_routing.setting import load_setting, save_setting

__all__ = [
    "ROOT",
    "RUNS_ROOT",
    "Run",
    "SPLITS",
    "SPLIT_SUFFIX",
    "CANDIDATES_DIR",
    "MODEL_INDEPENDENT_INDEX",
    "MODEL_INDEPENDENT_INDEX_PATH",
    "QUERY_DERIVED_INDEX",
    "SELECTION_REPORT_PATH",
    "build_selection_report",
    "compute_scorecard",
    "evaluate_on_split",
    "iter_run_dirs",
    "list_candidate_settings",
    "load_oracle_rows_by_id",
    "merge_oracle_rows",
    "message",
    "oracle_path",
    "permanent_run_name",
    "pick_winner",
    "run_all",
    "run_development",
    "stable_run_name",
    "stage_cross_model",
    "stage_eval",
    "stage_evaluate",
    "stage_model_dependent",
    "stage_model_independent",
    "stage_model_independent_all",
    "stage_oracle",
    "stage_prepare",
    "stage_scorecard",
    "stage_selection_report",
    "stage_signal_extraction_cross_model",
    "stage_signal_extraction_dependent",
    "stage_signal_extraction_independent",
    "stage_signal_validation",
    "utc_now",
    "write_json",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _is_run_dir(path: Path) -> bool:
    return path.is_dir() and (path / "manifest.json").exists()


def iter_run_dirs(runs_root: Path | None = None) -> list[Path]:
    """Scratch timestamp runs plus grouped permanent runs."""
    runs_root = runs_root or paths.RUNS_ROOT
    if not runs_root.exists():
        return []
    found: list[Path] = []
    stable = runs_root / "permanent"
    if stable.is_dir():
        for group in sorted(stable.iterdir()):
            if group.is_dir():
                found.extend(sorted(p for p in group.iterdir() if _is_run_dir(p)))
    found.extend(
        sorted(
            p
            for p in runs_root.iterdir()
            if p.name not in ("permanent", "stable") and not p.name.startswith(".") and _is_run_dir(p)
        )
    )
    return found


def stable_run_name(dataset: str, stage: str, split: str) -> str:
    """Stable run dir name, e.g. arc_oracle_pilot or mmlu_pro_model_independent_eval."""
    slug = dataset.lower().replace("-", "_").removesuffix("_challenge")
    if stage == "model_independent" and split == "eval":
        return f"{slug}_model_independent_eval"
    suffix = SPLIT_SUFFIX.get(split, split)
    return f"{slug}_{stage}_{suffix}"


permanent_run_name = stable_run_name


@dataclass(frozen=True)
class Run:
    root: Path

    @property
    def run_id(self) -> str:
        return self.root.name

    @property
    def setting_path(self) -> Path:
        return self.root / "setting.yaml"

    @property
    def corpus_dir(self) -> Path:
        return self.root / "corpus"

    @property
    def oracle_dir(self) -> Path:
        return self.root / "oracle"

    @property
    def signals_dir(self) -> Path:
        return self.root / "signals"

    @property
    def routing_dir(self) -> Path:
        return self.root / "routing"

    @property
    def router_package_dir(self) -> Path:
        return self.root / ROUTER_PACKAGE_DIRNAME

    @property
    def scorecard_path(self) -> Path:
        return self.root / "scorecard.json"

    def manifest(self) -> dict[str, Any]:
        return json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))

    def setting(self) -> dict[str, Any]:
        return load_setting(self.setting_path)

    def stage_done(self, name: str, **info: Any) -> None:
        m = self.manifest()
        m.setdefault("stages", {})[name] = {"status": "done", "finished_at": utc_now(), **info}
        write_json(self.root / "manifest.json", m)

    @classmethod
    def create(
        cls,
        setting_src: Path | str,
        *,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> Run:
        setting_src = Path(setting_src)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        rid = f"{stamp}-{name or setting_src.stem}"
        root = paths.RUNS_ROOT / rid
        if root.exists():
            raise FileExistsError(f"run already exists: {root}")
        root.mkdir(parents=True)
        save_setting(root / "setting.yaml", load_setting(setting_src))
        write_json(
            root / "manifest.json",
            {
                "run_id": rid,
                "created_at": utc_now(),
                "setting_source": str(
                    setting_src.relative_to(ROOT) if setting_src.is_relative_to(ROOT) else setting_src
                ),
                "stages": {},
                "config": config or {},
            },
        )
        return cls(root)

    @classmethod
    def open(cls, path: Path | str) -> Run:
        root = Path(path)
        if not (root / "manifest.json").exists():
            raise FileNotFoundError(f"not a run directory: {root}")
        return cls(root)


# Stage re-exports and multi-stage helpers (import after Run to avoid cycles).
from llm_routing.design import (  # noqa: E402
    build_selection_report,
    compute_scorecard,
    message,
    pick_winner,
    stage_eval,
    stage_prepare,
    stage_scorecard,
    stage_selection_report,
)
from llm_routing.develop import (  # noqa: E402
    load_oracle_rows_by_id,
    merge_oracle_rows,
    oracle_path,
    run_development,
    stage_cross_model,
    stage_model_dependent,
    stage_model_independent,
    stage_oracle,
    stage_signal_extraction_cross_model,
    stage_signal_extraction_dependent,
    stage_signal_extraction_independent,
)
from llm_routing.signal_validation import stage_signal_validation  # noqa: E402
from llm_routing.evaluate import evaluate_on_split, stage_evaluate  # noqa: E402
from llm_routing.paths import (  # noqa: E402
    CANDIDATES_DIR,
    MODEL_INDEPENDENT_INDEX_PATH,
    RUNS_ROOT,
    SELECTION_REPORT_PATH,
    SPLITS,
)

MODEL_INDEPENDENT_INDEX = MODEL_INDEPENDENT_INDEX_PATH
QUERY_DERIVED_INDEX = MODEL_INDEPENDENT_INDEX


def list_candidate_settings() -> list[Path]:
    return sorted(CANDIDATES_DIR.glob("*.yaml"))


def stage_model_independent_all(
    *,
    limit: int | None = None,
    force_partition: bool = False,
    smoke: bool = False,
) -> Path:
    if smoke and limit is None:
        limit = 5
    index_runs: list[dict[str, Any]] = []
    for setting_path in list_candidate_settings():
        setting = load_setting(setting_path)
        dataset = setting["corpus"]["dataset"]
        run = Run.create(setting_path, name=f"phi-{setting_path.stem}")
        print(f"\n=== {dataset} ({run.run_id}) ===")
        stage_prepare(run, force_partition=force_partition)
        stage_eval(run)
        out = stage_model_independent(run, limit=limit, allow_full_corpus=False)
        meta_path = run.signals_dir / "model_independent_meta.json"
        if not meta_path.exists():
            meta_path = run.signals_dir / "query_derived_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        index_runs.append(
            {
                "dataset": dataset,
                "run_id": run.run_id,
                "run_root": str(run.root.relative_to(ROOT)),
                "output": str(out.relative_to(ROOT)),
                "n_queries": meta["n_queries"],
                "n_calib": meta.get("n_calib"),
                "n_test": meta.get("n_test"),
                "holdout_excluded": meta.get("holdout_excluded", True),
            }
        )
    report = {"timestamp": utc_now(), "limit": limit, "runs": index_runs}
    write_json(MODEL_INDEPENDENT_INDEX, report)
    print(f"\n[model-independent-all] {len(index_runs)} datasets → {MODEL_INDEPENDENT_INDEX}")
    return MODEL_INDEPENDENT_INDEX


def run_all(
    setting_src: Path,
    *,
    name: str = "pilot",
    smoke: bool = False,
    limit: int | None = None,
    split: str = "selection_holdout",
    force_partition: bool = False,
) -> Run:
    cap = limit if limit is not None else (20 if smoke else None)
    run = Run.create(
        setting_src,
        name=name,
        config={"split": split, "limit": cap, "smoke": smoke},
    )
    print(f"Run → {run.root}")
    stage_prepare(run, force_partition=force_partition)
    stage_oracle(run, split=split, limit=cap)
    stage_scorecard(run)
    return run
