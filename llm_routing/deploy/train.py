"""Fit routing policy on calib analysis_table (optional; rewrite alongside Stage 6–8)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from llm_routing.deploy.extract import default_runtime_columns
from llm_routing.paths import FEATURE_SPEC_FILENAME
from llm_routing.signal_schema import (
    SIGNAL_LAYER_MODEL_DEPENDENT,
    SIGNAL_LAYER_MODEL_INDEPENDENT,
)


def _feature_spec_path(run_root: Path | str) -> Path:
    return Path(run_root) / "signals" / "analysis" / FEATURE_SPEC_FILENAME


def _load_feature_spec(run_root: Path | str) -> dict[str, Any] | None:
    path = _feature_spec_path(run_root)
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _parse_float(val: str | float | None) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _load_xy(
    rows: list[dict[str, str]], columns: tuple[str, ...]
) -> tuple[list[list[float]], list[int]]:
    xs: list[list[float]] = []
    ys: list[int] = []
    for row in rows:
        vec: list[float] = []
        for col in columns:
            x = _parse_float(row.get(col))
            if x is None:
                vec = []
                break
            vec.append(x)
        if not vec:
            continue
        xs.append(vec)
        ys.append(int(row["r"]))
    return xs, ys


def _tune_threshold(probs: list[float], labels: list[int]) -> tuple[float, float]:
    best_tau = 0.5
    best_acc = -1.0
    for i in range(1, 100):
        tau = i / 100.0
        correct = sum(int((p > tau) == bool(y)) for p, y in zip(probs, labels))
        acc = correct / len(labels)
        if acc > best_acc:
            best_acc = acc
            best_tau = tau
    return best_tau, best_acc


def train_routing_policy(
    run_root: Path | str,
    *,
    feature_columns: tuple[str, ...] | None = None,
    include_query: bool = False,
    seed: int = 42,
    policy_name: str = "policy.json",
) -> Path:
    """Fit logistic weights on calib analysis_table; write routing/policy.json."""
    run_root = Path(run_root)
    table_path = run_root / "signals" / "analysis" / "analysis_table_calib.csv"
    if not table_path.exists():
        table_path = run_root / "signals" / "analysis" / "analysis_table.csv"
    if not table_path.exists():
        raise FileNotFoundError(
            f"{table_path} missing — run: python run.py signal-validation --run {run_root}"
        )

    spec = _load_feature_spec(run_root)
    if feature_columns is not None:
        columns = feature_columns
    elif spec and spec.get("runtime_columns"):
        columns = tuple(spec["runtime_columns"])
    else:
        columns = default_runtime_columns(include_query=include_query)
    if any(col.startswith("chi.") for col in columns):
        raise ValueError("runtime policy cannot include cross-model (chi.*) features")

    from llm_routing.deploy.policy import POLICY_SCHEMA_VERSION, RoutingPolicy, save_policy

    with table_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    xs, ys = _load_xy(rows, columns)
    if len(xs) < 10 or len(set(ys)) < 2:
        raise ValueError("insufficient calib rows for policy training")

    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        raise ImportError('policy training requires scikit-learn') from e

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(np.asarray(xs, dtype=float))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(x_scaled, np.asarray(ys, dtype=int))
    probs = clf.predict_proba(x_scaled)[:, 1].tolist()
    tau, calib_acc = _tune_threshold(probs, ys)

    layers: list[str] = []
    if any(c.startswith("phi.") for c in columns):
        layers.append(SIGNAL_LAYER_MODEL_INDEPENDENT)
    if any(c.startswith("psi.") for c in columns):
        layers.append(SIGNAL_LAYER_MODEL_DEPENDENT)

    policy = RoutingPolicy(
        schema_version=POLICY_SCHEMA_VERSION,
        feature_columns=columns,
        scaler_mean=tuple(float(x) for x in scaler.mean_),
        scaler_scale=tuple(float(x) for x in scaler.scale_),
        weights=tuple(float(x) for x in clf.coef_[0]),
        intercept=float(clf.intercept_[0]),
        threshold=tau,
        signal_layers=tuple(layers) or (SIGNAL_LAYER_MODEL_DEPENDENT,),
        training={
            "split": "calib",
            "method": "logistic_regression",
            "seed": seed,
            "n_calib": len(xs),
            "positive_rate": sum(ys) / len(ys),
            "calib_accuracy_at_tau": calib_acc,
            "excludes_cross_model": True,
            "source_table": str(table_path.relative_to(run_root)),
        },
    )

    out_dir = run_root / "routing"
    out_path = save_policy(out_dir / policy_name, policy)
    (out_dir / "policy_meta.json").write_text(
        json.dumps(
            {
                "policy_path": str(out_path.relative_to(run_root)),
                "n_features": len(columns),
                "feature_columns": list(columns),
                **policy.training,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return out_path


def evaluate_policy_on_table(
    policy_path: Path | str,
    table_path: Path | str,
) -> dict[str, Any]:
    from llm_routing.deploy.policy import load_policy
    from llm_routing.deploy.router import route_features

    policy = load_policy(policy_path)
    with Path(table_path).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    correct = 0
    n = 0
    for row in rows:
        try:
            feats = {col: float(row[col]) for col in policy.feature_columns}
        except (KeyError, TypeError, ValueError):
            continue
        decision = route_features(policy, row["query_id"], feats)
        y = int(row["r"])
        if int(decision.route_hi) == y:
            correct += 1
        n += 1
    return {
        "n": n,
        "accuracy": correct / n if n else float("nan"),
        "threshold": policy.threshold,
    }
