"""Frozen routing policy — loaded at deployment from policy.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_routing.signal_schema import (
    SIGNAL_LAYER_MODEL_DEPENDENT,
    normalize_signal_layers,
)

POLICY_SCHEMA_VERSION = "v1"


@dataclass
class RoutingPolicy:
    """Deployable router config: φ/ψ features only (no χ at runtime)."""

    schema_version: str
    feature_columns: tuple[str, ...]
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    weights: tuple[float, ...]
    intercept: float
    threshold: float
    signal_layers: tuple[str, ...] = (SIGNAL_LAYER_MODEL_DEPENDENT,)
    scoring_method: str = "logistic_probability"
    pool_lo: str = "M_lo"
    pool_hi: str = "M_hi"
    training: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "runtime_scope": "deploy",
            "signal_layers": list(self.signal_layers),
            "feature_columns": list(self.feature_columns),
            "scaler": {
                "mean": list(self.scaler_mean),
                "scale": list(self.scaler_scale),
            },
            "model": {
                "scoring_method": self.scoring_method,
                "weights": list(self.weights),
                "intercept": self.intercept,
                "threshold": self.threshold,
            },
            "pool": {"lo": self.pool_lo, "hi": self.pool_hi},
            "route_hi_when": "score > threshold",
            "training": self.training,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RoutingPolicy:
        scaler = data.get("scaler") or {}
        model = data.get("model") or {}
        pool = data.get("pool") or {}
        cols = tuple(data["feature_columns"])
        mean = tuple(float(x) for x in scaler.get("mean", [0.0] * len(cols)))
        scale = tuple(float(x) for x in scaler.get("scale", [1.0] * len(cols)))
        weights = tuple(float(x) for x in model.get("weights", [0.0] * len(cols)))
        return cls(
            schema_version=str(data.get("schema_version", POLICY_SCHEMA_VERSION)),
            feature_columns=cols,
            scaler_mean=mean,
            scaler_scale=scale,
            weights=weights,
            intercept=float(model.get("intercept", 0.0)),
            threshold=float(model.get("threshold", 0.5)),
            signal_layers=normalize_signal_layers(
                tuple(data.get("signal_layers") or (SIGNAL_LAYER_MODEL_DEPENDENT,))
            ),
            scoring_method=str(model.get("scoring_method", "logistic_probability")),
            pool_lo=str(pool.get("lo", "M_lo")),
            pool_hi=str(pool.get("hi", "M_hi")),
            training=dict(data.get("training") or {}),
        )


def load_policy(path: Path | str) -> RoutingPolicy:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return RoutingPolicy.from_dict(data)


def resolve_policy_path(
    run_root: Path | str,
    policy_path: Path | str | None = None,
) -> Path:
    """Prefer explicit path, then router_package/, then routing/policy.json."""
    if policy_path is not None:
        return Path(policy_path)
    run_root = Path(run_root)
    from llm_routing.paths import ROUTER_PACKAGE_DIRNAME

    pkg = run_root / ROUTER_PACKAGE_DIRNAME / "policy.json"
    if pkg.exists():
        return pkg
    return run_root / "routing" / "policy.json"


def save_policy(path: Path, policy: RoutingPolicy) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path
