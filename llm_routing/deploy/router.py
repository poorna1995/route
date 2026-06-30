"""Deployable router — score frozen features and choose pool member."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from llm_routing.deploy.extract import extract_from_run, vectorize_features
from llm_routing.deploy.policy import RoutingPolicy, load_policy


@dataclass(frozen=True)
class RouteDecision:
    query_id: str
    score: float
    route_hi: bool
    model: str
    feature_columns: tuple[str, ...]


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def score_vector(policy: RoutingPolicy, values: list[float]) -> float:
    if len(values) != len(policy.feature_columns):
        raise ValueError("feature dimension mismatch")
    scaled = [
        (v - m) / (s if s != 0.0 else 1.0)
        for v, m, s in zip(values, policy.scaler_mean, policy.scaler_scale)
    ]
    logit = policy.intercept + sum(w * x for w, x in zip(policy.weights, scaled))
    if policy.scoring_method == "logistic_probability":
        return _sigmoid(logit)
    return logit


def route_features(
    policy: RoutingPolicy,
    query_id: str,
    features: dict[str, float],
) -> RouteDecision:
    values = vectorize_features(features, policy.feature_columns)
    score = score_vector(policy, values)
    route_hi = score > policy.threshold
    return RouteDecision(
        query_id=query_id,
        score=score,
        route_hi=route_hi,
        model=policy.pool_hi if route_hi else policy.pool_lo,
        feature_columns=policy.feature_columns,
    )


def route_query(
    policy: RoutingPolicy,
    run_root: Path | str,
    query_id: str,
    *,
    include_query: bool | None = None,
) -> RouteDecision:
    """Route using signals loaded from a research run (replay / eval mode)."""
    if include_query is None:
        include_query = any(col.startswith("phi.") for col in policy.feature_columns)
    features = extract_from_run(
        run_root,
        query_id,
        include_query=include_query,
    )
    return route_features(policy, query_id, features)


def load_router(run_root: Path | str, policy_path: Path | str | None = None) -> RoutingPolicy:
    return load_policy(resolve_policy_path(run_root, policy_path))
