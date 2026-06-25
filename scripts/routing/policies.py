"""Study IV — hold-out routing evaluation and exploratory preview (RH4)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from routing.constants import (
    ANALYSIS_ROUTING_HOLDOUT,
    ANALYSIS_ROUTING_PREVIEW,
    COL_COMPLEXITY,
    COL_ENTROPY_WEAK,
    COL_MARGIN_WEAK,
    COL_OPPORTUNITY,
    ROUTING_ROUTER_FEATURES,
    STRONG_COST,
    WEAK_COST,
)
from routing.evaluation import build_feature_matrix, fit_logistic_probs_holdout


def oracle_choice(weak_ok: bool, strong_ok: bool) -> str:
    if weak_ok and strong_ok:
        return "weak"
    if strong_ok:
        return "strong"
    if weak_ok:
        return "weak"
    return "weak"


def routed_correct(choice: str, weak_ok: bool, strong_ok: bool) -> bool:
    return strong_ok if choice == "strong" else weak_ok


def routed_cost(choice: str, *, weak_cost: float, strong_cost: float) -> float:
    return strong_cost if choice == "strong" else weak_cost


def evaluate_policy(
    df: pd.DataFrame,
    name: str,
    chooser: Callable[[pd.Series], str],
    *,
    weak_cost: float,
    strong_cost: float,
) -> dict[str, Any]:
    choices = df.apply(chooser, axis=1)
    correct = [routed_correct(c, w, s) for c, w, s in zip(choices, df["weak_ok"], df["strong_ok"])]
    costs = [routed_cost(c, weak_cost=weak_cost, strong_cost=strong_cost) for c in choices]

    opp_mask = df["bucket"] == "opportunity"
    weak_only_mask = df["bucket"] == "weak_only"

    opp_routed_strong = int((choices[opp_mask] == "strong").sum()) if opp_mask.any() else 0
    opp_n = int(opp_mask.sum())
    weak_only_routed_weak = int((choices[weak_only_mask] == "weak").sum()) if weak_only_mask.any() else 0
    weak_only_n = int(weak_only_mask.sum())

    oracle_choices = [oracle_choice(w, s) for w, s in zip(df["weak_ok"], df["strong_ok"])]
    oracle_acc = float(np.mean([
        routed_correct(c, w, s) for c, w, s in zip(oracle_choices, df["weak_ok"], df["strong_ok"])
    ]))

    by_bucket: dict[str, Any] = {}
    for bucket, grp in df.groupby("bucket"):
        idx = grp.index
        ch = choices.loc[idx]
        by_bucket[bucket] = {
            "n": int(len(grp)),
            "routed_strong": int((ch == "strong").sum()),
            "routed_weak": int((ch == "weak").sum()),
            "accuracy": float(np.mean([
                routed_correct(c, w, s) for c, w, s in zip(ch, grp["weak_ok"], grp["strong_ok"])
            ])),
        }

    return {
        "policy": name,
        "n": int(len(df)),
        "accuracy": float(np.mean(correct)),
        "avg_cost": float(np.mean(costs)),
        "pct_routed_strong": float((choices == "strong").mean()),
        "opportunity_recall_strong": opp_routed_strong / opp_n if opp_n else None,
        "opportunity_n": opp_n,
        "weak_only_correct_routing": weak_only_routed_weak / weak_only_n if weak_only_n else None,
        "weak_only_n": weak_only_n,
        "delta_accuracy_vs_always_weak": float(np.mean(correct)) - float(df["weak_ok"].mean()),
        "delta_accuracy_vs_always_strong": float(np.mean(correct)) - float(df["strong_ok"].mean()),
        "oracle_accuracy_upper_bound": oracle_acc,
        "by_bucket": by_bucket,
    }


def build_routing_policies(df: pd.DataFrame) -> list[tuple[str, Callable[[pd.Series], str]]]:
    med_ew = float(df["entropy_w"].median())
    med_mw = float(df["margin_w"].median())
    med_es = float(df["entropy_s"].median())
    med_de = float(df["delta_entropy"].median())

    def always_weak(_: pd.Series) -> str:
        return "weak"

    def always_strong(_: pd.Series) -> str:
        return "strong"

    def oracle_policy(row: pd.Series) -> str:
        return oracle_choice(bool(row["weak_ok"]), bool(row["strong_ok"]))

    def random_policy(row: pd.Series) -> str:
        return "strong" if hash(row["query_id"]) % 2 == 0 else "weak"

    def high_entropy_w(row: pd.Series) -> str:
        return "strong" if row["entropy_w"] > med_ew else "weak"

    def low_margin_w(row: pd.Series) -> str:
        return "strong" if row["margin_w"] < med_mw else "weak"

    def high_entropy_s(row: pd.Series) -> str:
        return "strong" if row["entropy_s"] > med_es else "weak"

    def positive_delta_entropy(row: pd.Series) -> str:
        return "strong" if row["delta_entropy"] > 0 else "weak"

    def high_delta_entropy(row: pd.Series) -> str:
        return "strong" if row["delta_entropy"] > med_de else "weak"

    def combined_hw(row: pd.Series) -> str:
        votes = int(row["entropy_w"] > med_ew) + int(row["margin_w"] < med_mw)
        return "strong" if votes >= 1 else "weak"

    return [
        ("always_weak", always_weak),
        ("always_strong", always_strong),
        ("oracle", oracle_policy),
        ("random", random_policy),
        ("high_entropy_w", high_entropy_w),
        ("low_margin_w", low_margin_w),
        ("high_entropy_s", high_entropy_s),
        ("positive_delta_entropy", positive_delta_entropy),
        ("high_delta_entropy", high_delta_entropy),
        ("combined_entropy_or_margin_w", combined_hw),
    ]


def run_routing_preview(
    df: pd.DataFrame,
    *,
    weak_cost: float = WEAK_COST,
    strong_cost: float = STRONG_COST,
) -> dict[str, Any]:
    policies = build_routing_policies(df)
    results = [
        evaluate_policy(df, name, fn, weak_cost=weak_cost, strong_cost=strong_cost)
        for name, fn in policies
    ]
    return {
        "analysis": ANALYSIS_ROUTING_PREVIEW,
        "note": (
            "Exploratory routing simulation on the same slice used to set median thresholds. "
            "Paper routing (EXP-03) uses route-eval: CALIB-fit logistic + threshold, TEST evaluation."
        ),
        "n": int(len(df)),
        "cost_model": {"weak": weak_cost, "strong": strong_cost},
        "bucket_counts": df["bucket"].value_counts().to_dict(),
        "policies": results,
        "reading_guide": {
            "accuracy": "Fraction of queries where the routed model answered correctly.",
            "opportunity_recall_strong": "On opportunity queries, fraction routed to strong (should be 1.0 for ideal router).",
            "weak_only_correct_routing": "On weak-only queries, fraction routed to weak (should be 1.0).",
            "oracle_accuracy_upper_bound": "Best accuracy achievable with perfect routing + cheap tie-break.",
        },
    }


def _sklearn_logistic():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return LogisticRegression, Pipeline, StandardScaler


def _route_from_probs(probs: np.ndarray, tau: float) -> np.ndarray:
    return np.where(probs >= tau, "strong", "weak")


def _policy_metrics(
    df: pd.DataFrame,
    choices: np.ndarray,
    *,
    weak_cost: float,
    strong_cost: float,
) -> dict[str, Any]:
    correct = np.array([
        routed_correct(c, w, s) for c, w, s in zip(choices, df["weak_ok"], df["strong_ok"])
    ])
    costs = np.array([routed_cost(c, weak_cost=weak_cost, strong_cost=strong_cost) for c in choices])
    opp_mask = df["bucket"] == "opportunity"
    weak_only_mask = df["bucket"] == "weak_only"
    opp_routed_strong = int((choices[opp_mask] == "strong").sum()) if opp_mask.any() else 0
    opp_n = int(opp_mask.sum())
    weak_only_routed_weak = int((choices[weak_only_mask] == "weak").sum()) if weak_only_mask.any() else 0
    weak_only_n = int(weak_only_mask.sum())
    return {
        "accuracy": float(correct.mean()),
        "avg_cost": float(costs.mean()),
        "pct_routed_strong": float((choices == "strong").mean()),
        "opportunity_recall_strong": opp_routed_strong / opp_n if opp_n else None,
        "opportunity_n": opp_n,
        "weak_only_correct_routing": weak_only_routed_weak / weak_only_n if weak_only_n else None,
        "weak_only_n": weak_only_n,
    }


def _utility(accuracy: float, avg_cost: float, *, cost_lambda: float) -> float:
    if cost_lambda <= 0:
        return accuracy
    return accuracy - cost_lambda * avg_cost


def tune_routing_threshold(
    probs: np.ndarray,
    df: pd.DataFrame,
    *,
    weak_cost: float,
    strong_cost: float,
    cost_lambda: float = 0.0,
) -> dict[str, Any]:
    """Pick τ on CALIB only. Default: maximize accuracy; optional cost penalty via cost_lambda."""
    candidates = sorted({0.0, 1.0, *map(float, probs)})
    best: dict[str, Any] | None = None
    for tau in candidates:
        choices = _route_from_probs(probs, tau)
        m = _policy_metrics(df, choices, weak_cost=weak_cost, strong_cost=strong_cost)
        score = _utility(m["accuracy"], m["avg_cost"], cost_lambda=cost_lambda)
        row = {"tau": tau, "utility": score, **m}
        if best is None or row["utility"] > best["utility"]:
            best = row
    assert best is not None
    return best


def run_routing_holdout(
    df: pd.DataFrame,
    *,
    calib_ids: set[str],
    test_ids: set[str],
    weak_cost: float = WEAK_COST,
    strong_cost: float = STRONG_COST,
    cost_lambda: float = 0.0,
    feature_cols: tuple[str, ...] = ROUTING_ROUTER_FEATURES,
) -> dict[str, Any]:
    """Study IV (EXP-03): independent CALIB-fit router; TEST-only reported metrics.

    Fits a fresh logistic P(y_opp | c, H_w, m_w) on CALIB — not the Study III model artifact.
    Tunes threshold τ on CALIB, freezes (coefficients + τ), evaluates policies on TEST.
    """
    LogisticRegression, Pipeline, StandardScaler = _sklearn_logistic()

    ids = df["query_id"].astype(str)
    calib_mask = ids.isin(calib_ids).to_numpy()
    test_mask = ids.isin(test_ids).to_numpy()
    if not calib_mask.any() or not test_mask.any():
        raise ValueError("CALIB and TEST must both be non-empty in merged CSV")

    missing_calib = calib_ids - set(ids[calib_mask])
    missing_test = test_ids - set(ids[test_mask])
    if missing_calib or missing_test:
        raise ValueError(
            f"merged CSV missing split ids: calib={len(missing_calib)} test={len(missing_test)}"
        )

    cols = list(feature_cols)
    for col in cols + [COL_OPPORTUNITY, "weak_ok", "strong_ok", "bucket"]:
        if col not in df.columns:
            raise ValueError(f"merged CSV missing column: {col}")

    complete = df[np.isfinite(df[cols + [COL_OPPORTUNITY]].to_numpy(dtype=float)).all(axis=1)].copy()
    ids_c = complete["query_id"].astype(str)
    calib_df = complete.loc[ids_c.isin(calib_ids)].copy()
    test_df = complete.loc[ids_c.isin(test_ids)].copy()

    X_cal = build_feature_matrix(calib_df, cols)
    y_cal = calib_df[COL_OPPORTUNITY].to_numpy(dtype=float)
    X_te = build_feature_matrix(test_df, cols)

    probs_cal = fit_logistic_probs_holdout(
        X_cal, y_cal, X_cal,
        LogisticRegression=LogisticRegression, Pipeline=Pipeline, StandardScaler=StandardScaler,
    )
    probs_te = fit_logistic_probs_holdout(
        X_cal, y_cal, X_te,
        LogisticRegression=LogisticRegression, Pipeline=Pipeline, StandardScaler=StandardScaler,
    )
    if probs_cal is None or probs_te is None:
        raise ValueError("logistic router could not be fit on CALIB (check class balance and n)")

    coefs = None
    if len(np.unique(y_cal)) >= 2 and X_cal.shape[0] >= max(5, X_cal.shape[1] + 2):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ])
        clf.fit(X_cal, y_cal)
        coefs = {name: float(c) for name, c in zip(cols, clf.named_steps["lr"].coef_[0])}
        intercept = float(clf.named_steps["lr"].intercept_[0])
    else:
        intercept = None

    tuning = tune_routing_threshold(
        probs_cal, calib_df, weak_cost=weak_cost, strong_cost=strong_cost, cost_lambda=cost_lambda,
    )
    tau = float(tuning["tau"])
    learned_choices = _route_from_probs(probs_te, tau)

    def always_weak(_: pd.Series) -> str:
        return "weak"

    def always_strong(_: pd.Series) -> str:
        return "strong"

    def oracle_policy(row: pd.Series) -> str:
        return oracle_choice(bool(row["weak_ok"]), bool(row["strong_ok"]))

    baselines = [
        ("always_weak", always_weak),
        ("always_strong", always_strong),
        ("oracle", oracle_policy),
    ]
    baseline_results = [
        evaluate_policy(test_df, name, fn, weak_cost=weak_cost, strong_cost=strong_cost)
        for name, fn in baselines
    ]

    learned_metrics = _policy_metrics(
        test_df, learned_choices, weak_cost=weak_cost, strong_cost=strong_cost,
    )
    learned_metrics["policy"] = "learned_router"
    learned_metrics["n"] = int(len(test_df))
    aw = next(r for r in baseline_results if r["policy"] == "always_weak")
    learned_metrics["delta_accuracy_vs_always_weak"] = learned_metrics["accuracy"] - aw["accuracy"]
    learned_metrics["delta_accuracy_vs_always_strong"] = (
        learned_metrics["accuracy"]
        - next(r for r in baseline_results if r["policy"] == "always_strong")["accuracy"]
    )
    learned_metrics["oracle_accuracy_upper_bound"] = next(
        r for r in baseline_results if r["policy"] == "oracle"
    )["accuracy"]

    return {
        "analysis": ANALYSIS_ROUTING_HOLDOUT,
        "study": "IV",
        "hypothesis": "RH4",
        "note": (
            "Independent CALIB-fit logistic router for routing utility (EXP-03). "
            "Does not reuse Study III complementarity model weights or thresholds."
        ),
        "nested_evaluation": {
            "protocol": "calib_fit_tune_test_eval",
            "fit_split": "CALIB",
            "threshold_split": "CALIB",
            "reported_split": "TEST",
            "n_calib": int(len(calib_df)),
            "n_test": int(len(test_df)),
        },
        "router": {
            "model": "logistic_regression",
            "features": cols,
            "target": COL_OPPORTUNITY,
            "coefficients": coefs,
            "intercept": intercept,
            "routing_rule": "route_strong if P(y_opp) >= tau else route_weak",
            "tau": tau,
            "tau_selection": {
                "criterion": "maximize accuracy" if cost_lambda <= 0 else "maximize accuracy - cost_lambda * avg_cost",
                "cost_lambda": cost_lambda,
                "calib_metrics_at_tau": {k: v for k, v in tuning.items() if k != "tau"},
            },
        },
        "cost_model": {"weak": weak_cost, "strong": strong_cost},
        "bucket_counts_test": test_df["bucket"].value_counts().to_dict(),
        "policies_test": baseline_results + [learned_metrics],
        "reading_guide": {
            "accuracy": "Fraction of TEST queries where the routed model answered correctly.",
            "avg_cost": "Mean routing cost on TEST under the fixed cost model.",
            "opportunity_recall_strong": "On TEST opportunity queries, fraction routed to strong.",
            "learned_router": "CALIB-fit logistic + CALIB-tuned τ; metrics reported on TEST only.",
        },
    }
