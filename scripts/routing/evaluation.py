"""Studies I–III characterization, complementarity, and statistical evaluation (RH1–RH3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from routing.constants import (
    ANALYSIS_COMPLEMENTARITY,
    ANALYSIS_ROUTING_RELEVANCE,
    ABLATION_LADDER,
    BOOTSTRAP_COUNT,
    BOOTSTRAP_SEED,
    CALIB_SIZE,
    CALIB_REDRAW_FOLDS,
    COL_COMPLEXITY,
    COL_ENTROPY_WEAK,
    COL_MARGIN_WEAK,
    COL_OPPORTUNITY,
    COMPLEMENTARITY_CV_FOLDS,
    COMPLEMENTARITY_CV_MIN_CLASS,
    COMPLEMENTARITY_STEPS,
    HEADLINE_SIGNALS,
    HYPOTHESES,
    LADDER_CROSS_FAMILY,
    PERMUTATION_COUNT,
    PROBE_DERIVED,
    PROBE_HEADLINE,
    REFERENCE_MODELS,
    ROUTING_RELEVANCE_SIGNALS,
    STABILITY_CALIB_DRAWS,
)
from routing.data import (
    _summary_stats,
    bucket_summary,
    distribution_by_bucket,
    list_signal_columns,
)
from routing.delong import compare_auc_delong

def _require_analysis_deps() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        from scipy.stats import kendalltau, mannwhitneyu, pearsonr, spearmanr
    except ImportError as exc:
        raise SystemExit("scipy required. Install: uv sync --extra analysis") from exc
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError as exc:
        raise SystemExit("scikit-learn required. Install: uv sync --extra analysis") from exc
    return spearmanr, kendalltau, pearsonr, roc_auc_score, mannwhitneyu, average_precision_score


def load_sklearn_deps():
    spearmanr, _, _, roc_auc_score, _, _ = _require_analysis_deps()
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit("Install: uv sync --extra analysis") from exc
    return LogisticRegression, roc_auc_score, Pipeline, StandardScaler, spearmanr


# --- Effect sizes & bootstrap ---


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float | None:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return None
    greater = sum(float(xi > yj) for xi in x for yj in y)
    less = sum(float(xi < yj) for xi in x for yj in y)
    return (greater - less) / (len(x) * len(y))


def cohens_d(x: np.ndarray, y: np.ndarray) -> float | None:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return None
    v1, v2 = np.var(x, ddof=1), np.var(y, ddof=1)
    n1, n2 = len(x), len(y)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled == 0:
        return None
    return float((np.mean(x) - np.mean(y)) / pooled)


def rank_biserial(x: np.ndarray, y: np.ndarray, mannwhitneyu) -> float | None:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 1 or len(y) < 1:
        return None
    try:
        u, _ = mannwhitneyu(x, y, alternative="two-sided")
    except ValueError:
        return None
    n1, n2 = len(x), len(y)
    return float(1.0 - (2.0 * u) / (n1 * n2))


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    stat_fn,
    n_boot: int = BOOTSTRAP_COUNT,
    seed: int = BOOTSTRAP_SEED,
    ci: float = 0.95,
) -> tuple[float | None, float | None]:
    mask = np.isfinite(x) & np.isfinite(y)
    xs, ys = x[mask], y[mask]
    n = len(xs)
    if n < 5:
        return None, None

    point = stat_fn(xs, ys)
    if point is None or not np.isfinite(point):
        return None, None

    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bx, by = xs[idx], ys[idx]
        if len(np.unique(by)) < 2:
            continue
        val = stat_fn(bx, by)
        if val is not None and np.isfinite(val):
            samples.append(float(val))

    if len(samples) < max(50, n_boot // 10):
        return None, None

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def summarize_values(values: np.ndarray) -> dict[str, float | int | None]:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
    return {
        "n": int(len(v)),
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
        "median": float(np.median(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def stat_spearman(xs: np.ndarray, ys: np.ndarray, *, spearmanr=None) -> float | None:
    if spearmanr is None:
        spearmanr = _require_analysis_deps()[0]
    if len(np.unique(ys)) < 2 or np.std(xs) < 1e-12:
        return None
    rho, _ = spearmanr(xs, ys)
    return float(rho) if np.isfinite(rho) else None


def stat_auroc(xs: np.ndarray, ys: np.ndarray, *, roc_auc_score=None) -> float | None:
    if roc_auc_score is None:
        roc_auc_score = _require_analysis_deps()[3]
    if len(np.unique(ys)) < 2 or np.std(xs) < 1e-12:
        return None
    try:
        return float(roc_auc_score(ys, xs))
    except ValueError:
        return None


def correlation_metrics(
    y: np.ndarray,
    x: np.ndarray,
    *,
    spearmanr,
    kendalltau,
    pearsonr,
    roc_auc_score,
    n_boot: int,
    bootstrap_seed: int,
    average_precision_score=None,
) -> dict[str, Any]:
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    empty: dict[str, Any] = {
        "n": n,
        "spearman_rho": None,
        "spearman_p": None,
        "spearman_ci_low": None,
        "spearman_ci_high": None,
        "kendall_tau": None,
        "kendall_p": None,
        "pearson_r": None,
        "pearson_p": None,
        "auroc": None,
        "auroc_abs": None,
        "auroc_ci_low": None,
        "auroc_ci_high": None,
        "auprc": None,
    }
    if n < 3:
        return empty

    xs = x[mask]
    ys = y[mask]
    rho, p_rho = spearmanr(xs, ys)
    tau, p_tau = kendalltau(xs, ys)
    r_pearson, p_pearson = pearsonr(xs, ys)

    result: dict[str, Any] = {
        "n": n,
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
        "spearman_ci_low": None,
        "spearman_ci_high": None,
        "kendall_tau": float(tau),
        "kendall_p": float(p_tau),
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "auroc": None,
        "auroc_abs": None,
        "auroc_ci_low": None,
        "auroc_ci_high": None,
        "auprc": None,
    }

    ci_lo, ci_hi = bootstrap_ci(
        xs, ys,
        stat_fn=lambda a, b: stat_spearman(a, b, spearmanr=spearmanr),
        n_boot=n_boot,
        seed=bootstrap_seed,
    )
    result["spearman_ci_low"] = ci_lo
    result["spearman_ci_high"] = ci_hi

    if len(np.unique(ys)) >= 2:
        try:
            auroc = float(roc_auc_score(ys, xs))
            result["auroc"] = auroc
            result["auroc_abs"] = abs(auroc - 0.5) + 0.5
            a_lo, a_hi = bootstrap_ci(
                xs, ys,
                stat_fn=lambda a, b: stat_auroc(a, b, roc_auc_score=roc_auc_score),
                n_boot=n_boot,
                seed=bootstrap_seed + 1,
            )
            result["auroc_ci_low"] = a_lo
            result["auroc_ci_high"] = a_hi
            if average_precision_score is not None:
                try:
                    result["auprc"] = float(average_precision_score(ys, xs))
                except ValueError:
                    pass
        except ValueError:
            pass

    return result


def effect_size_between_buckets(
    merged: pd.DataFrame,
    signal_col: str,
    bucket_a: str,
    bucket_b: str,
    mannwhitneyu,
) -> dict[str, Any]:
    a = merged.loc[merged["bucket"] == bucket_a, signal_col].to_numpy(dtype=float)
    b = merged.loc[merged["bucket"] == bucket_b, signal_col].to_numpy(dtype=float)
    return {
        "comparison": f"{bucket_a}_vs_{bucket_b}",
        f"n_{bucket_a}": int(np.isfinite(a).sum()),
        f"n_{bucket_b}": int(np.isfinite(b).sum()),
        "cliffs_delta": cliffs_delta(a, b),
        "cohens_d": cohens_d(a, b),
        "rank_biserial": rank_biserial(a, b, mannwhitneyu),
        bucket_a: _summary_stats(a),
        bucket_b: _summary_stats(b),
    }


def pairwise_bucket_comparisons(
    merged: pd.DataFrame,
    signal_col: str,
    mannwhitneyu,
) -> dict[str, Any]:
    pairs = [("opportunity", "easy"), ("opportunity", "weak_only"), ("opportunity", "too_hard")]
    return {
        f"{a}_vs_{b}": effect_size_between_buckets(merged, signal_col, a, b, mannwhitneyu)
        for a, b in pairs
    }


def probe_cost_table(*, pool_size: int = 2, max_new_tokens: int = 8) -> dict[str, Any]:
    m = pool_size
    return {
        "pool_size_m": m,
        "note": "Per-query inference cost for routing (oracle row is offline evaluation only).",
        "methods": [
            {
                "method": "always_weak",
                "role": "baseline",
                "prefill_probes_per_query": 0,
                "full_generations_per_query": 1,
                "routed_model": "weak",
                "description": "Always route to weak model — cheapest baseline.",
            },
            {
                "method": "always_strong",
                "role": "baseline",
                "prefill_probes_per_query": 0,
                "full_generations_per_query": 1,
                "routed_model": "strong",
                "description": "Always route to strong model — quality upper bound among static policies.",
            },
            {
                "method": "prefill_probes_only",
                "role": "this_work_signals",
                "prefill_probes_per_query": m,
                "full_generations_per_query": 0,
                "routed_model": "n/a",
                "description": "Prefill probe extraction — m forward passes, no generation.",
            },
            {
                "method": "signal_routed_one_generation",
                "role": "routing_policy",
                "prefill_probes_per_query": m,
                "full_generations_per_query": 1,
                "routed_model": "selected",
                "description": "Probe all pool members, then generate on chosen model only.",
            },
            {
                "method": "oracle_offline_labels",
                "role": "evaluation_only",
                "prefill_probes_per_query": 0,
                "full_generations_per_query": m,
                "max_new_tokens_per_generation": max_new_tokens,
                "routed_model": "all",
                "description": "Offline oracle — m full generations per query (never used for routing signals).",
            },
            {
                "method": "label_free_post_generation",
                "role": "literature_contrast",
                "prefill_probes_per_query": 0,
                "full_generations_per_query": m,
                "routed_model": "all",
                "description": "Smoothie-style: m full generations before routing decision.",
            },
        ],
        "ratio_probe_to_oracle_forward_equivalent": (
            f"1 prefill : 1 full generation ≈ 1:{max_new_tokens} token steps (order-of-magnitude)."
        ),
    }


# --- Routing relevance ---


def analyze_routing_relevance(
    merged: pd.DataFrame,
    oracle_path: Path,
    *,
    n_boot: int = BOOTSTRAP_COUNT,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    spearmanr, kendalltau, pearsonr, roc_auc_score, mannwhitneyu, average_precision_score = (
        _require_analysis_deps()
    )

    y_opp = merged["y_opp"].to_numpy(dtype=float)
    y_gap = merged["oracle_gap"].to_numpy(dtype=float)
    cols = list_signal_columns(merged)

    vs_opportunity: dict[str, Any] = {}
    vs_oracle_gap: dict[str, Any] = {}
    effect_sizes: dict[str, Any] = {}
    distributions: dict[str, Any] = {}

    boot_kw = {"n_boot": n_boot, "bootstrap_seed": bootstrap_seed}
    for col in cols:
        x = merged[col].to_numpy(dtype=float)
        vs_opportunity[col] = correlation_metrics(
            y_opp, x,
            spearmanr=spearmanr, kendalltau=kendalltau, pearsonr=pearsonr,
            roc_auc_score=roc_auc_score, average_precision_score=average_precision_score,
            **boot_kw,
        )
        vs_oracle_gap[col] = correlation_metrics(
            y_gap, x,
            spearmanr=spearmanr, kendalltau=kendalltau, pearsonr=pearsonr,
            roc_auc_score=roc_auc_score, average_precision_score=average_precision_score,
            **boot_kw,
        )
        if col in ROUTING_RELEVANCE_SIGNALS:
            effect_sizes[col] = pairwise_bucket_comparisons(merged, col, mannwhitneyu)

    for col in PROBE_HEADLINE:
        if col in merged.columns:
            distributions[col] = distribution_by_bucket(merged, col)

    if COL_COMPLEXITY in merged.columns:
        distributions[COL_COMPLEXITY] = distribution_by_bucket(merged, COL_COMPLEXITY)
        effect_sizes[COL_COMPLEXITY] = pairwise_bucket_comparisons(merged, COL_COMPLEXITY, mannwhitneyu)

    complementarity: dict[str, Any] = {}
    if {"entropy_w", "margin_w"}.issubset(merged.columns):
        ew = merged[COL_ENTROPY_WEAK].to_numpy(dtype=float)
        mw = merged[COL_MARGIN_WEAK].to_numpy(dtype=float)
        rho_hm, p_hm = spearmanr(ew, mw)
        complementarity = {
            "note": "Full complementarity ladder: run.py complementarity.",
            "rho_entropy_w_margin_w": float(rho_hm),
            "p_value": float(p_hm),
        }

    payload = json.loads(oracle_path.read_text())
    max_new_tokens = int(payload.get("max_new_tokens", 8))

    return {
        "analysis": ANALYSIS_ROUTING_RELEVANCE,
        "question": "How strongly do pre-inference signals relate to routing need?",
        "probe_headline": PROBE_HEADLINE,
        "probe_derived": PROBE_DERIVED,
        "hypotheses": HYPOTHESES,
        "headline_signals": HEADLINE_SIGNALS,
        "routing_relevance_signals": ROUTING_RELEVANCE_SIGNALS,
        "interpretation": {
            "focus": (
                "Characterize routing relevance via bucket distributions, "
                "Spearman rho, AUROC, and effect sizes — not routing policy tuning."
            ),
            "reporting_order": [
                "1. Signal distributions by all four oracle buckets (F1)",
                "2. Spearman rho + 95% bootstrap CI vs oracle_gap and vs opportunity",
                "3. Effect sizes (opportunity vs easy)",
                "4. AUROC + bootstrap CI (opportunity vs non-opportunity)",
                "5. ROC curves for headline signals (F2)",
            ],
        },
        "distributions_by_bucket": distributions,
        "correlation_vs_opportunity": vs_opportunity,
        "correlation_vs_oracle_gap": vs_oracle_gap,
        "pairwise_bucket_comparisons": effect_sizes,
        "effect_size_opportunity_vs_easy": {
            col: v.get("opportunity_vs_easy") for col, v in effect_sizes.items()
        },
        "bootstrap": {"n_resamples": n_boot, "ci_level": 0.95, "seed": bootstrap_seed},
        "probe_cost": probe_cost_table(max_new_tokens=max_new_tokens),
        "complementarity_preview": complementarity,
    }


def build_relevance_summary(
    *,
    merged: pd.DataFrame,
    weak_path: Path,
    strong_path: Path,
    oracle_path: Path,
    analysis: dict[str, Any],
) -> dict[str, Any]:
    return {
        "analysis": ANALYSIS_ROUTING_RELEVANCE,
        "weak_csv": str(weak_path),
        "strong_csv": str(strong_path),
        "oracle_json": str(oracle_path),
        "n_merged": len(merged),
        "schemas": {
            "weak": merged.get("schema_w", pd.Series(["unknown"])).iloc[0]
            if "schema_w" in merged.columns
            else "unknown",
            "strong": merged.get("schema_s", pd.Series(["unknown"])).iloc[0]
            if "schema_s" in merged.columns
            else "unknown",
        },
        "buckets": bucket_summary(merged),
        "routing_relevance": analysis,
        "notes": [
            "Primary characterization: signal distributions across all four oracle buckets (F1).",
            "Routing relevance: Spearman rho + 95% bootstrap CI (headline); Kendall/Pearson secondary.",
            "AUROC ranks opportunity vs non-opportunity — interpret with CI and effect sizes; no fixed cutoff.",
            "oracle_gap = strong_ok - weak_ok is a secondary target (appropriate-model RQ).",
            "Probe-derived: delta_entropy (H_w−H_s), delta_margin_gain (m_s−m_w); no extra forward passes.",
            "Routing thresholds are fit separately (routing evaluation).",
        ],
    }


# --- Complementarity ---


def _effective_cv_folds(y: np.ndarray, requested: int) -> int:
    """Pick fold count that stratification can support."""
    y_int = y.astype(int)
    if len(np.unique(y_int)) < 2:
        return 0
    min_class = int(np.min(np.bincount(y_int)))
    if min_class < COMPLEMENTARITY_CV_MIN_CLASS:
        return 0
    return max(2, min(requested, min_class))


def fit_logistic_auc_cv(
    X,
    y,
    *,
    LogisticRegression,
    Pipeline,
    StandardScaler,
    roc_auc_score,
    n_splits: int = COMPLEMENTARITY_CV_FOLDS,
) -> tuple[float | None, int | None]:
    """Out-of-fold AUROC from stratified k-fold logistic regression."""
    from sklearn.model_selection import StratifiedKFold

    y_arr = np.asarray(y, dtype=float)
    if len(np.unique(y_arr)) < 2 or X.shape[0] < max(5, X.shape[1] + 2):
        return None, None

    k = _effective_cv_folds(y_arr, n_splits)
    if k == 0:
        return None, None

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    oof_probs = np.full(len(y_arr), np.nan)
    for train_idx, test_idx in skf.split(X, y_arr):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, random_state=0)),
        ])
        clf.fit(X[train_idx], y_arr[train_idx])
        oof_probs[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

    if np.any(np.isnan(oof_probs)) or len(np.unique(y_arr)) < 2:
        return None, k
    try:
        return float(roc_auc_score(y_arr, oof_probs)), k
    except ValueError:
        return None, k


def fit_logistic_auc_insample(X, y, *, LogisticRegression, Pipeline, StandardScaler, roc_auc_score):
    """In-sample AUROC — appendix diagnostic only."""
    if len(np.unique(y)) < 2 or X.shape[0] < max(5, X.shape[1] + 2):
        return None
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, random_state=0)),
    ])
    clf.fit(X, y)
    return float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))


def bootstrap_auc_delta(X_small, X_large, y, *, boot_count: int, seed: int, fit_auc: Callable):
    a0, a1 = fit_auc(X_small, y), fit_auc(X_large, y)
    if a0 is None or a1 is None:
        return None, None, None
    delta = a1 - a0
    rng = np.random.default_rng(seed)
    n = len(y)
    samples: list[float] = []
    for _ in range(boot_count):
        idx = rng.integers(0, n, size=n)
        sa, la = fit_auc(X_small[idx], y[idx]), fit_auc(X_large[idx], y[idx])
        if sa is not None and la is not None:
            samples.append(la - sa)
    if len(samples) < max(50, boot_count // 10):
        return delta, None, None
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return delta, float(lo), float(hi)


def _make_logistic_pipeline(LogisticRegression, StandardScaler, Pipeline):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, random_state=0)),
    ])


def fit_logistic_auc_holdout(
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    LogisticRegression,
    Pipeline,
    StandardScaler,
    roc_auc_score,
) -> float | None:
    """CALIB-fit / TEST-eval AUROC (Study III primary protocol)."""
    y_tr = np.asarray(y_train, dtype=float)
    y_te = np.asarray(y_test, dtype=float)
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return None
    if X_train.shape[0] < max(5, X_train.shape[1] + 2):
        return None
    clf = _make_logistic_pipeline(LogisticRegression, StandardScaler, Pipeline)
    clf.fit(X_train, y_tr)
    try:
        return float(roc_auc_score(y_te, clf.predict_proba(X_test)[:, 1]))
    except ValueError:
        return None


def fit_logistic_probs_holdout(
    X_train,
    y_train,
    X_test,
    *,
    LogisticRegression,
    Pipeline,
    StandardScaler,
) -> np.ndarray | None:
    """CALIB-fit logistic; return P(y=1) on TEST."""
    y_tr = np.asarray(y_train, dtype=float)
    if len(np.unique(y_tr)) < 2 or X_train.shape[0] < max(5, X_train.shape[1] + 2):
        return None
    clf = _make_logistic_pipeline(LogisticRegression, StandardScaler, Pipeline)
    clf.fit(X_train, y_tr)
    return clf.predict_proba(X_test)[:, 1].astype(float)


def calibration_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, list[float] | int | None]:
    """Bin predicted opportunity probability vs observed rate (TEST only)."""
    try:
        from sklearn.calibration import calibration_curve
    except ImportError as exc:
        raise RuntimeError("scikit-learn required for calibration curve") from exc
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {"n_bins": n_bins, "mean_predicted": None, "fraction_positive": None}
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return {
        "n_bins": n_bins,
        "mean_predicted": [float(x) for x in mean_pred],
        "fraction_positive": [float(x) for x in frac_pos],
        "n_test": int(len(y_true)),
    }


def fit_logistic_coefficients(
    X,
    y,
    feature_names: list[str],
    *,
    LogisticRegression,
    Pipeline,
    StandardScaler,
) -> dict[str, float] | None:
    """Standardized logistic coefficients (after StandardScaler)."""
    y_arr = np.asarray(y, dtype=float)
    if len(np.unique(y_arr)) < 2 or X.shape[0] < max(5, X.shape[1] + 2):
        return None
    clf = _make_logistic_pipeline(LogisticRegression, StandardScaler, Pipeline)
    clf.fit(X, y_arr)
    coefs = clf.named_steps["lr"].coef_[0]
    return {name: float(c) for name, c in zip(feature_names, coefs)}


def bootstrap_holdout_auc(
    X_cal,
    y_cal,
    X_te,
    y_te,
    *,
    boot_count: int,
    seed: int,
    fit_holdout,
) -> tuple[float | None, float | None, float | None]:
    real = fit_holdout(X_cal, y_cal, X_te, y_te)
    if real is None:
        return None, None, None
    rng = np.random.default_rng(seed)
    n = len(y_te)
    samples: list[float] = []
    for _ in range(boot_count):
        idx = rng.integers(0, n, size=n)
        auc = fit_holdout(X_cal, y_cal, X_te[idx], y_te[idx])
        if auc is not None:
            samples.append(auc)
    if len(samples) < max(50, boot_count // 10):
        return real, None, None
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return real, float(lo), float(hi)


def bootstrap_holdout_auc_delta(
    X_small,
    X_large,
    y,
    calib_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    boot_count: int,
    seed: int,
    fit_holdout,
) -> tuple[float | None, float | None, float | None]:
    a0 = fit_holdout(X_small[calib_idx], y[calib_idx], X_small[test_idx], y[test_idx])
    a1 = fit_holdout(X_large[calib_idx], y[calib_idx], X_large[test_idx], y[test_idx])
    if a0 is None or a1 is None:
        return None, None, None
    delta = a1 - a0
    rng = np.random.default_rng(seed)
    n = len(test_idx)
    samples: list[float] = []
    for _ in range(boot_count):
        idx = rng.integers(0, n, size=n)
        te = test_idx[idx]
        sa = fit_holdout(X_small[calib_idx], y[calib_idx], X_small[te], y[te])
        la = fit_holdout(X_large[calib_idx], y[calib_idx], X_large[te], y[te])
        if sa is not None and la is not None:
            samples.append(la - sa)
    if len(samples) < max(50, boot_count // 10):
        return delta, None, None
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return delta, float(lo), float(hi)


def compute_rank_agreement(frame: pd.DataFrame, *, spearmanr, kendalltau) -> dict[str, Any]:
    """Rank correlation between c(q) and H_w orderings (RH3 interpretability)."""
    c = frame[COL_COMPLEXITY].to_numpy(dtype=float)
    h = frame[COL_ENTROPY_WEAK].to_numpy(dtype=float)
    m = frame[COL_MARGIN_WEAK].to_numpy(dtype=float)
    mask = np.isfinite(c) & np.isfinite(h)
    c, h = c[mask], h[mask]
    m = m[mask]
    rho_ch, p_ch = spearmanr(c, h)
    tau_ch, p_tau = kendalltau(c, h)
    rho_cm, _ = spearmanr(c, m)
    rho_hm, _ = spearmanr(h, m)
    return {
        "placement": "primary_support",
        "interpretation": "Low agreement ⇒ families rank queries differently (complementary).",
        "reporting_order": "spearman_rho → kendall_tau → p-values (supporting)",
        "complexity_entropy": {
            "spearman_rho": float(rho_ch) if rho_ch is not None else None,
            "spearman_p": float(p_ch) if p_ch is not None else None,
            "kendall_tau": float(tau_ch) if tau_ch is not None else None,
            "kendall_p": float(p_tau) if p_tau is not None else None,
        },
        "complexity_margin": {"spearman_rho": float(rho_cm) if rho_cm is not None else None},
        "entropy_margin": {"spearman_rho": float(rho_hm) if rho_hm is not None else None},
    }


def run_failure_analysis(
    frame: pd.DataFrame,
    *,
    signal_col: str = COL_ENTROPY_WEAK,
    n: int = 20,
) -> dict[str, Any]:
    """Top/bottom signal queries for qualitative Discussion (false pos/neg on opportunity)."""
    need = {"query_id", "user_content", "bucket", "y_opp", signal_col}
    missing = need - set(frame.columns)
    if missing:
        raise ValueError(f"failure analysis requires columns: {missing}")
    data = frame.dropna(subset=[signal_col]).copy()
    data = data.sort_values(signal_col, ascending=False)
    high = data.head(n)
    low = data.tail(n)

    def _rows(sub: pd.DataFrame, tail: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for _, r in sub.iterrows():
            opp = int(r["y_opp"])
            out.append({
                "query_id": r["query_id"],
                "bucket": r["bucket"],
                "y_opp": opp,
                signal_col: float(r[signal_col]),
                "user_content": (r["user_content"][:500] + "…") if len(str(r["user_content"])) > 500 else r["user_content"],
                "error_type": (
                    "false_positive_opportunity" if tail == "high" and opp == 0
                    else "false_negative_opportunity" if tail == "low" and opp == 1
                    else "true_positive" if tail == "high" and opp == 1
                    else "true_negative" if tail == "low" and opp == 0
                    else "other"
                ),
            })
        return out

    return {
        "analysis": "failure_cases",
        "signal": signal_col,
        "n_per_tail": n,
        "highest_entropy": _rows(high, "high"),
        "lowest_entropy": _rows(low, "low"),
        "note": "Use for Discussion: what question types confuse the signal?",
    }


def permutation_auroc_test(
    X,
    y,
    *,
    fit_auc,
    n_perm: int = PERMUTATION_COUNT,
    seed: int = BOOTSTRAP_SEED,
    shuffle_cols: list[int] | None = None,
    calib_idx: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
    fit_holdout=None,
) -> dict[str, Any]:
    """Null AUROC from shuffling signal columns; one-sided p vs observed.

    Nested eval: when calib_idx/test_idx/fit_holdout are set, shuffle features on
    CALIB only, fit on CALIB, evaluate on held-out TEST (never fits on TEST).
    """
    nested = (
        calib_idx is not None
        and test_idx is not None
        and fit_holdout is not None
    )

    if nested:
        real = fit_holdout(
            X[calib_idx], y[calib_idx], X[test_idx], y[test_idx],
        )
    else:
        real = fit_auc(X, y)

    if real is None:
        return {"observed_auroc": None, "p_value": None, "n_perm": n_perm}

    cols = shuffle_cols if shuffle_cols is not None else list(range(X.shape[1]))
    rng = np.random.default_rng(seed)
    null: list[float] = []
    for _ in range(n_perm):
        Xp = X.copy()
        for col in cols:
            if nested:
                vals = Xp[calib_idx, col].copy()
                rng.shuffle(vals)
                Xp[calib_idx, col] = vals
            else:
                rng.shuffle(Xp[:, col])
        if nested:
            auc = fit_holdout(
                Xp[calib_idx], y[calib_idx], Xp[test_idx], y[test_idx],
            )
        else:
            auc = fit_auc(Xp, y)
        if auc is not None:
            null.append(auc)
    if not null:
        return {"observed_auroc": real, "p_value": None, "n_perm": n_perm}
    p = (sum(v >= real for v in null) + 1) / (len(null) + 1)
    return {
        "observed_auroc": real,
        "null_mean": float(np.mean(null)),
        "null_std": float(np.std(null)),
        "p_value": float(p),
        "n_perm": n_perm,
        "n_valid": len(null),
        "eval": "calib_fit_test_eval" if nested else "in_sample",
    }


def score_ablation_ladder(
    frame: pd.DataFrame,
    opp_labels: np.ndarray,
    *,
    fit_auc,
    fit_holdout=None,
    LogisticRegression,
    Pipeline,
    StandardScaler,
    roc_auc_score,
    cv_folds: int,
    eval_mode: str,
    calib_idx: np.ndarray | None,
    test_idx: np.ndarray | None,
    boot_count: int = 0,
    boot_seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Four-model ablation ladder with AUROC + 95% CI per rung."""
    models: list[dict[str, Any]] = []
    for i, (label, cols) in enumerate(ABLATION_LADDER):
        X = build_feature_matrix(frame, cols)
        ci_lo, ci_hi = None, None
        if eval_mode == "holdout" and calib_idx is not None and test_idx is not None and fit_holdout:
            auroc, ci_lo, ci_hi = bootstrap_holdout_auc(
                X[calib_idx], opp_labels[calib_idx], X[test_idx], opp_labels[test_idx],
                boot_count=boot_count or BOOTSTRAP_COUNT,
                seed=boot_seed + i,
                fit_holdout=fit_holdout,
            )
            eval_label = "calib_fit_test_eval"
        elif eval_mode == "holdout" and calib_idx is not None and test_idx is not None:
            auroc = fit_logistic_auc_holdout(
                X[calib_idx], opp_labels[calib_idx],
                X[test_idx], opp_labels[test_idx],
                LogisticRegression=LogisticRegression,
                Pipeline=Pipeline,
                StandardScaler=StandardScaler,
                roc_auc_score=roc_auc_score,
            )
            eval_label = "calib_fit_test_eval"
        else:
            auroc = fit_auc(X, opp_labels)
            eval_label = "stratified_kfold_cv"
        models.append({
            "label": label,
            "columns": cols,
            "auroc": auroc,
            "auroc_ci_low": ci_lo,
            "auroc_ci_high": ci_hi,
            "auroc_eval": eval_label,
        })
    return models


def build_feature_matrix(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.column_stack([frame[c].to_numpy(dtype=float) for c in cols])


def filter_complete_rows(frame: pd.DataFrame) -> pd.DataFrame:
    need = {COL_OPPORTUNITY, COL_COMPLEXITY, COL_ENTROPY_WEAK, COL_MARGIN_WEAK}
    missing = need - set(frame.columns)
    if missing:
        raise ValueError(f"merged CSV missing columns: {missing}. Pass --features-csv when merging.")
    out = frame.copy()
    if "weak_ok" in out.columns:
        out["weak_correct"] = out["weak_ok"].astype(int)
    mask = np.ones(len(out), dtype=bool)
    for col in [COL_COMPLEXITY, COL_ENTROPY_WEAK, COL_MARGIN_WEAK, COL_OPPORTUNITY]:
        mask &= np.isfinite(out[col].to_numpy(dtype=float))
    out = out.loc[mask].copy()
    if len(out) < 10:
        raise ValueError(f"too few complete rows: {len(out)}")
    return out


def fit_model_spec(
    frame,
    label,
    cols,
    opp_labels,
    *,
    LogisticRegression,
    Pipeline,
    StandardScaler,
    roc_auc_score,
    role,
    cv_folds: int,
):
    X = build_feature_matrix(frame, cols)
    auroc, folds_used = fit_logistic_auc_cv(
        X, opp_labels,
        LogisticRegression=LogisticRegression,
        Pipeline=Pipeline,
        StandardScaler=StandardScaler,
        roc_auc_score=roc_auc_score,
        n_splits=cv_folds,
    )
    return {
        "label": label,
        "role": role,
        "columns": cols,
        "auroc": auroc,
        "auroc_eval": "stratified_kfold_cv",
        "cv_folds": folds_used,
        "auroc_insample_diagnostic": fit_logistic_auc_insample(
            X, opp_labels,
            LogisticRegression=LogisticRegression,
            Pipeline=Pipeline,
            StandardScaler=StandardScaler,
            roc_auc_score=roc_auc_score,
        ),
    }


def score_opportunity_models(frame, opp_labels, *, fit_auc_cv, cv_folds: int):
    c = frame[COL_COMPLEXITY].to_numpy(dtype=float).reshape(-1, 1)
    h = frame[COL_ENTROPY_WEAK].to_numpy(dtype=float).reshape(-1, 1)
    m = frame[COL_MARGIN_WEAK].to_numpy(dtype=float).reshape(-1, 1)
    specs = [
        ("complexity_only", c),
        ("complexity_entropy", np.column_stack([c, h])),
        ("complexity_joint", np.column_stack([c, h, m])),
    ]
    return {
        label: {"auroc": fit_auc_cv(X, opp_labels), "cv_folds": cv_folds}
        for label, X in specs
    }


def score_cross_family_diagnostics(frame, opp_labels, *, fit_auc_cv, cv_folds: int):
    """Appendix-only: c+m vs c+H when margin may dominate entropy."""
    c = frame[COL_COMPLEXITY].to_numpy(dtype=float).reshape(-1, 1)
    h = frame[COL_ENTROPY_WEAK].to_numpy(dtype=float).reshape(-1, 1)
    m = frame[COL_MARGIN_WEAK].to_numpy(dtype=float).reshape(-1, 1)
    specs = [
        ("complexity_margin", np.column_stack([c, m])),
        ("complexity_entropy", np.column_stack([c, h])),
    ]
    models = {
        label: {"auroc": fit_auc_cv(X, opp_labels), "cv_folds": cv_folds}
        for label, X in specs
    }
    cm, ch = models["complexity_margin"]["auroc"], models["complexity_entropy"]["auroc"]
    note = None
    if cm is not None and ch is not None and cm > ch + 0.02:
        note = "complexity_margin exceeds complexity_entropy — margin may dominate entropy increment."
    return {
        "placement": "appendix",
        "note": note or "Compare c+m vs c+H when interpreting family increments.",
        "models": models,
    }


def score_weak_diagnostic(
    frame,
    opp_labels,
    *,
    fit_eval,
    calib_idx: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
    eval_label: str = "calib_fit_test_eval",
):
    if "weak_correct" not in frame.columns:
        return None
    w = frame["weak_correct"].to_numpy(dtype=float).reshape(-1, 1)
    c = frame[COL_COMPLEXITY].to_numpy(dtype=float).reshape(-1, 1)
    h = frame[COL_ENTROPY_WEAK].to_numpy(dtype=float).reshape(-1, 1)
    m = frame[COL_MARGIN_WEAK].to_numpy(dtype=float).reshape(-1, 1)
    specs = [
        ("weak_only", w, ["weak_correct"]),
        ("weak_entropy", np.column_stack([w, h]), ["weak_correct", COL_ENTROPY_WEAK]),
        ("weak_complexity", np.column_stack([w, c]), ["weak_correct", COL_COMPLEXITY]),
        ("weak_joint", np.column_stack([w, c, h, m]), ["weak_correct", COL_COMPLEXITY, COL_ENTROPY_WEAK, COL_MARGIN_WEAK]),
    ]
    return {
        "placement": "primary_evidence",
        "note": "Opportunity ~ weak_correct + signals — all models fit CALIB, AUROC on TEST.",
        "auroc_eval": eval_label,
        "n_calib": int(len(calib_idx)) if calib_idx is not None else None,
        "n_test": int(len(test_idx)) if test_idx is not None else None,
        "models": {
            label: {"auroc": fit_eval(X, opp_labels), "columns": cols}
            for label, X, cols in specs
        },
    }


def _index_mask(data: pd.DataFrame, ids: set[str]) -> np.ndarray:
    return data["query_id"].astype(str).isin(ids).to_numpy()


def run_calib_redraw_eval(
    data: pd.DataFrame,
    opp_labels: np.ndarray,
    *,
    test_ids: set[str],
    calib_ids_canonical: set[str],
    calib_size: int,
    n_folds: int,
    base_seed: int,
    fit_holdout,
    model_mats: dict[str, np.ndarray],
) -> dict[str, Any]:
    """5-fold CALIB redraw with fixed TEST: fold 0 = canonical split, folds 1+ redrawn."""
    from routing.splits import draw_calib_ids

    pool = data["query_id"].astype(str).tolist()
    test_mask = data["query_id"].astype(str).isin(test_ids)
    test_idx = np.where(test_mask.to_numpy())[0]
    y_test = opp_labels[test_idx]

    calib_sets: list[set[str]] = [calib_ids_canonical]
    for f in range(1, n_folds):
        calib_sets.append(
            draw_calib_ids(
                pool, calib_size=calib_size, seed=base_seed + f * 97, exclude=test_ids,
            )
        )

    folds: list[dict[str, Any]] = []
    for fold, calib_set in enumerate(calib_sets):
        calib_idx = np.where(data["query_id"].astype(str).isin(calib_set).to_numpy())[0]
        row: dict[str, Any] = {"fold": fold, "n_calib": int(len(calib_idx)), "canonical": fold == 0}
        aurocs: dict[str, float | None] = {}
        for label, X in model_mats.items():
            auc = fit_holdout(
                X[calib_idx], opp_labels[calib_idx], X[test_idx], y_test,
            )
            aurocs[label] = auc
            row[f"auroc_{label}"] = auc
        c_auc = aurocs.get("complexity_only")
        j_auc = aurocs.get("complexity_joint")
        row["delta_auroc_c_to_joint"] = (
            (j_auc - c_auc) if c_auc is not None and j_auc is not None else None
        )
        folds.append(row)

    def _mean_std(key: str) -> dict[str, float | None]:
        vals = [f[key] for f in folds if f.get(key) is not None]
        if not vals:
            return {"mean": None, "std": None}
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }

    summary: dict[str, Any] = {
        "delta_auroc_c_to_joint": _mean_std("delta_auroc_c_to_joint"),
    }
    for label in model_mats:
        summary[f"auroc_{label}"] = _mean_std(f"auroc_{label}")

    return {
        "placement": "primary_support",
        "protocol": "fold 0 = canonical CALIB from splits.json; folds 1..K-1 redrawn; TEST fixed",
        "n_folds": n_folds,
        "folds": folds,
        "summary": summary,
    }


def _delong_pair(
    y_test: np.ndarray,
    X_cal: np.ndarray,
    X_te: np.ndarray,
    y_cal: np.ndarray,
    *,
    LogisticRegression,
    Pipeline,
    StandardScaler,
    Xb_cal: np.ndarray,
    Xb_te: np.ndarray,
) -> dict[str, float | None]:
    pa = fit_logistic_probs_holdout(
        X_cal, y_cal, X_te,
        LogisticRegression=LogisticRegression, Pipeline=Pipeline, StandardScaler=StandardScaler,
    )
    pb = fit_logistic_probs_holdout(
        Xb_cal, y_cal, Xb_te,
        LogisticRegression=LogisticRegression, Pipeline=Pipeline, StandardScaler=StandardScaler,
    )
    if pa is None or pb is None:
        return {"p_value": None, "delta_auroc": None, "z_statistic": None}
    try:
        return compare_auc_delong(y_test, pa, pb)
    except ValueError:
        return {"p_value": None, "delta_auroc": None, "z_statistic": None}


def run_complementarity_analysis(
    frame: pd.DataFrame,
    *,
    boot_count: int = BOOTSTRAP_COUNT,
    boot_seed: int = BOOTSTRAP_SEED,
    cv_folds: int = COMPLEMENTARITY_CV_FOLDS,
    eval_mode: str = "holdout",
    calib_ids: set[str] | None = None,
    test_ids: set[str] | None = None,
    permutation_n: int = 0,
    permutation_seed: int = BOOTSTRAP_SEED,
    allow_cv: bool = False,
    calib_redraw_folds: int = CALIB_REDRAW_FOLDS,
) -> dict[str, Any]:
    """Study III complementarity — nested eval: fit logistic on CALIB, AUROC on TEST."""
    LogisticRegression, roc_auc_score, Pipeline, StandardScaler, spearmanr = load_sklearn_deps()
    data = filter_complete_rows(frame)
    if "query_id" not in data.columns:
        raise ValueError("merged CSV must include query_id for nested evaluation")

    use_holdout = eval_mode == "holdout"
    if not use_holdout:
        if not allow_cv:
            raise ValueError(
                "Study III requires nested evaluation (eval_mode=holdout). "
                "Pass calib_ids + test_ids from splits.json. Dev only: allow_cv=True."
            )
    else:
        if not calib_ids or not test_ids:
            raise ValueError("holdout eval requires calib_ids and test_ids (from splits.json)")

    opp_labels = data[COL_OPPORTUNITY].to_numpy(dtype=float)
    folds_effective = _effective_cv_folds(opp_labels, cv_folds)

    calib_mask = test_mask = None
    calib_idx = test_idx = None
    if use_holdout:
        calib_mask = _index_mask(data, calib_ids)
        test_mask = _index_mask(data, test_ids)
        if not calib_mask.any() or not test_mask.any():
            raise ValueError("calib or test split has no rows in merged CSV")
        calib_idx = np.where(calib_mask)[0]
        test_idx = np.where(test_mask)[0]
        overlap = calib_ids & test_ids
        if overlap:
            raise ValueError(f"calib/test overlap in split manifest: {len(overlap)} ids")

    def fit_holdout(Xc, yc, Xt, yt):
        return fit_logistic_auc_holdout(
            Xc, yc, Xt, yt,
            LogisticRegression=LogisticRegression,
            Pipeline=Pipeline,
            StandardScaler=StandardScaler,
            roc_auc_score=roc_auc_score,
        )

    def fit_auc(X, y):
        if use_holdout and calib_idx is not None and test_idx is not None:
            return fit_holdout(
                X[calib_idx], y[calib_idx],
                X[test_idx], y[test_idx],
            )
        auroc, _ = fit_logistic_auc_cv(
            X, y,
            LogisticRegression=LogisticRegression,
            Pipeline=Pipeline,
            StandardScaler=StandardScaler,
            roc_auc_score=roc_auc_score,
            n_splits=cv_folds,
        )
        return auroc

    eval_label = "calib_fit_test_eval" if use_holdout else "stratified_kfold_cv_dev_only"

    col_map: dict[str, list[str]] = {}
    models: dict[str, dict[str, Any]] = {}
    ladder_labels = {n for n, _ in LADDER_CROSS_FAMILY}

    for label, cols in LADDER_CROSS_FAMILY + REFERENCE_MODELS:
        col_map[label] = cols
        role = "cross_family" if label in ladder_labels else "solo_reference"
        X = build_feature_matrix(data, cols)
        models[label] = {
            "label": label,
            "role": role,
            "columns": cols,
            "auroc": fit_auc(X, opp_labels),
            "auroc_eval": eval_label,
            "cv_folds": folds_effective if not use_holdout else None,
        }

    ablation = score_ablation_ladder(
        data, opp_labels,
        fit_auc=fit_auc,
        fit_holdout=fit_holdout if use_holdout else None,
        LogisticRegression=LogisticRegression,
        Pipeline=Pipeline,
        StandardScaler=StandardScaler,
        roc_auc_score=roc_auc_score,
        cv_folds=cv_folds,
        eval_mode=eval_mode,
        calib_idx=calib_idx,
        test_idx=test_idx,
        boot_count=boot_count,
        boot_seed=boot_seed,
    )

    X_c = build_feature_matrix(data, [COL_COMPLEXITY])
    X_joint_full = build_feature_matrix(
        data, [COL_COMPLEXITY, COL_ENTROPY_WEAK, COL_MARGIN_WEAK]
    )
    auroc_c = fit_auc(X_c, opp_labels)
    auroc_joint = fit_auc(X_joint_full, opp_labels)
    if use_holdout and calib_idx is not None and test_idx is not None:
        primary_delta, primary_lo, primary_hi = bootstrap_holdout_auc_delta(
            X_c, X_joint_full, opp_labels, calib_idx, test_idx,
            boot_count=boot_count, seed=boot_seed + 11, fit_holdout=fit_holdout,
        )
    else:
        primary_delta, primary_lo, primary_hi = bootstrap_auc_delta(
            X_c, X_joint_full, opp_labels,
            boot_count=boot_count, seed=boot_seed + 11, fit_auc=fit_auc,
        )

    auc_steps: dict[str, Any] = {}
    for small, large, step_name in COMPLEMENTARITY_STEPS:
        Xs = build_feature_matrix(data, col_map[small])
        Xl = build_feature_matrix(data, col_map[large])
        a0, a1 = models[small]["auroc"], models[large]["auroc"]
        if use_holdout and calib_idx is not None and test_idx is not None:
            delta, lo, hi = bootstrap_holdout_auc_delta(
                Xs, Xl, opp_labels, calib_idx, test_idx,
                boot_count=boot_count, seed=boot_seed, fit_holdout=fit_holdout,
            )
            y_te = opp_labels[test_idx]
            delong = _delong_pair(
                y_te,
                Xl[calib_idx], Xl[test_idx], opp_labels[calib_idx],
                LogisticRegression=LogisticRegression,
                Pipeline=Pipeline,
                StandardScaler=StandardScaler,
                Xb_cal=Xs[calib_idx],
                Xb_te=Xs[test_idx],
            )
        else:
            delta, lo, hi = bootstrap_auc_delta(
                Xs, Xl, opp_labels, boot_count=boot_count, seed=boot_seed, fit_auc=fit_auc
            )
            delong = {"p_value": None, "delta_auroc": None, "z_statistic": None}
        auc_steps[step_name] = {
            "from": small,
            "to": large,
            "delta_auroc": delta,
            "ci_low": lo,
            "ci_high": hi,
            "delong_p": delong.get("p_value"),
            "delong_z": delong.get("z_statistic"),
            "auc_from": models[small]["auroc"],
            "auc_to": models[large]["auroc"],
        }

    X_ent = build_feature_matrix(data, [COL_ENTROPY_WEAK])
    X_joint = build_feature_matrix(data, [COL_ENTROPY_WEAK, COL_MARGIN_WEAK])
    if use_holdout and calib_idx is not None and test_idx is not None:
        delta_em, em_lo, em_hi = bootstrap_holdout_auc_delta(
            X_ent, X_joint, opp_labels, calib_idx, test_idx,
            boot_count=boot_count, seed=boot_seed + 7, fit_holdout=fit_holdout,
        )
    else:
        delta_em, em_lo, em_hi = bootstrap_auc_delta(
            X_ent, X_joint, opp_labels, boot_count=boot_count, seed=boot_seed + 7, fit_auc=fit_auc
        )

    test_frame = data.iloc[test_idx] if use_holdout and test_idx is not None else data
    rho_em, p_em = spearmanr(
        test_frame[COL_ENTROPY_WEAK].to_numpy(dtype=float),
        test_frame[COL_MARGIN_WEAK].to_numpy(dtype=float),
    )

    perm_kwargs = {}
    if use_holdout and calib_idx is not None and test_idx is not None:
        perm_kwargs = {
            "calib_idx": calib_idx,
            "test_idx": test_idx,
            "fit_holdout": fit_holdout,
        }

    permutation: dict[str, Any] = {}
    if permutation_n > 0:
        for col, label in [
            (COL_COMPLEXITY, "complexity_solo"),
            (COL_ENTROPY_WEAK, "entropy_solo"),
            (COL_MARGIN_WEAK, "margin_solo"),
        ]:
            X = build_feature_matrix(data, [col])
            permutation[label] = permutation_auroc_test(
                X, opp_labels, fit_auc=fit_auc, n_perm=permutation_n,
                seed=permutation_seed, **perm_kwargs,
            )
        permutation["complexity_joint"] = permutation_auroc_test(
            X_joint_full, opp_labels, fit_auc=fit_auc,
            n_perm=permutation_n, seed=permutation_seed + 3,
            shuffle_cols=[0, 1, 2], **perm_kwargs,
        )

    from scipy.stats import kendalltau

    rank_agreement = compute_rank_agreement(test_frame, spearmanr=spearmanr, kendalltau=kendalltau)
    confound = score_weak_diagnostic(
        data, opp_labels, fit_eval=fit_auc,
        calib_idx=calib_idx, test_idx=test_idx, eval_label=eval_label,
    )

    nested_evaluation = {
        "protocol": "calib_fit_test_eval",
        "rule": "Every logistic model fits on CALIB only; AUROC reported on TEST only.",
        "d46_screening": "c(q) selected on CALIB before TEST merge (no re-screening on TEST).",
        "calib_redraw_folds": calib_redraw_folds if use_holdout else None,
        "n_calib": int(calib_mask.sum()) if calib_mask is not None else None,
        "n_test": int(test_mask.sum()) if test_mask is not None else None,
        "forbidden": "Do not fit logistic models on TEST rows for Study III reporting.",
    }

    delong_primary = {"p_value": None, "z_statistic": None}
    calibration: dict[str, Any] = {"placement": "primary_figure", "model": "complexity_joint"}
    repeated_calib: dict[str, Any] | None = None

    if use_holdout and calib_idx is not None and test_idx is not None and calib_ids:
        y_te = opp_labels[test_idx]
        delong_primary = _delong_pair(
            y_te,
            X_joint_full[calib_idx], X_joint_full[test_idx], opp_labels[calib_idx],
            LogisticRegression=LogisticRegression,
            Pipeline=Pipeline,
            StandardScaler=StandardScaler,
            Xb_cal=X_c[calib_idx],
            Xb_te=X_c[test_idx],
        )
        joint_probs = fit_logistic_probs_holdout(
            X_joint_full[calib_idx], opp_labels[calib_idx], X_joint_full[test_idx],
            LogisticRegression=LogisticRegression, Pipeline=Pipeline, StandardScaler=StandardScaler,
        )
        if joint_probs is not None:
            calibration.update(calibration_curve_data(y_te.astype(int), joint_probs))

        if calib_redraw_folds > 1:
            model_mats = {label: build_feature_matrix(data, cols) for label, cols in ABLATION_LADDER}
            repeated_calib = run_calib_redraw_eval(
                data, opp_labels,
                test_ids=test_ids,
                calib_ids_canonical=calib_ids,
                calib_size=len(calib_ids),
                n_folds=calib_redraw_folds,
                base_seed=boot_seed,
                fit_holdout=fit_holdout,
                model_mats=model_mats,
            )

    primary_endpoint = {
        "definition": "ΔAUROC: c(q) alone → c+H+m joint model",
        "from_model": "complexity_only",
        "to_model": "complexity_joint",
        "auroc_from": auroc_c,
        "auroc_to": auroc_joint,
        "delta_auroc": primary_delta,
        "ci_low": primary_lo,
        "ci_high": primary_hi,
        "delong_p": delong_primary.get("p_value"),
        "delong_z": delong_primary.get("z_statistic"),
        "permutation_p": (
            permutation.get("complexity_joint", {}).get("p_value") if permutation else None
        ),
        "reporting_order": "delta_auroc → 95% bootstrap CI → DeLong p → permutation p",
        "repeated_calib_summary": (
            repeated_calib.get("summary", {}).get("delta_auroc_c_to_joint")
            if repeated_calib else None
        ),
    }

    return {
        "analysis": ANALYSIS_COMPLEMENTARITY,
        "question": "Do model-independent and model-dependent families complement each other?",
        "hypotheses": {"RH3": HYPOTHESES["RH3"]},
        "nested_evaluation": nested_evaluation,
        "n": int(len(data)),
        "n_test_reported": int(test_mask.sum()) if test_mask is not None else int(len(data)),
        "auroc_eval": eval_label,
        "eval_mode": eval_mode,
        "holdout": {
            "n_calib": int(calib_mask.sum()) if calib_mask is not None else None,
            "n_test": int(test_mask.sum()) if test_mask is not None else None,
        } if use_holdout else None,
        "primary_endpoint": primary_endpoint,
        "repeated_calib_folds": repeated_calib,
        "calibration_curve": calibration,
        "ablation_ladder": {
            "placement": "primary_figure",
            "models": ablation,
        },
        "rank_agreement": rank_agreement,
        "auc_increments": {
            "placement": "primary_support",
            "reporting_order": "delta_auroc → 95% bootstrap CI → DeLong p",
            "steps": auc_steps,
        },
        "confound_control": {
            "placement": "primary_support",
            **(confound or {}),
        },
        "permutation_tests": {
            "placement": "primary_support",
            "note": "Report after ΔAUROC and CI in prose.",
            "tests": permutation if permutation else None,
        },
        "calibration_stability": {
            "placement": "appendix",
            "note": "Legacy alias; see repeated_calib_folds in main output",
            "run": "scripts/run.py stability",
        },
        "cross_family_ladder": {
            "placement": "primary_support",
            "models": [models[n] for n, _ in LADDER_CROSS_FAMILY],
        },
        "solo_reference": {
            "placement": "appendix",
            "models": [models[n] for n, _ in REFERENCE_MODELS],
        },
        "within_dep_family": {
            "placement": "appendix",
            "rho_entropy_margin": float(rho_em) if rho_em is not None else None,
            "p_value_report": float(p_em) if p_em is not None else None,
            "delta_auroc_entropy_margin": delta_em,
            "ci_low": em_lo,
            "ci_high": em_hi,
        },
        "cross_family_diagnostics": score_cross_family_diagnostics(
            data, opp_labels, fit_auc_cv=fit_auc, cv_folds=folds_effective
        ),
        "cv_folds_requested": cv_folds if not use_holdout else None,
        "cv_folds_effective": folds_effective if not use_holdout else None,
        "dev_cv_warning": (
            "stratified_kfold_cv on merged table — dev/smoke only; not for paper"
            if not use_holdout else None
        ),
        "bootstrap": {"count": boot_count, "ci_level": 0.95, "seed": boot_seed},
    }


def run_calibration_stability(
    frame: pd.DataFrame,
    *,
    test_ids: set[str],
    calib_size: int = CALIB_SIZE,
    n_draws: int = STABILITY_CALIB_DRAWS,
    base_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Redraw CALIB, fit joint logistic, evaluate fixed TEST; report mean ± std."""
    LogisticRegression, roc_auc_score, Pipeline, StandardScaler, _ = load_sklearn_deps()
    data = filter_complete_rows(frame)
    if "query_id" not in data.columns:
        raise ValueError("merged CSV must include query_id")

    from routing.splits import draw_calib_ids

    pool = data["query_id"].astype(str).tolist()
    test_mask = data["query_id"].astype(str).isin(test_ids)
    if not test_mask.any():
        raise ValueError("fixed TEST ids not found in merged CSV")

    joint_cols = [COL_COMPLEXITY, COL_ENTROPY_WEAK, COL_MARGIN_WEAK]
    draws: list[dict[str, Any]] = []

    for draw in range(n_draws):
        calib_set = draw_calib_ids(
            pool, calib_size=calib_size, seed=base_seed + draw * 97, exclude=test_ids,
        )
        calib_mask = data["query_id"].astype(str).isin(calib_set).to_numpy()
        test_idx = np.where(test_mask.to_numpy())[0]
        calib_idx = np.where(calib_mask)[0]
        X = build_feature_matrix(data, joint_cols)
        y = data[COL_OPPORTUNITY].to_numpy(dtype=float)
        auroc = fit_logistic_auc_holdout(
            X[calib_idx], y[calib_idx], X[test_idx], y[test_idx],
            LogisticRegression=LogisticRegression,
            Pipeline=Pipeline,
            StandardScaler=StandardScaler,
            roc_auc_score=roc_auc_score,
        )
        coefs = fit_logistic_coefficients(
            X[calib_idx], y[calib_idx], joint_cols,
            LogisticRegression=LogisticRegression,
            Pipeline=Pipeline,
            StandardScaler=StandardScaler,
        )
        draws.append({
            "draw": draw,
            "seed": base_seed + draw * 97,
            "n_calib": int(len(calib_idx)),
            "n_test": int(len(test_idx)),
            "auroc": auroc,
            "coefficients": coefs,
        })

    aurocs = [d["auroc"] for d in draws if d["auroc"] is not None]
    coef_summary: dict[str, dict[str, float | None]] = {}
    for col in joint_cols:
        vals = [d["coefficients"][col] for d in draws if d.get("coefficients")]
        coef_summary[col] = {
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }

    return {
        "analysis": "calibration_stability",
        "model": "complexity_joint",
        "columns": joint_cols,
        "n_draws": n_draws,
        "calib_size": calib_size,
        "n_test_fixed": int(test_mask.sum()),
        "draws": draws,
        "summary": {
            "auroc_mean": float(np.mean(aurocs)) if aurocs else None,
            "auroc_std": float(np.std(aurocs, ddof=1)) if len(aurocs) > 1 else 0.0,
            "coefficients": coef_summary,
        },
    }


def summarize_c2(oracle_path: Path, *, scientific_question: str | None = None) -> dict:
    data = json.loads(oracle_path.read_text())
    rows = data.get("rows", [])
    complete = [r for r in rows if "bucket" in r]
    n = len(complete)
    counts = {"easy": 0, "opportunity": 0, "weak_only": 0, "too_hard": 0}
    for r in complete:
        counts[r["bucket"]] += 1

    rates = {k: (v / n if n else 0.0) for k, v in counts.items()}
    opp = rates["opportunity"]
    easy = rates["easy"]
    too_hard = rates["too_hard"]

    if opp < 0.10:
        failure_mode = "too_easy" if easy >= 0.80 else "low_opportunity"
    elif too_hard >= 0.80:
        failure_mode = "too_hard"
    elif easy >= 0.95:
        failure_mode = "too_easy"
    else:
        failure_mode = None

    studiable = opp >= 0.10 and easy < 0.95 and too_hard < 0.80

    return {
        "claim": "C2",
        "derived_from": str(oracle_path),
        "scientific_question": scientific_question,
        "config": data.get("config"),
        "weak_model": data.get("weak_model"),
        "strong_model": data.get("strong_model"),
        "dataset": data.get("dataset"),
        "n": n,
        "seed": data.get("seed"),
        "max_new_tokens": data.get("max_new_tokens"),
        "outcomes": {
            "easy_weak_yes_strong_yes": counts["easy"],
            "opportunity_weak_no_strong_yes": counts["opportunity"],
            "weak_only_weak_yes_strong_no": counts["weak_only"],
            "too_hard_weak_no_strong_no": counts["too_hard"],
        },
        "rates": rates,
        "opportunity_query_ids": [r["id"] for r in complete if r["bucket"] == "opportunity"],
        "gate_pass_criterion": "opportunity_rate >= 0.10; not ~95% easy; not ~80% too_hard",
        "gate_pass": studiable,
        "studiable_on_sample": studiable,
        "failure_mode": failure_mode,
        "recommendation": (
            "Proceed to n=200 generalization"
            if studiable
            else "Drop or defer — report as C2 finding"
        ),
    }
