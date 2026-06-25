"""DeLong test for comparing correlated ROC AUCs (DeLong et al. 1988)."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _structural_components(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == 0]
    m, n = len(pos_scores), len(neg_scores)
    if m == 0 or n == 0:
        raise ValueError("DeLong requires both classes in y_true")
    v10 = np.array([
        np.mean(sp > neg_scores) + 0.5 * np.mean(sp == neg_scores)
        for sp in pos_scores
    ])
    v01 = np.array([
        np.mean(pos_scores > sn) + 0.5 * np.mean(pos_scores == sn)
        for sn in neg_scores
    ])
    return v10, v01


def compare_auc_delong(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> dict[str, float | None]:
    """Two-sided DeLong test: H0 AUROC(a) == AUROC(b) on paired TEST scores."""
    y_true = np.asarray(y_true, dtype=int)
    sa = np.asarray(scores_a, dtype=float)
    sb = np.asarray(scores_b, dtype=float)
    v10_a, v01_a = _structural_components(y_true, sa)
    v10_b, v01_b = _structural_components(y_true, sb)
    m, n = len(v10_a), len(v01_a)

    auc_a = float(np.mean(v10_a))
    auc_b = float(np.mean(v10_b))
    delta = auc_a - auc_b

    var_a = np.var(v10_a, ddof=1) / m + np.var(v01_a, ddof=1) / n
    var_b = np.var(v10_b, ddof=1) / m + np.var(v01_b, ddof=1) / n
    cov_v10 = np.cov(v10_a, v10_b, ddof=1)[0, 1] / m if m > 1 else 0.0
    cov_v01 = np.cov(v01_a, v01_b, ddof=1)[0, 1] / n if n > 1 else 0.0
    cov_ab = cov_v10 + cov_v01

    var_delta = var_a + var_b - 2 * cov_ab
    if var_delta <= 0:
        return {
            "auc_a": auc_a,
            "auc_b": auc_b,
            "delta_auroc": delta,
            "z_statistic": None,
            "p_value": None,
        }
    z = delta / np.sqrt(var_delta)
    p = float(2 * norm.sf(abs(z)))
    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "delta_auroc": delta,
        "z_statistic": float(z),
        "p_value": p,
    }
