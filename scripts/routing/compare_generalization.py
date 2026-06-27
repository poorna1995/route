"""Cross-regime dimension transfer analysis (RH7).

Compares qualitative signal patterns across datasets — escalation vs uncertainty
separation — not AUROC leaderboards.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from routing.constants import (
    BOOTSTRAP_COUNT,
    BOOTSTRAP_SEED,
    COL_COMPLEXITY,
    COL_ENTROPY_WEAK,
    PROBE_DERIVED_ENTROPY,
    PROBE_DERIVED_MARGIN,
)
from routing.evaluation import (
    bootstrap_ci,
    correlation_metrics,
    effect_size_between_buckets,
    stat_auroc,
    stat_spearman,
)

# Latent dimension → operationalization (frozen vocabulary).
DIMENSION_SIGNALS: dict[str, str] = {
    "task_difficulty": COL_COMPLEXITY,
    "model_uncertainty": COL_ENTROPY_WEAK,
    "model_disagreement": PROBE_DERIVED_ENTROPY,
    "escalation_potential": PROBE_DERIVED_MARGIN,
}

SIDE_DIFFICULTY = frozenset({"task_difficulty", "model_uncertainty"})
SIDE_ESCALATION = frozenset({"model_disagreement", "escalation_potential"})

_MMLU_SUBJECT_RE = re.compile(r"^mmlu_(.+?)_test_\d+$")


def mmlu_subject_from_query_id(query_id: str) -> str | None:
    m = _MMLU_SUBJECT_RE.match(str(query_id))
    return m.group(1) if m else None


def _require_deps():
    from scipy.stats import kendalltau, mannwhitneyu, pearsonr, spearmanr
    from sklearn.metrics import average_precision_score, roc_auc_score

    return spearmanr, kendalltau, pearsonr, roc_auc_score, mannwhitneyu, average_precision_score


def analyze_regime(
    merged: pd.DataFrame,
    *,
    regime: str,
    n_boot: int = BOOTSTRAP_COUNT,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Per-regime dimension statistics for opportunity vs too-hard contrast."""
    spearmanr, kendalltau, pearsonr, roc_auc_score, mannwhitneyu, average_precision_score = (
        _require_deps()
    )
    boot_kw = {"n_boot": n_boot, "bootstrap_seed": bootstrap_seed}

    frame = merged.copy()
    y_opp = frame["y_opp"].to_numpy(dtype=float)
    buckets = frame["bucket"].value_counts().to_dict()
    rates = {k: v / len(frame) for k, v in buckets.items()}

    dimensions: dict[str, Any] = {}
    for dim, col in DIMENSION_SIGNALS.items():
        if col not in frame.columns:
            continue
        x = frame[col].to_numpy(dtype=float)
        opp_hard = effect_size_between_buckets(
            frame, col, "opportunity", "too_hard", mannwhitneyu,
        )
        corr = correlation_metrics(
            y_opp,
            x,
            spearmanr=spearmanr,
            kendalltau=kendalltau,
            pearsonr=pearsonr,
            roc_auc_score=roc_auc_score,
            average_precision_score=average_precision_score,
            **boot_kw,
        )
        dimensions[dim] = {
            "signal": col,
            "cohens_d_opp_vs_too_hard": opp_hard.get("cohens_d"),
            "cliffs_delta_opp_vs_too_hard": opp_hard.get("cliffs_delta"),
            "n_opportunity": opp_hard.get("n_opportunity"),
            "n_too_hard": opp_hard.get("n_too_hard"),
            "spearman_rho_vs_opportunity": corr.get("spearman_rho"),
            "spearman_ci": [corr.get("spearman_ci_low"), corr.get("spearman_ci_high")],
            "auroc_vs_opportunity": corr.get("auroc"),
            "auroc_ci": [corr.get("auroc_ci_low"), corr.get("auroc_ci_high")],
        }

    return {
        "regime": regime,
        "n": int(len(frame)),
        "buckets": buckets,
        "rates": rates,
        "dimensions": dimensions,
        "pattern": classify_regime_pattern(dimensions),
    }


def classify_regime_pattern(dimensions: dict[str, Any]) -> dict[str, Any]:
    """Qualitative RH7 pattern tags (difficulty-side weak, escalation-side strong)."""
    d = {k: abs(v.get("cohens_d_opp_vs_too_hard") or 0.0) for k, v in dimensions.items()}
    auroc = {k: v.get("auroc_vs_opportunity") for k, v in dimensions.items()}

    esc_d = max(d.get("model_disagreement", 0.0), d.get("escalation_potential", 0.0))
    diff_d = max(d.get("task_difficulty", 0.0), d.get("model_uncertainty", 0.0))
    unc_d = d.get("model_uncertainty", 0.0)

    escalation_separates = esc_d >= 0.5 and esc_d > unc_d
    uncertainty_weak = unc_d < 0.2
    difficulty_escalation_gap = esc_d - diff_d

    if escalation_separates and uncertainty_weak:
        verdict = "matches_arc_template"
    elif escalation_separates:
        verdict = "escalation_holds_uncertainty_mixed"
    elif esc_d >= 0.35:
        verdict = "partial_escalation_signal"
    else:
        verdict = "pattern_break"

    return {
        "escalation_cohens_d_max": esc_d,
        "difficulty_side_cohens_d_max": diff_d,
        "uncertainty_cohens_d": unc_d,
        "escalation_separates": escalation_separates,
        "uncertainty_weak": uncertainty_weak,
        "difficulty_escalation_gap": difficulty_escalation_gap,
        "verdict": verdict,
        "auroc_escalation": auroc.get("escalation_potential"),
        "auroc_uncertainty": auroc.get("model_uncertainty"),
    }


def cross_regime_pattern_match(regimes: list[dict[str, Any]]) -> dict[str, Any]:
    """Check whether escalation/uncertainty qualitative story holds everywhere."""
    verdicts = [r["pattern"]["verdict"] for r in regimes]
    esc_holds = all(r["pattern"]["escalation_separates"] for r in regimes)
    unc_weak_all = all(r["pattern"]["uncertainty_weak"] for r in regimes)

    if esc_holds and unc_weak_all:
        summary = (
            "Escalation potential separates routable from irrecoverable queries on all regimes; "
            "weak-model uncertainty does not — despite different absolute accuracies."
        )
        match = "full"
    elif esc_holds:
        summary = (
            "Escalation-side dimensions separate opportunity from too-hard on all regimes; "
            "uncertainty separation varies by domain."
        )
        match = "escalation_invariant"
    else:
        summary = "Escalation separation does not hold uniformly — inspect per-regime table."
        match = "partial"

    return {
        "n_regimes": len(regimes),
        "escalation_invariant": esc_holds,
        "uncertainty_weak_all": unc_weak_all,
        "verdicts": verdicts,
        "match": match,
        "summary": summary,
    }


def build_transfer_table(regimes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flat rows for paper table: regime × dimension."""
    rows: list[dict[str, Any]] = []
    for reg in regimes:
        for dim, stats in reg["dimensions"].items():
            side = "difficulty" if dim in SIDE_DIFFICULTY else "escalation"
            rows.append(
                {
                    "regime": reg["regime"],
                    "dimension": dim,
                    "side": side,
                    "signal": stats["signal"],
                    "cohens_d": stats["cohens_d_opp_vs_too_hard"],
                    "spearman_rho": stats["spearman_rho_vs_opportunity"],
                    "auroc": stats["auroc_vs_opportunity"],
                    "n": reg["n"],
                    "opportunity_rate": reg["rates"].get("opportunity"),
                }
            )
    return rows


def compare_generalization(
    *,
    regimes: list[tuple[str, Path]],
    output: Path | None = None,
    mmlu_subject_splits: bool = True,
    n_boot: int = BOOTSTRAP_COUNT,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Run RH7 dimension transfer across named merged CSV regimes."""
    analyzed: list[dict[str, Any]] = []
    for name, path in regimes:
        merged = pd.read_csv(path)
        analyzed.append(
            analyze_regime(
                merged,
                regime=name,
                n_boot=n_boot,
                bootstrap_seed=bootstrap_seed,
            )
        )

    subject_regimes: list[dict[str, Any]] = []
    if mmlu_subject_splits:
        mmlu_paths = [p for n, p in regimes if "mmlu" in n.lower()]
        if len(mmlu_paths) == 1:
            merged = pd.read_csv(mmlu_paths[0])
            merged = merged.copy()
            merged["_subject"] = merged["query_id"].map(mmlu_subject_from_query_id)
            for subject, sub in merged.groupby("_subject", sort=True):
                if subject is None or len(sub) < 20:
                    continue
                subject_regimes.append(
                    analyze_regime(
                        sub.drop(columns=["_subject"]),
                        regime=f"mmlu/{subject}",
                        n_boot=n_boot,
                        bootstrap_seed=bootstrap_seed + hash(subject) % 10_000,
                    )
                )

    all_regimes = analyzed + subject_regimes
    payload: dict[str, Any] = {
        "analysis": "dimension_transfer",
        "hypothesis": "RH7",
        "question": "Do latent routing dimensions generalize across domains (pattern invariants)?",
        "dimension_map": DIMENSION_SIGNALS,
        "comparison": "opportunity_vs_too_hard",
        "regimes": all_regimes,
        "transfer_table": build_transfer_table(all_regimes),
        "pattern_match": cross_regime_pattern_match(analyzed),
    }
    if subject_regimes:
        payload["mmlu_subject_pattern_match"] = cross_regime_pattern_match(subject_regimes)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2))

    return payload


def format_summary(payload: dict[str, Any]) -> str:
    """Human-readable terminal summary."""
    lines = [
        f"RH7 dimension transfer — {payload['pattern_match']['summary']}",
        "",
        f"{'Regime':<28} {'Dim':<22} {'d(opp,hard)':>12} {'ρ_opp':>8} {'AUROC':>7}",
        "-" * 82,
    ]
    for row in payload["transfer_table"]:
        if "/" in row["regime"]:
            continue
        d = row["cohens_d"]
        rho = row["spearman_rho"]
        auroc = row["auroc"]
        lines.append(
            f"{row['regime']:<28} {row['dimension']:<22} "
            f"{d if d is not None else float('nan'):>12.3f} "
            f"{rho if rho is not None else float('nan'):>+8.3f} "
            f"{auroc if auroc is not None else float('nan'):>7.3f}"
        )

    lines.append("")
    lines.append("Per-regime pattern verdicts:")
    for reg in payload["regimes"]:
        if reg["regime"].startswith("mmlu/"):
            continue
        p = reg["pattern"]
        lines.append(
            f"  {reg['regime']}: {p['verdict']} "
            f"(esc_d={p['escalation_cohens_d_max']:.2f}, unc_d={p['uncertainty_cohens_d']:.2f}, "
            f"opp={reg['rates'].get('opportunity', 0):.1%})"
        )

    if payload.get("mmlu_subject_pattern_match"):
        lines.append("")
        lines.append("MMLU subjects:")
        for reg in payload["regimes"]:
            if not reg["regime"].startswith("mmlu/"):
                continue
            p = reg["pattern"]
            esc = reg["dimensions"].get("escalation_potential", {})
            unc = reg["dimensions"].get("model_uncertainty", {})
            lines.append(
                f"  {reg['regime']}: d_Δm={esc.get('cohens_d_opp_vs_too_hard', 0):.2f} "
                f"d_Hw={unc.get('cohens_d_opp_vs_too_hard', 0):.2f} "
                f"n={reg['n']} opp={reg['rates'].get('opportunity', 0):.1%}"
            )

    return "\n".join(lines)
