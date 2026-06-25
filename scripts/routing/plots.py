"""Paper figures from merged analysis tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from routing.constants import (
    BUCKET_LABELS,
    BUCKET_ORDER,
    COL_COMPLEXITY,
    COL_ENTROPY_WEAK,
    COL_MARGIN_WEAK,
    DISTRIBUTION_PANELS,
    ROC_CURVE_SIGNALS,
)

SCATTER_COLORS = {
    "easy": "#4c78a8",
    "opportunity": "#f58518",
    "weak_only": "#54a24b",
    "too_hard": "#e45756",
}

ABLATION_LABELS = {
    "complexity_only": r"$c(q)$",
    "complexity_entropy": r"$c+H$",
    "complexity_margin": r"$c+m$",
    "complexity_joint": r"$c+H+m$",
}


def plot_distributions(merged_csv: Path, output: Path, *, dpi: int = 150) -> None:
    import seaborn as sns

    df = pd.read_csv(merged_csv)
    if "bucket" not in df.columns:
        raise SystemExit(f"missing bucket column in {merged_csv}")

    df = df.copy()
    df["bucket_label"] = pd.Categorical(
        df["bucket"].map(BUCKET_LABELS),
        categories=[BUCKET_LABELS[b] for b in BUCKET_ORDER],
        ordered=True,
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for ax, (col, title) in zip(axes.ravel(), DISTRIBUTION_PANELS):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        sns.violinplot(
            data=df,
            x="bucket_label",
            y=col,
            order=[BUCKET_LABELS[b] for b in BUCKET_ORDER],
            inner="box",
            cut=0,
            linewidth=0.8,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(col.replace("_", " "))
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Prefill probe distributions by oracle bucket", y=1.02, fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(merged_csv: Path, output: Path, *, dpi: int = 150) -> None:
    try:
        from sklearn.metrics import auc, roc_curve
    except ImportError as exc:
        raise SystemExit("scikit-learn required. Install: uv sync --extra analysis") from exc

    df = pd.read_csv(merged_csv)
    if "y_opp" not in df.columns:
        raise SystemExit("merged CSV must include y_opp column")

    y = df["y_opp"].to_numpy()
    if len(set(y)) < 2:
        raise SystemExit("y_opp must contain both classes for ROC plot")

    fig, ax = plt.subplots(figsize=(6, 5))
    for col, label in ROC_CURVE_SIGNALS:
        if col not in df.columns:
            continue
        x = df[col].to_numpy(dtype=float)
        mask = ~pd.isna(x)
        if mask.sum() < 5:
            continue
        fpr, tpr, _ = roc_curve(y[mask], x[mask])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC: ranking routing opportunity")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_ladder(complementarity_json: Path, output: Path, *, dpi: int = 150) -> None:
    """Study III centerpiece: AUROC at four ablation rungs."""
    import json

    payload = json.loads(complementarity_json.read_text())
    models = payload.get("ablation_ladder", {}).get("models")
    if not models:
        raise SystemExit(f"no ablation_ladder in {complementarity_json}")

    labels = [ABLATION_LABELS.get(m["label"], m["label"]) for m in models]
    aurocs = [m.get("auroc") for m in models]
    if any(a is None for a in aurocs):
        raise SystemExit("ablation ladder has missing AUROC values")

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    ax.plot(x, aurocs, "o-", color="#4c78a8", linewidth=2, markersize=8)
    for xi, yi, m in zip(x, aurocs, models):
        lo, hi = m.get("auroc_ci_low"), m.get("auroc_ci_high")
        if lo is not None and hi is not None:
            ax.errorbar(xi, yi, yerr=[[yi - lo], [hi - yi]], fmt="none", color="#4c78a8", capsize=4)
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUROC (routing opportunity)")
    ax.set_ylim(max(0.4, min(aurocs) - 0.05), min(1.0, max(aurocs) + 0.08))
    ax.set_title("Cross-family ablation ladder (Study III)")
    ax.grid(axis="y", alpha=0.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_family_map(merged_csv: Path, output: Path, *, dpi: int = 150) -> None:
    """Discussion figure: c(q) vs H_w colored by oracle bucket."""
    df = pd.read_csv(merged_csv)
    for col in (COL_COMPLEXITY, COL_ENTROPY_WEAK, "bucket"):
        if col not in df.columns:
            raise SystemExit(f"merged CSV missing {col}")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for bucket in BUCKET_ORDER:
        sub = df[df["bucket"] == bucket]
        if sub.empty:
            continue
        ax.scatter(
            sub[COL_COMPLEXITY],
            sub[COL_ENTROPY_WEAK],
            label=BUCKET_LABELS[bucket],
            alpha=0.72,
            s=40,
            c=SCATTER_COLORS[bucket],
            edgecolors="white",
            linewidths=0.4,
        )
    ax.set_xlabel(r"Model-independent complexity $c(q)$")
    ax.set_ylabel(r"Weak entropy $H_w$")
    ax.set_title("Signal families in query space (oracle buckets)")
    ax.legend(loc="best", fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(complementarity_json: Path, output: Path, *, dpi: int = 150) -> None:
    """Calibration curve: predicted vs observed opportunity (joint model, TEST)."""
    import json

    payload = json.loads(complementarity_json.read_text())
    cal = payload.get("calibration_curve", {})
    mean_pred = cal.get("mean_predicted")
    frac_pos = cal.get("fraction_positive")
    if not mean_pred or not frac_pos:
        raise SystemExit(f"no calibration_curve data in {complementarity_json}")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color="#4c78a8", linewidth=2, markersize=7, label="c+H+m joint")
    ax.set_xlabel("Mean predicted P(opportunity)")
    ax.set_ylabel("Observed opportunity rate")
    ax.set_title("Calibration on TEST (joint logistic)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_entropy_margin_scatter(merged_csv: Path, output: Path, *, dpi: int = 150) -> None:
    df = pd.read_csv(merged_csv)
    fig, ax = plt.subplots(figsize=(6, 5))
    for bucket in BUCKET_ORDER:
        sub = df[df["bucket"] == bucket]
        if sub.empty:
            continue
        ax.scatter(
            sub[COL_ENTROPY_WEAK],
            sub[COL_MARGIN_WEAK],
            label=BUCKET_LABELS[bucket],
            alpha=0.75,
            s=36,
            c=SCATTER_COLORS[bucket],
            edgecolors="white",
            linewidths=0.4,
        )
    ax.set_xlabel(r"Weak entropy $H_w$")
    ax.set_ylabel(r"Weak margin $m_w$")
    ax.set_title("Weak prefill probes by oracle bucket")
    ax.legend(loc="best", fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
