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
    PROBE_DERIVED_MARGIN,
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


def plot_information_decomposition(complementarity_json: Path, output: Path, *, dpi: int = 150) -> None:
    """Flagship figure: solo-family AUROC + incremental cross-family ladder."""
    import json

    payload = json.loads(complementarity_json.read_text())
    ladder = payload.get("ablation_ladder", {}).get("models") or []
    solo = payload.get("solo_reference", {}).get("models") or []
    increments = payload.get("auc_increments", {}).get("steps") or {}

    if not ladder:
        raise SystemExit(f"no ablation_ladder in {complementarity_json}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

    # Left: solo families
    solo_labels = {
        "complexity_only": r"$c(q)$ solo",
        "entropy_solo": r"$H_w$ solo",
        "margin_solo": r"$m_w$ solo",
    }
    solo_aurocs: list[tuple[str, float]] = []
    c_solo = next((m for m in ladder if m["label"] == "complexity_only"), None)
    if c_solo and c_solo.get("auroc") is not None:
        solo_aurocs.append((solo_labels["complexity_only"], float(c_solo["auroc"])))
    for m in solo:
        key = m["label"]
        if m.get("auroc") is not None and key in solo_labels:
            solo_aurocs.append((solo_labels[key], float(m["auroc"])))

    ax = axes[0]
    if solo_aurocs:
        labels, vals = zip(*solo_aurocs)
        x = np.arange(len(labels))
        ax.bar(x, vals, color=["#4c78a8", "#f58518", "#54a24b"][: len(vals)], width=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        for xi, yi in zip(x, vals):
            ax.text(xi, yi + 0.008, f"{yi:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("AUROC (routing opportunity)")
    ax.set_ylim(0.45, 0.65)
    ax.set_title("Solo signal families")
    ax.grid(axis="y", alpha=0.3)

    # Right: incremental ladder with Δ annotations
    ax = axes[1]
    labels = [ABLATION_LABELS.get(m["label"], m["label"]) for m in ladder]
    aurocs = [float(m["auroc"]) for m in ladder]
    x = np.arange(len(labels))
    ax.plot(x, aurocs, "o-", color="#4c78a8", linewidth=2, markersize=8)
    for xi, yi in zip(x, aurocs):
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    inc_h = increments.get("entropy_beyond_complexity", {})
    if inc_h.get("delta_auroc") is not None:
        d = float(inc_h["delta_auroc"])
        ax.annotate(
            f"+Δ={d:+.3f}",
            xy=(1, aurocs[1]),
            xytext=(0.5, aurocs[1] + 0.04),
            arrowprops=dict(arrowstyle="->", color="#f58518", lw=1.2),
            fontsize=8,
            color="#f58518",
        )
    inc_m = increments.get("margin_beyond_joint", {})
    if inc_m.get("delta_auroc") is not None and len(aurocs) >= 4:
        d = float(inc_m["delta_auroc"])
        ax.annotate(
            f"Δ={d:+.3f}",
            xy=(3, aurocs[3]),
            xytext=(2.5, aurocs[3] - 0.05),
            arrowprops=dict(arrowstyle="->", color="#54a24b", lw=1.2),
            fontsize=8,
            color="#54a24b",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUROC (routing opportunity)")
    ax.set_ylim(max(0.4, min(aurocs) - 0.05), min(1.0, max(aurocs) + 0.1))
    ax.set_title("Incremental cross-family composition")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Routing information decomposition (Study III)", y=1.02, fontsize=12)
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


def plot_signal_calibration(merged_csv: Path, output: Path, *, dpi: int = 150) -> None:
    """Opportunity rate by entropy quantile band (bottom / middle / top 20%)."""
    from routing.evaluation import signal_quantile_calibration

    df = pd.read_csv(merged_csv)
    cal = signal_quantile_calibration(df, COL_ENTROPY_WEAK)
    bins = cal["bins"]
    labels = ["Bottom 20%", "Middle 20%", "Top 20%"]
    keys = ["bottom", "middle", "top"]
    rates = [bins[k]["opportunity_rate"] for k in keys]
    ci_lo = [bins[k]["opportunity_rate_ci_low"] for k in keys]
    ci_hi = [bins[k]["opportunity_rate_ci_high"] for k in keys]
    baseline = cal["baseline_opportunity_rate"]

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    x = np.arange(len(labels))
    colors = ["#4c78a8", "#bab0ac", "#f58518"]
    bars = ax.bar(x, rates, color=colors, width=0.55, zorder=2)
    yerr = [
        [r - lo for r, lo in zip(rates, ci_lo)],
        [hi - r for r, hi in zip(rates, ci_hi)],
    ]
    ax.errorbar(x, rates, yerr=yerr, fmt="none", color="#333", capsize=4, zorder=3)
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{rate:.0%}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    ax.axhline(baseline, color="#666", linestyle="--", linewidth=0.9, label=f"Overall {baseline:.0%}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Observed opportunity rate")
    ax.set_ylim(0, min(1.0, max(rates) + 0.15))
    ax.set_title(r"Calibration: weak entropy $H_w$ vs routing opportunity")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_probability_calibration(
    merged_full_csv: Path,
    splits_json: Path,
    output: Path,
    *,
    dpi: int = 150,
) -> None:
    """10-bin reliability: logistic predicted P(opp) vs observed rate (CALIB fit, TEST eval)."""
    from routing.evaluation import run_logistic_probability_calibration
    from routing.splits import load_splits

    splits = load_splits(splits_json)
    df = pd.read_csv(merged_full_csv)
    cal = run_logistic_probability_calibration(
        df,
        test_ids=splits["test"],
        feature_cols=[COL_COMPLEXITY, COL_ENTROPY_WEAK, COL_MARGIN_WEAK],
        model_label="complexity_joint",
    )
    bins = cal.get("bins") or []
    if not bins:
        raise SystemExit("probability calibration produced no bins")

    x = np.arange(len(bins))
    pred = [b["mean_predicted"] for b in bins]
    obs = [b["observed_rate"] for b in bins]
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, pred, width, label="Predicted", color="#4c78a8", zorder=2)
    ax.bar(x + width / 2, obs, width, label="Observed", color="#f58518", zorder=2)
    ax.plot([x[0] - 0.5, x[-1] + 0.5], [0, 0], color="white", alpha=0)
    verdict = cal.get("calibration_verdict", "")
    gap = cal.get("mean_signed_gap")
    title = r"Logistic calibration: $c+H+m$ joint model (TEST)"
    if verdict and gap is not None:
        title += f"\nmean gap {gap:+.3f} → {verdict.replace('_', ' ')}"
    ax.set_title(title, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b["bin"]) for b in bins])
    ax.set_xlabel("Bin (low → high predicted probability)")
    ax.set_ylabel("Routing opportunity rate")
    ax.set_ylim(0, min(1.0, max(max(pred), max(obs)) + 0.12))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
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


def _bucket_axis_positions(
    df: pd.DataFrame,
    signal_col: str,
    buckets: tuple[str, ...],
    *,
    margin: float = 0.1,
) -> dict[str, float]:
    means = {
        b: float(df.loc[df["bucket"] == b, signal_col].mean())
        for b in buckets
        if (df["bucket"] == b).any()
    }
    lo, hi = min(means.values()), max(means.values())
    span = hi - lo
    if span < 1e-9:
        return {b: 0.5 for b in means}
    return {b: margin + (1.0 - 2.0 * margin) * (means[b] - lo) / span for b in means}


def _vertical_kde(
    ax: plt.Axes,
    x_center: float,
    values: np.ndarray,
    *,
    width: float,
    color: str,
    label: str,
) -> None:
    from scipy.stats import gaussian_kde

    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return
    kde = gaussian_kde(vals)
    pad = max(0.05 * (vals.max() - vals.min()), 0.02)
    y_grid = np.linspace(vals.min() - pad, vals.max() + pad, 200)
    dens = kde(y_grid)
    dens = dens / dens.max() * width
    ax.fill_betweenx(y_grid, x_center - dens, x_center + dens, color=color, alpha=0.35, linewidth=0)
    ax.plot(x_center - dens, y_grid, color=color, lw=1.2, label=label)
    ax.plot(x_center + dens, y_grid, color=color, lw=1.2)
    ax.scatter(
        [x_center],
        [float(np.median(vals))],
        color=color,
        s=32,
        zorder=5,
        edgecolors="white",
        linewidths=0.5,
    )


def _draw_concept_axis(ax: plt.Axes, left: str, right: str) -> None:
    ax.annotate(
        "",
        xy=(0.97, 0.04),
        xytext=(0.03, 0.04),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="#666666", lw=1.8, mutation_scale=12),
    )
    ax.text(0.03, 0.10, left, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, color="#333333")
    ax.text(0.97, 0.10, right, transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color="#333333")


def plot_signal_semantics(merged_csv: Path, output: Path, *, dpi: int = 150) -> None:
    """Difficulty vs recoverability: entropy on Easy–Hard, Δmargin on No rescue–Rescue."""
    df = pd.read_csv(merged_csv)
    if "bucket" not in df.columns:
        raise SystemExit(f"missing bucket column in {merged_csv}")
    for col in (COL_ENTROPY_WEAK, PROBE_DERIVED_MARGIN):
        if col not in df.columns:
            raise SystemExit(f"missing {col} in {merged_csv}")

    panels = (
        (COL_ENTROPY_WEAK, r"Weak entropy $H_w$", "Difficulty", "Easy", "Hard"),
        (PROBE_DERIVED_MARGIN, r"$\Delta m_{\mathrm{gain}}$ ($m_s - m_w$)", "Recoverability", "No rescue", "Rescue"),
    )

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    kde_width = 0.11
    legend_handles: list = []
    legend_labels: list[str] = []

    for ax, (signal_col, signal_label, concept, left, right) in zip(axes, panels):
        positions = _bucket_axis_positions(df, signal_col, BUCKET_ORDER)
        vals = df[signal_col].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        y_lo = float(np.percentile(vals, 1))
        y_hi = float(np.percentile(vals, 99))
        pad = 0.08 * (y_hi - y_lo) if y_hi > y_lo else 0.05
        ax.set_ylim(y_lo - pad, y_hi + pad)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_ylabel(signal_label, fontsize=10)
        ax.set_title(f"{concept}: {left} $\\rightarrow$ {right}", fontsize=11, loc="left", pad=18)
        _draw_concept_axis(ax, left, right)

        for bucket in BUCKET_ORDER:
            sub = df.loc[df["bucket"] == bucket, signal_col].to_numpy(dtype=float)
            if len(sub) == 0:
                continue
            color = SCATTER_COLORS[bucket]
            label = BUCKET_LABELS[bucket]
            _vertical_kde(
                ax,
                positions[bucket],
                sub,
                width=kde_width,
                color=color,
                label=label,
            )
            if ax is axes[0]:
                legend_handles.append(
                    plt.Line2D([0], [0], color=color, lw=2, marker="o", markersize=5, markerfacecolor=color)
                )
                legend_labels.append(label)

        if signal_col == COL_ENTROPY_WEAK:
            opp_x = positions["opportunity"]
            hard_x = positions["too_hard"]
            if abs(opp_x - hard_x) < 0.12:
                mid = 0.5 * (opp_x + hard_x)
                y_note = y_hi + 0.35 * pad
                ax.annotate(
                    "overlap",
                    xy=(mid, y_note),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#666666",
                    style="italic",
                )
                ax.plot([opp_x, hard_x], [y_note - 0.15 * pad] * 2, color="#999999", lw=0.8)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=9,
    )
    fig.suptitle("What pre-inference signals measure", y=1.06, fontsize=12)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_recovery_matrix(merged_csv: Path, output: Path, *, dpi: int = 150) -> None:
    """2×2 heatmap: weak uncertainty × strong recoverability → opportunity rate."""
    from routing.evaluation import recovery_matrix

    df = pd.read_csv(merged_csv)
    mat = recovery_matrix(df)
    counts = np.array(mat["counts"], dtype=int)
    opp = np.array(mat["opportunity_rates"], dtype=float)
    row_titles = mat["row_titles"]
    col_titles = mat["col_titles"]

    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    im = ax.imshow(opp, cmap="YlOrRd", vmin=0.25, vmax=0.65, aspect="auto")
    ax.set_xticks(range(2))
    ax.set_yticks(range(2))
    ax.set_xticklabels(col_titles, fontsize=10)
    ax.set_yticklabels(row_titles, fontsize=10)
    ax.set_xlabel("Weak-model difficulty ($H_w$)", fontsize=10)
    ax.set_ylabel(r"Cross-model recoverability ($\Delta m_{\mathrm{gain}}$)", fontsize=10)
    ax.set_title("Recovery matrix: difficulty × recoverability", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Opportunity rate", fontsize=9)

    for ri in range(2):
        for ci in range(2):
            cell_key = f"{mat['row_labels'][ri]}__{mat['col_labels'][ci]}"
            cell = mat["cells"][cell_key]
            n = counts[ri, ci]
            rate = opp[ri, ci]
            dom = BUCKET_LABELS.get(cell["dominant_bucket"], "")
            text_color = "white" if rate > 0.52 else "black"
            ax.text(
                ci,
                ri,
                f"n={n}\n{rate:.0%} opp\n({dom})",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                linespacing=1.25,
            )

    h_thr = mat["thresholds"][COL_ENTROPY_WEAK]
    dm_thr = mat["thresholds"][PROBE_DERIVED_MARGIN]
    ax.text(
        0.5,
        -0.22,
        f"Median splits: $H_w$={h_thr:.2f}, $\\Delta m_{{\\mathrm{{gain}}}}$={dm_thr:.2f}",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
        color="#555555",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_conceptual_model(output: Path, *, dpi: int = 150) -> None:
    """Paper figure F0: routing problem → observations → signals → structure → selection."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.2, 9.2), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(cx: float, cy: float, w: float, h: float, title: str, subtitle: str = "", *, color: str) -> None:
        rect = FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=color,
            facecolor=color,
            alpha=0.14,
        )
        ax.add_patch(rect)
        ax.text(cx, cy + (0.012 if subtitle else 0), title, ha="center", va="center", fontsize=11, fontweight="bold", color="#222222")
        if subtitle:
            ax.text(cx, cy - 0.028, subtitle, ha="center", va="center", fontsize=9, color="#444444")

    def arrow(y0: float, y1: float) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (0.5, y0),
                (0.5, y1),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.6,
                color="#666666",
            )
        )

    ax.text(
        0.5,
        0.97,
        "Unsupervised pre-inference routing",
        ha="center",
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.925,
        "Latent dimension → Operationalization → Opportunity prediction",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    box(0.5, 0.87, 0.62, 0.08, "Routing problem", "Appropriate model selection", color="#9e9ac8")
    arrow(0.83, 0.795)
    box(0.5, 0.755, 0.66, 0.09, "Pre-inference signals", "Query text; prefill logits", color="#6baed6")
    arrow(0.71, 0.665)
    box(0.5, 0.62, 0.72, 0.10, "Latent routing dimensions", "Task difficulty · Model uncertainty · Disagreement · Escalation", color="#fdae6b")

    # Dimension branch
    box(0.15, 0.50, 0.20, 0.07, "Task difficulty", r"$c(q)$", color="#fdd0a2")
    box(0.38, 0.50, 0.20, 0.07, "Model uncert.", r"$H_w$", color="#fdd0a2")
    box(0.61, 0.50, 0.20, 0.07, "Disagreement", r"$\Delta H$", color="#fd8d3c")
    box(0.84, 0.50, 0.22, 0.07, "Escalation", r"$\Delta m_{\mathrm{gain}}$", color="#fd8d3c")
    for x in (0.15, 0.38, 0.61, 0.84):
        ax.add_patch(
            FancyArrowPatch((0.5, 0.575), (x, 0.535), arrowstyle="-|>", mutation_scale=11, linewidth=1.2, color="#888")
        )
    ax.text(0.5, 0.44, "Operationalizations (prefill statistics)", ha="center", fontsize=9, color="#666666")

    arrow(0.395, 0.36)
    box(0.5, 0.315, 0.58, 0.09, "Appropriate model selection", "Routing validation + oracle bound", color="#74c476")

    ax.text(
        0.5,
        0.19,
        "Evidence: latent dimensions predict opportunity and encode different routing need\n"
        "Validation: simple policies do not fully exploit available information",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f7f7f7", edgecolor="#cccccc"),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_layer_evolution(
    trace_path: Path,
    merged_csv: Path,
    output: Path,
    *,
    dpi: int = 150,
) -> None:
    from routing.formation_analysis import F7_BUCKETS, bucket_medians, load_traces, trace_depth_fraction

    traces = load_traces(trace_path)
    merged = pd.read_csv(merged_csv)
    medians = bucket_medians(traces, merged)
    xs = np.asarray(trace_depth_fraction(next(iter(traces.values()))), dtype=float)

    labels = {"easy": "Easy", "opportunity": "Opportunity", "too_hard": "Too hard"}
    colors = {"easy": SCATTER_COLORS["easy"], "opportunity": SCATTER_COLORS["opportunity"], "too_hard": SCATTER_COLORS["too_hard"]}

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for bucket in F7_BUCKETS:
        y = medians[bucket]
        ax.plot(xs, y, label=labels[bucket], color=colors[bucket], linewidth=2)

    ax.set_xlabel("Fraction of depth (ℓ / L)")
    ax.set_ylabel(r"Median margin $m_\ell$")
    ax.set_title("Layerwise Confidence Evolution Across Depth")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
