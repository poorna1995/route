"""Model-independent signals: query complexity c(q) and D46 screening (RH1)."""

from __future__ import annotations

import math
import zlib
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import pandas as pd

from routing.constants import (
    BOOTSTRAP_COUNT,
    BOOTSTRAP_SEED,
    CALIB_SIZE,
    FEATURE_CSV_FIELDS,
    FEATURE_FAMILIES,
    FEATURE_NAMES,
    INTERPRETABILITY_RANK,
    SCREEN_TAG,
    SCORE_OVERLAP,
)
from routing.evaluation import (
    _require_analysis_deps,
    bootstrap_ci,
    cliffs_delta,
    cohens_d,
    stat_auroc,
    stat_spearman,
    summarize_values,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class FeatureRow(TypedDict):
    piece_count: int
    mattr: float
    text_shannon: float
    text_shannon_norm: float
    compression_ratio: float
    unique_tokens: int


def tokenize_query(tokenizer: PreTrainedTokenizerBase, text: str) -> list[str]:
    """Subword pieces via encode → convert_ids_to_tokens (canonical HF path)."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.convert_ids_to_tokens(ids)


def compute_mattr(tokens: list[str], window: int = 15) -> float:
    """Moving-average type-token ratio in [0, 1]."""
    n = len(tokens)
    if n == 0:
        return 0.0
    w = min(window, n)
    freq: dict[str, int] = {}
    for tok in tokens[:w]:
        freq[tok] = freq.get(tok, 0) + 1
    ttr_sum, n_windows = len(freq) / w, 1
    for i in range(w, n):
        out = tokens[i - w]
        freq[out] -= 1
        if freq[out] == 0:
            del freq[out]
        inc = tokens[i]
        freq[inc] = freq.get(inc, 0) + 1
        ttr_sum += len(freq) / w
        n_windows += 1
    return min(ttr_sum / n_windows, 1.0)


def _shannon_from_counts(counts: Counter, n: int) -> float:
    """Lexical entropy over tokenizer pieces (nats) — NOT model prefill H(q, M)."""
    if n == 0:
        return 0.0
    return sum(-(c / n) * math.log(c / n) for c in counts.values())


def compute_shannon(tokens: list[str]) -> float:
    """Empirical unigram Shannon entropy (nats) over HF subword pieces."""
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    return _shannon_from_counts(counts, len(tokens))


def compute_shannon_norm(tokens: list[str]) -> float:
    """Shannon entropy normalized by log(type count)."""
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    n = len(tokens)
    h = _shannon_from_counts(counts, n)
    vocab = len(counts)
    if vocab <= 1:
        return 0.0
    return h / math.log(vocab)


def compute_compression(text: str) -> float:
    """zlib compressed size / UTF-8 bytes."""
    raw = text.encode("utf-8")
    if not raw:
        return 0.0
    return len(zlib.compress(raw, level=9)) / len(raw)


def compute_feature_row(
    user_content: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    mattr_window: int = 15,
) -> FeatureRow:
    pieces = tokenize_query(tokenizer, user_content)
    counts = Counter(pieces)
    return FeatureRow(
        piece_count=len(pieces),
        mattr=compute_mattr(pieces, window=mattr_window),
        text_shannon=_shannon_from_counts(counts, len(pieces)),
        text_shannon_norm=(
            0.0
            if len(counts) <= 1
            else _shannon_from_counts(counts, len(pieces)) / math.log(len(counts))
        ),
        compression_ratio=compute_compression(user_content),
        unique_tokens=len(counts),
    )


# --- D46 screening ---


def rank_score(values: list[float]) -> list[float]:
    """Average ranks in [1, n] — stable when candidates are added later."""
    from scipy.stats import rankdata

    if not values:
        return []
    return rankdata(values, method="average").tolist()


def match_feature_family(name: str) -> str:
    for key, spec in FEATURE_FAMILIES.items():
        if name in spec["features"]:
            return key
    return "unknown"


def score_feature(
    frame: pd.DataFrame,
    name: str,
    *,
    spearmanr,
    roc_auc_score,
    boot_count: int,
    boot_seed: int,
) -> dict[str, Any]:
    x = frame[name].astype(float).to_numpy()
    y = frame["y_opp"].astype(float).to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    opp_vals = x[y == 1]
    non_vals = x[y == 0]
    summary = summarize_values(x)
    summary_by_label = {
        "opportunity": summarize_values(opp_vals),
        "non_opportunity": summarize_values(non_vals),
    }

    if np.std(x) < 1e-12:
        return {
            "feature": name,
            "family": match_feature_family(name),
            "summary": summary,
            "summary_by_label": summary_by_label,
            "spearman_rho": 0.0,
            "spearman_p": 1.0,
            "spearman_abs_rho": 0.0,
            "spearman_ci_low": None,
            "spearman_ci_high": None,
            "auroc": 0.5,
            "auroc_ci_low": None,
            "auroc_ci_high": None,
            "cohens_d": cohens_d(opp_vals, non_vals),
            "cliffs_delta": cliffs_delta(opp_vals, non_vals),
            "composite_score": 0.0,
            "n": int(len(x)),
        }

    rho, p_rho = spearmanr(x, y)
    auroc = 0.5 if len(np.unique(y)) < 2 else float(roc_auc_score(y, x))
    rho_lo, rho_hi = bootstrap_ci(
        x, y,
        stat_fn=lambda a, b: stat_spearman(a, b, spearmanr=spearmanr),
        n_boot=boot_count,
        seed=boot_seed,
    )
    auc_lo, auc_hi = bootstrap_ci(
        x, y,
        stat_fn=lambda a, b: stat_auroc(a, b, roc_auc_score=roc_auc_score),
        n_boot=boot_count,
        seed=boot_seed + 1,
    )

    return {
        "feature": name,
        "family": match_feature_family(name),
        "summary": summary,
        "summary_by_label": summary_by_label,
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
        "spearman_abs_rho": float(abs(rho)),
        "spearman_ci_low": rho_lo,
        "spearman_ci_high": rho_hi,
        "auroc": auroc,
        "auroc_ci_low": auc_lo,
        "auroc_ci_high": auc_hi,
        "cohens_d": cohens_d(opp_vals, non_vals),
        "cliffs_delta": cliffs_delta(opp_vals, non_vals),
        "composite_score": None,
        "n": int(len(x)),
    }


def ci_ranges_overlap(a_lo, a_hi, b_lo, b_hi) -> bool:
    if None in (a_lo, a_hi, b_lo, b_hi):
        return True
    return not (a_hi < b_lo or b_hi < a_lo)


def pick_best_feature(
    scores: list[dict[str, Any]],
    *,
    overlap_margin: float = SCORE_OVERLAP,
) -> tuple[str, dict[str, Any]]:
    abs_rhos = [s["spearman_abs_rho"] for s in scores]
    aurocs = [s["auroc"] for s in scores]
    rank_rho = rank_score(abs_rhos)
    rank_auc = rank_score(aurocs)

    for s, rr, ra in zip(scores, rank_rho, rank_auc):
        s["composite_score"] = 0.5 * rr + 0.5 * ra
        s["rank_abs_rho"] = rr
        s["rank_auroc"] = ra

    ranked = sorted(scores, key=lambda s: s["composite_score"], reverse=True)
    top_score = ranked[0]["composite_score"]
    tie_group = [s for s in ranked if top_score - s["composite_score"] <= overlap_margin]

    tie_used = False
    tie_reason = None
    if len(tie_group) == 1:
        chosen = tie_group[0]
    else:
        tie_used = True
        tie_reason = "Composite scores tied; picked simplest feature."
        chosen = min(tie_group, key=lambda s: INTERPRETABILITY_RANK.get(s["feature"], 99))

    if len(ranked) > 1:
        first, second = ranked[0], ranked[1]
        if (
            ci_ranges_overlap(
                first["spearman_ci_low"], first["spearman_ci_high"],
                second["spearman_ci_low"], second["spearman_ci_high"],
            )
            and ci_ranges_overlap(
                first["auroc_ci_low"], first["auroc_ci_high"],
                second["auroc_ci_low"], second["auroc_ci_high"],
            )
            and first["feature"] != second["feature"]
        ):
            simpler = min((first, second), key=lambda s: INTERPRETABILITY_RANK.get(s["feature"], 99))
            if simpler["feature"] != first["feature"]:
                tie_used = True
                tie_reason = "Bootstrap CIs overlapped; picked simplest feature."
                chosen = simpler

    meta = {
        "rule": "composite = 0.5 * rank(|rho|) + 0.5 * rank(AUROC) over CALIB candidates",
        "overlap_margin": overlap_margin,
        "tie_break_used": tie_used,
        "tie_break_reason": tie_reason,
        "selected_feature": chosen["feature"],
        "representative_feature": chosen["feature"],
        "composite_score": chosen["composite_score"],
    }
    return chosen["feature"], meta


def build_corr_matrix(frame: pd.DataFrame, spearmanr) -> dict[str, Any]:
    names = list(FEATURE_NAMES)
    matrix: dict[str, dict[str, float | None]] = {}
    for a in names:
        matrix[a] = {}
        for b in names:
            if a == b:
                matrix[a][b] = 1.0
            elif b in matrix and a in matrix.get(b, {}):
                matrix[a][b] = matrix[b][a]
            else:
                rho, _ = spearmanr(frame[a].astype(float), frame[b].astype(float))
                matrix[a][b] = float(rho) if np.isfinite(rho) else None
    return {"spearman": matrix}


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray, spearmanr) -> float | None:
    from scipy.stats import rankdata

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if mask.sum() < 5:
        return None
    xr, yr, zr = rankdata(x[mask]), rankdata(y[mask]), rankdata(z[mask])
    zc = zr - zr.mean()
    if np.var(zc) < 1e-12:
        return None
    rx = xr - np.mean(xr) - np.cov(xr, zr, ddof=0)[0, 1] / np.var(zr) * zc
    ry = yr - np.mean(yr) - np.cov(yr, zr, ddof=0)[0, 1] / np.var(zr) * zc
    rho, _ = spearmanr(rx, ry)
    return float(rho) if np.isfinite(rho) else None


def check_length_confounds_all(frame: pd.DataFrame, spearmanr) -> dict[str, Any]:
    """Partial Spearman(feature, opportunity | piece_count) for every candidate."""
    opp = frame["y_opp"].astype(float).to_numpy()
    length = frame["piece_count"].astype(float).to_numpy()
    by_feature: dict[str, Any] = {}
    for name in FEATURE_NAMES:
        feat = frame[name].astype(float).to_numpy()
        rho_feat_len, _ = spearmanr(feat, length)
        rho_feat_opp, _ = spearmanr(feat, opp)
        by_feature[name] = {
            "rho_feature_length": float(rho_feat_len) if np.isfinite(rho_feat_len) else None,
            "rho_feature_opportunity": float(rho_feat_opp) if np.isfinite(rho_feat_opp) else None,
            "partial_rho_opportunity_given_length": partial_spearman(feat, opp, length, spearmanr),
        }
    return {
        "control_variable": "piece_count",
        "by_feature": by_feature,
    }


def check_length_confound(frame: pd.DataFrame, selected: str, spearmanr) -> dict[str, Any]:
    """Summary for the selected feature (convenience alias)."""
    all_checks = check_length_confounds_all(frame, spearmanr)
    selected_block = all_checks["by_feature"].get(selected, {})
    return {
        "selected_feature": selected,
        **selected_block,
    }


def run_feature_screen(
    frame: pd.DataFrame,
    *,
    expected_n: int = CALIB_SIZE,
    allow_preview: bool = False,
    boot_count: int = BOOTSTRAP_COUNT,
    boot_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    spearmanr, _, _, roc_auc_score, _, _ = _require_analysis_deps()

    if len(frame) != expected_n and not allow_preview:
        raise ValueError(
            f"CALIB screening requires n={expected_n}; got {len(frame)}. Use allow_preview for dev."
        )

    missing = [n for n in FEATURE_NAMES if n not in frame.columns]
    if missing:
        raise ValueError(f"Features CSV missing columns: {missing}")

    scores = [
        score_feature(
            frame, name,
            spearmanr=spearmanr,
            roc_auc_score=roc_auc_score,
            boot_count=boot_count,
            boot_seed=boot_seed + i * 10,
        )
        for i, name in enumerate(FEATURE_NAMES)
    ]
    selected, pick_meta = pick_best_feature(scores)
    corr = build_corr_matrix(frame, spearmanr)
    length_checks = check_length_confounds_all(frame, spearmanr)

    return {
        "screening_tag": "D46",
        "process": "signal_screening",
        "screen_version": SCREEN_TAG,
        "slice": "ARC validation (CALIB)",
        "n": len(frame),
        "calib_locked": len(frame) == expected_n and not allow_preview,
        "feature_families": FEATURE_FAMILIES,
        "selection_rule": "rank(|rho|) + rank(AUROC) composite; simplicity tie-break",
        "bootstrap": {"count": boot_count, "ci_level": 0.95, "seed": boot_seed},
        "features_scored": scores,
        "feature_correlation": corr,
        "length_confound_checks": length_checks,
        "length_confound_check": check_length_confound(frame, selected, spearmanr),
        "selection": pick_meta,
    }


def run_feature_extraction(
    *,
    queries: list[dict],
    tokenizer,
    output: Path,
    tokenizer_id: str,
    mattr_window: int = 15,
) -> int:
    """Extract model-independent complexity features (tokenizer only — no forward pass)."""
    import csv

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FEATURE_CSV_FIELDS)
        w.writeheader()
        for q in queries:
            row_data = compute_feature_row(q["user_content"], tokenizer, mattr_window=mattr_window)
            row = {
                "query_id": q["id"],
                "row_uid": q.get("row_uid"),
                "user_content": q["user_content"],
                "piece_count": row_data["piece_count"],
                "mattr": round(row_data["mattr"], 6),
                "text_shannon": round(row_data["text_shannon"], 6),
                "text_shannon_norm": round(row_data["text_shannon_norm"], 6),
                "compression_ratio": round(row_data["compression_ratio"], 6),
                "unique_tokens": row_data["unique_tokens"],
                "tokenizer_id": tokenizer_id,
                "screen_version": SCREEN_TAG,
                "extraction_method": "query_complexity_v2_llama_tokenizer",
            }
            w.writerow(row)
            print(
                f"{q['id']}: pieces={row['piece_count']} mattr={row['mattr']:.4f} "
                f"H_text={row['text_shannon']:.4f} H_norm={row['text_shannon_norm']:.4f} "
                f"compr={row['compression_ratio']:.4f}"
            )
    print(f"Wrote {output} ({len(queries)} rows, features={FEATURE_NAMES})")
    return 0
