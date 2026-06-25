"""Shared column labels, feature names, ladders, and analysis tags — single source of truth."""

from __future__ import annotations

# Populated only after D46 screening (selected candidate → unified column).
COL_COMPLEXITY = "c_q"
COL_ROW_UID = "row_uid"

COL_ENTROPY_WEAK = "entropy_w"
COL_MARGIN_WEAK = "margin_w"
COL_ENTROPY_STRONG = "entropy_s"
COL_MARGIN_STRONG = "margin_s"
COL_OPPORTUNITY = "y_opp"
COL_COMPLEXITY_CANDIDATE = "complexity_candidate"

# Raw probe CSV metric stems (before _w / _s suffix in merged table).
PROBE_METRIC_BASES = (
    "entropy",
    "entropy_norm",
    "entropy_top10",
    "n_eff",
    "margin",
    "max_prob",
    "top5_mass",
)

PROBE_DERIVED_ENTROPY = "delta_entropy"  # H_w − H_s
# Strong-minus-weak margin gain (higher ⇒ strong model more peaked than weak).
PROBE_DERIVED_MARGIN = "delta_margin_gain"  # m_s − m_w

FEATURE_PIECE_COUNT = "piece_count"
FEATURE_MATTR = "mattr"
FEATURE_SHANNON = "text_shannon"
FEATURE_SHANNON_NORM = "text_shannon_norm"
FEATURE_COMPRESSION = "compression_ratio"

FEATURE_NAMES = (
    FEATURE_PIECE_COUNT,
    FEATURE_MATTR,
    FEATURE_SHANNON,
    FEATURE_SHANNON_NORM,
    FEATURE_COMPRESSION,
)

FEATURE_FAMILIES: dict[str, dict[str, object]] = {
    "length": {"label": "Length", "features": (FEATURE_PIECE_COUNT,)},
    "lexical_diversity": {"label": "Lexical diversity", "features": (FEATURE_MATTR,)},
    "information": {
        "label": "Information",
        "features": (FEATURE_SHANNON, FEATURE_SHANNON_NORM),
        "note": "D46 selects one representative → c_q; do not preset Shannon vs normalized.",
    },
    "compressibility": {"label": "Compressibility", "features": (FEATURE_COMPRESSION,)},
}

# Tie-break only when rank composite scores or bootstrap CIs overlap (lower = preferred).
INTERPRETABILITY_RANK: dict[str, int] = {
    FEATURE_PIECE_COUNT: 1,
    FEATURE_COMPRESSION: 2,
    FEATURE_SHANNON: 3,
    FEATURE_SHANNON_NORM: 4,
    FEATURE_MATTR: 5,
}

# Paper-primary signal order (Results §5 tables / figures).
HEADLINE_SIGNALS = (
    COL_COMPLEXITY,
    COL_ENTROPY_WEAK,
    COL_MARGIN_WEAK,
)

# Study II probe panels (weak + strong prefill).
PROBE_HEADLINE = (
    COL_ENTROPY_WEAK,
    COL_MARGIN_WEAK,
    COL_ENTROPY_STRONG,
    COL_MARGIN_STRONG,
)

PROBE_DERIVED = (PROBE_DERIVED_ENTROPY, PROBE_DERIVED_MARGIN)

# Full routing-relevance reporting set (effect sizes, terminal summary).
ROUTING_RELEVANCE_SIGNALS = tuple(
    dict.fromkeys(HEADLINE_SIGNALS + PROBE_HEADLINE + PROBE_DERIVED)
)

# ROC figure curves (F2).
ROC_CURVE_SIGNALS: tuple[tuple[str, str], ...] = (
    (COL_ENTROPY_WEAK, r"Weak entropy $H_w$"),
    (COL_MARGIN_WEAK, r"Weak margin $m_w$"),
    (PROBE_DERIVED_ENTROPY, r"Derived $\Delta H = H_w - H_s$"),
    (PROBE_DERIVED_MARGIN, r"Derived $\Delta m_{\mathrm{gain}} = m_s - m_w$"),
)

# Distribution figure panels (F1).
DISTRIBUTION_PANELS: tuple[tuple[str, str], ...] = (
    (COL_ENTROPY_WEAK, "Weak entropy $H_w$"),
    (COL_ENTROPY_STRONG, "Strong entropy $H_s$"),
    (COL_MARGIN_WEAK, "Weak margin $m_w$"),
    (COL_MARGIN_STRONG, "Strong margin $m_s$"),
)

LADDER_CROSS_FAMILY: list[tuple[str, list[str]]] = [
    ("complexity_only", [COL_COMPLEXITY]),
    ("complexity_entropy", [COL_COMPLEXITY, COL_ENTROPY_WEAK]),
    ("complexity_joint", [COL_COMPLEXITY, COL_ENTROPY_WEAK, COL_MARGIN_WEAK]),
]

# Baseline single-family models for Study III reference (not the primary ladder).
REFERENCE_MODELS: list[tuple[str, list[str]]] = [
    ("entropy_solo", [COL_ENTROPY_WEAK]),
    ("margin_solo", [COL_MARGIN_WEAK]),
]


def _step_name(from_cols: list[str], to_cols: list[str], from_label: str, to_label: str) -> str:
    added = set(to_cols) - set(from_cols)
    if COL_ENTROPY_WEAK in added:
        return "entropy_beyond_complexity"
    if COL_MARGIN_WEAK in added:
        return "margin_beyond_joint"
    return f"{from_label}_to_{to_label}"


def build_complementarity_steps(
    ladder: list[tuple[str, list[str]]] | None = None,
) -> list[tuple[str, str, str]]:
    """ΔAUROC steps derived from consecutive ladder rungs."""
    ladder = ladder or LADDER_CROSS_FAMILY
    steps: list[tuple[str, str, str]] = []
    for i in range(len(ladder) - 1):
        from_label, from_cols = ladder[i]
        to_label, to_cols = ladder[i + 1]
        steps.append((from_label, to_label, _step_name(from_cols, to_cols, from_label, to_label)))
    return steps


COMPLEMENTARITY_STEPS = build_complementarity_steps()

# Study III centerpiece figure: four ablation models (AUROC at each rung).
ABLATION_LADDER: list[tuple[str, list[str]]] = [
    ("complexity_only", [COL_COMPLEXITY]),
    ("complexity_entropy", [COL_COMPLEXITY, COL_ENTROPY_WEAK]),
    ("complexity_margin", [COL_COMPLEXITY, COL_MARGIN_WEAK]),
    ("complexity_joint", [COL_COMPLEXITY, COL_ENTROPY_WEAK, COL_MARGIN_WEAK]),
]

STABILITY_CALIB_DRAWS = 5
CALIB_REDRAW_FOLDS = 5
PERMUTATION_COUNT = 1000

HYPOTHESES: dict[str, dict[str, object]] = {
    "RH1": {
        "question": "Representative model-independent signal relates to routing need",
        "signals": [COL_COMPLEXITY],
    },
    "RH2": {
        "question": "Model-dependent prefill probes relate to routing need",
        "signals": [COL_ENTROPY_WEAK, COL_MARGIN_WEAK],
    },
    "RH3": {
        "question": "Signal families complement each other beyond solo baselines",
        "ladder": LADDER_CROSS_FAMILY,
        "reference_models": REFERENCE_MODELS,
    },
}

BOOTSTRAP_COUNT = 2000
BOOTSTRAP_SEED = 42

# ARC-Challenge official HF splits (Option 1: validation → CALIB, test → paper eval).
# Train is intentionally unused — no LLM or router training in this study.
# Split sizes are discovered at manifest creation time (see split_source_metadata).
ARC_CALIB_SPLIT = "validation"
ARC_EVAL_SPLIT = "test"
CALIB_SIZE = 150  # legacy internal-partition default only; prefer splits manifest

SCORE_OVERLAP = 0.05
SCREEN_TAG = "D46-v2"

# Frozen D46 artifact written by `screen`, consumed by `merge` (prevents manual re-picking).
COMPLEXITY_SELECTION_SCHEMA = "complexity_selection_v1"
DEFAULT_COMPLEXITY_SELECTION_NAME = "selected_feature.json"

# Study III: stratified k-fold CV for logistic AUROC (avoids in-sample optimism).
COMPLEMENTARITY_CV_FOLDS = 5
COMPLEMENTARITY_CV_MIN_CLASS = 2

ANALYSIS_ROUTING_RELEVANCE = "routing_relevance"
ANALYSIS_COMPLEMENTARITY = "signal_complementarity"
ANALYSIS_ROUTING_PREVIEW = "routing_preview"
ANALYSIS_ROUTING_HOLDOUT = "routing_holdout"

# Study IV router features (independent fit from Study III characterization).
ROUTING_ROUTER_FEATURES = (COL_COMPLEXITY, COL_ENTROPY_WEAK, COL_MARGIN_WEAK)

DEFAULT_TOKENIZER_ID = "meta-llama/Llama-3.2-1B-Instruct"

FEATURE_CSV_FIELDS = (
    "query_id",
    "row_uid",
    "user_content",
    "piece_count",
    "mattr",
    "text_shannon",
    "text_shannon_norm",
    "compression_ratio",
    "unique_tokens",
    "tokenizer_id",
    "screen_version",
    "extraction_method",
)

PROBE_CSV_FIELDS = (
    "query_id",
    "row_uid",
    "model_id",
    "user_content",
    "entropy",
    "entropy_norm",
    "entropy_top10",
    "n_eff",
    "margin",
    "max_prob",
    "top5_mass",
    "top1_token",
    "top2_token",
    "vocab_size",
    "prompt_tokens",
    "protocol_version",
    "prompt_hash",
    "chat_template",
    "tokenizer_id",
    "extraction_method",
)

WEAK_COST = 1.0
STRONG_COST = 3.0

BUCKET_ORDER = ("easy", "opportunity", "weak_only", "too_hard")
BUCKET_LABELS = {
    "easy": "Easy",
    "opportunity": "Opportunity",
    "weak_only": "Weak-only",
    "too_hard": "Too hard",
}
