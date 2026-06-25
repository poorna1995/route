# Figure specifications (§5–§6)

Generate after main characterization analysis (n=150–200). Filenames: `F1_signal_distributions.pdf`, etc.

## F1 — Signal distributions by outcome bucket (§5.1)

- **Type:** violin or box plots (4 panels or faceted)
- **X:** oracle bucket (easy, opportunity, weak-only, too-hard)
- **Y:** $H_{\text{weak}}$, $H_{\text{strong}}$, $m_{\text{weak}}$, $m_{\text{strong}}$
- **Purpose:** do signal distributions differ across routing-relevant buckets?
- **Command:** `run.py plot distributions`

## F2 — ROC curves (§5.2)

- **Type:** ROC per headline signal vs opportunity
- **Purpose:** ranking strength (AUROC + CI from `run.py merge`)
- **Command:** `run.py plot roc`

## F3 — Entropy vs margin scatter (§5.2)

- **Type:** scatter, points colored by oracle bucket
- **Axes:** $H_w$ vs $m_w$
- **Purpose:** visual complementarity / redundancy before complementarity analysis
- **Command:** `run.py plot scatter`

## F4 — Cost–quality routing plane (§6, conditional)

- **Type:** line or scatter (RouterBench-style)
- **Points:** always-weak, always-strong, $P(\text{opp})$ router, oracle
- **Axes:** accuracy vs normalized cost
- **When:** only if §6 routing evaluation runs

## Pipeline (after main characterization)

```bash
# 1. Oracle + signals → merge
.venv/bin/python scripts/run.py merge \
    --oracle experiments/M4/routing_opportunity/llama_arc_n200.json \
    --weak-csv experiments/M5/llama_arc_n200_weak_signals.csv \
    --strong-csv experiments/M5/llama_arc_n200_strong_signals.csv \
    --output analysis/llama_arc_n200_routing_relevance.json \
    --merged-csv analysis/llama_arc_n200_merged.csv

# 2. Figures
.venv/bin/python scripts/run.py plot distributions \
    --merged-csv analysis/llama_arc_n200_merged.csv \
    --output paper/figures/F1_signal_distributions.png
.venv/bin/python scripts/run.py plot roc \
    --merged-csv analysis/llama_arc_n200_merged.csv \
    --output paper/figures/F2_roc_curves.png
.venv/bin/python scripts/run.py plot scatter \
    --merged-csv analysis/llama_arc_n200_merged.csv \
    --output paper/figures/F3_entropy_margin_scatter.png

# 3. Complementarity ladder
.venv/bin/python scripts/run.py complementarity \
    --merged-csv analysis/llama_arc_n200_merged.csv \
    --output analysis/llama_arc_n200_complementarity.json

# 4. (Internal only) median-heuristic sanity check — NOT for paper
.venv/bin/python scripts/run.py route-preview \
    --merged-csv analysis/llama_arc_n200_merged.csv \
    --output analysis/llama_arc_n200_routing_preview.json
```
