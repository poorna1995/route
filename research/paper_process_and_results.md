# Paper 1 — Process, findings, and results

> **Vocabulary:** [`claims.md`](claims.md) · **Frozen design:** [`MASTER.md`](MASTER.md) · **Commands:** [`../experiments/README.md`](../experiments/README.md)

This document is the end-to-end account of what was done, in what order, and what was found on the locked ARC-Challenge configuration. All test numbers below come from filled paper tables (T2–T4) and oracle bucket counts reported in the draft.

---

## 1. Research question and answer

**Question:** Can unsupervised pre-inference signals support appropriate multi-LLM routing before generation?

**Operational meaning:** Given a query, extract statistics from query text and prefill logits **without routing labels at extraction time**, then ask whether those statistics carry **routing-relevant information** (association with routing opportunity and oracle buckets) and whether a **calibrated policy** can exploit that information.

**Locked answer (ARC TEST, \(n{=}1{,}172\)):** **Partially.**

| Aspect | Result |
| ------ | ------ |
| Information exists? | Yes — opportunity 43.3%; best dimension AUROC \(\approx 0.61\) |
| Dimensions differ? | Yes — difficulty-like vs recoverability-like structure |
| Complementarity? | Partial — task difficulty + model uncertainty adds \(\Delta\)AUROC \(+0.049\) (DeLong \(p{=}0.008\)) |
| Exploitable by calibrated policy? | No on TEST — policy matches always-strong (69.2%); oracle 74.4% leaves 5.2 pp unexploited |

---

## 2. Locked experimental configuration

| Component | Value |
| --------- | ----- |
| Weak model \(M_w\) | `meta-llama/Llama-3.2-1B-Instruct` |
| Strong model \(M_s\) | `meta-llama/Llama-3.2-3B-Instruct` |
| Dataset | ARC-Challenge (`allenai/ai2_arc`, Challenge split) |
| CALIB | Official validation, \(n{=}299\) |
| TEST | Official test, \(n{=}1{,}172\) |
| Train split | Unused |
| Prompt formatting | Deterministic chat template (`prompt_protocol.py`) |
| Seed | 42 |
| Complexity representative \(c(q)\) | `piece_count` (D46 winner on CALIB) |
| Cost model (routing eval) | weak \(=1\), strong \(=3\) |

**Selection rationale:** Screened small pilots (Llama vs Qwen, ARC vs GSM8K) for non-degenerate routing opportunity and successful full-vocabulary prefill logits. Llama 3.2 1B/3B on ARC showed usable opportunity (\(\sim\)24–50% on pilots; 43.3% on full test).

---

## 3. End-to-end process (step by step)

### Step 0 — Freeze science layer

Lock research question, hypotheses RH1–RH4, models, dataset, splits, metrics, and vocabulary (`MASTER.md`, `claims.md`). No methodology redesign after freeze (D62).

### Step 1 — Split manifest

Write which official HF splits map to CALIB vs TEST:

```bash
.venv/bin/python scripts/run.py splits \
  --dataset arc_challenge \
  --output analysis/splits.json
```

Policy: ARC **validation → CALIB** (representative selection + policy fitting); ARC **test → TEST** (all reported metrics).

### Step 2 — Configuration screening (validation pilots)

Before locking, run small-\(n\) pilots:

- Confirm prefill extraction returns full-vocabulary logits at final prompt position.
- Confirm oracle buckets are non-degenerate (not 100% too-hard or 0% opportunity).

Excluded examples from appendix: Qwen 2.5 1.5B/3B on GSM8K (100% too-hard); Qwen on ARC pilot (0% opportunity). Llama 3.2 on ARC retained.

### Step 3 — Offline oracle labels (full inference)

For each \((q, M_i)\) on CALIB and TEST:

1. Build the same chat prompt used for signal extraction.
2. Run greedy full inference (`max_new_tokens=8` for MCQ letter answers).
3. Parse output → binary correctness \(y(q,M_i)\).

Derive:

- **Routing opportunity:** \(y^{\text{opp}} = 1\) iff weak fails and strong succeeds.
- **Buckets:** easy, opportunity, weak-only, too-hard.

```bash
.venv/bin/python scripts/run.py oracle \
  --weak meta-llama/Llama-3.2-1B-Instruct \
  --strong meta-llama/Llama-3.2-3B-Instruct \
  --dataset arc_challenge \
  --splits-json analysis/splits.json --split-role calib \
  --max-new-tokens 8 --device cuda --dtype bfloat16 \
  --output experiments/M4/routing_opportunity/arc_validation_oracle.json
```

Repeat for `--split-role test`. Oracle JSON is reused for all downstream analysis.

**Supervision note:** Oracle labels are **offline only** — never used to compute signals.

### Step 4 — Model-independent signal candidates (query text)

Extract tokenizer-based complexity candidates on CALIB (no GPU forward pass):

```bash
.venv/bin/python scripts/run.py features \
  --dataset arc_challenge \
  --splits-json analysis/splits.json --split-role calib \
  --output experiments/M5/arc_validation_features.csv
```

Candidates: `piece_count`, `mattr`, `text_shannon`, `text_shannon_norm`, `compression_ratio`.

### Step 5 — D46 pre-study calibration (select \(c(q)\) once)

Screen candidates on CALIB only; freeze winner before any TEST reporting:

```bash
.venv/bin/python scripts/run.py screen \
  --features experiments/M5/arc_validation_features.csv \
  --oracle experiments/M4/routing_opportunity/arc_validation_oracle.json \
  --splits-json analysis/splits.json \
  --output analysis/d46_signal_screen_arc.json
```

**Winner:** `piece_count` (tokenizer piece count on formatted user content).

| Candidate | \(\rho_s\) [95% CI] | AUROC [95% CI] | Composite rank |
| --------- | ------------------- | -------------- | -------------- |
| **piece_count** | +0.093 [−0.024, +0.210] | 0.556 [0.486, 0.629] | **4.0 (winner)** |
| mattr | −0.067 [−0.179, +0.048] | 0.460 [0.395, 0.530] | 2.5 |
| text_shannon | −0.015 [−0.121, +0.099] | 0.491 [0.426, 0.560] | 2.5 |
| text_shannon_norm | −0.151 [−0.264, −0.039] | 0.409 [0.345, 0.481] | 3.0 |
| compression_ratio | −0.172 [−0.282, −0.061] | 0.397 [0.328, 0.465] | 3.0 |

Frozen in `analysis/selected_feature.json`. **Do not re-screen** without a new decision in `09_decision_register.md`.

### Step 6 — Model-dependent signals (prefill probes)

One prefill forward pass per model; logits at final prompt position \(t{=}T\):

- **Uncertainty:** \(H(q,M_i) = -\sum_v p_T(v)\log p_T(v)\) (full vocabulary, \(\tau{=}1\))
- **Confidence:** \(m(q,M_i) = p_T^{(1)} - p_T^{(2)}\)

Cross-model statistics:

- **Agreement:** \(\Delta H = H_w - H_s\)
- **Recoverability:** \(\Delta m_{\mathrm{gain}} = m_s - m_w\)

```bash
.venv/bin/python scripts/run.py probes \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dataset arc_challenge \
  --splits-json analysis/splits.json --split-role test \
  --device cuda --dtype bfloat16 --batch-size 8 \
  --output experiments/M5/arc_test_weak_signals.csv

# Repeat for strong model → arc_test_strong_signals.csv
```

**Supervision note:** Signal extraction is **unsupervised** — no routing labels in this step.

### Step 7 — Merge master table

Join oracle, probes, and frozen \(c(q)\):

```bash
.venv/bin/python scripts/run.py merge \
  --weak-csv experiments/M5/arc_test_weak_signals.csv \
  --strong-csv experiments/M5/arc_test_strong_signals.csv \
  --oracle experiments/M4/routing_opportunity/arc_test_oracle.json \
  --features-csv experiments/M5/arc_test_query_features.csv \
  --complexity-selection analysis/selected_feature.json \
  --output analysis/arc_routing_relevance.json \
  --merged-csv analysis/arc_merged.csv
```

Pre-flight:

```bash
.venv/bin/python scripts/run.py doctor \
  --oracle ... --weak-csv ... --strong-csv ... \
  --features-csv ... --complexity-selection analysis/selected_feature.json
```

### Step 8 — Signal characterization (Studies I–III)

**Study I–II (RH1, RH2):** Spearman \(\rho\), AUROC/AUPRC vs \(y^{\text{opp}}\), bucket distributions — reported **by information dimension** (Table T2, Figures F1, F2, F6).

**Study III (RH3):** Nested CALIB-fit / TEST-eval logistic ladder; \(\Delta\)AUROC and DeLong test (Table T3, Figure F3).

```bash
.venv/bin/python scripts/run.py complementarity \
  --merged-csv analysis/arc_merged.csv \
  --splits-json analysis/splits.json \
  --output analysis/arc_complementarity.json
```

Study III models are **not** reused for Study IV.

### Step 9 — Interpretation bundle (optional diagnostics)

```bash
.venv/bin/python scripts/run.py interpret \
  --merged-csv analysis/arc_merged.csv \
  --splits-json analysis/splits.json \
  --output analysis/arc_interpretation.json
```

Produces landscape, overlap, and decomposition summaries for opportunity vs too-hard contrasts (Finding 2).

### Step 10 — Routing evaluation (Study IV, RH4)

Fit calibrated logistic policy on CALIB; tune threshold \(\tau\) on CALIB; report TEST only:

\[
\hat{p}(q) = P(y^{\text{opp}}{=}1 \mid c, H_w, m_w), \quad
U = \text{accuracy} - \lambda \cdot \text{avg.\ cost}
\]

```bash
.venv/bin/python scripts/run.py route-eval \
  --merged-csv analysis/arc_merged.csv \
  --splits-json analysis/splits.json \
  --output analysis/arc_routing_holdout.json
```

Baselines: always-weak, always-strong, calibrated policy, offline oracle.

### Step 11 — Paper figures

```bash
.venv/bin/python scripts/run.py plot conceptual-model --output paper/figures/F0_conceptual_model.png
.venv/bin/python scripts/run.py plot distributions --merged-csv analysis/arc_merged.csv --output paper/figures/F1_bucket_distributions.png
# roc, scatter, decomposition, etc.
```

---

## Scientific unit: latent routing dimensions

Each experiment asks: **which latent routing dimension predicts routing opportunity?**

| Latent routing dimension | Operationalization |
| ------------------------ | ------------------ |
| Task difficulty | `piece_count` / $c(q)$ |
| Model uncertainty | $H_w$ |
| Model disagreement | $\Delta H$ |
| Escalation potential | $\Delta m_{\mathrm{gain}}$ |

AUROC and $\rho$ summarize detectability for a dimension—they are not a feature leaderboard.

---

## 4. Hypotheses → what was tested → outcome

### RH1 — Which dimensions predict opportunity? (Studies I–II)

**Test:** Do unsupervised pre-inference signals show predictive association with \(y^{\text{opp}}\) above chance?

**Oracle landscape (TEST):**

| Bucket | Count | Rate |
| ------ | ----- | ---- |
| Easy | 304 | 25.9% |
| **Opportunity** | **507** | **43.3%** |
| Weak-only | 61 | 5.2% |
| Too-hard | 300 | 25.6% |
| **Total** | **1,172** | 100% |

**Result:** **Supported.** All four main dimensions have AUROC \(> 0.5\); escalation potential and model disagreement reach \(\approx 0.60\)–\(0.61\).

| Latent routing dimension | Operationalization | \(\rho_s\) [95% CI] | AUROC [95% CI] | AUPRC |
| ------------------------ | ------------------ | ------------------- | -------------- | ----- |
| Task difficulty | \(c(q)\) = piece_count | +0.071 [+0.015, +0.128] | 0.541 [0.509, 0.574] | 0.474 |
| Model uncertainty | \(H_w\) | +0.138 [+0.082, +0.193] | 0.581 [0.547, 0.613] | 0.500 |
| Model disagreement | \(\Delta H\) | +0.174 [+0.118, +0.232] | 0.602 [0.569, 0.635] | 0.559 |
| Escalation potential | \(\Delta m_{\mathrm{gain}}\) | +0.179 [+0.123, +0.239] | **0.605** [0.572, 0.637] | 0.547 |

**Interpretation:** Modest but real signal — detection is far from perfect (best AUROC \(\sim 0.61\)), not random.

---

### RH2 — Dimensions encode distinct routing need (Study II + interpretation)

**Difficulty-side** (task difficulty, model uncertainty) vs **escalation-side** (model disagreement, escalation potential).

| Dimension | Cohen's \(d\) (opportunity vs too-hard) |
| --------- | --------------------------------------- |
| Model uncertainty \(H_w\) | \(\approx 0.03\) |
| Escalation potential \(\Delta m_{\mathrm{gain}}\) | \(\approx 0.72\) |
| Model disagreement \(\Delta H\) | \(\approx 0.82\) |

**Result:** **Supported.**

- **Task difficulty** and **model uncertainty** track overlapping aspects of routing difficulty.
- **Model disagreement** and **escalation potential** separate routable from irrecoverable hard queries.

**Recovery matrix (median splits on \(H_w\) and \(\Delta m_{\mathrm{gain}}\)):** Highest opportunity rate **57.2%** (\(n{=}353\)) in the weak-uncertain / strong-rescues cell.

**Routing implication:** A policy that treats high weak-model uncertainty as “escalate to strong” will confuse opportunity with too-hard queries.

---

### RH3 — Partial complementarity (Study III)

**Test:** Does adding dimensions improve opportunity detection beyond a single dimension?

**Ladder (logistic on CALIB, AUROC on TEST):**

| Dimensions included | Representatives | AUROC [95% CI] |
| ------------------- | --------------- | -------------- |
| Complexity | \(c(q)\) | 0.541 [0.509, 0.574] |
| Uncertainty (ref.) | \(H_w\) | 0.581 |
| Confidence (ref.) | \(m_w\) | 0.568 |
| Complexity + Uncertainty | \(c, H_w\) | 0.590 [0.558, 0.622] |
| Complexity + Confidence | \(c, m_w\) | 0.576 [0.543, 0.608] |
| Complexity + Uncertainty + Confidence | \(c, H_w, m_w\) | 0.578 [0.546, 0.612] |

**Primary increment:**

| Comparison | \(\Delta\)AUROC [95% CI] | DeLong \(p\) |
| ---------- | ------------------------ | ------------ |
| Complexity → Complexity + Uncertainty | **+0.049** [+0.014, +0.085] | **0.008** |
| Complexity + Uncertainty → joint (+ Confidence) | −0.013 [−0.031, +0.005] | — |

**Result:** **Partially supported.** Query complexity and weak-model uncertainty are **partially complementary**; adding confidence does not help further on ARC.

---

### RH4 — Routing evaluation (Study IV)

**Test:** Can a calibrated policy exploit characterized information for cost–quality gains vs static baselines?

**Policy:** Logistic \(P(y^{\text{opp}} \mid c, H_w, m_w)\) fit on CALIB; threshold \(\tau\) tuned on CALIB for \(U = \text{accuracy} - \lambda \cdot \text{cost}\); evaluate on TEST (\(\lambda{=}0\)).

| Policy | Accuracy | Avg. cost | \(U\) (\(\lambda{=}0\)) | Opp. recall |
| ------ | -------- | --------- | ----------------------- | ----------- |
| Always-weak | 31.1% | 1.00 | 0.311 | 0% |
| Always-strong | 69.2% | 3.00 | 0.692 | 100% |
| **Calibrated policy** | **69.2%** | **3.00** | **0.692** | **100%** |
| Oracle (bound) | 74.4% | 1.87 | — | 100% |

**Headroom:** Oracle − always-strong = **5.2 percentage points**; calibrated policy exploits **0 pp** of that gap on TEST (\(\tau{=}0\) → route all to strong).

**Result:** **Not supported for exploitation** (policy does not beat always-strong), but **informative for the routing question:** detection (\(\sim 0.61\) AUROC) exceeds assignment under this policy class. The gap is an **exploitation limit**, not proof that signals are uninformative.

---

## 5. Three reader-facing findings (paper Results spine)

### Finding 1 — Information exists

Before generation, routing-relevant information is measurable: 43.3% of TEST queries are routing opportunities; best dimension representative AUROC \(\sim 0.61\).

### Finding 2 — Dimensions differ

Information is structured: difficulty-like dimensions (Complexity, Uncertainty) vs escalation-like dimensions (Recoverability, Agreement). Uncertainty does not separate opportunity from too-hard; Recoverability does.

### Finding 3 — Exploitation gap

A CALIB-fit calibrated policy matches always-strong (69.2%) while an offline oracle reaches 74.4% — **5.2 pp** of improvement remains unexploited by this policy class on TEST.

---

## 6. Supervision boundary (reviewer-critical)

| Stage | Uses routing labels? | Role |
| ----- | -------------------- | ---- |
| Signal extraction | **No** | Unsupervised at inference |
| Signal characterization | Oracle offline | Measure routing-relevant information |
| Routing evaluation | CALIB only | Exploitation test; not the research claim |

---

## 7. What this study does not claim

- A new supervised router or SOTA routing method.
- That entropy alone “predicts routing.”
- Generalization beyond ARC-Challenge (MMLU transfer planned/deferred).
- Agent routing, task decomposition, or paraphrase stability (future work).

---

## 8. Reproducibility checklist

| Artifact | Purpose |
| -------- | ------- |
| `analysis/splits.json` | CALIB/TEST split policy |
| `analysis/selected_feature.json` | Frozen \(c(q)\) = piece_count |
| `experiments/M4/routing_opportunity/*.json` | Offline oracle |
| `experiments/M5/*_signals.csv` | Prefill probes |
| `analysis/arc_merged.csv` | Master merged table |
| `paper/tables/T2_correlation.tex` | RH1–RH2 numbers |
| `paper/tables/T3_complementarity.tex` | RH3 numbers |
| `paper/tables/T4_routing.tex` | RH4 numbers |

**Build paper:** `pdflatex paper/acl.tex` (see `research/11_paper_outline.md`).

---

## 9. One-paragraph synthesis (for Abstract/Discussion)

On ARC-Challenge with Llama 3.2 1B/3B, unsupervised pre-inference signals carry modest but measurable routing-relevant information before full generation. Information **dimensions** encode different aspects of routing need: complexity and uncertainty overlap with task difficulty, while recoverability and agreement separate routable from irrecoverable hard queries. Complexity and uncertainty are partially complementary for opportunity detection (\(\Delta\)AUROC \(+0.049\), \(p{=}0.008\)). A calibrated logistic routing policy, however, does not improve over always routing to the strong model on the held-out test set, leaving 5.2 percentage points of oracle headroom unexploited. The answer to whether pre-inference signals can **support** routing is therefore **partial**: information exists and is structured, but simple exploitation is limited.
