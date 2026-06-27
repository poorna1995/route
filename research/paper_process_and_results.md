# Paper 1 — End-to-end process, results, and findings

> **Vocabulary:** [`claims.md`](claims.md) · **Frozen design:** [`MASTER.md`](MASTER.md) · **Commands:** [`../experiments/README.md`](../experiments/README.md)

This document is the full account of what was run, in what order, and what was found on the locked configuration: **ARC-Challenge (primary)** and **MMLU (dimension-transfer / RH7)**. Numbers below come from merged analysis artifacts (`analysis/*.json`, `analysis/*_merged.csv`) unless noted.

---

## 1. Research question and answer

**Question:** Can unsupervised pre-inference signals support appropriate multi-LLM routing before generation?

**Operational meaning:** Extract statistics from query text and prefill logits **without routing labels at extraction time**, then measure **routing-relevant information** (association with routing opportunity and oracle buckets) and test whether a **calibrated policy** can exploit it.

### Answer on ARC TEST (\(n{=}1{,}172\)) — **Partially**

| Aspect | Result |
| ------ | ------ |
| Information exists? | Yes — opportunity 43.3%; best dimension AUROC \(\approx 0.61\) |
| Dimensions differ? | Yes — **difficulty vs recoverability** |
| Complementarity? | Partial — \(c + H_w\) adds \(\Delta\)AUROC \(+0.049\) (DeLong \(p{=}0.008\)) |
| Exploitable by calibrated policy? | No on TEST — policy matches always-strong (69.2%); oracle 74.4% leaves 5.2 pp unexploited |

### Answer on MMLU transfer (\(n{=}314\), RH7) — **Pattern holds pooled**

| Aspect | Result |
| ------ | ------ |
| C2 gate | Pass — opportunity 25.8%; four buckets present |
| Escalation invariant? | Yes pooled — \(\Delta m_{\mathrm{gain}}\) Cohen's \(d \approx 0.72\) (same as ARC) |
| Uncertainty weak? | Mostly — \(H_w\) \(d \approx 0.18\) (stronger than ARC's 0.03, still \(\ll\) escalation) |
| Subject nuance | US history strong; physics moderate; abstract algebra reverses (small \(n\)) |

---

## 2. Locked experimental configuration

| Component | Value |
| --------- | ----- |
| Weak model \(M_w\) | `meta-llama/Llama-3.2-1B-Instruct` |
| Strong model \(M_s\) | `meta-llama/Llama-3.2-3B-Instruct` |
| Prompt formatting | Chat template v1 (`prompt_protocol.py`) |
| Oracle decoding | Greedy, `max_new_tokens=8` (MCQ letter) |
| Seed | 42 |
| Complexity \(c(q)\) | `piece_count` — D46 winner on ARC CALIB (`analysis/selected_feature.json`) |
| Cost model (routing eval) | weak \(=1\), strong \(=3\) |

### Datasets and splits

| Campaign | Dataset | CALIB | TEST | \(n\) TEST |
| -------- | ------- | ----- | ---- | ---------- |
| **C0 — ARC (primary)** | ARC-Challenge | Official validation (299) | Official test (1,172) | 1,172 |
| **C2 — MMLU (transfer)** | MMLU `test` | None (reuse ARC D46) | Pooled subjects | 314 |

**MMLU subjects actually run (RunPod):** `high_school_us_history` (145), `abstract_algebra` (74), `high_school_physics` (95).  
*Note:* Repo default loader lists `high_school_physics` + `logical_fallacies`; the oracle JSON is the source of truth for this run.

**Why Instruct models:** Matches deployed multi-LLM routing; protocol uses `apply_chat_template`. See discussion in project notes — not optimal for calibration, but correct for the routing setting studied.

---

## 3. End-to-end pipeline (overview)

```text
splits → oracle (GPU) → features (CPU) → screen D46 on CALIB (ARC only)
      → probes weak + strong (GPU) → merge (CPU) → interpret / complementarity / route-eval
      → compare-generalization (ARC + MMLU)
```

**Supervision boundary**

| Stage | Routing labels? | Role |
| ----- | --------------- | ---- |
| Signal extraction | **No** | Unsupervised at inference |
| D46 screen | Oracle on CALIB only | Pick \(c(q)\) once |
| Signal characterization | Oracle offline | Studies I–III |
| Routing evaluation | CALIB only | Study IV |

---

## 4. Step-by-step process

### Step 0 — Freeze science layer

Lock RQ, RH1–RH4 (+ RH7 for transfer), models, metrics, vocabulary (`MASTER.md`, `claims.md`).

### Step 1 — Split manifest (ARC)

```bash
export PYTHONPATH=scripts:$PYTHONPATH
.venv/bin/python scripts/run.py splits \
  --dataset arc_challenge \
  --output analysis/splits.json
```

Policy: validation → CALIB; test → TEST. MMLU is test-only transfer eval.

### Step 2 — Configuration screening (pilots)

Small-\(n\) pilots before scaling:

- Prefill returns full-vocabulary logits at final prompt position.
- Oracle buckets non-degenerate (not 100% too-hard / 0% opportunity).

**Retained:** Llama 3.2 1B/3B Instruct on ARC.  
**Excluded (appendix):** Qwen 2.5 on ARC (0% opportunity pilot); Qwen on GSM8K (100% too-hard).

### Step 3 — Offline oracle (GPU)

For each query and each model: chat prompt → greedy generation → letter match → buckets.

```bash
.venv/bin/python scripts/run.py oracle \
  --weak meta-llama/Llama-3.2-1B-Instruct \
  --strong meta-llama/Llama-3.2-3B-Instruct \
  --dataset arc_challenge \
  --splits-json analysis/splits.json --split-role calib \
  --max-new-tokens 8 --device cuda --dtype bfloat16 \
  --output experiments/M4/routing_opportunity/arc_validation_oracle.json
# Repeat --split-role test → arc_test_oracle.json

# MMLU (RunPod example)
.venv/bin/python scripts/run.py oracle \
  --weak meta-llama/Llama-3.2-1B-Instruct \
  --strong meta-llama/Llama-3.2-3B-Instruct \
  --dataset mmlu --split test --limit 314 --seed 42 \
  --max-new-tokens 8 --device cuda --dtype bfloat16 \
  --output experiments/M4/routing_opportunity/mmlu_test_oracle.json
```

**Outputs:** `easy`, `opportunity`, `weak_only`, `too_hard`; \(y^{\text{opp}} = \mathbb{1}[\text{weak wrong} \land \text{strong right}]\).

### Step 4 — Model-independent features (CPU)

Tokenizer-only complexity candidates (no forward pass):

```bash
.venv/bin/python scripts/run.py features \
  --dataset arc_challenge \
  --splits-json analysis/splits.json --split-role calib \
  --output experiments/M5/arc_validation_features.csv
# TEST: arc_test_features.csv
```

**MMLU:** Regenerate features **from oracle JSON query IDs** (not the default subject loader) so `query_id` aligns with probes:

```python
# queries = oracle rows → run_feature_extraction(...)
# → experiments/M5/mmlu_test_features.csv
```

### Step 5 — D46 calibration (ARC CALIB only)

```bash
.venv/bin/python scripts/run.py screen \
  --features experiments/M5/arc_validation_features.csv \
  --oracle experiments/M4/routing_opportunity/arc_validation_oracle.json \
  --splits-json analysis/splits.json \
  --output analysis/d46_signal_screen_arc.json
```

**Winner:** `piece_count` → frozen in `analysis/selected_feature.json`. MMLU reuses this selection (test-only transfer).

### Step 6 — Prefill probes (GPU)

One forward pass per model; entropy \(H\) and margin \(m\) at final prompt position; derived \(\Delta H\), \(\Delta m_{\mathrm{gain}}\).

```bash
.venv/bin/python scripts/run.py probes \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dataset arc_challenge \
  --splits-json analysis/splits.json --split-role test \
  --device cuda --dtype bfloat16 --batch-size 8 \
  --output experiments/M5/arc_test_weak.csv
# Strong → arc_test_strong.csv; MMLU → mmlu_test_weak.csv / mmlu_test_strong.csv
```

### Step 7 — Merge + routing relevance (CPU)

```bash
.venv/bin/python scripts/run.py merge \
  --weak-csv experiments/M5/arc_test_weak.csv \
  --strong-csv experiments/M5/arc_test_strong.csv \
  --oracle experiments/M4/routing_opportunity/arc_test_oracle.json \
  --features-csv experiments/M5/arc_test_features.csv \
  --complexity-selection analysis/selected_feature.json \
  --output analysis/arc_routing_relevance.json \
  --merged-csv analysis/arc_merged.csv
```

MMLU: same pattern → `analysis/mmlu_merged.csv`, `analysis/mmlu_routing_relevance.json`.

### Step 8 — Doctor (pre-flight)

```bash
.venv/bin/python scripts/run.py doctor \
  --oracle experiments/M4/routing_opportunity/mmlu_test_oracle.json \
  --weak-csv experiments/M5/mmlu_test_weak.csv \
  --strong-csv experiments/M5/mmlu_test_strong.csv \
  --merged-csv analysis/mmlu_merged.csv \
  --features-csv experiments/M5/mmlu_test_features.csv \
  --complexity-selection analysis/selected_feature.json \
  --output analysis/mmlu_doctor.json
```

MMLU: **28/28 checks passed** after feature realignment.

### Step 9 — Interpretation bundle (CPU)

```bash
.venv/bin/python scripts/run.py interpret \
  --merged-csv analysis/arc_merged.csv \
  --features-csv experiments/M5/arc_test_features.csv \
  --splits-json analysis/splits.json \
  --output analysis/arc_interpretation.json
```

MMLU → `analysis/mmlu_interpretation.json`.

### Step 10 — Complementarity / Study III (ARC only)

Nested CALIB-fit / TEST-eval logistic ladder:

```bash
.venv/bin/python scripts/run.py complementarity \
  --merged-csv analysis/arc_merged_full.csv \
  --splits-json analysis/splits.json \
  --output analysis/arc_complementarity.json
```

### Step 11 — Routing evaluation / Study IV (ARC)

```bash
.venv/bin/python scripts/run.py route-eval \
  --merged-csv analysis/arc_merged_full.csv \
  --splits-json analysis/splits.json \
  --cost-lambda 0 \
  --output analysis/arc_route_eval_lambda0.json
```

### Step 12 — C2 screening summary (MMLU)

```bash
.venv/bin/python scripts/run.py summarize-c2 \
  --oracle experiments/M4/routing_opportunity/mmlu_test_oracle.json \
  --output analysis/c2_mmlu_summary.json
```

**Gate:** `gate_pass=true` (opportunity \(\geq 10\%\), not ~95% easy, not ~80% too-hard).

### Step 13 — RH7 dimension transfer (CPU)

```bash
.venv/bin/python scripts/run.py compare-generalization \
  --regime arc=analysis/arc_merged.csv \
  --regime mmlu=analysis/mmlu_merged.csv \
  --output analysis/C2_dimension_transfer.json
```

### Step 14 — Paper figures

```bash
.venv/bin/python scripts/run.py plot distributions \
  --merged-csv analysis/arc_merged.csv \
  --output paper/figures/F1_bucket_distributions.png
# roc, scatter, recovery-matrix, decomposition, etc.
```

---

## 5. Scientific unit: latent routing dimensions

| Latent dimension | Operationalization | Side |
| ---------------- | ------------------ | ---- |
| Task difficulty | `piece_count` / \(c(q)\) | Difficulty |
| Model uncertainty | \(H_w\) | Difficulty |
| Model disagreement | \(\Delta H = H_w - H_s\) | Escalation |
| Escalation potential | \(\Delta m_{\mathrm{gain}} = m_s - m_w\) | Escalation |

AUROC and \(\rho\) summarize detectability for a **dimension** — not a feature leaderboard.

---

## 6. Results — ARC primary (C0)

### Oracle landscape (TEST, \(n{=}1{,}172\))

| Bucket | Count | Rate |
| ------ | ----- | ---- |
| Easy | 304 | 25.9% |
| **Opportunity** | **507** | **43.3%** |
| Weak-only | 61 | 5.2% |
| Too-hard | 300 | 25.6% |

Weak accuracy 36.7%; strong 69.2%; oracle upper bound 74.4%.

### RH1 — Predictive content (Study I–II)

| Dimension | \(\rho_s\) | AUROC | AUPRC |
| --------- | ---------- | ----- | ----- |
| Task difficulty \(c(q)\) | +0.071 | 0.541 | 0.474 |
| Model uncertainty \(H_w\) | +0.138 | 0.581 | 0.500 |
| Model disagreement \(\Delta H\) | +0.174 | 0.602 | 0.559 |
| Escalation potential \(\Delta m_{\mathrm{gain}}\) | +0.179 | **0.605** | 0.547 |

**Finding:** Modest but real signal; best AUROC \(\sim 0.61\), far from random, far from perfect.

### RH2 — Distinct dimensions (opportunity vs too-hard)

| Dimension | Cohen's \(d\) |
| --------- | ------------- |
| \(c(q)\) | 0.09 |
| \(H_w\) | **0.03** |
| \(\Delta H\) | **0.82** |
| \(\Delta m_{\mathrm{gain}}\) | **0.72** |

**Recovery matrix:** Highest opportunity rate **57.2%** (\(n{=}353\)) in weak-uncertain / strong-rescues cell (median splits on \(H_w\), \(\Delta m_{\mathrm{gain}}\)).

**Finding:** **Difficulty** signals (query-derived, weak entropy) overlap; **recoverability** signals (cross-model \(\Delta m_{\mathrm{gain}}\)) separate routable from irrecoverable queries. Routing on \(H_w\) alone confuses opportunity with too-hard.

### RH3 — Complementarity (Study III)

| Model (CALIB fit → TEST AUROC) | AUROC [95% CI] |
| ------------------------------ | -------------- |
| \(c(q)\) only | 0.541 [0.509, 0.574] |
| \(c + H_w\) | 0.590 [0.558, 0.622] |
| \(c + H_w + m_w\) | 0.578 [0.546, 0.612] |

| Step | \(\Delta\)AUROC | DeLong \(p\) |
| ---- | ------------- | ------------ |
| \(c \to c + H_w\) | **+0.049** [+0.014, +0.085] | **0.008** |
| \(c + H_w \to\) joint (+ \(m_w\)) | −0.013 | 0.16 |

**Finding:** Partial complementarity between query complexity and weak entropy; adding margin does not help further on ARC.

### RH4 — Routing evaluation (Study IV, \(\lambda{=}0\))

| Policy | Accuracy | Avg. cost | Opp. recall (strong) |
| ------ | -------- | --------- | -------------------- |
| Always-weak | 31.1% | 1.00 | 0% |
| Always-strong | 69.2% | 3.00 | 100% |
| Calibrated policy | 69.2% | 3.00 | 100% |
| Oracle (bound) | 74.4% | 1.87 | 100% |

**Headroom:** 5.2 pp (oracle − always-strong); calibrated policy exploits **0 pp** at \(\lambda{=}0\).

**Finding:** Detection exceeds simple assignment — exploitation gap, not absence of information.

---

## 7. Results — MMLU transfer (C2 / RH7)

### Oracle landscape (TEST, \(n{=}314\))

| Bucket | Count | Rate |
| ------ | ----- | ---- |
| Easy | 53 | 16.9% |
| **Opportunity** | **81** | **25.8%** |
| Weak-only | 38 | 12.1% |
| Too-hard | 142 | 45.2% |

Harder than ARC (45% too-hard vs 26%), but C2 gate passes.

### Dimension detectability (MMLU TEST)

| Dimension | \(\rho_s\) | AUROC | Cohen's \(d\) (opp vs too-hard) |
| --------- | ---------- | ----- | ------------------------------- |
| \(c(q)\) | +0.115 | 0.576 | 0.44 |
| \(H_w\) | +0.085 | 0.556 | 0.18 |
| \(\Delta H\) | +0.197 | 0.630 | 0.78 |
| \(\Delta m_{\mathrm{gain}}\) | +0.195 | 0.629 | **0.72** |

### RH7 — Cross-regime pattern match

| Regime | \(d(\Delta m_{\mathrm{gain}})\) | \(d(H_w)\) | Verdict |
| ------ | ------------------------------- | ---------- | ------- |
| ARC | 0.72 | 0.03 | matches template |
| MMLU (pooled) | 0.72 | 0.18 | matches template |
| MMLU / US history | 0.89 | 0.29 | strong escalation |
| MMLU / physics | 0.38 | 0.14 | weaker |
| MMLU / abstract algebra | −0.21 | −0.17 | pattern break |

**Pooled summary (from `C2_dimension_transfer.json`):**

> Escalation potential separates routable from irrecoverable queries on all regimes; weak-model uncertainty does not — despite different absolute accuracies.

**Finding:** The **difficulty vs recoverability** structure **transfers at the pooled level**. Subject choice matters; math-heavy abstract algebra with \(n{=}74\) does not replicate the pattern.

### MMLU-specific notes

1. **Features must match oracle `query_id`s** — IDs use `mmlu_{subject}_test_{idx}`; regenerating from the default loader caused a merge failure (wrong subjects).
2. **D46 frozen on ARC** — \(c(q)\) stronger on MMLU (\(d{=}0.44\)) because long history passages inflate `piece_count`; still secondary to escalation signals.
3. **No MMLU routing eval** — \(\tau\), \(\lambda\) from ARC CALIB only; MMLU is pattern transfer, not a second leaderboard.

---

## 8. Three reader-facing findings (paper spine)

### Finding 1 — Information exists

Before generation, routing-relevant information is measurable. On ARC TEST, 43.3% of queries are routing opportunities; best dimension AUROC \(\sim 0.61\). On MMLU, 25.8% opportunity with AUROC up to \(\sim 0.63\).

### Finding 2 — Dimensions differ (and partially transfer)

Information is structured: **difficulty** (query-derived complexity, weak entropy) vs **recoverability** (cross-model disagreement, margin gain). On ARC, \(H_w\) barely separates opportunity from too-hard (\(d \approx 0.03\)); \(\Delta m_{\mathrm{gain}}\) does (\(d \approx 0.72\)). The same recoverability \(d\) appears on pooled MMLU.

### Finding 3 — Exploitation gap (ARC)

A CALIB-fit calibrated policy matches always-strong (69.2%) while an offline oracle reaches 74.4% — **5.2 pp** unexploited on TEST. Characterization \(\neq\) product router.

---

## 9. Infrastructure notes (reproducibility)

| Environment | Work | Notes |
| ----------- | ---- | ----- |
| **GPU (RunPod)** | Oracle + probes | `export PYTHONPATH=/workspace/llm_routing/scripts:$PYTHONPATH`; HF token for Llama weights |
| **Mac CPU** | Features, merge, interpret, doctor, compare-generalization | No GPU needed post-copy |
| **Copy from pod** | `scp` oracle + probe CSVs | Quote remote globs for zsh |

---

## 10. What this study does not claim

- A new supervised router or SOTA routing benchmark win.
- That weak entropy alone should drive escalation.
- Uniform transfer to every MMLU subject (abstract algebra is a counterexample).
- Agent routing, Qwen architecture transfer (C1), or paraphrase stability — future work.
- **C3 layerwise extension** — in progress (`c3_layerwise_concepts.md`, `c3_prefill_extensions_plan.md`); richer model-derived characterization, not a new information source.

---

## 11. Artifact index

| Artifact | Purpose |
| -------- | ------- |
| `analysis/splits.json` | ARC CALIB/TEST manifest |
| `analysis/selected_feature.json` | Frozen \(c(q)\) = piece_count |
| `experiments/M4/routing_opportunity/*.json` | Offline oracles |
| `experiments/M5/*_{weak,strong,features}.csv` | Probes + complexity |
| `analysis/arc_merged.csv` | ARC master table (\(n{=}1{,}172\)) |
| `analysis/mmlu_merged.csv` | MMLU master table (\(n{=}314\)) |
| `analysis/arc_interpretation.json` | RH2 diagnostics (ARC) |
| `analysis/mmlu_interpretation.json` | RH2 diagnostics (MMLU) |
| `analysis/arc_complementarity.json` | RH3 ladder |
| `analysis/arc_route_eval_lambda0.json` | RH4 policies |
| `analysis/C2_dimension_transfer.json` | RH7 cross-regime table |
| `analysis/c2_mmlu_summary.json` | MMLU screening gate |
| `paper/tables/T2_correlation.tex` | RH1–RH2 |
| `paper/tables/T3_complementarity.tex` | RH3 |
| `paper/tables/T4_routing.tex` | RH4 |

---

## 12. One-paragraph synthesis (Abstract / Discussion)

On ARC-Challenge with Llama 3.2 1B/3B Instruct, unsupervised pre-inference signals carry modest but measurable routing-relevant information before full generation. **Difficulty and recoverability are distinct:** query-derived complexity and weak entropy track difficulty-like variation, while cross-model disagreement and margin gain separate routable from irrecoverable hard queries (Cohen's \(d \approx 0.72\) for \(\Delta m_{\mathrm{gain}}\) vs \(0.03\) for \(H_w\) on ARC). The three information sources are partially complementary for opportunity detection (\(\Delta\)AUROC \(+0.049\), DeLong \(p{=}0.008\)). A calibrated logistic policy does not beat always routing to the strong model on ARC TEST, leaving 5.2 percentage points of oracle headroom unexploited. **Transfer to MMLU** preserves the recoverability pattern at the pooled level (again \(d \approx 0.72\) for \(\Delta m_{\mathrm{gain}}\)) while subject-level variation warns against treating transfer as automatic. The answer to whether pre-inference signals can **support** routing is **partial**: information exists, is structured, and partially generalizes — but simple exploitation and universal cross-subject invariance are limited.
