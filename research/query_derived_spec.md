# Query-derived representation — concepts and implementation

> **Architecture frozen** · manifest: [`experiments/query_derived_defaults.yaml`](../experiments/query_derived_defaults.yaml)  
> **Code:** [`llm_routing/query_derived/`](../llm_routing/query_derived/) · **Program:** [`program.md`](program.md) §5  
> **Layer:** **model-independent** (query-derived signals) — **H1**; no **routing-pool** forward pass at extraction (fixed encoder ∉ \(\mathcal{P}\) is allowed).

---

## Part A — Scientific framing

### The right question

Do not ask *"What features can I compute?"* Ask:

> **What properties of a query suggest \(M_{\mathrm{hi}}\) (not \(M_{\mathrm{lo}}\)) is the appropriate pool member?**

Each property is a **testable hypothesis** about positive oracle \(r(q)=1\) (appropriate-model label; program §3).

### Representation

**Scalar blocks in \(\phi(q)\)** (written to `query_derived.jsonl`):

\[
\phi(q) = \big[\,\phi_{\text{structural}},\; \phi_{\text{ambiguity}},\; \phi_{\text{embedding\_geometry}}\,\big]
\]

**Semantic representation** \(u(q)\) is stored separately (`signals/embeddings/{id}.npy`) and is **not** a column of \(\phi(q)\). Corpus-relative geometry scalars \(\phi_{\text{embedding\_geometry}}(q)\) are **derived from** \(u(q)\) after fit on \(R_c\).

| Property | Hypothesis | What it measures |
| -------- | ---------- | ---------------- |
| **\(\phi_{\text{structural}}\)** | Higher processing load → more likely to exceed \(M_{\mathrm{lo}}\) capacity | Prompt size, option stats, lexical diversity / density |
| **\(\phi_{\text{ambiguity}}\)** | Harder-to-distinguish options → more error-prone for weaker models | MCQ stem–choice and choice–choice overlap |
| **\(u(q)\)** (artifact) | Semantic content positions the query in task space | Frozen sentence encoder output (not in jsonl) |
| **\(\phi_{\text{embedding\_geometry}}\)** | Corpus-unusual queries behave differently | Geometry of \(u(q)\) **relative to \(R_c\)** (PCA, density, outliers) |

**Not generic difficulty. Not reasoning depth.** Each block targets one property linked to oracle \(r(q)\) (appropriate model choice), not whether the task is hard in the abstract.

### Conceptual pipeline (paper figure)

```text
Canonical prompt
        │
        ▼
Structural descriptors
        │
        ▼
Ambiguity descriptors
        │
        ▼
Semantic representation u(q)     → stored separately (.npy)
        │
        ▼
Calibration-only engineering   (fit on R_c only)
        │
        ▼
Embedding-geometry descriptors → part of φ(q) in jsonl
        │
        ▼
φ(q)
```

**Critical distinction:** \(u(q)\) is **extraction** (artifact on disk); \(\phi_{\text{embedding\_geometry}}\) is **corpus-relative engineering** derived from \(u(q)\). Do not treat the raw embedding as a jsonl column of \(\phi(q)\).

### Model-independent vs supervised routers

| | Query-derived (this work) | RouteLLM / Hybrid LLM |
|--|---------------------------|------------------------|
| Input | Frozen \(\phi(q)\) | Learned query encoder |
| Labels at extraction | **No** | Yes |
| Pool forward pass | **No** | N/A |

### Leakage rules

| Stage | Uses \(R_c\)? | Uses \(r(q)\)? |
| ----- | ------------- | --------------- |
| Extraction (load, ambiguity, semantic) | No | No |
| Engineering (embedding-geometry) | **Fit only** | No |
| Z-score (implementation) | \(\mu,\sigma\) only | No |
| Analysis (Stage 6) | Yes | **Yes** |

---

## Part B — Feature inventory

### \(\phi_{\text{structural}}\) — processing load (7 scalars in jsonl)

*How much information must the model process?*

| Feature | Role | Stage 7 prior |
|---------|------|---------------|
| `prompt_token_len` | Full canonical user prompt — actual load on \(M_{\mathrm{lo}}\) tokenizer | **Keep** |
| `question_token_len` | Stem only; complements prompt length | **Keep** |
| `mean_option_token_len` | Average choice length | **Keep** |
| `std_option_token_len` | Choice length spread | **Evaluate** (low variance on ARC) |
| `question_option_ratio` | `question_token_len / sum(option_token_lens)` | **Evaluate** (may correlate with lengths) |
| `mattr` | Lexical diversity (moving TTR) | **Keep** |
| `compression_ratio` | Text redundancy / compressibility (zlib) | **Keep** |

`option_count` is **not extracted** (constant at 4 on ARC MCQ).

### \(\phi_{\text{ambiguity}}\) — MCQ distinguishability (4 scalars)

*How separable are the answer options?*

| Feature | Role | Stage 7 prior |
|---------|------|---------------|
| `stem_choice_overlap_max` | Max word-Jaccard(stem, choice) | **Evaluate** (may duplicate mean) |
| `stem_choice_overlap_mean` | Mean stem–choice overlap | **Keep** |
| `choice_choice_overlap` | Mean pairwise choice overlap | **Keep** |
| `choice_length_range` | Char-length spread among choices | **Evaluate** |

### Semantic representation \(u(q)\) — frozen embedding (artifact)

*Where does this query lie in semantic space?*

- **No PCA, no clustering at this step** — just \(u(q)\) from a **frozen sentence encoder**.
- Stored per query: `signals/embeddings/{query_id}.npy`
- **Not** part of \(\phi(q)\) in jsonl (too large; enables re-engineering without re-encoding).
- Paper: "a frozen sentence encoder"; implementation: MiniLM in defaults yaml.

### \(\phi_{\text{embedding\_geometry}}\) — corpus-relative descriptors (6 scalars)

*How unusual is this query vs calibration corpus \(R_c\)?* Derived from \(u(q)\); semantic content enters scalar \(\phi(q)\) here.

Fit on **\(R_c\) embeddings only**; apply to calib + test. **ACL simplification (locked):** one local-density signal (`mean_knn_similarity` only — `knn_distance` dropped as redundant with \(1 - \text{similarity}\)).

| Feature | Role |
|---------|------|
| `pc1`, `pc2`, `pc3` | PCA coordinates on calib embedding manifold |
| `centroid_distance` | Cosine distance from calib mean embedding |
| `mean_knn_similarity` | Mean \((1 - \text{cosine dist})\) to \(k\) NN in \(R_c\) |
| `lof_score` | LOF outlier score (normalized vs \(R_c\)) |

**Mean kNN similarity** = typical vs rare question style in calib space (high = dense region; low = sparse / unusual).

Saved fit: `signals/engineering/embedding_geometry_model.json`

### Interpretation notes (locked — extraction v1)

| Component | Verdict | Paper / analysis guidance |
| --------- | ------- | ------------------------- |
| Canonical prompt + tokenizer stats | ✅ Freeze | Counts reflect exact user message the pool model receives |
| MATTR | ✅ Freeze | Prefer over plain TTR (length-stable) |
| Ambiguity (Jaccard) | ✅ Freeze | Simple, deterministic; tests lexical distinguishability |
| MiniLM \(u(q)\) + geometry on \(R_c\) | ✅ Freeze | Strongest H1 block; fit geometry on calib only |
| Z-score on \(R_c\) | ✅ Freeze | For joint evaluation models and Stage 8; not required for univariate AUROC |
| `compression_ratio` | ✅ Keep | Describe as **redundancy/compressibility**, not complexity |
| `pc1`–`pc3` | ✅ Keep | Describe as **manifold coordinates**, not “higher PC1 = harder” |
| `lof_score` | ✅ Keep | Check tail outliers at Stage 6; clip at analysis if needed |
| `option_count` | ❌ Removed | Not extracted (constant on ARC) |

**Stage 7 pruning rule (locked):** do **not** drop features before Stage 6. After univariate AUROC / AUPRC / Spearman on \(R_c\), remove only with evidence: AUROC \(\approx 0.5\), Spearman \(\approx 0\), or zero variance. Geometry redundancy (`knn_distance` vs `mean_knn_similarity`) is resolved at extraction — only `mean_knn_similarity` is kept.

**Prior tiers (hypothesis guide, not pre-deletion):**
- **~11 keep:** load block (5) + `stem_choice_overlap_mean`, `choice_choice_overlap` + geometry (`centroid_distance`, `mean_knn_similarity`, `lof_score`, `pc1`–`pc3`)
- **~4 evaluate:** `std_option_token_len`, `question_option_ratio`, `stem_choice_overlap_max`, `choice_length_range`

**Future (not v1):** `max(option_tokens) / question_tokens` or `mean(option_tokens) / question_tokens` as alternatives to sum-based `question_option_ratio`.

---

## Part C — Step-by-step implementation

### Step 0 — Load context

`run_query_derived(run_root)` reads setting, corpus, partition, defaults.  
Query IDs = `calib ∪ test` (or full corpus if M3 not run).  
**Does not read** oracle or \(r(q)\).

### Step 1 — Canonicalize

`canonical_user()` → exact user message the LLM sees (`render_user_message`).

### Step 2 — Load extraction

`extract_load()` = `extract_structural()` + `extract_lexical()`.

Token counts via `pool.M_lo` HF tokenizer (count only) or regex fallback.

### Step 3 — Ambiguity extraction

`extract_ambiguity()` — word Jaccard on stem and choices.

### Step 4 — Semantic extraction

`encode_canonical_texts()` batch-encodes canonical strings.  
Saves `signals/embeddings/{query_id}.npy`.

### Step 5 — Embedding-geometry engineering (\(R_c\) only)

If \(|R_c| \ge 2\): `GeometryModel.fit(calib_embeddings)` then `.transform(u(q))` for every query.

If \(|R_c| < 2\): `embedding_geometry: {}`.

### Step 6 — Z-score (locked default for Stages 6–9)

`ZScoreModel` fits \(\mu,\sigma\) on **\(R_c\) only**; applies the **same** transform to calib and test rows. Statistics are saved to `signals/engineering/zscore.json` and must not be recomputed on \(R_t\).

**Rules (no leakage):**

1. Fit mean and std on the calibration split (\(R_c\)) only.
2. Apply unchanged to both calib and test.
3. Do not recompute scaling on the test set.

**What gets z-scored:** all continuous H1 scalars in `zscore.continuous_keys` (see manifest). After scaling, each value is “how many calib standard deviations from the calib mean” — comparable across token counts, Jaccard overlaps, embedding distances, and PCA coordinates.

**Excluded:** nothing at extraction; `option_count` not stored (constant on ARC).

| Feature group | Z-score? |
| ------------- | -------- |
| structural (7) | Yes |
| ambiguity (4) | Yes |
| embedding_geometry (6) | Yes |

**Stage 6 nuance:** AUROC, AUPRC, and Spearman \(\rho\) are **invariant** to monotone transforms such as z-scoring (ranking unchanged) — **no scaling is required for univariate analysis** on any layer (e.g. MSP \(0.70\) vs its z-score \(-1.12\) yield identical AUROC). Z-scoring \(\phi\) is still the correct default for **evaluation models** (Stage 6 multivariate), coefficient comparability, numerical stability, and **Stage 8** routing-policy learning. \(\psi\) and \(\chi\) metrics stay in raw protocol units unless a later stage explicitly scales them for joint models.

**v1 default:** standard z-score for all continuous keys above. If a feature is pathologically skewed (`compression_ratio`, overlap features, `lof_score`), a robust scaler (median/IQR) is a future option — not required for v1 unless diagnostics show failure.

**Paper figure:** z-score is an implementation step between engineering and \(\phi(q)\) in jsonl; not part of the conceptual extraction diagram.

### Step 7 — Write outputs

```text
signals/query_derived.jsonl       # structural, ambiguity, embedding_geometry per query
signals/query_derived_meta.json   # semantic artifact paths
signals/embeddings/{id}.npy       # u(q) — not a jsonl column of φ(q)
signals/engineering/embedding_geometry_model.json
signals/engineering/zscore.json
```

**Example jsonl record:**

```json
{
  "query_id": "ARC-Challenge:validation:0",
  "structural": {
    "prompt_token_len": 42,
    "question_token_len": 12,
    "mean_option_token_len": 3.5,
    "std_option_token_len": 1.2,
    "question_option_ratio": 0.86,
    "mattr": 0.71,
    "compression_ratio": 0.48
  },
  "ambiguity": {
    "stem_choice_overlap_max": 0.15,
    "stem_choice_overlap_mean": 0.08,
    "choice_choice_overlap": 0.05,
    "choice_length_range": 12
  },
  "embedding_geometry": {
    "pc1": -0.31,
    "pc2": 0.12,
    "pc3": 0.04,
    "centroid_distance": 0.22,
    "mean_knn_similarity": 0.82,
    "lof_score": -0.5
  }
}
```

---

## Part D — How to run

```bash
python run.py eval --run experiments/runs/<id>
python run.py model-independent --run experiments/runs/<id> --mock-embed   # local smoke
python run.py model-independent --run experiments/runs/<id>                 # real encoder
```

---

## Part E — Stage 6: signal analysis (frozen methodology)

Answer **which signal representation carries information about** \(r(q)\). Stage 6 **evaluates informativeness** — it does **not** build the routing policy (Stage 8) or select features (Stage 7).

**Unit of analysis:** a **representation** (feature vector), not an individual scalar. H1 tests structural, ambiguity, and embedding-geometry blocks individually and combined \(\phi(q)\).

### Paper structure (locked)

| Section | Content |
| ------- | ------- |
| **6.1 Representation tests (primary)** | 5-fold stratified CV logistic per representation vs.\ \(r(q)\) on \(R_c\); Table T2 |
| **6.2 Scalar diagnostics (secondary)** | Per-feature Spearman \(\rho\), AUROC, AUPRC, class means — appendix / Stage 7 input |

### Analysis order (locked)

```text
analysis_table.csv
    ↓
representation_tests.json   ← PRIMARY (joint CV per representation → T2)
    ↓
univariate_h{1,2,3}.jsonl     ← SECONDARY (scalar diagnostics, no ranks)

Stage 7 (separate):
    univariate + analysis_table → ranking, Pearson redundancy, prune, freeze x(q)
```

1. **Representation tests (primary)** — 5-fold stratified CV logistic on each representation: `query_structural`, `query_ambiguity`, `query_geometry`, `query_combined`, `model_response`, `cross_model`. Output: `representation_tests.json` keyed by `representation_id`.
2. **Scalar diagnostics (secondary)** — per scalar: Spearman \(\rho\), **raw** AUROC, AUPRC, `direction`, `mean_r0`, `mean_r1`. **No** ranks, **no** `selection_metric`, **no** correlations in Stage 6.
3. **Table T2** — one row per representation; columns: joint CV AUROC, joint CV AUPRC.

**Do not prune features in Stage 6.** Stage 7 drops features after ranking and redundancy evidence.

### Master analysis table (required artifact)

Write `signals/analysis/analysis_table.csv` on \(R_c\) — one row per query, columns:

- `query_id`, `r`, `bucket` (easy / opportunity / lo\_only / too\_hard)
- hierarchical feature columns: `phi.<block>.<key>`, `psi.<key>`, `chi.<key>` (static schema from Stage 5 / protocol registries — not inferred from data rows)

### Stage 6 outputs

```text
signals/analysis/
  analysis_table.csv              # master join (required)
  representation_tests.json       # PRIMARY → Table T2
  univariate_query.jsonl          # SECONDARY diagnostic scalars (query / φ)
  univariate_response.jsonl       # model-response ψ
  univariate_cross.jsonl          # cross-model χ
  signal_analysis_meta.json       # manifest (schema, seed, dataset, artifact list)
```

`signal_analysis_meta.json` is the Stage~6 manifest: `schema_version`, `dataset=calib`, `seed`, `n_folds`, `evaluator`, and an explicit `artifacts` array listing every generated file.

Joint CV results in `representation_tests.json` include `fold_scores.auroc` / `fold_scores.auprc` per fold (mean/std derived; enables CIs and plots without reruns).

### Univariate row schema (diagnostic only)

| Field | Role |
| ----- | ---- |
| `spearman_rho`, `spearman_p` | Rank association vs.\ \(r(q)\) |
| `auroc` | Raw sklearn AUROC |
| `direction` | `positive` / `negative` |
| `auprc` | Raw AUPRC |
| `mean_r0`, `mean_r1` | Class-conditional means |
| `signal_layer`, `representation_id`, `block`, `feature` | Provenance |

Ranking (`auroc_abs`, `rank_by_*`), Pearson redundancy, and `top_feature` belong in **Stage 7** (`llm_routing/signal_selection/`).

### Table T2 (representation rows)

| Representation | Joint CV AUROC | Joint CV AUPRC |
| -------------- | -------------- | -------------- |
| H1 structural | … | … |
| H1 ambiguity | … | … |
| H1 embedding-geometry | … | … |
| H1 all φ | … | … |
| H2 ψ | … | … |
| H3 χ | … | … |

Combination models (\(\phi{+}\psi\), \(\phi{+}\psi{+}\chi\)) support Stage 7 selection, **not** H2 or H3 definitions.

Metrics: bucket stratification optional. No \(\pi(q)\) or accuracy–cost claims.

---

## Part F — Code map

```text
llm_routing/query_derived/
  __init__.py    # public API (lazy re-exports)
  core.py        # manifest, φ_structural, φ_ambiguity, u(q) encode, φ_embedding_geometry, z-score
  run.py         # run_query_derived()
```

| Property | Module |
| -------- | ------ |
| Structural, Ambiguity, \(u(q)\), Embedding-geometry | `core.py` |
| Orchestration | `run.py` |
| CLI stage | `llm_routing/pipeline.py` → `stage_model_independent()` |

---

## Part G — Configuration

File: `experiments/query_derived_defaults.yaml`

| Key | Role |
|-----|------|
| `load.mattr_window` | MATTR window |
| `load.zlib_level` | Compression level |
| `embedding_geometry.pca_components` | PC count |
| `embedding_geometry.knn_k` | NN count for density / LOF |
| `embedding.model_id` | Sentence encoder (implementation) |
| `tokenizer.source` | HF id for token counts |
| `zscore.continuous_keys` | Features to normalize |

Lock at M3 after smoke validation.
