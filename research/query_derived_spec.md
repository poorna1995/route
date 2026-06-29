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

\[
\phi(q) = \big[\,\phi_{\text{load}},\; \phi_{\text{ambiguity}},\; \phi_{\text{semantic}},\; \phi_{\text{novelty}}\,\big]
\]

| Property | Hypothesis | What it measures |
| -------- | ---------- | ---------------- |
| **\(\phi_{\text{load}}\)** | Higher processing load → more likely to exceed \(M_{\mathrm{lo}}\) capacity | Prompt size, option stats, lexical diversity / density |
| **\(\phi_{\text{ambiguity}}\)** | Harder-to-distinguish options → more error-prone for weaker models | MCQ stem–choice and choice–choice overlap |
| **\(\phi_{\text{semantic}}\)** | Semantic content positions the query in task space | Frozen sentence encoder \(u(q)\) |
| **\(\phi_{\text{novelty}}\)** | Corpus-unusual queries behave differently | Descriptors **relative to \(R_c\)** (PCA, density, outliers) |

**Not generic difficulty. Not reasoning depth.** Each block targets one property linked to oracle \(r(q)\) (appropriate model choice), not whether the task is hard in the abstract.

### Conceptual pipeline (paper figure)

```text
Canonical prompt
        │
        ▼
Load descriptors
        │
        ▼
Ambiguity descriptors
        │
        ▼
Frozen semantic embedding u(q)
        │
        ▼
Calibration-only engineering
        │
        ▼
Novelty descriptors
        │
        ▼
φ(q)
```

**Critical distinction:** semantic embedding is **extraction**; PCA / kNN / LOF are **corpus-relative engineering** → novelty. Do not collapse them into one "geometry" block.

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
| Engineering (novelty) | **Fit only** | No |
| Z-score (implementation) | \(\mu,\sigma\) only | No |
| Analysis (Stage 6) | Yes | **Yes** |

---

## Part B — Feature inventory

### \(\phi_{\text{load}}\) — processing load (8 scalars in jsonl)

*How much information must the model process?*

| Feature | Role |
|---------|------|
| `prompt_token_len` | Full canonical prompt length |
| `question_token_len` | Stem length |
| `option_count` | Number of choices (not z-scored) |
| `mean_option_token_len` | Average choice length |
| `std_option_token_len` | Choice length spread |
| `question_option_ratio` | Stem vs total choice tokens |
| `mattr` | Lexical diversity (moving TTR) |
| `compression_ratio` | Information density (zlib / raw) |

### \(\phi_{\text{ambiguity}}\) — MCQ distinguishability (4 scalars)

*How separable are the answer options?*

| Feature | Role |
|---------|------|
| `stem_choice_overlap_max` | Max word-Jaccard(stem, choice) |
| `stem_choice_overlap_mean` | Mean stem–choice overlap |
| `choice_choice_overlap` | Mean pairwise choice overlap |
| `choice_length_range` | Char-length spread among choices |

### \(\phi_{\text{semantic}}\) — frozen embedding (artifact)

*Where does this query lie in semantic space?*

- **No PCA, no clustering at this step** — just \(u(q)\) from a **frozen sentence encoder**.
- Stored per query: `signals/embeddings/{query_id}.npy`
- **Not** written as jsonl columns (too large; enables re-engineering without re-encoding).
- Paper: "a frozen sentence encoder"; implementation: MiniLM in defaults yaml.

### \(\phi_{\text{novelty}}\) — corpus-relative descriptors (7 scalars)

*How unusual is this query vs calibration corpus \(R_c\)?*

Fit on **\(R_c\) embeddings only**; apply to calib + test.

| Feature | Role |
|---------|------|
| `pc1`, `pc2`, `pc3` | PCA projection of \(u(q)\) in calib subspace |
| `centroid_distance` | Cosine distance from calib mean embedding |
| `knn_distance` | Mean cosine distance to \(k\) NN in \(R_c\) |
| `retrieval_density` | \(\frac{1}{k}\sum_i \cos(u(q), u_i)\) over \(k\) NN in \(R_c\) |
| `lof_score` | Local outlier factor (z-scored vs \(R_c\)) |

**Retrieval density** interprets *common vs rare question styles*: high density = typical calib region; low density = sparse / unusual.

Saved fit: `signals/engineering/novelty_model.json`

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

### Step 5 — Novelty engineering (\(R_c\) only)

If \(|R_c| \ge 2\): `NoveltyModel.fit(calib_embeddings)` then `.transform(u(q))` for every query.

If \(|R_c| < 2\): `novelty: {}`.

### Step 6 — Z-score (implementation detail)

`ZScoreModel` fits \(\mu,\sigma\) on \(R_c\) for continuous keys; applies to all rows.  
**Not part of the conceptual paper figure** — normalization for Stage 8 logistic regression.

`option_count` excluded (discrete).

### Step 7 — Write outputs

```text
signals/query_derived.jsonl       # load, ambiguity, novelty per query
signals/query_derived_meta.json   # semantic artifact paths
signals/embeddings/{id}.npy       # φ_semantic
signals/engineering/novelty_model.json
signals/engineering/zscore.json
```

**Example jsonl record:**

```json
{
  "query_id": "ARC-Challenge:validation:0",
  "load": {
    "prompt_token_len": 42,
    "question_token_len": 12,
    "option_count": 4,
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
  "novelty": {
    "pc1": -0.31,
    "pc2": 0.12,
    "pc3": 0.04,
    "centroid_distance": 0.22,
    "knn_distance": 0.18,
    "retrieval_density": 0.82,
    "lof_score": -0.5
  }
}
```

---

## Part D — How to run

```bash
python run.py lock-eval --run experiments/runs/<id>
python run.py query-derived --run experiments/runs/<id> --mock-embed   # local smoke
python run.py query-derived --run experiments/runs/<id>                 # real encoder
```

---

## Part E — Stage 6: block importance (planned)

Answer **which information source contributes**, not only which scalar feature matters.

### Single-property models

| Model | Features |
|-------|----------|
| **Load only** | \(\phi_{\text{load}}\) (8) |
| **Ambiguity only** | \(\phi_{\text{ambiguity}}\) (4) |
| **Semantic only** | \(u(q)\) from `.npy` (or PCA of \(u\) without other novelty dims) |
| **Novelty only** | \(\phi_{\text{novelty}}\) (7) |

### Combination models

| Model | Features |
|-------|----------|
| Load + Ambiguity | 12 scalars |
| Load + Semantic | load + \(u(q)\) |
| Load + Novelty | load + novelty |
| Ambiguity + Semantic | ambiguity + \(u(q)\) |
| … | all pairwise / triple / **All** |

Metrics: AUROC, AUPRC, Brier vs oracle \(r(q)\) (appropriate model); bucket stratification over easy / opportunity / lo\_only / too\_hard.

### Parallel layer tests H1, H2, H3 (program §10)

Each block tested **alone** vs \(r(q)\) — not nested ΔAUROC over the previous layer.

| Model | Layer | Hypothesis |
|-------|-------|------------|
| B1 | $\phi(q)$ only | **H1** |
| B2 | $\psi(q,M_{\mathrm{lo}})$ only | **H2** |
| B3 | $\chi(q)$ only | **H3** |

Combination models ($\phi{+}\psi$, $\phi{+}\psi{+}\chi$) support $x(q)$ selection (Stage 7), **not** the H2 or H3 layer hypotheses (ψ alone, χ alone).

---

## Part F — Code map

```text
llm_routing/query_derived/
  __init__.py    # public API (lazy re-exports)
  core.py        # manifest, φ_load, φ_ambiguity, φ_semantic, φ_novelty, z-score
  run.py         # run_query_derived()
```

| Property | Module |
| -------- | ------ |
| Load, Ambiguity, Semantic, Novelty | `core.py` |
| Orchestration | `run.py` |
| CLI stage | `llm_routing/pipeline.py` → `stage_query_derived()` |

---

## Part G — Configuration

File: `experiments/query_derived_defaults.yaml`

| Key | Role |
|-----|------|
| `load.mattr_window` | MATTR window |
| `load.zlib_level` | Compression level |
| `novelty.pca_components` | PC count |
| `novelty.knn_k` | NN count for density / LOF |
| `embedding.model_id` | Sentence encoder (implementation) |
| `tokenizer.source` | HF id for token counts |
| `zscore.continuous_keys` | Features to normalize |

Lock at M3 after smoke validation.
