# ACL implementation plan — geometry-first routing study

> **Purpose:** Operational plan for reframing Paper 1 from “another router” to **pre-decoding routing-state geometry**, with optional minimal replication (Qwen) and one novel signal (confidence dynamics).  
> **Companion:** `acl_scientific_strategy.md` (why) · `claims.md` (vocabulary) · `paper_process_and_results.md` (locked numbers).  
> **Last updated:** 2026-06-25

---

## 0. North star

**Scientific question (paper spine):**

> What is the structure of the latent routing state before decoding, and which pre-decoding observables operationalize it?

**Evaluation order (non-negotiable in write-up):**

```text
Observable signals  →  State estimation (axes / geometry)  →  Opportunity detection  →  Routing (validation only)
```

**Success criterion that does NOT require routing gains:**

- Two-axis model (difficulty vs escalation) is **empirically supported** on ARC TEST.
- Geometry is **visualized** (2D state plot) and **tested** (factor analysis / separation metrics).
- Optional: phenomenon **replicates** on one non-Llama family OR one non-ARC regime.
- Optional: one **dynamic** pre-decoding signal is screened; result reported honestly even if null.

---

## 1. Prerequisites checklist

### 1.1 Frozen science (already done — do not re-litigate)

| Artifact | Path | Status |
| -------- | ---- | ------ |
| Vocabulary SSOT | `research/claims.md` | Locked |
| RH1–RH4 + outcomes | `research/02_research_hypotheses.md`, `paper_process_and_results.md` | Locked |
| D46 winner | `analysis/selected_feature.json` → `piece_count` | Frozen |
| Splits | `analysis/splits.json` | Frozen |
| Paper reframe (draft) | `paper/draft/*.tex`, `paper/tables/T_dimensions.tex` | Draft exists |

### 1.2 Artifacts you must have locally before Tier A analysis

| Artifact | Expected path | How to obtain |
| -------- | ------------- | ------------- |
| Merged TEST table | `analysis/arc_merged.csv` | `python scripts/run.py merge` (after probes + features) |
| Weak/strong probe CSVs | `experiments/M5/*_probes.csv` | `python scripts/run.py probes` |
| Oracle JSON | `experiments/M4/oracle_*.json` | `python scripts/run.py oracle` |
| Features CSV | `experiments/M5/features.csv` | `python scripts/run.py features` |
| Interpret bundle | `analysis/interpret_*.json` | `python scripts/run.py interpret` |

**Gate:** If `analysis/arc_merged.csv` is missing, run the full Paper 1 pipeline once before geometry scripts. Locked numbers in docs assume a completed ARC Llama 1B/3B run.

### 1.3 Environment

```bash
# From repo root
uv sync
uv sync --extra analysis   # scipy, sklearn, matplotlib — required for geometry + plots
```

| Requirement | Minimum | Recommended |
| ----------- | ------- | ----------- |
| Python | 3.11+ | match `pyproject.toml` |
| GPU | 1× CUDA, ≥16 GB VRAM (3B probes) | 24 GB for 7B if Tier B uses Qwen 7B |
| Disk | ~30 GB (Llama 1B+3B HF cache) | +15 GB per extra model family |
| HF access | `huggingface-cli login` if gated models | Token in env `HF_TOKEN` |
| LaTeX | `pdflatex` + `bibtex` for paper build | `tectonic` acceptable |

### 1.4 Models (by tier)

| Tier | Weak | Strong | Notes |
| ---- | ---- | ------ | ----- |
| **Primary (locked)** | `meta-llama/Llama-3.2-1B-Instruct` | `meta-llama/Llama-3.2-3B-Instruct` | All headline numbers |
| **B1 replication** | `Qwen/Qwen2.5-1.5B-Instruct` | `Qwen/Qwen2.5-3B-Instruct` | Prior pilot: 0% opp on ARC n=10 — **screen before full run** |
| **B1 alt** | `Qwen/Qwen2.5-3B-Instruct` | `Qwen/Qwen2.5-7B-Instruct` | Larger gap; more VRAM |
| **B2 transfer** | Same Llama pool | Same | Dataset change only |

### 1.5 Datasets (by tier)

| Tier | Dataset | Loader status | Split policy |
| ---- | ------- | ------------- | ------------ |
| **Primary** | ARC-Challenge | `datasets.py` ✓ | val→CALIB, test→TEST |
| **B2** | MMLU (2 subjects) | **Not implemented** — add loader | subject holdout or official dev/test |
| **Deferred** | GSM8K | Loader exists; Qwen smoke failed | Do not use for primary claim |

---

## 2. Work packages (priority order)

### WP-A — Geometry & framing (Tier A, **week 1**)

**Goal:** Make “latent routing state” a figure and a statistical object, not a metaphor. No new GPU.

#### A1. Paper narrative pass (writing)

| Section | Change |
| ------- | ------ |
| Abstract | Lead with state geometry; routing = validation |
| Intro | RH: existence + factorization of `s(q)` |
| Method | Evaluation order; two-axis sufficient model |
| Results §1 | State estimation: 2D plot, factor loadings, Cohen’s d table |
| Results §2 | Opportunity AUROC (existing) |
| Results §3 | Routing (short); exploitation gap |
| Discussion | “What survives if routing never improves” |

**Files:** `paper/draft/01_introduction.tex` … `08_conclusion.tex`, `paper/acl.tex`, tables `T_framework.tex`, `T_dimensions.tex`.

#### A2. Offline geometry analysis (new code)

**New script:** `scripts/routing/geometry.py`

**Functions:**

1. `load_state_frame(merged_csv)` — columns: `c_q`, `entropy_w`, `delta_entropy`, `delta_margin_gain`, `y_opp`, `oracle_bucket`.
2. `factor_analysis_2d(frame)` — sklearn `FactorAnalysis(n_components=2)` or PCA; report loadings matrix.
3. `axis_scores(frame)` — difficulty score = mean z(H_w, c_q); escalation score = mean z(Δm_gain, ΔH) (or FA scores).
4. `separation_report(frame)` — Cohen’s d opportunity vs too-hard per axis (reuse `evaluation.effect_size_between_buckets`).
5. `write_geometry_json(...)` → `analysis/geometry_arc_test.json`.

**CLI hook:** extend `scripts/run.py`:

```bash
python scripts/run.py geometry \
  --merged analysis/arc_merged.csv \
  --output analysis/geometry_arc_test.json
```

**Tests:** unit test on synthetic 4-column frame (loadings shape, no NaN).

#### A3. Geometry figure (new plot)

**Extend:** `scripts/routing/plots.py` → `plot_routing_state_scatter(frame, out_path)`

- x = difficulty-side score, y = escalation-side score
- Color = oracle bucket (easy / opportunity / weak_only / too_hard)
- Marginal densities or convex hull optional

```bash
python scripts/run.py plot routing-state \
  --geometry analysis/geometry_arc_test.json \
  --merged analysis/arc_merged.csv \
  --output paper/figures/F_state_geometry.png
```

#### A4. Regenerate conceptual figure

```bash
python scripts/run.py plot conceptual-model --output paper/figures/F0_conceptual_model.png
```

#### A5. Gate (Tier A complete)

- [ ] `geometry_arc_test.json` shows 2 factors with difficulty vars loading axis 1, escalation vars axis 2 (or clear PCA equivalent)
- [ ] F_state_geometry shows opportunity cluster separated from too-hard on **y** more than **x**
- [ ] Paper PDF compiles; routing demoted to ≤1 page in main body

**Effort:** ~3–5 days writing + 1–2 days code.

---

### WP-B — Model-family replication (Tier B1, **week 2**, optional)

**Goal:** Answer “Is the two-axis structure Llama-specific?”

#### B0. Opportunity screen (mandatory — do not skip)

**Use existing:** `scripts/routing_opportunity_assessment.py` or `oracle` on n=50 ARC TEST seed=42.

| Outcome | Action |
| ------- | ------ |
| Opportunity rate ∈ [15%, 55%] | Proceed to full CALIB+TEST |
| ~0% opportunity | Try Qwen 3B↔7B or report as boundary condition in appendix |
| ~100% easy | Pool too strong; do not burn GPU on full replication |

#### B1. Run pipeline for Qwen pool

Same commands as Llama; different `--model` args. Store under `experiments/M4_qwen/`, `experiments/M5_qwen/` (do not overwrite Llama artifacts).

```bash
# Example pattern (exact flags: scripts/run.py --help)
python scripts/run.py oracle --dataset arc_challenge --weak-model Qwen/Qwen2.5-1.5B-Instruct --strong-model Qwen/Qwen2.5-3B-Instruct ...
python scripts/run.py probes  ...
python scripts/run.py features ...
python scripts/run.py merge --output analysis/arc_merged_qwen.csv
python scripts/run.py geometry --merged analysis/arc_merged_qwen.csv --output analysis/geometry_arc_qwen_test.json
```

#### B2. Cross-family comparison (analysis only)

**New:** `scripts/routing/geometry.py` → `compare_geometry(llama_json, qwen_json)`

Report:

- Same axis interpretation? (loadings correlation)
- Same opp vs too-hard separation pattern? (Δm_gain d > H_w d on both?)
- AUROC ordering preserved?

**Paper:** one subsection or appendix table — not a second full results chapter.

#### B3. Gate

- [ ] Screen passed
- [ ] Qwen geometry qualitatively matches Llama (escalation axis separates opp vs too-hard)
- [ ] If fails: honest negative result — “structure is pool-dependent” (still publishable)

**Effort:** 2–4 GPU-days + 1 day analysis/writing.

---

### WP-C — Dataset transfer (Tier B2, **alternative to B**, not both)

**Goal:** “Do the same axes appear on factual/knowledge MCQ?”

#### C1. Implement MMLU loader

**File:** `scripts/routing/datasets.py`

- Add `mmlu` to `SUPPORTED_DATASETS`
- `load_queries(dataset="mmlu", subjects=["high_school_physics", "college_medicine"], split=...)`
- Reuse MCQ prompt path from ARC (`format_arc_question` pattern)

#### C2. Screen + run

- n=50 opportunity screen on CALIB
- If viable: full pipeline with **Llama 1B/3B** (same pool as primary)
- Output: `analysis/geometry_mmlu_test.json`

#### C3. Gate

Same separation criteria as B. If MMLU shows degenerate opportunity, report and keep ARC as sole primary.

**Effort:** 2 days loader + 2–4 GPU-days. **Pick B1 OR C, not both** unless ahead of schedule.

---

### WP-D — Confidence dynamics (Tier C, **week 3**, optional)

**Goal:** Test RH2+ — pre-decoding signals are **dynamic** (confidence trajectory), not only final-layer scalars.

#### D1. Probe extension design

**Current:** `first_token_logits` → single vector at last prompt position (`model_dependent.py:76-78`).

**New:** `layerwise_first_token_logits(model, inputs, layer_indices) -> dict[int, Tensor]`

Implementation options (pick one):

| Option | Mechanism | Pros | Cons |
| ------ | --------- | ---- | ---- |
| **D-a** | Forward hooks on `model.model.layers[i]` | Works with most HF causal LMs | Hook boilerplate |
| **D-b** | `output_hidden_states=True` + `lm_head` per layer | Clean if `lm_head` shared | Extra matmul per layer |
| **D-c** | `early_exit` hidden @ selected layers | Matches literature | Architecture-specific |

**Recommended:** D-a hooks on 4 layers: `[L/4, L/2, 3L/4, L-1]` plus final.

#### D2. New metrics per query

For each layer ℓ, compute `H_ℓ`, `m_ℓ` via existing `extract_logits`.

**Derived scalars (store in probe CSV):**

| Column | Definition |
| ------ | ---------- |
| `slope_entropy` | OLS slope of H_ℓ vs layer index |
| `slope_margin` | OLS slope of m_ℓ vs layer index |
| `delta_entropy_early_late` | H_{L/4} − H_{final} |
| `delta_margin_early_late` | m_{final} − m_{L/4} |
| `entropy_var_layers` | Var(H_ℓ across layers) |

Keep raw layer series in optional JSONL (`experiments/M5/layer_traces/`) for plots only — do not bloat main CSV.

#### D3. Constants + merge

**File:** `scripts/routing/constants.py`

- Add `PROBE_DYNAMICS_BASES` and derived names
- Map to latent axis hypotheses: dynamics → escalation refinement

**File:** `scripts/routing/data.py` — merge weak/strong dynamics like existing entropy columns.

#### D4. Screen before full TEST

```bash
# CALIB n=100 only
python scripts/run.py probes --layerwise --layers quarter ...
python scripts/run.py screen --features slope_margin,slope_entropy,...
```

**Gate:** AUROC on CALIB opportunity ≥ `entropy_w` + 0.02 **or** ΔAUROC complementarity with Δm_gain p<0.05 → run full TEST. Else: appendix null result.

#### D5. Figure

`plot_confidence_trajectory(queries_by_bucket)` — spaghetti or median bands per oracle bucket.

**Effort:** 3–5 days implementation + 1–2 GPU-days + 1 day analysis.

---

### WP-E — Paper integration & submission hygiene (**week 4**)

| Task | Output |
| ---- | ------ |
| Sync tables with geometry JSON | `T_geometry_loadings.tex`, update `T2_correlation.tex` |
| Limitations paragraph | Pool, dataset, linear 2D model |
| Reproducibility | Pin `uv.lock`, record model revisions in `experiments/M4/metadata.json` |
| Build PDF | `cd paper && pdflatex acl.tex` |
| Internal checklist | `research/10_experiment_registry.md` update |

---

## 3. Infrastructure map

```text
┌─────────────────────────────────────────────────────────────────┐
│                        Hugging Face Hub                          │
│              Llama-3.2-1B/3B  (+ optional Qwen)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │ download
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     GPU worker (single machine)                  │
│  oracle.py → probes.py → features.py → merge → geometry.py      │
│       │              │                                           │
│       ▼              ▼                                           │
│  experiments/M4/   experiments/M5/                             │
│  oracle JSON       probe CSVs + optional layer_traces/         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   analysis/ (derived, small)                     │
│  arc_merged.csv · geometry_*.json · interpret_*.json           │
│  selected_feature.json · splits.json                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              plots.py → paper/figures/ · paper/draft/            │
└─────────────────────────────────────────────────────────────────┘
```

### What exists today vs what to build

| Component | Status | Action |
| --------- | ------ | ------ |
| Oracle / probes / features / merge | ✓ | Re-run if artifacts missing |
| `interpret` (Q2–Q3 bundle) | ✓ | Keep; geometry supplements it |
| `plots conceptual-model` | ✓ updated | Regenerate F0 |
| `geometry.py` + CLI | ✗ | **Build (WP-A2)** |
| `plot routing-state` | ✗ | **Build (WP-A3)** |
| Layerwise probes | ✗ | **Build (WP-D)** if Tier C |
| MMLU loader | ✗ | **Build (WP-C)** if Tier B2 |
| Qwen experiment dirs | partial smoke | Full pipeline if B1 passes screen |

### Storage budget

| Run | Approx size |
| --- | ----------- |
| ARC oracle (1B+3B, ~1.2k TEST) | ~5 MB JSON |
| Probe CSVs (full) | ~2 MB |
| Layer traces (optional, 4 layers × 2 models) | ~50–200 MB |
| Analysis JSON/CSV | <10 MB |

---

## 4. Decision register (log before running)

Add rows to `research/10_experiment_registry.md`:

| ID | Decision | Rationale |
| -- | -------- | --------- |
| D-GEO | 2D model = FA vs PCA vs hand-weighted z-scores | FA if n sufficient; else PCA |
| D-B1 | Qwen 1.5B/3B vs 3B/7B | After opportunity screen |
| D-B2 | B1 (Qwen) vs C (MMLU) | Pick one replication axis |
| D-DYN | Layer hook strategy D-a vs D-b | After smoke on 10 queries |
| D-DYN-NULL | If dynamics fail screen | Appendix only vs omit |

---

## 5. Four-week calendar

| Week | Focus | Deliverable |
| ---- | ----- | ----------- |
| **1** | WP-A | Geometry JSON + F_state_geometry + paper rewrite draft |
| **2** | WP-B or WP-C | One replication (screen → full or stop) |
| **3** | WP-D (optional) | Dynamics screen → TEST or null appendix |
| **4** | WP-E | Camera-ready, reproducibility, limit scope creep |

**If only one week:** Do WP-A only — still a substantially stronger paper than adding routers.

---

## 6. Explicit non-goals (defer to Paper 2)

- 4-model capability ladder (1B→3B→7B→8B)
- 5+ datasets (GSM8K, TruthfulQA, HellaSwag, …)
- Paraphrase representation stability (needs LLM paraphraser + 3× prefill cost)
- Attention entropy, VAE latent discovery, neural router
- Observation store / Paper 2 infrastructure

---

## 7. Risk matrix

| Risk | Mitigation |
| ---- | ---------- |
| `arc_merged.csv` missing | Run full Llama pipeline first (1–2 GPU-days) |
| Qwen ARC 0% opportunity (prior pilot) | Screen n=50; switch to 3B/7B or report boundary |
| Dynamics no gain | Pre-register as RH2+ test; null result in appendix |
| Reviewer “only two models” | B1 Qwen replication + family-independent framing |
| Reviewer “only ARC” | B2 MMLU 2 subjects OR explicit limitation |
| Scope creep | Tier A ships without B/C if week 1 slips |

---

## 8. Command quick reference (primary path)

```bash
# --- Environment ---
uv sync --extra analysis

# --- Full Llama ARC pipeline (if artifacts missing) ---
python scripts/run.py oracle   --help   # dataset, models, splits
python scripts/run.py probes     --help
python scripts/run.py features   --help
python scripts/run.py merge      --help
python scripts/run.py interpret  --help

# --- Tier A (after merge) ---
python scripts/run.py geometry   --merged analysis/arc_merged.csv \
  --output analysis/geometry_arc_test.json
python scripts/run.py plot routing-state \
  --merged analysis/arc_merged.csv \
  --geometry analysis/geometry_arc_test.json \
  --output paper/figures/F_state_geometry.png
python scripts/run.py plot conceptual-model \
  --output paper/figures/F0_conceptual_model.png

# --- Paper ---
cd paper && pdflatex acl.tex && bibtex acl && pdflatex acl.tex && pdflatex acl.tex
```

---

## 9. Definition of done (ACL submission)

1. **Contribution statement** in intro matches `claims.md` — geometry first, routing last.
2. **Figure:** 2D routing-state scatter with oracle buckets (main paper).
3. **Table:** Factor loadings or axis operationalization (`T_geometry_loadings` or `T_dimensions`).
4. **Existing evidence preserved:** recovery matrix, complementarity DeLong, routing exploitation gap.
5. **Optional appendix:** Qwen replication OR MMLU OR dynamics (at least one attempted with screen documented).
6. **Reproducibility:** splits, selected feature, model IDs, and analysis JSON committed; no secrets in repo.

---

*This plan intentionally trades breadth (many models/datasets/signals) for depth (one latent-state story with minimal falsifiable extensions).*
