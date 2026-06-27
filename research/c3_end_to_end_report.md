# C3 — End-to-end process, results, and findings

> **Related:** [`c3_layerwise_concepts.md`](c3_layerwise_concepts.md) · [`c3_prefill_extensions_plan.md`](c3_prefill_extensions_plan.md) · [`paper_process_and_results.md`](paper_process_and_results.md) · Commands: [`../experiments/README.md`](../experiments/README.md)

This document is the complete account of **Campaign C3 (layerwise confidence evolution)** on ARC-Challenge: what was run, in what order, how the pipeline works, and what was found. Numbers come from artifacts under `experiments/campaigns/C3_llama_confidence_formation/M5/`, `analysis/c3_*`, and `paper/figures/F7_*` unless noted.

**Status:** Complete (RunPod GPU extraction + Mac CPU postprocess; all 20 campaign artifacts verified on laptop).

---

## 1. Research question and headline answer

### RH5

> Does **layerwise evolution** of model confidence provide routing-relevant information **beyond terminal prefill confidence** (C0)?

C3 is **not** a fourth information source. It extends **model-derived characterization** using the same forward pass as C0, recording how margin (and entropy) evolve across transformer depth at the last prompt token.

### Answer on ARC — **Null for mid-network routing signal**

| Claim | Supported? | Evidence |
| ----- | ---------- | -------- |
| Terminal layerwise probes match C0 | **Yes** | Parity: Δmargin = 0; max \|Δlogit\| ≈ 0.0625 (bf16 noise) |
| Oracle buckets separate **before** final depth | **No** | Layers 1…L−1: median margin ≈ 0 for all buckets (logit-lens on pre-norm hiddens) |
| Buckets separate **at** terminal depth | **Yes** | Sharp spike at L; clearest on 3B strong (F7) |
| Layerwise scalars add routing relevance beyond C0 | **No** | `slope_margin` ≡ terminal margin (ρ identical); `stabilization_layer` degenerate (median 2 everywhere) |
| CALIB→TEST pattern stable | **Yes** | Same L\* and terminal bucket ordering on CALIB and TEST |

**Paper framing (locked):** Publishable **RH5 null** — on ARC with Llama 3.2 1B/3B, layerwise trajectories do **not** add mid-network routing signal beyond terminal C0 probes; bucket separation appears only at final depth. Optional §5.7 with F7 + explicit logit-lens limitation.

---

## 2. Locked configuration

| Component | Value |
| --------- | ----- |
| Weak \(M_w\) | `meta-llama/Llama-3.2-1B-Instruct` (L = 16) |
| Strong \(M_s\) | `meta-llama/Llama-3.2-3B-Instruct` (L = 28) |
| Dataset | ARC-Challenge |
| CALIB split | Official validation, \(n = 299\) |
| TEST split | Official test, \(n = 1{,}172\) |
| Prompt protocol | Chat template v1 (`prompt_protocol.py`), same as C0 |
| Oracle | Greedy decode, `max_new_tokens=8`; buckets from pinned oracle JSON |
| Complexity \(c(q)\) | `piece_count` (D46 winner; frozen in `analysis/selected_feature.json`) |
| GPU env | RunPod, bf16, `batch_size=1` |
| Formation params | `stab_eps=0.02`, `stab_k=2`, `margin_tol=0.001` |

**Supervision boundary (unchanged from C0):** Layerwise extraction uses **no routing labels**. Oracle buckets enter only in offline merge and RH5/F7 analysis.

---

## 3. End-to-end pipeline (overview)

```text
                    ┌─────────────────────────────────────────────────────────┐
                    │                    RunPod (GPU)                          │
                    └─────────────────────────────────────────────────────────┘
  A0 Parity ──► Smoke ──► CALIB extract (weak+strong) ──► [gate] ──► TEST extract
     │              │              │                                      │
     │              │              ▼                                      ▼
     │              │         layerwise CSV + JSONL traces          layerwise CSV + JSONL
     │              │              │                                      │
     └──────────────┴──────────────┴──────────────────────────────────────┘
                                        │
                          scp / pull_c3_from_runpod.sh / Jupyter zip
                                        ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │                    Mac (CPU)                             │
                    └─────────────────────────────────────────────────────────┘
              postprocess calib ──► merge + F7 + RH5 JSON + routing relevance
              postprocess test  ──► (same, TEST oracle + arc_test_features.csv)
```

**Phase order (mandatory):**

1. **Parity** — terminal margin must match C0 before any full run.
2. **Smoke** — 10 queries per model; inspect traces and CSV schema.
3. **CALIB extract** — full validation split, weak then strong.
4. **CALIB postprocess** — merge, F7 plots, RH5 divergence JSON, routing-relevance sanity.
5. **Decision gate** — CALIB F7 interpretable → proceed to TEST (passed).
6. **TEST extract** — full test split.
7. **TEST postprocess** — final figures and analysis for paper.

---

## 4. Step-by-step execution

### 4.1 One-time RunPod setup

```bash
cd /workspace/llm_routing
git pull   # commits through 7bf7674 (layerwise parity, postprocess fixes)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync && uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv sync --extra analysis
huggingface-cli login
chmod +x scripts/run_c3_runpod.sh scripts/run_c3_postprocess.sh
export PYTHONPATH=/workspace/llm_routing/scripts:$PYTHONPATH
```

Pin oracle + features on pod (now also in git; copy if pod predates commit):

```bash
# experiments/M4/routing_opportunity/arc_validation_oracle.json
# experiments/M4/routing_opportunity/arc_test_oracle.json
# experiments/M5/arc_validation_features.csv
# experiments/M5/arc_test_features.csv
# analysis/selected_feature.json
```

### 4.2 Phase A0 — Parity (`./scripts/run_c3_runpod.sh parity`)

**Purpose:** Verify that layer L margin equals C0 terminal margin for the same queries.

**What it does:** For \(n=10\) CALIB queries per model, runs full forward with `output_hidden_states=True`, computes terminal margin via the **C0 path** at layer L (`out.logits` at last prompt token), and compares to logit-lens at L−1 and intermediate layers.

**Result:** **PASSED** for 1B and 3B.

| Check | Outcome |
| ----- | ------- |
| Terminal Δmargin vs C0 | **0** (exact match in reported precision) |
| max \|Δlogit\| | ≈ **0.0625** (expected bf16 quantization) |
| Intermediate layers | Near-zero margins (expected under pre-norm logit-lens) |

**Why parity mattered:** Early implementation used `lm_head(norm(hs[L]))` vs `out.logits` and failed on RunPod because `CausalLMOutputWithPast` does not always expose `last_hidden_state`. Fix: terminal layer uses **`out.logits` directly**; layers 1…L−1 use `lm_head(hidden_states[ℓ])` without final RMSNorm.

### 4.3 Phase A1 — Smoke (`./scripts/run_c3_runpod.sh smoke`)

**Purpose:** End-to-end dry run on 10 weak + 10 strong TEST queries.

**Checks:** CSV columns match `PROBE_LAYERWISE_FIELDS`, JSONL traces written, no OOM, margins at L plausible.

**Result:** Passed; used to validate postprocess before CALIB burn.

### 4.4 Phase B — CALIB extraction (`./scripts/run_c3_runpod.sh extract calib all`)

**Purpose:** Extract layerwise probes for all 299 validation queries × 2 models.

**Outputs per pool (weak / strong):**

| File | Content |
| ---- | ------- |
| `arc_calib_{weak,strong}_layerwise.csv` | Terminal C0-compatible probes + `stabilization_layer`, `slope_margin`, method tag |
| `layer_traces/calib_{weak,strong}.jsonl` | Per-query arrays: `depth_fraction`, `margin`, `entropy`, formation scalars |

**Runtime:** Dominated by 598 forward passes with hidden states (≈ hours on single GPU; acceptable for 1B/3B at bs=1).

### 4.5 Phase C — CALIB postprocess (`./scripts/run_c3_postprocess.sh calib`)

**Purpose:** Merge weak+strong+features+oracle; plot F7; compute RH5 divergence; routing-relevance JSON.

**Steps inside script:**

1. **Merge** — join on `query_id`; fallback match on `user_content` when `prompt_hash` drifts (chat-template edge cases).
2. **F7** — median margin vs normalized depth ℓ/L, lines for easy / opportunity / too_hard.
3. **RH5 JSON** — bucket medians, pairwise dispersion, layer L\* maximizing opportunity vs too_hard separation.
4. **Routing relevance** — Spearman ρ vs opportunity for C0 and layerwise-derived columns (sanity that merge did not break C0 signals).

**Decision gate:** CALIB F7 shows terminal separation (especially 3B strong) → proceed to TEST.

### 4.6 Phase D — TEST extraction (`./scripts/run_c3_runpod.sh extract test all`)

**Purpose:** Full test split, \(n = 1{,}172\) × 2 models.

Same artifact layout as CALIB with `arc_test_*` and `layer_traces/test_*`.

### 4.7 Phase E — TEST postprocess (`./scripts/run_c3_postprocess.sh test`)

Same as CALIB but uses `arc_test_oracle.json` and **`arc_test_features.csv`** (critical: an early bug merged TEST layerwise CSVs against CALIB features, yielding zero overlap).

### 4.8 Artifact transfer to Mac

```bash
# From laptop (replace POD host):
./scripts/pull_c3_from_runpod.sh

# Or on pod: zip campaign + analysis + figures → download via Jupyter
```

**Verified on Mac:** 20/20 RunPod files present (8 campaign CSV/JSONL, 4 F7 PNGs, 8 analysis JSON/CSV).

Row counts: use pandas (`len(df)`), not `wc -l` — multiline CSV fields inflate line counts.

---

## 5. Technical design (what the code measures)

### 5.1 Logit-lens probe

At the **last prompt token** position:

```text
ℓ = 1 … L−1:   logits_ℓ = lm_head( hidden_states[ℓ] )      # pre-final-norm
ℓ = L:         logits_L = out.logits                         # C0-identical path
               → margin_ℓ, entropy_ℓ from softmax(logits_ℓ)
```

**Methods wording:** “We use a logit-lens style probe.” We do **not** claim intermediate layers are the model’s true next-token predictions.

### 5.2 Formation scalars (CSV)

| Column | Definition | RH5 role |
| ------ | ---------- | -------- |
| `stabilization_layer` | First layer after \(k=2\) consecutive margin steps with \|Δm\| < ε=0.02 | Intended primary scalar — **degenerate in results** |
| `slope_margin` | OLS slope of \(m_\ell\) vs ℓ | Redundant with terminal margin when only L carries signal |

### 5.3 RH5 divergence statistic

From JSONL + oracle buckets:

1. Compute **median** \(m_\ell\) per bucket at each depth.
2. **Pairwise dispersion** = mean absolute difference across bucket pairs at each ℓ.
3. Report **L\*** = argmax dispersion (opportunity vs too_hard focus) and **fraction_depth** = L\*/L.

Compare models only on **fraction depth**, never raw layer index (16 vs 28).

---

## 6. Infrastructure fixes (reproducibility notes)

These bugs were hit during the RunPod campaign and fixed on `main`:

| Issue | Symptom | Fix |
| ----- | ------- | --- |
| Terminal hidden-state parity | `lm_head(norm(hs[L]))` ≠ `out.logits` | Layer L uses `out.logits`; resolver tries multiple terminal hidden paths for diagnostics |
| Inference-mode crash | Tensor ops outside `torch.inference_mode()` | All post-forward ops inside inference mode |
| Missing oracle on pod | Postprocess merge failure | Oracles pinned in git under `experiments/M4/routing_opportunity/` |
| `prompt_hash` mismatch | Merge dropped rows | Allow merge when `user_content` matches |
| TEST postprocess 0 overlap | Empty TEST analysis | Use `arc_test_features.csv` for TEST role, not validation features |
| Mac transfer | Large JSONL over scp | `scripts/pull_c3_from_runpod.sh` + zip fallback |

---

## 7. Results

### 7.1 Merge coverage

| Split | Merged rows | Oracle buckets (TEST) |
| ----- | ----------- | --------------------- |
| CALIB | 299 | (validation distribution) |
| TEST | 1,172 | opportunity 43.3%, easy 25.9%, too_hard 25.6%, weak_only 5.2% |

### 7.2 C0 signal sanity on merged C3 tables (Spearman ρ vs opportunity)

Confirms layerwise CSV terminal columns still carry the same routing relevance as C0.

| Signal | CALIB ρ [95% CI] | TEST ρ [95% CI] |
| ------ | ---------------- | --------------- |
| `margin_w` | −0.122 [−0.231, −0.012] | −0.120 [−0.177, −0.064] |
| `margin_s` | +0.139 [+0.030, +0.245] | +0.080 [+0.023, +0.138] |
| **`delta_margin_gain`** | **+0.212 [+0.101, +0.325]** | **+0.172 [+0.115, +0.234]** |
| `slope_margin_w` | −0.122 (identical to `margin_w`) | −0.120 |
| `slope_margin_s` | +0.141 (≈ `margin_s`) | +0.076 |
| `stabilization_layer_*` | NaN (zero variance) | NaN (zero variance) |

**Interpretation:** Recoverability signal (`delta_margin_gain`) remains the strongest cross-model probe; layerwise slope adds nothing beyond terminal margin. Stabilization layer is constant (median **2**, std **0** for every query) because layers 1…L−1 margins are ≈0 and the first large jump is L−1→L.

### 7.3 RH5 divergence — layer L\* and terminal bucket medians

| Split | Model | L\* / L | d\* (fraction depth) | Easy @ L | Opportunity @ L | Too hard @ L |
| ----- | ----- | ------- | -------------------- | -------- | --------------- | ------------ |
| CALIB | 1B weak | 16/16 | 1.00 | 0.473 | 0.282 | 0.299 |
| CALIB | 3B strong | 26/28 | 0.929 | 0.914 | 0.711 | 0.287 |
| TEST | 1B weak | 16/16 | 1.00 | 0.500 | 0.320 | 0.330 |
| TEST | 3B strong | 26/28 | 0.929 | 0.936 | 0.691 | 0.277 |

**Mid-network (example TEST strong, L=8):** easy = opp = too_hard ≈ **0.000** (all buckets).

**Penultimate (TEST strong, L=27, d\*≈0.96):** margins rise but buckets still overlap (easy 0.017, opp 0.008, hard 0.009).

**Terminal (TEST strong, L=28):** clear separation — easy **0.936**, opportunity **0.691**, too_hard **0.277**.

### 7.4 Per-layer JSONL check (TEST weak, n=1,172)

| Layer | Median margin | % queries with margin > 0.01 |
| ----- | ------------- | ---------------------------- |
| L1 | 0.000001 | 0.0% |
| L8 | 0.000000 | 0.0% |
| L15 | 0.000017 | 0.0% |
| **L16** | **0.351** | **96.6%** |

Pattern: a **step function** — flat near zero through L−1, sharp activation at L.

### 7.5 F7 figures (visual summary)

Files: `paper/figures/F7_confidence_evolution_{calib,test}_{weak,strong}.png`

**Common pattern (all four):**

- x-axis: normalized depth ℓ/L
- y-axis: median margin by oracle bucket
- Layers 1…L−1: three buckets collapsed near zero
- Layer L: vertical separation; **3B strong** shows clearest spread (easy > opportunity > too_hard)

**Paper figure choice:** Prefer **`F7_confidence_evolution_test_strong.png`** for §5.7 (largest model, held-out split, cleanest bucket ordering at L).

### 7.6 Weak 1B vs strong 3B (qualitative)

| Model | Terminal separation | Routing interpretation |
| ----- | -------------------- | ------------------------ |
| **3B strong** | Large gaps at L (easy ~0.9, opp ~0.7, hard ~0.28) | Terminal strong margin aligns with recoverability story from C0 |
| **1B weak** | Opportunity and too_hard **overlap** at L (~0.32 vs ~0.33); easy higher (~0.50) | Weak model terminal margin poorly separates routing buckets — consistent with C0 RH2 |

---

## 8. Findings (numbered for paper cross-reference)

### Finding C3-1 — Terminal fidelity

Layerwise extraction at L reproduces C0 terminal probes exactly (parity gate). The C3 campaign does not alter the C0 signal definition at the decision boundary.

### Finding C3-2 — No mid-depth routing signal (RH5 null)

Under the locked logit-lens protocol, **no oracle bucket separation appears before the final layer**. Median margins at ℓ < L are statistically indistinguishable across easy, opportunity, and too_hard. Any RH5 claim of “early” routing information would require a different probe (e.g., applying final RMSNorm at intermediate layers) — out of scope for this locked design.

### Finding C3-3 — Separation is terminal-only

L\* equals L for 1B (d\* = 1.0) and near-L for 3B (L\* = 26/28, d\* ≈ 0.93). The 3B penultimate layer shows rising but **non-separating** margins; discriminative power concentrates at L.

### Finding C3-4 — Formation scalars are non-informative

- **`stabilization_layer`:** Degenerate (always 2) given near-zero early margins — **do not headline**.
- **`slope_margin`:** Numerically equivalent to terminal-margin routing relevance — adds no new dimension beyond C0.

### Finding C3-5 — CALIB/TEST stability

Divergence depth, terminal bucket ordering, and F7 shape replicate on held-out TEST. The RH5 null is not a CALIB overfit artifact.

### Finding C3-6 — Consistency with C0 narrative

3B strong terminal margins separate buckets; 1B weak margins conflate opportunity and too_hard. Layerwise analysis **refines** the model-dependent story (where in depth separation appears) without introducing a new exploitable routing feature.

---

## 9. Limitations (state explicitly in paper)

1. **Logit-lens on pre-norm hiddens:** Intermediate Llama hidden states omit final RMSNorm; LM-head projections at ℓ < L are analysis probes, not calibrated probabilities. Near-zero intermediate margins may partly reflect this mismatch rather than “no internal computation.”
2. **Single architecture family:** Llama 3.2 1B/3B only; no Qwen/C1 transfer.
3. **MCQ letter task:** ARC four-option setup; margins computed over choice logits only (same as C0).
4. **No new routing policy:** C3 is characterization-only (RH5), not RH4 policy tuning.
5. **`stabilization_layer` parameterization:** With ε=0.02 and k=2, the scalar collapses when trajectories are step functions — report as failed exploratory scalar, not evidence against “formation” in general.

---

## 10. Paper integration (§5.7 sketch)

**Methods (short):** One additional forward pass with `output_hidden_states=True`; logit-lens margins at last prompt token; terminal layer matched to C0 via `out.logits`; buckets from frozen oracle; report median trajectories (F7) and max-divergence depth L\*.

**Results (short):** Bucket trajectories overlap through ℓ/L < 1; separation appears at terminal depth only (F7). L\* = L (1B) or 26/28 (3B). `delta_margin_gain` routing relevance unchanged from C0; layerwise scalars non-informative.

**Limitation paragraph:** Pre-norm logit-lens; defer claims of progressive confidence formation to Discussion unless future work applies norm-aligned probes.

**Figure caption (F7 test strong):** Median margin vs normalized depth on ARC test (n=1,172); 3B strong model; easy, routing opportunity, and too-hard oracle buckets. Shaded regions optional (bootstrap CI not plotted in current F7).

---

## 11. Artifact index

| Path | Purpose |
| ---- | ------- |
| `experiments/campaigns/C3_llama_confidence_formation/M5/arc_{calib,test}_{weak,strong}_layerwise.csv` | Per-query terminal + formation scalars |
| `.../layer_traces/{calib,test}_{weak,strong}.jsonl` | Full depth trajectories |
| `analysis/c3_arc_{calib,test}_merged.csv` | Master merged tables |
| `analysis/c3_arc_{calib,test}_routing_relevance.json` | C0 + layerwise routing relevance |
| `analysis/c3_rh5_{calib,test}_{weak,strong}.json` | RH5 divergence + bucket medians |
| `paper/figures/F7_confidence_evolution_*.png` | Paper figures |
| `scripts/run_c3_runpod.sh` | GPU: parity, smoke, extract |
| `scripts/run_c3_postprocess.sh` | CPU: merge, F7, RH5 |
| `scripts/pull_c3_from_runpod.sh` | Mac artifact pull |
| `scripts/routing/layerwise.py` | Core extraction + parity |

Large campaign CSVs/JSONL are **gitignored**; oracles and feature CSVs are in git.

---

## 12. Reproduction checklist

```bash
# RunPod
./scripts/run_c3_runpod.sh parity
./scripts/run_c3_runpod.sh smoke
./scripts/run_c3_runpod.sh extract calib all
./scripts/run_c3_postprocess.sh calib
./scripts/run_c3_runpod.sh extract test all
./scripts/run_c3_postprocess.sh test

# Mac (after pull)
uv sync --extra analysis
./scripts/run_c3_postprocess.sh all   # re-generate figures/JSON from copied artifacts
```

**Success criteria:**

- [x] Parity PASS both models
- [x] CALIB/TEST merge n = 299 / 1172
- [x] Four F7 PNGs generated
- [x] Eight RH5 JSON files
- [x] Terminal `delta_margin_gain` ρ > 0.15 on TEST
- [x] Mid-layer bucket medians ≈ 0; terminal separation visible on F7 test strong

---

## 13. What C3 does not claim

- A new routing signal family beyond C0 terminal probes.
- That confidence “forms progressively” in Llama mid-layers under this probe (Discussion-only hypothesis).
- Improved routing policies (RH4 unchanged).
- Cross-architecture or paraphrase robustness.

**Bottom line:** C3 completes the model-derived **characterization** agenda for Paper 1 by showing **where** routing-relevant separation lives in depth (terminal only), strengthening the C0 focus on terminal prefill statistics for routing research on ARC.

---

## 14. Route B — representation geometry (RH5-repr)

> **Status:** Implemented in code; **requires re-extract** on GPU (existing JSONL traces predate Route B fields).

### 14.1 Why Route B exists

Route A (logit-lens margin) gave an **RH5 null**: bucket separation only at the final layer. A reviewer may ask:

> “Are early layers uninformative, or is the LM head a poor readout?”

Route B tests an alternative: **adjacent hidden-state drift** — how much the representation moves layer-to-layer — without projecting through the vocabulary head.

| Route | Probe | Question |
| ----- | ----- | -------- |
| **A (done)** | Logit-lens margin | When can the vocabulary head decode confidently? |
| **B (new)** | Adjacent drift 1−cos(h_ℓ, h_{ℓ+1}) | How much is the internal state still changing? |

A model may have margin ≈ 0 at layer 8 while representations are already stable (high adjacent cos). Route B distinguishes **decode lag** from **missing internal structure**.

### 14.2 Locked signals (not cosine-to-final)

We **do not** use cosine-to-final (`cos(h_ℓ, h_L)`) — it is structurally monotonic and unlikely to separate buckets.

**Headline scalars:**

| Column | Meaning |
| ------ | ------- |
| `total_representation_drift` | Σ(1 − cos(h_ℓ, h_{ℓ+1})) — total movement |
| `mean_adjacent_cos` | mean cos(h_ℓ, h_{ℓ+1}) — step stability (use for cross-model compare) |
| `repr_adjacent_std` | std of per-step drift — non-uniform evolution |

**Cross-model deltas (merge):** `delta_total_representation_drift`, `delta_mean_adjacent_cos`, `delta_repr_adjacent_std` (strong − weak).

### 14.3 Extraction

Route B is computed in the **same forward pass** as Route A inside `probe_layerwise()` — no extra LM head cost for drift.

```bash
# Full extract (Route A margin + Route B drift) — default
./scripts/run_c3_runpod.sh extract test all

# Faster re-extract: skip intermediate LM head, keep terminal margin + drift
REPR_ONLY=1 ./scripts/run_c3_runpod.sh extract test all
```

**CSV columns added:** `total_representation_drift`, `mean_adjacent_cos`, `repr_adjacent_std`.

**JSONL arrays added:** `adjacent_cos`, `drift`, `drift_depth_fraction`.

### 14.4 Postprocess

```bash
./scripts/run_c3_postprocess.sh test
```

Produces (when traces include `drift[]`):

| Artifact | Route |
| -------- | ----- |
| `F7_confidence_evolution_*.png` | A — margin curves |
| `c3_rh5_*.json` | A — RH5 divergence |
| `F8_representation_drift_*.png` | B — median drift curves |
| `c3_rh5_repr_*.json` | B — RH5-repr divergence |

If traces lack `drift[]`, postprocess skips F8 with a message (re-extract required).

### 14.5 Paper framing

- **Keep §5.7 Route A** (F7 null) as primary C3 result.
- **Add §5.7.1 or Appendix:** Route B ablation + F8.
- **Bridge sentence:** “Because intermediate logit-lens margins collapse, we test whether adjacent representation drift separates oracle buckets.”

**Two interesting outcomes:**

1. Drift separates buckets mid-depth → logit-lens inadequate readout.
2. Drift also fails → routing signal genuinely late for both probes.

### 14.6 Reproduction checklist (Route B)

- [ ] Re-extract TEST (and optionally CALIB) with current `layerwise.py`
- [ ] `./scripts/run_c3_postprocess.sh test` → four F8 PNGs + four `c3_rh5_repr_*.json`
- [ ] Report Spearman ρ vs opportunity for `total_representation_drift_w`, `mean_adjacent_cos_w`, cross-model deltas
- [ ] Compare F8 bucket curves to F7 (same oracle buckets)
