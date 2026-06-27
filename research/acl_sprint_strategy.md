# ACL sprint strategy — deep science, scalable infrastructure

> **Status:** Master plan (2026-06). Track 2 (geometry/PCA) **deferred**.  
> **Primary evidence:** ARC TEST, Llama 3.2 1B/3B — complete locally.  
> **Principle:** One phenomenon deeply — not another router paper.

---

## 1. North star

### Scientific question (paper title direction)

> **What routing-relevant information is available before decoding, and how does it factor into difficulty vs escalation?**

### Contribution hierarchy (reviewer-proof)

| Rank | Object | Status |
| ---- | ------ | ------ |
| **1** | Empirical characterization of **pre-decoding routing information** | ARC Llama — **done** |
| **2** | **Two-property model:** task difficulty ≠ escalation recoverability | ARC — **done** (d≈0.03 vs d≈0.72) |
| **3** | **Generalization:** architecture (Qwen) + domain (MMLU) | **To run** |
| **4** | **Model-derived extension (C3 / RH5):** layerwise evolution vs terminal | **To build + run** |
| **5** | Routing evaluation (exploitation gap) | ARC — **done**; replicate lightly |

**Router is validation, not the product.**

### Signal taxonomy (paper-facing — three information sources)

Organize by **where information comes from** (professor transcript), not extraction depth.

| Information source | What it measures | Signals (this paper) | Campaign |
| ------------------ | ---------------- | -------------------- | -------- |
| **Query-derived** | Task difficulty from query text | `piece_count` / \(c(q)\) | C0 (done) |
| **Model-derived** | Weak-model prefill confidence | $H_w$, $m_w$; **C3:** layerwise evolution | C0 (done); C3 (to build) |
| **Cross-model** | Recoverability / disagreement | $\Delta H$, $\Delta m_{\mathrm{gain}}$ | C0 (done) |
| *(Future) Perturbation-derived* | Paraphrase stability | — | Paper 2 |

**C3 (layerwise confidence evolution)** extends **model-derived** (terminal → layerwise)—not a fourth source. Concepts: [`c3_layerwise_concepts.md`](c3_layerwise_concepts.md).

Even if C3 layerwise signals do not beat terminal AUROC, the contribution stands: *pre-inference routing information characterized by source; difficulty \(\neq\) recoverability*.

### Theorem-like claim (say explicitly in intro + discussion)

> Routing need before generation requires estimating two **non-redundant** properties: **(i)** how difficult the query is for the weak model, and **(ii)** whether additional model capacity can recover the correct answer.

| Property | Operationalizations |
| -------- | ------------------- |
| Difficulty | `piece_count`, $H_w$ |
| Escalation / recoverability | $\Delta H$, $\Delta m_{\mathrm{gain}}$ |

---

## 2. Experiment matrix (only these runs)

Do **not** expand beyond this table without a new decision in `09_decision_register.md`.

| Campaign ID | Pool (weak → strong) | Dataset | Role | n (TEST) | Priority |
| ----------- | -------------------- | ------- | ---- | -------- | -------- |
| **C0** | Llama 1B → 3B | ARC-Challenge | **Primary** (main paper) | 1,172 | ✅ Done |
| **C1** | Qwen 3B → 7B | ARC-Challenge | Architecture generalization | 1,172 | **P1** |
| **C2** | Llama 1B → 3B | MMLU (2 subjects) | **RH7:** dimension generalization test | ~314 | **P2** |
| **C3** | Llama 1B → 3B | ARC-Challenge | Confidence dynamics (layerwise) | 1,172 | **P3** |

**Explicitly excluded:** GSM8K primary (Qwen smoke 100% too-hard), 4-model ladder, full MMLU, BoolQ unless C2 finishes early, paraphrase stability, Track 2 geometry.

### Qwen pool choice (learned from pilots)

| Pool | ARC pilot | Verdict |
| ---- | --------- | ------- |
| Qwen 1.5B → 3B | ~0% opportunity, 70% easy | **Do not use** |
| Qwen 3B → 7B | Not run | **Default for C1** — screen n=50 first |

### MMLU subjects (locked in MASTER)

- `high_school_physics` — STEM / reasoning overlap with ARC
- `logical_fallacies` — different skill (argumentation)

Reuse frozen `piece_count` from D46; **no re-screening on MMLU**.

### C2 is a hypothesis test, not another benchmark

**Wrong framing:** ARC AUROC vs MMLU AUROC leaderboard.

**Right framing (RH7):** After discovering dimensions on ARC, **freeze** pool, signals, and operationalizations — MMLU tests whether the **same routing dimensions** appear in new reasoning regimes.

```text
ARC  →  discover dimensions  →  freeze everything  →  MMLU  →  test generalization
```

**Paper table (T_transfer or §5.6):** dimension × regime alignment — not raw AUROC competition.

| Latent dimension | ARC | MMLU Physics | MMLU Logic | Pattern |
| ---------------- | --- | ------------ | ---------- | ------- |
| Task difficulty ($c$) | opp ↑ | same sign? | same sign? | ✓ / △ / ✗ |
| Model uncertainty ($H_w$) | opp ↑ | … | … | |
| Model disagreement ($\Delta H$) | opp ↑ | … | … | |
| Escalation potential ($\Delta m_{\mathrm{gain}}$) | opp ↑ | … | … | |
| **Escalation separates opp vs too-hard** | $d \approx 0.72$ | $d_{\mathrm{phys}}$ | $d_{\mathrm{logic}}$ | **Key row** |
| **Uncertainty fails separation** | $d \approx 0.03$ | … | … | **Key row** |

**Per-cell statistics:** Spearman ρ(sign), AUROC > 0.5, Cohen's d(opp vs too-hard) — report **qualitative pattern match**, e.g.:

> Escalation potential separates routable from irrecoverable queries on ARC and both MMLU subjects; uncertainty does not — despite different absolute accuracies.

**Subject design (keep):** Physics ≈ near ARC (STEM); Logical fallacies ≈ far (argumentation). A dimension that holds on both is **strong** generalization evidence.

---

## 3. Phased timeline (6–8 weeks, you have time)

### Phase 0 — Narrative lock (week 1, no GPU)

**Goal:** Paper reads as information science, not router benchmark.

| Task | Output |
| ---- | ------ |
| Sharpen two-property claim in intro, method, discussion | `paper/draft/*.tex` |
| Add RH5 (dynamics) + RH6 (cross-family) to `02_research_hypotheses.md` | Research docs |
| Demote routing to ≤1 page main body | `06_results_routing.tex` |
| Update `claims.md` contribution order | Locked vocabulary |

**Exit:** Abstract + intro pass the “another router” smell test.

---

### Phase 1 — Infrastructure (week 1–2, light GPU for smoke)

**Goal:** Run any campaign with one command, resume safely, no artifact collisions.

See **§5 Infrastructure** below. Minimum deliverables:

1. MMLU loader in `datasets.py` (screen script already calls it — **currently broken**)
2. `scripts/campaigns/manifest.yaml` — campaign definitions
3. `scripts/run_campaign.sh` — orchestrates oracle → probes → merge → interpret
4. `scripts/routing/layerwise.py` — hooks for dynamics (design + smoke on n=10)
5. `scripts/routing/compare_generalization.py` — pattern comparison across campaigns

**Exit:** Smoke pass on ARC n=10 for C1 config + dynamics hooks.

---

### Phase 2 — Screening gates (week 2, ~2 GPU-hours)

**Never scale a campaign without passing its gate.**

| Gate | Command pattern | Pass criterion |
| ---- | --------------- | -------------- |
| **G-C1** | Qwen 3B/7B oracle, ARC, n=50 | opportunity ∈ [15%, 55%], all four buckets present |
| **G-C2** | Llama oracle, MMLU, n=50 | same |
| **G-C3** | Layerwise probes, ARC CALIB, n=100 | CSV valid; terminal parity; columns present |

| Fail | Action |
| ---- | ------ |
| 0% opportunity | Try Qwen 3B/8B or report as boundary (appendix) |
| 100% too-hard | Drop dataset for that pool; do not force |

---

### Phase 3 — Campaign C1: Qwen on ARC (week 3, ~1.5 GPU-days)

**Scientific output:** Table comparing **pattern** Llama vs Qwen:

- Cohen’s d: opportunity vs too-hard for $H_w$ vs $\Delta m_{\mathrm{gain}}$
- Same qualitative story? (escalation separates, uncertainty does not)

**Not the goal:** Beat Llama AUROC.

**Artifacts:** `experiments/campaigns/C1_qwen_arc/`, `analysis/C1_*`

---

### Phase 4 — Campaign C2: MMLU dimension transfer (week 4, ~0.5 GPU-days)

**Scientific output:** RH7 — dimension × regime table (ARC | Physics | Logic).

Compare **pattern invariants** (escalation separates, uncertainty does not) — not AUROC leaderboard.

**Analysis:** `compare_generalization.py` → `analysis/C2_dimension_transfer.json`

**Calibration:** ARC CALIB only; MMLU is frozen-protocol transfer eval.

**Artifacts:** `experiments/campaigns/C2_llama_mmlu/`

---

### Phase 5 — Campaign C3: Confidence dynamics — structural signals (week 5, ~2 GPU-days)

**Scientific output:** RH5 — does **structural** (layer trajectory) information add beyond **distributional** (final logits)?

**Paper claim:** Model-derived extension characterized (layerwise vs terminal); maps to difficulty or escalation axis (or reports null). Not a fourth information source.

| Comparison | Metric |
| ---------- | ------ |
| Final \(H_w\) vs layerwise entropy (JSONL only) | Exploratory — not headline |
| Final \(\Delta m_{\mathrm{gain}}\) vs **`stabilization_layer`** | d(opp vs too-hard); primary scalar |
| F7 trajectories + divergence L* | Primary RH5 evidence |

**Publishable null:** “Dynamics do not beat final-layer escalation signal” — still tests RH5.

**Artifacts:** layer traces optional JSONL; scalars in probe CSV.

---

### Phase 6 — Paper integration (week 6–7)

| Section | Content |
| ------- | ------- |
| §5.1–5.3 | ARC Llama (existing) |
| §5.4 | Two-property synthesis (theorem-like paragraph) |
| §5.5 | C1 architecture table + 1 figure (recovery matrix or d-bar chart) |
| §5.6 | C2 MMLU transfer table |
| §5.7 | C3 dynamics: static vs dynamic figure |
| §6 | Routing — brief; exploitation gap |
| Appendix | Failed pilots (Qwen 1.5B/3B, GSM8K) as boundary conditions |

---

### Phase 7 — Buffer (week 8)

Rebuttal prep, figure polish, `doctor` on all campaigns, reproducibility manifest.

---

## 4. Hypothesis map (extended)

| ID | Hypothesis | Campaign | Evidence |
| -- | ---------- | -------- | -------- |
| RH1 | Pre-decoding dimensions predict opportunity | C0 | T2, F1 |
| RH2 | **Difficulty \(\neq\) recoverability** | C0 | F6, d table |
| RH3 | Partial complementarity | C0 | T3 |
| RH4 | Calibrated policy exploitation gap | C0 | T4 |
| **RH5** | **Layerwise confidence evolution adds information beyond terminal prefill (or null)** | C3 | F7 + RH5 JSON |
| **RH6** | **Two-property structure replicates across model families** | C1 | Generalization table |
| **RH7** | **Dimension structure generalizes across reasoning regimes (MMLU)** | C2 | Transfer table |

---

## 5. Infrastructure — scalability & efficiency

### 5.1 Design principles

1. **Resume everything** — oracle already checkpoints; probes should append-only CSV (already does).
2. **One model in VRAM** — weak then strong; never both.
3. **Batch prefill, not generation** — probes `batch_size=8–16` on CUDA; oracle stays batch=1.
4. **Campaign isolation** — separate dirs; never overwrite C0 artifacts.
5. **Screen before scale** — n=50 gates save days of wasted GPU.
6. **Analysis is cheap** — run on laptop; GPU only for inference.

### 5.2 Directory layout (new)

```text
experiments/
├── M4/routing_opportunity/          # C0 (existing — do not touch)
├── M5/                              # C0 probes (existing)
└── campaigns/
    ├── C1_qwen_arc/
    │   ├── M4/oracle_{calib,test}.json
    │   ├── M5/{weak,strong,features}_{calib,test}.csv
    │   └── M5/layer_traces/         # optional, C3-style
    ├── C2_llama_mmlu/
    └── C3_llama_arc_dynamics/

analysis/
├── arc_merged.csv                   # C0 (existing)
├── campaigns/
│   ├── C1_geometry_pattern.json     # not PCA — pattern comparison
│   ├── C2_mmlu_transfer.json
│   └── C3_dynamics_screen.json
└── reproducibility_manifest.json    # model revisions, seeds, dates
```

### 5.3 Campaign manifest (`scripts/campaigns/manifest.yaml`)

```yaml
campaigns:
  C1_qwen_arc:
    weak: Qwen/Qwen2.5-3B-Instruct
    strong: Qwen/Qwen2.5-7B-Instruct
    dataset: arc_challenge
    splits: analysis/splits.json
    max_new_tokens: 8
  C2_llama_mmlu:
    weak: meta-llama/Llama-3.2-1B-Instruct
    strong: meta-llama/Llama-3.2-3B-Instruct
    dataset: mmlu
    subjects: [high_school_physics, logical_fallacies]
    split: test
    limit: 400
    seed: 42
  C3_llama_dynamics:
    weak: meta-llama/Llama-3.2-1B-Instruct
    strong: meta-llama/Llama-3.2-3B-Instruct
    dataset: arc_challenge
    layerwise: true
    layer_fractions: [0.25, 0.5, 0.75, 1.0]
```

`run_campaign.sh C1 --stage oracle --role test` reads manifest and writes to `experiments/campaigns/C1_*/`.

### 5.4 GPU efficiency settings

| Stage | Device | Dtype | batch_size | Notes |
| ----- | ------ | ----- | ---------- | ----- |
| Oracle | CUDA | bfloat16 | 1 | `max_new_tokens=8`; resume + `--max-pending 100` for long jobs |
| Probes (static) | CUDA | bfloat16 | 8–16 | ARC prompts short |
| Probes (layerwise) | CUDA | bfloat16 | 1–4 | Hooks add memory; start 1 |
| Features | CPU | — | — | No GPU |
| Merge / interpret | CPU | — | — | `uv sync --extra analysis` |

**Estimated GPU budget (single L4/A10 24GB):**

| Campaign | Oracle | Probes | Total |
| -------- | ------ | ------ | ----- |
| C1 Qwen ARC | ~8–12 h | ~4–6 h | ~1.5 days |
| C2 MMLU | ~2 h | ~1 h | ~0.5 day |
| C3 Dynamics | 0 (reuse C0 oracle) | ~12–18 h | ~1 day |
| Screening | — | — | ~2 h |
| **Total** | | | **~3–4 GPU-days** |

### 5.5 Layerwise probes — efficient implementation

**One forward pass per query per model** — register hooks on layers at fractions `[0.25, 0.5, 0.75, 1.0]`:

```text
forward pass
  → hook captures hidden at layer ℓ
  → apply shared lm_head (or model-specific head)
  → compute H_ℓ, m_ℓ at last prompt position
  → detach + store scalars; discard activations
```

**Derived columns (weak + strong):**

| Column | Formula |
| ------ | ------- |
| `slope_entropy_w` | OLS slope of $H_\ell$ vs layer index |
| `slope_margin_w` | OLS slope of $m_\ell$ vs layer index |
| `entropy_early_late_w` | $H_{0.25L} - H_{L}$ |
| `margin_early_late_w` | $m_{L} - m_{0.25L}$ |

Store full trajectories only for CALIB n=100 debug plot; TEST gets scalars only.

### 5.6 Code tasks (build order)

| # | Task | File(s) | Blocks |
| - | ---- | ------- | ------ |
| 1 | MMLU + BoolQ loaders | `datasets.py` | C2 screen |
| 2 | Campaign runner | `scripts/run_campaign.sh`, manifest | All campaigns |
| 3 | Layerwise probe module | `routing/layerwise.py`, extend `model_dependent.py` | C3 |
| 4 | Dynamics columns in merge | `data.py`, `constants.py` | C3 |
| 5 | Cross-campaign pattern compare | `routing/compare_generalization.py` | C1, C2 |
| 6 | Dynamics screen CLI | `run.py screen-dynamics` | C3 gate |
| 7 | CUDA batch oracle script | `scripts/batch_oracle_gpu.sh` | Long runs |

### 5.7 Reproducibility checklist (end of sprint)

- [ ] `reproducibility_manifest.json`: model IDs, HF revision hashes, seeds, torch version
- [ ] `doctor` passes per campaign
- [ ] `analysis/splits.json` unchanged for ARC
- [ ] `selected_feature.json` unchanged (D46)
- [ ] Failed pilots cited in appendix (Qwen 1.5B/3B, GSM8K)

---

## 6. Evaluation protocol (what to compare across campaigns)

**Do not build AUROC leaderboards across campaigns.**

For C1 and C2, report **structural invariants**:

| Metric | Invariant claim |
| ------ | --------------- |
| $d(H_w)$: opportunity vs too-hard | Near zero on all campaigns |
| $d(\Delta m_{\mathrm{gain}})$: opportunity vs too-hard | Large positive on all campaigns |
| AUROC ordering | Escalation ≥ uncertainty ≥ complexity |
| Recovery matrix | Highest opportunity cell = high $\Delta m_{\mathrm{gain}}$ |

For C3, report **terminal vs layerwise** (primary scalar: `stabilization_layer`):

| Test | Pass for RH5 |
| ---- | ------------ |
| F7 curves separate buckets | Primary |
| d(`stabilization_layer`, opp vs too-hard) > d(\(H_w\)) | Strong support |
| Partial ρ(`stabilization_layer`, opp \| \(m_w\)) > 0 | Non-redundancy |
| `slope_margin` AUROC / effect size | Supplementary only — not a gate |
| None of above | Null result — still publish RH5 as tested |

---

## 7. Paper structure (final)

```text
1. Introduction        — two-property claim; pre-decoding information
2. Related work        — routers vs information characterization
3. Method              — signals, oracle, supervision, RH1–RH7
4. Setup               — C0 primary; C1–C3 generalization overview
5. Results
   5.1 Oracle landscape (ARC)
   5.2 Dimension characterization (ARC) — existing
   5.3 Complementarity (ARC)
   5.4 Two-property synthesis
   5.5 Architecture generalization (C1)
   5.6 Domain generalization (C2)
   5.7 Confidence dynamics (C3)
6. Routing evaluation  — brief; exploitation gap
7. Discussion          — what survives if routing never improves
8. Conclusion
Appendix               — failed pilots, md variants, calibration
```

---

## 8. Risk register

| Risk | Mitigation |
| ---- | ---------- |
| Qwen 3B/7B 0% opportunity | Screen G-C1; fallback Qwen 3B + Llama 8B cross-family |
| MMLU loader delay | Week 1 infra priority; BoolQ as backup (screen exists) |
| Dynamics null result | Frame as RH5 tested; paper still strong on RH2 |
| Scope creep | Manifest whitelist; no new campaigns without D-register |
| Time slip | Drop C2 before C1 or C3; never drop C0 narrative |

---

## 9. What to do Monday morning

```bash
# 1. Narrative (no GPU)
#    Edit intro/discussion two-property claim

# 2. Fix MMLU loader (blocks C2 screen)
#    Implement load_mmlu_queries in datasets.py

# 3. Screen Qwen (2 GPU-hours)
uv run python scripts/run.py oracle \
  --weak Qwen/Qwen2.5-3B-Instruct \
  --strong Qwen/Qwen2.5-7B-Instruct \
  --dataset arc_challenge --splits-json analysis/splits.json \
  --split-role test --limit 50 --seed 42 \
  --max-new-tokens 8 --device cuda --dtype bfloat16 \
  --output experiments/campaigns/C1_qwen_arc/M4/screen_n50.json

uv run python scripts/run.py summarize-c2 --oracle ... --output analysis/c2_qwen_arc_screen.json

# 4. If G-C1 passes → run_campaign.sh C1 full
# 5. Parallel: implement layerwise.py smoke on ARC n=10
```

---

## 10. Explicit non-goals

- Track 2 geometry (PCA, F_state_geometry) — **deferred**
- GSM8K as primary or co-primary
- Representation stability / paraphrases
- Neural router, RL, graph routing
- GPT-4 / closed API models
- Re-running D46 or changing `piece_count`

---

*This plan implements the advisor’s four priorities with gates, ~3–4 GPU-days total, and infrastructure that scales without artifact chaos.*
