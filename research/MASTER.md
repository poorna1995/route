# Research Master Document

> **Single source of truth (science layer)** — permanently frozen D56 (2026-06-22).  
> **Experiments & scripts:** `10_experiment_registry.md` · `WORKFLOW.md` · `../experiments/README.md` · **Decisions:** `09_decision_register.md`

---

## 1. One sentence

This project investigates whether **unsupervised pre-inference signals can support multi-LLM routing** in a fixed pool (Llama-3.2-1B / 3B on ARC-Challenge). To answer that routing question, we **extract unsupervised pre-inference signals** (model-independent \(c(q)\) + model-dependent \(H\), \(m\)), **characterize** routing-relevant information by **information dimension** (Studies I–III), and **evaluate** whether a calibrated policy can exploit it (Study IV). Signals are computed **before full model generation** (pre-inference).

**One-line problem (advisor):** Given a query, without supervision at signal time, estimate useful pre-inference signals that help **decide which LLM should solve it**.

**Headline contribution (ACL):** A **systematic empirical study** of whether unsupervised pre-inference signals can support multi-LLM routing — including what routing-relevant information they carry, how dimensions differ and complement each other, and what a calibrated policy can exploit.

**The taxonomy supports the science; do not lead with “we propose a framework.”**

**Acceptable in prose:** *unsupervised pre-inference signal-based routing* · *routing based on unsupervised pre-inference signals* · *unsupervised routing* **only when defined once** as routing driven by unsupervised signals (policy calibration on CALIB is the **last** step, not the research object).

**Avoid as sole headline:** *Characterizing routing information* (metrology only) · *we built a router* · *signal framework* as primary novelty claim.

### Professor's mental model (transcript — D66)

```text
Research problem: Can we perform unsupervised routing?
        │
        ▼
Need unsupervised signals (extract before generation)
        │
        ▼
Need to understand those signals (characterization — how we answer)
        │
        ▼
Need to know whether they help routing (complementarity + limits)
        │
        ▼
Simple routing demonstration (Study IV — last, not first)
```

**Central insight:** The professor asks a **routing question** but wants the paper to answer it **through signal characterization**, not through a sophisticated router.

### Goal vs contribution (do not conflate)

| Layer | What it is | Wording |
| ----- | ---------- | ------- |
| **Research problem** | Can unsupervised pre-inference signals support routing? | **Unsupervised routing** (using unsupervised signals) |
| **How we answer** | Empirical study of routing-relevant information + limits | **Signal characterization** (Studies I–III) — not a separate paper topic |
| **Scientific contribution** | What we learn about signals | **Empirical understanding** of pre-inference routing signals |
| **Routing evaluation** | Exploitability under calibrated policy | Study IV — **last**; not a product claim |

**The contribution is not** “we built a router” or “we proposed a framework.” **The contribution is** empirical evidence on **how much routing-relevant information exists before inference, and what its limits are** — in service of the routing problem.

### Framing (advisor — D63/D64/D65)

Weak framing (characterization-only):

```text
Query → Signals → measure information     ← sounds like a metrics paper
```

Strong framing (routing problem, characterization as means):

```text
Problem: Can unsupervised pre-inference signals support routing?
    ↓
Extract signals (model-independent | model-dependent)
    ↓
Characterize routing-relevant information (Studies I–III)
    ↓
Test calibrated routing policy (Study IV)
    ↓
Generate answer with chosen LLM
```

Emphasis: **routing is the objective**; **signal characterization is how we answer** whether these signals are viable for routing. Negative routing results (calibrated policy $\approx$ always-strong) are publishable if they answer the research question.

### Conceptual pipeline

```text
                    Query q
                        │
         ┌──────────────┴──────────────┐
         │                             │
         ▼                             ▼
  Model-independent              Model-dependent
  signals c(q)                   probe signals H(q), m(q)
         │                             │
         └──────────────┬──────────────┘
                        ▼
           Characterize: how much routing-relevant information?   ← Studies I–III (method)
                        │
                        ▼
           Test: can a calibrated policy exploit it?                 ← Study IV (routing evaluation)
         (weak default; escalate if needed)
                        │
                        ▼
              Generate final answer
```

---

## 2. Contributions

### Contribution hierarchy (ranked — D66)

| Rank | Contribution | Role |
| ---- | ------------ | ---- |
| **1. Primary** | Systematic **empirical study** of pre-inference routing signals (routing-relevant information, limits, interpretation) | The science |
| **2. Secondary** | Taxonomy of **information dimensions** + prefill-based extraction | Supports the science |
| **3. Tertiary** | Evidence on **complementarity** across families (ΔAUROC ladder) | Structure of information |
| **4. Fourth** | Simple routing policy — how much information is **exploitable** | Utility demonstration (last) |

### Paper presentation: four questions (not “Study I–IV” in prose)

Use these in **Results**; map to studies internally only.

| Question | Answer (ARC, TEST) | Studies |
| -------- | ------------------ | ------- |
| **Q1** Is routing-relevant information present pre-inference? | **Yes** — ρ and AUROC above chance | I–II |
| **Q2** How much? What are the limits? | **Modest** — AUROC ≈ 0.54–0.61; oracle gap ρ ≈ 0.18 | I–II + interpretation |
| **Q3** What aspects of routing need do different dimensions encode? | Difficulty (H_w, c(q)) vs recoverability (Δm_gain); dimensions partially complementary (+ΔAUROC 0.049) | III + interpretation |
| **Q4** Can a calibrated policy exploit it? | **Partially / not fully** — calibrated policy $\approx$ always-strong (69.2\%); oracle 74.4\% | IV |

**Overall answer to the routing question:** **Partially.** Different pre-inference signals encode different aspects of routing need (difficulty vs recoverability). Query proxies and weak-model probes track difficulty/uncertainty; cross-model disagreement tracks recoverability. A simple linear policy captures only part of the available oracle improvement.

```text
Query proxies          → coarse difficulty
Weak-model probes      → model uncertainty
Cross-model disagreement → recoverability
                         ↓
Simple routing captures only part of this structure
```

| Layer | Content | Studies | Paper share |
| ----- | ------- | ------- | ----------- |
| **Empirical characterization** | Q1–Q2: presence and magnitude | I + II | ~35% |
| **Information structure** | Q3: complementarity, decomposition, calibration | III + interpret | ~30% |
| **Routing evaluation** | Q4: exploitability | IV | ~15% |
| **Framework + setup** | Dimensions, extraction, pool | Method + §4 | ~20% |

**Contribution pathway:**

```text
Routing problem  →  Unsupervised signal extraction  →  Signal characterization  →  Routing evaluation
```

Studies I–III answer **whether and how** unsupervised signals carry routing-relevant information. Study IV answers **whether a minimal policy can use it**—not whether we beat supervised routers. A weak Study IV result (policy ≈ always-strong) is consistent with the routing problem: signals exist but are **limited** for simple exploitation.

**Hypothesis progression (each study depends on the prior):**

```text
RH1  Unsupervised pre-inference signals carry routing-relevant information
        ↓
RH2  Information dimensions encode distinct aspects of routing need
        ↓
RH3  Dimensions provide complementary information
        ↓
RH4  A calibrated policy can exploit available information (conditional on I–III)
```

**Contribution paragraph (paper):**

> We investigate whether **unsupervised pre-inference signals can support multi-LLM routing**. Through a systematic empirical study on ARC-Challenge, we show that weak-model entropy and cross-model probe disagreement carry measurable routing-relevant information before full generation, that model-independent query proxies add modest complementary signal, and that the two families are partially redundant yet incrementally informative when combined. A simple calibrated routing policy does not outperform always routing to the strong model, indicating that **current simple exploitation captures only part of the oracle routing improvement**—not that the signals are uninformative. Signal extraction is unsupervised; oracle labels evaluate informativeness and calibrate the demonstration policy on CALIB only.

**Contrast (intro figure / one paragraph):**

| Prior work | This work |
| ---------- | --------- |
| Query → supervised router → choose LLM | Query → unsupervised signals → characterize → calibrated policy → choose LLM |

**Not this paper:** entropy-only router · RouteLLM-style learned router · agent orchestration · task decomposition · graph routing.

**Future-work roadmap (one line each — not v1):** agents and orchestration; learned parameter tuning beyond simple logistic; larger pools; paraphrase stability.

### Framing hierarchy (advisor guidance — do not conflate)

```text
1. Problem       →  unsupervised routing (can pre-inference signals support it?)
2. Science       →  empirical understanding of routing-relevant information + limits
3. Method        →  characterization answers the routing question (not a separate goal)
4. Taxonomy      →  model-independent | model-dependent (supports science)
5. Closure       →  routing evaluation (last step — exploitation test)
```

> **Routing is the problem; characterization is how we answer it; the router is the last step.** Reviewers must not mistake this for “another routing paper”—the contribution is empirical understanding in service of unsupervised routing.

### Terminology (locked — avoid reviewer traps)

| Use in prose | Avoid as sole headline | Why |
| ------------ | ---------------------- | --- |
| **Unsupervised pre-inference routing signals** | *characterizing routing information* | Problem is routing, not metrology |
| **Routing based on unsupervised pre-inference signals** | *we built a router* | Study IV tests feasibility, not SOTA |
| **Unsupervised routing** (define once) | using without definition | Means routing *using* unsupervised signals |
| **Unsupervised routing signals** | conflating signals with full pipeline | Signals are unsupervised; policy step is separate |

### Terminology (locked — four distinct concepts)

| Concept | Meaning | Role in this work |
| ------- | ------- | ----------------- |
| **Unsupervised** | No routing labels used to **compute** signals \(c, H, m\) | Signal-extraction paradigm |
| **Routing signals** | Quantities intended to reflect routing-relevant information (e.g. \(c(q)\), \(H\), \(m\)) | **Research object** |
| **Model-independent / model-dependent** | Two families in taxonomy \(\mathcal{S}\) | Feature-vector structure |
| **Before full model generation** | Signals available before complete answer generation; prefill probe permitted | Operational constraint |
| **Calibrated policy (Study IV)** | Logistic + threshold fit on CALIB from oracle-derived \(y_{\text{opp}}\) | **Demonstration** — not primary contribution |

**Canonical phrase:** *unsupervised pre-inference routing signal* (the signal, not the system). Define once: a routing signal is **pre-inference** if it can be computed before complete model generation.

**Scientific core of the RQ:** *How much routing-relevant information is present in inexpensive pre-inference signals?* Formal RQ below uses *informativeness* as the operational term.

---

## 3. Research question & hypotheses

**RQ (frozen):** Can **unsupervised pre-inference signals support multi-LLM routing** before full model generation?

**Operational sub-question (Studies I–III):** How much **routing-relevant information** do those signals carry, and do **information dimensions** complement each other?

**Operational sub-question (Study IV):** Can a **simple calibrated policy** exploit whatever information exists under a cost–quality trade-off?

*Appropriate selection* means a desirable **cost–quality trade-off** within a fixed pool: route to the **weakest model expected to answer correctly**, escalating to a stronger model only when pre-inference signals indicate that additional capability is likely to improve the outcome—not “pick the strongest model” or a vague “suitable LLM.” Offline, routing opportunity is \(y_{\text{opp}} = 1\) when the weak model fails and the strong model succeeds (`07`).

### Operational definition of *informativeness*

We use **informativeness** operationally—not as Shannon mutual information \(I(S; Y)\). A signal is informative to the extent it shows **predictive association with routing need** (offline oracle buckets, routing opportunity, weak–strong correctness gap), measured by:

- Spearman \(\rho\) (+ bootstrap CI) vs routing need / oracle gap  
- AUROC / AUPRC for opportunity detection  
- **Complementary predictive gain:** \(\Delta\)AUROC when adding a signal family beyond another (Study III)

Weak or null association is a **valid scientific finding** if reported rigorously—but **changes the paper’s emphasis** (see §3b).

| ID | Layer | Hypothesis |
| -- | ----- | ---------- |
| **RH1** | Characterization | Unsupervised pre-inference signals carry measurable routing-relevant information. |
| **RH2** | Characterization | Information dimensions encode distinct aspects of routing need. |
| **RH3** | Understanding | Dimensions provide complementary information beyond any single dimension. |
| **RH4** | Routing evaluation | A calibrated policy can exploit available information under a cost–quality objective (conditional on I–III). |

### §3b Outcome scenarios (null results — D64)

| Outcome | RH1–RH3 | RH4 | Paper emphasis |
| ------- | ------- | --- | -------------- |
| **Mixed / positive** | Some signals informative | Utility may improve | Information present → calibrated policy helps |
| **Uniformly null** (e.g. all AUROC ≈ 0.50 on corrected TEST) | Rejected | Unlikely meaningful utility | **Limits paper:** empirical characterization of what these signal families *do not* provide under this methodology—not a practical routing method |

A uniformly null result is still publishable as ACL-style empirical science; Abstract/Intro/Discussion must foreground **understanding limits**, not enabling routing.

---

## 4. Workflow

```text
Research Question → Problem → Literature Gap → Signal Taxonomy
        ↓
Validation (V1/V2)
        ↓
D46 screening — representative c(q) on CALIB only  [not a paper experiment]
        ↓
Study I   — Characterization: model-independent     [RH1]
        ↓
Study II  — Characterization: model-dependent        [RH2]
        ↓
Study III — Understanding: complementarity           [RH3]
        ↓
Study IV  — Utility: lightweight routing             [RH4]
        ↓
Cross-dataset validation — MMLU (generalization)
        ↓
Discussion               — BoolQ optional (robustness)
```

Study ↔ notebook ID mapping: `10_experiment_registry.md` only.

---

## 5. Frozen configuration

### Signals (three headline probes)

| Signal | Family | Role | Definition |
| ------ | ------ | ---- | ---------- |
| **\(c(q)\)** | Model-independent | **Representative** complexity signal | Selected per D46 screening (`18`); exact formula recorded in `05` §8 when implemented |
| **\(H_i(q)\)** | Model-dependent | Uncertainty | Mean token entropy, prefill probe |
| **\(m_i(q)\)** | Model-dependent | Confidence separation | Mean \(\log p_{(1)}-\log p_{(2)}\) margin, prefill |

**\(c(q)\) is not an arbitrary heuristic.** One representative is chosen from the model-independent complexity family via the documented screening process (`18` §3–§7). The exact formula lives in `05` §8 after implementation—not in methodology prose until evaluated.

**Tokenization (D58):** Length/diversity/entropy candidates use the **Llama HF tokenizer** on raw `user_content` (no forward pass). Compression uses raw UTF-8. This aligns segmentation with the study pool without making \(c(q)\) model-dependent.

**Margin stays (D48):** entropy measures uncertainty; margin measures confidence separation. Testing redundancy and complementarity between them is core science (RH3), independent of any single advisor mention.

Derived: \(\Delta H = H_w - H_s\), \(\Delta m = m_s - m_w\).

**Deferred:** paraphrase (D04) · full lexical grids · hidden-state probes.

### Model pool

| Role | Purpose | Model |
| ---- | ------- | ----- |
| Weak | **Primary** characterization | `meta-llama/Llama-3.2-1B-Instruct` |
| Strong | **Primary** characterization | `meta-llama/Llama-3.2-3B-Instruct` |

**Generalization (optional appendix only):** Qwen 1.5B↔3B or Llama 8B↔70B — do not redesign primary pool.

### Datasets (by scientific purpose)

| Purpose | Dataset | Why | Size |
| ------- | ------- | --- | ---- |
| **Primary characterization** | ARC-Challenge test | Validated opportunity distribution (V2); MCQ oracle | 1,172 |
| **Generalization** | MMLU: `high_school_physics`, `logical_fallacies` | Different subject/format — tests whether signal patterns transfer | ~200–400 |
| **Robustness** (optional) | BoolQ | Yes/no format stress-test | if time |

**Rejected as primary:** GSM8K + Qwen (smoke failed) · full MMLU · benchmark sprawl.

### Splits (seed 42)

```text
ARC train (1,119)     →  NOT USED (no LLM/router training; evaluation-focused study)
ARC validation (299)  →  CALIB: D46 screening, Study III/IV logistic calibration
ARC test (1,172)      →  TEST: Studies I–IV reported results
MMLU                  →  TEST only; τ, λ from ARC validation CALIB
```

**Paper sentence (Methods):** *The official ARC-Challenge training split was not used, as our study does not train language models or discover task-specific representations. The official validation split was reserved exclusively for representative signal selection and router calibration; all reported results are computed on the official test split.*

### Experimental configuration

| Item | Value |
| ---- | ----- |
| Prompt formatting | Deterministic chat template (`05` §1) |
| Offline oracle | Greedy decode, MCQ letter match, `max_new_tokens=8` (`07`) |
| Prefill probe | Canonical model-dependent signal extraction (`05`) |
| Seed | 42 |

---

## 6. Four studies (paper naming)

| Study | Layer | Name | Outputs |
| ----- | ----- | ---- | ------- |
| **I** | Characterization | Model-independent | ρ, AUROC, distributions for representative \(c(q)\) |
| **II** | Characterization | Model-dependent | ρ, AUROC for \(H, m\); partial ρ given \(c\) |
| **III** | Understanding | Complementarity | **Primary:** ΔAUROC ladder \(c → c{+}H → c{+}H{+}m\) · **Secondary (appendix):** \(H → H{+}m\) |
| **IV** | Utility | Lightweight routing | Rule-based or logistic policy on CALIB; baselines: always-weak/strong, oracle |

### Metrics (do not add more)

Spearman ρ + bootstrap CI · AUROC + CI · bucket separability · ΔAUROC · probe cost.

---

## 6b. Architecture freeze (D62 — execute only)

**No further methodology redesign.** Remaining work: run CALIB → D46 → TEST pipeline, write paper.

```text
prompt_protocol → oracle → query features → probe signals → merge → EXP-01 (I–II) → EXP-02 (III) → EXP-03 (IV)
```

| Layer | Frozen decision |
| ----- | --------------- |
| D46 screening | ARC validation (~299) once; composite norm(ρ)+norm(AUROC); candidate ρ matrix + partial ρ diagnostic |
| Study III | Primary c→c+H→c+H+m; secondary H→m appendix |
| Study IV | **Independent** logistic on CALIB; tune τ on CALIB; evaluate on TEST (not Study III weights) |
| MMLU | Transfer table only |
| Stability | Bootstrap on TEST; optional 5 query-subset re-analyses (no re-inference) |

Deferred post-paper: `Signal` class abstraction · rank+stability D46 rescoring · full 7-model complementarity grid in main text.

### Supervision (three layers — state once in Methods; D64)

| Layer | Supervised? | Role |
| ----- | ----------- | ---- |
| **Signal computation** (\(c, H, m\)) | **No** | Unsupervised extraction — primary contribution |
| **Signal characterization** (Studies I–III) | Oracle labels **offline only** | Evaluate ρ, AUROC, ΔAUROC — not used to define signals |
| **Routing policy** (Study IV) | **Yes on CALIB** — logistic/threshold from \(y_{\text{opp}}\) | Simple **demonstration** that exploits signal information; not a learned neural router |

Reviewer FAQ: *“If you fit logistic regression on CALIB, why is this unsupervised?”*  
→ **Signal computation** is unsupervised. **Characterization** uses oracle labels only to measure informativeness. **Routing policy** is supervised calibration on top of those signals—it is not the primary contribution and is not claimed to be label-free.

### Router (Study IV — frozen)

- **Hold-out utility validation:** fit logistic \(P(y^{\text{opp}} \mid c, H_w, m_w)\) on CALIB; tune threshold τ on CALIB; report accuracy + cost on TEST only  
- **Independent from Study III:** same feature set allowed; **do not** reuse complementarity model artifacts or thresholds  
- **Simple only:** logistic + threshold (optional cost penalty λ on CALIB); **no** deep / neural / RL router  
- **Sanity only:** `route-preview` median heuristics (D37) — not EXP-03

---

## 7. Implementation (not science — see experiments/)

All scripts, artifact paths, and run commands live in:

- `10_experiment_registry.md` — validation runs, EXP IDs, status  
- `../experiments/README.md` — how to execute the pipeline

---

## 8. Out of scope

Paraphrase · 20-heuristic ablation · GSM8K primary · full MMLU · RouteLLM/GNN training · agents · methodology redesign.

**Future work (one line each):** paraphrase stability; agents; larger pools; BoolQ.

---

## 9. Next actions (paper-first — D63/D64)

**Guiding principle:** Every **experiment** must justify a **paragraph**; every **paragraph** must justify a **hypothesis**; every **hypothesis** must answer the **RQ**. If an experiment does not fill a row in RH1–RH4, it is future work.

**Paper-first workflow (not “invent conclusions first”):**

```text
Research question → Hypotheses (RH1–RH4) → Table shells (T2–T4) → Experiments → Fill tables
```

| Step | Deliverable |
| ---- | ----------- |
| 1 | Draft `paper/draft/01–06` with TBD cells + RH1–RH4 prose — `11_paper_outline.md` |
| 2 | Complete CALIB oracle (299) + fix ARC label normalization |
| 3 | D46 screen → freeze `selected_feature.json` → record D46 in `09` |
| 4 | TEST: oracle + probes + merge → EXP-01 (Studies I–II) |
| 5 | EXP-02 (Study III), EXP-03 (Study IV), MMLU transfer table |
| 6 | Write Abstract/Discussion from **actual** outcome scenario (§3b) |

---

## 10. Other documents (specialized — not duplicates)

Read **this file first**. Use others only for depth:

| File | Use when you need |
| ---- | ----------------- |
| `05_computation_protocol.md` | Exact H, m, prefill probe math |
| `07_oracle_definition.md` | Bucket definitions |
| `04_signal_design.md` | Taxonomy 𝒮, literature rationale |
| `03_literature_gap.md` | Related work positioning |
| `09_decision_register.md` | Why decision Dxx was made |
| `10_experiment_registry.md` | V1/V2/EXP ID traceability |
| `11_paper_outline.md` | Section → draft file map |
| `claims.md` | Claim → evidence dashboard |
| `14_literature_record.md` | Paper deep-read notes |
| `18_model_independent_signal_selection.md` | \(c(q)\) literature survey + D46 screening candidates |
| `notes.md` | Daily lab notes (ephemeral) |
