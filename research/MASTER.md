# Research Master Document

> **Single source of truth (science layer)** — permanently frozen D56 (2026-06-22).  
> **Experiments & scripts:** `10_experiment_registry.md` · `WORKFLOW.md` · `../experiments/README.md` · **Decisions:** `09_decision_register.md`

---

## 1. One sentence

This project characterizes **routing-relevant information in unsupervised pre-inference signals**—representative model-independent \(c(q)\), plus model-dependent \(H\) and \(m\)—for LLM selection between Llama-3.2-1B and 3B on ARC. Study IV demonstrates whether a simple policy can exploit whatever information exists; MMLU tests transfer. Signals are measured **before full model generation** (pre-inference).

**Headline contribution (ACL):** **unsupervised pre-inference signal extraction and characterization for routing** — not “unsupervised routing” (Study IV uses supervised calibration on CALIB) and not “we built a router.”

**Acceptable alternates in prose:** *routing based on unsupervised pre-inference signals* · *unsupervised pre-inference routing signals* (signals only, not the full system).

**Avoid as contribution headline:** *unsupervised routing* — reviewers will ask why logistic regression on CALIB is unsupervised.

### Goal vs contribution (do not conflate)

| Layer | What it is | Wording |
| ----- | ---------- | ------- |
| **Research goal** | Why routing matters | Different queries require different reasoning capability; using one LLM for every query is inefficient |
| **Scientific object** | What we study | **Unsupervised pre-inference signals** — model-independent features \(c(q)\) and model-dependent probe statistics \(H, m\) forming a feature vector that may govern routing |
| **Application** | Demonstration (Study IV) | Simple calibrated policy: route to weakest model expected to succeed; escalate when signals indicate benefit |

**Not the contribution:** “We built a router.” **The contribution:** estimate and characterize those signals; routing is the **demonstration** that exploits whatever information Studies I–III establish.

### Contribution shift (advisor — D63/D64)

Weak framing (router-centric):

```text
Query → Signals → Router → LLM     ← router looks like the contribution
```

Strong framing (signal-centric):

```text
Query → Unsupervised signal extraction → Signal characterization → Simple routing policy (demonstration)
```

Emphasis: **“Estimate those signals.”** Model-independent features · model-dependent features · feature vector · how features govern routing.

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
           Routing-relevant information?     ← Studies I–III: how much, from which family?
                        │
                        ▼
           Simple routing policy            ← Study IV: can we exploit what exists? (demonstration)
         (weak default; escalate if needed)
                        │
                        ▼
              Generate final answer
```

---

## 2. Contributions

| Layer | Content | Studies | Paper share |
| ----- | ------- | ------- | ----------- |
| **Characterization** | Quantify informativeness of model-independent and model-dependent unsupervised routing signals | I + II | ~50% |
| **Understanding** | Test whether the two signal families provide complementary information | III | ~20% |
| **Utility** | Can a simple policy **exploit whatever information exists**? | IV | ~20% |

**Contribution pathway:**

```text
Information  →  Characterization  →  Understanding  →  Utility
   (signals)        (I + II)              (III)          (IV)
```

Routing is a **demonstration**, not the research object. Primary science: **how much routing-relevant information** inexpensive pre-inference signals carry, and how the families relate. Study IV is meaningful only because Studies I–III establish whether exploitable information exists.

**Hypothesis progression (each study depends on the prior):**

```text
RH1  Model-independent signals contain routing-relevant information
        ↓
RH2  Model-dependent signals contain routing-relevant information
        ↓
RH3  Together they provide additional information (complementarity)
        ↓
RH4  That information supports a simple routing policy (utility — conditional on I–III)
```

**Contribution paragraph (paper):**

> Current LLM routing methods largely rely on supervised routers, preference data, or information available only after full answer generation. We study **routing based on unsupervised pre-inference signals**: inexpensive quantities computed from the query alone (model-independent) and from a lightweight prefill probe (model-dependent). We characterize **how much routing-relevant information** these signals carry, test whether the families complement each other, and evaluate whether a simple calibrated policy can exploit whatever information exists—without using routing labels to **extract** signals.

**Contrast (intro figure / one paragraph):**

| Prior work | This work |
| ---------- | --------- |
| Query → supervised router → choose LLM | Query → extract unsupervised signals → characterize information → simple policy → choose LLM |

**Not this paper:** entropy-only router · RouteLLM-style learned router · agent orchestration · task decomposition · graph routing.

**Future-work roadmap (one line each — not v1):** agents and orchestration; learned parameter tuning beyond simple logistic; larger pools; paraphrase stability.

### Framing hierarchy (advisor guidance — do not conflate)

```text
1. Main idea     →  routing based on unsupervised pre-inference signals
2. How           →  signal extraction + characterization: model-independent | model-dependent
3. When          →  before full model generation (pre-inference)
4. Closure       →  simple calibrated policy (Study IV — demonstration, not headline)
```

> The **signals** are the research object; the **router** is the demonstration. Unsupervised applies to **signal computation**, not to the entire system (Study IV fits logistic weights on CALIB).

### Terminology (locked — avoid reviewer traps)

| Use in prose | Avoid as headline | Why |
| ------------ | ----------------- | --- |
| **Unsupervised pre-inference signal extraction for routing** | *unsupervised routing* | Study IV uses oracle-derived labels on CALIB for logistic calibration |
| **Routing based on unsupervised pre-inference signals** | *we built a router* | Contribution is characterization, not a routing product |
| **Unsupervised routing signals** (the signals themselves) | conflating signals with the full pipeline | Signals are unsupervised; policy calibration is separate |

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

**RQ (frozen):** How informative are **unsupervised pre-inference routing signals** for LLM selection **before full model generation**?

**Scientific reading (same question):** How much **routing-relevant information** is present in inexpensive pre-inference signals—model-independent and model-dependent—and can a simple policy exploit whatever information exists?

*Appropriate selection* means a desirable **cost–quality trade-off** within a fixed pool: route to the **weakest model expected to answer correctly**, escalating to a stronger model only when pre-inference signals indicate that additional capability is likely to improve the outcome—not “pick the strongest model” or a vague “suitable LLM.” Offline, routing opportunity is \(y_{\text{opp}} = 1\) when the weak model fails and the strong model succeeds (`07`).

### Operational definition of *informativeness*

We use **informativeness** operationally—not as Shannon mutual information \(I(S; Y)\). A signal is informative to the extent it shows **predictive association with routing need** (offline oracle buckets, routing opportunity, weak–strong correctness gap), measured by:

- Spearman \(\rho\) (+ bootstrap CI) vs routing need / oracle gap  
- AUROC / AUPRC for opportunity detection  
- **Complementary predictive gain:** \(\Delta\)AUROC when adding a signal family beyond another (Study III)

Weak or null association is a **valid scientific finding** if reported rigorously—but **changes the paper’s emphasis** (see §3b).

| ID | Layer | Hypothesis |
| -- | ----- | ---------- |
| **RH1** | Characterization | Model-independent signals contain measurable routing-relevant information. |
| **RH2** | Characterization | Model-dependent probe signals contain measurable routing-relevant information. |
| **RH3** | Understanding | Together, the families provide additional information beyond either alone. |
| **RH4** | Utility | A simple routing policy can **exploit whatever information exists** for cost–quality gains over static baselines (conditional on I–III). |

### §3b Outcome scenarios (null results — D64)

| Outcome | RH1–RH3 | RH4 | Paper emphasis |
| ------- | ------- | --- | -------------- |
| **Mixed / positive** | Some signals informative | Utility may improve | Information present → simple policy helps |
| **Uniformly null** (e.g. all AUROC ≈ 0.50 on corrected TEST) | Rejected | Unlikely meaningful utility | **Limits paper:** empirical characterization of what these signal families *do not* provide under this protocol—not a practical routing method |

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

### Protocol

| Item | Value |
| ---- | ----- |
| Prompt protocol | v1 (`05` §1) |
| Oracle | Greedy decode, MCQ letter match, `max_new_tokens=8` (`07`) |
| Prefill probe | Canonical signal extraction (`05`) |
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
