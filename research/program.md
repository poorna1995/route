# Research program

> unsupervised signals for **LLM routing** + weight learning in a **fixed LLM pool of models with distinct capabilities**. see §0.1a.  
> **Vocabulary frozen 2026-06-28** — [`nomenclature.md`](nomenclature.md). No term changes unless compelling technical reason.  
> **Related work:** [`literature_record.md`](literature_record.md)

**Canonical sections:** **§0 research design** · **§0.7 methodology (Parts I–IV)** · signal assumptions **§5** · unsupervised §2 · routing need §3 · routing policy §8 (refs [`nomenclature.md`](nomenclature.md) **§2.1** for \(s\), \(\pi\))

---

## 0. Research design (Stage 0)

### 0.1 Title and pitch

**Working title: _Routing from Unsupervised Signals in a Fixed LLM Pool_**

Given a fixed pool of language models with different capabilities, can pre-defined routing signals computed without routing supervision identify which model is appropriate for a query?

**30 s pitch:**
Current LLM routers are typically trained from supervision such as preference labels, oracle routing decisions, reward scores, or model outcomes. In contrast, we investigate whether routing-relevant information already exists in signals that can be computed without routing labels during signal extraction.

We organize these signals into three families:

1. **Model-independent (ϕ):** derived from the query alone.
2. **Model-dependent (ψ):** derived from the response of a candidate model.
3. **Cross-model (χ):** derived from comparisons between models (offline analysis).

Using an oracle-defined routing label r(q) only on a calibration set, we evaluate which signal families contain routing information, freeze a routing feature specification x(q), optionally learn a lightweight routing policy π(q), and evaluate the resulting accuracy–cost trade-off on held-out test queries.

**Contributions**

This paper makes four contributions.

1. **Signal taxonomy (C1)** — predefined routing signals computed without routing supervision during extraction (Stage 5).

2. **Signal validation (C2)** — statistical evidence that model-independent, model-dependent, and cross-model families associate with oracle routing need (Stage 6).

3. **Signal combination and policy calibration (C3)** — a routing score \(s(q)\) and threshold \(\tau\) learned from validated signals on \(R_c\) (Stage 8).

4. **Routing evaluation (C4)** — accuracy–cost of the frozen policy on held-out \(R_t\) (Stage 9).

**One-line problem (scientific):** Can pre-specified, label-free routing signals provide sufficient information for model selection in a fixed LLM pool?

**Three core objects:**

| Object | Symbol | Stage |
|--------|--------|-------|
| Signals | φ, ψ, χ | 5 — define |
| Representation | \(x(q)\) | 7 — freeze |
| Policy | \(\pi(q)\), \(s(q)\) | 8 — calibrate; 9 — evaluate |

**Scope**

We study routing within a fixed model pool

M={M1,…,MK},

where models have distinct capabilities and computational costs.

Currently, ∣M∣=2,

consisting of

1. Llama-3.2-3B-Instruct
2. Llama-3.1-8B-Instruct

Later work may extend the framework to larger model pools.

**0.1 Paper Voice (locked)**

Use throughout the paper:

1. LLM routing
2. Model selection
3. Select the appropriate model
4. Routing policy
5. Fixed model pool

Avoid as contribution language:

1. "Cascade routing"
2. "Escalation framework"
3. "Sequential invocation"

Those describe one possible deployment strategy, not the scientific contribution.

The routing policy is

π(q):x(q)→Mi,

which selects one member of the model pool.

**0.2 Motivation**

Large language models exhibit a well-known trade-off between inference cost and answer quality. Running the strongest available model for every query maximizes accuracy but incurs unnecessary computational cost, while always selecting a smaller model sacrifices quality on difficult queries. The central challenge of LLM routing is therefore to select the most appropriate model for each query.

Existing routing systems—including RouteLLM, HybridLLM, RouterBench, and related methods—typically formulate routing as a supervised prediction problem. They learn a direct mapping from queries (or model outputs) to routing decisions using preference labels, oracle outcomes, reward models, or other forms of supervision.

This naturally raises a different research question:

How much routing-relevant information is already present in signals that can be computed without routing supervision?

Rather than learning a router directly from routing labels, we first investigate the signals themselves. We define routing signals that require no routing labels during extraction, organize them into model-independent, model-dependent, and cross-model families, and evaluate whether they are associated with oracle routing need.

This separation is important. The proposed signals are label-free to compute and can therefore be extracted for any query without access to routing supervision. Oracle routing labels are used only during an offline calibration phase to evaluate signal informativeness and, optionally, to fit a lightweight routing policy. During deployment, routing decisions are made solely from the extracted signals and a frozen routing policy.

**Setting**

We consider a fixed pool of language models

> M={M1,…,MK},

whose members exhibit different capabilities and computational costs. For each incoming query q, the routing policy selects exactly one model from this pool.

The present work focuses on the simplest non-trivial setting,

> ∣M∣=2,

where routing is a binary model-selection problem between a lower-cost and a higher-capability model.

> Can unsupervised signal extraction provide sufficient information for effective LLM routing?

**Deployment picture (professor / paper):**

```text
Query → compute signals x(q) → π(q) → select M_lo or M_hi → run that model
```

**Not our contribution:** FrugalGPT-style **cascades** (sequential invoke with feedback) — cite as related work only.

Full voice table: [`nomenclature.md`](nomenclature.md) §1.

### 0.3 Primary question and sub-questions

**Primary question (PQ):**

> Can routing-relevant information be obtained from signals that are extracted without routing supervision, and can these validated signals support effective model selection in a fixed LLM pool?

**Sub-questions (advisor formulation):**

| ID      | Question                                                                    | Maps to                              |
| ------- | --------------------------------------------------------------------------- | ------------------------------------ |
| **SQ1** | What routing-relevant signals can be extracted without routing supervision? | RQ1 · §5 · Stage **5**               |
| **SQ2** | Which signal families are informative of oracle routing need? | RQ2 · H1–H3 · Part II Stage **6** |
| **SQ3** | How can validated signals be combined into a routing score and policy? | Part II Stages **7–8** · H4 |
| **SQ4** | Does the resulting policy improve the accuracy–cost trade-off compared with simple routing baselines? | RQ3 · H5 · Part IV (Stage **9**) |

**Research questions (paper-facing):**

| ID                              | Question                                                                                                               | Hypotheses                   |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| **RQ1 : Signal Definition**     | What model-independent, model-dependent, and cross-model routing signals can be extracted without routing supervision? | — (descriptive; Stage **5**) |
| **RQ2: Signal Informativeness** | Which unsupervised routing signals are associated with oracle routing need?                                            | **H1–H3** (Stage **6**)      |
| **RQ3: Combination**            | Does combining validated families yield a more informative routing representation than any single family?             | **H4** (Stage **8**)         |
| **RQ4: Routing**                | Does the calibrated policy improve accuracy–cost vs baselines on held-out test?                                        | **H5** (Stage **9**)         |

### 0.5 Supervision model (three layers)

**Unsupervised** describes **how signals are obtained** (layer 1), not “zero labels in the entire paper.”

| Layer                     | Purpose                               | Uses oracle labels?            | Supervised?     |
| ------------------------- | ------------------------------------- | ------------------------------ | --------------- |
| **1. Signal extraction**  | Compute routing signals               | No                             | **No**          |
| **2. Signal validation**  | Measure association with routing need | Yes (offline calibration only) | Evaluation only |
| **3. Policy calibration** | Construct a routing policy            | Yes (offline calibration only) | Yes             |

Signal definitions do not depend on routing labels. Oracle labels are introduced only after signal extraction to evaluate signal informativeness and, optionally, calibrate a routing policy.

> **Labels evaluate predefined signals; they do not create new signals.**

### 0.6 Problem formulation

**Per-model correctness:** \(y(q,M)\in\{0,1\}\) under frozen protocol.

**Primary routing pair (v1):** Stage 3 designates two pool members with ordered capability: **\(M\_{\mathrm{lo}}\)** (lower capability, lower cost) and **\(M\_{\mathrm{hi}}\)** (higher capability, higher cost), with \(\mathrm{cap}(M*{\mathrm{lo}}) < \mathrm{cap}(M*{\mathrm{hi}})\). Write \(y*{\mathrm{lo}}=y(q,M*{\mathrm{lo}})\), \(y*{\mathrm{hi}}=y(q,M*{\mathrm{hi}})\). The threshold policy and \(r(q)\) are defined on this pair; the full pool \(\mathcal{M}\) may contain additional members for comparative signals and future extensions.

**Binary LLM routing (deployable):** For each query, compute signals \(x(q)\), score \(s(q)=\lambda^\top x(q)\), and **select** \(M*{\mathrm{lo}}\) or \(M*{\mathrm{hi}}\) via threshold policy \(\pi(q)\) (nomenclature §2.1). This is **model selection**, not a cascade contribution.

**Routing need \(r(q)\) (oracle label only):**

\[
r(q) = \mathbb{1}\big[\, y_{\mathrm{lo}} = 0 \;\wedge\; y_{\mathrm{hi}} = 1 \,\big], \qquad \Delta(q) = y*{\mathrm{hi}} - y*{\mathrm{lo}}
\]

**Definition (locked):** \(r(q)=1\) iff **\(M\_{\mathrm{hi}}\) was the appropriate pool member ex post** — \(M*{\mathrm{lo}}\) failed and \(M*{\mathrm{hi}}\) would succeed. Also called **escalation utility** (technical name). Not generic difficulty.

| Bucket          | \((y*{\mathrm{lo}}, y*{\mathrm{hi}})\) | \(r(q)\) | Appropriate model (oracle) |
| --------------- | -------------------------------------- | -------- | -------------------------- |
| easy            | \((1,1)\)                              | 0        | \(M\_{\mathrm{lo}}\)       |
| **opportunity** | \((0,1)\)                              | **1**    | **\(M\_{\mathrm{hi}}\)**   |
| lo_only         | \((1,0)\)                              | 0        | \(M\_{\mathrm{lo}}\)       |
| too_hard        | \((0,0)\)                              | 0        | \(M\_{\mathrm{lo}}\)       |

**\(r(q)\) is an oracle label, not \(\pi(q)\).** Computed offline from both models. The **deployable router** \(\pi(q)\) uses \(x(q)\) only and **selects** the pool member; it never observes \(r(q)\), \(y*{\mathrm{lo}}\), or \(y*{\mathrm{hi}}\) online.

**Oracle** \(\pi^\*(q)\) (eval upper bound only): route to the **cheapest correct** pool member — for the primary pair: \(M*{\mathrm{lo}}\) if \(y*{\mathrm{lo}}=1\); else \(M*{\mathrm{hi}}\) if \(y*{\mathrm{hi}}=1\); else \(M\_{\mathrm{lo}}\).

**Invariant:** \(r(q)\) is computed offline for layers 2–3; it is **never** an input to signal extraction (§5). Detail: §3.

**Important:** \(r(q)\), bucket rates, and signal behaviour emerge from the coupled Experimental Setting \(\mathcal{S}\) (§0.7). Stages **1–3** select \(\mathcal{S}\) in a reproducible order; Stage **3** freezes it before oracle runs.

# 3 Problem Formulation

## 3.1 Experimental Setting

We consider routing within a **fixed pool of language models**

$$
\mathcal{P} = \{M_1, \ldots, M_K\},
$$

whose members differ in capability and inference cost. The model pool is fixed before any routing experiments are performed.

For the present study, routing is formulated as binary model selection between a designated primary pair,

$$
(M_{\mathrm{lo}}, M_{\mathrm{hi}}),
$$

where

$$
\mathrm{cap}(M_{\mathrm{lo}}) < \mathrm{cap}(M_{\mathrm{hi}})
$$

and

$$
\kappa(M_{\mathrm{lo}}) < \kappa(M_{\mathrm{hi}}).
$$

The remaining models, if any, are not part of the routing decision in this version of the framework but may be used for comparative analyses or future extensions.

---

## 3.2 Routing Objective

For every query $q$, the routing policy

$$
\pi(q)
$$

selects exactly one model from the primary pair,

$$
\pi(q) \in \{M_{\mathrm{lo}}, M_{\mathrm{hi}}\}.
$$

The routing decision is made from a routing feature vector

$$
x(q),
$$

constructed from the predefined signal families introduced in Section 5.

A routing score

$$
s(q) = f(x(q))
$$

is computed, where $f(\cdot)$ denotes the routing policy.

In the present work we instantiate

$$
f(x) = \lambda^\top x,
$$

yielding the threshold policy

$$
\pi(q) =
\begin{cases}
M_{\mathrm{hi}}, & s(q) > \tau, \\
M_{\mathrm{lo}}, & \text{otherwise}.
\end{cases}
$$

The linear score is one implementation of the routing policy rather than a defining component of the framework.

---

## 3.3 Oracle Correctness

Under the frozen experimental protocol, model correctness is defined as

$$
y(q, M) \in \{0, 1\},
$$

where

$$
y(q, M) = 1
$$

indicates that model $M$ answers query $q$ correctly.

For the primary routing pair we write

$$
y_{\mathrm{lo}} = y(q, M_{\mathrm{lo}}), \qquad y_{\mathrm{hi}} = y(q, M_{\mathrm{hi}}).
$$

---

## 3.4 Oracle Routing Need

To evaluate routing signals, we define the oracle routing label

$$
r(q) = \mathbf{1}\left[y_{\mathrm{lo}} = 0 \land y_{\mathrm{hi}} = 1\right].
$$

This label identifies queries for which the higher-capability model is the **appropriate** choice because the lower-capability model fails while the higher-capability model succeeds.

Equivalently,

$$
r(q) = 1
$$

if and only if routing to $M_{\mathrm{hi}}$ would improve correctness over $M_{\mathrm{lo}}$.

The correctness difference

$$
\Delta(q) = y_{\mathrm{hi}} - y_{\mathrm{lo}}
$$

is reported for analysis only.

Importantly, $r(q)$ is **not** a measure of generic task difficulty. It is a routing-specific oracle label defined only with respect to the selected model pair.

---

## 3.5 Oracle Buckets

Each query belongs to one of four mutually exclusive oracle buckets.

| Bucket      | $(y_{\mathrm{lo}}, y_{\mathrm{hi}})$ | $r(q)$ | Appropriate model        |
| ----------- | -----------------------------------: | -----: | ------------------------ |
| easy        |                              $(1,1)$ |      0 | $M_{\mathrm{lo}}$        |
| opportunity |                              $(0,1)$ |      1 | $M_{\mathrm{hi}}$        |
| lo_only     |                              $(1,0)$ |      0 | $M_{\mathrm{lo}}$        |
| too_hard    |                              $(0,0)$ |      0 | none in the primary pair |

> **Note on the "too_hard" bucket.** Neither model is correct, so writing "$M_{\mathrm{lo}}$" as the appropriate model would incorrectly imply the oracle prefers it. The correct label is **"none in the primary pair"** (or **"no correct model"**). A deployment policy may still default to $M_{\mathrm{lo}}$ for cost reasons, but that is a **policy decision**, not an oracle fact.

---

## 3.6 Oracle vs Routing Policy

The oracle routing label $r(q)$ is computed **offline** from the correctness of both models. It is used exclusively for signal analysis and policy calibration.

The deployment policy $\pi(q)$ never observes

$$
r(q), \quad y_{\mathrm{lo}}, \quad y_{\mathrm{hi}}.
$$

Instead, it operates solely on the routing feature vector $x(q)$, computed from the predefined signal extraction pipeline.

---

## 3.7 Oracle Policy (Evaluation Upper Bound)

For evaluation we define the oracle routing policy $\pi^\star(q)$, which always selects the cheapest correct model.

For the primary pair,

$$
\pi^\star(q) =
\begin{cases}
M_{\mathrm{lo}}, & y_{\mathrm{lo}} = 1, \\
M_{\mathrm{hi}}, & y_{\mathrm{lo}} = 0,\ y_{\mathrm{hi}} = 1, \\
M_{\mathrm{lo}}, & y_{\mathrm{lo}} = 0,\ y_{\mathrm{hi}} = 0.
\end{cases}
$$

This oracle policy is used only as an evaluation reference and is never available during deployment.

---

# Experimental Setting

## Definition

An **Experimental Setting** is the tuple

$$
\mathcal{S} = (\mathcal{D}, \mathcal{P}, \Pi),
$$

where

- $\mathcal{D}$ denotes the benchmark,
- $\mathcal{P}$ the fixed model pool,
- $\Pi$ the evaluation protocol.

All remaining components, including the evaluation corpus, calibration/test partition, and routing feature specification, are deterministically derived from $\mathcal{S}$.

### Design process

**Part I** selects and freezes one experimental setting. **Part II–IV** conduct the routing study on that locked setting.

```text
Part I    M1 → M2 → M3     (Experimental Design)
Part II   Stages 4–8        (Development → Freeze Router)
Part III  deploy            (Online Deployment)
Part IV   evaluate on R_t   (Test Evaluation, H4)
```

#### 0.7.1 Experimental Setting object (single config, all modules)

Every module consumes the same structure. M1 **writes** the spec; M2 **evaluates** candidates; M3 **freezes** one instance; M4 **reads** it read-only.

```yaml
# Template: experiments/defaults.yaml (M1 shared) → runs/<id>/setting.yaml (M3 frozen)
setting:
  dataset:
    name: ARC-Challenge
    repo: allenai/ai2_arc
    config: ARC-Challenge

  evaluation_corpus: # C — labeled queries for this study (M1)
    splits: [validation, test] # HF loading detail; see §13 fact sheet
    exclude_splits: [train] # few-shot / train only

  partition: # fixed counts — same pilot cost across benchmarks (not %)
    method: split_dataset
    seed: 42
    selection_holdout_n: 150 # M1/M2 holdout |H|
    test_n: 150 # M3 test |R_t| (winning benchmark only)
    # |R_c| = |C| - selection_holdout_n - test_n  (remainder, M3)

  pool: # FROZEN in M1 before dataset pilot
    deployment_scenario: model_pool # model_pool | heterogeneous_pool | api_router | edge_cloud
    vendor: meta-llama
    M_lo: meta-llama/Llama-3.2-3B-Instruct
    M_hi: meta-llama/Llama-3.1-8B-Instruct
    kappa: null # relative cost M_hi / M_lo — document at M3

  protocol:
    system_prompt: ...
    dataset_template: ...
    decoding: ...
    grading: objective_mcq

  splits: # populated at M3 freeze (query IDs)
    selection_holdout: [] # filled after M2
    calib: []
    test: []
```

One source of confusion in LLM routing is mixing **experimental design**, **offline development**, **deployment**, and **evaluation** into a single linear pipeline. They are four distinct systems with different purposes and different access to oracle labels.

---

## 0.7 Methodology — four parts

The complete methodology has **four parts**. Parts I and IV are about the **experiment**; Part II builds the router **once offline**; Part III applies the frozen router **at inference time**.

```text
Part I    Experimental Design          M1 → M2 → M3
          (nothing learned; nothing deployed)

Part II   Development                  Stage 4 → 5 → 6 → 7 → 8 → Freeze Router
          (offline; uses r(q) on R_c)

Part III  Deployment                   new query → signals → x(q) → score → π → model
          (no labels; no learning)

Part IV   Evaluation                   frozen π on R_t → accuracy, cost, Pareto
          (offline test; not deployment)
```

| Part | Name | Purpose | Uses \(r(q)\)? | When |
| ---- | ---- | ------- | -------------- | ---- |
| **I** | Experimental Design | Establish a valid Experimental Setting \(\mathcal{S}\) | No (gates only) | Once per benchmark screen |
| **II** | Development | Build and freeze the deployable router | Yes, on \(R_c\) only | Once per locked \(\mathcal{S}\) |
| **III** | Deployment | Route live queries with frozen artifacts | **Never** | Every new query |
| **IV** | Evaluation | Measure accuracy–cost trade-off of frozen \(\pi\) | Yes, on \(R_t\) (grading only) | Once after freeze |

**Terminology (locked):** use **Offline Development** (Part II) and **Online Deployment** (Part III). Avoid vague “offline/online” without naming which part.

---

### Part I — Experimental Design (M1–M3)

**Purpose:** select and freeze one Experimental Setting \(\mathcal{S} = (\mathcal{D}, \mathcal{P}, \Pi)\). No hypotheses are tested. No router is built.

```text
Original dataset D
        ↓
M1a  Dataset preparation (large D only)
     • normalize labels / text
     • drop invalid examples
     • stratified subsample if |C_raw| > threshold (e.g. 5000 → 2000)
        ↓
M1b  Selection holdout H  (fixed |H|, e.g. 150)
        ↓
M2   Feasibility on H
        ↓
M3   Split C \ H → R_c , R_t  (e.g. 80/20 of remainder)
```

Example (MMLU-Pro): ~12k raw → ~2000 prepared → 150 holdout → ~1480 remainder → ~296 test, ~1184 calib (at 20% test fraction).

```text
M1  Experimental Setting Specification
↓
M2  Feasibility Assessment
↓
M3  Experimental Setting Lock
```

| Step | Objective | CLI / artifact |
| ---- | --------- | -------------- |
| **M1** | Load \(C\), **prepare** corpus if large, sample holdout \(H\) | `prepare` → `corpus/` (+ `corpus_preparation` in manifest), `setting.yaml` |
| **M2** | Pilot oracle on \(H\); scorecard gates A–D; pick winning benchmark | `oracle` (split=`selection_holdout`) → `scorecard.json`; `selection-report` |
| **M3** | Lock \(R_c\) and \(R_t\) on winner only; freeze query IDs in `setting.yaml` | `eval` → `corpus/partition.json` |

**Outputs:** frozen `setting.yaml`, `corpus/partition.json` with \(H\), \(R_c\), \(R_t\).

**Invariant:** M1–M3 never fit routing weights, never validate signals, never deploy.

---

### Part II — Development (Stages 4–8 + Freeze Router)

**Purpose:** construct the deployable router **once**, offline, on the locked setting. Oracle labels \(r(q)\) are available on \(R_c\) for validation and calibration only.

```text
Stage 4   Oracle
↓
Stage 5   Signal Extraction          (unsupervised — no r(q))
↓
Stage 6   Signal Validation          (H1–H3; research evidence only)
↓
Stage 7   Representation freeze      (freeze x(q))
↓
Stage 8   Policy calibration           (learn s(q), τ; H4)
↓
Frozen policy                          (research output)
──
Export policy artifact                 (engineering → routing/policy.json)
```

| Stage | Name | Objective | Uses \(r(q)\)? | CLI |
| ----- | ---- | --------- | -------------- | --- |
| **4** | Oracle | Run \(M_{\mathrm{lo}}\), \(M_{\mathrm{hi}}\); construct \(y_{\mathrm{lo}}, y_{\mathrm{hi}}, r(q)\) | Defines \(r\) | `oracle` (calib/test) |
| **5** | Signal Extraction | Compute φ(q), ψ(q), χ(q) — predefined, label-free at extraction | **No** | `model-independent`, `model-dependent`, `cross-model` |
| **6** | Signal Validation | Do predefined signals contain routing information? (H1–H3) | Yes (\(R_c\) only) | `signal-validation` |
| **7** | Representation freeze | Select columns for deployable \(x(q)\) from Stage 6 evidence | Uses rankings | *(planned)* |
| **8** | Policy calibration | Learn routing score \(s(q)\) and \(\tau\) on \(R_c\) (H4) | Yes | `deploy/train` *(interim)* |
| **—** | Export | Serialize frozen policy | No | `routing/policy.json` |

### Stage 6 — Signal validation (locked)

**Scientific question:** Do these predefined signals contain routing information?

**Inputs only:** signals φ, ψ, χ (Stage 5) and oracle label \(r(q)\) (Stage 4). Nothing else.

**Outputs (research only — no deployable artifact):**

| Artifact | Level | Content |
| -------- | ----- | ------- |
| `analysis_table_calib.csv` | join | \(R_c\) only — labels + all signal columns |
| `analysis_table_test.csv` | join | \(R_t\) only — for Stage 9; **not** used in Stage 6 validation |
| `validation_meta.json` | meta | `purpose`, split, n_queries, positive_rate, primary_metric |
| `univariates.json` | L1 | AUROC, AUPRC, Spearman, direction, mean_positive/negative |
| `family_summary.json` | L2 | mean/median/std/IQR/quartiles of AUROC per block |
| `linear_representation_probes.json` | L3 | CV linear probe AUROC + AUPRC (secondary, diagnostic) |

**Code structure:** `validate_signals()` → univariates + family summaries; `probe_linear_representations()` → Level 3 only; `stage_signal_validation()` orchestrates both on **`analysis_table_calib.csv` only**.

**Level 1 — individual signals:** for each feature, association with \(r(q)\). No ranking, no top-k, no feature selection.

**Level 2 — signal families:** median/mean/min/max AUROC over predefined blocks (φ structural / ambiguity / geometry / all; ψ_lo; ψ_hi; χ all). Answers whether a *family* carries routing information.

**Level 3 — linear representation probes (secondary):** stratified CV logistic readout per block — how much routing information is *linearly readable*? Not primary evidence; not the routing method.

**Primary validation metric:** AUROC. Secondary: AUPRC. Exploratory: correlation.

**Invariant:** Stage 6 never fits \(\lambda\), never freezes \(x(q)\), never exports policy.

**Stage 5 is unsupervised.** φ, ψ, and χ are computed without routing labels. χ is **privileged offline analysis** (requires both model traces); it is validated in Stage 6 but excluded from the default runtime policy.

**Target freeze artifact** (`router_package/`):

```text
router_package/
  feature_spec.yaml    # frozen column names and signal-layer scope
  preprocessing.pkl    # scaler (mean, scale) fit on R_c
  weights.npy          # λ (and intercept)
  threshold.json       # τ
  metadata.json        # pool ids, schema version, training provenance
```

**Current code:** Stage 8 writes `routing/policy.json` (combines the above fields). A standalone `router_package/` export is planned.

**Invariant:** Part II runs on the research run directory. It is **never** executed during Part III deployment.

---

### Part III — Deployment (Online)

**Purpose:** for every **new query**, select \(M_{\mathrm{lo}}\) or \(M_{\mathrm{hi}}\) using only frozen signal definitions and frozen policy. No oracle, no validation, no retraining.

```text
New query
↓
Signal computation        φ(q) [optional]  +  ψ(q) after M_lo pass
↓
Feature vector            x(q)  — frozen subset from Stage 7
↓
Preprocessing             z(q) = scale(x(q))
↓
Score                     s(q) = λᵀ z(q)  (+ intercept)
↓
Policy                    if s(q) > τ → M_hi  else  M_lo
↓
Chosen model              return answer from selected pool member
```

| At deploy time | Allowed | Forbidden |
| -------------- | ------- | --------- |
| Signals | φ, ψ per frozen `feature_spec` | χ (default), new feature selection |
| Labels | — | \(r(q)\), \(y_{\mathrm{lo}}\), \(y_{\mathrm{hi}}\) |
| Learning | — | AUROC, CV, refit λ or τ |

**Code:** `llm_routing/runtime/` (`extract.py`, `router.py`, `policy.py`); CLI `route-demo` replays from a run dir.

**ψ-primary note:** ψ requires one \(M_{\mathrm{lo}}\) inference pass before the routing decision; this is cascade **semantics**, not the paper contribution.

---

### Part IV — Evaluation (held-out test)

**Purpose:** measure how the **frozen** router from Part II performs on \(R_t\). This is **not** deployment — it is a one-time offline benchmark of accuracy, cost, and Pareto position (H4).

```text
Run frozen router π on R_t
↓
Accuracy                  task correctness under π vs baselines
↓
Cost                      inference cost (κ-weighted pool usage)
↓
Pareto                    accuracy–cost frontier vs always-M_lo, always-M_hi, π*
```

| Compared to Part III | Part III (deploy) | Part IV (eval) |
| -------------------- | ----------------- | -------------- |
| Queries | Live, unseen | Fixed held-out \(R_t\) |
| Grading | None | Uses frozen protocol labels |
| Output | Model choice + answer | Metrics, tables, H4 |

**Stage mapping:** Part IV = **Stage 9** (test evaluation, H5). CLI: `evaluate`.

---

### Supervision model (cross-cutting)

**Unsupervised** refers to **Stage 5 (signal extraction) only**, not the entire methodology.

| Layer | Part / Stage | Purpose | Uses routing labels? |
| ----- | ------------ | ------- | -------------------- |
| Signal extraction | Part II · Stage 5 | Compute φ, ψ, χ | **No** |
| Signal validation | Part II · Stage 6 | Measure association with routing need | Yes (\(R_c\)) |
| Feature engineering | Part II · Stage 7 | Freeze \(x(q)\) for deploy | Uses Stage 6 evidence |
| Policy calibration | Part II · Stage 8 | Fit \(\lambda\), \(\tau\) | Yes (\(R_c\)) |
| Deployment | Part III | Apply frozen \(\pi\) | **No** |
| Evaluation | Part IV | Grade \(\pi\) on \(R_t\) | Yes (grading only) |

**Paper claim (locked):** *routing using unsupervised signals* — not *unsupervised routing*. Supervised routing methods learn the entire mapping from labels; we separate **label-free signal construction** from **offline calibration** on \(R_c\).

---

## 0.8 Deployment architecture (Part III detail)

The deployed router consists only of frozen components from Part II.

```text
Incoming query q
↓
Compute signals           φ(q) [if in policy]  +  ψ(q) from M_lo trace
↓
Construct x(q)            frozen columns from Stage 7
↓
Preprocess & score        s(q) = f(x(q); λ, τ)
↓
Select                    M_lo  or  M_hi
↓
Return answer
```

Neither oracle labels nor the calibration set \(R_c\) are required at inference time.

**Not at runtime (default):** χ(q), \(r(q)\), AUROC, logistic refit, feature re-ranking.

---

## 0.9 Scientific Contributions

This work makes four contributions (see §0.1).

1. A taxonomy of unsupervised routing signals (φ, ψ, χ).
2. A three-level validation protocol measuring association with \(r(q)\) on \(R_c\) (Stage 6).
3. Policy calibration: routing score \(s(q)\) and \(\tau\) from validated signals (Stage 8).
4. Held-out routing evaluation vs baselines (Stage 9).

---

I would also recommend one important terminology change throughout the document. Instead of saying "Offline" and "Online", use **"Offline Development"** and **"Online Deployment"**. This makes it immediately clear to reviewers that the offline phase is the research and calibration workflow, while the online phase is the actual deployed inference pipeline. That distinction is standard in machine learning systems and avoids the confusion you've been encountering.
