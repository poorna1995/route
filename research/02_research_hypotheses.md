# Research Hypotheses

> **Scientific decision this document supports:**  
> What scientific claims will this research test?

> **Lifecycle**
>
> - **M1:** Draft hypotheses from the research question and advisor discussion.
> - **M2:** Refine after literature review.
> - **M5:** Finalize before large-scale experiments.

Related documents

- Research Question → `00_research_question.md`
- Problem Statement → `01_problem_statement.md`
- Literature Gap → `03_literature_gap.md`
- Signal Design → `04_signal_design.md`
- Evaluation Design → `08_evaluation_design.md`
- Experiment Registry → `10_experiment_registry.md`
- Computation Protocol → `05_computation_protocol.md`

---

# Purpose

The research question defines **what we want to investigate**.

Research hypotheses define the **scientific claims** that experiments will verify or refute.

These hypotheses are informed by

1. the research question,
2. advisor discussion,
3. initial literature review.

They are **not** tied to any specific implementation.

---

# Framing — routing problem, unsupervised pre-inference signals

The paper investigates **unsupervised pre-inference routing**: can inexpensive signals computed before full generation support appropriate LLM selection in a fixed pool?

```text
Problem: multi-LLM routing with unsupervised pre-inference signals
    ↓
Information sources: query-derived | model-derived | cross-model
    ↓
Signal characterization (Studies I–III)
    ↓
Routing evaluation (Study IV)
```

**v1 scope:** one **representative** query-derived signal (`piece_count` / \(c(q)\), D46) plus model-derived probes (\(H_w\), \(m_w\)) and cross-model derived signals (\(\Delta H\), \(\Delta m_{\mathrm{gain}}\)). The contribution is empirical understanding of **unsupervised pre-inference routing signals**—not assuming any single statistic suffices. Taxonomy 𝒮 organizes **information sources**; C3 extends model-derived (layerwise), not a fourth source.

**Informativeness (operational):** predictive association with routing need via Spearman ρ, AUROC, and complementary predictive gain (ΔAUROC)—not Shannon mutual information. See `MASTER.md` §3.

**Negative results:** Weak association is valid. If signals exist but routing evaluation fails: **routing limited by exploitation**. If all dimensions null on TEST: **limits paper** for these information sources.

---

# Hypothesis progression (D64)

Each study depends on the previous. Study IV is meaningful only if Studies I–III establish whether exploitable information exists.

```text
RH1  Unsupervised pre-inference signals carry routing-relevant information        → Studies I–II
        ↓
RH2  Information dimensions encode distinct aspects of routing need               → Study II + interpret
        ↓
RH3  Dimensions provide complementary information                                  → Study III
        ↓
RH4  A calibrated policy can exploit available information                         → Study IV
```

**Not this progression:** H1 router works → H2 router works better → H3 saves money.

---

# Paper-level Research Hypotheses

These represent the primary scientific claims of the paper.

| ID      | Research Hypothesis                                                                                                | Related Objective | Status |
| ------- | ------------------------------------------------------------------------------------------------------------------ | ----------------- | ------ |
| **RH1** | Unsupervised pre-inference signals carry measurable **routing-relevant information**. | O1, O2 | Frozen |
| **RH2** | **Information dimensions** encode **distinct** aspects of routing need. | O1, O2 | Frozen |
| **RH3** | Dimensions provide **complementary** information beyond any single dimension. | O2 | Frozen |
| **RH4** | A **calibrated routing policy** can **exploit available information** for cost–quality gains (conditional on I–III). | O3 | Frozen |

---

# Signal-level Hypotheses

These hypotheses study the behaviour of individual signals.

They support RH1–RH3.

---

## H1 — Query Complexity (representative \(c(q)\))

**Signal**

Representative **query-derived** complexity statistic \(c(q)\) — selected per D46 screening process (`18`); exact formula in `05` §8 when implemented.

**Claim**

The chosen representative contains statistically measurable information about routing need—not that every complexity measure works, nor that \(c(q)\) alone suffices for routing.

**Expected Behaviour**

Complex queries should correspond to larger performance gaps between weak and strong models.

**Possible Failure**

Surface complexity may not reflect actual reasoning difficulty.

**Tier-1 literature (deep read)**

- Hybrid LLM (Ding et al., 2024) routes on predicted difficulty but via a **supervised** DeBERTa router and BARTScore labels from offline S/L generations — not unsupervised s(q).
- Hybrid LLM §5 lists **task-aware routing** and richer pre-query signals as future work; current router is **query-only**.
- RouteLLM and RouterBench predictive routers use **query embeddings**, not explicit complexity features — different mechanism from s(q).

**Supports**

RH1

---

## H2 — Token Entropy

**Signal**

Entropy of the next-token distribution for model Mᵢ.

**Claim**

Weak-model entropy **contains statistically measurable information about routing need** — i.e., how informative $H(q, M_{\text{weak}})$ is for identifying queries where the strong model succeeds and the weak model fails. We do **not** claim entropy alone “predicts routing” or is sufficient.

**Expected Behaviour**

Higher weak-model entropy may associate with routing opportunity; direction is **tested**, not assumed. Within-query $\Delta H = H_w - H_s$ may align more directly with weak–strong divergence.

**Possible Failure**

Models may remain confidently incorrect on some factual or adversarial queries.

**Tier-1 literature (deep read)**

- **None** of RouteLLM, FrugalGPT, GraphRouter, Hybrid LLM, or RouterBench evaluated baselines use **per-model pre-inference entropy** as routing information.
- FrugalGPT uses post-generation DistilBERT scoring g(q, answer), not token-level entropy on a probe pass.
- *Entropy for uncertainty estimation is widespread outside routing (Tier-2); Tier-1 does not establish it as a routing probe.*

**Supports**

RH2

---

## H3 — Log-Probability Margin

**Signal**

Difference between the highest and second-highest token probabilities.

**Claim**

Weak-model log-probability margin **contains statistically measurable information about routing need**. Study III tests whether margin adds incremental information beyond entropy, and whether model-derived signals add information beyond the query-derived representative.

**Expected Behaviour**

Lower weak-model margin may associate with routing opportunity; direction is **tested**, not assumed. $\Delta m = m_s - m_w$ captures how differently the two models respond to the same prompt.

**Possible Failure**

Margin may not always correlate with final answer correctness.

**Tier-1 literature (deep read)**

- Tier-1 routing papers do not use **log-probability margin on a pre-inference probe** as routing information.
- Closest related idea: FrugalGPT's learned scorer on (q, answer) — **post-generation**, not next-token margin before full decode.

**Supports**

RH2

---

## H4 — Paraphrase Stability

**Signal**

Consistency of model behaviour across paraphrased versions of the same query.

**Claim**

Robust model-query pairs produce stable probe signals across paraphrases.

**Expected Behaviour**

Suitable models remain stable under paraphrasing, while unsuitable models show greater variation.

**Possible Failure**

Poor paraphrases may unintentionally alter task semantics.

**Tier-1 literature (deep read)**

- Not studied in any Tier-1 routing paper reviewed (`03` §2.1–2.5).

**Supports**

RH2

---

# Hypothesis Traceability

Every hypothesis maps to a **paper study**. Notebook IDs and scripts: `10_experiment_registry.md` only.

| Hypothesis | Study | Primary metric |
| ---------- | ----- | -------------- |
| RH1 | **I–II** — Signal characterization | Spearman ρ, AUROC across dimension representatives |
| RH2 | **II** + interpretation — Dimensional structure | Bucket separability; opportunity vs.\ too-hard contrasts |
| RH3 | **III** — Complementarity | ΔAUROC ladder across dimensions |
| RH4 | **IV** — Routing evaluation | Cost–quality vs.\ static baselines |
| H1 | Study I | \(c(q)\) informativeness |
| H2 | Study II | Entropy informativeness |
| H3 | Studies II–III | Margin informativeness + complementarity with entropy |

**Deferred:** H4 (paraphrase stability) — out of main study scope (D04).

---

# Extension hypothesis — RH5 (C3, optional)

**Scope:** Extends **model-derived** signals only. Not a fourth information source. See [`c3_layerwise_concepts.md`](c3_layerwise_concepts.md).

**Hypothesis RH5 (locked wording):**

> Does layerwise evolution of model confidence provide routing-relevant information **beyond terminal** prefill confidence?

**Measurable form:**

> Do easy, opportunity, and too-hard queries exhibit **different confidence evolution** across transformer layers?

**Operationalizations:** `stabilization_layer` (primary), `slope_margin` (secondary), F7 trajectories, divergence at fraction depth ℓ/L.

**Wording:** Use *confidence evolution* / *trajectories* in Methods and Results. Reserve *confidence formation* for Discussion only if results support it.

**Null result:** Publishable — “terminal prefill probes suffice on ARC for routing characterization.”

---

# Literature Refinement Log

Update hypotheses after reviewing literature.

| Hypothesis | Revision | Reason |
| ---------- | -------- | ------ |
| RH1        | Frozen (D55) | Representative **query-derived** signals — measurable, not "at least one" |
| RH2        | Frozen (D55) | Model-dependent probe signals — measurable information |
| RH3        | Frozen (D55) | Dimensions partially complementary beyond any single dimension |
| RH4        | Strengthened, tempered | RouteLLM/Hybrid LLM show supervised routing gains; RouterBench §5.1 — simple predictive routers often fail to beat **Zero router** on several tasks, so probe-policy gains are not guaranteed |
| RH5        | C3 optional  | Layerwise **confidence evolution** beyond terminal — null publishable |
| H1         | Clarified  | Hybrid LLM "difficulty" is supervised query-only routing; our H1 targets **unsupervised** s(q) — related motivation, different mechanism |
| H2, H3     | Clarified  | Not routing probes in Tier-1; Tier-2 uncertainty literature is separate |
| H4         | Unchanged  | No Tier-1 precedent |

---

# Notes

- Hypotheses should remain implementation-independent.
- Individual signals may be added, removed, or replaced during signal screening.
- Negative results are valid research outcomes.
- Any substantial changes should be recorded in `09_decision_register.md`.
