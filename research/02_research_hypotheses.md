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

# Framing — signal space, not a single heuristic

The paper studies a **signal space** 𝒮 of candidate pre-inference probes, not entropy alone:

```text
𝒮 = 𝒮_indep ∪ 𝒮_dep

𝒮_indep   model-independent family   →  representative c(q) in v1
𝒮_dep     model-dependent probes      →  H (uncertainty), m (confidence separation)
```

**v1 scope:** one **representative** model-independent signal (selected from the complexity family per `18`) plus two model-dependent probes. The contribution is **understanding pre-inference information for unsupervised routing**: characterization before routing policy—not assuming any one signal works. Taxonomy 𝒮 is the organizing framework.

**Informativeness (operational):** predictive association with routing need via Spearman ρ, AUROC, and complementary predictive gain (ΔAUROC)—not Shannon mutual information. See `MASTER.md` §3.

**Negative results:** Weak association is valid—but if **every** family is null on corrected TEST (AUROC ≈ 0.50), the paper becomes a **limits** characterization (D64), not a routing-methods claim.

---

# Hypothesis progression (D64)

Each study depends on the previous. Study IV is meaningful only if Studies I–III establish whether exploitable information exists.

```text
RH1  Model-independent signals contain routing-relevant information        → Study I
        ↓
RH2  Model-dependent probe signals contain routing-relevant information   → Study II
        ↓
RH3  Together they provide additional information (complementarity)         → Study III
        ↓
RH4  A simple routing policy can exploit whatever information exists        → Study IV (demonstration)
```

**Not this progression:** H1 router works → H2 router works better → H3 saves money.

---

# Paper-level Research Hypotheses

These represent the primary scientific claims of the paper.

| ID      | Research Hypothesis                                                                                                | Related Objective | Status |
| ------- | ------------------------------------------------------------------------------------------------------------------ | ----------------- | ------ |
| **RH1** | Model-independent signals contain measurable **routing-relevant information**. | O1, O2 | Frozen (D55, D64) |
| **RH2** | Model-dependent probe signals contain measurable **routing-relevant information**. | O1, O2 | Frozen (D55, D64) |
| **RH3** | The two signal families provide **additional information** beyond either family alone. | O2 | Frozen (D55, D64) |
| **RH4** | A simple routing policy can **exploit whatever information exists** for cost–quality gains over static baselines (conditional on I–III). | O3 | Frozen (D55, D64) |

---

# Signal-level Hypotheses

These hypotheses study the behaviour of individual signals.

They support RH1–RH3.

---

## H1 — Query Complexity (representative \(c(q)\))

**Signal**

Representative model-independent complexity statistic \(c(q)\) — selected per D46 screening process (`18`); exact formula in `05` §8 when implemented.

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

Weak-model log-probability margin **contains statistically measurable information about routing need**. Study III tests whether margin adds incremental information beyond entropy, and whether model-dependent signals add information beyond the model-independent representative.

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
| RH1 | **I** — Model-independent characterization | Spearman ρ, AUROC for \(c(q)\) vs \(y_{\text{opp}}\) |
| RH2 | **II** — Model-dependent characterization | Spearman ρ, AUROC for \(H, m\) |
| RH3 | **III** — Complementarity | ΔAUROC / predictive-gain ladder across families |
| RH4 | **IV** — Routing evaluation | Cost–quality vs static baselines |
| H1 | Study I | \(c(q)\) informativeness |
| H2 | Study II | Entropy informativeness |
| H3 | Studies II–III | Margin informativeness + complementarity with entropy |

**Deferred:** H4 (paraphrase) — out of v1 scope (D04).

---

# Literature Refinement Log

Update hypotheses after reviewing literature.

| Hypothesis | Revision | Reason |
| ---------- | -------- | ------ |
| RH1        | Frozen (D55) | Representative model-independent signals — measurable, not "at least one" |
| RH2        | Frozen (D55) | Model-dependent probe signals — measurable information |
| RH3        | Frozen (D55) | Two families complementary beyond either alone |
| RH4        | Strengthened, tempered | RouteLLM/Hybrid LLM show supervised routing gains; RouterBench §5.1 — simple predictive routers often fail to beat **Zero router** on several tasks, so probe-policy gains are not guaranteed |
| H1         | Clarified  | Hybrid LLM "difficulty" is supervised query-only routing; our H1 targets **unsupervised** s(q) — related motivation, different mechanism |
| H2, H3     | Clarified  | Not routing probes in Tier-1; Tier-2 uncertainty literature is separate |
| H4         | Unchanged  | No Tier-1 precedent |

---

# Notes

- Hypotheses should remain implementation-independent.
- Individual signals may be added, removed, or replaced during signal screening.
- Negative results are valid research outcomes.
- Any substantial changes should be recorded in `09_decision_register.md`.
