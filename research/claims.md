# Claims — Vocabulary, narrative, and evidence map

> **Frozen design:** [`MASTER.md`](MASTER.md) · **Metrics:** `08_evaluation_design.md` · **Outline:** `11_paper_outline.md`

---

## Locked vocabulary (use consistently in `research/` and `paper/`)

Define each term once; do not alternate synonyms in prose.

| Term | Definition | Do not substitute |
| ---- | ---------- | ----------------- |
| **Multi-LLM routing** | Assigning each query to one model in a fixed pool \(\mathcal{M}\) under a cost–quality objective | model selection, dispatch, orchestration |
| **Appropriate model selection** | Route to the weakest model expected to answer correctly; escalate only when additional capability is likely to help | best model, suitable LLM |
| **Unsupervised pre-inference signals** | Statistics computed from query text and/or prefill logits **before full answer generation**, without routing labels at extraction time | pre-generation features (ambiguous), probes alone |
| **Query-derived signals** | Statistics from query text alone (model-independent) | query-only features, linguistic features (in lists only) |
| **Model-derived signals** | Statistics from a **prefill probe** on one model's logits | model-dependent alone, terminal confidence alone |
| **Cross-model signals** | Statistics from paired weak/strong prefill probes | relational, disagreement alone |
| **Perturbation-derived signals** | Statistics under meaning-preserving query perturbations (future) | paraphrase stability, semantic entropy |
| **Prefill probe** | One forward pass on the formatted prompt; logits at the final prompt position | observation, acquisition pass |
| **Latent routing dimension** | Unobserved aspect of routing need (task difficulty, model uncertainty, model disagreement, escalation potential) | signal, feature, raw statistic |
| **Operationalization** | Prefill statistic that measures a dimension (e.g., $H_w$ for model uncertainty) | representative statistic, probe |
| **Routing-relevant information** | Predictive association between a signal and routing need (opportunity, oracle buckets, weak–strong gap) | informativeness alone without definition |
| **Signal characterization** | Empirical analysis of routing-relevant information (Studies I–III) | measurement, metrology, protocol |
| **Routing evaluation** | Held-out test of whether a calibrated policy exploits characterized information (Study IV) | router contribution, routing system |
| **Offline oracle** | Full greedy inference labels used only for evaluation and calibration | ground truth routing |
| **Routing opportunity** | \(y^{\text{opp}}=1\) when weak fails and strong succeeds | escalation label |
| **Calibration split (CALIB)** | ARC validation (\(n{=}299\)): representative selection and policy fitting | validation set (ambiguous) |
| **Test split (TEST)** | ARC test (\(n{=}1{,}172\)): all reported metrics | held-out (define once) |
| **Calibrated policy** | Logistic + threshold fit on CALIB; routing evaluation instrument | learned router, simple router |

**Banned in paper prose:** Protocol v1/v2, L1–L4, framework (as novelty claim), measurement protocol, observation store, Paper v1.

**Notebook-only IDs** (never in paper): V1/V2 pilots, EXP-01–03, D46, Phase A/B/C.

---

## Research question (frozen)

> **Can unsupervised pre-inference signals support appropriate multi-LLM routing before generation?**

**Contrast sentence:** Prior work develops routing *algorithms*; we ask what *routing-relevant information is available before any routing decision*.

**Answer (ARC TEST):** **Partially.** Information exists and dimensions differ; a calibrated policy does not exploit available headroom.

---

## Information sources (paper taxonomy — three + future)

Organize signals by **where information comes from**, not by extraction depth.

| Information source | Examples (this paper) | Study |
| ------------------ | --------------------- | ----- |
| **Query-derived** | `piece_count` / $c(q)$ | I |
| **Model-derived** | $H_w$, $m_w$ (terminal prefill) | II |
| **Cross-model** | $\Delta H$, $\Delta m_{\mathrm{gain}}$ | II–III |
| *(Future) Perturbation-derived* | paraphrase stability (D04 defer) | — |

**Model-derived (C3 extension):** terminal statistics (C0) → **layerwise confidence evolution** (C3). Richer characterization within model-derived — **not** a fourth information source. Concepts: [`c3_layerwise_concepts.md`](c3_layerwise_concepts.md).

**Anchor sentence (transcript + co-author):** *Study unsupervised signals first. Routing is only an application.*

## Narrative spine (ACL-style)

```text
Routing problem (unsupervised, pre-inference)
    → Signal extraction (query-derived | model-derived | cross-model)
    → Signal characterization (existence, structure, complementarity)
    → Routing evaluation (exploitation test)
```

**Routing is the problem; characterization is how we answer it; routing evaluation is the final test—not a product claim.**

---

## Contributions (ranked)

| Rank | Contribution |
| ---- | ------------ |
| **1** | Empirical study of **unsupervised pre-inference signals** for multi-LLM routing |
| **2** | Taxonomy of **information dimensions** with representative statistics and reproducible prefill-based extraction |
| **3** | Evidence that dimensions encode **different routing need** and are **partially complementary** |
| **4** | Routing evaluation showing an **exploitation gap** between oracle and a calibrated policy |

---

## Scientific question (unit of analysis)

> **Which latent routing dimension predicts routing opportunity?**

Not: *which feature has the largest AUROC?*  
Dimensions are latent; statistics are operationalizations. AUROC summarizes detectability for a dimension—not a feature leaderboard.

## Latent routing dimensions (main analysis)

| Latent routing dimension | Operationalization | Opportunity AUROC (TEST) |
| ------------------------ | ---------------- | ------------------------ |
| **Task difficulty** | `piece_count` / $c(q)$ | 0.541 |
| **Model uncertainty** | $H_w$ | 0.581 |
| **Model disagreement** | $\Delta H$ | 0.602 |
| **Escalation potential** | $\Delta m_{\mathrm{gain}}$ | 0.605 |

**Difficulty vs recoverability:** query-derived + model-derived uncertainty track **difficulty**; cross-model signals track **recoverability**. Task difficulty + model uncertainty overlap; model disagreement + escalation potential separate opportunity from too-hard.

**Legacy aliases (decision register / code only):** *model-independent* = query-derived; *model-dependent* = model-derived. Do not use as primary paper taxonomy.

**Appendix operationalization:** $m_w$ as alternate measure of model uncertainty (AUROC 0.432 on TEST; does not improve joint model).

**Deferred dimensions** (future work): dynamics, representation, stability, calibration.

---

## Hypotheses (RH1–RH4)

| ID | Hypothesis | Study | Evidence |
| -- | ---------- | ----- | -------- |
| **RH1** | Latent routing dimensions carry measurable predictive content for routing opportunity | I–II | T2; F1 |
| **RH2** | Dimensions encode **distinct** aspects of routing need (difficulty vs recoverability) | II + interpret | F2, F6 |
| **RH3** | Dimensions provide **complementary** information beyond any single dimension | III | T3; F3 |
| **RH4** | A **calibrated policy** can exploit available information under a cost–quality objective | IV | T4 |

---

## Three findings (reader-facing)

| Finding | Claim | Key numbers |
| ------- | ----- | ----------- |
| **F1** | Latent dimensions **predict opportunity** before generation | Opportunity 43.3%; escalation potential AUROC ~0.61 |
| **F2** | **Difficulty \(\neq\) recoverability** before decoding | $H_w$ $d{\approx}0.03$ vs $\Delta m_{\mathrm{gain}}$ $d{\approx}0.72$ |
| **F3** | Routing **leaves headroom** | Always-strong 69.2%; calibrated policy 69.2%; oracle 74.4% (5.2 pp unexploited) |

---

## Supervision (state once in Methods)

| Stage | Routing labels? | Role |
| ----- | ----------------- | ---- |
| Signal extraction | **No** | Unsupervised at inference |
| Signal characterization | Oracle **offline only** | Measure routing-relevant information |
| Routing evaluation | CALIB only | Test exploitation; not the research claim |

---

## Title

> Can Unsupervised Pre-Inference Signals Support Appropriate Multi-LLM Routing?

---

## Hypothesis → table map

| Hypothesis | Table | Figure |
| ---------- | ----- | ------ |
| RH1–RH2 | T2 | F1, F2, F6 |
| RH3 | T3 | F3 |
| RH4 | T4 | — |

**Principle:** Every experiment fills a table that answers a hypothesis that answers the research question.

---

## Keywords (ACL / paper metadata)

**Primary:** unsupervised pre-inference signals · multi-LLM routing · prefill probe · signal characterization · routing opportunity · appropriate model selection

**Information sources:** query-derived · model-derived · cross-model · perturbation-derived (future)

**Findings:** difficulty vs recoverability · exploitation gap · complementary predictive gain · offline oracle · calibrated policy

**Methods:** ARC-Challenge · MMLU transfer · Cohen's d · AUROC · Spearman ρ

**Contrast (not our contribution):** supervised routing · post-generation routing · agent orchestration · RouteLLM · GraphRouter
