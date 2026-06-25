# Problem Statement

> **Scientific decision this document supports:** Why does this problem matter, and what is in scope?

---

## Problem statement (3–5 sentences)

Modern multi-LLM systems increasingly rely on multiple foundation models with different capabilities, costs, and inference latencies. **Different queries require different reasoning capability**; using the same LLM for every query is inefficient. The challenge is to **estimate routing need before generating an answer**—without supervised routing labels or expensive full inference on every pool member. Our approach uses **unsupervised pre-inference signals**—model-independent features and model-dependent probe statistics—extracted before full generation. The research question: **how much routing-relevant information** do these signals carry for LLM selection?

**Headline contribution:** signal **extraction and characterization** for routing — not “unsupervised routing” (Study IV uses CALIB calibration) and not “we built a router.”

### Conceptual flow (one problem)

```text
Query → unsupervised signal extraction → signal characterization → simple routing policy (demonstration) → generate answer
```

**Appropriate selection:** route to the **weakest model expected to answer correctly**; escalate only when signals indicate additional capability is likely to improve outcome (cost–quality trade-off within the pool).

### Scope discipline — “solve one problem” (D63)

One ACL paper = one problem, four hypotheses, one methodology. Do **not** expand into agent routing, task decomposition, graph routing, or benchmark sprawl in v1.

```text
✓  Unsupervised pre-inference signals → LLM routing (this paper)
✗  Agent routing, orchestration, supervised neural routers (future work)
```

---

## Research assumptions

Every claim in this project rests on these. State them explicitly so scope of claims is clear.

| ID     | Assumption                                                                                                                                      |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **A1** | Deployment uses a **fixed probe pool** (same models offline and online for this study).                                                         |
| **A2** | **Prefill-probe cost** (signal extraction) is significantly cheaper than **expensive full inference** (full answer generation) on all pool models.                                                |
| **A3** | The selected evaluation benchmarks contain sufficient diversity in query difficulty to evaluate routing decisions across the chosen model pool. |
| **A4** | All routing signals are computed **before expensive full inference** (pre-inference). A **prefill probe** on the prompt is allowed; `generate()` to completion is not used for signals. |
| **A5** | Oracle labels (**full inference** / full answer generation on all pool models) are available **offline only** for evaluation—not at deployment.                                  |
| **A6** | Model-dependent probe signals can be computed consistently across all models in the selected pool.                                              |

Add or revise assumptions here; log changes in `09_decision_register.md`.

---

## Research objectives

Objectives organize the research program. Testable predictions live in `02_research_hypotheses.md`.

| ID     | Objective                                                                                       |
| ------ | ----------------------------------------------------------------------------------------------- |
| **O1** | Define and taxonomize unsupervised routing signals (model-independent and model-dependent). |
| **O2** | Characterize how informative those signals are for routing need **under a cost–quality objective**. |
| **O3** | Determine whether a simple routing policy can exploit **validated** signals for cost–quality gains (utility). |

---

## Scope

### In scope

- Fixed pool of LLMs (probe pool)
- Pre-inference signals (before expensive full inference; prefill probe permitted)
- Signal design and characterization
- Simple routing policy (rule or calibration)
- Offline evaluation with oracle comparison and probe-cost analysis
- Signal computation and probe efficiency analysis

### Out of scope

- [ ] Agent routing (ReAct, multi-step orchestration)
- [ ] Task decomposition pipelines
- [ ] Supervised end-to-end neural routers as main contribution
- [ ] Post-inference / post-hoc routing signals
- [ ] Must beat all SOTA routers

---

## Success criteria (scientific, venue-agnostic)

1. Understanding what pre-inference information exists and **how informative** it is (operational metrics: ρ, AUROC, ΔAUROC—not mutual information).
2. Signal characterization and joint understanding before any routing claim.
3. Probe efficiency analysis.
4. Simple routing policy demonstrating that **validated** signals can be exploited (policy is last, not first).
5. Comprehensive offline evaluation.

---

## Daily log

**Decision (2026-06-25):** Problem statement aligned with D63 — goal (efficient routing) vs contribution (signal informativeness); scope discipline explicit.

**Evidence:** Advisor: one problem, one paper; signals before router.

**Uncertainty:** None on scope; execution remains CALIB → D46 → TEST.
