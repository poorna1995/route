# Problem Statement

> **Vocabulary:** [`claims.md`](claims.md) · **Frozen design:** [`MASTER.md`](MASTER.md)

---

## Problem statement

Modern multi-LLM deployments use several foundation models with different capabilities, costs, and latencies.
**Different queries require different reasoning capability**; using one LLM for every query is inefficient.
The **research problem** is **unsupervised pre-inference routing**: given a query, extract inexpensive signals before full generation—without routing labels at extraction time—and determine **which LLM in a fixed pool should answer**.
We characterize routing-relevant information because routing depends on understanding those signals, then evaluate whether a calibrated policy can exploit them.

**Headline:** a systematic empirical study of **unsupervised pre-inference signals for multi-LLM routing**—not a characterization-only paper and not “we built a router.”

---

## Conceptual flow

```text
Query
  → unsupervised signal extraction (query-derived | model-derived | cross-model)
  → signal characterization
  → calibrated routing evaluation
  → generate answer with selected LLM
```

**Appropriate selection:** weakest model expected to succeed; escalate when signals indicate benefit.

---

## Scope discipline

One ACL paper = one problem, four hypotheses, one methodology.

```text
✓  Unsupervised pre-inference signals → multi-LLM routing (this paper)
✗  Agent routing, orchestration, supervised neural routers as main contribution (future work)
```

---

## Research assumptions

| ID | Assumption |
| -- | ---------- |
| **A1** | Fixed probe pool (same models offline and online for this study). |
| **A2** | Prefill-probe cost is significantly cheaper than full answer generation on all pool models. |
| **A3** | Evaluation benchmarks contain sufficient difficulty diversity for the chosen pool. |
| **A4** | All routing signals are computed **before full generation**; prefill probe permitted. |
| **A5** | Offline oracle labels (full inference on all pool models) are available for evaluation only. |
| **A6** | Model-dependent signals are computable consistently across all pool members. |

---

## Research objectives

| ID | Objective |
| -- | --------- |
| **O1** | Define and taxonomize unsupervised pre-inference signals by **information source** (query-derived, model-derived, cross-model). |
| **O2** | Characterize routing-relevant information under a cost–quality objective. |
| **O3** | Evaluate whether a calibrated routing policy can exploit characterized signals. |

Testable predictions: `02_research_hypotheses.md`.

---

## Scope

### In scope

- Fixed LLM pool; pre-inference signals; signal characterization; calibrated routing evaluation; offline oracle comparison; probe-cost analysis.

### Out of scope

- Agent routing, task decomposition, supervised end-to-end neural routers as main contribution, post-generation routing signals, beating all SOTA routers.

---

## Success criteria

1. Clear answer whether unsupervised pre-inference signals **can support routing** (including limits).
2. Rigorous signal characterization before routing claims.
3. Routing evaluation interpreted as **exploitation test**, not product claim.
4. Probe efficiency analysis and comprehensive offline evaluation.
