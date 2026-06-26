# Advisor meeting — structured guidance

> Verbatim transcript follows in §Raw notes. This section is the actionable research direction in academic form.

---

## Core problem

**Unsupervised query routing to large language models using pre-inference signals.**

Focus on one publishable problem: given a query, extract inexpensive signals before full generation and decide which LLM in a fixed pool should answer—without routing labels at signal-extraction time.

## Scope discipline

| In scope (this paper) | Out of scope (future work) |
| --------------------- | -------------------------- |
| Multi-LLM routing in a fixed pool | Agent routing and orchestration |
| Model-independent + model-dependent pre-inference signals | Task decomposition pipelines |
| Signal characterization + simple calibrated routing evaluation | Supervised neural routers as the contribution |
| Honest limits and negative exploitation results | Benchmark sprawl |

## Narrative contrast

- **Prior work:** supervised routers, preference data, or post-generation selection.
- **This work:** what routing-relevant information is available **before generation** from unsupervised signals, and whether a calibrated policy can use it.

## Signal families

1. **Model-independent:** query complexity and related text statistics.
2. **Model-dependent:** prefill-probe entropy, confidence, and cross-model disagreement—stronger than query-only features because they condition on the candidate model.

## Learning component

Learn weights over signals (e.g., logistic regression on CALIB) rather than hand-crafted routing rules. Negative results are acceptable if they answer the hypotheses.

## Publication target

ACL (language-model routing contribution). **Workflow:** write the paper first; run only experiments needed to fill hypothesis-linked tables.

## Progression (advisor roadmap)

1. Unsupervised pre-inference routing for LLMs *(this paper)*
2. Richer signal combination and exploitation
3. Extension to agents *(deferred)*

---

## Raw notes

