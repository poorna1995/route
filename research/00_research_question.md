# Research Question

> **Frozen project design:** [`MASTER.md`](MASTER.md) · **Vocabulary:** [`claims.md`](claims.md)

---

## Research question

**Can unsupervised pre-inference signals support appropriate multi-LLM routing before full model generation?**

**One-line problem:** Given a query, without routing labels at signal-extraction time, can inexpensive pre-inference signals help **decide which LLM should answer**?

**How we answer it:** Extract unsupervised signals from three **information sources** (query-derived, model-derived, cross-model) → characterize routing-relevant information (Studies I–III) → evaluate whether a calibrated policy can exploit it (Study IV).

**Anchor:** *Study unsupervised signals first. Routing is only an application.*

---

## Motivation

Different queries require different reasoning capability; routing every query through the same LLM is inefficient.
Existing routing is largely **supervised** (labels, preferences, trained routers) or uses information **after** full generation.
We ask whether **unsupervised pre-inference signals**—computable before expensive inference—can support routing decisions.

---

## Headline contribution

A **systematic empirical study** of unsupervised pre-inference signals for multi-LLM routing: what routing-relevant information they carry, how **information dimensions** differ and complement each other, and what a **calibrated policy** can exploit.

**Avoid as sole headline:** signal framework · characterization-only metrology · we built a router.

---

## Results structure (four questions)

| Q | Question | Studies |
| - | -------- | ------- |
| Q1 | Is routing-relevant information present pre-inference? | I–II |
| Q2 | How much? What are the limits? | I–II + interpretation |
| Q3 | What aspects of routing need do dimensions encode? | III + interpretation |
| Q4 | Can a calibrated policy exploit it? | IV |

**Locked answer (ARC TEST):** **Partially** — see `claims.md`.

---

## Appropriate model selection

Route to the **weakest model expected to answer correctly**, escalating only when pre-inference signals indicate that additional capability is likely to improve the outcome—a **cost–quality trade-off** within fixed pool \(\mathcal{M}\).

---

## Informativeness (operational)

Predictive association with routing need (offline oracle buckets, routing opportunity, weak–strong gap), measured by Spearman \(\rho\), AUROC/AUPRC, and complementary predictive gain (ΔAUROC).
**Not** Shannon mutual information \(I(S;Y)\).

---

## Hypothesis chain

```text
RH1  Unsupervised pre-inference signals carry routing-relevant information     → Studies I–II
RH2  Information dimensions encode distinct aspects of routing need            → Study II + interpret
RH3  Dimensions provide complementary information                               → Study III
RH4  A calibrated policy can exploit available information                      → Study IV
```

---

## Supervision (three layers)

| Layer | Labels? | Role |
| ----- | ------- | ---- |
| Signal extraction | **No** | Unsupervised at inference |
| Signal characterization | Oracle offline only | Evidence for routing question |
| Routing evaluation | CALIB only | Exploitation test |

---

## Framing

**Routing is the problem; signal characterization is how we answer it; routing evaluation is the final test.**

Primary contribution: **empirical understanding** of pre-inference routing signals—not a routing product.

---

## Information sources (paper taxonomy)

| Source | Examples | Study |
| ------ | -------- | ----- |
| **Query-derived** | `piece_count` / \(c(q)\) | I |
| **Model-derived** | \(H_w\), \(m_w\) | II |
| **Cross-model** | \(\Delta H\), \(\Delta m_{\mathrm{gain}}\) | II–III |
| *(Future) Perturbation-derived* | paraphrase stability | — |

C3 (**layerwise confidence evolution**) extends **model-derived** signals (terminal → layerwise); it is not a fourth source. See [`c3_layerwise_concepts.md`](c3_layerwise_concepts.md).

---

## Advisor alignment

See [`MASTER.md`](MASTER.md) §1 and [`claims.md`](claims.md): one problem, unsupervised pre-inference signals, three information sources, calibrated policy on CALIB only, agents deferred, ACL target, paper-first workflow. Professor examples (entropy, complexity, paraphrases) illustrate signal types—not the contribution.
