# Signal Design and Characterization

> **Frozen v1 signals:** [`MASTER.md`](MASTER.md) §5 · **Computation:** `05_computation_protocol.md`

> **Milestone M3** — taxonomy frozen (2026-06-20)
>
> - Keep / Defer / Drop in Part F are literature-informed; validated in main study
> - Workflow: taxonomy → implement probes → main study characterization → routing
> - Computation → `05_computation_protocol.md`
> - **Next:** implement probes → CSV → `07` oracle → `08` evaluation (no routing yet)
>
> Related documents:
>
> - Research Hypotheses → `02_research_hypotheses.md`
> - Literature Gap → `03_literature_gap.md`
> - Computation Protocol → `05_computation_protocol.md`
> - Decision Register → `09_decision_register.md`

---

# Purpose

The objective of this document is to identify, justify, and characterize candidate **unsupervised routing signals** before any routing policy is designed.

Every candidate signal should answer three questions:

1. **What information does this signal provide?**
2. **Why should this information help routing?**
3. **Is the signal practical enough to justify its probe cost?**

Only signals that satisfy the screening criteria will be retained for the routing policy.

**Scope boundary:** This document covers *why* signals matter. *How* they are computed lives in `05_computation_protocol.md`.

**Contribution note:** \(c(q)\), entropy, and margin are **instances** in 𝒮 — the paper studies families via representatives. Contribution pathway: **characterization → understanding → routing** (`MASTER` §2).

---

# Part A — Candidate Signal Taxonomy

Let **𝒮** denote the pre-inference signal space. The paper **organizes** 𝒮 by **information source** (matches professor transcript); v1 studies one representative per source branch used in this paper.

```text
              Pre-inference signal space 𝒮
         /              |              \
        /               |               \
 Query-derived    Model-derived    Cross-model
 c(q)             H_w, m_w         ΔH, Δm_gain
        \               |               /
         \         (C3: layerwise       /
          \         extension)         /
           \              |           /
            Characterize → Complementarity → Routing evaluation
```

**C3 (layerwise confidence evolution):** methodological extension within **model-derived** (terminal → layerwise)—not a fourth information source. See [`c3_layerwise_concepts.md`](c3_layerwise_concepts.md).

## Query-derived — \(c(q)\)

No forward pass on the routed LLM pool. *(Legacy alias: model-independent.)*

| Family | Examples | v1 status |
| ------ | -------- | --------- |
| **Query complexity** | \(c(q)\) — text entropy, compression, TF-IDF (survey `18`) | **Study one** (D46) → RH1 |
| Lexical complexity | TTR, readability, word rarity | Surveyed; not studied |
| Structural features | length, punctuation density, sentence count | Length dropped D07 |
| Semantic ambiguity | text Shannon entropy, compression ratio | Candidates for \(c(q)\) |
| Graph / interaction features | task–query graphs (GraphRouter contrast) | Future work |

## Model-derived — \(s(q, M_i)\)

Prefill probe on each pool member (canonical protocol in `05`). *(Legacy alias: model-dependent.)*

| Family | Examples | v1 status |
| ------ | -------- | --------- |
| **Entropy** | token entropy \(H\) | **Studied** → RH2 |
| **Confidence / margin** | log-prob margin \(m\) | **Studied** → RH2, RH3 |
| **Layerwise evolution (C3)** | `stabilization_layer` ⭐, `slope_margin` (secondary) | Extension within model-derived |
| Robustness | paraphrase stability | Deferred D04 → perturbation-derived |
| Hidden-state signals | activation probes (Lugoloobi contrast) | Optional discussion; not our method |

**Experiments test hypotheses, not the full taxonomy.** Perturbation-derived and hidden-state signals position future work—they are not v1 experiment scope.

---

## Cross-model — derived from paired probes

Computed at merge from weak + strong **model-derived** terminal logits:

| Signal | Notation | Role |
| ------ | -------- | ---- |
| Model disagreement | \(\Delta H = H_w - H_s\) | Recoverability / disagreement |
| Escalation potential | \(\Delta m_{\mathrm{gain}} = m_s - m_w\) | Recoverability / margin gain |

**Role:** Tests **RH2–RH3** — difficulty vs recoverability structure.

---

## Query-derived signals (implementation detail)

\(c(q)\) — also written **query-derived** (legacy: *model-independent*)

| Signal | Notation | What it is | Cost |
| ------ | -------- | ---------- | ---- |
| **Query complexity** (selecting) | \(c(q)\) | One unsupervised difficulty proxy from query text (D46) | No model pass |

**Role:** Tests **RH1** — Experiment 1.

**Status:** Selection in progress (`18`). Length dropped (D07).

## Model-derived signals (implementation detail)

\(s(q, Mᵢ)\) — legacy alias: *model-dependent*

| Signal | Notation | v1 |
| ------ | -------- | -- |
| Token entropy | \(H\) | **Keep** |
| Log-prob margin | \(m\) | **Keep** |

**Role:** Tests **RH2** (Exp 2), **RH3** within-family (Exp 3).

Initial candidates (not v1):

- Token entropy
- Confidence / log-probability margin
- Paraphrase stability
- Additional candidates identified from literature

### Additional candidates (from literature)

| Signal                     | References | Decision | Decision rationale |
| -------------------------- | ---------- | -------- | ------------------ |
| Self-consistency (cheap k) |            |          |                    |
| Perplexity                 |            |          |                    |
| Sequence length            |            |          |                    |

---

> **Computation:** Definitions, prefill probe protocol, implementation, and cost → `05_computation_protocol.md`. Do not add method detail here.

---

# Part C — Initial Candidate Signals

## Signal: Query Complexity

| Field                         | Content                                                                                               |
| ----------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Scientific role**           | Estimate query difficulty before inference                                                            |
| **Type**                      | Model-agnostic                                                                                        |
| **Notation**                  | `s(q)`                                                                                                |
| **Mechanism**                 | More complex queries are expected to require stronger models.                                         |
| **Theoretical justification** | Query complexity is expected to correlate with reasoning requirements.                                |
| **Evidence**                  | **Tier-1 (contrast):** Hybrid LLM (Ding et al., 2024) routes via a **supervised** DeBERTa encoder on query text with BARTScore labels from offline S/L generations — motivates difficulty-aware routing but not unsupervised surface s(q). **Tier-2:** Lugoloobi et al. (2026) show prefill activations encode human IRT difficulty (Spearman ρ ≈ 0.83–0.87) and model-specific success (ρ ≈ 0.40–0.64) as **distinct** signals; probe-guided routing saves cost but requires **success-labelled** linear probes, not our unsupervised s(q). Surface proxies split by target on E2H-AMC (Table 1): **length** ρ ≈ 0.15 vs human IRT, ρ ≈ 0.19–0.30 vs model success rate; **TF-IDF** ρ ≈ 0.72–0.74 vs human IRT but only ρ ≈ 0.25–0.47 vs model success (degrades with reasoning budget). Table 2: TF-IDF AUROC for binary success is setting-dependent (≈ 0.58–0.86) — not uniformly weak, but routing-relevant model-success signal still trails activation probes and decouples from human-aligned lexical cues under extended reasoning. CoT output length tracks **human** difficulty, not model success (Lugoloobi et al., 2026, §3). |
| **References**                | Ding et al. (2024) Hybrid LLM — arXiv:2404.14618. Lugoloobi et al. (2026) — arXiv:2602.09924. |
| **Expected behaviour**        | If a valid unsupervised s(q) exists, harder queries should show larger weak–strong model performance gaps. Tier-2: lexical features may track **human** difficulty better than **model** success; length alone ranks poorly for either target (Table 1). Under extended reasoning, human- and model-perceived difficulty diverge — so s(q) aligned with surface/human complexity may mis-route even when non-trivial (Lugoloobi et al., 2026). |
| **Failure modes**             | **Theoretical:** Surface complexity (length, lexical features) does not reliably reflect model-relative reasoning difficulty; human and model difficulty decouple under extended reasoning (Lugoloobi et al., 2026). **Implementation:** Short hard queries mis-scored by length; TF-IDF is corpus-dependent and may encode human-IRT rather than pool-specific success; unsupervised heuristics underperform supervised activation probes for model-success prediction (Lugoloobi et al., 2026 — contrast, not our method). |
| **Decision**                  | **Defer**                                                                                             |
| **Decision rationale**        | RH1 remains open but no unsupervised s(q) spec (`05` §5 deferred). Tier-2 does **not** rule out a lexical/corpus s(q), but the routing-relevant target (model success / weak–strong gap) is weakly or mis-aligned with best surface proxies (Lugoloobi Table 1–2). Model-dependent Keep signals (entropy, margin) cover the stronger pre-inference path. Revisit after entropy/margin pilot if a model-agnostic signal is still needed for RH1. → D05 |

**Supports:** H1, RH1

---

## Signal: Token Entropy

| Field                         | Content                                                                                     |
| ----------------------------- | ------------------------------------------------------------------------------------------- |
| **Scientific role**           | Estimate model uncertainty                                                                  |
| **Type**                      | Model-dependent                                                                             |
| **Notation**                  | `s(q, Mᵢ)`                                                                                  |
| **Mechanism**                 | Higher entropy indicates greater uncertainty about the next token.                          |
| **Theoretical justification** | Predictive uncertainty should increase when the model is less capable of solving the query. |
| **Evidence**                  | **Tier-1:** Not used as pre-inference routing information in reviewed routing papers (`03` §2). **Tier-2:** Token-level uncertainty can discriminate correct vs incorrect generations: semantic entropy outperforms raw (length-normalised) token entropy and P(True) on TriviaQA/CoQA QA (Kuhn et al., 2023, Figs 1–2). Larger pre-trained LMs are well-calibrated on formatted MCQ; calibration improves with size and few-shot (Kadavath et al., 2022, §2–3). Aligned/chat LMs are **overconfident** — alignment shifts **answer uncertainty** and conflates it with format uncertainty (He et al., 2023, §3–4). Raw token entropy used for cascade thresholding is **weaker** than calibrated token-margin in production NER cascade (UCCI: Kotte et al., 2026 — beats entropy thresholding at matched F1). |
| **References**                | Kadavath et al. (2022) — arXiv:2207.05221. He et al. (2023) — arXiv:2310.11732. Kuhn et al. (2023) — arXiv:2302.09664. Plaut et al. (2025) — arXiv:2402.13213. Kotte et al. (2026) UCCI — arXiv:2605.18796. |
| **Expected behaviour**        | Higher token entropy on difficult queries for weaker models; incorrect answers associated with more semantically distinct sample clusters (Kuhn et al., 2023, Table 2). Uncertainty–accuracy link may strengthen with model capability on MCQ (Kadavath et al., 2022; Plaut et al., 2025 — via MSP/entropy family). For routing: entropy may **rank** query difficulty per model even when miscalibrated (Plaut et al., 2025 — ranking ≠ calibration). |
| **Failure modes**             | **Theoretical:** Semantic equivalence — different surface forms, same meaning — inflates token entropy (Kuhn et al., 2023, §3–4); models may stay confidently wrong (Kadavath et al., 2022 — self-eval challenge). **Implementation:** Chat/aligned models miscalibrated and overconfident vs pre-trained (He et al., 2023; Plaut et al., 2025); cross-model comparison requires consistent prompt/format (He et al., 2023 — format uncertainty); raw entropy sensitive to tokenizer and decoding temperature (Kuhn et al., 2023, §6.2); cascade work finds raw entropy thresholding suboptimal vs calibrated margin (Kotte et al., 2026). |
| **Dependencies**              | Access to token log-probabilities                                                           |
| **Probe cost**                | One forward pass                                                                            |
| **Decision**                  | **Keep**                                                                                    |
| **Decision rationale**        | Tier-2 motivates mechanism; gap in Tier-1 routing; one forward pass (A2); spec complete (`05` §2). Primary **model-derived** probe for pilot. → D02 |

**Supports:** H2, RH2

---

## Signal: Log-Probability Margin

| Field                         | Content                                                                     |
| ----------------------------- | --------------------------------------------------------------------------- |
| **Scientific role**           | Measure top-1 vs top-2 separation in \(P(\text{next token} \mid q, M_i)\)   |
| **Type**                      | Model-dependent                                                             |
| **Notation**                  | `s(q, Mᵢ)`                                                                  |
| **Mechanism**                 | Smaller margins indicate less reliable predictions.                         |
| **Theoretical justification** | Ambiguous token distributions indicate uncertainty.                         |
| **Evidence**                  | **Tier-1:** FrugalGPT uses post-gen DistilBERT scorer g(q, answer) — different timing (`03` §2.2). **Tier-2:** Pre-trained LMs produce well-calibrated MCQ probabilities with appropriate format; RLHF models need temperature rescaling (Kadavath et al., 2022, §3.3). Chat LLM **maximum softmax probability (MSP)** and **max logit** predict correctness (AUROC) even when miscalibrated; predictiveness correlates with Q&A accuracy (R² = 0.94 MSP, Plaut et al., 2025) but **calibration error does not improve** with capability. Aligned LMs overconfident due to shifted **answer uncertainty** vs format uncertainty (He et al., 2023). UCCI aggregates per-token margin mₜ = p₁ − p₂ over generation; **calibrated** margin beats raw **entropy thresholding** for cascade routing (Kotte et al., 2026, §4.1, §6). |
| **References**                | Kadavath et al. (2022) — arXiv:2207.05221. He et al. (2023) — arXiv:2310.11732. Plaut et al. (2025) — arXiv:2402.13213. Kotte et al. (2026) UCCI — arXiv:2605.18796. |
| **Expected behaviour**        | Smaller top-1 vs top-2 margin on first probe token(s) when model is uncertain; wrong MCQ answers associated with lower MSP/max logit for capable chat models (Plaut et al., 2025). Margin may support **query ranking** across difficulty within a model without calibrated absolute probabilities. Weak models should show lower margins on queries where strong models succeed. |
| **Failure modes**             | **Theoretical:** Margin/confidence may not equal correctness — monotonic ranking can hold without calibration (Plaut et al., 2025); models overconfident on own samples (Kadavath et al., 2022, §4). **Implementation:** Alignment destroys pre-trained calibration; ICL fixes format preference but not aligned overconfidence on MCQ (He et al., 2023); MSP conflates answer and format logits; cross-tokenizer margin comparison unreliable; raw margin weak for routing until calibrated (Kotte et al., 2026 — author-stated: raw scores miscalibrated, workload-specific thresholds). |
| **Dependencies**              | Token log-probabilities                                                     |
| **Probe cost**                | One forward pass                                                            |
| **Decision**                  | **Keep**                                                                    |
| **Decision rationale**        | Same probe pass as entropy; Tier-2 ranking evidence (Plaut et al., 2025); spec complete (`05` §3). Test complementarity vs entropy in pilot (RH3). → D03 |

**Supports:** H3, RH2

---

## Signal: Paraphrase Stability

| Field                         | Content                                                                         |
| ----------------------------- | ------------------------------------------------------------------------------- |
| **Scientific role**           | Estimate robustness of the model for a query                                    |
| **Type**                      | Model-dependent                                                                 |
| **Notation**                  | `s(q, Mᵢ)`                                                                      |
| **Mechanism**                 | Suitable models should respond consistently to semantically equivalent queries. |
| **Theoretical justification** | Robust representations should remain stable under surface-level reformulations. |
| **Evidence**                  | **Tier-1:** Not in reviewed routing papers (`03` §2). **Tier-2:** Kuhn et al. (2023) formalise **semantic equivalence** — paraphrases sharing meaning should not inflate uncertainty. Token entropy treats “Paris” vs “It's Paris” as distinct, overstating uncertainty; **semantic entropy** (cluster by bidirectional entailment, entropy over meaning-clusters) better predicts QA correctness than token entropy and P(True) (Kuhn et al., 2023, §4, §6). Entailment clustering accuracy ~93–96% on TriviaQA/CoQA (Kuhn et al., 2023, §B.2). Implies probe signals should be stable under meaning-preserving paraphrase; unstable probes may indicate model–query misfit. |
| **References**                | Kuhn et al. (2023) — arXiv:2302.09664. |
| **Expected behaviour**        | For a well-matched model, meaning-preserving paraphrases of q should yield similar probe statistics (entropy/margin); large drift suggests uncertainty or poor fit. Incorrect answers show more semantically distinct generations under resampling (Kuhn et al., 2023, Table 2) — paraphrase **instability** may co-occur with error. |
| **Failure modes**             | **Theoretical:** Poor paraphrases change task semantics — stability drops for wrong reasons; semantic clustering errors (~7% entailment mistakes, Kuhn et al., 2023) propagate to stability scores; method does not detect deliberate deception, only spread over meanings (Kuhn et al., 2023, §7). **Implementation:** Requires paraphrase generator + k probe calls (cost, A2); bidirectional entailment model adds dependency; temperature trade-off between sample diversity and accuracy affects stability estimates (Kuhn et al., 2023, §6.2). |
| **Dependencies**              | Paraphrase generation and repeated probing                                      |
| **Probe cost**                | Multiple inexpensive probe calls                                                |
| **Decision**                  | **Defer**                                                                         |
| **Decision rationale**        | Tier-2 motivates (Kuhn et al., 2023) but cost scales with paraphrase count × pool size + paraphraser (A2 borderline). Spec exists (`05` §4) for phase-2 after core probes validated. → D04 |

**Supports:** H4, RH2

---

# Part D — Signal Screening

Each candidate signal will be evaluated using the following criteria.

A signal is retained only if it satisfies **all** criteria.

| Criterion                                        | Pass / Fail | Assumption |
| ------------------------------------------------ | ----------- | ---------- |
| Computable before expensive full inference (A4) |             | A4         |
| Theoretically justified                          |             | —          |
| Probe cost acceptable                            |             | A2         |
| Potentially discriminative across the probe pool |             | —          |
| Reproducible using available infrastructure      |             | A6         |

Signals that fail one or more criteria will be dropped or deferred, and the decision will be recorded in `09_decision_register.md`.

**Target:** ~6 screened → **2 kept** for pilot, 3 deferred, 4 dropped (see Part F).

### Screening — kept signals (pilot)

| Signal | A4 pre-gen | Justified | A2 cost | Discriminative (hypothesis) | A6 reproducible |
| ------ | ---------- | --------- | ------- | --------------------------- | --------------- |
| Token entropy | Pass | Pass | Pass (1 pass) | Pass (to test) | Pass (needs logprobs) |
| Log-probability margin | Pass | Pass | Pass (shared pass) | Pass (to test) | Pass (needs logprobs) |

---

# Part E — Signal Relationships

Document how signals relate to one another after characterization.

| Signal A   | Signal B             | Relationship                                      |
| ---------- | -------------------- | ------------------------------------------------- |
| Entropy    | Margin               | Same forward pass; highly correlated — test RH3   |
| Entropy    | Complexity (defer) | Would be complementary if complexity spec added   |
| Complexity | Paraphrase (defer)   | Mostly independent                                |

This analysis helps identify redundant signals before routing experiments.

---

# Part F — Signal Taxonomy (frozen — D56)

**Taxonomy frozen:** 2026-06-20. **Main-study scope:** representative \(c(q)\), entropy, margin. Decisions in `09_decision_register.md`.

**Decision labels:** **Keep** = main study. **Defer** = out of v1. **Drop** = out of scope.

## Main-study scope

Implement **Keep** signals: representative \(c(q)\) (D46 pending), token entropy, log-probability margin.

### Model-dependent signals (canonical reference)

| Signal | Notation | What it is | Cost |
| ------ | -------- | ---------- | ---- |
| **Token entropy** | \(H(q, M_i)\) | Spread of \(P(\text{next token} \mid q, M_i)\) | 1 forward pass |
| **Log-probability margin** | \(m(q, M_i)\) | \(p^{(1)} - p^{(2)}\) on the same distribution | Same pass (no extra cost) |
| **Max next-token probability** (auxiliary) | \(p_{\max}(q, M_i)\) | Peak probability; CSV `max_prob`; **not a primary signal** | Same pass |

Primary paper signals: \(H\) and \(m\) only. Do not call \(p_{\max}\) or \(m\) “confidence.”

## Model-agnostic signals

| Signal | Decision | Scientific rationale | Engineering rationale |
| ------ | -------- | -------------------- | --------------------- |
| Query complexity \(c(q)\) | **Keep** | RH1 — representative 𝒮_indep; D46 screening (`18`); formula in `05` §8 when implemented. | `05` §8 pending. |
| Query length | **Drop** | ρ ≈ 0.15 vs human IRT; ρ ≈ 0.19–0.30 vs model success (Lugoloobi Table 1) — poor discriminator for routing hypotheses. | Redundant with complexity if a composite s(q) is ever defined. |
| Domain / task category | **Drop** | Tier-1 task routing uses **supervised** task nodes (GraphRouter); no unsupervised s(q) mechanism within RQ scope. | Requires labels or metadata not assumed in probe-only setting. |

## Model-dependent signals

| Signal | Decision | Scientific rationale | Engineering rationale |
| ------ | -------- | -------------------- | --------------------- |
| Token entropy | **Keep** | Tier-2 motivates uncertainty–correctness link; absent as pre-gen routing probe in Tier-1 (`03`); satisfies A4; directly tests RH2 / H2. | One forward pass per (q, Mᵢ); needs logprobs; spec complete (`05` §2). |
| Log-probability margin | **Keep** | Tier-2 ranking evidence (Plaut et al., 2025); distinct statistic from entropy for RH3 / H3 complementarity test; satisfies A4. | Zero marginal probe cost vs entropy (same pass); spec complete (`05` §3). |
| Paraphrase stability | **Defer** | Tier-2 (Kuhn et al., 2023): meaning-preserving reformulation should not inflate uncertainty; tests robustness (H4). | P paraphrases × pool size probe multiplier + paraphraser infra; spec drafted (`05` §4) for phase-2. |
| Perplexity | **Drop** | Mechanism overlaps prompt-token entropy; no distinct Tier-1/Tier-2 advantage for routing over H. | No separate spec; would duplicate entropy probe pass. |
| Self-consistency (cheap k) | **Defer** | Multiple cheap samples may estimate epistemic spread; related to uncertainty family (RH2). | k-sample generation cost; closer to cascade than single prefill probe (A2 borderline). Evaluate after single-pass pilot. |
| Sequence length (predicted) | **Drop** | No distinct pre-inference routing mechanism identified in Tier-1/Tier-2. | Overlaps entropy/perplexity; requires generation or extra prediction step. |

**Count:** 3 Keep · 2 Defer · 4 Drop

---

# Part G — Open Research Questions

| Question | Status |
| -------- | ------ |
| Which representative \(c(q)\)? | **D46 pending** — implement + record in `05` §8 |
| Does entropy remain comparable across tokenizers? | **Open** — test in main study |
| Is paraphrase stability worth its probe cost? | **Deferred** — D04 |
| Entropy vs margin complementarity (RH3)? | **Open** — Study III |

---

---

> **Reading archive:** `14_literature_record.md`

