# Oracle Definition

> **Scientific decision this document supports:** How do we define the "correct" routing decision during offline evaluation?  
> **Frozen (D33, D55):** ARC-Challenge · Llama 3.2 1B / 3B · greedy decode · MCQ letter match.  
> Assumption A5: oracle is **offline only**. Routing never uses oracle labels.

---

## What is the oracle?

For each query \(q\), the oracle identifies which model in the probe pool would have been best **if we had run expensive full inference (full answer generation) on all pool members**.

The oracle is an **evaluation construct**, not a routing method.

Pre-inference signals are **never** used to compute the oracle.

This follows the same spirit as RouterBench: offline outcome labels on a fixed pool, same prompt, deterministic generation, **cheapest correct model wins** on ties.

---

## Per-model correctness

For each \((q, M_i)\):

1. Build prompt \(P(q)\) — **same template** as prefill probe (`05` §1).
2. Run **full inference** (full answer generation) with \(M_i\).
3. Score output with the task correctness metric (below).
4. Assign success \(y(q, M_i) \in \{0, 1\}\).

**Frozen pool (primary):** weak \(M_{\text{weak}}\) = Llama-3.2-1B-Instruct; strong \(M_{\text{strong}}\) = Llama-3.2-3B-Instruct.

---

## Oracle model selection

\[
M^*(q) = \arg\max_{M_i \in \mathcal{M}} \; y(q, M_i)
\]

**Tie-break:** prefer **lowest cost** among models tied at maximum quality (aligns with A2).

---

## Routing-opportunity label (primary binary target)

Experiments use a **binary routing-opportunity indicator**, not only the correctness gap \(\Delta(q)\).

**Correctness gap** (auxiliary):

\[
\Delta(q) = y(q, M_{\text{strong}}) - y(q, M_{\text{weak}})
\]

**Routing opportunity** (what AUROC / opportunity detection use):

\[
y_{\text{opp}}(q) =
\begin{cases}
1, & y(q, M_{\text{weak}}) = 0 \;\wedge\; y(q, M_{\text{strong}}) = 1 \\
0, & \text{otherwise}
\end{cases}
\]

Interpretation: the weak model fails but the strong model succeeds — the case where routing to the strong model would have helped.

---

## Four-bucket stratification (characterization)

For distribution plots and pairwise separability, each query is also assigned one bucket:

| Bucket | Condition | Routing interpretation |
| ------ | --------- | ---------------------- |
| **easy** | \(y_w = 1,\; y_s = 1\) | Weak model sufficient |
| **opportunity** | \(y_w = 0,\; y_s = 1\) | Routing to strong would help (\(y_{\text{opp}} = 1\)) |
| **weak_only** | \(y_w = 1,\; y_s = 0\) | Rare; strong harmful |
| **too_hard** | \(y_w = 0,\; y_s = 0\) | No pool member succeeds |

where \(y_w = y(q, M_{\text{weak}})\) and \(y_s = y(q, M_{\text{strong}})\).

---

## Correctness metric (frozen — primary)

| Dataset | Metric | Normalization | Status |
| ------- | ------ | ------------- | ------ |
| ARC-Challenge | MCQ letter exact match | Extract A/B/C/D from generation | **Locked (D33)** |
| MMLU (subjects) | MCQ letter exact match | Same protocol | Locked for gen |
| BoolQ (optional) | Yes/no match | Same protocol | If run |

Requirements: objective, binary success, standard in routing literature.

---

## Generation settings (oracle only)

Distinct from prefill probe settings (`05`: prefill, \(\tau = 1\)).

| Setting | Frozen value | Rationale |
| ------- | ------------ | --------- |
| Temperature | \(0\) (greedy) | Reproducible labels |
| Max new tokens | \(8\) (ARC/MCQ) | Sufficient for letter answer |
| Seed | Fixed per \((q, M_i)\) | A6 |

---

## Tie-breaking

| Situation | Oracle choice |
| --------- | ------------- |
| One model correct, other incorrect | Correct model |
| Both correct | Lower-cost model |
| Both incorrect | Lower-cost model (document as no routing gain) |

---

## Evaluation splits

| Split | Purpose | Size (ARC) |
| ----- | ------- | ---------- |
| **Train** | — | 1,119 (**not used** — no model/router training) |
| **CALIB** | D46 screening + Study III/IV calibration | ~299 (official validation) |
| **TEST** | Studies I–IV reported metrics | 1,172 (official test) |

Seed 42. See `MASTER.md` §5.

---

## What the oracle is NOT used for

- Training a supervised router
- Online routing
- Computing pre-inference signals
- Signal extraction or probe passes

---

## Analysis questions (evidence, not routing)

1. When \(y_{\text{opp}} = 1\), do query-derived, model-derived, and cross-model signals shift as hypothesized? (RH1, RH2)
2. Do the three information sources provide complementary information beyond any single source? (RH3)
