# Research Question

> **Frozen project design:** [`MASTER.md`](MASTER.md)

> What is the primary research question and scope of this study?

---

## Research Question

**RQ (frozen):** How informative are **unsupervised pre-inference routing signals** for LLM selection before full model generation?

**Scientific core (same question, ACL-facing):** How much **routing-relevant information** is present in inexpensive pre-inference signals?

**Motivation (research goal — not the headline RQ):** Different queries require different levels of reasoning capability; routing every query through the same LLM is inefficient. For each incoming query, we want to **estimate those signals before generating an answer**—not train another supervised router as the primary contribution.

**Headline contribution phrasing (paper title/abstract):**

- ✓ **Unsupervised pre-inference signal extraction and characterization for routing**
- ✓ **Routing based on unsupervised pre-inference signals**
- ✗ *unsupervised routing* (Study IV uses supervised calibration on CALIB — imprecise)
- ✗ *we built a router* (router is Study IV **demonstration**)

**Operational meaning of *appropriate selection*:** Route to the **weakest model expected to answer correctly**, escalating to a stronger model only when pre-inference signals indicate that additional capability is likely to improve the outcome. Formally: a desirable **cost–quality trade-off** within fixed pool \(\mathcal{M}\)—not “best model” or vague “suitable LLM.”

**Operational meaning of *informativeness*:** Predictive association with routing need (offline oracle buckets, opportunity labels, weak–strong gap)—measured by Spearman \(\rho\), AUROC/AUPRC, and complementary predictive gain (ΔAUROC when adding signal families). **Not** Shannon mutual information \(I(S;Y)\). See `MASTER.md` §3.

**RQ vs application:** The RQ asks *how informative* signals are (characterization). Whether they *reliably enable* cost-effective routing is **RH4 / Study IV**—tested only after Studies I–III.

---

## Goal vs contribution

| | Research goal | Research contribution |
| --- | --- | --- |
| **Question** | Can we route efficiently across LLMs? | **How much routing-relevant information** do pre-inference signals carry? |
| **Object** | Cost–quality routing | Signal extraction → characterization → (demonstration) simple policy |
| **Paper share** | Motivation in Intro | Studies I–III (~70%); Study IV demonstration (~20%) |

**Contribution shift:** Signals are the **research object**; the router is the **demonstration**.

**Central hypothesis chain (each study depends on the prior):**

```text
RH1  Model-independent signals contain routing-relevant information     → Study I
        ↓
RH2  Model-dependent signals contain routing-relevant information       → Study II
        ↓
RH3  Together they provide additional information                       → Study III
        ↓
RH4  A simple policy can exploit whatever information exists            → Study IV
```

---

## Supervision (three layers — D64)

| Layer | Labels? | Role |
| ----- | ------- | ---- |
| **Signal computation** | **No** | Unsupervised — \(c(q)\), \(H\), \(m\) |
| **Characterization** (I–III) | Oracle offline only | Measure informativeness (ρ, AUROC) |
| **Routing policy** (IV) | **CALIB** — logistic from \(y_{\text{opp}}\) | Demonstration; not primary contribution |

*Unsupervised* describes **signal extraction**, not the full system. Reviewer FAQ: logistic on CALIB calibrates a policy **on top of** unsupervised signals—it is not claimed to be an unsupervised router.

### Outcome scenarios (null results)

If **every** signal in **every** family has AUROC ≈ 0.50 on corrected TEST: RH1–RH3 are rejected; RH4 unlikely to show utility. The paper remains valid as a **limits** characterization—Abstract/Intro/Discussion must emphasize what these signals *do not* provide, not enabling routing.

---

## Terminology — locked wording

Use **one phrase per concept**. Do not alternate synonyms in prose.

| Concept | **Preferred wording** | **Role** |
| ------- | --------------------- | -------- |
| Main object | **unsupervised pre-inference routing signals** | Scientific object studied |
| Headline contribution | **unsupervised pre-inference signal extraction for routing** | Paper framing |
| Taxonomy | **model-independent** / **model-dependent** | Feature-vector structure |
| Study IV | **simple routing demonstration** | Exploits whatever information I–III establish |
| Operational property | **before full model generation**; define **pre-inference** once | Deployment constraint |
| Study I–II | **characterization** | Quantify routing-relevant information per family |
| Study III | **understanding** (complementarity) | Incremental information across families |
| Study IV | **utility** (demonstration) | Can a simple policy exploit what exists? |

| Concept | **Avoid as headline** | **Why** |
| ------- | --------------------- | ------- |
| Unsupervised routing | System headline | Study IV uses CALIB-supervised calibration |
| We built a router | Contribution claim | Signals are the contribution |
| Pre-inference signals | Title/RQ lead | Timing is constraint, not novelty |
| Model-independent = unsupervised | Conflation | Taxonomy ≠ learning paradigm |

**Signal extraction:** **prefill probe** (canonical) · **Oracle labels:** offline **full answer generation** only (never define routing signals)

---

## Sub-questions (paper spine)

```text
Information (signals)
        ↓
Characterization (Studies I–II)
        ↓
Understanding (Study III)
        ↓
Utility (Study IV — demonstration)
```

```text
What unsupervised pre-inference signals can we extract?
        ↓
How much routing-relevant information does each family carry?
        ↓
Do families provide additional information together?
        ↓
Can a simple policy exploit whatever information exists?
```

---

## Why this question (2–3 sentences)

Modern multi-LLM systems route queries across models with different capability and cost. Existing routing predominantly uses **supervised routers**, preference data, or information available only **after** full answer generation. This work asks whether **unsupervised pre-inference signals**—model-independent query features and model-dependent probe statistics—carry information about routing need, measured before full model generation. The primary contribution is **signal characterization**, not another trained router. Weak or negative associations are valid findings if reported rigorously.

---

## Contribution framing

See **`MASTER.md`** §1–§4. Pathway: **signal families → characterization (I–II) → understanding (III) → utility (IV)**. Taxonomy \(\mathcal{S}\) organizes model-independent vs model-dependent families; v1 studies representative \(c(q)\), plus \(H\) and \(m\).

---

## Daily log → also record in `09_decision_register.md` if a decision was made

**Decision (2026-06-20):**
Reframe RQ from sufficiency to **informativeness**. Contribution = characterization of unsupervised routing signals, not a single heuristic.

**Decision (2026-06-23):**
Harmonize vocabulary: headline **unsupervised routing signals**; demote **pre-inference** to one-time definition (before full model generation); layer Studies I–IV as characterization / understanding / utility.

**Decision (2026-06-25):**
ACL framing refined (D64): contribution = **signal extraction + characterization**; router = **demonstration**. Avoid headline *unsupervised routing*; use *routing based on unsupervised pre-inference signals*. Three-layer supervision. Null-all-signals outcome → limits paper, not failure.

**Decision (2026-06-25):**
Story sharpened (D63): one problem → four hypotheses → one paper. Paper-first: RQ → hypotheses → tables → experiments.

**Evidence:**
Advisor discussion: “pick one problem and solve it”; “given a query, estimate those signals” (not “train a better router”); write paper then run only hypothesis-driven experiments. Scope concern is **research breadth**, not rejection of the RQ.

**Prior (2026-06-20):**
Reframe RQ from sufficiency to **informativeness**. Advisor: study properties available before inference, not “build another router.” Negative correlations and weak signals are scientifically valid if analysis is rigorous.
