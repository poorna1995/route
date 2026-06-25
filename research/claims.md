# Claims — Paper vocabulary & evidence map

> **Frozen design:** [`MASTER.md`](MASTER.md) · **Metrics detail:** `08_evaluation_design.md` · **Paper outline:** `11_paper_outline.md`

---

## Contribution (one paragraph — paper Intro / Abstract)

> Current LLM routing methods largely rely on supervised routers, preference data, or information available only after full answer generation. We study **routing based on unsupervised pre-inference signals**: inexpensive model-independent features and model-dependent probe statistics extracted before full generation. We characterize **how much routing-relevant information** these signals carry, test whether the families complement each other, and evaluate whether a simple calibrated policy can exploit whatever information exists. Signal **computation** is unsupervised; oracle labels are used only to **evaluate** informativeness and to **calibrate** the demonstration policy on a held-out validation split—not to extract signals.

**Headline (not):** “We built a router” · “unsupervised routing.”  
**Headline (yes):** unsupervised pre-inference **signal extraction and characterization** for routing.

**Contribution pathway:**

```text
Query → Unsupervised signal extraction → Signal characterization → Simple routing policy (demonstration)
```

---

## Goal vs contribution

| | Goal | Contribution |
| --- | --- | --- |
| **Framing** | Route efficiently across LLMs | **Estimate and characterize** pre-inference signals |
| **Studies I–III** | — | How much routing-relevant information? Complementarity? |
| **Study IV** | Demonstration | Simple policy exploits whatever information exists |

---

## Supervision (Methods — three layers; D64)

> **Signal computation** is unsupervised: \(c(q)\), \(H\), and \(m\) are extracted without routing labels or router training. **Signal characterization** (Studies I–III) uses offline oracle labels (\(y_{\text{opp}}\), buckets) only to measure predictive association (ρ, AUROC, ΔAUROC)—never to define signals. **Routing policy** (Study IV) fits a simple logistic model and threshold on CALIB from oracle-derived labels; this supervised **calibration** is a demonstration built on unsupervised signals, not the primary contribution and not an end-to-end supervised router.

**Reviewer FAQ:** *“Why call this unsupervised if you fit logistic regression?”* → Unsupervised refers to **signal extraction**. Policy calibration is explicitly secondary and uses labels only on CALIB.

---

## Outcome scenarios (Abstract/Discussion branching)

| TEST outcome | Paper emphasis |
| ------------ | -------------- |
| Some signal informative | Information present; policy may help (RH4) |
| All families null (AUROC ≈ 0.50) | **Limits paper:** what these probes do *not* provide under this protocol; do not claim enabling routing |

---

## Paper vocabulary (use in `paper/` only)

```text
Signal families → Characterization (I–II) → Understanding (III) → Utility (IV)
```

**Studies (ACL naming):**

| Study | Layer | Name | RH |
| ----- | ----- | ---- | -- |
| I | Characterization | Model-independent | RH1 |
| II | Characterization | Model-dependent | RH2 |
| III | Understanding | Complementarity | RH3 |
| IV | Utility | Lightweight routing | RH4 |

**Terminology (locked):**

| Concept | Wording |
| ------- | ------- |
| Main object | **unsupervised pre-inference routing signals** |
| Headline contribution | **signal extraction and characterization for routing** |
| Avoid headline | *unsupervised routing* · *we built a router* |
| Taxonomy | **model-independent** / **model-dependent** (feature vector) |
| Study IV | **routing demonstration** (simple calibrated policy) |
| Operational property | **before full model generation** (define *pre-inference* once) |

Notebook IDs (V1, EXP-01–03) stay in `research/` only — never in paper prose. EXP-01 = signal characterization (Studies I–II).

---

## Informativeness (one paragraph for Methods / Intro)

> We use _informativeness_ operationally: the degree to which an unsupervised routing signal shows predictive association with routing need (offline oracle opportunity and weak–strong correctness gap). We quantify this with Spearman correlation, AUROC/AUPRC, and complementary predictive gain (ΔAUROC when combining signal families)—not with information-theoretic mutual information.

---

## Literature positioning (intro / related work)

| Category | Routing information | Examples |
| -------- | ------------------- | -------- |
| Supervised routers | Labels, prefs, offline gens | RouteLLM, Hybrid LLM, GraphRouter |
| Label-free post-generation | Pool outputs | Smoothie, CASCAL |
| **This work** | Unsupervised pre-inference signals (before full generation) | \(c(q)\), \(H\), \(m\) |

---

## Datasets (by purpose — paper §4)

| Purpose | Dataset |
| ------- | ------- |
| Primary characterization | ARC-Challenge |
| Generalization | MMLU (2 subjects) |
| Robustness (optional) | BoolQ |

---

## Hypothesis → evidence → table (paper-first)

Each hypothesis maps to **one primary table**. Run only experiments that fill these tables.

| Hypothesis | Study | Primary table | Key outputs |
| ---------- | ----- | ------------- | ----------- |
| **RH1** | I (model-independent) | T2 (indep. column) | ρ, AUROC, distributions for \(c(q)\) |
| **RH2** | II (model-dependent) | T2 (dep. column) | ρ, AUROC for \(H, m\); partial ρ \| \(c\) |
| **RH3** | III (complementarity) | T3 | ΔAUROC ladder \(c → c{+}H → c{+}H{+}m\) |
| **RH4** | IV (demonstration) | T4 | Exploit whatever information exists; cost–quality vs baselines |

**Guiding principle (D64):** Every experiment → justifies a paragraph → justifies a hypothesis → answers the RQ. Otherwise → future work.

**Null results:** Fill tables honestly. If all signals null on corrected TEST, write a **limits** paper (Abstract/Intro/Discussion)—do not add rescue experiments.

---

## Paper metrics (v1 — no additions)

Spearman ρ + CI · AUROC/AUPRC · bucket separability · ΔAUROC · opportunity rate · probe cost (T5).

**Not in v1:** ECE suite · MI headlines · median-heuristic routing in paper.

---

## Future work (not v1 — one line in Discussion)

| Paper | Topic |
| ----- | ----- |
| **This paper** | Unsupervised pre-inference signal extraction + characterization → LLM routing demonstration |
| Future | Learned combination of signals (beyond simple logistic) |
| Future | Extend to agents and multi-step orchestration |

Paraphrase stability deferred (D04)—not evaluated in v1.
