# Paper Outline

> **Scientific decision this document supports:** How does the research notebook compress into prose?

> **Milestone M5** — Skeleton only. Venue: ACL (Aug 3) — changeable without editing `research/`.

> **Workflow (D63/D64):** RQ → hypotheses → tables → experiments → fill tables. See `08_evaluation_design.md`.

---

## Paper-first workflow

```text
Research question
        ↓
Hypotheses (RH1–RH4)
        ↓
Table shells (T2–T4)
        ↓
Experiments (only what fills tables)
        ↓
Fill tables → Discussion (outcome scenario: positive/mixed vs limits)
```

**Guiding principle:** Every experiment → paragraph → hypothesis → RQ.

**Do not:** add agents, new signals, or extra benchmarks before v1 tables are filled.

---

## Section order (ACL-conventional)

```text
1 Introduction
2 Related Work
3 Method
4 Experimental Setup
5 Signal Characterization      ← primary results
6 Routing Evaluation           ← conditional on §5
7 Discussion
8 Conclusion
```

Method → setup → results. Do not place routing before experimental setup.

---

## Paper vocabulary vs notebook

**In `paper/` prose:** Experimental Setup → Signal Characterization → Routing Evaluation only.

**In `research/` only:** Phase A/B/C/D, C1–C6, V1/V2, EXP-01–03, smoke/apparatus/configuration validation as named phases. Mapping table → `claims.md` §Paper vocabulary vs notebook.

---

## Source mapping

| Paper section | Notebook source | Draft file |
|---------------|-----------------|------------ |
| §1 Introduction | `00`, `01`, `02` | `paper/draft/01_introduction.tex` |
| §2 Related Work | `03_literature_gap.md` | `paper/draft/02_related_work.tex` |
| §3 Method | `04`, `05` (signals, protocol, prefill probe) | `paper/draft/03_method.tex` |
| §4 Experimental Setup | `MASTER` §5, `07`, `08` (pool, oracle, datasets by purpose, metrics) | `paper/draft/04_experimental_setup.tex` |
| §5 Signal Characterization | EXP-01, EXP-02 — Studies I–III | `paper/draft/05_results_characterization.tex` |
| §6 Routing Evaluation | EXP-03 — Study IV | `paper/draft/06_results_routing.tex` |
| §7 Discussion | results interpretation | `paper/draft/07_discussion.tex` |
| §8 Conclusion | `00` contribution framing | `paper/draft/08_conclusion.tex` |

---

## Title (working)

Characterizing Unsupervised Routing Signals for LLM Selection

*(Alt: Unsupervised Routing Signal Analysis for Multi-LLM Selection)*

---

## §1 Introduction

- [ ] **Problem:** Current LLM routing is largely **supervised** (labels, prefs, trained routers) or uses information **after** full generation
- [ ] **Gap:** Little is known about whether **inexpensive unsupervised pre-inference signals** can estimate routing need before full inference
- [ ] **Contrast figure:** supervised router path vs query → signals → routing need → LLM (`MASTER.md` §2)
- [ ] **Motivation:** Different queries need different capability; one LLM for all is inefficient
- [ ] **RQ (frozen):** How **informative** are unsupervised routing signals for LLM selection before full model generation?
- [ ] **Appropriate selection:** Weakest model expected to succeed; escalate only when signals indicate benefit (not vague “suitable LLM”)
- [ ] **Supervision:** Signals unsupervised at inference; oracle labels offline for evaluation only (`claims.md`)
- [ ] **Contribution (headline):** unsupervised pre-inference **signal extraction and characterization** for routing — **not** “unsupervised routing” or “we built a router”
- [ ] **Scientific core:** How much **routing-relevant information** is in inexpensive pre-inference signals?
- [ ] **RQ (frozen):** How **informative** are unsupervised pre-inference routing signals…?
- [ ] **Contribution pathway:** Query → signal extraction → characterization → simple policy (**demonstration**)
- [ ] **Supervision (three layers):** unsupervised signal computation · oracle for evaluation · CALIB calibration for demonstration only
- [ ] **Hypothesis chain:** RH1 (indep. info) → RH2 (dep. info) → RH3 (complementarity) → RH4 (exploit what exists)
- [ ] **Outcome branching:** prepare limits-paper framing if all AUROC ≈ 0.50 (`claims.md` §Outcome scenarios)
- [ ] Contributions (four layers):
  1. Problem formulation — estimate routing need before expensive full inference
  2. Signal space 𝒮 — model-independent + model-dependent; v1: \(c(q)\), \(H\), \(m\)
  3. Empirical characterization — **primary** (Studies I–III)
  4. Simple routing + cost–quality — **conditional** (Study IV)

## §2 Related Work

- [ ] From `03_literature_gap.md` — contrast supervised routers vs pre-inference signal analysis

## §3 Method

- [ ] 3.1 Formulation (fixed pool, pre-inference constraint A4, oracle as eval construct)
- [ ] 3.2 Signal taxonomy 𝒮 — pre-inference space; model-independent vs model-dependent **families**
- [ ] 3.3 Studied signals: representative \(c(q)\) (complexity family), entropy \(H\) (uncertainty), margin \(m\) (confidence separation)
- [ ] 3.4 Operational informativeness definition (`claims.md`)
- [ ] 3.5 **Supervision (three layers)** — unsupervised signal computation; oracle labels for characterization only; CALIB-supervised policy as demonstration (`claims.md`)
- [ ] 3.6 Prompt protocol + probe computation (`05`, `prompt_protocol.py`)
- [ ] **Not here:** locked pool/dataset (§4), routing policy details (§6)

## §4 Experimental Setup

- [ ] 4.1 Configuration selection (requirements-first; candidate screening)
- [ ] 4.2 Locked / candidate triple \((\text{pool}, \text{dataset}, \text{protocol})\)
- [ ] 4.3 Datasets · 4.4 Models and inference · 4.5 Oracle labels · 4.6 Metrics · 4.7 Reproducibility
- [ ] Table T1 (`tables/T1_setup.tex`)

## §5 Signal Characterization (primary results)

- [ ] 5.1 Setup recap + probe cost (T5 analytical)
- [ ] 5.2 **Study I** — model-independent (RH1): \(c(q)\)
- [ ] 5.3 **Study II** — model-dependent (RH2): \(H\), \(m\)
- [ ] 5.4 Generalization — MMLU subset (BoolQ appendix only if time)
- [ ] 5.5 **Study III** — complementarity (RH3)
- [ ] Paraphrase: **future work** paragraph only (D04)—not an experiment
- [ ] Failure modes / negative results (within §5 or §7)
- [ ] Tables T2–T3; figures F1–F2

## §6 Routing Evaluation (Study IV — RH4, demonstration)

- [ ] **Framing:** Can a simple policy **exploit whatever information Studies I–III establish**? (Not “our router works.”)
- [ ] Simple policy: logistic + threshold (+ optional λ) fit on CALIB from \(y_{\text{opp}}\)
- [ ] vs always-weak, always-strong, oracle; cost–quality on TEST
- [ ] Table T4; figure F3
- [ ] If signals uniformly null: one honest paragraph; shrink or omit large utility table — emphasize limits in §7

## §7 Discussion

- [ ] What 𝒮 teaches; calibration vs ranking; limits; future signals
- [ ] **Future work roadmap:** agents/orchestration; learned signal combination; larger pools (one paragraph — not new experiments)

## §8 Conclusion

- [ ] One paragraph: RQ answer, main finding, contribution (including negative results)

---

## Hypothesis → table checklist

| Done | Item | Draft file |
| ---- | ---- | ---------- |
| [ ] | T1 Setup | `tables/T1_setup.tex` |
| [ ] | T2 Characterization (RH1 + RH2) | `tables/T2_correlation.tex` |
| [ ] | T3 Complementarity (RH3) | `tables/T3_complementarity.tex` |
| [ ] | T4 Routing (RH4) | `tables/T4_routing.tex` |

---

## Paper paragraph (prompt protocol — draft from code)

> **Prompt protocol.** Each benchmark instance is converted into a task-specific user message using deterministic formatting rules (e.g., numeric-only output for GSM8K, option-letter output for ARC). The user message is wrapped with the model's native chat template. The identical prompt is used for signal extraction (**prefill probe**) and oracle evaluation (**full inference**, offline labels only).

---

## LaTeX

`paper/acl.tex` — section order matches outline above.
