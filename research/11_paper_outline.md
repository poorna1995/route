# Paper Outline

> **Vocabulary:** [`claims.md`](claims.md) · **Frozen design:** [`MASTER.md`](MASTER.md)  
> **Venue:** ACL · **Workflow:** RQ → hypotheses → tables → experiments → fill tables

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
Fill tables → Discussion
```

**Do not:** add agents, new signals, or extra benchmarks before main tables are filled.

---

## Execution order (locked — two-week sprint)

**Resolve C3 uncertainty first**, then write from evidence.

| Day | Task | Outcome |
| --- | ---- | ------- |
| **1** | Implement `layerwise.py` + probe flag; RunPod **smoke n=10** | Stop if smoke fails — C6 → future work |
| **2** | ARC TEST weak + strong layerwise; **F7** + RH5 JSON | Paper scope frozen (with or without §5.7) |
| **3+** | Write (see order below) | Submit-ready draft |

---

## Writing order (not reading order)

Draft in this sequence; **Intro last**.

```text
1. Results + figures     (C1–C5 from claims_evidence_matrix.md; C6 if Day 2 succeeded)
2. Methods + Setup       (only experiments actually run)
3. Discussion
4. Introduction + Related Work
5. Abstract              (one matrix row per sentence)
6. Claim ↔ evidence pass (claims_evidence_matrix.md)
```

**Reading order in PDF** stays ACL-conventional (Intro first for reviewers).

---

## Section order (ACL-conventional)

Present **Results** around three findings; use Study I–IV labels only in Method / appendix.

```text
1 Introduction          — routing problem; pre-inference information question
2 Related Work          — supervised / post-generation vs pre-inference
3 Method                — three information sources; prefill extraction; supervision boundary
4 Experimental Setup    — pool, splits, oracle, metrics
5 Empirical Evidence    — F1 existence; F2 difficulty vs recoverability; MMLU transfer
5.7 Extension (C3)     — layerwise confidence evolution within model-derived (optional subsection)
6 Routing Evaluation    — F3 exploitation gap
7 Discussion            — limits, perturbation-derived future work
8 Conclusion
```

**Anchor:** Study unsupervised signals first. Routing is only an application.

---

## Paper vocabulary (locked — see `claims.md`)

| Use in `paper/` | Never in `paper/` |
| --------------- | ----------------- |
| unsupervised pre-inference signals | Protocol v1/v2, L1–L4 |
| information source (query-derived, model-derived, cross-model) | model-independent / model-dependent (legacy) |
| information dimension | framework (as novelty) |
| difficulty vs recoverability | difficulty-side / escalation-side (internal only) |
| signal characterization | observation store |
| routing evaluation | EXP-01, V1/V2 |

---

## Source mapping

| Paper section | Notebook source | Draft file |
|---------------|-----------------|------------ |
| §1 Introduction | `00`, `01`, `02` | `paper/draft/01_introduction.tex` |
| §2 Related Work | `03_literature_gap.md` | `paper/draft/02_related_work.tex` |
| §3 Method | `04`, `05` | `paper/draft/03_method.tex` |
| §4 Experimental Setup | `MASTER` §5, `07`, `08` | `paper/draft/04_experimental_setup.tex` |
| §5 Empirical Evidence | Studies I–III | `paper/draft/05_results_characterization.tex` |
| §6 Routing Evaluation | Study IV | `paper/draft/06_results_routing.tex` |
| §7 Discussion | interpretation | `paper/draft/07_discussion.tex` |
| §8 Conclusion | `00`, `claims.md` | `paper/draft/08_conclusion.tex` |

---

## Title

**Can Unsupervised Pre-Inference Signals Support Appropriate Multi-LLM Routing?**

---

## Hypothesis → table checklist

See **[`claims_evidence_matrix.md`](claims_evidence_matrix.md)** — every Intro/Abstract claim must map to a row.

| Done | Item | File |
| ---- | ---- | ---- |
| [x] | T1 Setup | `tables/T1_setup.tex` |
| [x] | T2 Characterization (RH1–RH2) | `tables/T2_correlation.tex` |
| [x] | T3 Complementarity (RH3) | `tables/T3_complementarity.tex` |
| [x] | T4 Routing (RH4) | `tables/T4_routing.tex` |

---

## LaTeX

`paper/acl.tex` — build: `pdflatex acl && bibtex acl && pdflatex acl && pdflatex acl`
