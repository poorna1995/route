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

## Section order (ACL-conventional)

Present **Results** around three findings; use Study I–IV labels only in Method / appendix.

```text
1 Introduction          — routing problem; answer = partially
2 Related Work          — supervised / post-generation vs pre-inference
3 Method                — dimensions, prefill extraction, supervision boundary
4 Experimental Setup    — pool, splits, oracle, metrics
5 Empirical Evidence    — F1 existence; F2 structure; complementarity
6 Routing Evaluation    — F3 exploitation gap
7 Discussion            — limits, future measurements
8 Conclusion
```

---

## Paper vocabulary (locked — see `claims.md`)

| Use in `paper/` | Never in `paper/` |
| --------------- | ----------------- |
| unsupervised pre-inference signals | Protocol v1/v2, L1–L4 |
| information dimension | framework (as novelty) |
| representative statistic | measurement protocol |
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

| Done | Item | File |
| ---- | ---- | ---- |
| [x] | T1 Setup | `tables/T1_setup.tex` |
| [x] | T2 Characterization (RH1–RH2) | `tables/T2_correlation.tex` |
| [x] | T3 Complementarity (RH3) | `tables/T3_complementarity.tex` |
| [x] | T4 Routing (RH4) | `tables/T4_routing.tex` |

---

## LaTeX

`paper/acl.tex` — build: `pdflatex acl && bibtex acl && pdflatex acl && pdflatex acl`
