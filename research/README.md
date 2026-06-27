# Research Notebook

```text
research/  →  experiments/  →  paper/
```

## Document hierarchy (do not mix layers)

```text
Research Question     00_research_question.md
        ↓
Problem Statement     01_problem_statement.md
        ↓
Hypotheses            02_research_hypotheses.md
        ↓
Methodology           MASTER.md, 03–08, claims.md, 11_paper_outline.md
        ↓
Experiments           10_experiment_registry.md  (study ↔ notebook IDs)
        ↓
Implementation        ../experiments/README.md, scripts/
```

**Rule:** Script names (`extract_signals.py`, `merge_and_analyze.py`, etc.) belong in **Experiments** and **Implementation** only—not in hypotheses, oracle, signal design, or paper prose.

---

## Start here

**[`MASTER.md`](MASTER.md)** — frozen science: RQ, hypotheses, signals, studies, datasets, metrics, **goal vs contribution** (D63).

**Vocabulary (locked):** [`claims.md`](claims.md) — query-derived · model-derived · cross-model · perturbation-derived (future). Anchor: *Study unsupervised signals first. Routing is only an application.*

**Paper-first workflow:** [`11_paper_outline.md`](11_paper_outline.md) · [`claims.md`](claims.md) §Hypothesis → table

**Run commands & scripts:** [`../experiments/README.md`](../experiments/README.md)

**Notebook IDs (V1, EXP-01–03):** [`10_experiment_registry.md`](10_experiment_registry.md)

---

## Science documents

| Layer | File | Purpose |
| ----- | ---- | ------- |
| RQ | `00_research_question.md` | Research question + terminology |
| Problem | `01_problem_statement.md` | Scope, assumptions, objectives |
| Hypotheses | `02_research_hypotheses.md` | RH1–RH4 (no implementation) |
| Method | `MASTER.md` | Single source of truth |
| Literature | `03_literature_gap.md`, `14_literature_record.md` | Related work |
| Signals | `04_signal_design.md`, `18_…selection.md`, `05_computation_protocol.md` | Taxonomy + math |
| Oracle | `07_oracle_definition.md` | Offline labels, \(y_{\text{opp}}\) |
| Evaluation | `08_evaluation_design.md` | Metrics, study outputs |
| Claims | `claims.md` | Hypothesis → evidence map |
| Claim audit | `claims_evidence_matrix.md` | Intro/Abstract ↔ table/figure gate |
| Milestones | `submission_milestones.md` | M1 draft → M2 C3 → M3 revision |
| Process & results | `paper_process_and_results.md` | End-to-end pipeline + locked findings |
| ACL sprint strategy | `acl_sprint_strategy.md` | C1/C2/C3 campaigns, infra, timeline |
| C3 extensions | `c3_layerwise_concepts.md`, `c3_prefill_extensions_plan.md` | Model-derived layerwise evolution (RH5) |
| Paper | `11_paper_outline.md` | Section structure |

## Engineering documents

| Layer | File | Purpose |
| ----- | ---- | ------- |
| Experiments | `10_experiment_registry.md` | V1/V2, EXP-01–03, artifacts |
| Decisions | `09_decision_register.md` | Audit log D01–D64 |
| Lab notes | `notes.md` | Ephemeral daily log |
