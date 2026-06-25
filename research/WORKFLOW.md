# Final Research Workflow (ACL Version)

> **Execution layer** — how to run the frozen study.  
> **Science SSOT:** [`MASTER.md`](MASTER.md) · **Registry:** [`10_experiment_registry.md`](10_experiment_registry.md) · **CLI:** [`../experiments/README.md`](../experiments/README.md)

This document is frozen (D62, D63, D64). Remaining work is **paper-first writing**, then **hypothesis-driven execution** only.

---

## Paper-first rule (D63/D64)

```text
RQ → Hypotheses → Table shells → Experiments → Fill tables
```

**Guiding principle:** Every experiment → paragraph → hypothesis → RQ.

See `11_paper_outline.md` and `claims.md` §Hypothesis → table.

---

## Research question

> **RQ:** How informative are **unsupervised pre-inference routing signals** for LLM selection before full model generation?

> **Scientific core:** How much **routing-relevant information** is present in inexpensive pre-inference signals?

Everything else supports this question.

---

## Scientific hierarchy

```text
Research Question (routing-relevant information in pre-inference signals)
        │
        ▼
Unsupervised signal extraction  →  feature vector [c(q), H, m]
        │
        ├──────────────┬──────────────┐
        ▼              ▼
Model-independent   Model-dependent
     c(q)             H(q), m(q)
        │              │
        └──────┬───────┘
               ▼
      Characterization (Studies I & II)     ← primary
               │
               ▼
      Complementarity (Study III)
               │
               ▼
     Routing demonstration (Study IV)        ← exploit what exists
               │
               ▼
      Generalization (MMLU)
```

The paper is **not** “we built a router” or “unsupervised routing.” It is **unsupervised pre-inference signal extraction and characterization**; Study IV is a **demonstration** that exploits whatever information exists (~20% of paper).

**Terminology:** Avoid headline *unsupervised routing* — Study IV uses CALIB-supervised calibration (D64).

---

## Phase 0 — Freeze

Freeze: research question, hypotheses, datasets, prompt protocol, models, metrics, signal taxonomy. **No changes afterwards.**

---

## Phase 1 — Dataset policy

| Dataset | Calibration | Evaluation |
| ------- | ----------- | ---------- |
| ARC     | validation  | test       |
| MMLU    | dev         | test       |
| GSM8K   | train       | test       |
| BoolQ   | train       | validation |

```bash
.venv/bin/python scripts/run.py splits --dataset arc_challenge --output analysis/splits.json
```

---

## Phase 2 — Prompt protocol

Every branch uses the same path:

```text
dataset row → task formatting → user_content → chat template → chat_prompt
```

Same prompt for tokenizer features, probe extraction, and oracle generation.

---

## Phase 3 — Signal extraction

| Branch | Path | Generation? |
| ------ | ---- | ----------- |
| **A — Model-independent** | `chat_prompt` → tokenizer → candidate \(c(q)\) features | No |
| **B — Model-dependent** | `chat_prompt` → prefill → \(H, m\) | No answer generation |
| **C — Oracle** | `chat_prompt` → `generate()` → `weak_ok`, `strong_ok`, `bucket`, `y_opp` | Labels only |

---

## Phase 4 — D46 (pre-study calibration)

**Not a paper experiment.** One-time calibration to select representative \(c(q)\).

| Item | Value |
| ---- | ----- |
| Purpose | Pick one \(c(q)\) from word count, MATTR, Shannon, compression |
| Data | ARC **validation** only (~299) |
| Metrics | Spearman, AUROC vs `y_opp` |
| Output | `analysis/selected_feature.json` |
| Rule | Run **once**; never repeat without new decision in `09` |

```bash
.venv/bin/python scripts/run.py features --dataset arc_challenge --split-role calib \
  --splits-json analysis/splits.json --output experiments/M5/arc_validation_features.csv
.venv/bin/python scripts/run.py oracle ... --split-role calib ... \
  --output experiments/M4/arc_validation_oracle.json
.venv/bin/python scripts/run.py screen \
  --features experiments/M5/arc_validation_features.csv \
  --oracle experiments/M4/arc_validation_oracle.json \
  --splits-json analysis/splits.json \
  --output analysis/d46_signal_screen_arc.json
```

---

## Phase 5 — Master dataset

Extract oracle, weak probes, strong probes, and frozen \(c(q)\) on CALIB + TEST. Merge:

```text
query_id | bucket | weak_ok | strong_ok | y_opp | c_q | H_w | m_w | ...
```

```bash
.venv/bin/python scripts/run.py merge \
  --weak-csv ... --strong-csv ... --oracle ... \
  --features-csv ... --complexity-selection analysis/selected_feature.json \
  --output analysis/arc_routing_relevance.json \
  --merged-csv analysis/arc_merged.csv
```

---

## Studies (paper experiments)

### Study I (RH1) — model-independent

Does \(c(q)\) contain routing information? Metrics: Spearman, AUROC, AUPRC, distributions.  
**CLI:** `merge` → EXP-01 (part 1)

### Study II (RH2) — model-dependent

Do \(H, m\) contain routing information? Same metrics. Strong probes exploratory.  
**CLI:** `merge` → EXP-01 (part 2)

### Study III (RH3) — complementarity (centerpiece)

Do the two families measure different information? Ladder: \(c \to H \to m \to c{+}H \to c{+}m \to c{+}H{+}m\).

**Primary endpoint:** ΔAUROC \(c \to c{+}H{+}m\) on TEST.

Supporting: bootstrap CI, permutation, DeLong, confound regression, calibration, stability.

**CLI:** `complementarity --splits-json analysis/splits.json` → EXP-02

> Study III fits logistic models for **characterization** (AUROC). These artifacts are **not** reused for Study IV.

### Study IV (RH4) — utility validation

**Hold-out routing evaluation** — separate from Study III.

```text
Fit router on CALIB  →  tune threshold τ on CALIB  →  freeze  →  evaluate on TEST
```

Compare on TEST: always-weak, always-strong, oracle upper bound, learned router.

Metrics: accuracy, average cost, cost–quality tradeoff.

**CLI:** `route-eval --splits-json analysis/splits.json` → EXP-03

`route-preview` is **D37 sanity only** (median heuristics, in-sample) — not the paper experiment.

### Generalization — MMLU

Same protocol, models, and frozen \(c(q)\). **Do not repeat D46.** One transfer table.

---

## Nested evaluation rule

| Stage | Split | Uses routing labels? |
| ----- | ----- | -------------------- |
| D46 calibration | CALIB | Yes (screening only) |
| Study III logistic fit | CALIB | Yes (AUROC characterization) |
| Study IV router fit + τ | CALIB | Yes (routing utility) |
| **All reported TEST metrics** | **TEST** | **Evaluate only** |

---

## Writing order

**Now:** Introduction, Related Work, Method, Experimental Setup.

**As results arrive:** Study I → II → III → IV → Generalization.

**Last:** Discussion, Conclusion.

---

## Execution schedule

| Week | Tasks |
| ---- | ----- |
| **1** | Splits, CALIB extraction, D46, freeze `selected_feature.json` |
| **2** | Full ARC TEST extraction, merge, doctor |
| **3** | Studies I–III, figures, fill Results §5 |
| **4** | Study IV, MMLU, Discussion, final edit |

---

## What not to do

Do **not** redesign experiments, add signals, change hypotheses, add datasets, invent new routers, refactor code, or rerun D46.

---

## Reviewer narrative (one paragraph)

> We are **not proposing a new router**. We ask what information about routing exists **before expensive model generation**. We organize unsupervised routing signals into **model-independent** and **model-dependent** families, characterize each, test complementarity, and examine whether this understanding supports a **simple lightweight routing policy** evaluated with proper hold-out calibration.
