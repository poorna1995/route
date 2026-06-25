# Research Notes

Daily log. Important decisions also go in `09_decision_register.md`.

---

## Template

### YYYY-MM-DD — Day N

**1. Decision**

**2. Evidence**

**3. Uncertainty**

---

## Entries

### 2026-06-25 — ACL framing refined (D64)

**1. Decision**

Contribution = **signal extraction + characterization**; router = **demonstration**. Avoid headline *unsupervised routing*; three-layer supervision (unsupervised compute · oracle evaluate · CALIB calibrate policy). Null-all-signals → limits paper. Guiding principle: experiment→paragraph→hypothesis→RQ.

**2. Evidence**

Updated: `MASTER`, `00`, `01`, `claims`, `08`, `11`, `09` (D64).

**3. Uncertainty**

None on narrative; execution unchanged.

---

### 2026-06-25 — Story sharpened (D63)

**1. Decision**

Advisor transcript: **one problem, one paper** — unsupervised pre-inference signals for LLM routing. Signals are the contribution (“estimate those signals”); router is Study IV only. **Paper-first:** write prose + table shells, then run only RH1–RH4 experiments. Agents/orchestration → future work.

**2. Evidence**

Updated: `MASTER.md`, `00`, `01`, `claims.md`, `08`, `11`, `WORKFLOW.md`, `09` (D63).

**3. Uncertainty**

None on scope. Execution order unchanged: CALIB oracle → D46 → TEST.

---

### 2026-06-20 — Day 1 (M4-A)

**M4-A goal:** Verify that candidate models can provide the information required to compute the retained pre-inference signals (entropy and confidence margin). This phase evaluates **probe feasibility only** — not routing performance or signal informativeness.

**Step 0:** Rubric accepted → **D16 frozen** (`09`) — evaluation criteria only, not full methodology.

**Step 1:** Candidate register in `06` §4.1 (C1–C4). Not final pool.

**Evaluation sequence (sequential):**

```text
C1 (Llama 3.2 1B) → verify pipeline
  → if OK: C2 (Llama 3.2 3B)
  → if OK: C3 (Qwen2.5 1.5B)
  → populate §4.3 → stop (no pool, no dataset)
```

If C1 fails (script, access, load), fix before C2/C3.

**Methodology changes:** Avoid rubric edits unless evidence shows a real limitation. Any change → new row in `09`, not silent edits to `06` §2.

**Observation log**

```
(C1 — meta-llama/Llama-3.2-1B-Instruct)
  env: project .venv (Python 3.12), NOT conda base
  model_type: llama
  config.vocab_size: 128256
  logits shape: (1, 12, 128256)
  PASS: full-vocabulary logits returned
  load time: ~200s CPU (mps available: False)
  revision: main (default) — not pinned yet

(C2 — if C1 OK)
  skipped day 1 (C1 OK; C3 run for cross-family check)

(C2 — meta-llama/Llama-3.2-3B-Instruct)  [2026-06-20]
  env: project .venv; torch 2.11.0; mps=False; cuda=False
  CPU/disk offload (meta device warning)
  revision: main (default) — not pinned
  model_type: llama | vocab_size: 128256
  prompt tokens: 12
  logits shape: (1, 12, 128256)
  load time (s): 501.7 | forward time (s): 9.916 | total: 511.6
  peak rss (MB): 1853
  PASS: full-vocabulary logits returned

(C3 — Qwen/Qwen2.5-1.5B-Instruct)
  env: project .venv (Python 3.12)
  model_type: qwen2
  config.vocab_size: 151936
  logits shape: (1, 11, 151936)
  PASS: full-vocabulary logits returned
  load time: ~263s CPU (mps available: False)
  revision: main (default) — not pinned yet

(C4 — Qwen/Qwen2.5-3B-Instruct)  [2026-06-20]
  env: project .venv; torch 2.11.0; mps=False; cuda=False
  CPU/disk offload (meta device warning)
  revision: main (default) — not pinned
  model_type: qwen2 | vocab_size: 151936
  prompt tokens: 11
  logits shape: (1, 11, 151936)
  load time (s): 495.9 | forward time (s): 9.980 | total: 505.9
  peak rss (MB): 2445
  PASS: full-vocabulary logits returned
```

**M4 survey summary (all candidates):**

| ID | Model | Total (s) | Load (s) | Forward (s) | Peak (MB) | Prompt tok | Logits | Feasibility |
| -- | ----- | --------- | -------- | ----------- | --------- | ---------- | ------ | ----------- |
| C1 | Llama-3.2-1B | ~200 | — | — | — | 12 | `(1,12,128256)` | PASS |
| C2 | Llama-3.2-3B | 511.6 | 501.7 | 9.9 | 1853 | 12 | `(1,12,128256)` | PASS |
| C3 | Qwen2.5-1.5B | ~263 | — | — | — | 11 | `(1,11,151936)` | PASS |
| C4 | Qwen2.5-3B | 505.9 | 495.9 | 10.0 | 2445 | 11 | `(1,11,151936)` | PASS |

Structured copy → `06` §4.3. Supports **claim C1** (`claims.md`).

**Survey note:** Same default prompt → Llama tokenizes to 12 tokens, Qwen to 11. Compare timings within family, not across.

### 2026-06-20 — M4-B complete

**1. Decision**

M4-B survey complete. C1–C4 all **PASS** signal feasibility (small + medium, Llama + Qwen). M4 scientific objective met — probe requirements satisfied across surveyed tiers. No pool lock.

**Next milestone is not dataset selection.** Finish hypothesis-driven experimental design in `08` (requirements before names).

**2. Evidence**

C2 + C4 stdout above; §4.3 updated (4 rows).

**3. Uncertainty**

- Dev path: ~8–9 min load per 3B model on M2 8GB CPU/disk; feasible but slow for iteration.
- Prompt token count differs by family (12 vs 11) — use fixed template per family for M5 timing.
- Next: pin `--revision` for C1–C4; M4-C pool pairings in §5 (discussion only).

---

### 2026-06-20 — C2 routing opportunity assessment (pilot)

**Config (draft, not locked):** Config A — Qwen 1.5B (weak) ↔ Qwen 3B (strong) + GSM8K test (n=5, seed=42).

**Script:** `scripts/routing_opportunity_assessment.py` → `experiments/c2/qwen_gsm8k_n5.json`

**Results:**

| Bucket | Count | Rate |
| ------ | ----- | ---- |
| Opportunity (weak ✘, strong ✔) | 0 | 0% |
| Easy (both ✔) | 0 | 0% |
| Weak-only | 0 | 0% |
| Too hard (both ✘) | 5 | 100% |

**1. Decision**

C2 **not supported** for this (pool, dataset) pair on this sample. Do not lock. Try another dataset candidate (e.g. ARC-Challenge MCQ) or re-run with n=10–20 before concluding GSM8K is unsuitable.

**2. Evidence**

Both models failed all 5 GSM8K items (exact numeric match). Example: gold=109, weak=11, strong=162. ~5.5h total on M2 CPU/disk offload.

**3. Uncertainty**

- n=5 is small; GSM8K may still work with larger n or different seed.
- Parsing may miss correct answers buried in long chain-of-thought — spot-check raw outputs if re-running.
- Config B (Llama) not tested yet.

**End-of-day checklist**

---

### 2026-06-20 — Validation 2 (C3): Llama 1B + 3B signals — ARC (n=10) **COMPLETE**

**Config:** Same queries as Validation 1 — ARC-Challenge test, n=10, seed=42, protocol v1.

**Scripts:** `scripts/extract_signals.py`  
**Raw:** `experiments/M5/llama_arc_n10_weak_signals.csv`, `experiments/M5/llama_arc_n10_strong_signals.csv`

#### Strong model (3B)

| id | H | m | tokens |
| ---- | --- | --- | --- |
| arc_test_0 | 0.0147 | 1.0000 | 90 |
| arc_test_1 | 0.6289 | 0.7070 | 81 |
| arc_test_2 | 1.3047 | 0.1611 | 163 |
| arc_test_3 | 0.0918 | 0.9688 | 102 |
| arc_test_4 | 0.0427 | 0.9883 | 116 |
| arc_test_5 | 0.4629 | 0.8477 | 99 |
| arc_test_6 | 0.0044 | 1.0000 | 90 |
| arc_test_7 | 1.1719 | 0.2773 | 100 |
| arc_test_8 | 0.3984 | 0.8789 | 124 |
| arc_test_9 | 0.6836 | 0.7188 | 125 |

#### Weak vs strong (combined)

| id | bucket (V1) | weak H | strong H | weak m | strong m |
| ---- | ----------- | ------ | -------- | ------ | -------- |
| arc_test_0 | easy | 1.336 | 0.015 | 0.438 | 1.000 |
| arc_test_1 | opportunity | 0.445 | 0.629 | 0.820 | 0.707 |
| arc_test_2 | opportunity | 0.805 | 1.305 | 0.730 | 0.161 |
| arc_test_3 | opportunity | 0.731 | 0.092 | 0.609 | 0.969 |
| arc_test_4 | easy | 0.746 | 0.043 | 0.734 | 0.988 |
| arc_test_5 | opportunity | 0.443 | 0.463 | 0.844 | 0.848 |
| arc_test_6 | easy | 0.197 | 0.004 | 0.938 | 1.000 |
| arc_test_7 | opportunity | 0.887 | 1.172 | 0.777 | 0.277 |
| arc_test_8 | easy | 0.293 | 0.398 | 0.914 | 0.879 |
| arc_test_9 | easy | 1.039 | 0.684 | 0.539 | 0.719 |

**C3 checks (both models):**

| Check | Result |
| ----- | ------ |
| Computable weak + strong | **PASS** (20/20) |
| H varies per model | **PASS** |
| m varies per model | **PASS** |
| Weak vs strong differ (all 10 q) | **PASS** |
| Informativeness (ρ) | **Not tested** — pilot only |

**Note:** Strong m=1.0 on arc_test_0, arc_test_6 = saturated margin (single dominant next token). Apparatus OK.

**Decision:** Validation 2 **PASS** on n=10. C3 supported for candidate triple. **Next:** Validation 1 n≈50 → freeze in `09` if still studiable.

**Evidence:** `analysis/c3_arc_llama_n10_signals_summary.json`

---

### 2026-06-20 — Validation 2 (C3): Llama 1B weak signals — ARC (n=10)

**Config:** Same queries as Validation 1 (`llama_arc_n10.json`) — ARC-Challenge test, n=10, seed=42, protocol v1.

**Script:** `scripts/extract_signals.py` → `experiments/M5/llama_arc_n10_weak_signals.csv`

**Model:** `meta-llama/Llama-3.2-1B-Instruct` (weak). **Probe:** prefill only (no generation).

| id | H | m | tokens |
| ---- | --- | --- | --- |
| arc_test_0 | 1.3359 | 0.4375 | 90 |
| arc_test_1 | 0.4453 | 0.8203 | 81 |
| arc_test_2 | 0.8047 | 0.7305 | 163 |
| arc_test_3 | 0.7305 | 0.6094 | 102 |
| arc_test_4 | 0.7461 | 0.7344 | 116 |
| arc_test_5 | 0.4434 | 0.8438 | 99 |
| arc_test_6 | 0.1973 | 0.9375 | 90 |
| arc_test_7 | 0.8867 | 0.7773 | 100 |
| arc_test_8 | 0.2930 | 0.9141 | 124 |
| arc_test_9 | 1.0391 | 0.5391 | 125 |

**C3 checks (weak only):**

| Check | Result |
| ----- | ------ |
| Computable (10/10) | **PASS** |
| H varies across queries | **PASS** (≈0.20–1.34 nats) |
| m varies across queries | **PASS** (≈0.44–0.94) |
| Weak vs strong differ | **PASS** — see combined entry above |

**Decision:** Superseded by combined weak+strong entry above (C3 PASS).

**Evidence:** `analysis/c3_arc_llama_n10_weak_signals_summary.json`. Dev: CPU/disk offload warning (same as V1).

---

### 2026-06-20 — Validation 1 (C2): Llama + ARC-Challenge (n=10)

**Config (candidate, not locked):** Llama 3.2 1B (weak) ↔ Llama 3.2 3B (strong) + ARC-Challenge test (n=10, seed=42, max_new_tokens=64).

**Script:** `scripts/routing_opportunity_assessment.py` → `experiments/M4/routing_opportunity/llama_arc_n10.json`

**Prompt protocol:** `scripts/prompt_protocol.py` (task formatting + user-only chat template).

**Per-query:**

| id | weak | strong | bucket |
| ---- | ---- | ------ | ------ |
| arc_test_0 | ✔ | ✔ | easy |
| arc_test_1 | ✘ | ✔ | opportunity |
| arc_test_2 | ✘ | ✔ | opportunity |
| arc_test_3 | ✘ | ✔ | opportunity |
| arc_test_4 | ✔ | ✔ | easy |
| arc_test_5 | ✘ | ✔ | opportunity |
| arc_test_6 | ✔ | ✔ | easy |
| arc_test_7 | ✘ | ✔ | opportunity |
| arc_test_8 | ✔ | ✔ | easy |
| arc_test_9 | ✔ | ✔ | easy |

**Aggregate:**

| Bucket | Count | Rate |
| ------ | ----- | ---- |
| Opportunity (weak ✘, strong ✔) | 5 | 50% |
| Easy (both ✔) | 5 | 50% |
| Weak-only | 0 | 0% |
| Too hard (both ✘) | 0 | 0% |

**Validation 1 criterion (`08` §8.2):** opportunity ≥ 10% → **PASS** (50%).

**1. Decision**

Candidate triple **(Llama 1B, Llama 3B, ARC-Challenge)** is **studiable on this sample** — not frozen yet (D21: n≈50 for freeze decision + Validation 2). C2 **supported for this configuration** on n=10.

**Contrast with Qwen + ARC (same dataset, n=10):** Qwen 0% opportunity / 70% easy; Llama 50% / 50%. Pool spread matters on ARC; not a dataset-only verdict.

**2. Evidence**

Analysis: `analysis/c2_arc_llama_n10_summary.json`. Dev: CPU/disk offload (M2 8GB); weak-model generate times spiked on some queries (e.g. arc_test_3 ~399s, arc_test_5 ~456s) — note for dev path only, not paper.

**3. Uncertainty / next**

- n=10 is below D21 target (≈50) for freeze — rates may shift at scale.
- **Next:** Validation 2 (C3) on **same 10 queries** (extract H, m); then Validation 1 at n≈50 before `09` lock.

---

### 2026-06-22 — C2 Gate 1: Llama + ARC-Challenge (n=50) — **COMPLETE**

**Config (candidate):** Llama 3.2 1B (weak) ↔ Llama 3.2 3B (strong) + ARC-Challenge test (n=50, seed=42, max_new_tokens=8).

**Script:** `scripts/routing_opportunity_assessment.py` → `experiments/M4/routing_opportunity/llama_arc_n50.json`

**Environment:** Mac CPU, torch.float32 (~2–5 min/query per model).

**Aggregate:**

| Bucket | Count | Rate |
| ------ | ----- | ---- |
| Opportunity (weak ✘, strong ✔) | 12 | 24% |
| Easy (both ✔) | 21 | 42% |
| Weak-only (weak ✔, strong ✘) | 4 | 8% |
| Too hard (both ✘) | 13 | 26% |

**Validation 1 criterion (`08` §8.2):** opportunity ≥ 10% → **PASS** (24%).

**Decision:** Candidate triple remains **studiable** at n=50. Bucket mix is non-degenerate (all four buckets present). Compared to n=10: opportunity dropped 50%→24%, too_hard appeared 0%→26% — expected at larger n.

**Evidence:** `analysis/c2_arc_llama_n50_summary.json`

**Next:** `extract_signals.py` on same 50 queries (weak + strong CSVs) → `merge_and_analyze.py` → lock in `research/09` if C3 passes on n=50.

---

### 2026-06-22 — C3: Weak-model prefill signals (n=50)

**Script:** `extract_signals.py` → `experiments/M5/llama_arc_n50_weak_signals.csv`

**Config:** Llama 3.2 1B-Instruct, ARC-Challenge test, n=50, seed=42, CPU, batch_size=1.

**Result:** 50/50 extracted. H ∈ [0.12, 1.62] nats; m ∈ [0.17, 0.97]; entropy_norm ∈ [0.010, 0.138]; top5_mass ∈ [0.88, 1.00]. Current CSV schema (entropy_norm, entropy_top10, n_eff, top5_mass, vocab_size=128256).

**Evidence:** `analysis/c3_arc_llama_n50_weak_signals_summary.json`

**Next:** Strong model (`meta-llama/Llama-3.2-3B-Instruct`) on same 50 queries → `merge_and_analyze.py` with `llama_arc_n50.json`.

---

### 2026-06-22 — C3: Strong-model prefill signals + merge (n=50)

**Script:** `extract_signals.py` → `experiments/M5/llama_arc_n50_strong_signals.csv`

**Config:** Llama 3.2 3B-Instruct, ARC-Challenge test, n=50, seed=42, CPU, batch_size=1.

**Result:** 50/50 extracted. H ∈ [0.005, 1.38] nats; m ∈ [0.001, 0.999]. All 50 queries differ weak vs strong on H or m; prompt_hash aligned.

**Merge:** `merge_and_analyze.py` → `analysis/llama_arc_n50_informativeness.json`, `analysis/llama_arc_n50_merged.csv`. Preview only (n=50): weak H/m alone ~no signal (|ρ| < 0.05); best |ρ| among deltas ~0.14 (delta_entropy_top10) — not significant at n=50; EXP-01 at n=150–200 is authoritative.

**Evidence:** `analysis/c3_arc_llama_n50_strong_signals_summary.json`, `analysis/c3_arc_llama_n50_signals_summary.json`

**C3 gate:** PASS. V1 + V2 validation complete on n=50.

---

### 2026-06-20 — C2 Gate 1: Qwen + ARC-Challenge (n=10)

**Config (draft, not locked):** Config A — Qwen 1.5B (weak) ↔ Qwen 3B (strong) + ARC-Challenge test (n=10, seed=42, max_new_tokens=64).

**Script:** `scripts/routing_opportunity_assessment.py` → `experiments/M4/routing_opportunity/qwen_arc_n10.json`

**Prompt protocol:** `scripts/prompt_protocol.py` (task formatting + user-only chat template).

**Per-query:**

| id | weak | strong | bucket |
| ---- | ---- | ------ | ------ |
| arc_test_0 | ✔ | ✔ | easy |
| arc_test_1 | ✔ | ✔ | easy |
| arc_test_2 | ✘ | ✘ | too_hard |
| arc_test_3 | ✘ | ✘ | too_hard |
| arc_test_4 | ✔ | ✔ | easy |
| arc_test_5 | ✔ | ✘ | weak_only |
| arc_test_6 | ✔ | ✔ | easy |
| arc_test_7 | ✔ | ✔ | easy |
| arc_test_8 | ✔ | ✔ | easy |
| arc_test_9 | ✔ | ✔ | easy |

**Aggregate:**

| Bucket | Count | Rate |
| ------ | ----- | ---- |
| Opportunity (weak ✘, strong ✔) | 0 | 0% |
| Easy (both ✔) | 7 | 70% |
| Weak-only (weak ✔, strong ✘) | 1 | 10% |
| Too hard (both ✘) | 2 | 20% |

**Gate 1 criterion (`08` §8.2):** opportunity ≥ 10% → **FAIL** (0%).

**1. Decision**

C2 **not supported** for the **(Qwen 1.5B, Qwen 3B, ARC-Challenge)** configuration on this sample — not a verdict on ARC alone. Do not lock. **Next:** Experiment 3 — Llama 1B ↔ 3B on ARC (one variable changed: pool). Gate 2 on Qwen+ARC queries optional for **C3 apparatus only** (computable H, m — not setup selection).

**Scientific interpretation (two failure modes):**

| Config | Dominant bucket | Failure type | Meaning |
| ------ | --------------- | ------------ | ------- |
| Qwen + GSM8K (n=5, old protocol) | 100% too_hard | **Too difficult** | Both models fail — gap not exploitable |
| Qwen + ARC (n=10) | 70% easy | **Too easy (for this pair)** | Weak already succeeds — little need to route to strong |

Same 0% opportunity, **different mechanisms**. ARC may be easy **for Qwen 1.5B↔3B**, not necessarily easy in general.

**Correct claim:** *(pool, dataset, protocol)* offers limited routing opportunity — not "ARC is unsuitable."

**2. Evidence**

Analysis: `analysis/c2_arc_qwen_summary.json`. Notable: arc_test_5 — weak ✔ (B), strong ✘ (A), gold B (weak-only). Both-wrong on arc_test_2, arc_test_3. ~6 min weak + ~6 min strong model load/generate on CPU/disk offload.

**3. Uncertainty**

- n=10 is smoke size; larger n may surface opportunity queries.
- 70% easy suggests weak model already strong on this MCQ slice — pool gap may be too narrow.
- Config B (Llama 1B ↔ 3B) on ARC not tested yet.

---

### 2026-06-20 — C2 Gate 1: Qwen + GSM8K re-val (n=10, frozen protocol)

**Config (draft, not locked):** Config A — Qwen 1.5B (weak) ↔ Qwen 3B (strong) + GSM8K test (n=10, seed=42, max_new_tokens=64).

**Script:** `scripts/routing_opportunity_assessment.py` → `experiments/M4/routing_opportunity/qwen_gsm8k_n10.json`

**Prompt protocol:** `scripts/prompt_protocol.py` (`Answer using only one number.` + user-only chat template).

**Per-query:** all 10 → too_hard (weak ✘, strong ✘).

**Aggregate:**

| Bucket | Count | Rate |
| ------ | ----- | ---- |
| Opportunity | 0 | 0% |
| Easy | 0 | 0% |
| Weak-only | 0 | 0% |
| Too hard | 10 | 100% |

**Gate 1:** FAIL (0% opportunity). **Failure mode:** too hard (same as n=5 — not fixed by protocol change).

**Evidence:** `analysis/c2_gsm8k_qwen_n10_summary.json`. Example gsm8k_test_0: gold=109, weak=84, strong=88. ~8 min weak + ~8 min strong generate phase on CPU/disk offload.

**Decision:** *(Qwen, Qwen, GSM8K)* not viable for routing-opportunity setup validation. Do not use GSM8K for this pool in pilot. **Next remains Exp 3:** Llama + ARC.

**End-of-day checklist**

- [x] D16 dated in `09`
- [x] §4.1 candidate register filled
- [x] §4.3 has C1 + C3 rows (M4-A ≈ complete)
- [x] Signal feasibility separated from reproducibility in `06` §4.2
- [x] No §5 pool pairing; no dataset lock

**Dev environment:** Apple M2, 8 GB RAM — prototyping only (`06` §2c). Use project `.venv`.

**1. Decision**

**M4-A ≈ complete.** Signal feasibility PASS for Llama (C1) and Qwen (C3). The probe computation protocol was successfully executed on multiple open-weight LLM families (Llama and Qwen), indicating that the proposed signal computation is not tied to a single model architecture. Reproducibility (revision pin) is **pending**, not a blocker for M4-B.

No pool or family lock. Conda failure is engineering-only; not paper evidence.

**M4 research question (not model selection):** Which candidate models satisfy the requirements needed to compute our proposed pre-inference signals?

**End-of-M4 deliverable:** models that pass/fail and why; candidate pools worth evaluating — not "Llama is best."

**What M4 proves:** full-vocab logits at prefill → entropy + margin computable.

**What M4 does not prove:** signal informativeness, routing, cost–quality improvement.

**2. Evidence**

C1 + C3 `verify_logprobs.py` PASS in project `.venv`; §4.3 updated.

**3. Uncertainty**

- MPS not used in venv runs (CPU offload; slow but works).
- Final pilot/oracle hardware still undecided.
- Family / capability spread / oracle practicality — after M4-B survey.

---

## Project progress (2026-06-20)

| Stage                        | Status                                            |
| ---------------------------- | ------------------------------------------------- |
| Research question            | ✅ Complete                                       |
| Problem formulation          | ✅ Complete                                       |
| Literature review (Tier-1)   | ✅ Complete                                       |
| Hypotheses                   | ✅ Complete (draft/refined)                       |
| Signal taxonomy              | ✅ Complete (pilot scope)                         |
| Probe requirements           | ✅ Complete                                       |
| Probe feasibility (M4-A)     | ✅ Complete                                       |
| Candidate survey (M4-B)      | ✅ Complete (C1–C4 PASS)                          |
| Pool selection               | ⏳ M4-C — configurations with dataset placeholder |
| Experimental design          | 🔄 In progress (`08`)                             |
| Dataset requirements         | ⏳ In `08` §5 — **not names yet**                 |
| Signal characterization (M6) | ⏳ After pool + dataset lock                      |
| Routing experiments (M7)     | ⏳ Later                                          |

**Evidence ladder:**

```text
RQ → Signal taxonomy → Can signals be computed? (M4) → YES
  → Do signals contain information? (M6 / RH2, RH3)
  → Can routing exploit that? (M7)
  → Cost–quality improvement?
```

**Working balance:** ~80% experiments, ~20% documentation (evidence-driven updates only).

---

## M4 plan

| Phase    | Goal                                     | Stop when                  |
| -------- | ---------------------------------------- | -------------------------- |
| **M4-A** | Signal feasibility across ≥1 family      | ≈ **done** (C1 + C3 PASS)  |
| **M4-B** | Candidate survey (small + medium)        | ✅ C1–C4 PASS              |
| **M4-C** | Candidate configurations (§5)            | Pool + dataset placeholder |
| **M4-D** | Routing opportunity + signal feasibility | Lock in `09`               |
| **M5**   | Signal extraction pipeline               | H, m → CSV                 |

**M4-B sequence:**

```text
C2 (Llama 3.2 3B)
C4 (Qwen 3B)
→ pin --revision for PASS models
→ §5 pool discussion (not lock)
```

If C1–C4 all PASS, M4's scientific objective is met (small → medium covered). A 7–8B attempt is **optional for pool construction**, not required to prove probe feasibility.

**Timing protocol:** use the same default prompt for all survey runs; record `prompt tokens`, `total time`, `peak rss` from script output.

**M5 (after lock):** Signal extraction pipeline — query → entropy, margin → CSV. Test RH2/RH3 on pilot (M6).

**Discipline rule:** _Which claim does this support?_ (`claims.md`) — if none, do not run.

**Execution plan (ordered):**

| #   | Task                                                              | Claim supported      | Outcome                     |
| --- | ----------------------------------------------------------------- | -------------------- | --------------------------- |
| 1   | Pin model revisions (C1–C4)                                       | C1 (reproducibility) | Revisions locked            |
| 2   | Finish `08` §3–§7                                                 | All                  | Experimental design frozen  |
| 3   | Candidate configurations + purpose (`06` §5.1)                    | C2 prep              | Pool + dataset + hypotheses |
| 4   | Shortlist datasets vs R1–R5                                       | C2 prep              | Dataset candidates          |
| 5   | **Routing opportunity assessment** (10–20 q, full gen, no probes) | **C2**               | Opportunity / bucket rates  |
| 6   | **Signal feasibility check** (H, m: computable? vary? differ?)    | **C3**               | Sanity — not correlation    |
| 7   | Freeze configuration (`09`)                                       | —                    | Setup locked                |
| 8   | Signal extraction pipeline (M5)                                   | C3 at scale          | CSV                         |
| 9   | Pilot + analysis (M6)                                             | **C4, C5**           | ρ, complementarity          |
| 10  | Routing (M7)                                                      | **C6**               | Cost–quality                |

**Roadmap:** see `claims.md` (frozen).
