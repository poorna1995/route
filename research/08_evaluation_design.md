# Evaluation Design

> **Scientific decision this document supports:** How will each hypothesis be tested?
>
> **Principle (D63/D64):** RQ → hypotheses → tables → experiments → fill tables. **Guiding principle:** every experiment → paragraph → hypothesis → RQ.
>
> **Contribution order:** Characterize **informativeness of unsupervised routing signals** before evaluating a routing policy. Taxonomy 𝒮 organizes what is measured; routing (M7, C6) is the **utility** component — not the first.
>
> **Operating model (D21):** Validation identifies one **studiable** `(pool, dataset, protocol)` — not the best router or benchmark. See `claims` §Operating model.
>
> **Milestone M5** — Metrics, requirements, experiment matrix **before** dataset names or downloads.
> **Freeze rule:** Methodology and metrics locked in `MASTER.md`, `05`, `07`, `08` §4. Reopen only on genuine bugs — log in `09`.
> Hypotheses: `02` · Oracle: `07` · Frozen config: `MASTER` · Registry: `10` · Claims: `claims.md`

---

## 1. Where we are

```text
✓ Research question, problem, hypotheses — frozen (D55, D63 story sharpen)
✓ Signal taxonomy: c(q), H, m (D15/D56)
✓ Probe feasibility (M4 — C1–C4 PASS)
✓ V2 configuration lock — Llama 1B/3B + ARC (D33)
→ Paper-first: draft Intro/Method/Setup + table shells (11_paper_outline.md)
→ CALIB oracle complete + D46 freeze c(q)
→ TEST pipeline: EXP-01 (Studies I–II) → EXP-02 (III) → EXP-03 (IV)
→ MMLU generalization table (after ARC interpretation)
```

**Publication strategy (D63/D64):**

```text
Research question → Hypotheses → Table shells → Experiments → Fill tables → Discussion (outcome scenario)
```

**Guiding principle:** Every experiment must justify a paragraph; every paragraph must justify a hypothesis; every hypothesis must answer the RQ.

**Outcome branching:** If all signals null on corrected TEST, write limits-paper emphasis (`MASTER.md` §3b, `claims.md` §Outcome scenarios)—do not add rescue experiments.

**Scope discipline:** One problem = unsupervised pre-inference signals for LLM routing. Agent routing, task decomposition, graph routing, supervised neural routers → future work only.

**Dependency graph:** Pool and dataset are **siblings** — both depend on experimental design; neither comes before the other. They meet at routing opportunity assessment + signal feasibility check.

```text
Research Question → Signal Taxonomy → Experimental Design (this doc)
        ├──────────────────┬──────────────────┐
        ▼                  ▼                  │
   Model survey      Dataset requirements      │
   (06 §4)            (§5–§6)                  │
        └──────────┬─────────┘                  │
                   ▼                            │
        Candidate configurations (pool + dataset placeholder)
                   ▼
        Phase A — smoke validation (~10 q: apparatus + signal extraction)
                   ▼
        Phase B — configuration validation (~50 q: outcome distribution)
                   ▼
        Lock pool + dataset (09)
                   ▼
        Phase C — main study (ARC) → Studies I–IV
        Phase D — generalization (optional second dataset)
```

You are **not blocked** on methodology. Remaining work is **paper-first prose** and **hypothesis-driven execution** (CALIB → D46 → TEST).

---

## 1b. Paper-first workflow (D63)

| Step | Action | Output |
| ---- | ------ | ------ |
| 1 | Draft paper sections 1–4 + empty Results | `paper/draft/*.tex` |
| 2 | Lock RH1–RH4 + table shells | T1–T4 in `paper/tables/` |
| 3 | Complete CALIB oracle + D46 screen | `selected_feature.json`, D46 in `09` |
| 4 | TEST: oracle, probes, merge | EXP-01 JSON/CSVs |
| 5 | EXP-02 complementarity, EXP-03 routing | T3, T4 filled |
| 6 | MMLU transfer (no D46 repeat) | Generalization paragraph |

**Rule:** No new experiment unless it maps to RH1–RH4 and a named table.

---

## 2. Main-study signal scope (D15, D56 — frozen)

| Signal                  | Hypothesis   | In main study?        |
| ----------------------- | ------------ | --------------------- |
| Representative \(c(q)\) | H1 → RH1     | **Yes** (D46 pending) |
| Token entropy           | H2 → RH2     | **Yes**               |
| Log-probability margin  | H3 → RH2–RH3 | **Yes**               |
| Paraphrase stability    | H4           | No (D04)              |

Studies I–IV test RH1–RH4. See `10_experiment_registry.md`.

---

## 3. Hypothesis → evidence needed

Before naming datasets, specify what each claim requires.

| Hypothesis | Claim (summary)                                      | Evidence required                             | Main study?         |
| ---------- | ---------------------------------------------------- | --------------------------------------------- | ------------------- |
| **RH1**    | Representative **query-derived** signals informative | ρ, AUROC for \(c(q)\) vs \(y\_{\text{opp}}\)  | **Yes** (Study I)   |
| **RH2**    | Model-dependent probes informative                   | ρ, AUROC for \(H, m\)                         | **Yes** (Study II)  |
| **RH3**    | Families complementary                               | ΔAUROC across families                        | **Yes** (Study III) |
| **RH4**    | Simple policy improves cost–quality                  | Accuracy + cost vs baselines                  | **Yes** (Study IV)  |
| **H2**     | Higher entropy → weak model less suitable            | Spearman ρ(entropy, oracle gap) on weak model | **Yes**             |
| **H3**     | Lower margin → less reliable                         | Spearman ρ(margin, correctness / gap)         | **Yes**             |
| **H1**     | Complexity predicts difficulty                       | Objective labels + difficulty spread          | Defer               |
| **H4**     | Paraphrase stability reflects robustness             | Paraphrases that preserve answer semantics    | Defer               |

**Litmus test for any dataset:** _If I remove it, which hypothesis becomes impossible to support?_ If none → it does not belong in the paper.

---

## 4. Evaluation metrics

Define before implementation. Ground truth is **for evaluation only** — never for computing signals.

| Phase                   | Question                             | Primary metrics                                                                 | Paper slot |
| ----------------------- | ------------------------------------ | ------------------------------------------------------------------------------- | ---------- |
| **Availability**        | Can signals be computed?             | Success rate; probe tokens; wall time                                           | T1         |
| **Informativeness**     | Do signals relate to oracle?         | Spearman ρ(signal, quality gap); **AUROC** (weak fail / strong success)         | T2, F1     |
| **Complementarity**     | Do H and m add distinct information? | Pairwise ρ; incremental gain                                                    | T3         |
| **Routing opportunity** | Is the setting studiable?            | % easy / weak-only / strong-only / mixed (Phase B + pilot)                      | T2 / setup |
| **Routing**             | Does a simple policy help?           | Accuracy, avg cost, Δ vs always-small/large                                     | T4, F3     |
| **Overhead**            | Is probe cost acceptable? (A2)       | m prefills vs m full generations (analytical + measured); tokens saved vs spent | T5         |

**v1 metric discipline (D29):** Spearman, AUROC, routing opportunity, cost — **only**. No ECE/AUPRC/MI headline stack. Optional: fraction of routing-relevant variation captured — **one result**, not the RQ.

**Oracle gap (informative phase):** per query $q$, $\texttt{oracle\_gap} = y(q, M_{\text{strong}}) - y(q, M_{\text{weak}}) \in \{-1,0,1\}$ (`07`). **Routing opportunity:** $y^{\text{opp}}=1$ iff weak ✘ and strong ✔.

### EXP-01 outputs (frozen — D34, D35)

| #   | Output                                      | Metrics / artifact                                                                                                                                                                         |
| --- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | **Signal distributions (all four buckets)** | Violin/box plots + summary stats for $H_w$, $m_w$, $H_s$, $m_s$ by easy / opportunity / weak-only / too-hard — **primary characterization** (`plot_signal_distributions.py` → F1)          |
| 1b  | **Pairwise bucket separability**            | Mann–Whitney + Cliff's $\delta$, Cohen's $d$, rank-biserial: opportunity vs easy, vs weak-only, vs too-hard — per headline signal (`merge_and_analyze.py` → `pairwise_bucket_comparisons`) |
| 2   | **Correlation + bootstrap CI**              | Spearman $\rho$ + 95% bootstrap CI (headline) vs $y^{\text{opp}}$ and vs `oracle_gap`; Kendall $\tau$, Pearson $r$ secondary                                                               |
| 3   | **AUROC + AUPRC + bootstrap CI**            | Ranking opportunity vs non-opportunity — interpret with CI and effect sizes; **no universal cutoff**                                                                                       |
| 4   | **Effect size (legacy headline)**           | `opportunity_vs_easy` subset of pairwise comparisons (back-compat alias)                                                                                                                   |
| 5   | **Probe cost (T5)**                         | always-weak / always-strong / m prefills / probe+routed 1 gen / oracle m gens                                                                                                              |
| 6   | **ROC curves (F2)**                         | Per-signal ROC vs opportunity (`plot_roc_curves.py`)                                                                                                                                       |
| 7   | **Entropy–margin scatter (F3)**             | $H_w$ vs $m_w$ colored by bucket (`plot_signal_scatter.py`)                                                                                                                                |

**Reporting order (paper §5):** distributions (all buckets) → pairwise separability → ρ + CI → AUROC/AUPRC + CI + ROC → entropy–margin scatter → EXP-02 complementarity → probe cost.

**EXP-02 / Study III** (`complementarity`): **Primary RH3** — logistic ladder `c(q) → c+H → c+H+m` with ΔAUROC + bootstrap CI on TEST. **Secondary** — `H → H+m` (within-family; appendix). Study III logistic models characterize predictive gain only — **not** reused for EXP-03.

**EXP-03 / Study IV** (`route-eval`): **Independent** CALIB-fit logistic \(P(y^{\text{opp}} \mid c, H_w, m_w)\); tune τ on CALIB; report accuracy + cost on TEST vs always-weak / always-strong / oracle. `route-preview` = D37 sanity only.

**Discipline:** EXP-01 characterizes signals only — **no routing thresholds**. Median-heuristic routing (`route-preview`) is **sanity-check only** (D37), not a paper experiment.

**Robustness (appendix):** bootstrap CIs on TEST; optional 5 random query subsets (re-analysis only, no re-inference).

**MMLU (§5.5):** one transfer table — ρ(c), ρ(H), ρ(m), ΔAUC — not full Studies I–III rerun.

**Scripts:** `run.py merge`, `complementarity`, `route-eval`, `plot`.

---

## 5. Dataset requirements (no names yet)

Datasets must satisfy these **before** shortlisting. Requirements are hypothesis-driven, not benchmark-driven.

| ID     | Requirement                         | Why                                                                                   |
| ------ | ----------------------------------- | ------------------------------------------------------------------------------------- |
| **R1** | **Objective evaluation**            | Automatic correctness (exact match, MCQ letter, pass@k) — no subjective judging in v1 |
| **R2** | **Difficulty variation**            | Spread of easy → hard; all-easy or all-impossible → no routing opportunity (A3)       |
| **R3** | **Stable ground truth**             | Reliable reference answers for oracle labels only                                     |
| **R4** | **Compatible with all pool models** | Same prompt template; models can attempt the task                                     |
| **R5** | **Manageable probe cost**           | Full ARC TEST + oracle generations feasible on deployment path                        |

**Not required for v1:** 8–10 benchmarks; subjective open-ended grading; datasets that only support hypotheses already deferred.

---

## 6. Task families (v1 paper — 2–3 max)

Choose **families** that satisfy R1–R5 and map to hypotheses. Names come **after** requirements.

| Task family            | Supports         | Why consider                                                  |
| ---------------------- | ---------------- | ------------------------------------------------------------- |
| Knowledge / MCQ        | RH2, H2, H3      | Clean objective labels; difficulty varies by item             |
| Mathematical reasoning | RH2, H2, H3, RH4 | Exact answers; strong weak/strong gap potential               |
| Open-domain QA         | RH2, RH4         | Generation beyond MCQ; harder eval — use only if R1 satisfied |

**Target:** 2–3 families, not a benchmark suite. Each family must support at least one pilot hypothesis.

---

## 7. Experiment matrix

Every run exists to support a claim or hypothesis. **Canonical registry:** `10_experiment_registry.md`.

| ID         | Type                    | Claim  | Hypothesis       | Metric                          |
| ---------- | ----------------------- | ------ | ---------------- | ------------------------------- |
| **V1**     | Validation              | C1, C3 | —                | Extraction success; apparatus   |
| **V2**     | Validation              | C2     | — (setup gate)   | Outcome bucket distribution     |
| **EXP-01** | Signal characterization | C4     | RH1, RH2 · H1–H3 | Spearman ρ; AUROC; effect sizes |
| **EXP-02** | Complementarity         | C5     | RH3              | ΔAUROC; incremental gain        |
| **EXP-03** | Routing evaluation      | C6     | RH4 · O3         | Accuracy, cost                  |

**Deferred (no EXP ID):** H4 / paraphrase (D04).

**Do not** compute Spearman ρ during validation — that belongs to **EXP-01** / **EXP-02** (main characterization, n=150–200).

**M4 complete:** signal availability on pool candidates (`06` §4.3).  
**Next:** candidate configurations + dataset shortlist → routing opportunity assessment → signal feasibility check → **then** signal extraction pipeline (M5).

---

## 8. Candidate experimental configurations

Pool and dataset are designed **together**, tied to hypotheses — not pool-first, **not literature-first**.

Define **Phase B candidates** (placeholders until validation). Validation **chooses one** studiable triple — popularity and prior citations do not.

| ID    | Pool (weak ↔ strong) | Dataset (candidate) | Purpose (methodology)                                     | Phase B status                  |
| ----- | -------------------- | ------------------- | --------------------------------------------------------- | ------------------------------- |
| **A** | Llama 1B ↔ 3B        | ARC-Challenge       | MCQ + letter eval; low protocol risk; setup validation    | Smoke promising → **n≈50 next** |
| **B** | Llama 1B ↔ 3B        | MMLU (1–2 subjects) | Same MCQ protocol; knowledge breadth; fallback or Phase D | Not run                         |
| **C** | Qwen 1.5B ↔ 3B       | ARC-Challenge       | Controlled pool lever (same dataset as A)                 | Smoke done — mostly easy        |

**Model pattern (from literature, not model IDs):** same-family, two capability tiers — justified by Hybrid LLM, Plaut; implemented as Llama 1B↔3B or Qwen 1.5B↔3B on available hardware. Do **not** reproduce GPT-4 / 70B pools.

Record in `06` §5. **No lock** until Phase A + Phase B pass on the chosen candidate.

Pool-only comparison (`06` §5 matrix) scores Must criteria; **routing opportunity** is measured only after pairing with a dataset candidate.

---

## 8. Validation experiments (before main study)

> **Three phases — three questions (D21, D22):**
>
> | Phase | Name                        | Claim             | Question                                                     |
> | ----- | --------------------------- | ----------------- | ------------------------------------------------------------ |
> | **A** | Smoke validation            | C3 (C1 apparatus) | Can I run the experiment?                                    |
> | **B** | Configuration validation    | C2                | Is this `(pool, dataset, protocol)` scientifically suitable? |
> | **C** | Hypothesis evaluation       | C4–C6             | Do signals support the research claims?                      |
> | **D** | Generalization (post-pilot) | C4–C6 extend      | Do findings hold on a second dataset?                        |
>
> Fail Phase B → do not run main study (EXP-01–03) on that configuration.
>
> **Sample sizes:** ~10 for Phase A (smoke/debug); **≈50** for Phase B (lock/no-lock decision — not a magic number, minimum evidence for configuration suitability); 150–200 for Phase C after lock.

| Phase                            | Legacy name             | Question                                                                                        | Claim | If NO                                                   |
| -------------------------------- | ----------------------- | ----------------------------------------------------------------------------------------------- | ----- | ------------------------------------------------------- |
| **A — Smoke validation**         | V1 (was “Validation 2”) | Can signals be **extracted** and do they **behave sensibly**? Can the apparatus run end-to-end? | C3    | Fix methodology before scaling                          |
| **B — Configuration validation** | V2 (was “Validation 1”) | Is this **candidate** (pool, dataset, protocol) **studiable** — sufficient outcome diversity?   | C2    | Try next candidate — change one lever (pool or dataset) |

Phase A typically runs first at ~10 queries. Phase B uses full generation on **≈50 queries** (same seed) when Phase A looks clean. Same queries can serve both when Phase B runs; they answer **different research questions**. Phase B uses full generation only (§8.2). Phase A uses prefill probes only (§8.3). **Both use the same prompt protocol** (§8.0).

---

## 8.0 Prompt protocol (shared by Phase A and Phase B)

Implementation: `scripts/prompt_protocol.py`. Full spec: `05` §1.

```text
Dataset item → task formatting (user_content) → chat template → P(u)
                      │                                    │
                      │                                    ├─ prefill forward pass (Phase A — C3)
                      │                                    └─ model.generate() (Phase B — C2 oracle only)
```

Phase A and Phase B must use **identical** `user_content` and chat wrapping so signals and correctness labels refer to the same query formulation. Lock in `09` after Phase A + Phase B pass.

---

## 8.1 Dataset shortlist (after §5–§7 frozen)

Shortlist 2–3 datasets against R1–R5 and the litmus test (§3). **Not a commitment** — candidates for routing opportunity assessment and signal feasibility check. See §9 survey table.

---

## 8.2 Phase B — Configuration validation (C2)

> **Research question:** Is this **candidate configuration** suitable for testing hypotheses about signal informativeness?
>
> Not: is it optimal? Is it the best benchmark? Simply: is the outcome **distribution** non-degenerate enough to study?

Run weak + strong models to completion on **≈50 queries** (seed fixed) for lock decisions; fewer only for smoke/debug (Phase A may include a preliminary C2 peek, but that does not substitute for Phase B). **No entropy, margin, or other probe signals.**

| Query | Weak correct? | Strong correct? | Outcome                                   |
| ----- | ------------- | --------------- | ----------------------------------------- |
| q1    |               |                 | Easy / Opportunity / Weak-only / Too hard |
| …     |               |                 |                                           |

**Outcome buckets:**

| Weak | Strong | Label                   | Research meaning                   |
| ---- | ------ | ----------------------- | ---------------------------------- |
| ✔    | ✔      | Easy                    | No routing needed                  |
| ✘    | ✔      | **Routing opportunity** | Strong adds value over weak        |
| ✔    | ✘      | Weak-only               | Weak uniquely succeeds (note rate) |
| ✘    | ✘      | Too hard                | Neither model solves               |

**Aggregate metrics:**

| Metric            | Use                                                       |
| ----------------- | --------------------------------------------------------- |
| Opportunity rate  | % weak ✘, strong ✔ — descriptive, not sole pass criterion |
| Weak-only rate    | % weak ✔, strong ✘                                        |
| Strong-only rate  | Same as opportunity rate                                  |
| Both-correct rate | High (~95%) → dataset too easy for this pool              |
| Both-wrong rate   | High (~80%) → dataset too hard for this pool              |

**If opportunity low → diagnose failure mode first** (too easy vs too hard), then change **one** lever: pool **or** dataset — not both at once (`claims` controlled sequence).

**Phase B — proceed criteria (D22):**

> **Proceed if the observed distribution contains sufficient diversity to study the relationship between pre-inference signals and model suitability.**

Look for a **pattern**, not a single rate:

| Signal in distribution                               | Interpretation                            |
| ---------------------------------------------------- | ----------------------------------------- |
| Opportunity present, not vanishing, not overwhelming | Routing is scientifically meaningful here |
| Some easy cases                                      | Weak model is not useless; sanity check   |
| Some hard cases, but not dominant                    | Signals have something to predict         |
| No obvious pathology                                 | Eval/parsing bugs ruled out               |

**Operational guidelines** (internal proceed/stop — not paper claims):

| Heuristic         | Promising         | Reject / swap              |
| ----------------- | ----------------- | -------------------------- |
| Opportunity rate  | ≥ ~10% of queries | 0% on smoke set            |
| Both-correct rate | Not ~95%+         | Too easy **for this pool** |
| Both-wrong rate   | Not ~80%+         | Too hard **for this pool** |

**Paper wording:** Report that the locked configuration exhibited **non-trivial routing opportunity** with a **mixed outcome distribution** (weak ✘, strong ✔ on a meaningful fraction of queries; not dominated by easy or too-hard). Justify from the bucket table — do not cite internal heuristics as scientific thresholds.

---

## 8.3 Phase A — Smoke validation (C3)

> **Research question:** Can the proposed signals be extracted on the same queries, and do they behave sensibly (not degenerate)? Does the experimental apparatus work end-to-end?

Main-study signals (D15): \(c(q)\), token entropy \(H\), log-probability margin \(m\).

| Check                    | Question                                                 | Pass criterion (propose)                  |
| ------------------------ | -------------------------------------------------------- | ----------------------------------------- |
| **Computable**           | Do H and m extract without error on every (q, model)?    | 100% success on smoke set                 |
| **Vary**                 | Do values differ across queries (not constant)?          | Non-zero variance per model               |
| **Differ across models** | On the same q, do weak and strong show separable H or m? | Visual / descriptive separation on sample |

**Do not** compute Spearman ρ or claim informativeness during validation — that belongs to **EXP-01** / **EXP-02** (Phase C).

**Phase A — pass criteria (preliminary evidence, not correlation):**

| Criterion            | Promising                                   | Reject / debug             |
| -------------------- | ------------------------------------------- | -------------------------- |
| Computable           | 100% (q, model) extract H, m without error  | Any systematic failure     |
| Vary across queries  | Non-zero variance per model (not constant)  | Degenerate — all identical |
| Differ across models | Visible separation weak vs strong on sample | Identical on every q       |

Phase A and Phase B answer **different questions**: Phase B = suitable configuration? Phase A = signals usable + apparatus OK?

| Metric                      | Record                                                                |
| --------------------------- | --------------------------------------------------------------------- |
| Signal compute success rate | % (q, model) with valid H, m                                          |
| **Example rows (5–10)**     | Query × weak/strong entropy, margin, max_prob — for paper + debugging |

Example table to save in `notes.md` or `analysis/`:

| Query | Weak H | Strong H | Weak m | Strong m |
| ----- | ------ | -------- | ------ | -------- |
| q1    |        |          |        |          |

---

## 8.4 Lock and main study (Phase A + Phase B must pass)

```text
Phase A pass (§8.3) — apparatus works; signals extractable
      ↓
Phase B pass (§8.2) — configuration scientifically suitable
      ↓
Lock ONE configuration in 09 (pool + dataset + protocol)
      ↓
Signal extraction pipeline
      ↓
Phase C — main study (ARC) → Studies I–IV (EXP-01–03)
      ↓
Routing policy (C6) — only after characterization
```

---

Phase C — main study (ARC) → Studies I–IV (EXP-01–03)
↓
Routing policy (C6) — only after characterization
↓
Phase D — generalization (optional second dataset, e.g. MMLU if ARC primary)

```

---

## 9. Candidate dataset survey

> **Decision order (D23):** Research question → R1–R5 → candidate triples → literature plausibility (`14` §6) → validation → lock. **Not:** literature → pick benchmark → run.
>
> Score candidates against R1–R5 and the litmus test (§3). Literature citations are **supporting evidence**, not selection criteria.

### 9.1 Methodology-first priority (v1)

| Priority | Dataset | Why it fits *your* methodology | Literature support (secondary) | Role |
| -------- | ------- | -------------------------------- | ------------------------------ | ---- |
| 1 | **ARC-Challenge** | Objective MCQ; simple letter eval; good for setup validation | Plaut (H/m), RouterBench | Phase B candidate A |
| 2 | **MMLU (1–2 subjects)** | Same MCQ protocol; asks *do signals generalize across knowledge domains?* | RouteLLM, Plaut, He, Kadavath | Phase B fallback or **Phase D** |
| 3 | HellaSwag / Winogrande | Different reasoning style; same MCQ interface | Plaut, RouterBench | Phase D optional |
| 4 | GSM8K | Free-form numeric answer; parsing complexity | RouteLLM, GraphRouter, Lugoloobi | **Defer** — not primary validation (Qwen smoke: too hard) |
| — | MATH | Harder than GSM8K; small instruct models struggle | Lugoloobi (activation probes) | **Not v1** — defer unless math is explicit RQ |

**ARC wording (paper):** ARC-Challenge is the **strongest initial candidate** because it satisfies experimental requirements (objective labels, simple evaluation, compatible prompting) and is widely used in closely related uncertainty work — **not** because it is the most popular benchmark.

### 9.2 Requirement scorecard

| Dataset | R1 | R2 | R3 | R4 | R5 | Hypothesis if included | Concerns |
| ------- | -- | -- | -- | -- | -- | ---------------------- | -------- |
| ARC-Challenge | ✓ | Moderate | ✓ | ✓ (smoke) | ✓ | RH2, H2, H3 | Subject/domain fixed to science MCQ |
| MMLU (1–2 subjects) | ✓ | Varies | ✓ | TBD | ✓ | RH2; generalization | Subject choice affects spread — pick deliberately |
| HellaSwag / Winogrande | ✓ | Moderate | ✓ | TBD | ✓ | RH2; generalization | Completion-style MCQ |
| GSM8K | ✓ | Moderate | ✓ | TBD | ✓ | RH2, RH4 | Numeric parsing; risk too_hard on small pools |
| MATH (subset) | ✓ | High | ✓ | TBD | Partial | RH2 | Both models may fail; defer past v1 |
| HumanEval | ✓ | High | ✓ | TBD | Partial | RH4 | Code family; different from MCQ pilot |

---

## 10. Dataset selection (locked — D33)

| Field | Value |
| ----- | ----- |
| **Primary dataset** | *TBD — Phase B chooses among candidates A/B (`08` §8)* |
| **Generalization dataset (Phase D)** | *TBD — likely MMLU (1–2 subjects) if ARC primary* |
| **Split** | *TBD* |
| **N (main study ARC)** | Full TEST split (1,022) or n≈200 preview first |
| **N (Phase D)** | 50–100 (optional) |
| **Justification** | Requirements first (`§5`); validation evidence; literature as support (`14` §6) |

---

## Baselines (routing phase)

| Baseline | Description |
| -------- | ----------- |
| Always-small | Cheapest pool model for every q |
| Always-large | Strongest pool model for every q |
| Random | Uniform over pool |
| Oracle | Best model per q (upper bound) |
| Ours | Entropy + margin + simple policy |

---

## Table shells

### T2: Signal–oracle correlation

| Signal | ρ | Informative? |
| ------ | - | ------------ |
| Entropy (weak M) | | |
| Margin (weak M) | | |

### T4: Main routing results

| Method | Accuracy | Avg cost | Δ vs always-small |
| ------ | -------- | -------- | ------------------- |
| Always-small | | | — |
| Always-large | | | |
| Ours | | | |
| Oracle | | | |

---

## Daily log

**Decision:**


**Evidence:**


**Uncertainty:**

```
