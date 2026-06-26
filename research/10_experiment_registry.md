# Experiment Registry

> **Implementation layer** — scripts, artifacts, notebook IDs.  
> **Science:** [`MASTER.md`](MASTER.md) · **Workflow:** [`WORKFLOW.md`](WORKFLOW.md) · **Claims:** `claims.md`

---

## At a glance

```text
Validation     V1 Smoke  →  V2 Configuration  →  lock (09)
Calibration    D46 on ARC validation (~299)  →  freeze c(q)  [pre-study; NOT a paper experiment]
Main study     EXP-01 (Studies I–II) → EXP-02 (III) → EXP-03 (IV)
Paper          §5.2–5.3                  §5.4          §6
Supplement     §5.5 generalization (MMLU) · §7.2 error analysis
```

**Rule (D47):** Four paper studies — one hypothesis each. Three notebook experiment IDs group them.

**Frozen blueprint:** `MASTER.md`

---

## Notebook ID structure (D56)

| Notebook ID | Name                        | Contains                                                         | Hypotheses | Paper    |
| ----------- | --------------------------- | ---------------------------------------------------------------- | ---------- | -------- |
| **V1, V2**  | Validation                  | Apparatus + configuration lock                                   | —          | §4 setup |
| **EXP-01**  | **Signal characterization** | **Study I** + **Study II** | RH1, RH2   | §5.2–5.3 |
| **EXP-02**  | **Complementarity**         | **Study III**                                                    | RH3        | §5.5     |
| **EXP-03**  | **Routing evaluation**      | **Study IV**                                                     | RH4        | §6       |

EXP-01 is one analysis pipeline with two paper subsections—not two separate notebook experiments.

---

## Paper study ↔ notebook mapping

> **RH definitions:** `claims.md` §Hypotheses. Studies I–II jointly test RH1; Study II + interpretation tests RH2.

| Paper study | Hypothesis | Notebook ID     | §    |
| ----------- | ---------- | --------------- | ---- |
| **I**       | RH1        | EXP-01 (part 1) | §5.2 |
| **II**      | RH1, RH2   | EXP-01 (part 2) | §5.3 |
| **III**     | RH3        | EXP-02          | §5.4 |
| **IV**      | RH4        | EXP-03          | §6   |
| —           | —          | Phase D (MMLU)  | §5.5 |

---

## Hypothesis ↔ main study

| Hypothesis              | Main-study run     | Claim | Primary metric            | Status       |
| ----------------------- | ------------------ | ----- | ------------------------- | ------------ |
| **RH1** · H1 (\(c(q)\)) | EXP-01 / Study I   | C4    | ρ, AUROC for \(c(q)\)     | Planned      |
| **RH2** · H2, H3        | EXP-01 / Study II  | C4    | ρ, AUROC for \(H, m\)     | Planned      |
| **RH3**                 | EXP-02 / Study III | C5    | ΔAUROC across families    | Planned      |
| **RH4**                 | EXP-03 / Study IV  | C6    | Cost–quality vs baselines | Planned      |
| H4 (paraphrase)         | —                  | —     | —                         | Deferred D04 |

**Validation (V1, V2)** gates the setup; they do **not** test RH1–RH4.

---

## Two registries (do not mix numbering)

| Registry       | IDs                        | Tests hypothesis? | Maps to claims |
| -------------- | -------------------------- | ----------------- | -------------- |
| **Validation** | **V1, V2**                 | **No**            | C1–C3          |
| **Main study** | **EXP-01, EXP-02, EXP-03** | **Yes**           | C4–C6          |

**Signal scope (frozen):** representative \(c(q)\) + entropy \(H\) + margin \(m\). Paraphrase deferred (D04).

---

## Validation registry

| ID     | Name              | Question                                              | Claim  | Typical n | Script / artifact                          | Status   |
| ------ | ----------------- | ----------------------------------------------------- | ------ | --------- | ------------------------------------------ | -------- |
| **V1** | **Smoke**         | Apparatus end-to-end? Prefill probes extract cleanly? | C1, C3 | ~10       | `verify_logprobs.py`, `extract_signals.py` | **Done** |
| **V2** | **Configuration** | Is `(pool, dataset, protocol)` studiable?             | C2     | 50        | `routing_opportunity_assessment.py`        | **Done** |

**Gate:** V1 + V2 pass → lock in `09` → **D46 signal screening on CALIB** → **EXP-01–03** on ARC TEST.

---

## D46 pre-study calibration (NOT an experiment ID)

| Process | Name | Question | Paper? | Slice | Status |
| ------- | ---- | -------- | ------ | ----- | ------ |
| **D46** | **Pre-study calibration** | Which model-independent candidate is the best **representative** \(c(q)\) on CALIB? | **No** — Methods setup only | ARC validation (~299) | Protocol locked (D57/D60); outcome pending |

Families: Length · Lexical diversity · Information · Compressibility. CLI: `run.py screen` → `analysis/d46_signal_screen_*.json`, `analysis/selected_feature.json`.

Selection: composite of normalized \(|\rho|\) + AUROC with bootstrap CIs. Output: `selected_candidate` → unified `c_q`.

Does **not** test RH1–RH4; gates which column becomes \(c(q)\) for Studies I–IV.

---

## Main study registry

Same locked config as V2. ARC primary; MMLU after ARC interpretation.

| ID         | Name                        | Question                                                                     | Claim | Studies | Hypotheses | Paper    |
| ---------- | --------------------------- | ---------------------------------------------------------------------------- | ----- | ------- | ---------- | -------- |
| **EXP-01** | **Signal characterization** | How informative are \(c(q)\), \(H\), \(m\) (and \(\Delta H\), \(\Delta m\))? | C4    | I, II   | RH1, RH2   | §5.2–5.3 |
| **EXP-02** | **Complementarity**         | Do families add information beyond each other?                               | C5    | III     | RH3        | §5.5     |
| **EXP-03** | **Routing evaluation**      | Does a simple hold-out router improve cost–quality vs baselines?             | C6    | IV      | RH4        | §6       |

**EXP-03** runs after EXP-01 and EXP-02. **Independent CALIB-fit logistic + τ** — do not reuse Study III model artifacts. CLI: `run.py route-eval`.

**Study III (D61):** Primary RH3 ladder `c(q) → c+H → c+H+m` on TEST; secondary `H → H+m` in appendix. Script: `analyze_exp02.py` (requires `c_q` in merged CSV).

### Optional / deferred (no EXP ID)

| Topic                  | Status                 |
| ---------------------- | ---------------------- |
| Paraphrase stability   | Deferred D04           |
| Failure-case deep dive | Qualitative supplement |
| Probe overhead (A2)    | Appendix table         |

---

## Evidence ladder

```text
V1  Smoke           ─┐
V2  Configuration   ─┼─► Lock config (09)
                      │
EXP-01  Signal char.  ─┼─► Studies I–II (§5)
  (RH1, RH2 / C4)     │
EXP-02  Complement.   ─┘─► Study III (§5.5)
  (RH3 / C5)          │
EXP-03  Routing       ───► Study IV (§6)
  (RH4 / C6)
```

---

## Rules

1. **V1, V2** — C1–C3 only.
2. **EXP-01–03** — C4–C6, RH1–RH4, `claims.md`.
3. Status: Planned → Running → Done / Failed / Superseded.
4. No EXP-04+ without decision in `09`.
5. New notes use **current IDs only** (§Superseded IDs for archival).

---

## Superseded IDs (archival)

| Old ID                       | Current ID |
| ---------------------------- | ---------- |
| Phase A / old Validation 2   | **V1**     |
| Phase B / old Validation 1   | **V2**     |
| Old EXP-01, EXP-02 (logprob) | **V1**     |
| Old EXP-03 (informativeness) | **EXP-01** |
| Old EXP-06 (complementarity) | **EXP-02** |
| Old EXP-08 (routing)         | **EXP-03** |
