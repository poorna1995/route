# Claims ↔ Evidence matrix

> **One-page gate (print this):** see [Claim → Evidence Map](#claim--evidence-map-one-page) below.  
> **Milestones:** [`submission_milestones.md`](submission_milestones.md)  
> **Numbers:** [`paper_process_and_results.md`](paper_process_and_results.md) · **Vocabulary:** [`claims.md`](claims.md)

**Mindset:** Write the **shortest paper that proves these claims** — not “an ACL paper.”

---

## Claim → Evidence Map (one page)

| Claim | Evidence | Status |
| ----- | -------- | ------ |
| Pre-inference routing information exists | **T2**, **F1** | ✅ |
| Difficulty and recoverability differ | **T2**, **F2**, **F6** (Cohen's d) | ✅ |
| Information sources are partially complementary | **T3**, **F3** | ✅ |
| Pattern generalizes (pooled) | **RH7** / MMLU table, `C2_dimension_transfer.json` | ✅ |
| Current routing under-exploits available information | **T4** | ✅ |
| *(Optional)* Layerwise evolution adds model-derived information beyond terminal | **F7** (§5.7) | ⏳ M2 |

**Rule:** If a row has no evidence → **weaken or remove** from Intro and Abstract.

*Note:* Difficulty ≠ recoverability lives in **T2 + figures**, not T3 (T3 = complementarity).

---

## Submission gate

For each claim below, confirm:

1. A table or figure exists on disk  
2. The LaTeX file `\ref{}` matches  
3. Wording in Intro/Abstract does not oversell (use “partially,” “modest,” “pooled” where noted)

---

## Core claims (C0 + MMLU — evidence on disk)

| # | Claim (paper sentence) | Primary evidence | Supporting artifacts | Status |
| - | ---------------------- | ---------------- | -------------------- | ------ |
| **C1** | Routing-relevant **information exists before decoding** | **T2**; **F1** | `analysis/arc_merged.csv`; opportunity 43.3%; best AUROC ~0.61 | ✅ Done |
| **C2** | **Difficulty ≠ recoverability** are distinct pre-inference properties | **T2** (Cohen's d); **F2**, **F6** | \(H_w\) \(d≈0.03\) vs \(\Delta m_{\mathrm{gain}}\) \(d≈0.72\); recovery matrix | ✅ Done |
| **C3** | Information sources are **partially complementary** | **T3**; **F3** | \(\Delta\)AUROC \(+0.049\), DeLong \(p{=}0.008\) | ✅ Done |
| **C4** | Structure **generalizes (pooled)** to MMLU transfer | MMLU transfer table / §5.x | `analysis/C2_dimension_transfer.json`; pooled \(d≈0.72\) for \(\Delta m_{\mathrm{gain}}\) | ✅ Done |
| **C5** | Simple calibrated routing leaves an **exploitation gap** | **T4** | Policy 69.2% = always-strong; oracle 74.4% (5.2 pp) | ✅ Done |

**Do not claim:** SOTA router · uniform MMLU subject transfer · agents · paraphrase stability.

---

## Extension claim (C3 — resolve before writing Intro)

| # | Claim | Primary evidence | Supporting artifacts | Status |
| - | ----- | ---------------- | -------------------- | ------ |
| **C6** | Layerwise **confidence evolution** adds routing information beyond terminal (or null) | **F7**; RH5 JSON | C3 campaign | ⏳ M2 |

**Decision rule (Day 2):**

| C3 outcome | Paper action |
| ---------- | ------------ |
| F7 separates buckets / RH5 supported | §5.7 + one Methods paragraph; C6 row ✅ |
| Null / redundant with terminal | One future-work sentence; **remove C6 from Intro** |

---

## Hypothesis map (internal)

| Hypothesis | Claim rows | Table |
| ---------- | ---------- | ----- |
| RH1 | C1 | T2 |
| RH2 | C2 | T2, F2, F6 |
| RH3 | C3 | T3 |
| RH4 | C5 | T4 |
| RH7 | C4 | MMLU table |
| RH5 | C6 | F7 (optional) |

---

## Reader-facing findings → claims

| Finding | Maps to |
| ------- | ------- |
| **F1** — dimensions predict opportunity | C1 |
| **F2** — difficulty ≠ recoverability | C2 |
| **F3** — exploitation gap | C5 |
| *(Results subsection)* MMLU transfer | C4 |
| *(Optional §5.7)* layerwise confidence evolution | C6 |

---

## Writing order (Milestone 1 — ugly draft first)

See [`submission_milestones.md`](submission_milestones.md).

```text
M1  Results → Methods → Discussion (rough) → Intro → Abstract (draft)
M2  C3 smoke → TEST → F7 → include or future work (no endless tuning)
M3  Figures → Intro → Discussion → Abstract (final) + claim audit
```

---

## Intro / Abstract audit (fill before submit)

| Sentence (draft) | Claim row | Table/Fig | Pass? |
| ---------------- | --------- | --------- | ----- |
| | | | |
| | | | |

Copy each Abstract sentence into the table above. Empty evidence column → cut or weaken.
