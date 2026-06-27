# ACL submission milestones

> **Mindset:** Submit the **shortest paper that proves the claims** — not “an ACL paper.”  
> **Gate:** [`claims_evidence_matrix.md`](claims_evidence_matrix.md) — every Intro/Abstract sentence maps to evidence.

**Narrative (locked):** Characterize unsupervised pre-inference information first; **routing is validation**, not primary novelty.

---

## Milestone 1 — Complete draft (2–3 days)

**Goal:** A **full ugly draft** end-to-end. Missing polish is fine; missing sections is not.

**Write order (internal):**

```text
Results + figures  →  Methods + Setup  →  Discussion (rough)
  →  Introduction + Related Work  →  Abstract (draft)
```

**Evidence on disk today:** C0 ARC + MMLU transfer (C1–C5). C6 (layerwise evolution) = placeholder “future work” until Milestone 2.

**Exit criteria:**

- [ ] All sections exist in `paper/draft/*.tex`
- [ ] T2, T3, T4 filled; MMLU transfer subsection drafted
- [ ] F1–F3 referenced (F7 stub or omitted)
- [ ] Claim → Evidence Map started (audit table in matrix doc)

**Do not:** new benchmarks, agents, paraphrases, hypothesis redesign.

---

## Milestone 2 — C3 decision (2 days, optional scope lock)

**Goal:** Resolve the **only remaining experimental uncertainty**. No endless tuning.

| Day | Task | Stop rule |
| --- | ---- | --------- |
| 1 | `layerwise.py` + smoke n=10 | Fail → C6 = future work; move to M3 |
| 2 | ARC TEST weak + strong → F7 | Include §5.7 **or** one future-work sentence |

**Decision (binary):**

| Outcome | Paper |
| ------- | ----- |
| F7 / RH5 informative | Add Methods paragraph + §5.7; C6 row ✅ |
| Null or smoke fails | Remove C6 from Intro; single future-work line |

**Then:** Patch Results/Methods/figure list only — no full rewrite.

**Plan:** [`c3_prefill_extensions_plan.md`](c3_prefill_extensions_plan.md)

---

## Milestone 3 — Revision pass (2–3 days)

**Goal:** What reviewers read most — not more experiments.

**Focus (in order):**

1. **Figures** — each figure supports exactly one claim row  
2. **Introduction** — precise RQ; no unsupported promises  
3. **Discussion** — difficulty vs recoverability; exploitation gap; MMLU caveats; limits  
4. **Abstract** — one sentence per claim row; written **last**

**Secondary:** Methods tightening, Related Work trim, caption polish.

**Exit criteria:**

- [ ] Claim → Evidence Map complete (every Abstract + major Intro claim)  
- [ ] No claim without table/figure  
- [ ] Wording conservative: “partially,” “modest,” “pooled” where needed  
- [ ] `pdflatex` builds clean  

---

## Timeline sketch (~7–8 days)

| Days | Milestone |
| ---- | --------- |
| 1–3 | M1 — ugly full draft |
| 4–5 | M2 — C3 run + include/future-work decision |
| 6–8 | M3 — Intro, Discussion, figures, Abstract |

---

## What success looks like at submission

| Property | Test |
| -------- | ---- |
| Precise RQ | One sentence; contrast with supervised/post-inference routing |
| Supported claims | Every claim row has T/F evidence |
| Conservative scope | No agents, paraphrases, SOTA router claims |
| Routing as validation | Study IV ≤ ~1 page; characterization is the story |

That is the version to submit to ACL.
