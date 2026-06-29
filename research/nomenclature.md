# Nomenclature

> **Vocabulary frozen — 2026-06-28 (ACL v1).** Terms below are locked. Do not rename unless a **compelling technical reason** forces it. Log any break-glass change in §7.
>
> Every name states *what it is* — no campaign codes or informal aliases.

**Program:** [`program.md`](program.md) · **Related work:** [`literature_record.md`](literature_record.md)

### Locked vocabulary (summary)

| Category | Canonical home | Do not substitute |
| -------- | -------------- | ----------------- |
| Paper voice | §1 | *support routing*, *solve routing*, Family 1/2/3 |
| Model pool | program §3 | *weak* / *strong* as model labels |
| Unsupervised vs supervised | program §2 | “no labels ever”; pre- vs post-inference as headline |
| Routing need, oracle | program §3 | informal “opportunity” without \(r(q)\) |
| Signal assumptions | program §5 · §0.8 | informal “try entropy” without prior |
| Signal types | nomenclature §4 | model-derived only; Family 1/2/3 |
| Pipeline stages | §3–§4, program §11 | ad-hoc *Phase* names; “Stage” = execution only (§11) |
| \(s(q)\), \(\pi(q)\), \(\lambda\), \(\tau\) | **§2.1** | redefining score/policy elsewhere |
| Hypotheses H1–H4 | program §10 | informal claim lists |
| Evaluation | program §9 | vague “better routing” without Pareto |

---

## 1. Paper voice

| Context | Wording |
| ------- | ------- |
| **Paper / Intro / RQ** | **Enable routing decisions** · **Guide routing decisions** |
| **Avoid** | *Support routing* (advisor-cautious) · *Solve routing* · *Family 1/2/3* |

---

## 2. Domain terms

Quick lookup. **Routing score and policy:** full math in **§2.1 only** — do not repeat elsewhere.

| Term | Symbol | Definition |
| ---- | ------ | ---------- |
| **Unsupervised (signals)** | — | Layer 1 ([`program.md`](program.md) §2): no routing labels at extraction |
| **Supervised (field baseline)** | — | Routers trained on prefs/outcomes (RouteLLM, …) |
| **Model pool** | \(\mathcal{M}\) | Fixed set of models with distinct capabilities — program §3 |
| **Primary pair** | \(M_{\mathrm{lo}}, M_{\mathrm{hi}}\) | Lower- / higher-capability members for threshold routing |
| Per-model correctness | \(y(q,M)\) | Correct answer under frozen protocol |
| **Routing need** | \(r(q)\) | \(\mathbb{1}[y_{\mathrm{lo}}=0 \wedge y_{\mathrm{hi}}=1]\) — [`program.md`](program.md) §3 |
| Opportunity query | \(r(q)=1\) | Bucket name for routing need |
| Correctness gap | \(\Delta(q)=y_{\mathrm{hi}}-y_{\mathrm{lo}}\) | \(+1\) iff routing need |
| Feature vector | \(x(q) \in \mathbb{R}^d\) | Unsupervised signals; no \(r(q)\) at extraction |
| **Routing score** | \(s(q)\) | **§2.1** |
| **Routing policy** | \(\pi(q)\) | **§2.1** |
| Weights / threshold | \(\lambda, \tau\) | Fit on calib only; **§2.1** |
| **Task accuracy** | \(\mathrm{Acc}(\pi)\) | [`program.md`](program.md) §9 |
| **Average cost** | \(\mathrm{Cost}(\pi)\) | [`program.md`](program.md) §9 |
| **Scalar objective** | \(J_\alpha(\pi)\) | [`program.md`](program.md) §9; calib tuning optional |
| Oracle route | \(\pi^*(q)\) | [`program.md`](program.md) §3; eval upper bound |
| **Experimental Setting** | \(\mathcal{S}\) | \(\mathcal{S}=(\mathcal{D},\mathcal{P},\Pi)\) — benchmark, pool, protocol; frozen at M3 — [`program.md`](program.md) §0.7 |
| **Evaluation corpus** | \(C\) | Labeled queries for the study; methodology defined in terms of \(C\), not HF splits |
| **Project calib** | \(R_c\) | H1–H3 + policy fit; partition of \(C\) — §0.7.1 |
| **Project test** | \(R_t\) | H4 only; partition of \(C\) |
| **Setting validity** | — | Environment passes Gates A–E; does **not** imply signals will work — §0.7 |
| **Deployment scenario** | — | `homogeneous_pool` / `heterogeneous_pool` — pool homogeneity (vendor, tokenizer) — §0.7 |
| **Gap-size robustness** | — | Optional post-primary pairs (1B→3B ablation; 8B→70B) — not Phase A grid — §0.7 |
| **Tuning policy** | — | Only \(\lambda, \tau\) on \(R_c\); all else frozen — §0.17 |
| **Corpus partition** | Option 1 | \(C \to H \mid R_c \mid R_t\); \(H\) never reused — §0.7.2 |
| **H1–H4** | — | [`program.md`](program.md) §10 |

### 2.1 Routing mathematics (canonical)

**Define once here.** Paper and [`program.md`](program.md) refer to this block; **never redefine** \(s(q)\) or \(\pi(q)\) elsewhere.

**Score** — combine frozen features with weights \(\lambda \in \mathbb{R}^d\):

\[
s(q) = \lambda^\top x(q) = \sum_{i=1}^{d} \lambda_i \, x_i(q)
\]

Higher \(s(q)\) = stronger evidence of routing need \(r(q)=1\) (escalate to \(M_{\mathrm{hi}}\)).

**Policy** — threshold router given \(\tau \in \mathbb{R}\):

\[
\pi(q) =
\begin{cases}
M_{\mathrm{hi}}, & s(q) > \tau \\
M_{\mathrm{lo}}, & \text{otherwise}
\end{cases}
\]

**Fit:** \((\lambda, \tau)\) on **calib** only (layer 3, [`program.md`](program.md) §2); lock before test. Tracks 4a (hand \(\lambda\)) / 4b (learned \(\lambda\)): [`program.md`](program.md) §8.

---

## 3. Pipeline vs execution stages

**Scientific pipeline** (paper Method — program §4):

| Pipeline step | Section | Question |
| ----- | ------- | -------- |
| Signals | program §5 | What can we measure? (layer 1) |
| Signal analysis | program §6 | What predicts \(r(q)\)? (layer 2) |
| Signal selection | program §7 | What enters \(x(q)\)? |
| Routing policy | program §8 | \(\pi(q)\) from §2.1 |
| Evaluation | program §9 | Accuracy–cost trade-off? |

**Execution workflow** (program **§11**, Stages **0–9**; Phase A = M1–M3, Phase B = M4): research design → setting specification → feasibility assessment → setting lock → oracle → signals → analysis → selection → policy → evaluation.

Supervision layers: program §2 (defined once).

---

## 4. Signal types

| Type | Meaning |
| ---- | ------- |
| **Query-derived** | Query text only (model-independent) |
| **Model-response** | One model on query (model-dependent) |
| **Cross-model comparative** | Pool gaps, disagreement (our refinement) |

Detail and **signal assumptions:** [`program.md`](program.md) §5.

Canonical refs: §2.1 (\(s\), \(\pi\)) · program §0 · §2 · §3 · §4 · §5 · §9 · §10 · §11 (Stages 0–9).

---

## 5. Project layout

**Planned** (not all paths exist yet — fresh start):

```text
llm_routing/
  scripts/               ← future implementation (see scripts/README.md)
  experiments/           ← stage artifacts
  paper/
  research/
    program.md
    nomenclature.md
    literature_record.md
  old_llm_routing/       ← retired; gitignored
```

---

## 6. Retired

Do not extend: prefill traces, layerwise campaigns, numbered experiment codes. Archive: `old_llm_routing/`.

---

## 7. Terminology log

**Pre-freeze history** (2026-06-28 and earlier). **Post-freeze:** append only when breaking the freeze; include technical justification.

| Date | Old | New | Reason |
| ---- | --- | --- | ------ |
| **2026-06-28** | *(open vocabulary)* | **FROZEN** | ACL v1 — no renames without compelling technical reason |
| 2026-06-28 | T1/T2 IDs | RouteLLM, Plaut et al. | Researcher-style names |
| 2026-06-28 | support routing | enable / guide routing decisions | Paper voice |
| 2026-06-28 | Family 1/2/3 | query-derived, model-response, cross-model comparative | Concept words |
| 2026-06-28 | opportunity (informal) | routing need \(r(q)\) | Precise definition |
| 2026-06-28 | 9 working docs + stable/ | 3 docs: program, nomenclature, literature | Remove duplication |
| 2026-06-28 | implicit score | \(s(q)\), \(\pi(q)\) in nomenclature §2.1 only | Single routing-math home |
| 2026-06-28 | informal claim list | H1–H4 nested ablation hypotheses | Maps to experiments |
| 2026-06-28 | accuracy + cost (vague) | \(\mathrm{Acc}\), \(\mathrm{Cost}\), Pareto; \(J_\alpha\) optional | Defines “better routing” |
| 2026-06-28 | mixed signal/routing stages | 5-stage pipeline | Signal vs routing |
| 2026-06-28 | repeated concepts | define once, cross-ref | Reduce duplication |
| 2026-06-28 | informal phase names | execution Stages 0–9 (program §11) | Setup vs scientific pipeline |
| 2026-06-28 | weak / strong model labels | **model pool** + \(M_{\mathrm{lo}}\) / \(M_{\mathrm{hi}}\) | Distinct capabilities, not binary branding |
| 2026-06-28 | joint Setting in one stage | Phase A M1–M3: specification → feasibility → lock | Reproducible coupled design |
| 2026-06-25 | HF validation → calib (methodology) | **Evaluation corpus** \(C\) + uniform partition | Future-proof; no HF split-name dependency |
| 2026-06-25 | Pool A / Pool B | **Deployment scenarios** (`narrow_gap`, `wide_gap`) | Scientific meaning for pools |
| 2026-06-28 | `pool.scenario` | **`pool.deployment_scenario`** | Distinguish from task scenario; extensible values (same_family, api_router, …) |
| 2026-06-25 | Dataset × pool grid; `narrow_gap`/`wide_gap` as primary axis | **`homogeneous_pool`**; pool frozen in M1; M2 varies **dataset only** | Align Phase A with fixed-pool RQ |
| 2026-06-25 | Setting | **Experimental Setting** \(\mathcal{S}\) | Unambiguous in paper; avoids generic "setting" |
| 2026-06-25 | fixed pilot N=150 | `selection_holdout_n` parameter | Methodology budget-independent |
| 2026-06-25 | — | **Setting validity** definition + architecture **FROZEN** | Reviewer protection; stop redesign |

**Break-glass procedure:** (1) document technical reason in a new row above; (2) update [`program.md`](program.md), paper, and code together; (3) do not introduce synonyms in parallel.
