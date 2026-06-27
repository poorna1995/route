# Model-Independent Signal Selection (survey archive)

> **\(c(q)\) selection (D46):** One **representative query-derived** complexity signal (paper term; legacy filename: model-independent), chosen via **D46 signal screening** (`05` §8, `screen_cq_candidates.py`) — **not a paper experiment**. CALIB 150 only to lock. Families: Length · Lexical diversity · Information · Compressibility.

This file retains the **literature survey** (papers Lugoloobi, RouteLLM, etc.) for Methods justification. Do not use for frozen config — see MASTER.

---

## 1. Why this matters for the RQ

**Research question:** How informative are **unsupervised pre-inference signals** for selecting an appropriate LLM?

The RQ says **signals**, not only model-derived probes. Taxonomy 𝒮 already defines:

```text
𝒮 = 𝒮_indep ∪ 𝒮_dep

𝒮_indep   query-only        s(q)           e.g. c(q)
𝒮_dep     prefill probes    s(q, Mᵢ)       H, m
```

**Current experiments:** only \(H_w, m_w, H_s, m_s\) ∈ 𝒮_dep.

**Reviewer risk:** "Why introduce 𝒮 if only one branch is evaluated?"

**Fix:** Add **one** query-derived signal \(c(q)\) with a **literature-justified** definition — not a kitchen-sink of lexical features.

---

## 2. Target signal set (after selection)

| Family | Signal | Notation | Status |
| ------ | ------ | -------- | ------ |
| Model-independent | **Representative complexity** | \(c(q)\) | **D46 pending** — screen C01/C03/C10/C11 on CALIB |
| Model-dependent | Token entropy (uncertainty) | \(H\) | Frozen |
| Model-dependent | Log-prob margin (confidence separation) | \(m\) | Frozen |

**Selection logic:** Survey complexity family → score candidates (`§7`) → freeze **one representative** for v1. Paper claims characterize the **family via a representative**, not that one formula exhausts 𝒮_indep.

**Not in v1:** length alone, TF-IDF stack, readability suite, paraphrase, extra models for query embedding.

---

## 3. Hard requirements for \(c(q)\)

Must satisfy **all** before freeze:

| # | Requirement |
| - | ----------- |
| R1 | Computed **before** any model forward pass on the routed pool |
| R2 | **Unsupervised** — no oracle labels, no routing history, no fine-tuned encoder |
| R3 | **Deterministic** — same \(q\) → same \(c(q)\) |
| R4 | **Reproducible** — documented formula, no proprietary API |
| R5 | **Inexpensive** — CPU-only, negligible vs prefill |
| R6 | **Domain-agnostic** — same code for ARC, MMLU, BoolQ |
| R7 | **Objective** — no human judgment at runtime |

**Disqualify:** supervised difficulty (Hybrid LLM), query embeddings trained on routing labels (RouteLLM), features requiring a second LLM forward pass (unless advisor explicitly accepts as boundary case).

---

## 4. Workflow (this week)

```text
Step 1  Survey literature (10–20 measures)          ← this doc §5
Step 2  Categorize (lexical / structural / …)      ← §6
Step 3  Score each against §3 requirements         ← §7 rubric
Step 4  Pick ONE → decision in 09 (D46)            ← freeze
Step 5  Spec in 05 §8 + implement extract step     ← after freeze
Step 6  Merge: query_id, c(q), H, m, bucket        ← unified CSV
```

**Parallel track OK:** C2 screening (MMLU, BoolQ) and ARC n=200 can proceed **without** \(c(q)\) only if you accept re-running merge/analysis once \(c(q)\) is added. **Advisor preference:** freeze \(c(q)\) **before** authoritative ARC n=200 if possible.

---

## 5. Literature survey — candidate measures

**Deep-read sources (2026-06-22):** Lugoloobi et al. (T2-10), RouteLLM (T1-01), Hybrid LLM (T1-03), RouterBench (T1-05). Full paper notes → `14_literature_record.md` §3–4.

### 5.1 Master table (unsupervised candidates + supervised contrasts)

| ID | Measure | Category | Definition / source | Model pass? | Supervised? | Key evidence (paper) | Eligible §3? | Prior decision |
| -- | ------- | -------- | ------------------- | ----------- | ----------- | -------------------- | ------------ | -------------- |
| C01 | **Question length** | Lexical | Token or char count of prompt \(q\) | No | No | Lugoloobi Table 1: Spearman \(\rho=0.15\) vs human IRT \(b(q)\); \(\rho=0.19\)–\(0.30\) vs model success \(\hat{s}_{MC}\) on E2H-AMC — **weakest** text baseline [Lugoloobi et al., 2026, arXiv:2602.09924] | Yes | **Drop** D07 |
| C02 | **Mean word length** | Lexical | avg chars per token/word | No | No | Not isolated in Tier-1/2; subsumed by length family | Yes | — |
| C03 | **Type-token ratio (TTR)** | Lexical | unique / total words | No | No | Not evaluated in deep-read set; standard lexical diversity | Yes | — |
| C04 | **TF-IDF → linear model** | Lexical | Bag-of-words TF-IDF features → ridge/logistic for difficulty or success | No* | No* | Lugoloobi Table 1: \(\rho=0.72\)–\(0.74\) vs **human** IRT; only \(\rho=0.25\)–\(0.47\) vs **model** success (degrades with reasoning budget). Table 2 AUROC 0.58–0.86 for binary success — setting-dependent, trails activation probes [Lugoloobi et al., 2026] | Yes* | Warn D05 |
| C05 | **Readability** | Lexical | Flesch-Kincaid, SMOG, etc. | No | No | Not in deep-read set; MCQ science text may be low-variance | Yes | — |
| C06 | **Punctuation / digit density** | Structural | counts / \(\|q\|\) | No | No | Not in deep-read set | Yes | — |
| C07 | **WH-word / interrogative patterns** | Structural | regex counts | No | No | Not in deep-read set; format-sensitive | Yes | — |
| C08 | **MCQ option count** | Structural | \#choices in prompt | No | No | ARC/MMLU=4, BoolQ=2 — cross-dataset confound | Partial | — |
| C09 | **Syntactic depth** | Structural | max parse-tree depth (spaCy/Stanza) | No | No | Not in deep-read set; parser cost only | Yes | — |
| C10 | **Text Shannon entropy** | Info-theoretic | \(H_{\text{text}}=-\sum_w p(w)\log p(w)\) over query tokens | No | No | Not in deep-read set; parallels model entropy naming | Yes | Leading candidate |
| C11 | **Compression ratio** | Info-theoretic | \(\| \text{zlib}(q) \| / \|q\|\) | No | No | Not in deep-read set; cheap structural proxy | Yes | Leading candidate |
| C12 | **Static embedding norm** | Semantic | \(\|E q\|\) with fixed word2vec/GloVe | No | No | Not in deep-read set | Yes | — |
| C13 | **Query perplexity (small LM)** | Semantic | PPL of \(q\) under tiny LM | **Yes** | No | Lugoloobi §2 cites perplexity as prior routing proxy [Chen et al., 2024; Ding et al., 2023] — **second model** | **No** R1 | Disqualify |
| C14 | **Hybrid LLM router score** | Supervised contrast | DeBERTa-v3-large on raw query → \(p_w(x)\approx\Pr[\text{small wins}]\); labels from offline BARTScore on 10× gens/query | Encoder only | **Yes** (BARTScore labels) | Pre-generation **query-only at inference**, but training needs offline gens + quality model [Ding et al., 2024, arXiv:2404.14618, ICLR 2024]. §5 future work: add **task labels** | No R2 | Contrast |
| C15 | **RouteLLM P(strong wins)** | Supervised contrast | SW-ranking / matrix factorization on `text-embedding-3-small`; or BERT/Llama-3-8B classifiers on query text | Embedding API / FT | **Yes** (Arena prefs, MMLU gold, GPT-4 judge) | No explicit length/TF-IDF — **learned embedding** encodes difficulty implicitly [Ong et al., 2024/2025, arXiv:2406.18665, ICLR 2025]. Inference = query text only | No R2 | Contrast |
| C16 | **RouterBench KNN / MLP** | Supervised contrast | SentenceTransformer (`all-MiniLM-L12-v2` best) → KNN or 2-layer MLP predicts per-model quality \(P_{ij}\) | Encoder only | **Yes** (70% precomputed outcomes) | Prompt embedding only — no hand-crafted lexical features [Hu et al., 2024, arXiv:2403.12031, ICML 2024]. RAG: routes on surface cues like year “2024” | No R2 | Contrast |
| C17 | **IRT-Router latent difficulty** | Supervised contrast | Item Response Theory: learned item + model embeddings → latent difficulty [Song et al., 2025, ACL 2025] | Encoder | **Yes** | Lugoloobi routing baseline; “requires additional training” vs probe [Lugoloobi et al., 2026, §4.1] | No R2 | Contrast |
| C18 | **Lugoloobi activation probe** | Model-dep contrast | Linear probe on **pre-generation hidden states** | **Yes** (routed pool) | **Yes** (success labels) | \(\rho=0.40\)–\(0.64\) model difficulty; beats TF-IDF/length [Lugoloobi et al., 2026] | No | Wrong family |
| C19 | **CoT output length** | Post-gen contrast | Total generated tokens | **Yes** | No | Fig. 2: CoT length tracks **human** IRT, **anti**-correlates with model success under extended reasoning [Lugoloobi et al., 2026] | No R1 | Disqualify |
| C20 | **Human IRT \(b(q)\)** | Oracle contrast | Psychometric difficulty from student performance (E2H-AMC) | No | N/A (human data) | Upper bound for human-aligned difficulty; not available at deploy time | No R6 | Oracle only |
| C21 | **FrugalGPT DistilBERT \(g(q,a)\)** | Post-gen contrast | Scorer on **query + answer** after each cascade step | **Yes** (gens) | **Yes** (task labels) | 12 APIs, length-3 cascade; not pre-inference [Chen et al., 2023, arXiv:2305.05176] | No R1 | Contrast |
| C22 | **Normalized length** | Lexical | \(\|q\| / \text{median}(\|q\|)\) per dataset | No | No | Scales length; still length family — weak per C01 | Yes | Unlikely |
| C23 | **Lexical sophistication** | Lexical | mean inverse document frequency / frequency rank | No | No | Mechanistically related to C04 TF-IDF | Yes | — |
| C24 | **Sentence count** | Structural | \# sentences in \(q\) | No | No | Correlates with C01 | Yes | — |

\*C04: no **routed-pool** forward pass; needs corpus IDF stats (fixed reference corpus = hyperparameter).

**Count:** 14 unsupervised-eligible (C01–C12, C22–C24), 7 supervised/post-gen contrasts (C14–C21), 1 wrong-family (C18).

---

### 5.2 Deep-read synthesis (four papers)

#### T2-10 — Lugoloobi et al. (2026) · [arXiv:2602.09924](https://arxiv.org/abs/2602.09924)

**Relevance:** Only deep-read paper that **systematically evaluates unsupervised text baselines** against the same targets we care about (human difficulty vs **model-specific success**).

| Text-only measure | vs human IRT \(b(q)\) | vs model success \(\hat{s}_{MC}\) | vs binary success (AUROC, Table 2) |
| ----------------- | --------------------- | --------------------------------- | ---------------------------------- |
| **TF-IDF + linear** | \(\rho \approx 0.72\)–\(0.74\) | \(\rho \approx 0.25\)–\(0.47\) (↓ with reasoning) | 0.58–0.86 (model-dependent) |
| **Length** | \(\rho = 0.15\) | \(\rho \approx 0.19\)–\(0.30\) | 0.46–0.73 |

**Takeaways for RH1:**

1. Best published **unsupervised** lexical proxy (TF-IDF) aligns with **human** difficulty, not **model-relative** routing need.
2. Length is a poor discriminator for either target on math (confirms D07).
3. CoT **output** length is misleading for model success (Fig. 2) — do not use post-gen proxies.
4. Paper explicitly positions prior routing as using “input length, perplexity, or heuristic confidence” — our \(c(q)\) should be justified as a **stronger** unsupervised text statistic than length, with eyes open that TF-IDF already sets a high bar for human-IRT alignment only.

**Candidates elevated:** C04 (negative-result baseline worth beating), C10/C11 (not in paper — room to test non-lexical unsupervised stats).

---

#### T1-01 — RouteLLM (Ong et al., 2024/2025) · [arXiv:2406.18665](https://arxiv.org/abs/2406.18665)

**Query-side signal at inference:** Raw query text → router; **no** hand-crafted complexity features.

| Router | Query representation | Supervision |
| ------ | -------------------- | ----------- |
| Similarity-weighted ranking | `text-embedding-3-small` | 65k Arena pairwise prefs |
| Matrix factorization | Same embedding × model vector | Same |
| BERT-base classifier | CLS embedding of \(q\) | Same + GPT-4 judge aug. |
| Llama-3-8B classifier | Next-token over win/tie/loss tokens | Same + judge aug. |

**Implication:** Tier-1 “query complexity” is **learned**, not explicit \(c(q)\). Difficulty is discussed conceptually (“infer intent, **complexity**, and domain”) but **not operationalized** as length, TF-IDF, or readability. **Cannot adopt** as our unsupervised \(c(q)\) without violating R2 — use as **contrast**: supervised embeddings vs our single statistic.

**No new unsupervised candidate rows** — confirms the gap our RH1 tests.

---

#### T1-03 — Hybrid LLM (Ding et al., 2024) · [arXiv:2404.14618](https://arxiv.org/abs/2404.14618)

**Query-side signal at inference:** Single forward pass of **DeBERTa-v3-large** on query text → \(p_w(x)\); latency **0.036 s/query** vs 7.99 s (7B LLM).

| Aspect | Detail |
| ------ | ------ |
| Training labels | BARTScore comparing 10 offline responses from small vs large per query |
| “Difficulty” | Implicit in encoder — **not** a scalar \(c(q)\) reported |
| Future work (§5) | “Task labels for query examples” to distinguish easy vs hard — **acknowledges query-only signal is incomplete** |

**Implication:** Motivates RH1 (query-only routing works when supervised) but **not** a formula for unsupervised \(c(q)\). Contrast paper for Discussion: Hybrid LLM needs BARTScore pipeline; we test **label-free** \(c(q)\).

---

#### T1-05 — RouterBench (Hu et al., 2024) · [arXiv:2403.12031](https://arxiv.org/abs/2403.12031)

**Query-side signal:** SentenceTransformer embedding of prompt → KNN (40 NN, cosine) or MLP (2×100 ReLU).

| Embedding model tested | Best config |
| ---------------------- | ----------- |
| `all-MiniLM-L12-v2`, `all-mpnet-base-v2`, `all-distilroberta-v1` | MiniLM-L12-v2 |

**Supervision:** 70% of precomputed (prompt, model, quality, cost) tuples per task.

**Implicit “features”:** No lexical table — embedding subsumes length, topic, format. Qualitative: RAG router picks online models for time-sensitive surface cues (e.g. “2024”).

**ARC-Challenge:** KNN/MLP routers **do not beat** Zero router on cost–quality AIQ (Figure 4) — predictive routing is hard even with supervised embeddings on our primary benchmark.

**Implication:** Supervised embedding routers are the RouterBench ceiling for query-only **learned** signals; our single \(c(q)\) is a deliberate **interpretable** alternative, not a competitor to MLP-on-embeddings.

---

### 5.3 Category map (updated)

| Category | Candidates | Best literature anchor |
| -------- | ---------- | ------------------------ |
| **Lexical** | C01–C05, C22–C23 | Lugoloobi TF-IDF/length (C04/C01) |
| **Structural** | C06–C09, C24 | RouterBench RAG year heuristic (qualitative) |
| **Information-theoretic** | C10–C11 | *No direct anchor in deep-read set* |
| **Semantic (static)** | C12 | — |
| **Learned query representation** | C14–C17 | RouteLLM, Hybrid, RouterBench, IRT-Router — **all supervised** |
| **Post-generation** | C19, C21 | Lugoloobi CoT length; FrugalGPT |

---

### 5.4 Rubric pre-scores (from literature only — empirical tie-break on ARC n=50 later)

Scores 0–2 per §7 column; **total /10** (higher = better candidate). Literature-only — not final pick.

| ID | Unsup. | Routing target | Cross-dataset | Cost | Lit. precedent | **Total** | Notes |
| -- | ------ | -------------- | ------------- | ---- | -------------- | --------- | ----- |
| C01 Length | 2 | 0 | 2 | 2 | 1 | **7** | Known weak — good RH1 floor, bad winner |
| C04 TF-IDF | 2 | 1 | 1 | 2 | 2 | **8** | Strong human-IRT; weak model-success |
| C10 Text entropy | 2 | 0 | 2 | 2 | 0 | **6** | No lit. anchor; clean §3 pass |
| C11 Compression | 2 | 0 | 2 | 2 | 0 | **6** | Same |
| C03 TTR | 2 | 0 | 2 | 2 | 0 | **6** | Untested in lit. |

**Literature-informed shortlist for D46 decision:** C10, C11, C04 (expect RH1 partial/negative), C03. **Eliminate as primary:** C01 (D07), C13–C21 (disqualified contrasts).

---

### 5.5 Reading log entry

| Paper | ID | Deep-read date | 𝒮_indep measures found | Supervised? | Added to table |
| ----- | -- | -------------- | ---------------------- | ----------- | -------------- |
| Lugoloobi et al. (2026) | T2-10 | 2026-06-22 | Length, TF-IDF (+ CoT length post-gen) | Probes yes; text baselines no | C01, C04, C19 — **anchor** |
| RouteLLM | T1-01 | 2026-06-22 | None explicit — embeddings only | Yes | C15 contrast |
| Hybrid LLM | T1-03 | 2026-06-22 | None explicit — DeBERTa encoder | Yes | C14 contrast |
| RouterBench | T1-05 | 2026-06-22 | None explicit — SentenceTransformer | Yes | C16 contrast |

**Next reads (optional):** IRT-Router [Song et al., 2025] for C17 detail; FrugalGPT already in `14` for C21.

---

## 6. Categories (for discussion section)

```text
Lexical           length, TTR, readability, TF-IDF
Structural        punctuation, option count, sentence count
Information-theoretic   text Shannon entropy, compression
Semantic (static) word embeddings — only if no LLM pass
```

Paper reports **which category** the chosen \(c(q)\) belongs to — not all categories.

---

## 7. Selection rubric (score each candidate 0–2)

| Criterion | 0 | 1 | 2 |
| --------- | - | - | - |
| Unsupervised | needs labels/model | grey area | pure \(q\) statistics |
| Routing-relevant target | human-IRT only (Lugoloobi) | mixed evidence | motivated for **model gap** |
| Cross-dataset | format-specific | partial | same code ARC/MMLU/BoolQ |
| Probe cost | >prefill | comparable | negligible |
| Literature precedent | none | indirect | explicit pre-inference routing |

**Pick:** highest total among candidates that pass all §3 hard requirements. **Tie-break:** simpler formula, fewer hyperparameters.

**Known tension (D05):** Best **lexical** proxies may predict human difficulty, not **weak–strong gap** on a fixed pool. A **negative RH1** ( \(c(q)\) uninformative ) is still a valid paper result — but the signal must be **defensible**, not arbitrary.

---

## 8. Leading hypotheses (post deep-read — not frozen)

Literature narrows the field:

| Candidate | Pros | Cons | Lit. score |
| --------- | ---- | ---- | ---------- |
| **Text Shannon entropy** (C10) | Info-theoretic; parallels model \(H\); no corpus; passes §3 | No routing paper anchor; may correlate with length | 6/10 |
| **Compression ratio** (C11) | Structural; cheap; not tested in Lugoloobi | Opaque; may correlate with length | 6/10 |
| **TF-IDF mean-IDF** (C04) | Strongest **published** unsupervised text baseline | Tracks human IRT (\(\rho\approx0.72\)), weak on model success (\(\rho\approx0.25\)–\(0.47\)); corpus hyperparameter | 8/10 lit., risky for RH1 |
| **Type-token ratio** (C03) | Beyond raw length | Untested; lexical family | 6/10 |

**Unlikely winner:** raw length (D07; Lugoloobi \(\rho=0.15\) vs human IRT).

**Contrast papers (RouteLLM, Hybrid, RouterBench):** confirm Tier-1 uses **learned** query representations, not explicit \(c(q)\) — strengthens motivation for testing one interpretable unsupervised statistic.

**Recommended D46 process:** (1) pick C10 or C11 as primary (cleanest §3 + info-theoretic story), **or** C04 if you want the published lexical ceiling; (2) run ARC n=50 quick correlation vs \(y^{\text{opp}}\) before n=200 freeze.

---

## 9. Experiment structure (D47 — one hypothesis per study)

| Paper study | Hypothesis | Question | Signals |
| ----------- | ---------- | -------- | ------- |
| **I** | RH1 | Model-independent characterization | representative \(c(q)\) |
| **II** | RH2 | Model-dependent characterization | \(H\), \(m\) |
| **III** | RH3 | Complementarity | ΔAUROC within + cross-family |
| **IV** | RH4 | Routing evaluation + cost–quality | P(opp \| validated features) |

**Supplement:** §5.4 cross-dataset interpret — not a fifth hypothesis.

**Not in v1:** graph features, lexical suite, paraphrase, hidden-state probes — taxonomy only.

---

## 10. Pipeline (implementation — not science)

See `../experiments/README.md` and `10_experiment_registry.md` for scripts and artifact paths.

---

## 11. Decisions log

| ID | Status | Content |
| -- | ------ | ------- |
| D05 | Superseded for **timing** | Defer arbitrary s(q) — still valid warning on length/TF-IDF |
| D45 | **Active** | RH1: evaluate representative 𝒮_indep signal; survey before freeze |
| D46 | **Pending** | Lock exact \(c(q)\) formula in `05` §8 + `09` after implementation |

---

## 12. This week's priority order

1. ~~**Complete §5 survey**~~ ✅ deep-read 4 papers → `18` §5
2. **Score candidates** (§7 + §5.4) → **pick one** → **D46 in `09`**
3. **Spec in `05` §8** + minimal implementation
4. C2 screen MMLU/BoolQ (parallel)
5. **ARC n=200** with all **three** signals in merge CSV

**Do not** add Winogrande loader, routing, or extra lexical features until \(c(q)\) is frozen.
