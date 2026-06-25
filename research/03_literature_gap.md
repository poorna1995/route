# Literature Gap

> **Scientific decision this document supports:**  
> Does existing literature already solve our research question?

> Related: `00_research_question.md` · `02_research_hypotheses.md` · `04_signal_design.md` · `09_decision_register.md`

---

## 1. Purpose

This document evaluates existing research to determine whether the proposed research question remains open, identifies the limitations of current approaches, and refines the hypotheses and methodology accordingly.

It is **not** a paper summary, reading notebook, bibliography, or Related Work draft. Every section answers one question:

> **Does existing literature already solve my research question?**

The literature review **validates** the research question defined in `00` and `01`; it does not create it.

**Evidence rule:** Section 2 records **what each paper does**. Sections 3–5 record **what we conclude after reviewing the literature**. Every conclusion must be traceable to Section 2.

---

## 2. Literature Mapping

### 2.0 Summary table

Compact index. Detail in §2.1–2.10.

| Paper | Problem | Method | Routing information used | Supervised? | Pre-inference? | Model-dependent? |
| ----- | ------- | ------ | ------------------------ | ----------- | --------------- | ---------------- |
| RouteLLM (Ong et al., 2024) | Minimize cost while meeting a quality target when routing between strong and weak LLM classes | Win predictor P(strong wins \| q) + cost threshold α; one LLM per query | Query text → embedding or classifier (SW ranking, matrix factorization, BERT, causal LLM) | Yes | Yes | No |
| FrugalGPT (Chen et al., 2023) | Maximize task performance subject to a budget over multiple LLM APIs | LLM cascade: sequential APIs; stop when generation score exceeds τ | DistilBERT scorer g(q, answer) on query + generated output | Yes | No | No |
| GraphRouter (Feng et al., 2025) | Recommend LLM per query balancing effect and cost across tasks | Inductive heterogeneous GNN; edge prediction query–LLM | PLM (BERT) embeddings of task/query/LLM descriptions; edge features from past performance + cost | Yes | Yes | No |
| Hybrid LLM (Ding et al., 2024) | Reduce inference cost while maintaining response quality (small vs large model) | DeBERTa-style encoder on query; threshold on p_w(x); one LLM per query | Query text only | Yes | Yes | No |
| RouterBench (Hu et al., 2024) | Standardize evaluation of multi-LLM routers (not a router) | 405k precomputed outcomes; compares router families | Evaluated: prompt embeddings (KNN, MLP) or FrugalGPT-style cascade | N/A | — | — |
| Smoothie (Guha et al., 2024) | Label-free routing: select best LLM output per sample without human labels | Weak-supervision graphical model over output embeddings; Smoothie-Global / Local / Train | **Post-generation** output embeddings from all pool LLMs; SentenceBERT | **No** | No (post-gen) | No |
| CASCAL / RGD (Niu et al., 2026) | Annotation-free router training when no in-domain ground truth | Consensus-weighted voting + hierarchical clustering on generated queries | **Post-generation** pool responses; confidence-weighted majority; query embeddings | **No** (consensus proxy) | No (post-gen on train) | No |
| UniRoute (Jitkrittum et al., 2025) | Dynamic routing to unseen LLMs without router retraining | K-means prompt clusters + per-cluster LLM error vector Ψ(h) | Unsupervised prompt clusters; **supervised** per-cluster validation errors for Ψ(h) | **Partial** | Yes | No |
| CP-Router (Su et al., 2025) | Route between standard LLM and LRM to save tokens | Conformal prediction set size on MCQ logits; FBE adaptive α | **Pre-answer** MCQ option softmax → CP set size | **No** (CP cal only) | Yes | Yes (LLM side) |
| Mu RAG routing (Mu et al., 2025) | Unsupervised search-engine routing for RAG | Multi-source RAG upper bound → pseudo-labels; GTE-large + ListMLE | Query-only router; labels from BertScore + LLM coherence vs upper bound | **No** (pseudo-labels) | Yes | No (retriever routing) |

---

### 2.1 RouteLLM (Ong et al., 2024) — arXiv:2406.18665

**Problem.** Choose between a **strong** and **weak** LLM class per query to minimize inference cost while meeting a performance target (e.g., 90% of strong-model performance).

**Assumptions (from paper).**

- Access to **preference data** (q, l_{s,w}) comparing strong vs weak model classes on queries.
- Models cluster into **strong** and **weak** tiers; routing is **binary** between classes.
- **Ties** in preference labels count as wins for the **weak** model (Eq. routing setup).
- For cost analysis, routers handle mostly **short, single-turn** prompts (Appendix D).
- Routers should be **economical** relative to full generation on the strong model.

**Routing information used.**

- At inference: **query text only** → P(strong wins | q) via SW ranking (query embedding similarity to training queries + Bradley–Terry), matrix factorization (query × model-class score), BERT classifier, or causal LLM classifier.
- Training on Chatbot Arena: **omits model response text**; keeps query + model identities + comparison label.
- Augmentation: MMLU golden labels; GPT-4-judge labels on Nectar queries (requires generating weak-model responses).

**Supervision.**

- Human pairwise preferences (Chatbot Arena, ~80k battles).
- Augmented: automatically derived labels from MMLU; LLM-judge pairwise labels (GPT-4 judge on strong vs weak outputs).

**Limitation (paper-stated, §6).**

- Evaluation benchmarks may **differ from real-world query distributions** (in-domain augmentation helps).
- **Two-model-class** setting only; extension to multiple models left for future work.
- No single best router architecture; performance varies across routers **without clear explanation**.
- Latency and holistic router selection not fully addressed.

---

### 2.2 FrugalGPT (Chen et al., 2023) — arXiv:2305.05176

**Problem.** Use multiple commercial LLM APIs within a **budget** while maximizing task performance on natural-language query answering.

**Assumptions (from paper).**

- Task has a **correct answer** a for each query q; reward r(a, â) measures alignment.
- **Generation scorer** g(q, answer) ∈ [0,1] correlates with correctness (implemented as DistilBERT regression).
- Cascade length **3** in experiments; training and test data from **same or similar distribution**.
- Learning the cascade is a **one-time upfront cost** worthwhile when deployment volume exceeds training set size.

**Routing information used.**

- **Post-generation:** g(q, f_{L_i}(q)) after each API call in the ordered list L.
- LLM router selects API **order** L and thresholds τ; not query-only prediction before any generation.

**Supervision.**

- **Task labels** (ground-truth answers) on HEADLINES, OVERRULING, COQA to train the generation scorer and optimize cascade order/thresholds under budget constraint.

**Limitation (paper-stated, §5).**

- Requires **labeled examples** to train cascade; distribution shift hurts performance.
- Cascade learning itself requires compute (one-time).
- Does not address latency, fairness, privacy, or environmental impact.
- Paper presents a **vision** (prompt adaptation, approximation, cascade); cascade is the main empirical instantiation.

---

### 2.3 GraphRouter (Feng et al., 2025) — arXiv:2410.03834, ICLR 2025

**Problem.** Select an appropriate LLM per user query (with implied **task**) balancing **performance (effect)** and **cost** across a growing model pool.

**Assumptions (from paper).**

- **Historical interaction records** exist: task, query, selected LLM, response, measured performance, cost.
- Best LLM per training query is identifiable from **observed performance**; other query–LLM edges labeled 0, best labeled 1.
- Task and LLM nodes initialized from **GPT-4o-generated descriptions** encoded by BERT; query nodes from query text via same PLM.
- Standard split: **70% / 10% / 20%** train/val/test by query; new-LLM tests use **80-query** auxiliary interaction data (paper §3).

**Routing information used.**

- At inference: **heterogeneous GNN** over task, query, LLM nodes; **edge prediction** (dot product of query–task and LLM embeddings) — no live forward pass on each candidate LLM.
- LLM-query edge features initialized from **past performance and cost** concatenated.

**Supervision.**

- **Supervised edge labels** from historical query–LLM outcomes (best LLM per query in training set).
- Effect–cost trade-off controlled by user **preference weights** at recommendation time.

**Limitation (paper-stated, §6).**

- **Exploratory** validation of graph-based selection; richer graph structure (paths, LLM family taxonomy) not studied.
- Does not address **prompting-method selection** (CoT, ToT) or **multi-agent** simultaneous LLM selection.

---

### 2.4 Hybrid LLM (Ding et al., 2024) — arXiv:2404.14618, ICLR 2024

**Problem.** Route each query to a **small** or **large** LLM to save cost while maintaining response quality (hybrid inference / MLaaS).

**Assumptions (from paper).**

- **Exactly two models** L (large) and S (small); **one LLM call** per query at inference (not ensemble or cascade).
- Router (DeBERTa-style encoder) cost is **negligible** vs autoregressive decoding on L.
- **Smaller models are more efficient** than larger; cost advantage = fraction routed to S.
- Training labels from **BARTScore** comparing offline outputs q(S(x)) vs q(L(x)); probabilistic router uses **10 samples** per model per query.
- When S ≪ L in quality, **transformed labels** Pr[H(x) ≥ −t] with grid-searched t improve trainability (§3.3).

**Routing information used.**

- At inference: **query text only** → p_w(x) thresholded to route to S or L.
- No probe on candidate LLMs at routing time.

**Supervision.**

- **Offline-generated responses** from S and L on training queries; labels from BARTScore (deterministic, probabilistic, or transformed variants).

**Limitation (paper-stated, §5).**

- Router uses **query inputs only**; authors propose **task labels** and richer signals as future work.
- **Two-model** routing only; N-model and load balancing open.
- **Fixed model pair and data distribution** at train/test; OOD generalization to new pairs/distributions not solved.
- BARTScore (or better metrics) limits router quality.

---

### 2.5 RouterBench (Hu et al., 2024) — arXiv:2403.12031

**Problem.** No standardized way to evaluate **cost–quality** of LLM routing systems; proliferation of models makes selection hard.

**Assumptions (from paper).**

- **Precomputed** outputs o_i^j and quality q(o_i^j), cost c(o_i^j) for each (prompt, model) pair enable offline router evaluation.
- Routers compared in **cost–quality plane**; **AIQ** summarizes non-decreasing convex hull over cost range.
- **Oracle router:** always picks best-performing model per prompt; tie-break **cheapest**.
- Predictive routers (KNN, MLP): **70% train / 30% test** per task; SentenceTransformer prompt embeddings.

**Routing information used (evaluated baselines, not RouterBench itself).**

- **Predictive:** prompt embedding → predicted performance P_{ij}; route to argmax λ·P − cost.
- **Non-predictive (cascade):** FrugalGPT-style sequential generation + scoring (with optional error rate ε in simulation).

**Supervision.**

- Benchmark provides **labeled outcomes** per (prompt, model); predictive routers trained on held-out performance from same benchmark.

**Limitation (paper-stated, §6).**

- Evaluates **performance and dollar cost only** (not latency, throughput, etc.).
- **Subset** of LLMs and tasks; will expand in future iterations.
- Only **predictive and cascading** routers evaluated; RAG eval limited to models with **inherent retrieval**; two-stage retriever+LLM routing not covered.
- Simple predictive routers **do not consistently beat Zero router** across all tasks (§5.1).

---

### 2.6 Smoothie (Guha et al., 2024) — arXiv:2412.04692, NeurIPS 2024

**Problem.** Given an unlabeled test set and a pool of LLMs, route each sample to the LLM that yields the highest-quality generation **without** labeled data or routers trained on labels.

**Assumptions (from paper).**

- All \(m\) LLMs can be run on each test sample (or Smoothie-Train uses \(n_{\text{train}}=250\) held-out samples with precomputed generations).
- Output embeddings from a fixed model \(g_0\) (SentenceBERT `all-mpnet-base-v2`) capture semantic agreement with a latent “true” output.
- Weak-supervision Gaussian graphical model with diagonal covariance is adequate.

**Routing information used.**

- **Post-generation:** embeddings \(\lambda_i(x)=z_{g_0}([x,g_i(x)])\); pairwise distances estimate quality scores \(\hat{\theta}_i(x)\) (Algorithm 1).
- **Smoothie-Local:** sample-conditional scores via KNN smoothing (\(n_0=1\) best on mixed tasks).
- **Not used:** query-only routing, logprobs, cost.

**Supervision.**

- **None.**

**Limitation (paper-stated, §6).**

- Diagonal covariance assumes independent errors.
- No cost trade-off between models.
- Embedding-only semantics; requires \(n\times m\) generations (unless Smoothie-Train).

---

### 2.7 CASCAL / Routing with Generated Data (Niu et al., 2026) — arXiv:2601.09692, ACL 2026

**Problem.** Train LLM routers when **no ground-truth in-domain labels** exist; introduce **RGD** (routers trained on synthetic queries/answers from task descriptions).

**Assumptions (from paper).**

- Generator LLM can produce queries that differentiate models even when generated answers are unreliable.
- Tasks have **discrete answer classes** for consensus (BBEH tasks without discrete format removed).
- Pool responses available for all training queries.

**Routing information used.**

- **Consensus score** \(C_{i,j}\): confidence-weighted agreement across pool models on each query.
- **Hierarchical clustering** on query embeddings (Qwen3-Embedding-8B) over queries where each model matches majority — identifies skill niches.
- Inference: nearest task → nearest centroid → top-ranked models → consensus aggregation.

**Supervision.**

- **None** for correctness (consensus as proxy); training data is **synthetic** from generator LLMs.

**Limitation (from paper).**

- Query-answer routers degrade sharply with weak generators; ranking quality on large pools collapses with weak generators.
- Requires full pool inference on generated training queries; not pre-inference routing.

---

### 2.8 UniRoute (Jitkrittum et al., 2025) — arXiv:2502.08773, ICLR 2026 Poster

**Problem.** Route among **dynamic LLM pools** where new, previously unobserved models appear at test time without retraining the router.

**Assumptions (from paper).**

- Small labeled validation set \(S_{\text{val}}\) (~\(\mathcal{O}(10^3)\)) available; any new LLM can be evaluated on it efficiently.
- K-means on **training** prompt embeddings (Gecko 1B) yields representative clusters.
- Per-cluster average error on validation prompts approximates per-prompt expected loss.

**Routing information used.**

- **Prompt:** unsupervised K-means cluster membership \(\Phi_{\text{clust}}(x)\).
- **LLM:** supervised per-cluster error vector \(\Psi_{\text{clust}}(h)\in[0,1]^K\) from validation correctness.
- Score: \(\gamma_{\text{clust}}(x,h)=\Phi^\top\Psi(h)\); route with cost penalty \(\lambda\cdot c(h)\).

**Supervision.**

- **Partial:** prompt clustering unsupervised; **LLM features require labeled validation errors.**

**Limitation (from paper).**

- Not fully label-free; every new LLM needs labeled validation runs.
- Does not use live logprob probes on candidates.

---

### 2.9 CP-Router (Su et al., 2025) — arXiv:2505.19970

**Problem.** Dynamically route each prompt to a **standard LLM** or **large reasoning model (LRM)** to reduce token use while maintaining accuracy.

**Assumptions (from paper).**

- **MCQ format** (or hackable to multi-choice, e.g. GSM8K with 5 options).
- LLM exposes option logits; CP exchangeability on calibration set.
- Prediction set **size** correlates with whether LLM or LRM is preferable.

**Routing information used.**

- **Pre-answer:** softmax over MCQ options → CP nonconformity \(S(x,y)=1-f(y)\) → prediction set \(C(x)\).
- Route: small \(|C(x)|\) → LLM; large → LRM.
- **FBE** selects CP error rate \(\alpha\) to separate easy/hard prompts.

**Supervision.**

- **Training-free;** CP uses calibration split (not routing labels).

**Limitation (from paper).**

- Binary LLM↔LRM only; MCQ-focused; FBE grid search uses test-set set-size distribution.

---

### 2.10 Unsupervised Query Routing for RAG (Mu et al., 2025) — arXiv:2501.07793

**Problem.** Route user queries to the best **search engine** (Quark, Bing, Google) for RAG without gold answers on real user queries.

**Assumptions (from paper).**

- Multi-source retrieval (all engines) yields higher-quality RAG than any single engine (empirically validated in Table 1–2).
- BertScore + LLM pairwise coherence against multi-source upper bound approximates engine quality.

**Routing information used.**

- **Label construction (unsupervised):** single-source vs multi-source RAG responses; BertScore + coherence ranking.
- **Router inference:** query-only GTE-large → ListMLE-trained scores.

**Supervision.**

- **No human labels;** pseudo-supervision from upper-bound construction.

**Limitation (from paper).**

- Routes **retrievers**, not LLMs; label construction requires expensive multi-engine RAG runs offline.
- Not comparable to fixed LLM pool pre-inference probes.

---

## 3. Gap Analysis

Synthesis across §2. Scoped to Tier-1 deep read (original five + Smoothie, CASCAL, UniRoute) and selected Tier-2 (CP-Router, Mu RAG); other Tier-2 may refine.

| Observation | Evidence |
| ----------- | -------- |
| Four routing **methods** use **supervised** labels: preferences (RouteLLM), task correctness (FrugalGPT), historical outcomes (GraphRouter), BARTScore on offline generations (Hybrid LLM). | §2.1–2.4 |
| **Label-free LLM routing** exists: Smoothie (output-embedding agreement) and CASCAL (consensus on generated data) — but both require **post-generation** pool responses, not pre-inference probes. | §2.6–2.7 |
| UniRoute enables **pre-generation** routing decisions and generalization to unseen LLMs, but **LLM profiling uses supervised validation errors**; prompt clustering alone is unsupervised. | §2.8 |
| CP-Router uses **unsupervised pre-answer** MCQ logits (CP set size) for routing — closest to pre-inference uncertainty routing — but only **binary LLM↔LRM**, not multi-pool logprob probes. | §2.9 |
| Mu RAG routing constructs **unsupervised pseudo-labels** via multi-source upper bounds — different object (search engines) and post-retrieval signals. | §2.10 |
| At inference, RouteLLM, Hybrid LLM, GraphRouter, and RouterBench **predictive** baselines decide from **query and/or task representations** (and graph context) without live probes on each candidate LLM. | §2.1, §2.3–2.5 |
| FrugalGPT routing information is **post-generation**: g(q, answer) after each cascade step; may invoke multiple LLMs. | §2.2; RouterBench §2 |
| GraphRouter requires **past interaction data** with measured performance and cost; not unsupervised probe statistics. | §2.3 |
| Hybrid LLM and RouteLLM (judge augmentation) require **offline full generation** from pool models to construct training labels. | §2.1, §2.4 |
| Among papers reviewed in §2, we did not find a systematic study of **unsupervised, model-dependent, pre-inference logprob probes** (entropy, margin) as the **primary routing basis across a fixed multi-LLM pool**. | §2.0–2.10 |
| RouterBench shows routing can improve cost–quality but **simple predictive routers often fail to beat Zero router** on several tasks — routing is hard even with supervision. | §2.5 |

**Pattern:** Reviewed routing work improves cost–quality via supervised learned routers, graph-based outcome prediction, post-generation cascades, or label-free **post-generation** selection (Smoothie, CASCAL). UniRoute uses hybrid supervision; CP-Router uses pre-answer logits for **pairwise** LLM/LRM routing. Unsupervised model-dependent **prefill logprob probes** across a fixed pool remain uncharacterized.

### Research gap statement

The reviewed literature demonstrates that existing LLM routing approaches primarily rely on supervised learning, historical routing outcomes, or post-generation information to select models. While these methods improve the cost–quality trade-off, they do not systematically characterize **how informative unsupervised routing signals are for LLM selection before full model generation**. Consequently, unsupervised routing-signal characterization remains under-studied. This work addresses that gap by organizing model-independent and model-dependent signals (taxonomy 𝒮) and **empirically analyzing their informativeness** for routing within a fixed model pool.

---

## 4. Research Position

Existing methods show supervised routing can reduce cost (RouteLLM, Hybrid LLM, GraphRouter) or match best-model performance under budget (FrugalGPT), and RouterBench formalizes cost–quality evaluation — but all rely on supervision, historical outcomes, or post-generation scoring. **Label-free** routing (Smoothie, CASCAL) still requires **full pool generations** before selection. UniRoute routes pre-generation but needs **labeled validation errors** for new models. CP-Router routes on **pre-answer** uncertainty but only between **one LLM and one LRM**. **Therefore**, how informative **unsupervised pre-inference information** is for routing within a fixed multi-LLM pool remains unanswered. **This work investigates** what model-independent and model-dependent signals exist before expensive full inference and characterizes their predictive information for routing, as in `00_research_question.md`.

---

## 5. Effect on Research Design

Record decisions in `09_decision_register.md`; hypothesis revisions in `02`.

### Research question

The Tier-1 literature supports the relevance of the research question and does not provide evidence that it has already been answered. Refine "effective routing" as measurable **cost–quality trade-off** (RouterBench AIQ; RouteLLM CPT/APGR).

### Hypotheses

| Hypothesis | Effect |
| ---------- | ------ |
| RH1, RH2 | Strengthened — supervised and query-only Tier-1 methods unchanged; label-free Smoothie/CASCAL are **post-generation**, not unsupervised pre-inference probes |
| RH3 | Unchanged — complementarity not tested in reviewed routing papers |
| RH4 | Strengthened but tempered — RouteLLM/Hybrid LLM/CASCAL show gains; RouterBench shows simple routers often **do not beat Zero** |
| H1 | Hybrid LLM §5 notes query-only routing as limitation; suggests richer pre-query signals — supports studying unsupervised s(q), distinct from their supervised BERT router |
| H2, H3 | Entropy/margin not used as routing probes in Tier-1; remain candidates (Tier-2 calibration literature separate) |
| H4 | Not studied for routing in Tier-1 |

### Signals

| Signal | Tier-1 deep-read effect |
| ------ | ----------------------- |
| Query complexity | Hybrid LLM uses difficulty via **supervised** router; authors call for task labels / richer signals (§5 future work) — motivates unsupervised s(q), not duplicate of Hybrid LLM |
| Entropy / margin | **Not identified** as primary routing information in Tier-1 routing papers. CP-Router (T2-05) uses **CP set size** from MCQ logits (pre-answer), not entropy/margin across a pool. *Tier-2 calibration literature separate.* |
| Paraphrase stability | Not in Tier-1 routing papers |
| Label-free routing | Smoothie (embeddings), CASCAL (consensus) — **contrast** for Related Work; both post-generation |

Update `04_signal_design.md` Supporting literature and Part H.

### Model pool

- Multiple capability tiers required (RouteLLM strong/weak classes; RouterBench 11-model spread).
- Logprob access for entropy/margin — pool frozen in `MASTER.md` §5.
- Hybrid LLM: two-model focus; our work extends to **fixed pool > 2** (aligned with `01` scope).

### Evaluation

| Literature contribution | Use in this work |
| ----------------------- | ---------------- |
| RouterBench Oracle (best per prompt; tie-break cheapest) | Align `07_oracle_definition.md` |
| RouterBench Zero router + AIQ | Baseline / metric (adapt) |
| RouteLLM CPT, APGR | Secondary cost–quality framing |
| RouterBench finding: predictive routers vs Zero | Sets bar — probe-based router must beat simple baselines |
| RouteLLM, Hybrid LLM, FrugalGPT, GraphRouter | Related-work baselines — supervised/post-gen, not probe-signal methods |
| Smoothie, CASCAL | Related-work — **label-free** but post-generation pool responses |
| UniRoute | Related-work — dynamic pool; hybrid supervision |
| CP-Router | Related-work — pre-answer uncertainty routing (LLM↔LRM contrast) |

---

## 6. Reading Tracker

| Paper | Status |
| ----- | ------ |
| RouteLLM (Ong et al., 2024) | Deep read |
| FrugalGPT (Chen et al., 2023) | Deep read |
| GraphRouter (Feng et al., 2025) | Deep read |
| Hybrid LLM (Ding et al., 2024) | Deep read |
| RouterBench (Hu et al., 2024) | Deep read |
| Smoothie (Guha et al., 2024) | Deep read |
| CASCAL / RGD (Niu et al., 2026) | Deep read |
| UniRoute (Jitkrittum et al., 2025) | Deep read |
| CP-Router (Su et al., 2025) | Deep read |
| Unsupervised RAG Query Routing (Mu et al., 2025) | Deep read |
