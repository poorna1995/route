# Literature record — models, datasets, experiments

> **Role:** Factual catalog of routing and uncertainty papers — models, datasets, signals, supervision, results.  
> **Not:** gap analysis or Related Work prose (write from [`program.md`](program.md) §1).  
> **Program:** [`program.md`](program.md) · [`nomenclature.md`](nomenclature.md)

---

## 1. Reading status

| Group | Paper | arXiv / venue | Status |
| ----- | ----- | ------------- | ------ |
| Routing | RouteLLM — Ong et al. | [2406.18665](https://arxiv.org/abs/2406.18665) · ICLR 2025 | Deep read |
| Routing | FrugalGPT — Chen et al. | [2305.05176](https://arxiv.org/abs/2305.05176) · TMLR 2024 | Deep read |
| Routing | Hybrid LLM — Ding et al. | [2404.14618](https://arxiv.org/abs/2404.14618) · ICLR 2024 | Deep read |
| Routing | GraphRouter — Feng et al. | [2410.03834](https://arxiv.org/abs/2410.03834) · ICLR 2025 | Deep read |
| Routing | RouterBench — Hu et al. | [2403.12031](https://arxiv.org/abs/2403.12031) · ICML 2024 | Deep read |
| Routing | Smoothie — Guha et al. | [2412.04692](https://arxiv.org/abs/2412.04692) · NeurIPS 2024 | Deep read |
| Routing | CASCAL / RGD — Niu et al. | [2601.09692](https://arxiv.org/abs/2601.09692) · ACL 2026 | Deep read |
| Routing | UniRoute — Jitkrittum et al. | [2502.08773](https://arxiv.org/abs/2502.08773) · ICLR 2026 Poster | Deep read |
| Uncertainty | Language Models (Mostly) Know What They Know — Kadavath et al. | [2207.05221](https://arxiv.org/abs/2207.05221) | Deep read |
| Uncertainty | Uncertainty Calibration of Aligned LMs — He et al. | [2310.11732](https://arxiv.org/abs/2310.11732) | Deep read |
| Uncertainty | Chat LLM Probabilities Miscalibrated but Predict Correctness — Plaut et al. | [2402.13213](https://arxiv.org/abs/2402.13213) | Deep read |
| Uncertainty | Semantic Uncertainty — Kuhn et al. | [2302.09664](https://arxiv.org/abs/2302.09664) | Deep read |
| Uncertainty | UCCI — Kotte et al. | [2605.18796](https://arxiv.org/abs/2605.18796) | Deep read (routing-adjacent) |
| Uncertainty | CP-Router — Su et al. | [2505.19970](https://arxiv.org/abs/2505.19970) | Deep read (routing-adjacent) |
| Uncertainty | LLMs Encode Their Failures — Lugoloobi et al. | [2602.09924](https://arxiv.org/abs/2602.09924) | Deep read (contrast) |
| Uncertainty | Unsupervised RAG Query Routing — Mu et al. | [2501.07793](https://arxiv.org/html/2501.07793) | Deep read (retriever routing) |
| Uncertainty | Farquhar, Self-Consistency, AutoMix, Guo, Kumar CP | — | Not read |

---

## 2. Master comparison — routing vs signal papers

| Paper | Primary task | # models (typical) | Supervision | Signal timing | Uses H / margin as routing basis? | One LLM call / query? |
| -- | ----- | ------------ | ------------------ | ----------- | ------------- | --------------------------------- | --------------------- |
| RouteLLM | Strong vs weak class routing | 2 at eval (GPT-4, Mixtral-8x7B); 64 in train prefs | Human prefs + MMLU + GPT-4 judge | Pre-generation (query-only) | No | Yes |
| FrugalGPT | Budget-constrained QA | 12 APIs, cascade length 3 | Task labels → DistilBERT scorer | **Post-generation** per step | No | No (sequential cascade) |
| Hybrid LLM | Small vs large routing | 2 per experiment | BARTScore on offline gens | Pre-generation (query-only) | No | Yes |
| GraphRouter | Multi-LLM + multi-task | 10 LLMs | Historical query–LLM outcomes | Pre-generation (GNN, no live LLM probe) | No | Yes |
| RouterBench | Router **benchmark** | 11 (+ 3 RAG-only) | Precomputed outcomes | N/A (offline eval) | No (entropy only in cited cascade baselines elsewhere) | Depends on router |
| Kadavath | Self-eval / calibration | 800M–52B | P(IK) trained; P(True) uses samples | MC: pre-answer logits; open: needs draft | No (uncertainty study) | Yes |
| He | MC calibration | Llama 7B–70B, Vicuna, Llama-2-Chat | None for logit extraction | Pre-answer (MCQ logits) | No | Yes |
| Plaut | MC correctness prediction | 15 chat + base pairs | ~20 pts/dataset for abstention demo only | Pre-answer (MSP, max logit) | No | Yes |
| Kuhn | NLG uncertainty | OPT 2.7B–30B | Unsupervised clustering | **Multi-sample** generation | No | No (M samples) |
| UCCI | 4B↔12B cascade | 2 | Isotonic cal on held-out set | During small-model **generation** (token margin) | Margin **calibrated** for cascade | No (cascade) |
| Lugoloobi | Probe routing | 5–7 in routing demos | **Supervised** linear probes on activations | **Pre-generation** (hidden states) | No (activations, not logprobs) | Yes |
| Smoothie | Multi-LLM output selection | 4–11 per ensemble | **None** (weak-supervision graphical model) | **Post-generation** (all pool outputs) | No (output embeddings) | No (needs m gens/query) |
| CASCAL / RGD | Multi-LLM routing + aggregation | 6 per pool (2 pools) | **None** for correctness (consensus proxy) | **Post-generation** on training queries | No | No (pool responses on train) |
| UniRoute | Dynamic pool cost–quality routing | 30+ unseen at test | **Partial** — prompt K-means unsupervised; **Ψ(h) needs labeled val errors** | **Pre-generation** decision | No | Yes |
| CP-Router | LLM ↔ LRM binary route | 2 (pair) | **None** (CP; calibration split only) | **Pre-answer** (MCQ option logits) | Partial (CP set size, not H/margin) | Yes (one model forward for route) |
| Mu RAG routing | Search-engine selection for RAG | 3 engines | **None** for pseudo-labels (upper-bound construction) | **Pre-generation** at inference | No | Yes |

**Gap (this project):** Prior routing work is largely **supervised** or **post-generation**. We study **unsupervised** signals from the advisor proposal — query **complexity**, per-model **entropy**, **paraphrase stability** — combined via rules or learned weights on a fixed LLM pool. Closest neighbors (CP-Router, Lugoloobi et al.) use different supervision or activation probes; our story is professor-aligned unsupervised routing + parameter weighting.

---

## Routing papers

### Routing papers — summary

| ID | Training / construction data | Evaluation data | Eval model pair or pool | Router / method |
| -- | --------------------------- | --------------- | ----------------------- | --------------- |
| Chatbot Arena ~80k → 65k; + MMLU val ~1.5k; + Nectar/GPT-4 judge ~120k | MMLU, MT-Bench, GSM8K | GPT-4-1106-preview vs Mixtral-8x7B | SW ranking, MF, BERT, Llama-3-8B |
| HEADLINES, OVERRULING, COQA train splits | Same, test splits | 12 commercial APIs in learned cascade | DistilBERT g(q, answer), length-3 cascade |
| MixInstruct 10k train (+10 responses/model/query) | MixInstruct 5k test | e.g. Llama-2 7b↔13b, FLAN-T5, GPT-3.5-turbo | DeBERTa-v3-large |
| 600 queries × 4 tasks × 10 LLMs interaction graph | Same split 70/10/20 | 10 LLMs (Table 3.3 below) | Heterogeneous GNN + BERT |
| 405k precomputed (prompt, model) outcomes | Same benchmark | 11 LLMs × 8 datasets | KNN, MLP, Zero, cascade baselines |
| Unlabeled \(\mathcal{D}_{\text{test}}\); all \(m\) model generations per sample | 7 NLG + AlpacaEval + MixInstruct + GSM8K | 3B (4 LLMs), 7B (5 LLMs) ensembles | Smoothie-Global / Smoothie-Local (Algorithm 1) |
| RGD: synthetic \((\hat{q},\hat{a})\) from task descriptions; 5k/domain/generator | MMLU-Pro, MedMCQA, SuperGPQA, BBEH | Pool-Large (6), Pool-Small (6) | CASCAL (consensus + hierarchical clustering) |
| Train split + labeled \(S_{\text{val}}\) for per-cluster errors | EmbedLLM, RouterBench, Math+Code, SPROUT, Chatbot Arena | Train/test LLM pools disjoint; >30 unseen LLMs | UniRoute cluster-based (+ learned cluster map) |

---

### RouteLLM (Ong et al., 2024/2025)

| Field | Detail (from paper) |
| ----- | ------------------- |
| **Problem** | Binary route between strong and weak LLM classes; minimize cost at performance target |
| **Formulation** | P(strong wins \| q) + cost threshold α → route to M_weak or M_strong |
| **Training — primary** | ~80k Chatbot Arena battles; pruned to **65k** pairwise comparisons, **64 models**, prompts ≥16 chars |
| **Model tiers (RouteLLM paper)** | Models clustered into **10 tiers** (Chatbot Arena leaderboard + DP); strong = tiers 1–2, weak = tier 3 |
| **Training — augmentation A** | MMLU **validation** ~**1,500** MCQ; labels from comparing M_s, M_w to golden answer |
| **Training — augmentation B** | Nectar queries with GPT-4 responses; Mixtral-8x7B weak responses; GPT-4 judge → ~**120k** pairs (~**$700**) |
| **Router architectures** | (1) SW ranking + Bradley–Terry, (2) matrix factorization, (3) BERT_base full FT, (4) **Llama-3-8B** causal classifier |
| **Query embedding** | OpenAI `text-embedding-3-small` (SW, MF) |
| **Evaluation datasets** | **MMLU** (14,042), **MT-Bench** (160), **GSM8K** (1k+); cross-contamination check reported |
| **Evaluation pair** | Strong: **`gpt-4-1106-preview`**; weak: **Mixtral-8x7B** |
| **Metrics** | CPT, APGR, PGR, % calls to strong model |
| **Reported result** | >**2×** cost reduction on public benchmarks with minimal quality loss (abstract) |
| **Inference signal** | Query text only — **no** per-model prefill probe |
| **Paper-stated limits** | Two-class only; eval ≠ deployment distribution; no single best router; latency not fully addressed |

---

### FrugalGPT (Chen et al., 2023)

| Field | Detail |
| ----- | ------ |
| **Problem** | Maximize task performance subject to API budget |
| **Strategies (vision)** | Prompt adaptation, LLM approximation, **LLM cascade** (main experiment) |
| **Cascade length** | **3** APIs per learned strategy |
| **Scorer** | **DistilBERT** regression → g(q, answer) ∈ [0,1] |
| **APIs (12, Table 1)** | See API table below |
| **Datasets** | See dataset table below |
| **Supervision** | Ground-truth answers on train split; optimize order L and thresholds τ under budget |
| **Reported cost savings (match best single API)** | HEADLINES **98.3%**; OVERRULING **73.3%**; COQA **59.2%** |
| **Inference signal** | **Post-generation** after each cascade step |
| **Paper-stated limits** | Labeled examples required; distribution shift hurts; cascade learning is upfront cost; no latency/fairness/privacy |

**Table 3.2a — FrugalGPT commercial APIs (12)**

| Provider | API | Size (B) | Cost per 10M input tokens (USD) | Cost per 10M output tokens (USD) |
| -------- | --- | -------- | --------------------------------- | -------------------------------- |
| OpenAI | GPT-Curie | 6.7 | 2 | 2 |
| OpenAI | ChatGPT | — | 2 | 2 |
| OpenAI | GPT-3 | 175 | 20 | 20 |
| OpenAI | GPT-4 | — | 30 | 60 |
| AI21 | J1-Large | 7.5 | 0 | 30 |
| AI21 | J1-Grande | 17 | 0 | 80 |
| AI21 | J1-Jumbo | 178 | 0 | 250 |
| Cohere | Xlarge | 52 | 10 | 10 |
| ForeFrontAI | QA | 16 | 5.8 | 5.8 |
| Textsynth | GPT-J | 6 | 0.2 | 5 |
| Textsynth | FAIRSEQ | 13 | 0.6 | 15 |
| Textsynth | GPT-Neox | 20 | 1.4 | 35 |

*Costs retrieved March 2023 per paper Table 1.*

**Table 3.2b — FrugalGPT datasets**

| Dataset | Domain | Size | Few-shot examples in prompt |
| ------- | ------ | ---- | ----------------------------- |
| HEADLINES | Finance (gold trend from news titles) | 10,000 | 8 |
| OVERRULING | Law (is sentence an overruling?) | 2,400 | 5 |
| COQA | Passage reading (adapted to direct QA) | 7,982 | 2 |

---

### Hybrid LLM (Ding et al., 2024)

| Field | Detail |
| ----- | ------ |
| **Problem** | One LLM call per query: route to small S or large L |
| **Dataset** | **MixInstruct** (Jiang et al., 2023) |
| **Split** | **10k** train (uniform sample), **5k** val, **5k** test |
| **Offline generations** | **10 responses** per model per training query |
| **Labels** | **BARTScore** comparing S(x) vs L(x); soft labels from 10 samples |
| **Router** | **DeBERTa-v3-large** (300M); 5 epochs; NVIDIA A100 80GB |
| **Router variants** | r_det (single sample), r_prob (10 samples), r_trans (relaxed threshold t\*) |
| **Quality metric** | BARTScore |
| **Cost metric** | **Cost advantage** = fraction routed to small model |
| **Inference signal** | Query text → p_w(x) threshold — **no** LLM probe |
| **Paper-stated limits** | Query-only; two models; fixed pair/distribution; BARTScore ceiling |

**Table 3.3a — Hybrid LLM model pairs evaluated**

| Small (S) | Large (L) | Notes (from paper) |
| --------- | --------- | ------------------- |
| Llama-2 (7b) | Llama-2 (13b) | Small performance gap; **40%** cost adv., ~0% BART drop (r_trans) |
| Llama-2 (13b) | GPT-3.5-turbo | Medium gap; ~20% cost adv., ≤1% drop at 20% adv. |
| FLAN-T5 (800m) | Llama-2 (13b) | Large gap; r_trans needed; 40% adv. → 10.3% quality drop |

**Table 3.3b — Router latency vs LLMs (paper Table 2)**

| Model | Latency (s/query, mean ± SE) |
| ----- | ---------------------------- |
| Router (DeBERTa-v3-large) | 0.036 ± 0.002 |
| FLAN-T5 (800m) | 0.46 ± 0.039 |
| Llama-2 (7b) | 7.99 ± 0.15 |
| Llama-2 (13b) | 14.61 ± 0.27 |

---

### GraphRouter (Feng et al., 2025)

| Field | Detail |
| ----- | ------ |
| **Problem** | Select LLM per query balancing effect (performance) and cost across tasks |
| **Method** | Inductive heterogeneous **GNN**; edge prediction query–LLM |
| **Node init** | GPT-4o task/LLM descriptions + **BERT**; query text via same PLM |
| **Interaction data** | Per query: all LLMs answer → performance metric + token cost |
| **Split** | **70% / 10% / 20%** train/val/test by query |
| **New-LLM setting** | 6 seen LLMs + 4 new; **80-query** auxiliary interaction set (few-shot) |
| **Metrics** | Performance, Cost, **Reward** (user preference weights) |
| **Baselines** | Hybrid LLM (LLaMA-2 7b / Llama-3.1-Turbo 70b, RoBERTa); FrugalGPT (RoBERTa) |
| **Inference signal** | Graph embeddings from **past** interactions — no live candidate LLM forward pass |
| **Paper-stated limits** | Exploratory; richer graph structure not studied; no prompting-method routing |

**Table 3.4a — GraphRouter tasks and datasets**

| Task | Dataset | Metric | Queries in interaction set |
| ---- | ------- | ------ | -------------------------- |
| Hybrid QA | **Alpaca** (52k corpus; 52k self-instruct) | F1 | 600 |
| Math reasoning | **GSM8K** | Accuracy | 600 |
| Reading comprehension | **SQuAD** | (QA metric per paper) | 600 |
| Multi-doc summarization | **Multi-News** | F1 | 600 |

**Table 3.4b — GraphRouter LLM pool (10, Together API — paper Table 3)**

| LLM | Reported size | Cost per 1M tokens (paper units) |
| --- | ------------- | -------------------------------- |
| LLaMA-3 (7b) | 7b | 0.2 |
| Mixtral-8x7B | 56b | 0.6 |
| NousResearch | 34b | 0.8 |
| LLaMA-2 (7b) | 7b | 0.2 |
| Mistral-7b | 7b | 0.2 |
| LLaMA-3 (70b) | 70b | 0.9 |
| LLaMA-3-Turbo (8b) | 8b | 0.2 |
| LLaMA-3-Turbo (70b) | 70b | 0.9 |
| Llama-3.1-Turbo (70b) | 70b | 0.9 |
| Qwen-1.5 (72b) | 72b | 0.9 |

---

### RouterBench (Hu et al., 2024)

| Field | Detail |
| ----- | ------ |
| **Role** | Standardized **evaluation framework** + precomputed dataset |
| **Scale** | **405,467** samples; **11 models**, **8 datasets**, **64 tasks** |
| **Construction** | Run each LLM on each prompt; store output, quality q(·), cost c(·) |
| **Oracle router** | Best-performing model per prompt; tie-break **cheapest** |
| **Zero router** | NDCH over all LLMs — baseline |
| **Predictive routers** | **KNN**, **MLP** on **SentenceTransformer** embeddings; score = λ·P_ij − cost |
| **Train/test** | **70% / 30%** per task for predictive routers |
| **Metric** | Cost–quality plane, **AIQ** (area under NDCH) |
| **Key finding** | Simple predictive routers **do not consistently beat Zero** on all tasks |
| **Paper-stated limits** | Performance + dollar cost only; subset of models/tasks; RAG limited |

**Table 3.5a — RouterBench datasets (8 + RAG)**

| Category | Dataset |
| -------- | ------- |
| Commonsense reasoning | **HellaSwag**, **Winogrande**, **ARC-Challenge** |
| Knowledge | **MMLU** |
| Conversation | **MT-Bench** |
| Math | **GSM8K** |
| Coding | **MBPP** |
| RAG (practical) | **800** client search queries (sports, history, media, politics) + ground truth |

**Table 3.5b — RouterBench models (11 main benchmark)**

| Type | Model |
| ---- | ----- |
| Open source | Llama-70B-chat, Mixtral-8x7B, Yi-34B, Code Llama-34B, Mistral-7B, WizardLM-13B |
| Proprietary | GPT-4, GPT-3.5-turbo, Claude-instant-v1, Claude-v1, Claude-v2 |
| RAG-specific (3 additional) | sonar-small-online, sonar-medium-online (Perplexity), You.com API |

*Paper runs inference with 14 LLMs total; main benchmark aggregates **11** across 8 datasets.*

---

### Smoothie (Guha et al., 2024)

| Field | Detail (from paper) |
| ----- | ------------------- |
| **Authors / venue** | Neel Guha, Mayee F. Chen, Trevor Chow, Ishan S. Khare, Christopher Ré — **NeurIPS 2024** |
| **Problem** | Route each test sample to the best LLM in a pool **without** human labels or routers trained on labels |
| **Signals** | SentenceBERT embeddings \(\lambda_i(x)=z_{g_0}([x,g_i(x)])\); pairwise squared distances \(\|\lambda_i-\lambda_j\|^2\); quality scores \(\hat{\theta}_i(x)\) from weak-supervision Gaussian graphical model (Algorithm 1; adapted from Fu et al. 2022) |
| **Embedding model** | `all-mpnet-base-v2` SentenceBERT |
| **Variants** | **Smoothie-Global** (dataset-wide \(\hat{\delta}_{ij}\)); **Smoothie-Local** (KNN kernel smoothing, best \(n_0=1\) on mixed tasks); **Smoothie-Train** (\(n_{\text{train}}=250\) held-out gens — route test without generating on test) |
| **Routing rule** | \(\text{route}(x)=\arg\max_i \hat{\theta}_i(x)\) |
| **Supervision** | **None** |
| **Signal timing** | **Post-generation** — requires \(g_i(x)\) for all \(i\) (or Smoothie-Train held-out train gens) |
| **Not used** | Query-only features, logprobs, entropy, margin, cost model |
| **Single-task eval** | 7 NLG tasks (CNN/DM, XSum, SQuAD, TriviaQA, E2E, WebNLG, Def. Ext.); 3B ensemble (Pythia-2.8B, Gemma-2B, Incite-3B, Dolly-3B); 7B ensemble (Llama-2, Mistral, Vicuna, Gemma-7B, Nous Capybara); AlpacaEval; MixInstruct (11 LLMs); GSM8K (Gemma-7B, Phi-2, Llemma-7b + CoT) |
| **Multi-task eval** | Distr-Acc (SQuAD+TriviaQA+Def.Ext.); Distr-Rouge2 (CNN+XSum+WebNLG+E2E) |
| **Baselines** | Random; Best-on-Val (50 labeled samples); Labeled-kNN (50 val, 20 NN); PairRM (supervised reward model) |
| **Reported results** | Best-model identification 9/14 tasks; Spearman \(\rho\approx 0.72\) avg NLG; Smoothie-Local up to **~10 pts** over unsupervised baselines, up to **5 pts** over supervised (Table 2); AlpacaEval up to **27 pts** win-rate over Random; GSM8K Smoothie-Global **37.5%** vs Random **28.3%** |
| **Efficiency** | Closed-form — no SGD; ~2.14 s/1000 samples (Smoothie-Local, 7B multi-task); needs \(n\times m\) generations |
| **Paper-stated limits** | Diagonal covariance (independent errors); no cost trade-off; embedding-only semantics; requires all model outputs per sample |

---

### CASCAL / Routing with Generated Data (Niu et al., 2026)

| Field | Detail (from paper) |
| ----- | ------------------- |
| **Authors / venue** | Tianyi Niu et al. (UNC, Capital One, UT Austin) — **ACL 2026**; code: [RoutingGenData](https://github.com/tianyiniu/RoutingGenData) |
| **Problem** | Train LLM routers when **no in-domain ground-truth labels** exist; introduces **RGD** setting |
| **RGD setting** | Routers trained on synthetic \((\hat{q},\hat{a})\) from task descriptions via generator LLMs; held-out **real** test sets |
| **Signals** | **Consensus score** \(C_{i,j}=\sum_k \mathbb{I}(a_{i,j}=a_{i,k})\cdot Z_{i,k}\) (Z = dataset-normalized logprob confidence); **majority answer** \(a_i^{\text{maj}}\); **query embeddings** (Qwen3-Embedding-8B) for k-means skill clusters |
| **Method** | Identify \(Q^{\text{strong}}_{m,t}\) where model matches majority → k-means centroids per model → merge close centroids → rank models per cluster by avg consensus → inference: task match → nearest centroid → top-k models → consensus aggregation |
| **Variants** | CASCAL Top-1 (no aggregation); CASCAL-GT (uses generated answers as labels — query-answer router) |
| **Supervision** | **None** for correctness (consensus proxy); requires **discrete answer classes** |
| **Signal timing** | **Post-generation** — pool responses on generated training queries |
| **Model pools** | **Pool-Large (6):** GPT-OSS 120B, LLaMA-3.3 70B, Qwen-3 32B, GLM-4 32B, Exaone-4 32B, Gemma-3 27B; **Pool-Small (6):** Gemma-2 9B, GLM-4 9B, Yi-1.5 9B, Qwen-3 8B, Exaone-3.5 7.8B, DeepSeek-Math 7B |
| **Benchmarks** | MMLU-Pro, MedMCQA, SuperGPQA, BigBench-Extra-Hard (BBEH) |
| **Generators** | Real validation, Gemini-2.5-Flash, Qwen3-32B, Exaone-3.5-7.8B — **5,000 queries/domain** each |
| **Split** | 6:4 train–test stratified by task/subject |
| **Baselines** | Top-1 / Top-3 Vote (real val ceiling); LLMRank; Avengers (k=64); Smoothie-Train; Random |
| **Reported results** | Pool-Large Exaone: CASCAL **61.1%** vs LLMRank **57.1%**, Avengers **58.9%**; query-answer routers drop **8–10 pts** with weak generators vs CASCAL **~2.5 pts**; **+4.6%** absolute over best QA router on weak Exaone (abstract); filtered Exaone data **62.3%** Top-3 vs **63.6%** real validation |
| **Key analysis** | Weak generators: bad answer labels but useful queries; query-only more robust than query-answer; Kendall \(\tau\) ranking vs validation collapses for Pool-Large with weak generators |
| **Paper-stated limits** | Discrete outputs required; depends on generator quality; consensus needs full pool responses on training queries |

---

### UniRoute (Jitkrittum et al., 2025/2026)

| Field | Detail (from paper) |
| ----- | ------------------- |
| **Authors / venue** | Wittawat Jitkrittum et al. (Google) — **ICLR 2026 Poster** |
| **Problem** | **Dynamic routing** — generalize to **unseen LLMs** at test time without retraining router |
| **Signals — prompt** | \(\Phi_{\text{clust}}(x)\in\{0,1\}^K\): K-means on **training** prompt embeddings (Gecko 1B, 768-d) — **unsupervised** clustering step |
| **Signals — LLM** | \(\Psi_{\text{clust},k}(h)\): per-cluster **error rate** on validation prompts in cluster \(k\) — **supervised** (needs \((x,y)\) correctness on \(S_{\text{val}}\)) |
| **Routing score** | \(\gamma_{\text{clust}}(x,h)=\Phi_{\text{clust}}(x)^\top\Psi_{\text{clust}}(h)\); route: \(\arg\min_n[\gamma(x,h^{(n)})+\lambda\cdot c(h^{(n)})]\) |
| **Variant** | **Learned cluster map:** softmax cluster assignment from labeled training LLMs; \(\Psi_{\text{clust}}\) still from val errors |
| **K-NN special case** | RouterBench K-NN (Hu et al.) when \(K=N_{\text{val}}\) |
| **Supervision** | **Partial** — prompt clustering unsupervised; LLM representation requires labeled validation errors |
| **Signal timing** | **Pre-generation** routing decision; LLM profile built offline on \(S_{\text{val}}\) (~\(\mathcal{O}(10^3)\) prompts) |
| **Datasets** | EmbedLLM (112 LLMs), RouterBench (11), Math+Code, SPROUT o3-mini (15), Chatbot Arena |
| **Protocol** | Train/test LLM pools disjoint (e.g. EmbedLLM 2/3 train, 1/3 test); data 60/10/30; **400 trials** |
| **Metrics** | Deferral curve — area under curve, area to 50% cost, quality-neutral cost (QNC) |
| **Baselines** | K-NN, ZeroRouter, MLP, Matrix Factorization (static-pool oracles) |
| **Reported results** | Significant gains over K-NN on EmbedLLM (>30 unseen LLMs); beats ZeroRouter on four datasets; excess risk bound (Proposition 2) |
| **Paper-stated limits** | Every new LLM needs inference on labeled \(S_{\text{val}}\); not fully label-free |

---

## Uncertainty papers

### Uncertainty papers — summary

| ID | Uncertainty statistic | When computed | Supervision | Primary datasets | Primary models |
| -- | --------------------- | ------------- | ----------- | ---------------- | -------------- |
| P(True), P(IK), MC calibration | MC logits pre-answer; P(True) needs samples | P(IK) classifier trained | MMLU, BIG-Bench MC, TriviaQA, Lambada, GSM8k, HumanEval, … | 800M, 3B, 12B, **52B** |
| MSP / logits on MCQ | Pre-answer (MCQ format) | None for extraction | HellaSwag, OpenbookQA, TruthfulQA, LogiQA, MMLU, CivilComments, IMDB | Llama 7B–70B, Vicuna, Llama-2-Chat |
| MSP, max logit (+ margin, entropy in appendix) | Pre-answer (MCQ) | ~20 labeled pts/dataset for abstention only | ARC, HellaSwag, MMLU, TruthfulQA, WinoGrande | 15 chat LMs + base variants |
| Semantic entropy | After **M** sampled answers | Unsupervised NLI clustering | TriviaQA, CoQA | OPT 2.7B–**30B** |
| Token margin → isotonic → P(error) | During small-model generation | Cal + val labels | Production NER (75k queries) | **4B** vs **12B** instruct |
| Linear probe on activations | **Pre-generation** (prefill) | Success labels from rollouts | E2H-AMC, GSM8K, MATH, AIME, LiveCodeBench | Qwen2.5-Math, GPT-OSS-20B, DeepSeek-R1-Distill, … |
| CP prediction set size (MCQ option softmax) | **Pre-answer** (MCQ logits) | CP calibration split only | MMLU-STEM, GPQA, LogiQA, CN-Chemistry, GSM8K | Llama-3.1-8B, Qwen-2.5-14B + Distill-R1 |
| BertScore + LLM coherence vs multi-source RAG upper bound | Label construction post-RAG; query-only router at inference | Pseudo-labels (no human gold) | NLPCC-MH, CDQA, WebQA, SogouQA, PrivateQA | Quark/Bing/Google; Qwen2-max, GPT-4 |

---

### Kadavath et al. (2022)

| Field | Detail |
| ----- | ------ |
| **Question** | Can LMs evaluate validity of own claims? Predict which questions they answer correctly? |
| **Models** | Anthropic LM series: **800M, 3B, 12B, 52B** |
| **MC calibration** | BIG-Bench multiple-choice subtasks, **MMLU**, others; 5-shot where noted |
| **Sampling tasks** | **TriviaQA**, Lambada, **GSM8k**, Codex **HumanEval**, arithmetic, natural function synthesis |
| **P(True)** | Model proposes answer, then evaluates P(answer is true) |
| **P(IK)** | “I know” classifier — trained; tested ID and OOD |
| **Main findings** | Larger base models **well-calibrated** on MC; calibration improves with size; P(True) improves with capability; P(IK) partial OOD generalization |
| **Relevance** | Logprobs informative; P(True) is **not** pure prefill; chat vs base differs (see He et al., Plaut et al.) |

---

### He et al. (2023)

| Field | Detail |
| ----- | ------ |
| **Question** | Why are **aligned** chat LMs overconfident vs pre-trained? |
| **Models** | **Llama** (7B–70B); **Vicuna**; **Llama-2-Chat** |
| **Alignment pipelines studied** | **Alpaca-Farm** (Llama-1 7B, SFT+PPO); **Zephyr** (Mistral 7B, SFT+DPO) |
| **Synthetic alignment** | Llama-1 7B on synthetic MCQ (Lieberum et al. task); LoRA |
| **Datasets (7)** | HellaSwag, OpenbookQA, TruthfulQA, LogiQA, **MMLU**, CivilComments, IMDB |
| **Formats** | Zero-shot vs ICL; choice “A” vs “(A)” |
| **Core mechanism** | **Answer uncertainty** vs **format uncertainty** conflated during alignment |
| **Main findings** | All aligned LMs higher ECE than pre-trained; accuracy–calibration gap persists at scale |
| **Relevance** | Failure mode for **margin/MSP probes on chat models** under fixed MCQ-style prompts |

---

### Plaut et al. (2024)

| Field | Detail |
| ----- | ------ |
| **Question** | Are chat LLM MSPs calibrated? Do they still **predict correctness**? |
| **Setting** | Multiple-choice Q&A (HF leaderboard style); 4-bit quant on open models |
| **Datasets (5)** | **ARC-Challenge** (full 2,590), HellaSwag (6k sample), **MMLU** (6k sample), TruthfulQA (full 817), WinoGrande (6k sample); **GSM8K excluded** (not MCQ) |
| **Signals** | **MSP**, max logit; appendix: **margin**, **entropy** |
| **Metrics** | ECE, AUROC for correctness prediction |
| **Models (15 chat)** | Falcon 7B/40B; Llama 2 7B/70B; Llama 3.0/3.1 8B/70B; Mistral 7B v0.2; Mixtral 8x7B; SOLAR 10.7B; Yi 6B/34B; GPT-3.5 Turbo; GPT-4o (logprobs via API) |
| **Base models** | Same 13 open-weight families — base vs chat compared |
| **Main findings** | Chat MSP **miscalibrated**, does **not** improve with capability; MSP **predicts correctness** (AUROC), improves with Q&A accuracy; base models better calibrated |
| **Abstention** | ~**20** data points per dataset for threshold selection |
| **Relevance** | Supports **ranking** by margin without calibrated probabilities (ranking without calibration) |

**Table 4.3a — Plaut et al. open-weight model families**

| Family | Sizes evaluated |
| ------ | --------------- |
| Falcon | 7B, 40B |
| Llama 2 | 7B, 70B |
| Llama 3.0 / 3.1 | 8B, 70B |
| Mistral | 7B v0.2 |
| Mixtral | 8x7B |
| SOLAR | 10.7B |
| Yi | 6B, 34B |

---

### Kuhn et al. (2023)

| Field | Detail |
| ----- | ------ |
| **Method** | **Semantic entropy**: sample M answers → DeBERTa-large NLI clustering → entropy over meaning clusters |
| **Models** | **OPT** 2.7B, 6.7B, 13B, **30B** (headline results at 30B) |
| **Datasets** | **TriviaQA**, **CoQA** (free-form QA) |
| **Samples** | Typically **M ≈ 10** generations per question |
| **NLI model** | DeBERTa-large fine-tuned on **MNLI** |
| **Baselines** | Token entropy, length-normalized entropy, P(True), lexical similarity (Rouge-L) |
| **Correctness label** | Rouge-L > 0.3 vs reference (validated on 200 manual labels) |
| **Main findings** | Semantic entropy beats token entropy and P(True) on AUROC; scales with model size |
| **Relevance** | Token entropy **overcounts** when paraphrases share meaning; not single-pass prefill |

---

### UCCI (Kotte et al., 2025)

| Field | Detail |
| ----- | ------ |
| **Problem** | Cost-optimal **two-model cascade** with calibrated routing score |
| **Workload** | Production **named entity recognition**; **75,000** labeled queries; 6 entity types |
| **Models** | **4B** and **12B** instruction-tuned LMs on **H100** |
| **Signal** | **Token margin** aggregated over generation → **isotonic regression** → P(error) |
| **Policy** | Threshold on calibrated P(error) to escalate to large model |
| **Baselines** | Raw **entropy** thresholding, conformal routing, FrugalGPT-style learned threshold |
| **Results** | **31%** cost reduction (95% CI 27–35%) at micro-F1 = **0.91**; ECE **0.12 → 0.03** |
| **Supervision** | Calibration + validation labels required |
| **Not compared** | RouteLLM, Hybrid LLM (need preference/quality-gap labels) |
| **Relevance** | Raw margin weak; **calibrated** margin works for cascade — contrast to **uncalibrated** 𝒮 probes |

---

### Lugoloobi et al. (2026)

| Field | Detail |
| ----- | ------ |
| **Question** | Is success predictable from **pre-generation activations**? Can probes route across a pool? |
| **Signal** | **Supervised linear probes** on hidden states (not logprobs) |
| **Datasets** | **E2H-AMC** (human IRT + model rollouts), **GSM8K**, **MATH**, **AIME**, **LiveCodeBench** |
| **Rollouts** | K=50 (Qwen); K=5 (GPT-OSS-20B) for success-rate labels |
| **Models (evaluation pool)** | Qwen2.5-Math-1.5B/7B, Qwen2.5-1.5B, Qwen2.5-Coder-3B/7B, DeepSeek-R1-Distill-Qwen-7B, GPT-OSS-20B (low/med/high reasoning) |
| **Routing pool (demo)** | 5 models: Qwen2.5-Math-7B, DeepSeek-R1-Qwen-7B, GPT-OSS-20B × 3 reasoning levels |
| **Key findings** | Human IRT difficulty ≠ model difficulty (ρ human 0.83–0.87 vs model 0.40–0.64 on E2H-AMC); probe AUROC **degrades** with extended reasoning; routing up to **~70%** cost cut on MATH at matched accuracy |
| **Relevance** | Closest **prefill routing** neighbor — but **supervised activations**, not unsupervised logprob 𝒮 |

**Table 4.6a — Lugoloobi et al. probe AUROC snapshot (paper Table 2, selected rows)**

| Model | Decoding | Task acc. | Probe AUROC (avg) |
| ----- | -------- | --------- | ----------------- |
| Qwen2.5-Math-1.5B | Greedy | 0.84 | 0.83 |
| Qwen2.5-Math-7B | Greedy | 0.79 | 0.79 |
| GPT-OSS-20B | Reasoning: Low | 0.866 | 0.78 |
| GPT-OSS-20B | Reasoning: High | 0.920 | 0.64 |

---

### CP-Router (Su et al., 2025)

| Field | Detail |
| ----- | ------ |
| **Authors / venue** | Jiayuan Su et al. (Zhejiang U, HKU, Tsinghua, PKU, UIC) — arXiv:2505.19970 (preprint; no venue stated in source) |
| **Problem** | Route each prompt to **standard LLM** or **large reasoning model (LRM)** to reduce tokens while preserving accuracy (“overthinking”) |
| **Signals** | MCQ option **softmax probabilities** \(f(y)\); CP **nonconformity** \(S(x,y)=1-f(y)\); **prediction set** \(C(x)=\{y:S(x,y)\leq\hat{q}\}\); set **size** = uncertainty proxy; **FBE** (Full + Binary Entropy) for adaptive \(\alpha\) selection |
| **Routing rule** | Small \(|C(x)|\) (e.g. singleton) → **LLM**; large set → **LRM** |
| **Supervision** | **Training-free**; standard CP **calibration set** for quantile \(\hat{q}\) (not routing labels) |
| **Signal timing** | **Pre-answer** — single forward pass extracting MCQ logits |
| **Pairings** | Llama-3.1-8B + DeepSeek-R1-Distill-Llama-8B; Qwen-2.5-14B + DeepSeek-R1-Distill-Qwen-14B; also DeepSeek-V3 + DeepSeek-R1 on GPQA |
| **Benchmarks** | MMLU-STEM (elem / HS / college math), STEM-MCQA, GSM8K (open-ended → 5-choice hack), GPQA, LogiQA, CN-Chemistry |
| **Baselines** | LLM-only, LRM-only, Random routing, Top-1 prob, Response entropy, Dynathink (majority vote), Explicit self-awareness |
| **Metrics** | Accuracy, TRR (token reduction ratio), \(U_{\text{token}}\) (accuracy gain per token saved) |
| **Reported results** | Highest \(U_{\text{token}}\) on all 6 benchmarks (Llama pairing) and 5/6 (Qwen); GSM8K **77.9%** acc vs LRM **79.0%**, **32.9%** questions skip LRM; GPQA V3/R1: **54.7%** routed to R1, beats both standalone |
| **Relevance** | Closest published **pre-inference logit-based** router — but **binary LLM↔LRM**, not multi-pool entropy/margin probes |

---

### Unsupervised Query Routing for RAG (Mu et al., 2025)

| Field | Detail |
| ----- | ------ |
| **Authors / venue** | Feiteng Mu et al. (PolyU, Alibaba, Nankai) — arXiv:2501.07793 (preprint; no venue stated in source) |
| **Problem** | Route queries to best **search engine** (Quark, Bing, Google) for RAG **without** gold answers on real user queries |
| **Label construction (unsupervised)** | Single-source RAG \(r_m=\text{LLM}(q,\text{DOC}_m)\); multi-source upper bound \(r^*=\text{LLM}(q,\text{DOC}^*)\) with merged top \(k/M\) docs per engine; **BertScore** similarity + **LLM pairwise coherence** ranking → combined score \(s_m\) |
| **Router** | GTE-large encoder → scores \(\mathbf{p}=\mathcal{M}(q)\); **ListMLE** listwise ranking loss |
| **Inference** | **Query-only** — no search call at routing time |
| **Train data** | ~**110k** in-house real user queries (non-i.i.d. vs public test) |
| **Test datasets** | NLPCC-MH, CDQA, WebQA, SogouQA, PrivateQA |
| **Generation LLMs (label construction)** | Qwen2-max, GPT-4 |
| **Eval metric** | End-to-end RAG **Correctness** (Llama-index) |
| **Reported results** | Scalability with more unlabeled training data; better OOD generalization than supervised methods on public (q, answer) train data (§5.3) |
| **Relevance** | Label-free **pseudo-label construction** pattern; routes **retrievers**, not LLM pool; response-similarity signals, not pre-inference logprob probes |

---

## 5. Consolidated indexes

### 5.1 Dataset index (papers → benchmarks)

| Dataset | Routing papers | Uncertainty papers | This project |
| ------- | ------------- | ------------- | ---------------- |
| ARC-Challenge | RouterBench | Plaut | **Planned** |
| GSM8K | RouteLLM, GraphRouter, RouterBench | Kadavath, Lugoloobi, **Smoothie**, **CP-Router** | Planned |
| MMLU-Pro / MedMCQA / SuperGPQA | — | **CASCAL** | No |
| EmbedLLM / RouterBench (dynamic pool) | RouterBench | **UniRoute** | RouterBench eval only |
| NLPCC-MH / CDQA / WebQA / SogouQA | — | **Mu RAG routing (Mu RAG routing)** | No |
| MMLU | RouteLLM, RouterBench, Hybrid (via MixInstruct) | Kadavath, He, Plaut | No |
| MT-Bench | RouteLLM, RouterBench | — | No |
| HellaSwag | RouterBench | He, Plaut | No |
| Winogrande | RouterBench | Plaut | No |
| MBPP | RouterBench | — | No |
| MixInstruct | Hybrid LLM | **Smoothie** | No |
| Alpaca / SQuAD / Multi-News | GraphRouter | **Smoothie** | No |
| HEADLINES / OVERRULING / COQA | FrugalGPT | Kuhn (CoQA) | No |
| TriviaQA | — | Kadavath, Kuhn | No |
| MATH | — | Lugoloobi | Shortlisted, not in code |
| Production NER | — | UCCI | No |

### 5.2 Model index (frequently cited)

| Model / family | Papers |
| -------------- | ------ |
| GPT-4 (variants) | RouteLLM, FrugalGPT, RouterBench, GraphRouter (descriptions) |
| Mixtral-8x7B | RouteLLM, RouterBench, GraphRouter, Plaut |
| Llama-2 / Llama-3 | Hybrid LLM, GraphRouter, RouterBench, Plaut, He |
| GPT-3.5-turbo | Hybrid LLM, RouterBench, Plaut |
| DeBERTa / BERT / RoBERTa | Hybrid LLM, FrugalGPT, GraphRouter baselines, RouteLLM |
| OPT | Kuhn |
| Anthropic LMs 800M–52B | Kadavath |
| Qwen (1.5, 2.5-Math, 3) | GraphRouter; Lugoloobi; **CASCAL**; **CP-Router**; **this repo** |
| Llama-3.2-3B / Llama-3.1-8B Instruct | Primary pool (M1 frozen) | **CP-Router** Distill-Llama-8B |
| SentenceBERT / GTE / Gecko | **Smoothie**; **UniRoute** Gecko 1B; **Mu RAG routing** GTE-large |

### 5.3 Signal timing matrix

| Signal | Example papers | Pre-inference? | Needs generation? | Needs labels? |
| ------ | -------------- | -------------- | ----------------- | ------------- |
| Query embedding → router | RouteLLM, Hybrid, RouterBench-KNN | Yes (query-only) | No | Yes (prefs, BARTScore, outcomes) |
| g(q, answer) cascade scorer | FrugalGPT | No | Yes (partial/full) | Yes |
| GNN edge score | GraphRouter | Yes (no live LLM probe) | No (uses stored responses in train) | Yes |
| MC MSP / max logit | Plaut, He | Yes (first answer token) | No | No (eval labels only) |
| Unsupervised entropy (planned) | **This project** | Yes (model forward) | No | No |
| P(True) | Kadavath, Kuhn baseline | Partial | Yes (draft answer) | Optional samples |
| Semantic entropy | Kuhn | No | Yes (M samples) | No |
| Calibrated token margin | UCCI | During small-model gen | Yes | Yes (cal set) |
| Activation linear probe | Lugoloobi | Yes (prefill) | No | Yes (rollout labels) |
| Output embedding agreement | Smoothie | No | Yes (all pool outputs) | No |
| Consensus voting across pool | CASCAL | No | Yes (pool responses on train) | No |
| Per-cluster error vector Ψ(h) | UniRoute | Yes (route decision) | No (val profile offline) | Yes (val correctness) |
| CP prediction set size | CP-Router | Yes (MCQ logits) | No | CP cal split only |
| Multi-source RAG upper bound | Mu RAG routing | Yes (query-only router) | Yes (for label construction) | No (pseudo-labels) |

---

## Project use of this catalog

Literature **informs** design; it does not pick datasets by popularity alone.

| Do | Don't |
| -- | ----- |
| Check a setting appears in related routing / uncertainty work | Pick ARC only because one paper used it |
| Borrow patterns (same-family pool, MCQ probes) | Reproduce GPT-4 / 70B pools |
| Cite as supporting evidence | Treat citation count as selection criterion |

### Dataset candidates (methodology first)

| Priority | Dataset | Why | Literature support | Role |
| -------- | ------- | --- | ------------------ | ---- |
| Primary | **ARC-Challenge** | Objective MCQ; stable eval | RouterBench, Plaut et al. | Setup validation; in code |
| Secondary | **MMLU (1–2 subjects)** | Cross-domain MCQ | RouteLLM, RouterBench, Kadavath/He/Plaut | Generalization follow-on |
| Optional | HellaSwag / Winogrande | Same probe interface | RouterBench, He, Plaut | Optional |
| Defer | GSM8K | Parsing complexity | RouteLLM, GraphRouter, Kadavath, Lugoloobi | After primary lock |
| Not v1 | MATH | Hard reasoning benchmark | Lugoloobi et al. | Future |

### Model pool pattern

| Pattern | Literature example | This project |
| ------- | ------------------ | ------------ |
| Same-family weak ↔ strong | Llama-2 7B↔13B (Hybrid LLM); Llama 8B↔70B (Plaut) | Llama **3B↔8B** (primary); 1B↔3B ablation |
| Same-family alternative | Qwen pools in GraphRouter, Lugoloobi | Qwen 1.5B↔3B (optional) |
| Proprietary pairs | GPT-4 vs Mixtral (RouteLLM) | Out of scope |

### Llama family capability reference (M1)

Meta’s Llama line is **not one uniform size ladder** — capability tiers depend on **generation**. Sizes below are **text-only Instruct** models relevant to MCQ routing (vision/MoE omitted unless noted).

| Generation | Text-only sizes | Notes |
| ---------- | --------------- | ----- |
| **Llama 3.2** | **1B**, **3B** | Edge/lightweight; distilled from 3.1 8B/70B logits |
| **Llama 3.1** | **8B**, **70B**, **405B** | Main dense line; 128K context |
| **Llama 3** | **8B**, **70B** | Earlier 3.x (mostly superseded by 3.1) |
| **Llama 3.3** | **70B** only | Improved 70B checkpoint |
| **Llama 2** | **7B**, **13B**, **70B** | Classic routing pairs (Hybrid LLM 7B↔13B) |
| **Llama 4** | Scout, Maverick | MoE (~17B active each); multimodal — defer for v1 |

**Vision-only (out of scope v1):** Llama 3.2-Vision **11B**, **90B**.

**Cross-generation note:** **3B** is Llama **3.2**; **8B** is Llama **3.1**. Pair **3B → 8B** is “Meta Llama stack” but **not** same-release 3.2. The only native **same-generation 3.2 text step** is **1B → 3B**.

#### Selection criteria (M1 pool — logged 2026-06-25, advisor)

Choose the primary pair by **experimental validity**, not same-release purity. The paper tests whether unsupervised signals predict \(r(q)\) in a **fixed homogeneous pool** — not Llama generation comparisons.

| Criterion | Importance | Why |
| --------- | ---------- | --- |
| Same tokenizer / chat format | High | Avoid prompt confounds |
| Similar architecture (same vendor line) | High | Signals reflect capability, not family shift |
| Clear capability gap | **Very high** | Otherwise \(P(r(q)=1)\) is too rare (Gate D) |
| Affordable oracle (both models on every \(q\)) | **Very high** | Oracle dominates compute |
| Common in literature | Medium | Easier to justify |

**Same generation is not a selection criterion.**

#### Pair evaluation

| Pair | Pros | Cons | Role |
| ---- | ---- | ---- | ---- |
| **3B → 8B** (3.2-3B, 3.1-8B) | Meta ecosystem; shared instruct lineage; larger gap → more **opportunity**; feasible oracle | Cross-release label (not a scientific confound if tokenizer/chat aligned) | **Primary (ACL v1)** |
| **1B → 3B** (both 3.2) | Cheapest; same release/tokenizer | Gap may be too small → “no routing opportunity” failure mode | Pilot sanity check or **gap-size ablation** |
| **8B → 70B** (3.1) | Huge routing gap | Oracle cost; reproducibility; API/GPU burden | **Robustness only** (post-primary) |

**Do not cross vendor families in v1** (e.g. Llama + Qwen): tokenizer, alignment, and calibration confounds dominate.

**Paper wording (pool justification):** *We select a fixed pool of two instruction-tuned Llama models with a clear capability difference while maintaining a common architecture and tokenizer, minimizing confounds unrelated to model capability.*

**Terminology note:** **`homogeneous_pool`** (same vendor, architecture, tokenizer) vs **`heterogeneous_pool`** (cross-family). Gap size (3B→8B vs 8B→70B) is a **robustness axis**, not pool homogeneity. Canonical spec: [`program.md`](program.md) §0.7, [`experiments/m1/pool.frozen.yaml`](../experiments/m1/pool.frozen.yaml).

#### Candidate pairs by role

Pool is **frozen in M1** (before dataset pilot); pairs below are **pre-specified roles**, not a search grid.

| Role | M_lo | M_hi | Rationale |
| ---- | ---- | ---- | --------- |
| **Primary (ACL v1)** | Llama-3.2-3B-Instruct | Llama-3.1-8B-Instruct | Balance capability gap + oracle cost |
| Ablation / small gap | Llama-3.2-1B-Instruct | Llama-3.2-3B-Instruct | Behavior when \(P(r(q)=1)\) is low |
| Robustness (wide gap) | Llama-3.1-8B-Instruct | Llama-3.1-70B-Instruct | Generalization if resources permit |
| Literature baseline | Llama-2-7b-chat-hf | Llama-2-13b-chat-hf | Hybrid LLM–style (optional) |
| Defer | Llama-3.1-8B-Instruct | Llama-3.1-405B-Instruct | Extreme compute |

#### Hugging Face IDs (Instruct, text)

```text
meta-llama/Llama-3.2-1B-Instruct
meta-llama/Llama-3.2-3B-Instruct
meta-llama/Llama-3.1-8B-Instruct
meta-llama/Llama-3.1-70B-Instruct
meta-llama/Llama-3.3-70B-Instruct
meta-llama/Llama-2-7b-chat-hf
meta-llama/Llama-2-13b-chat-hf
```

#### ACL v1 recommendation (logged 2026-06-25, revised)

| Rank | Pair | Role |
| ---- | ---- | ---- |
| ⭐⭐⭐⭐⭐ | **3B → 8B** | **Primary** frozen pool |
| ⭐⭐⭐⭐ | **1B → 3B** | Pilot sanity check or gap-size ablation |
| ⭐⭐ | **8B → 70B** | Wide-gap robustness (optional, post-primary) |

Sources: [meta-llama/llama-models](https://github.com/meta-llama/llama-models) model cards (Llama 3.2, 3.1, 3.3, 4).

### Gap (this project)

Among papers above, supervised and post-generation routing dominate. **This project** follows the advisor proposal: unsupervised **complexity**, **entropy Q|Mᵢ**, and **paraphrase stability** on a fixed weak/strong pool.

---

## Run logging checklist

Record on every experiment run:

| Field | Example |
| ----- | ------- |
| Pool + dataset | Llama-3.2-3B / Llama-3.1-8B + ARC-Challenge |
| Protocol version | MCQ letter-eval v1 |
| Models | HuggingFace id + revision |
| Split | CALIB / TEST, n, seed |
| Signals extracted | complexity, entropy, paraphrase stability, … |
| Scorer | rule routing or learned weights |
| Paths | `experiments/` raw; `analysis/` summaries |

---

## Unread papers (no entry yet)

| Paper | Authors |
| ----- | ------- |
| Detecting Hallucinations via Semantic Entropy | Farquhar et al. |
| Self-Consistency | Wang et al. |
| AutoMix | Madaan et al. |
| On Calibration of Modern Neural Networks | Guo et al. |
| Conformal Prediction for MCQA | Kumar et al. |

---

## Document maintenance

 Document maintenance

| Event | Action |
| ----- | ------ |
| Deep-read unread uncertainty papers | Add §4 sections; update reading table |
| New routing paper | Add routing section + master comparison row |

*Last updated: 2026-06-25 — Llama family capability reference for M1 pool instantiation.*
