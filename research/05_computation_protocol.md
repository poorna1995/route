# Computation Protocol

> **Scientific decision this document supports:**  
> How is each retained pre-inference signal computed — exact definition, **prefill probe** protocol, implementation steps, and cost?

> **Milestone M4** — retained signals (entropy, margin) specification complete
>
> Related documents:
>
> - Signal Design (why) → `04_signal_design.md`
> - Decision Register (frozen parameters) → `09_decision_register.md`
> - Frozen pool → `MASTER.md` §5
> - Problem assumptions A2, A4, A6 → `01_problem_statement.md`

---

# Purpose

`04_signal_design.md` explains **why** a signal is meaningful (candidates, literature, mechanisms, failure modes).

This document explains **how** each retained signal is computed. Implementation should be mechanical once taxonomy is frozen in `04` Part F.

**Do not implement probes until:** (1) signal is **Keep** in `04` Part F, (2) corresponding section below is complete, (3) pool locked in `MASTER` §5.

---

# §1 — Shared notation and prefill probe protocol

All model-dependent signals share this **prefill probe** protocol unless stated otherwise. Do not use generic *probe* when *prefill probe* is meant — *probe* alone is ambiguous (prefix probe, paraphrase probe, activation probe, etc.).

| Symbol | Meaning |
| ------ | ------- |
| `u` | Task-formatted user message (benchmark item + format-only suffix) |
| `P(u)` | Chat-wrapped prompt: `apply_chat_template` on user message (model-native template; see §1 Prompt protocol) |
| `q` | Raw benchmark text before task formatting (internal only) |
| `Mᵢ` | Model *i* in the fixed probe pool |
| `x = P(u)` | Tokenized chat prompt; length `T = \|x\|` |
| `y` | Full model response (**not** generated during prefill probe) |
| `V` | Vocabulary of `Mᵢ` |
| `z_t` | Pre-softmax logits at generation step *t* |
| `p_t(v) = softmax(z_t)_v` | Probability of token *v* at step *t* |

## Prefill probe (A4-compliant, canonical)

**Prefill probe:** One forward pass on prompt `x` only. Statistics at position `t = T` (first **post-prompt** token): the model’s distribution over the **first generated token** before any decoding.

### Model-dependent signals (canonical reference)

All statistics below are computed from the same predictive distribution \(P(\text{next token} \mid q, M_i)\) at prefill step \(t = T\) (first post-prompt token). **Primary signals:** entropy and margin (D15). **`max_prob` is auxiliary** — logged for analysis, not a paper claim.

| Signal | Notation | What it is | Cost |
| ------ | -------- | ---------- | ---- |
| **Token entropy** | \(H(q, M_i)\) | Spread of \(P(\text{next token} \mid q, M_i)\) | 1 forward pass |
| **Log-probability margin** | \(m(q, M_i)\) | \(p^{(1)} - p^{(2)}\) on the same distribution | Same pass (no extra cost) |
| **Max next-token probability** (auxiliary) | \(p_{\max}(q, M_i)\) | Peak probability \(p^{(1)}\); CSV column `max_prob`; **not a primary signal** | Same pass |

Terminology: use **log-probability margin** (or **margin** in prose) for \(m\). Do **not** call \(p_{\max}\) or \(m\) “confidence” — confidence implies calibration; these are raw predictive-distribution statistics.

### Interpretation boundary (what signals are and are not)

Each retained probe implements a model-dependent function \(s_i = f(q, M_i)\): logits—and therefore \(H\), \(m\), and \(p_{\max}\)—change when either the query or the pool member changes.

| Signal | Measures | Does **not** directly measure |
| ------ | -------- | ----------------------------- |
| \(H(q, M_i)\) | Spread of \(P(\text{next token} \mid q, M_i)\) | Final answer correctness |
| \(m(q, M_i)\) | Separation between top-1 and top-2 next-token probabilities | Reasoning-chain length or step count |
| \(p_{\max}(q, M_i)\) (auxiliary) | Peak next-token probability | Task difficulty in isolation from \(M_i\) |

These are **predictive-distribution statistics** available before full response generation. Their relationship to offline routing need (weak failure / strong success) is an **empirical hypothesis** tested in Phase C—not a definitional claim. Do **not** describe entropy as “reasoning complexity”; describe it as an unsupervised uncertainty signal whose informativeness for routing is evaluated empirically.

**Optional ablation — prefix probe:** Greedy-decode exactly `T_probe` tokens (default `T_probe = 8`, logged in `09`), aggregate over steps `t = T, …, T + T_probe - 1`. Not full answer generation; use only if prefill probe is weak in pilots.

## Prompt protocol (fixed)

Three layers — full spec in this section. Implementation: `../experiments/README.md`.

```text
Dataset item
      ↓
Task formatting  →  user_content u  (benchmark-specific; format-only suffixes)
      ↓
Chat template    →  P(u) = tokenizer.apply_chat_template([{role: user, content: u}], …)
      ↓
      ├─ Prefill probe (forward pass only)  →  signal extraction (V1 / pilot)
      └─ model.generate()                  →  oracle labels only (V2 / offline eval)
```

**No researcher-defined system prompt.** User message only; any default system text comes from the **model's native** chat template (tokenizer-dependent — record model ids in `09`).

**Task formatting (frozen per benchmark):**

| Benchmark | `user_content` suffix (format only — no reasoning aids) |
| --------- | ------------------------------------------------------- |
| GSM8K | `Answer using only one number.` |
| ARC-Challenge | Options + `Reply with the letter of the correct answer only (A, B, C, or D).` |

The identical `P(u)` is used for prefill probe extraction and oracle generation on the same query.

The final token of `P(u)` must start the assistant turn so step `t = T` predicts the first content token. Frozen parameters → `09_decision_register.md` after setup validation passes.

## Decoding settings for probe

| Setting | Value | Rationale |
| ------- | ----- | --------- |
| Temperature | `τ = 1` on logits at prefill probe step(s) | Standard predictive distribution |
| Top-*k* / top-*p* | Disabled at prefill probe step | Full softmax for entropy/margin |
| Random seed | Fixed per `(q, Mᵢ)` | Reproducibility (A6) |

## Cross-model comparability

Raw `s(q, Mᵢ)` are **not** assumed comparable across `Mᵢ` (`04` Part C — He, Plaut). Routing analysis uses **within-query ranks** or **per-model z-scores** on a validation split. Cross-model raw thresholds out of scope unless calibration ablation added in `09`.

## Infrastructure requirements

- **Logprob access:** Full-vocabulary softmax at prefill probe step(s), or API `logprobs` with documented top-*k* bias if approximate.
- **No fine-tuning** of pool models for probe extraction.

---

# §2 — Token entropy

## Signal specification

```text
Signal:           Token entropy
Notation:         H(q, Mᵢ)
Type:             Model-dependent (04 Part F — Keep)

Input:            Query q; model Mᵢ; frozen prompt template P(·)
Output:           H(q, Mᵢ) ∈ [0, log |V|]  (nats)
Requires:         Full-vocabulary log-probabilities at prefill probe step t = T
Prefill probe:    Canonical — one forward pass on P(u) only; no answer generation
Prompt format:    §1 fixed template (System / User / Assistant turn start)
Decoding:         τ = 1; no top-k / top-p at prefill probe step
Cost:             1 forward pass per (q, Mᵢ)
Assumptions:      A4 (pre-gen); A6 (same protocol all Mᵢ); raw H not cross-model comparable (use ranks/z-scores)
Complexity:       O(T) prompt tokens processed; O(|V|) for softmax at probe step
```

## Definition

At prefill probe step `t = T`, let `p_t ∈ Δ^{|V|-1}` be the next-token distribution:

\[
H(q, M_i) = -\sum_{v \in V} p_T(v)\,\log p_T(v)
\]

Natural log (nats) unless noted otherwise.

**Prefix-probe variant** (ablation only):

\[
H_{\text{prefix}}(q, M_i) = \frac{1}{T_{\text{probe}}} \sum_{j=0}^{T_{\text{probe}}-1} H_{T+j}
\]

with teacher-forced prefix tokens when `j > 0`.

**Routing direction:** Higher `H` → greater uncertainty → candidate for stronger model (hypothesis; tested in experiments).

## Prefill probe procedure

1. Build `x = P(u)` with frozen prompt protocol.
2. One forward pass on `Mᵢ` (prefill only).
3. Read logits `z_T` at index `T`.
4. `p_T = softmax(z_T / τ)`, `τ = 1`.
5. `s(q, Mᵢ) ← H(q, Mᵢ)`.

## Implementation protocol

| Step | Action |
| ---- | ------ |
| 1 | Tokenize `P(u)`; truncate to context limit minus 1 token for probe position. |
| 2 | **Local (HF/vLLM):** `outputs = model(input_ids, use_cache=False)`; `outputs.logits[0, T-1, :]`. |
| 3 | **API:** `max_tokens=1`, `logprobs` as full as available; flag approximate entropy if top-*k* ≪ \|V\|. |
| 4 | `H = -(p * log p).sum()` via `log_softmax`. |
| 5 | Store row in probe CSV (schema §5). |

## Output fields

| Field | Type | Description |
| ----- | ---- | ----------- |
| `query_id` | str | Stable ID from dataset |
| `model_id` | str | Pool member identifier |
| `entropy` | float | `H(q, Mᵢ)` in nats |
| `extraction_method` | str | `prefill_probe` (canonical prefill probe) |
| `prompt_tokens` | int | `T` |
| `timestamp` | str | ISO run id for reproducibility |

## Computational cost (entropy + margin)

| Resource | Prefill (canonical) |
| -------- | ------------------- |
| Forward passes per `(q, Mᵢ)` | **1** |
| Tokens processed | `T` (prompt only) |
| Pool cost per query | `\|M\|` passes |
| Signals extracted | `H` and `m` from same pass |

**vs full generation (A2):** O(`T`) ≪ O(\|answer\|).

---

# §3 — Log-probability margin

## Signal specification

```text
Signal:           Log-probability margin (paper name; CSV column: margin)
Notation:         m(q, Mᵢ)
Type:             Model-dependent (04 Part F — Keep)

Input:            Query q; model Mᵢ; frozen prompt template P(·)
Output:           m(q, Mᵢ) = p_T^(1) − p_T^(2)  ∈ [0, 1]
Requires:         Top-2 token probabilities (or logits) at prefill probe step t = T
Prefill probe:    Same forward pass as §2 (canonical)
Prompt format:    §1 fixed template
Decoding:         τ = 1; no top-k / top-p at prefill probe step
Cost:             0 additional passes (extract from §2 pass)
Assumptions:      A4, A6; raw m not cross-model comparable; no calibration at probe time
Complexity:       O(|V|) for top-2 from same logits as entropy
```

## Definition

Top-two probabilities at `t = T`:

\[
m(q, M_i) = p_T^{(1)} - p_T^{(2)}
\]

**Logit margin** (when softmax unavailable): `m_logit = z_T^{(1)} - z_T^{(2)}`. Canonical: `m` (probability gap).

**Prefix-probe variant:** `m_prefix = min_j ( p_{T+j}^{(1)} - p_{T+j}^{(2)} )`.

**Hypothesis direction (empirical):** Lower `m` → smaller top-1/top-2 gap → candidate for stronger model.

## Prefill probe procedure

Same prefill setup as §1–§2. One pass → top-2 from `p_T` → `s(q, Mᵢ) ← m(q, Mᵢ)`.

## Implementation protocol

| Step | Action |
| ---- | ------ |
| 1–2 | Identical to §2. |
| 3 | `top2 = torch.topk(log_softmax(z_T), k=2)` → `m = exp(top2[0]) - exp(top2[1])`. |
| 4 | **API:** `logprobs=2` at probe step. |
| 5 | Append `margin` to same CSV row as `entropy` (schema below). |

No isotonic calibration or temperature scaling at probe time (raw margin as unsupervised probe; calibration = optional ablation in `09`).

See `04` Part E for redundancy between entropy and margin.

## Joint extraction (implementation)

Both signals from **one** forward pass per `(q, Mᵢ)`:

```text
logits z_T  →  log_softmax  →  H(q, Mᵢ)
                          └→  top-2  →  m(q, Mᵢ), p_max(q, Mᵢ)
```

Never run separate passes for entropy, margin, or `max_prob`.

---

# §3b — Max next-token probability (auxiliary — not a primary signal)

## Definition

At the same probe step as §2–§3:

\[
p_{\max}(q, M_i) = p_T^{(1)} = \max_v P(v \mid q, M_i)
\]

**Role:** Optional diagnostic logged as CSV column `max_prob`. Supports cross-model context alongside `vocab_size`. **Not** listed as a pilot signal in D15; do not treat as a paper claim unless explicitly promoted in `09`.

**Cost:** Zero — extracted from the same `top-2` / softmax as §2–§3.

---

# §4 — Paraphrase stability (deferred — not pilot)

## Definition

Paraphrases `{q̃ⱼ}_{j=1}^{P}` of `q` (`P = 3` default). Base statistic `b(q̃, Mᵢ)` = token entropy `H(q̃, Mᵢ)` from §2 (alternative: margin `m` → `09`).

Values `hⱼ = b(q̃ⱼ, Mᵢ)`, median `h̃ = median({hⱼ})`:

\[
\text{MAD}(q, M_i) = \frac{1}{P} \sum_{j=1}^{P} | h_j - \tilde{h} |
\]

\[
s_{\text{ps}}(q, M_i) = \frac{1}{1 + \text{MAD}(q, M_i)}
\]

Report `s_ps` (higher = more stable). Optional instability: `s_instab = MAD`.

**Routing direction:** Low `s_ps` (high MAD) → misfit → candidate for stronger model.

## Paraphrase generation

| Parameter | Default | Notes |
| --------- | ------- | ----- |
| `P` | 3 | Original `q` excluded from MAD |
| Paraphraser | Frozen `M_para` (fixed in `06`) | Not a pool member |
| Prompt | “Rephrase the following question preserving meaning. Output only the rephrased question.\n\n{q}” | τ=0.7, max 256 tokens |
| Filter | Drop if ROUGE-L with `q` > 0.95 or < 0.3 | Thresholds in `09` |

Paraphrases cached per `q`.

## Prefill probe procedure

1. Generate/cache `{q̃₁, …, q̃_P}` for `q`.
2. For each `j`, each `Mᵢ`: `hⱼ = H(q̃ⱼ, Mᵢ)` via §2.
3. MAD → `s_ps(q, Mᵢ)`.

## Implementation protocol

| Step | Action |
| ---- | ------ |
| 1 | `paraphrases[q] → list[str]` on disk. |
| 2 | For `Mᵢ ∈ pool`, `q̃ⱼ ∈ paraphrases[q]`: run §2 on `P(q̃ⱼ)`. |
| 3 | `{hⱼ}` → MAD → `s_ps`. |
| 4 | Store `(q, Mᵢ, s_ps, {hⱼ}, P)`. |

## Computational cost

| Resource | Formula | Example (`\|M\|=5`, `P=3`) |
| -------- | ------- | --------------------------- |
| Paraphrase generation | 1 call per unique `q` | 1 × `M_para` |
| Prefill probes | `P × \|M\|` passes | 15 |

---

# §5 — Probe CSV schema

One row per `(query_id, model_id)`. Generated by probe pipeline (Week 2); consumed by signal analysis (Week 3).

| Column | Type | Required | Description |
| ------ | ---- | -------- | ----------- |
| `query_id` | str | ✓ | Dataset identifier |
| `model_id` | str | ✓ | e.g. `small`, `large` (see `06`) |
| `entropy` | float | ✓ | `H(q, Mᵢ)` nats |
| `margin` | float | ✓ | `m(q, Mᵢ)` |
| `max_prob` | float | optional | Top-1 next-token probability at probe step \(T\) (auxiliary; not a paper signal) |
| `vocab_size` | int | optional | `\|V\|` at probe step — documents cross-model entropy context |
| `extraction_method` | str | ✓ | `prefill_probe` (canonical). Legacy column name: `probe_mode` with value `prefill` on n=10 CSVs. |
| `prompt_tokens` | int | ✓ | Token count `T` |
| `run_id` | str | ✓ | Reproducibility tag (date + git hash) |

Example:

| query_id | model_id | entropy | margin |
| -------- | -------- | ------- | ------ |
| gsm8k_001 | small | 8.42 | 0.31 |
| gsm8k_001 | large | 6.15 | 0.58 |
| gsm8k_002 | small | 9.01 | 0.12 |
| gsm8k_002 | large | 5.88 | 0.62 |

Oracle labels (`07`) join on `query_id` after full generation — not in probe CSV.

---

# §6 — Specification status

**Main-study scope (D12/D56):** Implement **Keep** signals — \(c(q)\) (§8), token entropy (§2), margin (§3).

| Signal | § | Spec complete | Main study |
| ------ | - | ------------- | ----- |
| Token entropy | §2 | ✓ | **Yes** |
| Log-probability margin | §3 | ✓ | **Yes** |
| Paraphrase stability | §4 | Draft (deferred) | No |
| Query complexity \(c(q)\) | §8 | Pending D46 | **Yes** (when implemented) |

**M4 gate:** Computation protocol complete for entropy and margin. \(c(q)\) completes in §8 after D46.

---

# §7 — Deferred signals

Add a new § here only when `04` Part F records **Keep** for a signal not yet specified. Do not expand `04_signal_design.md` with computation detail.

---

# §8 — Query complexity \(c(q)\) (model-independent — D46 pending)

## Signal specification

```text
Signal:           Query complexity (representative)
Notation:         c(q)  —  model-independent
Type:             One member of the model-independent complexity family,
                  selected by predefined D46 signal screening (not a paper experiment)

Input:            Formatted query text (user_content — protocol layer 1)
Output:           c(q) ∈ ℝ
Requires:         No model forward pass
Cost:             O(|q|) text processing
Hypothesis:       RH1 / H1
Status:           Selection protocol LOCKED (D57); winning formula pending D46 outcome
```

RH1 tests whether a **representative** model-independent signal carries routing-relevant information—not whether one fixed statistic (e.g. Shannon entropy) is optimal a priori.

## D46 signal screening (frozen — D57, D60)

**Not a paper experiment.** One-time feature screening on ARC **CALIB only** (150 queries, seed 42). V2 n=50 is validation only — **do not lock D46 from n=50**.

**Family structure (one representative chosen across all):**

| Family | Column(s) |
| ------ | ----------- |
| Length | `word_count` |
| Lexical diversity | `mattr` |
| Information | `text_shannon`, `text_shannon_norm` |
| Compressibility | `compression_ratio` |

**Reported per candidate:** descriptive stats (mean, std, median, min, max); Spearman \(\rho\) + **95% bootstrap CI**; AUROC + **95% bootstrap CI**; Cohen's \(d\) and Cliff's \(\delta\) (opportunity vs non-opportunity). **p-values reported but not used for selection.**

**Selection rule (D60):**

```text
composite = 0.5 * norm(|rho|) + 0.5 * norm(AUROC)   # min-max over candidates on CALIB
```

Maximize composite. If composite scores or bootstrap CIs overlap, choose the **simplest / most interpretable** measure (length → compressibility → information → lexical diversity) — not a hard-coded preference for any one statistic.

**After D46:** `selected_candidate` becomes **`c_q`** on TEST and all main runs (EXP-01–03). Record in `09`; copy formula below. **Freeze — do not re-screen** without a new decision.

**Out of scope:** QCE / TaskComplexityAnalyzer, dataset profiles, tool/multihop priors, `fit()` calibrators, paraphrase (D04).

## Tokenization for \(c(q)\) candidates (frozen — D58)

**Model-independent ≠ architecture-agnostic.** Features use **only query text** with **no forward pass**; a tokenizer is deterministic preprocessing, not model-dependent inference.

| Feature | Input representation |
| ------- | -------------------- |
| C01 `word_count` | Count of HF **subword pieces** (`tokenizer.tokenize(user_content)`) |
| C03 `mattr` | MATTR over subword pieces (window = 15 pieces) |
| C10 `text_shannon` | \(H_{\text{text}}(q) = -\sum_{t \in V_q} p(t)\log p(t)\), \(p(t)=\text{count}(t)/N\), **nats** (`math.log`) |
| C10b `text_shannon_norm` | \(H_{\text{text}}(q) / \log |V_q|\) when \(|V_q|>1\), else 0 |
| C11 `compression_ratio` | zlib ratio on **raw UTF-8** `user_content` (unchanged) |

### Notation: lexical vs model entropy (do not conflate)

| Symbol | Family | Object |
| ------ | ------ | ------ |
| \(H_{\text{text}}(q)\) or column `text_shannon` | Model-independent | Empirical unigram entropy over **query subword pieces** |
| \(H(q, M_i)\) or column `entropy` | Model-dependent | Shannon entropy of **next-token distribution** at prefill probe (§2) |

These share the Shannon formula but operate on **different random variables**. In prose: “lexical entropy” / \(H_{\text{text}}\) vs “probe entropy” / \(H_i\). If D46 selects an entropy variant, \(c(q)\) denotes that lexical quantity only.

**What \(H_{\text{text}}\) is not:** LM predictive entropy, zlib/compression entropy, character entropy, or conditional entropy \(H(X|Y)\).

**Tokenizer:** `AutoTokenizer` for **`meta-llama/Llama-3.2-1B-Instruct`** (weak pool model; same vocabulary as 3B Instruct). Applied to **layer-1 `user_content`**, not chat-templated prompt.

**Not used:** regex word tokenization (too misaligned with LLM segmentation); GPT/tiktoken (wrong model family).

**Cross-family generalization (future):** If comparing Llama vs Qwen vs Gemma, either use a language-level tokenizer or document weak-model tokenizer consistently — do not silently mix families.

## Selected formula (fill after D46)

```text
c(q) = [ pending — e.g. mattr(q) | text_shannon(q) | … ]
```

**Do not** preset Shannon (or any candidate) in methodology prose until D46 is logged.

## Role in main study

| Quantity | Family | Role |
| -------- | ------ | ---- |
| \(c(q)\) | Model-independent | Representative complexity signal (D46) |
| \(H(q, M_i)\), \(m(q, M_i)\) | Model-dependent | Prefill probes |
| \(y_{\text{opp}}(q)\), bucket | Oracle (eval only) | `07` |

**Complementarity (RH3 / Study III — D61):** Primary: model-dependent probes add beyond frozen \(c(q)\) — ladder c → c+H → c+H+m. Secondary: margin beyond entropy within dep family (appendix). Not a 7-model feature search.
