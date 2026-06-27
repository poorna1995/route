# C3 — Layerwise confidence evolution (concepts & verified references)

> **Implementation plan:** [`c3_prefill_extensions_plan.md`](c3_prefill_extensions_plan.md)  
> **Campaign folder (legacy name):** `C3_llama_confidence_formation/` — internal paths only.  
> **Paper vocabulary:** prefer **layerwise confidence evolution** before results; reserve **confidence formation** for Discussion _if_ results support it.

---

## 1. What C3 is (one paragraph)

**C3 extends model-derived signals** with a **richer characterization** of prefill confidence: not a fourth information source.

C0 extracts **terminal** statistics (\(H_w\), \(m_w\)) at the last prompt token after a full forward pass. C3 uses the **same forward pass** with `output_hidden_states=True` to measure how margin (and optionally entropy) **evolve across transformer depth** at that token, then asks:

> **RH5:** _Does layerwise evolution of model confidence provide routing-relevant information beyond terminal prefill confidence?_

**Measurable question (Methods / Results):**

> Do easy, opportunity, and too-hard queries exhibit **different confidence evolution** across transformer layers?

**Do not claim before results:** “confidence is formed progressively” — that is **interpretation** for Discussion only if F7/RH5 support it.

---

## 2. Wording rules (ACL reviewer-safe)

| Phase                           | Use                                                                               | Avoid                                                    |
| ------------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------- |
| **Before / during experiments** | layerwise confidence evolution · confidence trajectories · evolution across depth | confidence is formed progressively · formation (as fact) |
| **Methods**                     | we **characterize** / **investigate** evolution                                   | we **show** confidence forms                             |
| **Results**                     | curves differ · trajectories diverge at L\* · d(stabilization_layer)              | queries **form** confidence differently                  |
| **Discussion (if supported)**   | confidence **appears to form** progressively for opportunity queries              | stating formation as established literature              |

**Campaign code name** `confidence_formation` is fine on disk; **paper prose** follows the table above.

---

## 3. Technical pipeline (verified)

### 3.1 Logit-lens probe (locked — Methods wording)

**Methods sentence (use as-is):**

> We use a logit-lens style probe.

**What we claim vs what we do not claim:**

| Do **not** claim                                      | Do claim                                                                                 |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Layer ℓ **is** the model's true next-token prediction | Hidden state at layer ℓ → **LM head** → softmax → **probe statistics** (margin, entropy) |
| Intermediate layers are "correct" early answers       | Depth-indexed **observables** for routing characterization (RH5)                         |

Reviewer concern _"LM head on intermediate layers is theoretically shaky"_ is fair — and it is exactly the **logit lens** convention. LayerSkip, early-exit work, and related layer-analysis methods (e.g. TIDE) project intermediate hidden states through the shared head for analysis or early stopping. **This is not a reason to redesign the paper** — name the convention and move on.

```text
h_ℓ  →  lm_head  →  logits_ℓ  →  margin_ℓ, entropy_ℓ     (probe — not "prediction at layer ℓ")
```

Same pipeline as §3.2 below; terminal layer only must match C0 via final RMSNorm (§3.3).

### 3.2 Standard intermediate logits (implementation)

Early-exit and layer-analysis work obtain vocabulary logits at layer ℓ by projecting hidden states through the **shared LM head**:

```text
hidden_state(layer ℓ)  →  (optional norm — see §3.3)  →  lm_head  →  logits_ℓ  →  margin_ℓ, entropy_ℓ
```

This matches:

- **Early-exit literature:** intermediate logits via LM head on hidden states ([arXiv:2603.23701](https://arxiv.org/html/2603.23701v1) — “logit similarity” across layers).
- **Hugging Face API:** `output_hidden_states=True` returns `(embeddings, layer_1, …, layer_L)` ([HF Model outputs](https://huggingface.co/docs/transformers/main/main_classes/output)).

### 3.3 Llama 3.2 — RMSNorm before LM head (critical smoke check)

Llama applies **final RMSNorm** (`model.norm`) before `lm_head` on the **last** hidden state. Intermediate `hidden_states[i]` are typically **pre-final-norm**.

**Implementation rule:**

| Layer                               | Logits computation                                                                                              |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Final (terminal, must match C0)** | `logits = lm_head(model.norm(h_L))` **or** use `model(**inputs).logits` directly                                |
| **Intermediate ℓ < L**              | `logits_ℓ = lm_head(h_ℓ)` — standard early-exit convention; do **not** apply final `model.norm` to early layers |

**Smoke test (required before TEST):**

```text
| margin from lm_head(norm(h_L)) − margin from model(...).logits | < ε
```

**Mandatory pre-TEST smoke (RunPod):** Before full layerwise extraction, run `layerwise-parity` on several queries. Verify explicitly:

```text
margin( lm_head( model.model.norm( hidden_states[-1][:, last_pos, :] ) ) )
≈ margin( out.logits[:, last_pos, :] )     within margin_tol (default 1e-3)
```

```bash
python scripts/run.py layerwise-parity \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dataset arc_challenge --splits-json analysis/splits.json --split-role test \
  --limit 10 --device cuda --dtype bfloat16 --margin-tol 0.001
```

**Pass** → proceed to `--layerwise` extraction. **Fail** → stop and diagnose (HF tuple indexing, dtype, or architecture) before TEST.

Architecture note: final-layer `model.model.norm` is **Llama-specific** — acceptable for this paper (Llama 3.2 pool only).

Also compare to existing C0 `arc_test_weak.csv` for the same queries after parity passes.

### 3.4 One forward pass (unchanged cost model)

```text
outputs = model(**inputs, output_hidden_states=True)
last_pos = attention_mask.sum(dim=1) - 1   # same as C0 probe_batch

For ℓ = 1 … L:
    h_ℓ = hidden_states[ℓ][:, last_pos, :]   # HF tuple: index 0 = embeddings; layer ℓ → hidden_states[ℓ]
    logits_ℓ = project_to_vocab(h_ℓ, ℓ)    # §3.3
    m_ℓ, H_ℓ = extract_logits(logits_ℓ)     # reuse model_dependent.py
```

- **`batch_size=1`**, bf16 on GPU (memory).
- JSONL: full `{depth_fraction[], margin[], entropy[]}` per query (F7 + divergence).
- CSV: **`stabilization_layer`**, **`slope_margin`** (+ terminal columns for sanity). Headline scalar: **`stabilization_layer` only**.

### 3.5 Layer counts — never compare raw indices across pool models

| Model            | Transformer layers (`num_hidden_layers`) | Source                                                                                                                      |
| ---------------- | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Llama 3.2 **1B** | **16**                                   | [Meta torchtune builders](https://github.com/meta-pytorch/torchtune/blob/main/torchtune/models/llama3_2/_model_builders.py) |
| Llama 3.2 **3B** | **28**                                   | same                                                                                                                        |

**Methods rule:**

- F7 and divergence: **analyze weak and strong separately**, or plot **fraction of depth** ℓ/L on the x-axis.
- **Never** compare “layer 12 vs layer 12” across 1B and 3B.
- Cross-model merge uses **scalars** (`stabilization_layer_w/s`, `slope_margin_w/s`) and **normalized depth** in JSONL analysis — not raw ℓ.

Official model overview: [Llama 3.2 MODEL_CARD](https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_2/MODEL_CARD.md) · [HF 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) · [HF 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).

---

## 4. Formation scalars (what we extract)

**Priority (locked):** If reporting **one** layerwise statistic, use **`stabilization_layer` only**. `slope_margin` is secondary — kept in CSV for completeness but not headline RH5 evidence.

### 4.1 `stabilization_layer` ⭐ (primary — emphasize in paper)

**Methods sentence (use as-is):**

> The stabilization layer is defined as the first layer immediately following the beginning of a run of K stable adjacent margin transitions.

**Operational detail:** Build adjacent margin steps `transitions[i] = |m_{i+2} − m_{i+1}|` (equivalently `|margins[i+1] − margins[i]|`). Find the first index `start` where `transitions[start : start + K]` are all `< ε`. Return **`start + 2`** (1-indexed layer). Example: stable transition 1→2 only (`K=1`, `start=0`) ⇒ **stabilization layer = 2**. If none, return L.

**Layer-1 rule:** Stabilization requires observing at least one transition. Layer 1 has no prior margin, so **stabilization at layer 1 is undefined**. When L ≥ 2, the **minimum possible** `stabilization_layer` is **2**.

**Defaults (freeze on smoke):** ε = 0.02, K = 2.

**Also report:** `stabilization_frac` = `depth_fraction[stabilization_layer − 1]` in JSONL (cross-model readable).

### 4.2 `slope_margin` (secondary)

**Definition:** OLS slope of \(m\_\ell\) vs layer index ℓ = 1…L.

**Role:** Exploratory summary of overall rate of margin change. **Weaker and less interpretable than `stabilization_layer`** for routing — report only as supplement; do not gate RH5 on slope AUROC or effect size.

### 4.3 Probe space — raw softmax (no logit rescaling)

At each layer we compute **margin and entropy from `softmax(logits_ℓ)`** with **no temperature scaling or other logit normalization**. This matches the model's own next-token distribution at that depth — the same probe space as C0 terminal statistics.

**Do not adopt:** entropy temperature / calibrated rescaling at probe time — that changes the probe and invites reviewer questions (“why this normalization?”).

**Limitation (Discussion):** Chat/aligned models can be miscalibrated; we treat raw prefill softmax as the **unsupervised observable**, consistent with C0, not as a calibrated probability estimate.

### 4.4 JSONL trace format

Per query: `num_layers`, `depth_fraction[]`, `margin[]`, `entropy[]` — **no raw `layers` list** (reconstruct `1…L` from `num_layers` if needed). Layerwise entropy traces are debug/F7 context only; no `slope_entropy` CSV column.

---

## 5. Literature — motivation vs justification

### 5.1 Logit lens — not a redesign trigger

**Reviewer concern:** intermediate LM-head projections are theoretically imperfect.

**Response:** Agreed — and standard. We use a **logit-lens style probe** (§3.1). LayerSkip, early-exit, and related layer-analysis work use the same machinery; our objective is **routing signal characterization**, not claiming layer ℓ equals the final model prediction.

### 5.2 LayerSkip (ACL 2024) — **motivation only**

**Citation:** Elhoushi et al., [LayerSkip](https://aclanthology.org/2024.acl-long.681.pdf) (ACL 2024).

**What it shows:** Intermediate layers can produce **usable** next-token predictions via a shared LM head (early-exit training).

**What it does NOT show:** That intermediate confidence is **useful for routing**.

**Correct citation chain in Related Work / Methods:**

```text
LayerSkip → intermediate logits are meaningful
         → therefore they can be measured at prefill time
         → we investigate whether they carry routing-relevant information (RH5)
```

### 5.3 Early-exit papers — same machinery, different objective

**Example:** [The Diminishing Returns of Early-Exit Decoding in Modern LLMs](https://arxiv.org/html/2603.23701v1) (arXiv:2603.23701).

| Early-exit literature              | This paper (C3)                                                       |
| ---------------------------------- | --------------------------------------------------------------------- |
| Can we **stop computation early**? | Can we **characterize routing-relevant information before decoding**? |
| Latency / cost                     | Signal characterization                                               |
| Logit similarity across layers     | Trajectories vs oracle buckets                                        |

**Discussion note:** Newer LLMs may show less layer redundancy → **null RH5 is publishable** (“terminal probes suffice on ARC”).

### 5.4 Routing contrast (project literature — not C3-specific)

Use `03_literature_gap.md`: RouteLLM, RouterBench, He/Plaut prefill uncertainty. C3 does not replace that story — it **deepens model-derived** characterization.

---

## 6. RH5 — hypothesis, evidence, null

### Hypothesis (locked)

> **RH5:** Does layerwise evolution of model confidence provide routing-relevant information **beyond terminal** prefill confidence?

### Supported if any hold (no AUROC gate)

| #   | Evidence               | Example                                                                      |
| --- | ---------------------- | ---------------------------------------------------------------------------- |
| 1   | **F7 geometry**        | Median \(m\_\ell\) curves separate easy / opportunity / too-hard             |
| 2   | **Divergence L\***     | Opp and too-hard similar until ~fraction depth d\*, then diverge             |
| 3   | **Effect size**        | d(`stabilization_layer`, opp vs too-hard) > d(\(H_w\)) ≈ 0.03                |
| 4   | **Non-redundancy**     | Partial ρ(`stabilization_layer`, opp \| terminal \(m_w\)) > 0, CI excludes 0 |
| 5   | **Null (publishable)** | Trajectories mirror terminal — terminal probes suffice on ARC                |

**Primary deliverable:** F7 + one observation sentence. AUROC secondary.

### Figure F7

- **Title:** **Layerwise Confidence Evolution Across Depth**
- **X-axis:** `depth_fraction` from JSONL (ℓ/L) — **always**, never raw layer index (comparable 16L vs 28L)
- **Y-axis:** median \(m\_\ell\) at last prompt token
- **Curves:** Easy · Opportunity · Too-hard — **weak and strong separately** (`F7_*_{weak,strong}.png`)
- **File:** `paper/figures/F7_confidence_evolution.png`

---

## 7. Stop rule (project management)

See [`c3_prefill_extensions_plan.md`](c3_prefill_extensions_plan.md) §10 for full sequence. Summary:

```text
parity → smoke + manual trajectories → CALIB + F7 decision gate → TEST (if pass)
    ↓ fail at any gate
C3 = future work (no redesign, no multi-day debug)
```

Aligns with professor guidance: one problem, hypothesis-driven experiments, paper-first.

---

## 8. Claim map (C6 optional)

| Claim                                                        | Evidence                                           |
| ------------------------------------------------------------ | -------------------------------------------------- |
| Layerwise evolution adds routing information beyond terminal | F7, RH5 JSON, optional d(`stabilization_layer`)    |
| C3 is a richer **model-derived** characterization            | Methods taxonomy — not a fourth information source |

See [`claims_evidence_matrix.md`](claims_evidence_matrix.md).

---

## 9. Verified reference list

| Resource                                | URL                                                                                              | Use                                                           |
| --------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------- |
| HF `CausalLMOutput` / `hidden_states`   | https://huggingface.co/docs/transformers/main/main_classes/output                                | Implementation API                                            |
| HF output tracing                       | https://huggingface.co/docs/transformers/en/model_output_tracing                                 | Hook / collection behavior                                    |
| Logit lens (intermediate LM-head probe) | §3.1 — Methods one-liner; not claiming ℓ = final prediction                                      |
| LayerSkip (ACL 2024)                    | https://aclanthology.org/2024.acl-long.681.pdf                                                   | Same machinery; motivation only                               |
| Early-exit / logit similarity           | https://arxiv.org/html/2603.23701v1                                                              | Related work; intermediate logits via LM head                 |
| Llama 3.2 model card                    | https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_2/MODEL_CARD.md     | Pool models                                                   |
| Llama 3.2 layer configs                 | https://github.com/meta-pytorch/torchtune/blob/main/torchtune/models/llama3_2/_model_builders.py | 16 vs 28 layers                                               |
| Project C0 protocol                     | `scripts/routing/model_dependent.py`, `05_computation_protocol.md`                               | Terminal probe parity                                         |
| Project C3 plan                         | `c3_prefill_extensions_plan.md`                                                                  | Commands, checklist, artifacts                                |
|                                         | `scripts/routing/layerwise.py`                                                                   | Probes, scalars, JSONL traces, `verify_terminal_logit_parity` |
|                                         | `scripts/run.py layerwise-parity`                                                                | Pre-TEST terminal parity smoke                                |
|                                         | `scripts/routing/formation_analysis.py`                                                          | RH5 divergence analysis                                       |
|                                         | `scripts/run.py`                                                                                 | `--layerwise`, `analyze-formation`, `plot formation`          |
