# Research program

> **ACL v1:** unsupervised routing + weight learning in a **fixed pool of models with distinct capabilities**.  
> **Vocabulary frozen 2026-06-28** — [`nomenclature.md`](nomenclature.md). No term changes unless compelling technical reason.  
> **Related work:** [`literature_record.md`](literature_record.md)  
> Prior experiments retired — see nomenclature §6. **Benchmark, pool, and protocol not locked** (§13): chosen via reproducible Stages **1–3**, then frozen before oracle runs.

**Canonical sections:** **§0 research design (Stage 0)** · signal assumptions **§5** · unsupervised §2 · routing need §3 · pipeline §4 · execution workflow **§11 (Stages 0–9)** · routing policy §8 (refs [`nomenclature.md`](nomenclature.md) **§2.1** for \(s\), \(\pi\))

---

## 0. Research design (Stage 0)

**Execution stage:** 0 — Research design. **Status:** locked (2026-06-28). Benchmark, pool, and protocol are **not** locked — Stages **1–3** select and freeze them (§0.7, §13). Everything else in §0 is **method-level**.

**Purpose:** Fix the scientific contract before Stages 1–9. Stage 0 locks *what* we test; Stages 1–3 lock *where* we test it; Stages 4–9 run the study.

### 0.1 Title and pitch

**Working title:** *Routing from Unsupervised Signals in a Fixed LLM Pool*

**One sentence:** We study whether **unsupervised signals** — query-derived, model-response, and cross-model comparative — can **enable routing decisions** in a fixed **model pool** with distinct capabilities, and whether **learned combination weights** on calib improve over hand-written rules, in contrast to **supervised** routers trained on preferences or outcomes.

**30 s pitch:** Supervised routing dominates the field. We pre-specify unsupervised statistics at extraction time, test whether they predict **routing need** \(r(q)\), freeze \(x(q)\), fit a threshold policy via nomenclature §2.1, and judge routing on **accuracy–cost Pareto** curves — reporting signal analysis and routing evaluation as **separate** claims.

### 0.2 Motivation

Multi-LLM systems must trade **quality** against **inference cost**. Recent routers (RouteLLM, HybridLLM, GraphRouter, RouterBench, …) **learn** routing from human preferences, win/loss labels, or reward scores. That leaves a prior question open:

> *What routing-relevant information exists in signals computed **without** routing supervision at extraction time?*

Uncertainty and calibration work asks whether models “know what they know,” but routing papers rarely run **nested ablations** over signal types against a single **routing-need** target, nor separate **informativeness** from **deployable routing quality**.

**Setting:** A fixed **model pool** \(\mathcal{M}=\{M_1,\ldots,M_K\}\) whose members have **distinct, documented capabilities** (not interchangeable). For each query \(q\), pool members run under a **frozen** prompt and decoding protocol. The router selects one member per query; cost scales with the chosen model.

### 0.3 Primary question and sub-questions

**Primary question (PQ):**

> Can unsupervised signals **guide routing decisions** in a fixed capability-differentiated pool, and does learned weighting on calib beat simple rules on the **same** frozen features?

**Sub-questions (advisor formulation):**

| ID | Question | Maps to |
| -- | -------- | ------- |
| **SQ1** | What query-derived, model-response, and cross-model comparative signals can we estimate without routing labels at extraction? | RQ1 · §5 · Stage **5** |
| **SQ2** | How should those signals be combined into a score \(s(q)\)? | nomenclature §2.1 · §8 · Stage **8** |
| **SQ3** | Do resulting policies **guide** model choice vs baselines on accuracy–cost? | §9 · H4 · Stage **9** |

**Research questions (paper-facing):**

| ID | Question | Hypotheses |
| -- | -------- | ---------- |
| **RQ1** | What unsupervised signal types can we measure at layer 1? | — (descriptive; Stage **5**) |
| **RQ2** | Which signals predict routing need \(r(q)\) on calib? | **H1–H3** (Stage **6**) |
| **RQ3** | After freezing \(x(q)\), do learned weights outperform hand-designed rules? | **H4** (Stage **9**) |

**Critical separation:** RQ2 (informativeness) and RQ3 (routing) are **independent** scientific claims. Either may succeed or fail without implying the other (§0.12).

### 0.4 Contrast with prior work

**Headline contrast:** **Supervised vs unsupervised at signal extraction** — not agent orchestration, not pre- vs post-inference timing as the headline.

| | Typical supervised router | This work |
| - | ------------------------- | --------- |
| Router input | Classifier on query (+ history), trained on prefs/outcomes | Pre-specified **unsupervised signals** → \(x(q)\) |
| Where labels enter | **Define** the router end-to-end | **Calib only:** analysis (§6) + \((\lambda,\tau)\) fit (§8) |
| Signal source | Learned representations from routing data | Query-derived, model-response, cross-model comparative (§5) |
| Evaluation | Often accuracy or win-rate alone | **Pareto** \((\mathrm{Acc}, \mathrm{Cost})\) on test |

We cite supervised systems as field context; **reproducing** them is out of scope (§14).

### 0.5 Supervision model (three layers)

**Unsupervised** describes **how signals are obtained** (layer 1), not “zero labels in the entire paper.”

| Layer | Program § | Uses \(r(q)\) or routing prefs? | Unsupervised? |
| ----- | ---------- | ------------------------------- | ------------- |
| **1 — Signal extraction** | §5 | **No** | **Yes** — \(x_i(q)\) from query / model only |
| **2 — Signal analysis** | §6 | **Yes, calib only** | Analysis supervised; **signal definitions** not |
| **3 — Policy fit** | §8 | **Yes, calib only** | Combination supervised; **signal definitions** not |

**Paper label:** *Routing from unsupervised signals* — layers 2–3 use offline \(r(q)\) on calib to **evaluate** and **combine**, not to invent features.

**What we do not mean:** (i) “no labels anywhere”; (ii) classical unsupervised routing with no calib oracle; (iii) that model-response signals avoid inference — “unsupervised” means **routing supervision**, not compute.

Full definition and contrast table: §2.

### 0.6 Problem formulation

**Per-model correctness:** \(y(q,M)\in\{0,1\}\) under frozen protocol.

**Primary routing pair (v1):** Stage 3 designates two pool members with ordered capability: **\(M_{\mathrm{lo}}\)** (lower capability, lower cost) and **\(M_{\mathrm{hi}}\)** (higher capability, higher cost), with \(\mathrm{cap}(M_{\mathrm{lo}}) < \mathrm{cap}(M_{\mathrm{hi}})\). Write \(y_{\mathrm{lo}}=y(q,M_{\mathrm{lo}})\), \(y_{\mathrm{hi}}=y(q,M_{\mathrm{hi}})\). The threshold policy and \(r(q)\) are defined on this pair; the full pool \(\mathcal{M}\) may contain additional members for comparative signals and future extensions.

**Routing need:**

\[
r(q) = \mathbb{1}\big[\, y_{\mathrm{lo}} = 0 \;\wedge\; y_{\mathrm{hi}} = 1 \,\big], \qquad \Delta(q) = y_{\mathrm{hi}} - y_{\mathrm{lo}}
\]

| Bucket | \((y_{\mathrm{lo}}, y_{\mathrm{hi}})\) | \(r(q)\) | Escalation warranted? |
| ------ | ----------------------- | -------- | --------------------- |
| easy | \((1,1)\) | 0 | No — \(M_{\mathrm{lo}}\) suffices |
| **opportunity** | \((0,1)\) | **1** | **Yes** — use \(M_{\mathrm{hi}}\) |
| lo\_only | \((1,0)\) | 0 | No |
| too\_hard | \((0,0)\) | 0 | No — \(M_{\mathrm{hi}}\) also fails |

**Oracle** \(\pi^*(q)\) (eval upper bound only): route to the **cheapest correct** pool member — for the primary pair: \(M_{\mathrm{lo}}\) if \(y_{\mathrm{lo}}=1\); else \(M_{\mathrm{hi}}\) if \(y_{\mathrm{hi}}=1\); else \(M_{\mathrm{lo}}\).

**Invariant:** \(r(q)\) is computed offline for layers 2–3; it is **never** an input to signal extraction (§5). Detail: §3.

**Important:** \(r(q)\), bucket rates, and signal behaviour emerge from the coupled Experimental Setting \(\mathcal{S}\) (§0.7). Stages **1–3** select \(\mathcal{S}\) in a reproducible order; Stage **3** freezes it before oracle runs.

### 0.7 Experimental Setting selection (Stages 1–3) — **FROZEN**

> **Architecture frozen (2026-06-25).** M1–M4 structure is locked. Further changes must improve **scientific validity**, not organization. Next design effort: **signal definitions (§5)**, oracle protocol (§3), feature vector (§7), routing policy (§8).

Pool, benchmark, protocol, entropy, and \(\kappa\) are **not independent**. An **Experimental Setting** is everything needed to run the routing study under one frozen protocol.

**Core notation (paper-facing):**

\[
\mathcal{S} = (\mathcal{D},\; \mathcal{P},\; \Pi)
\]

| Symbol | Meaning |
| ------ | ------- |
| \(\mathcal{D}\) | Benchmark / task (e.g. ARC-Challenge) |
| \(\mathcal{P}\) | Model pool with primary pair \((M_{\mathrm{lo}}, M_{\mathrm{hi}})\), capability spec, cost \(\kappa\) |
| \(\Pi\) | Protocol: prompt, template, decoding, grading |

Everything else — evaluation corpus \(C\), splits \((H, R_c, R_t)\), concrete features — is **derived from** \(\mathcal{S}\) via M1–M3 configuration. The expanded registry form (for YAML) is:

\[
\mathcal{S} \equiv \big(\,\mathcal{D},\; \mathcal{P},\; \Pi,\; C,\; \text{partition policy}\,\big)
\]

**Reproducible design process** — two phases, four modules, one **Experimental Setting** object (§0.7.1):

```text
PHASE A — Experimental setting selection (not hypothesis testing)
  M1  Experimental Setting Specification     → candidate grid + corpus partition + spec YAML
  M2  Setting Feasibility Assessment         → selection holdout pilot; bucket scorecard
  M3  Experimental Setting Lock              → winner → freeze IDs → setup table

PHASE B — Routing study (H1–H4)
  M4  Routing Evaluation Pipeline            → Stages 4–9 (oracle → … → evaluation)
```

> **Phase A exists solely to establish a valid experimental environment. It is not part of the hypothesis-testing pipeline.**

Execution stages 1–3 map to M1–M3; stages 4–9 map to M4. No main oracle until M3 completes.

**Selection principle (locked):** Phase A evaluates **experimental suitability**, not **expected routing performance**. It establishes that routing *can* be studied in this Experimental Setting — it must **not** pre-test whether proposed signals (entropy, disagreement, etc.) already look informative. Signal informativeness belongs in Stages **5–6** (H1–H3), after M3 lock.

Stages 1–3 (M1–M3) are **documented and pre-specified** before the pilot runs. M3 **commits** one primary \(\mathcal{S}^*\); M4 **only reads** the frozen Experimental Setting.

**Experimental Setting validity:** An Experimental Setting is **valid** if it satisfies the requirements necessary to evaluate routing policies under a fixed protocol (Gates A–E on selection holdout). Validity concerns the **experimental environment only** — it does **not** imply that the proposed routing signals will be effective. Signal effectiveness is tested in Phase B (H1–H4).

#### 0.7.1 Experimental Setting object (single config, all modules)

Every module consumes the same structure. M1 **writes** the spec; M2 **evaluates** candidates; M3 **freezes** one instance; M4 **reads** it read-only.

```yaml
# Template: experiments/setting.schema.yaml (M1 draft → M3 frozen)
setting:
  dataset:
    name: ARC-Challenge
    repo: allenai/ai2_arc
    config: ARC-Challenge

  evaluation_corpus:          # C — labeled queries for this study (M1)
    splits: [validation, test]   # HF loading detail; see §13 fact sheet
    exclude_splits: [train]      # few-shot / train only

  partition:                    # fixed counts — same pilot cost across benchmarks (not %)
    method: random_split
    seed: 42
    selection_holdout_n: 150      # M1/M2 holdout |H|
    test_n: 150                   # M3 test |R_t| (winning benchmark only)
    # |R_c| = |C| - selection_holdout_n - test_n  (remainder, M3)

  pool:                         # FROZEN in M1 before dataset pilot
    deployment_scenario: homogeneous_pool    # homogeneous_pool | heterogeneous_pool | api_router | edge_cloud
    vendor: meta-llama
    M_lo: meta-llama/Llama-3.2-3B-Instruct
    M_hi: meta-llama/Llama-3.1-8B-Instruct
    kappa: null                   # relative cost M_hi / M_lo — document at M3

  protocol:
    system_prompt: ...
    dataset_template: ...
    decoding: ...
    grading: objective_mcq

  splits:                       # populated at M3 freeze (query IDs)
    selection_holdout: []       # filled after M2
    calib: []
    test: []
```

**Evaluation corpus \(C\):** The set of labeled queries used for this study. **Methodology is defined entirely in terms of \(C\)** — not Hugging Face split names. M1 declares which queries compose \(C\) (HF splits are one loading mechanism; see §13 fact sheet).

```text
C                          (evaluation corpus — locked in M1)
 ↓
H ⊂ C                      (selection holdout — M2 pilot only; |H| = selection_holdout_n)
 ↓
R_c , R_t ⊂ C \ H          (calib + test — sizes locked in M1; IDs at M3)
```

| Split | Symbol | Role |
| ----- | ------ | ---- |
| Evaluation corpus | \(C\) | All labeled queries for Phase A partition + Phase B study |
| Selection holdout | \(H\) | M2 feasibility only; never reused in M4 |
| Calib | \(R_c\) | H1–H3, policy fit (Stages 6–8) |
| Test | \(R_t\) | H4 only (Stage 9) |

**Invariant:** \(H\), \(R_c\), and \(R_t\) are **disjoint**. No query appears in more than one split. **\(H\) is never reused** after M2 — not in oracle, signals, analysis, policy fit, or H4.

#### 0.7.2 Corpus partition policy (Option 1 — default)

**Default (recommended):** Treat publicly labeled evaluation data as one corpus \(C\), then partition **once** with a fixed seed:

\[
C \;\xrightarrow{\text{random split}}\; H \;\|\; R_c \;\|\; R_t
\]

| Option | Policy | When to use |
| ------ | ------ | ----------- |
| **1 (default)** | \(C\) = native validation + test (exclude train/dev); random split into \(H, R_c, R_t\) | Routing methodology paper — larger \(R_c\), consistent across benchmarks |
| **2 (conservative)** | Native validation → \(H \cup R_c\); native test → \(R_t\) only | Only if claiming official leaderboard scores on native test |
| **3 (forbidden)** | Reuse pilot queries in \(R_c\) or \(R_t\); or run pilot on \(R_t\) | **Never** — invalidates hypothesis testing |

This paper studies **unsupervised routing signals**, not benchmark leaderboard submission → **Option 1**.

**Paper sentence (locked):** *We treat the publicly labeled evaluation portion of each benchmark as an evaluation corpus \(C\). Before experimentation, we partition \(C\) into three disjoint subsets: a selection holdout \(H\) for experimental-setting selection only, a calibration split \(R_c\) for signal analysis and policy fitting, and a test split \(R_t\) for final routing evaluation. No example is reused across these stages.*

**Example (ARC-Challenge, Option 1 — sizes pre-specified in M1, not fixed by methodology):**

| Native (excluded from \(C\)) | Evaluation corpus \(C\) | After partition |
| ---------------------------- | ----------------------- | --------------- |
| train (1,119) — few-shot only | validation + test = **1,471** | \(H\)=150 · \(R_t\)=150 · \(R_c\)=1,171 |

Native HF split names do not appear in the partition rule — only in M1 config for loading \(C\). Under Option 1, \(H\), \(R_c\), and \(R_t\) are a **single random partition** of \(C\) (fixed seed); do not assign splits by native HF split name unless using Option 2.

**Deployment scenarios (pools):** The pool is **part of the problem statement** — a **fixed** weak/strong pair, not an optimization variable in Phase A. M1 commits **one primary scenario** for ACL v1 before any dataset pilot.

| Scenario | Definition | ACL v1 | Example pair |
| -------- | ---------- | ------ | ------------ |
| **`homogeneous_pool`** | Same vendor, architecture line, and tokenizer; minimizes confounds unrelated to capability. | **Primary** | Llama-3.2-3B-Instruct → Llama-3.1-8B-Instruct |
| **`heterogeneous_pool`** | Cross-family pool (different vendors, tokenizers, alignment). | **Out of scope v1** | Llama + Qwen |

**Pool selection criteria (M1 — before dataset pilot):** Choose the primary pair by **experimental validity**, not same-release purity.

| Criterion | Importance | Why |
| --------- | ---------- | --- |
| Same tokenizer / chat format | High | Avoid prompt confounds |
| Similar architecture (same vendor line) | High | Signals reflect capability, not family shift |
| Clear capability gap | **Very high** | Otherwise \(P(r(q)=1)\) is too rare (Gate D) |
| Affordable oracle (both models on every \(q\)) | **Very high** | Oracle dominates compute |
| Common in literature | Medium | Easier to justify |

**ACL v1 primary pair (frozen in M1):** `meta-llama/Llama-3.2-3B-Instruct` → `meta-llama/Llama-3.1-8B-Instruct`. Cross-release within Meta Llama is intentional — the paper tests routing between **different capabilities**, not Llama generations. Canonical YAML: `experiments/m1/pool.frozen.yaml`.

**Gap-size robustness (post-primary only — not Phase A grid):**

| Role | Pair | When |
| ---- | ---- | ---- |
| Small-gap ablation | 1B → 3B (both 3.2) | Optional — behavior when opportunity is scarce |
| Wide-gap robustness | 8B → 70B (3.1) | Optional §5.5 — if compute permits |

If the frozen homogeneous pool fails Gates A–E on **every** dataset candidate, **redesign the scenario** (document rationale) and restart Phase A — do not silently search additional pools in the main study.

**Paper sentence (pool):** *We select a fixed pool of two instruction-tuned Llama models with a clear capability difference while maintaining a common architecture and tokenizer, minimizing confounds unrelated to model capability.*

#### Phase A sequence (locked)

```text
Step 1  Choose deployment scenario          → homogeneous_pool (ACL v1)
Step 2  Instantiate concrete model pair     → 3B → 8B (HF ids in M1 YAML)
Step 3  FREEZE pool + shared protocol       → experiments/m1/pool.frozen.yaml
Step 4  M2 dataset pilot (H only)           → ≤4 datasets; pool identical every cell
Step 5  M3 freeze winning dataset           → one S* = (D*, P, Pi); split IDs
Phase B Routing study (M4)
```

**What varies in Phase A:** dataset \(\mathcal{D}\) only. **What is fixed:** \(\mathcal{P}\), \(\Pi\), partition policy, gates, tie-break.

#### What couples to what

| Component | What it determines downstream |
| --------- | ----------------------------- |
| **Benchmark** \(\mathcal{D}\) | Task difficulty; \(P(r(q)=1)\); bucket mix; query complexity |
| **Pool** \(\mathcal{M}\) | Capability spread; which members enter comparative signals; cost structure |
| **Primary pair** \((M_{\mathrm{lo}}, M_{\mathrm{hi}})\) | Capacity gap; entropy scale; rescue rate; relative cost \(\kappa\) |
| **Prompt + format** | Confidence, entropy, gradability, \(y(q,M)\) |
| **Decoding** | Output length, \(y(q,M)\), signal variance |
| **Grading rule** | Definition of \(r(q)\) and oracle buckets |
| **\(\kappa\)** | Pareto cost axis; value of escalation |

##### M1 — Experimental Setting Specification (Stage 1)

**Goal:** Lock **pool + protocol + partition policy** and list **dataset candidates** before any inference. Pool is frozen **before** the dataset pilot — not co-selected with benchmarks.

```text
Pool (frozen):     homogeneous_pool — Llama-3.2-3B → Llama-3.1-8B
Protocol:          system prompt + dataset templates + shared decoding/grading
Dataset candidates:{ARC-Challenge, MMLU, TruthfulQA-MC, HellaSwag?}
Corpus + partition per candidate:  evaluation corpus C + holdout/calib/test sizes
Gates A–E, tie-break, selection_holdout_n
Output:            pool.frozen.yaml + one YAML per dataset candidate
```

**Deliverable:** `experiments/m1/pool.frozen.yaml` + dataset candidate YAMLs (`experiments/candidates/*.yaml`) — schema: `experiments/setting.schema.yaml`.

##### M2 — Setting Feasibility Assessment (Stage 2)

**Goal:** Run dataset-only pilot on **selection holdout** \(H \subset C\); bucket scorecard; Gates A–E per candidate \(\mathcal{S} = (\mathcal{D}, \mathcal{P}, \Pi)\) with **fixed** \(\mathcal{P}, \Pi\).

Sample holdout from \(C\) only (sizes from M1). **No signal statistics.** Recommend winning dataset for M3.

**Deliverable:** bucket scorecard + pass/fail per dataset (pool column omitted — pool is constant).

##### M3 — Experimental Setting Lock (Stage 3)

**Goal:** Administrative freeze only — feasibility was M2.

1. **Winner** — one primary \(\mathcal{S}^* = (\mathcal{D}^*, \mathcal{P}, \Pi)\) with \(\mathcal{P}, \Pi\) unchanged since M1 (pre-specified tie-break).
2. **Freeze IDs** — instantiate `splits.selection_holdout`, `splits.calib`, `splits.test` from M1 partition rule applied to \(C\).
3. **Setup table** — `paper/tables/T1_setup.tex` + frozen Experimental Setting YAML.

**Not in M3:** split policy design (M1), feasibility checks (M2), second pilot, pool re-selection.

**Deliverable:** frozen Experimental Setting file + setup table. **Gate to M4.**

##### M4 — Routing Evaluation Pipeline (Stages 4–9)

Read-only frozen Experimental Setting. Oracle → signals → H1–H3 (calib) → policy fit (calib) → H4 (test).

---

#### Benchmark selection (Phase A detail)

Benchmark selection answers one question: **Is this a scientifically valid environment to test our hypotheses?** — not *which benchmark makes our signals look best*.

**Not a contribution:** Benchmark selection is **not** part of the scientific contribution. It is a **reproducibility protocol** designed to prevent ad hoc benchmark and model selection before hypothesis testing.

**Why \(\mathcal{S} = (\mathcal{D}, \mathcal{P}, \Pi)\)?** Routing feasibility depends jointly on task, pool, and protocol. M2 selects **\(\mathcal{D}\)** only; \(\mathcal{P}\) and \(\Pi\) are frozen in M1.

**M2 grid:** ≤ 4 **datasets** × **1 frozen pool** = ≤ 4 candidate cells. Pilot cost = **150 queries × 2 models** per dataset (fixed in M1).

**Selection-bias guard:** no pool search; no entropy/disagreement thresholds; no AUROC vs \(r(q)\); no post-hoc numeric cutoffs.

**Protocol layers (Gate B):** system prompt · dataset templates · shared decoding/grading — all locked in M1 with the pool.

**Selection holdout:** Sample \(H \subset C\) (evaluation corpus) only. \(H\) is never reused in M4. Partition \(C \setminus H\) into calib + test per M1 policy (§0.7.1).

##### Feasibility gates (M2)

| Gate | Question | Pass |
| ---- | -------- | ---- |
| **A** | Objective MCQ grading? | Yes |
| **B** | Frozen protocol layers? | Yes |
| **C** | Meaningful capability gap? \(\mathrm{Acc}(M_{\mathrm{hi}}) - \mathrm{Acc}(M_{\mathrm{lo}}) \ge\) `min_accuracy_gap` | Yes |
| **D** | Routing feasible? opportunity rate ≥ `opportunity_min`; too_hard rate < `too_hard_max` | Yes (rationale below) |
| **E** | Full M4 oracle affordable on \(R_c \cup R_t\)? | Yes |

**Tie-break (pre-specified in M1):** among gate passers, prefer larger `acc_gap`, then higher `opportunity_rate`, then literature preference (`ARC-Challenge` → `MMLU` → `TruthfulQA-MC` → `HellaSwag`).

**Gate C rationale (locked — for reviewers):** *The minimum accuracy gap excludes settings where the designated strong model is only marginally better than the weak model on the holdout — differences that are often sampling noise at pilot size and do not justify a weak→strong routing study.* Concrete value (`min_accuracy_gap` = 0.03) is pre-specified in `experiments/phase_a_defaults.yaml` before any pilot runs.

**Gate D rationale (locked — for reviewers):** *Thresholds exclude degenerate routing settings — too few routing opportunities to study, or too many queries where even the strong model fails — rather than optimize routing performance.* Concrete values (`opportunity_min` = 0.05, `too_hard_max` = 0.70) are pre-specified in `experiments/phase_a_defaults.yaml` before any pilot runs. There is no upper bound on opportunity rate: a high opportunity share means the weak model fails often while the strong model rescues, which is a valid (if aggressive-gap) routing scenario when `too_hard` remains low.

##### M2 bucket scorecard

Pool fixed: Llama-3.2-3B → Llama-3.1-8B (`homogeneous_pool`).

| \(\mathcal{D}\) | easy | opportunity | lo\_only | too\_hard | Acc gap | Cost | Pass? |
| --------------- | ---- | ----------- | -------- | --------- | ------- | ---- | ----- |

#### What Stage 0 locks vs what M3 locks

| Locked in §0 (method) | Locked in M1 | Locked in M3 (instance) |
| --------------------- | ------------ | ------------------------- |
| Definition of \(r(q)\), \(\pi^*\) | \(\mathcal{P}\), \(\Pi\), partition **rule**, dataset **candidates** | Which \(\mathcal{D}^*\) |
| Signal **assumptions** + three **types** (§5) | Primary pair \(M_{\mathrm{lo}}, M_{\mathrm{hi}}\) | Concrete features for this \(\mathcal{S}\) |
| H1–H4 structure | Gates, tie-break, `selection_holdout_n` | Frozen Experimental Setting YAML + split IDs |
| Pareto eval | — | \(\kappa\), prompt text (measured/documented) |

### 0.8 Signal assumptions

Each signal type rests on a **routing-relevant assumption** — *why* it might predict \(r(q)=1\). These motivate H1–H3; the hypotheses **test** the assumptions empirically. A rejected hypothesis means the assumption failed for this Setting — not that the study failed.

| Type | Assumption | Link to routing need | Hypothesis |
| ---- | ---------- | -------------------- | ---------- |
| **Query-derived** | More complex or ambiguous queries are more likely to exceed \(M_{\mathrm{lo}}\) capacity while remaining within \(M_{\mathrm{hi}}\) capacity. | Higher query complexity → higher \(P(r(q)=1)\) | **H1** |
| **Model-response** | Higher model uncertainty reflects higher probability of error on that query. | Higher uncertainty on \(M_{\mathrm{lo}}\) → higher \(P(y_{\mathrm{lo}}=0)\); when \(y_{\mathrm{hi}}=1\), routing need | **H2** |
| **Cross-model comparative** | Differences between pool members on the same query reveal where escalation changes the outcome. | Larger capability-gap signals (confidence, entropy, disagreement) → higher \(P(r(q)=1)\) | **H3** |

**Literature grounding:** query complexity and task difficulty (Hybrid LLM, FrugalGPT); uncertainty and error (calibration literature); model disagreement and complementary strengths (RouterBench).

Full detail and examples: **§5**. Do not restate assumptions in §6 or §10 — cross-ref §5.

### 0.9 Signal taxonomy (conceptual)

Three types — nested for H1–H3 ablations:

| Type | Input needed | Role in ablation |
| ---- | ------------ | ---------------- |
| **Query-derived** | Query text only | H1 baseline set |
| **Model-response** | One model × query (logits, entropy, confidence, …) | H2 adds to H1 |
| **Cross-model comparative** | Both pool members (Δ entropy, disagreement, …) | H3 adds to H2 |

Concrete feature list is chosen at Stage **3** lock and documented in `paper/tables/T1_setup.tex`. Conceptual examples and assumptions: §5.

### 0.10 Scientific pipeline

Two threads run through five pipeline steps (§4):

```text
Thread A — Signal understanding          Thread B — Routing performance
  H1–H3 on calib                           H4 on test
       ↓                                        ↓
Signals → Analysis → Selection ────────→ Policy → Evaluation
  §5        §6         §7                  §8        §9
 layer 1   layer 2    layer 1–2           layer 3    test only
```

| Step | Layer | Split | Output |
| ---- | ----- | ----- | ------ |
| Signals | 1 | no \(r(q)\) | Raw inventory (Stage **5**) |
| Signal analysis | 2 | **calib** | Informativeness; H1–H3 (Stage **6**) |
| Signal selection | 1–2 | calib | Frozen \(x(q)\) (Stage **7**) |
| Routing policy | 3 | **calib** fit; **test** eval | Locked \(\pi(q)\) (Stage **8**) |
| Evaluation | — | **test only** | Pareto; H4 (Stage **9**) |

**Lock rules (design-level):**

1. After **Stage 7:** do not refit or expand \(x(q)\) using test labels.
2. After **Stage 8:** do not tune \((\lambda,\tau)\) on test.
3. Signal extraction must not use \(r(q)\) as input.
4. After **Stage 3:** do not change benchmark, pool, or protocol without restarting from Stage 1.

### 0.11 Hypotheses and tests

Motivated by **signal assumptions** (§0.8, §5).

| ID | Hypothesis | Layer | Exec. stage | Split | Test statistic / criterion |
| -- | ---------- | ----- | ----------- | ----- | -------------------------- |
| **H1** | Query-derived signals predict routing need. | 2 | **6** | calib | AUROC / AUPRC vs \(r(q)\); query-derived only |
| **H2** | Model-response improves prediction beyond query-derived. | 2 | **6** | calib | ΔAUROC over H1 (nested) |
| **H3** | Cross-model comparative improves beyond model-response. | 2 | **6** | calib | ΔAUROC over H2 (nested) |
| **H4** | Learned weighting outperforms manual rules on same \(x(q)\). | 3 | **9** | test | 4b **Pareto-dominates** best 4a |

**H4 tracks (policy fit on calib, eval on test):**

| Track | \(\lambda\) | \(\tau\) |
| ----- | ----------- | -------- |
| **4a Rule** | Hand-specified (e.g. single-feature weights) | Sweep on calib; lock |
| **4b Learned** | Fit on calib (e.g. logistic regression on \(r(q)\)) | Tune on calib; lock |

Score and policy: nomenclature §2.1 — \(s(q)=\lambda^\top x(q)\), threshold \(\pi(q)\). Detail: §8.

**Negative results:** Any hypothesis may fail; report H1–H3 and H4 **separately** (§0.12, §14).

### 0.12 Evaluation design

**Primary metrics (test only):**

\[
\mathrm{Acc}(\pi) = \frac{1}{|\mathcal{Q}_{\text{test}}|} \sum_{q} y\big(q, \pi(q)\big), \qquad
\mathrm{Cost}(\pi) = \frac{1}{|\mathcal{Q}_{\text{test}}|} \sum_{q} c\big(\pi(q)\big)
\]

with \(c(M_{\mathrm{lo}})=1\), \(c(M_{\mathrm{hi}})=\kappa>1\) fixed from the locked pool.

**Primary comparison:** **Pareto** curves over \((\mathrm{Acc}, \mathrm{Cost})\). Report **dominance** vs baselines and **oracle gap** vs \(\pi^*\).

**Baselines:** always-\(M_{\mathrm{lo}}\) · always-\(M_{\mathrm{hi}}\) · oracle \(\pi^*\) · best 4a · 4b.

**Optional (calib only):** scalar \(J_\alpha(\pi)=\mathrm{Acc}(\pi)-\alpha\,\mathrm{Cost}(\pi)\) for tuning — reported numbers always from test.

**Splits (project):**

| Split | Role | Labels used for |
| ----- | ---- | ---------------- |
| **selection holdout** | M2 feasibility pilot only | Buckets, gates A–E |
| **calib** | H1–H3, policy fit (Stages 6–8) | \(r(q)\), \(y(q,M)\) |
| **test** | H4 only (Stage 9) | \(y(q,M)\) for metrics — no tuning |

**Evaluation corpus partition (locked in M1):** For each benchmark, declare corpus \(C\) (which HF splits compose it). Apply one rule: \(C \to H\) (holdout) \(\to R_c\) (calib) + \(R_t\) (test). Methodology does not depend on HF split names — only M1 corpus config does. Detail: §0.7.1, §13.

**Joint feasibility gate (M2):** Gates A–E on selection holdout before M3.

Full metrics and baselines: §9.

### 0.13 Contributions (paper claims)

1. **Formulation** — \(r(q)\), signal assumptions + taxonomy, supervised vs unsupervised contrast (§2–§3, §5).
2. **Signal analysis** — nested H1–H3 tests on calib (§6).
3. **Signal selection** — frozen \(x(q)\) before routing claims (§7).
4. **Routing policies** — 4a rule + 4b learned via §2.1 (§8).
5. **Pareto evaluation** — accuracy and cost jointly on test (§9).

Paper drafts: `paper/sections/01_introduction.tex`, `03_problem_definition.tex`.

### 0.14 Scope, assumptions, and exclusions

**In scope:** §0 design · signal assumptions (§5) · three-layer unsupervised definition · reproducible Stages 1–3 · honest negative results.

**Out of scope:** claiming “no labels ever”; supervised router reproduction; benchmark chasing; agent/task decomposition; reusing retired configs as defaults; **tuning prompt or pool on test** to rescue a failed setting.

**Method-level assumptions (not setting-specific):**

- Fixed **model pool** with distinct capabilities **once Stage 3 locks Setting**; primary pair \((M_{\mathrm{lo}}, M_{\mathrm{hi}})\) designated.
- Objective, automated task grading for \(y(q,M)\) **under the locked protocol**.
- Same frozen protocol for oracle labeling, signal extraction, and routing evaluation.

**Setting-level assumptions (validated in M2):** non-degenerate \(P(r(q)=1)\); \(\mathrm{Acc}(M_{\mathrm{lo}}) < \mathrm{Acc}(M_{\mathrm{hi}})\); \(\kappa>1\) documented. Signal informativeness is **not** assumed at selection — tested in H1–H3.

**Success criterion:** Answer §10 / §0.11 honestly — report §6 and §9 as separate result sections regardless of outcome.

Detail: §14. Advisor alignment: §15.

### 0.15 Stage 0 deliverables and gate to Stage 1

| Deliverable | Location | Status |
| ----------- | -------- | ------ |
| Research design (this section) | program §0 | ✓ locked |
| Signal assumptions | §0.8 · §5 | ✓ |
| Problem + RQs | program §1 · paper §3 | ✓ |
| Supervision definition | program §2 · paper §3 | ✓ |
| Routing need + oracle | program §3 | ✓ |
| Pipeline + hypotheses | program §4, §10 | ✓ |
| Eval plan | program §9 · paper §5 | ✓ |
| Scope | program §14 | ✓ |
| Paper skeleton | `paper/main.tex` | ✓ (results TBD) |

**Gate to M1:** §0 method + Experimental Setting architecture are complete. **Recommended order from here:**

1. **Signal definitions (§5)** — operationalize query complexity, entropy, paraphrase stability, cross-model comparison *(scientific contribution)*
2. **Oracle protocol (§3)** — full labeling procedure on \(R_c \cup R_t\)
3. **Feature vector (§7)** — schema for \(x(q)\)
4. **Routing policy (§8)** — 4a/4b tracks
5. **M1 → M2 → M3** — fill candidate YAMLs, run feasibility pilot, lock \(\mathcal{S}^*\)
6. **M4** — implement routing study (read-only frozen Experimental Setting)

Do **not** begin M4 until M3 completes. Do **not** redesign M1–M4 structure.

### 0.16 Paper voice (locked)

Use **enable** / **guide routing decisions**. Avoid *support routing*, *solve routing*, Family 1/2/3. See nomenclature §1.

### 0.17 Tuning policy (frozen)

> **Tune only the routing policy. Everything else is frozen.**

| Component | Tuned? | When locked | Notes |
| --------- | ------ | ----------- | ----- |
| Benchmark, pool, protocol | **No** | M3 | Selected in Phase A; not optimized on routing metrics |
| Prompt, decoding, grading | **No** | M1 / M3 | Fixed \(\Pi\); changing decoding would confound entropy signals |
| Signal definitions | **No** | M3 / §7 | Research contribution — pre-specified, not fit to \(r(q)\) |
| Signal extraction \(x(q)\) | **No** | Stage 5 | Computed from query/model; no thresholds tuned on calib |
| Signal thresholds (e.g. length > 30) | **No** | M1 / §5 | Use raw features or literature-fixed cutoffs only |
| Routing weights \(\lambda\) | **Yes** | Stage 8 | Fit on \(R_c\) only (4b); 4a uses hand-specified \(\lambda\) |
| Routing threshold \(\tau\) | **Yes** | Stage 8 | Tuned on \(R_c\) only; locked before \(R_t\) |
| Final evaluation on \(R_t\) | **No** | Stage 9 | One-shot; no refit of \(x(q)\), \(\lambda\), or \(\tau\) |
| Phase A pilot (M2) | **No** | — | Suitability gates only; no prompt/temperature/signal tuning |

**Learned parameters:** \(\lambda\) and \(\tau\) only — both on **calib** \(R_c\). Detail: §8, nomenclature §2.1.

**Optional:** scalar \(J_\alpha\) for calib-only operating-point selection — reported numbers always from \(R_t\) (§9).

---

## 1. Problem

> **Full research design:** §0 (Stage 0, locked). This section is the concise problem statement for cross-reference.

Many systems route each query to one LLM from a pool to save **cost** and **latency**. Recent work (**RouteLLM**, **GraphRouter**, **RouterBench**, …) trains **supervised routers** on preferences, outcomes, or labels.

**Our question:** Given a query and a fixed pool, can **unsupervised signals** **enable routing decisions** — and does **learning weights** on calib improve over simple rules? **Unsupervised** is defined precisely in **§2** (not “no labels anywhere”).

**Sub-questions (advisor meeting):**

1. What **query-derived**, **model-response**, and **cross-model comparative** signals can we estimate?
2. How should those signals be combined into a score?
3. Do those scores **guide** model choice vs baselines?

**One sentence (Intro):** We study whether unsupervised signals — query-derived, model-response, and cross-model comparative — can **enable routing decisions** between LLMs in a fixed pool, and whether **learned combination weights** on calib improve over hand-written rules — vs **supervised** routers in prior work.

**Paper voice:** **Enable** / **guide routing decisions** — not *support* (advisor-cautious) or *solve routing*.

**Not the problem:** agent orchestration, task decomposition, “pre-inference routing” as the headline (contrast is **supervised vs unsupervised**, §2).

### Contributions

See §0.13 for the full list. Two threads: **signal understanding** (H1–H3, §6) and **routing** (H4, §8–§9).

---

## 2. What “unsupervised” means

**Defined once here.** Other sections refer back; do not re-define.

### In prior routing work (literature)

| Approach | What is “supervised” | Examples |
| -------- | -------------------- | -------- |
| **Supervised routing** | Router trained on **routing labels** — human preferences, win/loss between models, reward-model scores, task-type labels | RouteLLM (Chatbot Arena prefs); Hybrid LLM (DeBERTa + BARTScore labels); GraphRouter (task nodes); Zooter (reward-model labels) |
| **Other “unsupervised” uses** | No **preference** labels for the router; may still use clustering, similarity, or **multiple generations** | Prompt embedding k-means (UniRoute: clustering step); semantic entropy (Kuhn: sample then cluster); similarity routing in surveys |

**Field gap we target:** Most LLM routers are **supervised** in the table’s first row. We ask whether **routing-relevant information** exists in signals computed **without** that kind of routing training data.

### Advisor notes (summary)

Prior work **learns** routing from labels; we **propose and measure** fixed signals, then combine them with calib labels. Main contrast: **supervised vs unsupervised at signal extraction** — not pre- vs post-inference timing. Full quotes: §15.

### Our definition (three supervision layers)

**Unsupervised** describes **how signals are obtained**, not “the entire system never sees a label.”

| Layer | Program § | Uses \(r(q)\) or routing prefs? | Unsupervised? |
| ----- | ---------- | ------------------------------- | ------------- |
| **1 — Signal extraction** | Signals (§5) | **No** | **Yes** — \(x_i(q)\) from query / model only; no routing classifier training |
| **2 — Signal analysis** | Analysis (§6) | **Yes, calib only** — to score informativeness | **Analysis is supervised; signals stay unsupervised** |
| **3 — Policy fit** | Routing policy (§8) | **Yes, calib only** — fit \((\lambda,\tau)\) | **Combination is supervised; signal definitions stay unsupervised** |

**Short label for the paper:** *Routing from unsupervised signals* — signals in layer 1 are unsupervised; layers 2–3 use offline \(r(q)\) on calib to **evaluate** and **combine**, not to invent features.

### What we do **not** mean

- **Not** “zero labels in the whole paper” — we use \(r(q)\) on calib (§3, §6, §8).
- **Not** classical unsupervised learning (e.g. k-means router with no oracle) as the method.
- **Not** that **model-response** signals avoid model inference — they require running \(M_i\) on \(q\); “unsupervised” refers to **routing supervision**, not compute.
- **Not** the same as every paper that says “unsupervised” (e.g. RAG retriever routing with synthetic judge scores).

### Contrast table (Intro / Related Work)

| | Typical supervised router | This work |
| - | ------------------------- | --------- |
| Router input | Classifier on query (+ history), trained on prefs/outcomes | **Unsupervised signals** → \(x(q)\) |
| Where labels enter | **Define** the router end-to-end | **Only** calib: analysis (§6) + \((\lambda,\tau)\) fit (§8) |
| Signal source | Learned representations from routing data | Query-derived, model-response, cross-model comparative (§5) |

---

## 3. Routing need

**Setup:** Model pool \(\mathcal{M}=\{M_1,\ldots,M_K\}\) with distinct capabilities; query \(q\); full inference under frozen protocol. Stage 3 designates primary pair \(M_{\mathrm{lo}}\), \(M_{\mathrm{hi}}\) with \(\mathrm{cap}(M_{\mathrm{lo}}) < \mathrm{cap}(M_{\mathrm{hi}})\).

\[
y(q, M) \in \{0, 1\}, \quad y_{\mathrm{lo}} = y(q, M_{\mathrm{lo}}),\; y_{\mathrm{hi}} = y(q, M_{\mathrm{hi}})
\]

\[
r(q) = \mathbb{1}\big[\, y_{\mathrm{lo}} = 0 \;\wedge\; y_{\mathrm{hi}} = 1 \,\big], \qquad \Delta(q) = y_{\mathrm{hi}} - y_{\mathrm{lo}}
\]

| Bucket | \((y_{\mathrm{lo}}, y_{\mathrm{hi}})\) | \(r(q)\) | Oracle route |
| ------ | ----------------------- | -------- | ------------ |
| easy | \((1,1)\) | 0 | \(M_{\mathrm{lo}}\) |
| opportunity | \((0,1)\) | **1** | \(M_{\mathrm{hi}}\) |
| lo\_only | \((1,0)\) | 0 | \(M_{\mathrm{lo}}\) |
| too\_hard | \((0,0)\) | 0 | \(M_{\mathrm{lo}}\) |

**Oracle** \(\pi^*(q)\): cheapest correct pool member — on the primary pair: \(M_{\mathrm{lo}}\) if \(y_{\mathrm{lo}}=1\); else \(M_{\mathrm{hi}}\) if \(y_{\mathrm{hi}}=1\); else \(M_{\mathrm{lo}}\). Evaluation upper bound only (§9).

\(r(q)\) is offline (layer 2–3 in §2) — never an input to signal extraction (§5). Pipeline steps: §4.

---

## 4. Pipeline

Signal work and routing work are **separate**. **Calib** \(R_c\) for analysis and policy fit; **test** \(R_t\) for H4 only. Corpus partition locked in M1; IDs frozen in M3 (§0.7.1).

```text
Signals  →  Signal analysis  →  Signal selection  →  Routing policy  →  Evaluation
  §5              §6                  §7                  §8              §9
```

| Step | Layer (§2) | Output |
| ----- | ---------- | ------ |
| Signals | 1 | Raw signal inventory |
| Signal analysis | 2 | Informativeness tables (§6) |
| Signal selection | 1–2 | Frozen \(x(q)\) schema (§7) |
| Routing policy | 3 | Locked \(\pi(q)\) (§8) |
| Evaluation | — | \((\mathrm{Acc}, \mathrm{Cost})\) Pareto (§9) |

Hypotheses **H1–H4**: §10 only. **Execution order:** §11 (Stages 0–9).

---

## 5. Signals

### Signal assumptions

Each signal type rests on a **routing-relevant assumption** — why it might predict \(r(q)=1\). H1–H3 **test** these assumptions; they are not axioms.

| Type | Assumption | Predicted link to \(r(q)\) | Hypothesis |
| ---- | ---------- | -------------------------- | ---------- |
| **Query-derived** | More complex or ambiguous queries are more likely to require a higher-capability pool member. | Higher query complexity → higher \(P(r(q)=1)\) | **H1** |
| **Model-response** | Higher uncertainty on a model reflects higher probability of error on that query. | Higher \(H(q\mid M_{\mathrm{lo}})\) or lower confidence → higher \(P(y_{\mathrm{lo}}=0)\); when \(y_{\mathrm{hi}}=1\), routing need | **H2** |
| **Cross-model comparative** | Differences between pool members on the same query reveal routing opportunities. | Larger confidence/entropy gap or disagreement → higher \(P(r(q)=1)\) | **H3** |

**If an assumption fails** (e.g.\ complexity uncorrelated with \(r(q)\)), rejecting H1 is a valid, publishable outcome for this Setting.

Canonical summary: §0.8. Do not duplicate in §6 or §10.

### Signal types

Concept only --- concrete signal list is chosen at **Stage 3** lock (§13); not fixed in this program.

| Signal type | Needs | Examples |
| ----------- | ----- | -------- |
| **Query-derived** | Query text only | Complexity, length, lexical/syntactic cues, ambiguity |
| **Model-response** | One model × query | Entropy Q\|Mᵢ, confidence, log-prob, paraphrase stability |
| **Cross-model comparative** | Compare pool members | ΔH, confidence gap, disagreement |

**Cross-model comparative** distinguishes between-model signals from within-model model-response signals.

Why three types: query-derived needs no model run; model-response is within-model; cross-model comparative is between-model.

**Pipeline output:** candidate features (layer 1, §2). Selection: §7 after §6.

---

## 6. Signal analysis

**Purpose:** Which unsupervised signals predict \(r(q)\)? — layer 2; does not redefine signals.

**Split:** calib only; \(r(q)\) from §3.

| Analysis | Statistics |
| -------- | ---------- |
| Univariate | Spearman \(\rho\), AUROC / AUPRC per feature |
| Nested ablations | ΔAUROC — **§10 H1–H3** |
| Bucket view (optional) | Stratified plots over \((y_{\mathrm{lo}}, y_{\mathrm{hi}})\) |

No \(\pi(q)\) claims. **Deliverable:** analysis table.

---

## 7. Signal selection

**Input:** §6 on calib. **Output:** frozen feature specification and \(x(q)\).

| Rule | Action |
| ---- | ------ |
| Redundant | Drop (high correlation, no ΔAUROC) |
| Non-informative | Drop unless needed for §10 ablation narrative |
| Final | Compact \(x(q) \in \mathbb{R}^d\) for §8–§9 |

**Deliverable:** keep / drop / merge table.

---

## 8. Routing policy

Uses frozen \(x(q)\) from §7. Layer 3 (§2). **Score and router:** [`nomenclature.md`](nomenclature.md) **§2.1** (not repeated here).

**Only learned parameters:** \(\lambda\) and \(\tau\) — fit on calib \(R_c\) only; lock before test \(R_t\). Nothing else is tuned (§0.17).

| Track | \((\lambda, \tau)\) |
| ----- | ------------------- |
| **4a Rule** | Hand-specified \(\lambda\); sweep \(\tau\) on calib |
| **4b Learned** | Fit \(\lambda\) on calib (e.g. logistic on \(r(q)\)); tune \(\tau\); lock for test |

**H4:** §10. Quality: §9.

---

## 9. Evaluation

**Split:** project **test** \(R_t\) only (subset of evaluation corpus \(C\); partition locked in M1, IDs in M3 — §0.7.1, §13). Separate from §6.

### Metrics

\(c(M_{\mathrm{lo}})=1\), \(c(M_{\mathrm{hi}})=\kappa>1\).

\[
\mathrm{Acc}(\pi) = \frac{1}{|\mathcal{Q}_{\text{test}}|} \sum_{q} y\big(q, \pi(q)\big), \qquad
\mathrm{Cost}(\pi) = \frac{1}{|\mathcal{Q}_{\text{test}}|} \sum_{q} c\big(\pi(q)\big)
\]

**Optional calib only:** \(J_\alpha(\pi) = \mathrm{Acc}(\pi) - \alpha \cdot \mathrm{Cost}(\pi)\).

### Baselines and comparison

**Baselines:** always-\(M_{\mathrm{lo}}\) · always-\(M_{\mathrm{hi}}\) · \(\pi^*\) (§3) · 4a and 4b from §8.

**Primary:** Pareto \((\mathrm{Acc}, \mathrm{Cost})\). Report **dominance** vs baselines and **oracle gap** vs \(\pi^*\).

Do not tune on test.

---

## 10. Hypotheses

Motivated by **signal assumptions** (§5). Each hypothesis tests one assumption on the locked Setting.

| ID | Hypothesis | Assumption | Program § | Exec. stage | Test |
| -- | ---------- | ---------- | --------- | ----------- | ---- |
| **H1** | Query-derived signals predict routing need. | Query complexity → routing need | §6 | **6** | AUROC / AUPRC vs \(r(q)\); query-derived only |
| **H2** | Model-response improves prediction beyond query-derived. | Uncertainty → error | §6 | **6** | ΔAUROC over H1 |
| **H3** | Cross-model comparative improves beyond model-response. | Disagreement → opportunity | §6 | **6** | ΔAUROC over H2 |
| **H4** | Learned weighting outperforms manual rules. | (policy; not signal assumption) | §8–§9 | **9** | 4b Pareto-dominates best 4a on test |

H1–H3: **informativeness** (layer 2). H4: **routing** (layer 3). Either may fail (§14).

---

## 11. Execution workflow (Stages 0–9)

**Two views:** §4 is the **scientific pipeline** (what the paper claims). This section is the **execution workflow** (what you run, in order). Use **Stage** here for execution only.

```text
Stage 0   Research design
    ↓
PHASE A — Experimental setting selection
  M1 / Stage 1   Experimental Setting Specification
  M2 / Stage 2   Setting Feasibility Assessment      [selection holdout]
  M3 / Stage 3   Experimental Setting Lock           [freeze IDs → setup table]
    ↓
PHASE B — Routing study
  M4 / Stage 4   Oracle labeling
  M4 / Stage 5   Signal extraction
  M4 / Stage 6   Signal analysis                     [calib — H1–H3]
  M4 / Stage 7   Signal selection
  M4 / Stage 8   Routing policy fit                  [calib]
  M4 / Stage 9   Routing evaluation                  [test — H4]
```

**Design principle:** Phase A establishes a valid Setting; Phase B answers H1–H4. M4 consumes the frozen Setting only.

### Stage map

| Stage | Name | Program § | Split | Output / gate |
| ----- | ---- | --------- | ----- | ------------- |
| **0** | **Research design** | **§0**, §1–§3, §5, §10, §14 | — | RQs, assumptions, H1–H4, eval plan (**method locked**) |
| **1** | **M1 — Experimental Setting Specification** | §0.7 | — | Setting spec + candidate YAMLs |
| **2** | **M2 — Setting Feasibility Assessment** | §0.7 | selection **holdout** | Bucket scorecard; gates A–E |
| **3** | **M3 — Experimental Setting Lock** | §0.7 | — | **Frozen Setting** YAML + setup table |
| **4** | **Oracle labeling** | §3 | calib + test | \(y(q,M)\), \(r(q)\), \(\pi^*(q)\) |
| **5** | **Signal extraction** | §5 | no \(r(q)\) | Raw signal inventory (three types) |
| **6** | **Signal analysis** | §6 | **calib** | Informativeness; **H1–H3** |
| **7** | **Signal selection** | §7 | calib | Frozen \(x(q)\) and feature specification |
| **8** | **Routing policy fit** | §8 | **calib** | Locked \(\pi(q)\); 4a + 4b |
| **9** | **Routing evaluation** | §9 | **test only** | Pareto; **H4** |

### Terminology (locked)

| Use | Avoid |
| --- | ----- |
| **M1 / M2 / M3** module names | “Pick dataset” without Setting spec |
| **Evaluation corpus** + **partition** | Hard-coded “HF validation → calib” in methodology |
| **Deployment scenario** | Arbitrary Pool A / Pool B |
| **Selection holdout** | Reusing holdout in calib/test |
| **Oracle labeling** | “Oracle generation” (labeling preferred) |
| **Routing policy fit** | *Routing policy learning* |
| **Signal analysis** | *Analysis & paper* (conflicts with Stage 6) |

**Lock rules:** After Stage **3**, no benchmark/pool/protocol changes without restarting Stages 1–3. After Stage **7**, no refit \(x(q)\) on test. After Stage **8**, no tune \((\lambda,\tau)\) on test.

### Mapping §4 ↔ Stages 5–9

| §4 pipeline step | Execution stage |
| ---------------- | --------------- |
| Signals (§5) | **5** Signal extraction |
| Signal analysis (§6) | **6** Signal analysis |
| Signal selection (§7) | **7** Signal selection |
| Routing policy (§8) | **8** Routing policy fit |
| Evaluation (§9) | **9** Routing evaluation |

Stages 0–3: design + **reproducible setting selection**. Stages 4–9: main study.

### Paper map

Intro · Related Work · Problem · Method (Stages 5–8) · Setup (Stages 1–4) · Results H1–H3 (Stage 6) · Results H4 (Stage 9) · Discussion.

**Checklist:** §0 ✓ · Stages 1–3 (not started) · Stages 4–9 (not started) · paper skeleton ✓.

---

## 12. Artifacts

| Stage | Output location |
| ----- | ---------------- |
| **1** | M1 Setting spec + candidate YAMLs |
| **2** | M2 feasibility scorecard |
| **3** | M3 frozen Setting YAML + setup table |
| **4–9** | Oracle labels, signals, analysis, policies, evaluation results |

Store under `experiments/` by stage. Implementation mapping (future): [`../scripts/README.md`](../scripts/README.md).

**Status:** research design ✓ · workflow ✓ · paper skeleton ✓ · **Stages 1–9 not started**.

---

## 13. Experimental setting (TBD — Stages 1–3)

**Nothing locked.** Benchmark, pool, and protocol are selected via **Phase A (M1–M3)** (§0.7) — joint pilot on selection holdout, then frozen together. Retired choices under `old_llm_routing/` are not design authority. [`literature_record.md`](literature_record.md) is a **candidate catalog** only.

**Selection principle:** Benchmark selection evaluates **experimental suitability**, not **expected routing performance** (§0.7).

### Fresh-start status (Stages 0–9)

| Stage | Name | Status |
| ----- | ---- | ------ |
| **0** | Research design | **§0 method locked** · signal assumptions ✓ |
| **1** | M1 — Experimental Setting Specification | **Not started** |
| **2** | M2 — Setting Feasibility Assessment | **Not started** |
| **3** | M3 — Experimental Setting Lock | **Not started** |
| **4** | Oracle labeling | **Not started** |
| **5** | Signal extraction | **Not started** |
| **6** | Signal analysis (H1–H3) | **Not started** |
| **7** | Signal selection | **Not started** |
| **8** | Routing policy fit | **Not started** |
| **9** | Routing evaluation (H4) | **Not started** |

**No** locked IDs or `experiments/` artifacts yet.

### M1 — Experimental Setting Specification (Stage 1)

See §0.7 M1 checklist + `experiments/setting.schema.yaml`. **Deliverable:** Setting spec record (§12).

#### Phase 1 candidates — dataset fact sheet (2026-06-25)

**Status:** factual inventory for Stage 1; **not locked**. Counts from official Hugging Face dataset cards unless noted. Three split layers must not be conflated:

| Layer | Meaning |
| ----- | ------- |
| **Native splits** | What dataset authors publish (train / validation / test / dev) |
| **Standard eval split** | What leaderboard tools score on ([lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) task YAMLs) |
| **Project splits (`calib` / `test`)** | Partition of **evaluation corpus** \(C\) — policy locked in M1; IDs in M3 |

**Phase 1 scope (advisor direction, not locked):** shared MCQ protocol — ARC-Challenge (likely primary), MMLU selected subjects (secondary), TruthfulQA MCQ (secondary), HellaSwag (optional pilot).

##### Summary

| Dataset | HF repo | Native splits | Total (config) | Standard eval split | Eval *N* |
| ------- | ------- | --------------- | -------------- | ------------------- | -------- |
| **ARC-Challenge** | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | train / val / test | 2,590 | **test** | **1,172** |
| **MMLU** | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) (`all`, 57 subjects) | dev / val / test (+ `auxiliary_train`) | 14,042 (test) | **test** | **14,042** |
| **TruthfulQA MCQ** | [truthfulqa/truthful_qa](https://huggingface.co/datasets/truthfulqa/truthful_qa) | **validation only** | 817 | **validation** | **817** |
| **TruthfulQA-MC (alt.)** | [EleutherAI/truthful_qa_mc](https://huggingface.co/datasets/EleutherAI/truthful_qa_mc) | **validation only** | 684 | **validation** | **684** |
| **HellaSwag** | [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) | train / val / test | 59,950 | **validation** | **10,042** |

Combined standard-eval pool (all four at full size, official TruthfulQA): **26,073** queries. With EleutherAI TruthfulQA-MC: **25,940**.

##### Per-dataset detail

**ARC-Challenge** — Clark et al., 2018 ([arXiv:1803.05457](https://arxiv.org/abs/1803.05457))

| Split | *N* |
| ----- | --- |
| train | 1,119 |
| validation | 299 |
| test | 1,172 |

- MCQ: 4 choices (A–D), gold `answerKey`.
- lm-eval (`arc_challenge`): scores **test**; few-shot from **train** (Open LLM Leaderboard: 25-shot).
- Separate config `ARC-Easy` exists (5,197 total) — not in Phase 1 shortlist.

**MMLU** — Hendrycks et al., 2021 ([ICLR 2021](https://openreview.net/forum?id=d7KBjmI3GmQ))

| Split | *N* | Role (HF card) |
| ----- | --- | -------------- |
| test | 14,042 | Main held-out eval (≥100 per subject) |
| validation | 1,531 | Hyperparameter / model selection |
| dev | 285 | 5 per subject — **few-shot demos only** |
| auxiliary_train | 99,842 | Extra MCQ from ARC, OBQA, RACE, etc. — **not** standard MMLU eval |

- MCQ: 4 choices (A–D); 57 subject configs.
- **No split named `train`** in config `all` (do not confuse with `auxiliary_train`).
- lm-eval (`mmlu`): scores **test**; few-shot from **dev** (5-shot). **validation** not scored by default.

**TruthfulQA MCQ** — Lin et al., 2021 ([arXiv:2109.07958](https://arxiv.org/abs/2109.07958))

| Repo | Split | *N* | Notes |
| ---- | ----- | --- | ----- |
| truthfulqa/truthful_qa | validation | 817 | `multiple_choice` config; same 817 questions as `generation` |
| EleutherAI/truthful_qa_mc | validation | 684 | Drops questions with &lt;4 choices; fixed 4 options |

- **No train / test split** in either HF repo.
- lm-eval (`truthfulqa_mc1`, `truthfulqa_mc2`): scores **validation**; 0-shot; uses official repo, config `multiple_choice`. MC1 = single correct; MC2 = multi-label correct.

**HellaSwag** — Zellers et al., 2019 ([ACL 2019](https://rowanzellers.com/hellaswag/))

| Split | *N* |
| ----- | --- |
| train | 39,905 |
| validation | 10,042 |
| test | 10,003 |

- MCQ-style: 4 endings, one correct (`label` 0–3); sentence completion.
- lm-eval (`hellaswag`): scores **validation** (`test_split: null`); few-shot from **train** (10-shot on Open LLM Leaderboard).
- [Homepage](https://rowanzellers.com/hellaswag/): test leaderboard closed Nov 2024; community has long reported on **validation**.

##### Routing literature note (RouterBench)

[RouterBench](https://arxiv.org/abs/2403.12031) (Hu et al., 2024) includes ARC-Challenge, HellaSwag, and MMLU. For predictive-router experiments it **randomly partitions each task 70% train / 30% eval** — not the native splits above. MT-Bench excluded from that partition due to size.

##### Evaluation corpus — per-benchmark M1 config (§13 fact sheet)

M1 declares which native splits compose \(C\) (implementation only). The **partition rule** is uniform Option 1 (§0.7.2): random split of \(C\) into \(H, R_c, R_t\).

| Benchmark | \(C\) (`evaluation_corpus.hf_splits`) | Exclude from \(C\) |
| --------- | ------------------------------------- | ------------------ |
| **ARC-Challenge** | validation + test (1,471) | train (few-shot) |
| **MMLU** | validation + test | dev (few-shot), auxiliary_train |
| **TruthfulQA MCQ** | validation only (817) | — |
| **HellaSwag** | validation only (10,042) | train |

**Example partition (fixed counts — same M2 pilot cost on every benchmark):**

| Split | Symbol | Role | Size |
| ----- | ------ | ---- | ---- |
| Selection holdout | \(H\) | M2 pilot only | `selection_holdout_n` = **150** (all benchmarks) |
| Test | \(R_t\) | H4 only | `test.n` = **150** |
| Calib | \(R_c\) | H1–H3, \((\lambda,\tau)\) fit | \|C\| − 150 − 150 (remainder) |

| Benchmark | \|C\| | \|H\| | \|R_t\| | \|R_c\| |
| --------- | ----- | ----- | ------- | ------- |
| ARC | 1,471 | 150 | 150 | 1,171 |
| TruthfulQA | 817 | 150 | 150 | 517 |
| HellaSwag | 10,042 | 150 | 150 | 9,742 |

M2 oracle cost = **150 × 2 models** per benchmark — not percentage-dependent.

All three drawn from \(C\) via one random partition (fixed `seed`). **No query in \(H\) may appear in \(R_c\) or \(R_t\).**

Sizes and `seed` are fixed in M1 before M2. M3 assigns query IDs; M4 reads frozen lists.

**Do not use without pre-specification:**

- MMLU `dev` (reserved for few-shot prompts)
- MMLU `auxiliary_train` (different data mixture)
- ARC / HellaSwag **train** (unless an explicit supervised baseline — out of scope for unsupervised routing claims)

**Compute note:** MMLU full test × pool oracle is expensive; benchmark selection specification should pre-define subject subset or cost cap before pilot.

**M2 sampling note:** Holdout \(H\) is sampled from \(C\) per M1. M3 assigns \(R_c, R_t\) from \(C \setminus H\); M4 uses frozen ID lists only.

**Sources:** HF dataset cards linked above; lm-eval task YAMLs ([arc](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/arc/arc_easy.yaml), [mmlu](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlu/default/_default_template_yaml), [truthfulqa_mc1](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/truthfulqa/truthfulqa_mc1.yaml), [hellaswag](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hellaswag/hellaswag.yaml)); RouterBench [arXiv:2403.12031](https://arxiv.org/abs/2403.12031).

### M2 — Setting Feasibility Assessment (Stage 2)

Factorial pilot on selection holdout; bucket scorecard (§0.7). **Deliverable:** scorecard (§12).

### M3 — Experimental Setting Lock (Stage 3)

Winner → freeze split IDs → setup table. **Deliverable:** frozen Setting YAML + `T1_setup.tex`. **Gate to M4.**

### After lock (Stages 4–9 — M4)

- Stage 4: full oracle on calib + test.
- Stage 5: layer-1 signals (no \(r(q)\) input).
- Stages 6–9: analysis → selection → policy → evaluation.

**Paper:** update experimental setup section when Stage 3 completes.

---

## 14. Scope & assumptions

| In scope | Out of scope |
| -------- | ------------ |
| §2 three-layer unsupervised definition | Claiming “no labels ever” |
| §5 signal assumptions and types | Supervised router reproduction |
| §8 rule + learned policies | Benchmark suite for its own sake |
| Honest negative results | Chasing SOTA |
| Requirements-first Stages 1–3 (§0.7) | Reusing retired experiment configs as locked defaults |

**Assumptions (method-level):** fixed model pool with distinct capabilities **after Setting lock**; primary pair \((M_{\mathrm{lo}}, M_{\mathrm{hi}})\) designated; objective task metric **under locked protocol**; \(\kappa\) documented.

**Assumptions (validated in M2):** non-degenerate routing mix; \(\mathrm{Acc}(M_{\mathrm{lo}}) < \mathrm{Acc}(M_{\mathrm{hi}})\) on selection holdout.

**Success:** answer §10 honestly — report §6 and §9 separately.

---

## 15. Advisor notes

| Topic | Common drift | Advisor guidance |
| ----- | ------------- | ---------------- |
| Main contrast | Pre- vs post-inference | **Supervised vs unsupervised** (§2) |
| “Unsupervised” | No labels anywhere | **No prior routing signal**; weight fit uses calib labels |
| Signal source | Learned routing representations | Query text and model outputs — not a routing classifier’s hidden state |
| Combining signals | Same as discovering signals | **Combining** with calib labels ≠ supervised **signal discovery** |
| Scope | Agent orchestration | **LLM pool routing first** |

*“If this one and two you can do, then this itself is publishable.”*
