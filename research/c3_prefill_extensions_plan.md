# C3 implementation plan — layerwise confidence evolution (ACL scope)

> **Concepts & verified references:** [`c3_layerwise_concepts.md`](c3_layerwise_concepts.md) — read first (wording rules, literature, RMSNorm, layer depth).  
> **Campaign folder (legacy):** `C3_llama_confidence_formation/`  
> **Code:** `scripts/routing/layerwise.py`, `formation_analysis.py`, `model_dependent.py`, `data.py`, `run.py`

---

## 0. Does C3 strengthen the main claim?

**Main claim (locked):** Pre-inference signals contain routing-relevant information before decoding.

C3 **extends model-derived signals** with a richer characterization (terminal → layerwise). It is **not** a fourth information source.

**C0 already supports the main claim** (verified on disk):

| Evidence                                                        | Source                                 |
| --------------------------------------------------------------- | -------------------------------------- |
| Opportunity 43.3% (ARC TEST)                                    | `analysis/arc_merged.csv`              |
| Escalation vs uncertainty: d(Δm_gain)≈**0.72**, d(H_w)≈**0.03** | `analysis/arc_interpretation.json`     |
| MMLU pooled pattern transfer                                    | `analysis/C2_dimension_transfer.json`  |
| Exploitation gap 5.2 pp                                         | `analysis/arc_route_eval_lambda0.json` |

---

## 1. Scientific framing (Methods — neutral wording)

**We investigate** how prefill confidence **evolves across transformer layers** at the last prompt token. Terminal statistics (C0) capture only the **endpoint** of that computation.

**RH5 (locked):**

> Does layerwise evolution of model confidence provide routing-relevant information **beyond terminal** prefill confidence?

**Measurable question:**

> Do easy, opportunity, and too-hard queries exhibit **different confidence evolution** across transformer layers?

**After results only (Discussion):** e.g. “confidence **appears to form** progressively for opportunity queries” — if F7 supports it.

**Not studying:** decoding, CoT, multi-token time series, hidden-state geometry, paraphrases.

**Methods (one line):** We use a logit-lens style probe (§3.1 in concepts doc). Not claiming intermediate layers equal final predictions — depth-indexed probe statistics only.

See [`c3_layerwise_concepts.md`](c3_layerwise_concepts.md) §3.1, §5 for literature chain (LayerSkip = motivation; early-exit = related work, different objective).

---

## 2. Paper-facing taxonomy

| Information source | Examples (this paper)                                                                                            |
| ------------------ | ---------------------------------------------------------------------------------------------------------------- |
| **Query-derived**  | `piece_count` / \(c(q)\)                                                                                         |
| **Model-derived**  | Terminal: \(H_w\), \(m_w\). **C3 extension:** layerwise evolution (`stabilization_layer` ⭐, `slope_margin`; F7) |
| **Cross-model**    | \(\Delta H\), \(\Delta m\_{\mathrm{gain}}\)                                                                      |

```text
Model-derived → terminal statistics (C0) → layerwise evolution (C3)
```

---

## 3. RH5 — evidence criteria (no AUROC gate)

| #   | Evidence               | Example                                                          |
| --- | ---------------------- | ---------------------------------------------------------------- |
| 1   | **F7 geometry**        | Median \(m\_\ell\) curves separate easy / opportunity / too-hard |
| 2   | **Divergence L\***     | Opp vs too-hard similar until fraction depth d\*, then diverge   |
| 3   | **Effect size**        | d(`stabilization_layer`) > d(\(H_w\)) ≈ 0.03                     |
| 4   | **Non-redundancy**     | Partial ρ(scalar, opp \| terminal \(m_w\)) > 0                   |
| 5   | **Null (publishable)** | Evolution mirrors terminal — terminal probes suffice on ARC      |

**Emphasize `stabilization_layer`** in Results prose (when confidence stabilizes). **`slope_margin`** secondary only — do not gate RH5 on slope.

**Probe space:** raw `softmax(logits_ℓ)` at each layer; **no logit temperature rescaling** (same as C0). Mention miscalibration as limitation, not a probe change.

Primary: **F7 + one observation sentence**. AUROC secondary.

---

## 4. Scope lock

| In ACL scope                                  | Deferred                                    |
| --------------------------------------------- | ------------------------------------------- |
| Layerwise forward pass + JSONL traces         | Representation drift                        |
| **`stabilization_layer`**, **`slope_margin`** | Paraphrase, Qwen C1                         |
| F7 + divergence (fraction depth ℓ/L)          | AUROC gate before TEST                      |
| Reuse C0 oracle, D46                          | Overwriting `experiments/M5/arc_test_*.csv` |

---

## 5. Codebase state

**Exists:** `probe_batch`, `extract_logits`, `merge_tables`, `run_interpretation_analysis`.

**Missing (optional polish):** none for smoke/TEST pipeline.

**Implemented:** `layerwise.py`, `--layerwise`, formation merge, `plot formation`, `analyze-formation`.

**Invariants:** Same `build_chat_prompt()` / `prompt_hash` as C0.

**CLI flags:** `--layerwise`, `--layer-trace`, `--stab-eps`, `--stab-k`, `--margin-tol`, `--overwrite`.

---

## 6. Technical design

### 6.1 Forward pass + Llama RMSNorm (see concepts §3.2)

```text
outputs = model(**inputs, output_hidden_states=True)
last_pos = attention_mask.sum(dim=1) - 1

Terminal (must match C0):
    logits_T = model(**inputs).logits[:, last_pos, :]
    OR lm_head(model.norm(hidden_states[-1][:, last_pos, :]))

For intermediate ℓ = 1 … L−1:
    logits_ℓ = lm_head(hidden_states[ℓ][:, last_pos, :])

For ℓ = L:
    logits_L = lm_head(model.norm(hidden_states[L][:, last_pos, :]))  # match terminal
```

**Smoke checks:**

1. `|margin(logits_L) − margin(C0 csv)|` < ε for same queries
2. `|margin(logits_L) − margin(model.logits)|` < ε

Default ε = **`margin_tol = 1e-3`** (bf16 RunPod Llama; exposed as `--margin-tol`).

**Compute:** 1B/3B, `batch_size=1`, bf16 — in scope; no OOM gate.

### 6.2 Scalars (CSV)

| Column                    | Role                                                                                                         |
| ------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **`stabilization_layer`** | ⭐ Primary — when margin stabilizes (ε=0.02, K=2). Requires ≥1 observed transition; minimum ℓ = 2 when L ≥ 2 |
| **`slope_margin`**        | Secondary / exploratory — OLS slope of \(m\_\ell\) vs ℓ; not headline RH5 evidence                           |

Also store `stabilization_frac` (= `depth_fraction[stabilization_layer − 1]`) in JSONL or analysis output.

`extraction_method`: `prefill_probe_layerwise`

### 6.3 Layer depth (1B: L=16, 3B: L=28)

- F7 / divergence: per model separately, or x-axis = **ℓ/L**
- Never compare raw layer index across 1B vs 3B

### 6.4 Cross-model merge

- `delta_slope_margin` = `slope_margin_s` − `slope_margin_w`
- `delta_stabilization_layer` = `stabilization_layer_s` − `stabilization_layer_w`

---

## 7. Divergence analysis (post-hoc)

On JSONL + oracle buckets: median \(m\_\ell\) per bucket → dispersion vs ℓ → report **L\*** and **`fraction_depth = L*/L`**.

**Results sentence template:**

> Opportunity and too-hard queries exhibit similar weak-model margin until approximately **fraction depth** d\*, after which their confidence **trajectories diverge**.

**Discussion only (if supported):** “formation” language.

---

## 8. Figure F7

**Title:** **Layerwise Confidence Evolution Across Depth**

Median \(m\_\ell\) vs ℓ (or ℓ/L) — Easy · Opportunity · Too-hard (weak, ARC TEST).

`paper/figures/F7_confidence_formation.png` (legacy filename ok).

---

## 9. Implementation checklist

| Step | File                      | Task                                                                             |
| ---- | ------------------------- | -------------------------------------------------------------------------------- |
| 1    | `layerwise.py`            | Margin series; **RMSNorm on final layer**; `stabilization_layer`; `slope_margin` |
| 2    | `layerwise.py`            | Dedicated module; reuses `extract_logits` from `model_dependent.py`              |
| 3    | `constants.py`            | `PROBE_FORMATION_BASES`                                                          |
| 4    | `run.py`                  | `--layerwise`, trace dir, ε, K                                                   |
| 5    | `data.py`                 | Merge formation + deltas                                                         |
| 6    | `plots.py`                | F7: `depth_fraction` vs median margin (always ℓ/L axis)                          |
| 7    | `formation_analysis.py`   | Divergence + RH5 JSON (fraction depth)                                           |
| 8    | `tests/test_layerwise.py` | Synthetic stabilization + slope                                                  |

---

## 10. Run sequence (locked)

```text
Phase A0  layerwise-parity n=10     lm_head(norm(h[-1])) vs out.logits
Phase A   smoke n=10 (--layerwise)  terminal margin ≈ C0 csv within margin_tol
          manual check              inspect a few JSONL trajectories (not flat/noisy)
Phase B   CALIB run                 stabilization_layer + slope_margin; preliminary F7
          decision gate             meaningful pattern OR clear null → proceed;
                                    noisy/inconsistent → C3 future work (no redesign)
Phase C   TEST once (if gate pass)  weak + strong → F7 → analyze-formation → §5.7 or one sentence
```

**Do not skip Phase B decision gate.** A clearly interpretable null RH5 is publishable; incoherent trajectories are not worth a full TEST burn.

### Phase A0 — Terminal parity

```bash
python scripts/run.py layerwise-parity \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dataset arc_challenge --splits-json analysis/splits.json --split-role test \
  --limit 10 --device cuda --dtype bfloat16 --margin-tol 0.001
```

### Phase A — Smoke + manual inspection

```bash
python scripts/run.py probes --layerwise --overwrite ... --limit 10 ...
```

Verify `|margin_layerwise − margin_C0| < margin_tol` on overlapping queries. Open JSONL: trajectories should evolve (not constant or wild oscillation).

### Phase B — CALIB + preliminary F7

Run layerwise on CALIB (`split-role calib`). Merge with oracle. Plot F7 (`depth_fraction` vs median margin by Easy / Opportunity / Too-hard). **Decision gate** before TEST.

### Phase C — TEST (only if gate passes)

Weak + strong TEST extraction → merge → F7 → `analyze-formation`.

---

## 11. Stop rule (summary)

```text
Any gate FAIL → stop (no multi-day debug) → C3 = future work
```

```bash
python scripts/run.py merge ...
python scripts/run.py plot formation ... --output paper/figures/F7_confidence_formation.png
python scripts/run.py analyze-formation ... --output analysis/c3_rh5_summary.json
```

---

## 12. RunPod integration

Use the batch scripts (same pattern as `screen_c2_candidates.sh`):

```bash
chmod +x scripts/run_c3_runpod.sh scripts/run_c3_postprocess.sh

# GPU
./scripts/run_c3_runpod.sh parity
./scripts/run_c3_runpod.sh smoke
./scripts/run_c3_runpod.sh extract calib all
./scripts/run_c3_postprocess.sh calib    # decision gate on F7

./scripts/run_c3_runpod.sh extract test all   # only if gate passes
./scripts/run_c3_postprocess.sh test
```

See `experiments/README.md` § C3 layerwise for env vars (`DEVICE`, `CAMPAIGN`, oracle paths).

### Manual command reference

```bash
export PYTHONPATH=scripts:$PYTHONPATH

python scripts/run.py probes \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dataset arc_challenge \
  --splits-json analysis/splits.json --split-role test \
  --layerwise \
  --layer-trace experiments/campaigns/C3_llama_confidence_formation/M5/layer_traces/weak.jsonl \
  --stab-eps 0.02 --stab-k 2 \
  --margin-tol 0.001 --overwrite \
  --batch-size 1 --device cuda --dtype bfloat16 \
  --output experiments/campaigns/C3_llama_confidence_formation/M5/arc_test_weak_layerwise.csv
```

---

**Next step:** RunPod smoke (n=10), then TEST once → F7 → `analyze-formation`.
