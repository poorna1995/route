# Experiments — environment, packages, and execution

> **Project (frozen):** [`research/MASTER.md`](../research/MASTER.md)

> **Registry:** `research/10_experiment_registry.md` · **Design:** `research/MASTER.md`  
> **Dependencies:** root `pyproject.toml` · **Python:** 3.12 (`.python-version`)

---

## Directory layout

```text
experiments/
├── M4/routing_opportunity/   # V2 oracle JSON (full inference, offline labels)
├── M5/                       # Signal CSVs (prefill H, m; query complexity candidates)
└── README.md                 # this file

analysis/                     # Derived metrics (ρ, AUROC, summaries) — not raw runs
scripts/                      # Frozen pipeline (do not change without decision in 09)
```

Raw outputs live here; interpretation lives in `analysis/` and `research/notes.md`.

---

## Package manager: `uv`

Use **[uv](https://docs.astral.sh/uv/)** to create the venv, install dependencies, and run scripts without manual `activate`.

### One-time setup

```bash
# Install uv (once per machine)
curl -LsSf https://astral.sh/uv/install.sh | sh

cd /path/to/llm_routing
uv sync                      # inference dependencies → .venv
uv sync --extra analysis     # add scipy/sklearn/matplotlib when analyzing

# Hugging Face login (required for Llama weights)
uv run huggingface-cli login
```

### Run the pipeline

All stages use one entry point:

```bash
.venv/bin/python scripts/run.py --help
.venv/bin/python scripts/run.py oracle --weak ... --strong ... --output ...
.venv/bin/python scripts/run.py probes --model ... --dataset arc_challenge --output ...
```

Alternative: `source .venv/bin/activate` then `python scripts/run.py ...` (same environment).

---

## Code layout

```
scripts/
  run.py              # CLI entry (all subcommands)
  routing/            # shared library
    constants.py        # column names, feature families, ladder specs
    model_independent.py  # c(q) features + D46 screening (RH1)
    model_dependent.py    # prefill probes H, m (RH2)
    evaluation.py         # Studies I–III stats + complementarity (RH1–RH3)
    policies.py           # routing policies + preview (RH4)
    data.py               # merge oracle + probe CSVs
    oracle.py             # offline routing-opportunity assessment (C2)
    plots.py              # paper figures
```

Logic lives in `routing/`; `run.py` is thin argument parsing only.

---

## Required packages

### Inference (core — always install)

| Package               | Role                                                           |
| --------------------- | -------------------------------------------------------------- |
| **torch**             | Model load, prefill logits, oracle `generate()`                |
| **transformers**      | Llama / Qwen via Hugging Face                                  |
| **datasets**          | ARC-Challenge, GSM8K loading (`routing.datasets.load_queries`) |
| **huggingface_hub**   | Model download; CLI login                                      |
| **accelerate**        | HF ecosystem helper                                            |
| **numpy**, **pandas** | Arrays / merge steps                                           |

Declared in `pyproject.toml` `[project] dependencies`.

### Analysis (optional — install when computing ρ, AUROC, plots)

| Package                     | Role                 |
| --------------------------- | -------------------- |
| **scipy**                   | Spearman ρ, p-values |
| **scikit-learn**            | AUROC                |
| **matplotlib**, **seaborn** | Paper figures F1–F3  |

Install with: `uv sync --extra analysis`

### Transitive (automatic — do not install manually)

`aiohttp`, `httpx`, `fsspec`, `pyarrow`, etc. — pulled in by `datasets` / `huggingface_hub`.

### Not required for v1 pilot

| Package                       | Why skip                                                         |
| ----------------------------- | ---------------------------------------------------------------- |
| **vLLM / TGI**                | Different serving stack; complicates full-vocab logprob protocol |
| **bitsandbytes / GPTQ / AWQ** | Quantization can alter logprobs (bad for H, m science)           |
| **flash-attn**                | Long-context optimization; ARC prompts are short                 |
| **wandb / mlflow**            | JSON + CSV + `analysis/` is sufficient                           |

Research needs **correct full-vocabulary prefill logits**, not maximum throughput at any cost.

---

## Hardware

| Environment                           | Recommendation                          |
| ------------------------------------- | --------------------------------------- |
| **Smoke / debug (n≤10)**              | Mac CPU or MPS acceptable               |
| **V2 configuration (n≈50)**           | GPU strongly preferred                  |
| **Main characterization (n=150–200)** | **1× NVIDIA GPU, 16–24 GB VRAM** (CUDA) |

CPU-only runs work but are very slow (~30s+ per oracle query on MacBook Air). For n=150–200, use a CUDA machine or short cloud GPU session (L4, T4, A10, etc.).

### CUDA PyTorch (GPU machine only)

Default `uv sync` installs CPU torch on macOS. On Linux + NVIDIA:

```bash
uv venv
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv sync --extra analysis
```

Match CUDA version to your driver (`cu124`, `cu121`, …) per [pytorch.org](https://pytorch.org).

---

## Efficiency settings (reduce load)

### Dtype by device

| Device           | Flag                               |
| ---------------- | ---------------------------------- |
| CUDA             | `--device cuda --dtype bfloat16`   |
| Mac MPS          | `--device mps --dtype float16`     |
| CPU (smoke only) | `--device cpu` (expect heavy load) |

### Oracle runs — one model in memory at a time

`run.py oracle` loads weak then strong sequentially. On low RAM, split into two passes:

```bash
# Pass 1: weak only
.venv/bin/python scripts/run.py oracle \
  --weak meta-llama/Llama-3.2-1B-Instruct \
  --strong meta-llama/Llama-3.2-3B-Instruct \
  --dataset arc_challenge --limit 50 --seed 42 \
  --max-new-tokens 8 \
  --device cuda --dtype bfloat16 \
  --weak-only \
  --output experiments/M4/routing_opportunity/llama_arc_n50.json

# Pass 2: strong only (resumes from same JSON — omit --no-resume)
.venv/bin/python scripts/run.py oracle \
  ...same args... \
  --strong-only
```

- **`--max-new-tokens 8`** for ARC/MCQ (already correct).
- **Resume:** omit `--no-resume` so partial JSON is reused after crashes.
- **`release_model()`** runs after each model pass (built into scripts).

### Signal extraction — batch prefill on GPU

```bash
.venv/bin/python scripts/run.py probes \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dataset arc_challenge --limit 200 --seed 42 \
  --device cuda --dtype bfloat16 \
  --batch-size 8 \
  --output experiments/M5/llama_arc_n200_weak_signals.csv
```

Increase `--batch-size` until VRAM ~80% full (start at 4).

### D46 signal screening — select representative \(c(q)\)

**ARC validation (~299) only** to lock D46. Dev preview requires `--allow-preview` and cannot lock.

```bash
# 1. Write splits (validation → CALIB, test → TEST)
.venv/bin/python scripts/run.py splits \
  --dataset arc_challenge \
  --output analysis/splits.json

# zsh: use arrays (scalar $S=... is passed as one argument in zsh)
S=(--splits-json analysis/splits.json --split-role calib --seed 42)
T=(--splits-json analysis/splits.json --split-role test --seed 42)
GPU=(--device cuda --dtype bfloat16 --batch-size 8)

# 2. CALIB extraction (validation)
.venv/bin/python scripts/run.py features \
  --dataset arc_challenge $S \
  --output experiments/M5/llama_arc_validation_query_features.csv

.venv/bin/python scripts/run.py oracle \
  --weak meta-llama/Llama-3.2-1B-Instruct \
  --strong meta-llama/Llama-3.2-3B-Instruct \
  --dataset arc_challenge $S --max-new-tokens 8 $GPU \
  --output experiments/M4/routing_opportunity/llama_arc_validation.json
# Resume same output for TEST pass:
.venv/bin/python scripts/run.py oracle \
  --weak meta-llama/Llama-3.2-1B-Instruct \
  --strong meta-llama/Llama-3.2-3B-Instruct \
  --dataset arc_challenge $T --max-new-tokens 8 $GPU \
  --output experiments/M4/routing_opportunity/llama_arc_full.json

.venv/bin/python scripts/run.py screen \
  --features experiments/M5/llama_arc_validation_query_features.csv \
  --oracle experiments/M4/routing_opportunity/llama_arc_validation.json \
  --output analysis/d46_signal_screen_arc_validation.json
```

Log `selected_candidate` as **D46** in `research/09_decision_register.md`. **Run once; do not re-screen** after code tweaks without a new decision.

On a locked CALIB run, `screen` also writes `analysis/selected_feature.json` — the frozen D46 winner. All downstream `merge` / `complementarity` runs must pass `--complexity-selection analysis/selected_feature.json` (not ad-hoc `--c-q-column`).

```bash
.venv/bin/python scripts/run.py doctor \
  --oracle experiments/M4/routing_opportunity/llama_arc_validation.json \
  --weak-csv experiments/M5/llama_arc_validation_weak_signals.csv \
  --strong-csv experiments/M5/llama_arc_validation_strong_signals.csv \
  --features-csv experiments/M5/llama_arc_validation_query_features.csv \
  --complexity-selection analysis/selected_feature.json
```

---

Run oracle **once** per locked configuration; reuse the JSON for all signal analysis. Do not re-run `generate()` when only adding probe CSVs.

---

## Supported datasets (`prompt_protocol.load_queries`)

| ID              | Scientific question      | Split        | Notes                         |
| --------------- | ------------------------ | ------------ | ----------------------------- |
| `arc_challenge` | Science reasoning        | `test`       | Primary — locked              |
| `mmlu`          | Broad factual knowledge  | `test`       | 3 subjects, stratified sample |
| `boolq`         | Reading comprehension    | `validation` | C2 screening                  |
| `gsm8k`         | (legacy validation only) | `test`       | Excluded from paper           |

**C2 candidate screening (GPU):**

```bash
DEVICE=cuda DTYPE=bfloat16 ./scripts/screen_c2_candidates.sh
```

Summaries → `analysis/c2_*_summary.json`. Review `gate_pass` before generalization.

---

## Execution order

Matches `research/WORKFLOW.md` and `research/MASTER.md`:

```text
1. Validation — ARC locked (V1/V2)
2. D46 pre-study calibration — features → screen on validation (~299) → freeze c(q) (once)
3. Full ARC — oracle + probes + merge on TEST (1,172) → Studies I–III
4. Study IV — route-eval (CALIB fit, TEST report)
5. Generalization — MMLU (same c_q, no D46 repeat)
```

---

## CLI reference (`scripts/run.py`)

| Subcommand                | Stage                                         | Output                                                                     |
| ------------------------- | --------------------------------------------- | -------------------------------------------------------------------------- |
| `verify-logprobs`         | Feasibility (V1)                              | Terminal PASS/FAIL                                                         |
| `oracle`                  | V2 oracle                                     | JSON + buckets                                                             |
| `features`                | D46 screening input                           | CSV (candidates + tokenizer_id)                                            |
| `screen`                  | D46 pre-study calibration                     | `analysis/d46_signal_screen_*.json`, `analysis/selected_feature.json`      |
| `doctor`                  | Pre-flight artifact checks                    | terminal (+ optional JSON)                                                 |
| `probes`                  | Probe extraction                              | CSV (H, m, max_prob, …)                                                    |
| `merge`                   | Routing relevance                             | `analysis/*_routing_relevance.json`, merged CSV (`--complexity-selection`) |
| `plot distributions`      | Figure F1                                     | `paper/figures/F1_*.png`                                                   |
| `plot roc`                | Figure F2                                     | `paper/figures/F2_*.png`                                                   |
| `plot scatter`            | Figure F3                                     | `paper/figures/F3_*.png`                                                   |
| `complementarity`         | Study III complementarity                     | `analysis/*_complementarity.json`                                          |
| `route-eval`              | **Study IV** hold-out routing (EXP-03)        | `analysis/*_routing_holdout.json`                                          |
| `route-preview`           | Median-heuristic sanity (D37) — not paper     | preview JSON                                                               |
| `summarize-c2`            | C2 screening summary                          | `analysis/c2_*_summary.json`                                               |
| `screen_c2_candidates.sh` | Batch C2 on MMLU/BoolQ                        | oracle JSON + summaries                                                    |
| `run_c3_runpod.sh`        | **C3 GPU** — parity, smoke, layerwise extract | campaign CSV + JSONL                                                       |
| `run_c3_postprocess.sh`   | **C3 CPU** — merge, F7, RH5 JSON              | `analysis/c3_*`, `paper/figures/F7_*`                                      |

---

## C3 layerwise (RunPod)

Scripts wrap the phased workflow in [`research/c3_prefill_extensions_plan.md`](../research/c3_prefill_extensions_plan.md) §10.

**One-time pod setup:**

```bash
cd /workspace/llm_routing
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync && uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv sync --extra analysis
huggingface-cli login
chmod +x scripts/run_c3_runpod.sh scripts/run_c3_postprocess.sh
```

**GPU (in order):**

```bash
./scripts/run_c3_runpod.sh parity          # 1B + 3B — inspect margin Δ summary
./scripts/run_c3_runpod.sh smoke             # 10 weak + 10 strong
./scripts/run_c3_runpod.sh extract calib all # CALIB
**CPU postprocess** needs CALIB oracle + features (gitignored — copy from laptop or regenerate):

```bash
mkdir -p experiments/M4/routing_opportunity experiments/M5
# From laptop (replace POD):
# scp experiments/M4/routing_opportunity/arc_validation_oracle.json root@POD:/workspace/llm_routing/experiments/M4/routing_opportunity/
# scp experiments/M5/arc_validation_features.csv root@POD:/workspace/llm_routing/experiments/M5/
# scp analysis/selected_feature.json root@POD:/workspace/llm_routing/analysis/

./scripts/run_c3_postprocess.sh calib        # F7 + RH5 weak & strong → decision gate
# Or without c(q): ./scripts/run_c3_postprocess.sh calib --allow-no-features
# if interpretable:
./scripts/run_c3_runpod.sh extract test all
./scripts/run_c3_postprocess.sh test
```

Optional: `BATCH_SIZE=2` or `4` (layerwise still runs one query per forward until batched path lands).

Artifacts: `experiments/campaigns/C3_llama_confidence_formation/M5/`. F7 outputs: `paper/figures/F7_confidence_evolution_{calib,test}_{weak,strong}.png`.

---

## Locked candidate configuration (promising — not final)

| Component | Value                                                         |
| --------- | ------------------------------------------------------------- |
| Weak      | `meta-llama/Llama-3.2-1B-Instruct`                            |
| Strong    | `meta-llama/Llama-3.2-3B-Instruct`                            |
| Dataset   | ARC-Challenge test, `seed=42`                                 |
| Oracle    | greedy, `max_new_tokens=8`                                    |
| Signals   | \(c(q)\) from D46 + prefill H, m                              |
| Main n    | Full official ARC **test** (1,172); CALIB = validation (~299) |

---

## Rough runtime expectations (n=200, 2 models)

| Step                      | CPU Mac    | CUDA GPU                    |
| ------------------------- | ---------- | --------------------------- |
| Oracle (8 tokens)         | hours–days | ~30–90 min                  |
| Prefill signals (batched) | hours      | ~5–15 min                   |
| Analysis                  | minutes    | minutes (can run on laptop) |

---

## Reproducibility

- `seed=42` for query sampling
- `protocol_version=v1` in CSV / prompts
- `prompt_hash` per query in signal CSVs
- Pin model `--revision` in `09` when configuration locks
