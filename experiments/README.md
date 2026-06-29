# Experiments layout

Binary **LLM routing** study (select \(M_{\mathrm{lo}}\) or \(M_{\mathrm{hi}}\)); oracle \(r(q)\) = appropriate-model label. Voice: [`../research/program.md`](../research/program.md) §0.1a.

## Config

```
experiments/
  query_derived_defaults.yaml # draft φ(q) manifest (lock at M3)
  candidates/*.yaml       # corpus + meta only
  runs/
    <run_id>/             # scratch runs (timestamp-name)
    permanent/            # promoted, paper-citeable artifacts
      oracle/             # {slug}_oracle_{pilot|val|test}
      query_derived/      # {slug}_query_derived_{pilot|val|test|eval}
    selection_report.json # M2 aggregate — paper-citeable winner report
    query_derived_index.json
```

Candidates merge `defaults.yaml` at load time. Run dirs store the **expanded** setting.

## Partition timing (M1 vs M3)

| Phase | What gets created |
|-------|-------------------|
| **M1** (`prepare`) | Corpus C + selection holdout H (`selection_holdout_n=150`) |
| **M2** (`oracle` + `scorecard` + `select`) | Pilot on H; pick winning benchmark |
| **M3** (`lock-eval`) | R_c + R_t on **winning run only** (adaptive \|R_t\|) |

**Partition policy (locked in M1, IDs at M3 on winner):**

| Split | Rule |
|-------|------|
| **H** | `selection_holdout_n = 150` (all benchmarks; M2 pilot only) |
| **R_t** | `clamp(round(0.20 × \|C\\H\|), 150, 1000)` — frozen as `test_n` at M3 |
| **R_c** | remainder — maximize for φ, novelty, H1–H3, (λ, τ) |

M2 oracle cost = **150 × 2 models** per benchmark (unchanged).

## CLI

```bash
python run.py all --setting experiments/candidates/arc.yaml --name arc-smoke --smoke

python run.py new --setting experiments/candidates/arc.yaml --name arc-pilot
python run.py prepare --run experiments/runs/<run_id>
python run.py oracle  --run experiments/runs/<run_id>
python run.py oracle  --run experiments/runs/<run_id> --split calib --backfill  # fill missing trace only
python run.py model-response --run experiments/runs/<run_id> --role M_lo   # CPU: ψ from trace
python run.py model-response --run experiments/runs/<run_id> --role M_hi
python run.py cross-model --run experiments/runs/<run_id>                   # CPU: χ join
python run.py scorecard --run experiments/runs/<run_id>
python run.py select                                          # → selection_report.json
python run.py lock-eval --run experiments/runs/<winner_id>   # M3 only
python run.py query-derived-all --mock-embed                 # all benchmarks, full R_c∪R_t
python run.py query-derived-all --smoke --mock-embed         # 5 queries per dataset
python run.py query-derived --run experiments/runs/<run_id> --mock-embed
python run.py resume --run experiments/runs/<run_id>
```

## Pipeline stages

```
prepare → oracle (GPU) → scorecard
              ↓
         query-derived φ(q)     [5A]
         model-response ψ(q)    [5B]  per M_lo / M_hi
         cross-model χ(q)       [5C]  join only, no inference
              ↓
         signal analysis → freeze x(q) → routing → Pareto
```

## Code modules

| Module | Role |
|--------|------|
| `pipeline.py` | Run layout + stages + selection report |
| `corpus.py` | Query, QueryResult, load/partition C |
| `setting.py` | YAML load/save + Phase A defaults merge |
| `oracle.py` | MCQ prompts, grading, HF inference + full `model_response` trace (GPU, once) |
| `model_response/` | `protocol.py` (trace + extractors), `stage.py` (ψ metrics, CPU) |
| `cross_model/` | `stage.py` (χ join from ψ, CPU) |
| `query_derived/` | `core.py` (φ features + engineering), `run.py` (orchestration) |
| `signals.py` | SignalRecord artifact (signal_type + metrics dict) |

RunPod: `bash runpod.sh smoke` (env + HF cache persist on `/workspace`; see `scripts/README.md`)
