# Experiments layout

## Config

```
experiments/
  query_derived_defaults.yaml # draft φ(q) manifest (lock at M3)
  candidates/*.yaml       # corpus + meta only
  runs/
    <run_id>/             # per-candidate M2 pilot artifacts
    selection_report.json # M2 aggregate — paper-citeable winner report
```

Candidates merge `phase_a_defaults.yaml` at load time. Run dirs store the **expanded** setting.

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
python run.py scorecard --run experiments/runs/<run_id>
python run.py select                                          # → selection_report.json
python run.py lock-eval --run experiments/runs/<winner_id>   # M3 only
python run.py query-derived-all --mock-embed                 # all benchmarks, full R_c∪R_t
python run.py query-derived-all --smoke --mock-embed         # 5 queries per dataset
python run.py query-derived --run experiments/runs/<run_id> --mock-embed
python run.py resume --run experiments/runs/<run_id>
```

## Code modules

| Module | Role |
|--------|------|
| `pipeline.py` | Run layout + stages + selection report |
| `corpus.py` | Query, QueryResult, load/partition C |
| `setting.py` | YAML load/save + Phase A defaults merge |
| `oracle.py` | HF inference |
| `prompts.py` | MCQ prompt + grading |
| `query_derived/` | Stage 5: `config`, `extract`, `engineer`, `run` |
| `signals.py` | SignalRecord artifact |

RunPod: `bash runpod.sh smoke` (env + HF cache persist on `/workspace`; see `scripts/README.md`)
