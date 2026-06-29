# Experiments layout

## Config

```
experiments/
  phase_a_defaults.yaml   # M1 frozen pool, protocol, partition, gates, tie-break
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
| **M3** (`lock-eval`) | R_c + R_t on **winning run only** (`test_n=150`) |

Fixed counts — same pilot cost on ARC (1172), TruthfulQA (817), HellaSwag (10042).

## CLI

```bash
python run.py all --setting experiments/candidates/arc.yaml --name arc-smoke --smoke

python run.py new --setting experiments/candidates/arc.yaml --name arc-pilot
python run.py prepare --run experiments/runs/<run_id>
python run.py oracle  --run experiments/runs/<run_id>
python run.py scorecard --run experiments/runs/<run_id>
python run.py select                                          # → selection_report.json
python run.py lock-eval --run experiments/runs/<winner_id>   # M3 only
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
| `signals.py` | SignalRecord artifact |

RunPod: `bash runpod.sh smoke` (env + HF cache persist on `/workspace`; see `scripts/README.md`)
