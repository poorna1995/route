# Contributing / workflow

## Git workflow

- **`main`** — stable; merge via pull request (protect this branch on GitHub).
- **Feature branches** — `git checkout -b feat/oracle-test` (or `fix/…`, `paper/…`).
- **Commits** — code, paper, research docs, frozen manifests (`analysis/splits.json`).
- **Do not commit** — oracle JSON, probe CSVs, feature CSVs, merged tables (see `.gitignore`).

## Experiment artifacts

Store large outputs outside Git:

- Local: `experiments/M4/`, `experiments/M5/`, regenerable files under `analysis/`
- Cloud: upload/download checkpoints when using RunPod (S3/Drive until you add object storage)

Copy checkpoint before/after cloud runs:

```bash
experiments/M4/routing_opportunity/arc_validation_oracle.json
experiments/M4/routing_opportunity/arc_test_oracle.json
```

## Branches (examples)

| Branch | Purpose |
|--------|---------|
| `main` | Paper + scripts aligned to frozen science |
| `feat/runpod-oracle` | Cloud batch runs, batch scripts |
| `paper/results` | Table fills after experiments complete |
