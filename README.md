# Characterizing Unsupervised Routing Signals for Multi-LLM Selection

Research codebase and ACL paper for pre-hoc unsupervised routing signals (query complexity, entropy, margin) on ARC-Challenge with Llama 3.2 1B/3B.

## Where to start

| Path | Contents |
|------|----------|
| [`research/MASTER.md`](research/MASTER.md) | Frozen science (RQ, hypotheses, studies) |
| [`research/WORKFLOW.md`](research/WORKFLOW.md) | Study pipeline order |
| [`experiments/README.md`](experiments/README.md) | CLI commands, GPU/CPU settings, artifact paths |
| [`paper/acl.tex`](paper/acl.tex) | Paper source |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Git branches; what to commit vs ignore |

## Setup (local)

```bash
uv venv && source .venv/bin/activate
uv sync --extra analysis
huggingface-cli login   # gated Llama models
```

On **Linux + NVIDIA** (e.g. RunPod), install CUDA PyTorch before `uv sync` — see `experiments/README.md`.

## Main CLI

```bash
.venv/bin/python scripts/run.py oracle --help
.venv/bin/python scripts/run.py features --help
.venv/bin/python scripts/run.py probes --help
```

Resume-friendly CALIB/TEST oracle batches:

```bash
./scripts/batch_oracle_cpu.sh calib    # DEVICE=cuda DTYPE=bfloat16 on GPU
```

## Artifacts (not in Git)

Large outputs live under `experiments/` and regenerable files under `analysis/`. Frozen split manifest: `analysis/splits.json` (committed).
